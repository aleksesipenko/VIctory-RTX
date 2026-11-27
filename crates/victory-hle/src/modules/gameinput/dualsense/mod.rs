//! DualSense / DualSense Edge device discovery, overlapped Win32 HID I/O,
//! `DS5Dongle` (Pico 2 W) 1.5s liveness monitoring, PS/Touchpad chord filter,
//! ContainerId-verified 4-channel WASAPI haptics binding, and best-effort
//! lock-free bounded hardware reset on focus loss, exit, or crash.

pub mod audio_haptics;
pub mod gyro;
pub mod protocol;
pub mod triggers;

#[cfg(windows)]
use std::cell::UnsafeCell;
#[cfg(windows)]
use std::mem::MaybeUninit;
use std::sync::atomic::{
    AtomicBool, AtomicU8, AtomicU16, AtomicU32, AtomicU64, AtomicUsize, Ordering,
};
use std::sync::{Arc, Condvar, LazyLock, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use victory_core::config::{DualSenseConfig, HapticsMode};

use super::{BatteryState, PadState, SYSTEM_BUTTON_GUIDE};
use audio_haptics::{AudioHapticsEngine, ContainerId};
use gyro::{GyroTracker, ImuCalibration};
use protocol::{
    BT_REPORT_LEN, ControllerModel, FIRMWARE_VIBRATION2_MIN, OUTPUT_COMMON_LEN, OutputCommand,
    RawReport, Transport, USB_OUTPUT_REPORT_ID, USB_OUTPUT_REPORT_LEN, parse_input_report,
    touchpad_button,
};
use triggers::{ReportLimiter, encode_trigger_effect};

/// Maximum DualSense slots managed concurrently (`0..4`).
pub const MAX_DUALSENSE: usize = 4;

/// Maximum HID output report byte length supported for DualSense / `DS5Dongle`
/// (standard USB = 48, `DS5Dongle` = 64, Bluetooth `0x31` = 78). Rejecting
/// descriptors outside `min_len..=128` at open time guarantees emergency stack
/// reports match normal output report length identically with zero truncation.
pub const MAX_EMERGENCY_REPORT_LEN: usize = 128;

/// Total bounded budget for clean exit / `end_title` shutdown to wait for all
/// HID and audio-haptics workers to finish before proceeding.
pub const SHUTDOWN_JOIN_TIMEOUT: Duration = Duration::from_millis(250);

/// If `sensor_timestamp` stalls or valid packets cease for longer than 1.5 s
/// (e.g. `DS5Dongle` plugged into USB while the wireless PS5 controller is
/// turned off, out of range, or streaming corrupt packets), the slot
/// transitions back to disconnected so keyboard fallback (`WASD`) resumes and
/// stuck buttons are cleared.
pub const DONGLE_FREEZE_TIMEOUT: Duration = Duration::from_millis(1500);

/// Short-press ceiling for PS Guide button events; holds `>= 600 ms` (such as
/// 2-5 s power-off or multi-device Bluetooth slot switching) are never intercepted.
const PS_SHORT_PRESS_MAX: Duration = Duration::from_millis(600);

/// Virtual button pulse length (80 ms) so the poll loop reliably observes both
/// rising and falling edges on quick PS Guide or Touchpad taps.
const BUTTON_PULSE_DURATION: Duration = Duration::from_millis(80);

/// Simultaneous-press grace window (60 ms) held on a rising Touchpad click
/// before forwarding `VIEW`/`MENU`.
///
/// Policy: `PS + Touchpad` gyro recenter is a PS-first chord with a 60 ms
/// simultaneous-press grace window for natural thumb skew; pressing Touchpad
/// alone longer than 60 ms commits normally as `VIEW`/`MENU`, and pressing PS
/// afterwards cleanly aborts it and holds suppression until both release.
const TOUCHPAD_CHORD_GRACE: Duration = Duration::from_millis(60);

/// Bitmask packing host XInput HID topology seen during enumeration so
/// `xinput_topology()` reads both flags atomically without tearing.
const XINPUT_VIRTUAL_BIT: u8 = 0x01;
const XINPUT_PHYSICAL_BIT: u8 = 0x02;
static XINPUT_TOPOLOGY: AtomicU8 = AtomicU8::new(0);

/// Set on fatal crash, exit, `end_title`, or Ctrl+C so HID worker threads stop
/// sending new normal output reports and exit cleanly after releasing hardware.
static SHUTDOWN_REQUESTED: AtomicBool = AtomicBool::new(false);

/// Non-blocking check that an OS thread has finished completely (including
/// Win32 TLS teardown), so calling `JoinHandle::join()` afterwards cannot block.
#[cfg(any(windows, test))]
pub(crate) fn is_os_thread_finished(handle: &JoinHandle<()>) -> bool {
    #[cfg(windows)]
    {
        use std::os::windows::io::AsRawHandle;
        use windows_sys::Win32::Foundation::{HANDLE, WAIT_OBJECT_0};
        use windows_sys::Win32::System::Threading::WaitForSingleObject;

        let raw = handle.as_raw_handle() as HANDLE;
        // SAFETY: `handle` owns a live Win32 thread handle; timeout 0 polls without blocking.
        unsafe { WaitForSingleObject(raw, 0) == WAIT_OBJECT_0 }
    }
    #[cfg(not(windows))]
    {
        handle.is_finished()
    }
}

/// Waits for `handle` to exit at the OS level up to `deadline`. Only invokes
/// `JoinHandle::join()` once the OS thread object is already signaled; on
/// timeout returns `Err(handle)` without dropping or detaching it so callers
/// can retain ownership and never mistake a still-running thread for finished.
fn try_join_handle_bounded(handle: JoinHandle<()>, deadline: Instant) -> Result<(), JoinHandle<()>> {
    #[cfg(windows)]
    {
        use std::os::windows::io::AsRawHandle;
        use windows_sys::Win32::Foundation::{HANDLE, WAIT_OBJECT_0};
        use windows_sys::Win32::System::Threading::WaitForSingleObject;

        let rem = deadline.saturating_duration_since(Instant::now());
        // Never pass `INFINITE` (`u32::MAX`) to `WaitForSingleObject`.
        let wait_ms = rem.as_millis().min(u128::from(u32::MAX - 1)) as u32;
        let raw = handle.as_raw_handle() as HANDLE;
        // SAFETY: `handle` owns a valid Win32 thread handle across the bounded wait.
        let signaled = unsafe { WaitForSingleObject(raw, wait_ms) == WAIT_OBJECT_0 };
        if signaled {
            let _ = handle.join();
            Ok(())
        } else {
            Err(handle)
        }
    }
    #[cfg(not(windows))]
    {
        loop {
            if handle.is_finished() {
                let _ = handle.join();
                return Ok(());
            }
            if Instant::now() >= deadline {
                return Err(handle);
            }
            std::thread::sleep(Duration::from_millis(1));
        }
    }
}

/// Waits for `handle` to exit at the OS level up to `deadline`, detaching on timeout.
#[cfg(test)]
pub(crate) fn join_handle_bounded(handle: JoinHandle<()>, deadline: Instant) -> bool {
    try_join_handle_bounded(handle, deadline).is_ok()
}

struct SharedJoinState {
    handle: Option<JoinHandle<()>>,
    joining: bool,
    joined: Option<bool>,
}

/// Shared ownership wrapper around a worker's `JoinHandle<()>` so BOTH
/// `WorkerCoordinator::shutdown_and_wait` and `AudioHapticsEngine::stop` can
/// perform bounded OS-thread joins without double-joining, leaking handles,
/// returning `true` prematurely, or mistaking a timed-out handle for finished.
#[derive(Clone)]
pub(crate) struct SharedJoinHandle(Arc<(Mutex<SharedJoinState>, Condvar)>);

impl SharedJoinHandle {
    #[cfg(any(windows, test))]
    pub(crate) fn new(handle: JoinHandle<()>) -> Self {
        Self(Arc::new((
            Mutex::new(SharedJoinState {
                handle: Some(handle),
                joining: false,
                joined: None,
            }),
            Condvar::new(),
        )))
    }

    #[cfg(any(windows, test))]
    pub(crate) fn is_os_finished(&self) -> bool {
        let (lock, _) = &*self.0;
        let st = lock.lock().unwrap_or_else(|e| e.into_inner());
        if st.joining {
            false
        } else if st.joined == Some(true) {
            true
        } else {
            st.handle.as_ref().is_some_and(is_os_thread_finished)
        }
    }

    #[cfg(any(windows, test))]
    pub(crate) fn try_join_if_os_finished(&self) -> bool {
        let (lock, cv) = &*self.0;
        let done = {
            let mut st = lock.lock().unwrap_or_else(|e| e.into_inner());
            if st.joining {
                return false;
            }
            if st.joined == Some(true) {
                return true;
            }
            if st.handle.as_ref().is_some_and(is_os_thread_finished) {
                st.joining = true;
                st.handle.take()
            } else {
                return false;
            }
        };
        if let Some(h) = done {
            let _ = h.join();
            let mut st = lock.lock().unwrap_or_else(|e| e.into_inner());
            st.joining = false;
            st.joined = Some(true);
            cv.notify_all();
        }
        true
    }

    pub(crate) fn join_bounded(&self, deadline: Instant) -> bool {
        let (lock, cv) = &*self.0;
        let mut st = lock.lock().unwrap_or_else(|e| e.into_inner());
        if st.joined == Some(true) {
            return true;
        }
        if !st.joining
            && let Some(h) = st.handle.take()
        {
            st.joining = true;
            drop(st);
            let res = try_join_handle_bounded(h, deadline);
            let mut st = lock.lock().unwrap_or_else(|e| e.into_inner());
            st.joining = false;
            let ok = match res {
                Ok(()) => {
                    st.joined = Some(true);
                    true
                }
                Err(still_running) => {
                    // Put the still-running `JoinHandle` back so neither
                    // `is_os_finished()` nor a later waiter mistakes it for finished.
                    st.handle = Some(still_running);
                    st.joined = Some(false);
                    false
                }
            };
            cv.notify_all();
            return ok;
        }
        // Another caller already took the handle: wait until its join completes
        // (within our own deadline) and propagate the actual join outcome.
        while st.joining {
            let Some(rem) = deadline.checked_duration_since(Instant::now()) else {
                return false;
            };
            if rem.is_zero() {
                return false;
            }
            let (guard, wait_res) = cv
                .wait_timeout(st, rem)
                .unwrap_or_else(|e| e.into_inner());
            st = guard;
            if wait_res.timed_out() && st.joining {
                return false;
            }
        }
        st.joined == Some(true)
    }
}

/// Coordinates worker thread spawning and bounded shutdown joining across BOTH
/// DualSense HID device workers and the WASAPI 4-channel audio haptics worker.
///
/// Never locked from crash/panic/Ctrl+C `emergency_reset_all`.
#[derive(Default)]
struct ShutdownState {
    shutting_down: bool,
    live_workers: usize,
    pending_registrations: usize,
    joins_in_progress: usize,
    all_joined: Option<bool>,
    handles: Vec<SharedJoinHandle>,
}

pub(crate) struct WorkerCoordinator {
    state: Mutex<ShutdownState>,
    changed: Condvar,
}

impl WorkerCoordinator {
    fn new() -> Self {
        Self {
            state: Mutex::new(ShutdownState::default()),
            changed: Condvar::new(),
        }
    }

    /// Atomically reserves a live-worker slot AND a `pending_registrations`
    /// barrier BEFORE spawning a thread. Returns `None` if shutdown has already
    /// started, preventing any post-shutdown worker spawn.
    ///
    /// Completing the returned [`SpawnTicket::spawn`] pushes the worker's
    /// [`SharedJoinHandle`] into `WorkerCoordinator` and clears the pending
    /// barrier; dropping the ticket early or failing `spawn` cleans up both
    /// counters automatically so `shutdown_and_wait` never misses a worker's
    /// OS-thread handle even if the worker exits before registration completes.
    #[cfg(any(windows, test))]
    pub(crate) fn begin_spawn(self: &Arc<Self>) -> Option<SpawnTicket> {
        let mut st = self.state.lock().unwrap_or_else(|e| e.into_inner());
        if st.shutting_down {
            return None;
        }
        st.live_workers += 1;
        st.pending_registrations += 1;
        Some(SpawnTicket {
            coord: Arc::clone(self),
            reg: Some(WorkerRegistration {
                coord: Arc::clone(self),
            }),
            pending_active: true,
        })
    }

    /// Marks shutdown active, calls `wake_workers`, and waits up to `timeout`
    /// for `live_workers == 0 && pending_registrations == 0` AND for every
    /// stored OS thread handle (both HID and audio-haptics workers) to signal
    /// within the same bounded deadline (never calling unbounded `.join()`).
    /// Every concurrent caller waits until OS-thread joins actually finish (or
    /// its own deadline expires) and never returns `true` prematurely while
    /// another caller is still joining handles.
    ///
    /// Returns `true` if all registered workers exited cleanly within `timeout`.
    pub(crate) fn shutdown_and_wait(
        &self,
        timeout: Duration,
        wake_workers: impl FnOnce(),
    ) -> bool {
        let deadline = Instant::now() + timeout;
        {
            let mut st = self.state.lock().unwrap_or_else(|e| e.into_inner());
            st.shutting_down = true;
        }
        wake_workers();

        let mut st = self.state.lock().unwrap_or_else(|e| e.into_inner());
        while st.live_workers > 0 || st.pending_registrations > 0 {
            let Some(rem) = deadline.checked_duration_since(Instant::now()) else {
                return false;
            };
            if rem.is_zero() {
                return false;
            }
            let (guard, wait_res) = self
                .changed
                .wait_timeout(st, rem)
                .unwrap_or_else(|e| e.into_inner());
            st = guard;
            if wait_res.timed_out() && (st.live_workers > 0 || st.pending_registrations > 0) {
                return false;
            }
        }
        // `live_workers == 0 && pending_registrations == 0`: if handles remain,
        // this caller joins them while advertising `joins_in_progress += 1` so
        // concurrent callers wait until OS-thread joins complete rather than
        // observing an empty `handles` vec and returning `true` early.
        if !st.handles.is_empty() {
            let handles = std::mem::take(&mut st.handles);
            st.joins_in_progress += 1;
            drop(st);
            let mut joined_ok = true;
            let mut unjoined = Vec::new();
            for h in handles {
                if !h.join_bounded(deadline) {
                    joined_ok = false;
                    unjoined.push(h);
                }
            }
            let mut st = self.state.lock().unwrap_or_else(|e| e.into_inner());
            if !unjoined.is_empty() {
                st.handles.extend(unjoined);
            }
            st.joins_in_progress = st.joins_in_progress.saturating_sub(1);
            st.all_joined = Some(st.handles.is_empty() && joined_ok);
            self.changed.notify_all();
            while st.joins_in_progress > 0 {
                let Some(rem) = deadline.checked_duration_since(Instant::now()) else {
                    return false;
                };
                if rem.is_zero() {
                    return false;
                }
                let (guard, wait_res) = self
                    .changed
                    .wait_timeout(st, rem)
                    .unwrap_or_else(|e| e.into_inner());
                st = guard;
                if wait_res.timed_out() && st.joins_in_progress > 0 {
                    return false;
                }
            }
            return st.all_joined.unwrap_or(true);
        }
        // Another caller already drained `handles` (or zero workers ever spawned):
        // wait until `joins_in_progress == 0` within our deadline and return the
        // true OS-join outcome.
        while st.joins_in_progress > 0 {
            let Some(rem) = deadline.checked_duration_since(Instant::now()) else {
                return false;
            };
            if rem.is_zero() {
                return false;
            }
            let (guard, wait_res) = self
                .changed
                .wait_timeout(st, rem)
                .unwrap_or_else(|e| e.into_inner());
            st = guard;
            if wait_res.timed_out() && st.joins_in_progress > 0 {
                return false;
            }
        }
        st.all_joined.unwrap_or(true)
    }
}

#[cfg(any(windows, test))]
pub(crate) struct SpawnTicket {
    coord: Arc<WorkerCoordinator>,
    reg: Option<WorkerRegistration>,
    pending_active: bool,
}

#[cfg(any(windows, test))]
impl SpawnTicket {
    /// Spawns the named worker thread, registers its `SharedJoinHandle` with
    /// `WorkerCoordinator`, and clears the `pending_registrations` barrier.
    pub(crate) fn spawn(
        mut self,
        name: String,
        f: impl FnOnce(WorkerRegistration) + Send + 'static,
    ) -> Option<SharedJoinHandle> {
        let reg = self.reg.take().expect("SpawnTicket still holds reg");
        let handle = std::thread::Builder::new()
            .name(name)
            .spawn(move || f(reg))
            .ok()?;
        let shared = SharedJoinHandle::new(handle);
        let finished: Vec<SharedJoinHandle> = {
            let mut st = self.coord.state.lock().unwrap_or_else(|e| e.into_inner());
            let mut done = Vec::new();
            let mut i = 0;
            while i < st.handles.len() {
                if st.handles[i].is_os_finished() {
                    done.push(st.handles.swap_remove(i));
                } else {
                    i += 1;
                }
            }
            st.handles.push(shared.clone());
            self.pending_active = false;
            st.pending_registrations = st.pending_registrations.saturating_sub(1);
            self.coord.changed.notify_all();
            done
        };
        for h in finished {
            let _ = h.try_join_if_os_finished();
        }
        Some(shared)
    }
}

#[cfg(any(windows, test))]
impl Drop for SpawnTicket {
    fn drop(&mut self) {
        // Drop `reg` first (if spawn was never attempted) so `live_workers`
        // decrements before clearing `pending_registrations`.
        drop(self.reg.take());
        if self.pending_active {
            let mut st = self.coord.state.lock().unwrap_or_else(|e| e.into_inner());
            st.pending_registrations = st.pending_registrations.saturating_sub(1);
            self.coord.changed.notify_all();
        }
    }
}

#[cfg(any(windows, test))]
pub(crate) struct WorkerRegistration {
    coord: Arc<WorkerCoordinator>,
}

#[cfg(any(windows, test))]
impl Drop for WorkerRegistration {
    fn drop(&mut self) {
        let mut st = self.coord.state.lock().unwrap_or_else(|e| e.into_inner());
        st.live_workers = st.live_workers.saturating_sub(1);
        self.coord.changed.notify_all();
    }
}

static WORKERS: LazyLock<Arc<WorkerCoordinator>> =
    LazyLock::new(|| Arc::new(WorkerCoordinator::new()));

#[cfg(windows)]
pub(crate) fn worker_coordinator() -> &'static Arc<WorkerCoordinator> {
    &WORKERS
}

const SLOT_HUNG: u32 = 1 << 31;
const SLOT_CLOSING: u32 = 1 << 30;
const SLOT_WRITING: u32 = 1 << 29;
const SLOT_EMERGENCY: u32 = 1 << 28;
const SLOT_CLOSED: u32 = 1 << 27;
const SLOT_BORROW_MASK: u32 = (1 << 27) - 1;

const PHASE_PRIMARY_NEUTRAL: u32 = 1 << 30;
const PHASE_FALLBACK: u32 = 1 << 29;
const PHASE_CANCEL_MASK: u32 = (1 << 29) - 1;

#[cfg(windows)]
struct EmergencyIo {
    ov: MaybeUninit<windows_sys::Win32::System::IO::OVERLAPPED>,
    buf: [u8; MAX_EMERGENCY_REPORT_LEN],
}

#[cfg(windows)]
impl EmergencyIo {
    const fn new() -> Self {
        Self {
            ov: MaybeUninit::zeroed(),
            buf: [0u8; MAX_EMERGENCY_REPORT_LEN],
        }
    }
}

/// Lock-free per-slot handle, serialization, and static emergency I/O state
/// shared between a device worker thread and `emergency_reset_all`.
///
/// Invariants:
/// - A worker never closes its Win32 HID handle or events while any emergency
///   or cancel borrow holds a reference, and leaks them permanently if
///   `SLOT_HUNG` is set or a borrow fails to drain within budget.
/// - Normal worker writes cannot start while `SLOT_EMERGENCY`, `SLOT_CLOSING`,
///   `SLOT_CLOSED`, `SLOT_HUNG`, `emergency_in_flight`, or `SHUTDOWN_REQUESTED` is set.
/// - `io` lives in `'static` memory and is mutated exclusively by the single
///   holder of `SLOT_EMERGENCY` when `!SLOT_HUNG`; `fallback_io` lives in
///   `'static` memory and is mutated exclusively by the holder of
///   `PHASE_FALLBACK` in `io_phase`.
/// - `io_phase` coordinates ALL `CancelIoEx` calls (primary reset, fatal
///   fallback, and `cancel_all_open_handles`) against both primary and fallback
///   neutral writes so no `CancelIoEx` can cancel an in-flight neutral packet.
struct EmergencySlot {
    handle: AtomicUsize,
    emergency_event: AtomicUsize,
    fallback_event: AtomicUsize,
    transport: AtomicU8,
    out_len: AtomicU16,
    fw_ver: AtomicU16,
    bt_seq: AtomicU8,
    reset_epoch: AtomicU32,
    /// Dekker/`SeqCst` flag indicating primary emergency reset is actively driving a pass.
    emergency_in_flight: AtomicBool,
    /// Coordinates `PHASE_PRIMARY_NEUTRAL`, `PHASE_FALLBACK`, and active
    /// `PHASE_CANCEL_MASK` `CancelIoEx` calls lock-free on this slot.
    io_phase: AtomicU32,
    /// Monotonic counter incremented right AFTER each `CancelIoEx` returns
    /// (before decrementing `PHASE_CANCEL_MASK`) so neutral senders detect any
    /// delayed pre-existing `CancelIoEx` that overlapped their write.
    cancel_completed_seq: AtomicU32,
    /// Dekker handoff flag set when a neutral sender finishes while a delayed
    /// pre-existing `CancelIoEx` still holds `PHASE_CANCEL_MASK != 0`, handing
    /// off the guaranteed uncancelled trailing neutral to the last `CancelIoGuard`.
    neutral_after_cancel: AtomicBool,
    /// Strictly monotonic neutral request ticket counter (`SeqCst`, never reset).
    neutral_requested_seq: AtomicU64,
    /// Strictly monotonic neutral completion counter (`SeqCst`, never reset),
    /// advanced via `fetch_max(target)` ONLY after an uncancelled `Delivered`
    /// send with zero concurrent worker active write.
    neutral_completed_seq: AtomicU64,
    /// Highest requested ticket that requires `release_leds = true` (`SeqCst`).
    release_leds_requested_seq: AtomicU64,
    /// Highest ticket satisfied by a confirmed uncancelled `Delivered` send
    /// with `release_leds = true` and zero concurrent worker active write (`SeqCst`).
    release_leds_completed_seq: AtomicU64,
    state: AtomicU32,
    #[cfg(windows)]
    io: UnsafeCell<EmergencyIo>,
    #[cfg(windows)]
    fallback_io: UnsafeCell<EmergencyIo>,
}

// SAFETY: all metadata fields are atomic; `io` is accessed exclusively while
// holding `SLOT_EMERGENCY` with `SLOT_HUNG == 0`, `fallback_io` is accessed
// exclusively while holding `PHASE_FALLBACK` with `SLOT_HUNG == 0`, and once
// `SLOT_HUNG` is set neither buffer is ever touched again for the process lifetime.
unsafe impl Send for EmergencySlot {}
// SAFETY: as above.
unsafe impl Sync for EmergencySlot {}

impl EmergencySlot {
    const fn new() -> Self {
        Self {
            handle: AtomicUsize::new(0),
            emergency_event: AtomicUsize::new(0),
            fallback_event: AtomicUsize::new(0),
            transport: AtomicU8::new(0),
            out_len: AtomicU16::new(USB_OUTPUT_REPORT_LEN as u16),
            fw_ver: AtomicU16::new(FIRMWARE_VIBRATION2_MIN),
            bt_seq: AtomicU8::new(0),
            reset_epoch: AtomicU32::new(0),
            emergency_in_flight: AtomicBool::new(false),
            io_phase: AtomicU32::new(0),
            cancel_completed_seq: AtomicU32::new(0),
            neutral_after_cancel: AtomicBool::new(false),
            neutral_requested_seq: AtomicU64::new(0),
            neutral_completed_seq: AtomicU64::new(0),
            release_leds_requested_seq: AtomicU64::new(0),
            release_leds_completed_seq: AtomicU64::new(0),
            state: AtomicU32::new(SLOT_CLOSING | SLOT_CLOSED),
            #[cfg(windows)]
            io: UnsafeCell::new(EmergencyIo::new()),
            #[cfg(windows)]
            fallback_io: UnsafeCell::new(EmergencyIo::new()),
        }
    }

    #[cfg(any(windows, test))]
    fn is_hung(&self) -> bool {
        self.state.load(Ordering::SeqCst) & SLOT_HUNG != 0
    }

    #[cfg(any(windows, test))]
    fn mark_hung(&self) {
        self.state.fetch_or(SLOT_HUNG, Ordering::SeqCst);
    }

    /// Issues a strictly monotonic neutral request ticket (`SeqCst`).
    #[cfg(any(windows, test))]
    fn request_neutral(&self, release_leds: bool) -> u64 {
        let req = self.neutral_requested_seq.fetch_add(1, Ordering::SeqCst) + 1;
        if release_leds {
            self.release_leds_requested_seq
                .fetch_max(req, Ordering::SeqCst);
        }
        req
    }

    /// Returns `true` iff any neutral or LED-release ticket is still unsatisfied.
    #[cfg(any(windows, test))]
    fn has_pending_neutral(&self) -> bool {
        self.neutral_requested_seq.load(Ordering::SeqCst)
            > self.neutral_completed_seq.load(Ordering::SeqCst)
            || self.release_leds_requested_seq.load(Ordering::SeqCst)
                > self.release_leds_completed_seq.load(Ordering::SeqCst)
    }

    /// Single-atomic fatal confirmation check: `release_leds_completed_seq` can
    /// ONLY reach `>= my_req` after a confirmed uncancelled `Delivered` send
    /// whose `target >= my_req`, `want_release_leds == true`, and zero
    /// concurrent worker active write (`SLOT_WRITING == 0` before & after, or
    /// caller holds `SLOT_WRITING`).
    #[cfg(any(windows, test))]
    fn is_fatal_confirmed(&self, my_req: u64) -> bool {
        !self.is_hung() && self.release_leds_completed_seq.load(Ordering::SeqCst) >= my_req
    }

    #[cfg(any(windows, test))]
    fn publish(
        &self,
        handle: usize,
        emergency_event: usize,
        transport: Transport,
        out_len: usize,
    ) -> bool {
        if self.is_hung() {
            return false;
        }
        self.transport
            .store(u8::from(transport == Transport::Bluetooth), Ordering::Relaxed);
        self.out_len
            .store(out_len.min(u16::MAX as usize) as u16, Ordering::Relaxed);
        self.fw_ver
            .store(FIRMWARE_VIBRATION2_MIN, Ordering::Relaxed);
        self.bt_seq.store(0, Ordering::Relaxed);
        self.emergency_in_flight.store(false, Ordering::SeqCst);
        self.io_phase.store(0, Ordering::SeqCst);
        self.cancel_completed_seq.store(0, Ordering::SeqCst);
        self.neutral_after_cancel.store(false, Ordering::SeqCst);
        // Keep ticket counters strictly monotonic across reuse while starting
        // each newly opened handle with zero stale pending requests.
        let req = self.neutral_requested_seq.load(Ordering::SeqCst);
        self.neutral_completed_seq.fetch_max(req, Ordering::SeqCst);
        let led_req = self.release_leds_requested_seq.load(Ordering::SeqCst);
        self.release_leds_completed_seq
            .fetch_max(led_req, Ordering::SeqCst);
        self.emergency_event
            .store(emergency_event, Ordering::Relaxed);
        self.handle.store(handle, Ordering::Relaxed);
        // Release-publish: makes all fields visible before `SLOT_CLOSING`/`SLOT_CLOSED` clear.
        self.state.store(0, Ordering::Release);
        true
    }

    /// Runs `cancel_fn(handle)` ONLY when neither a primary neutral write nor a
    /// fatal fallback neutral write is currently in flight on this slot, while
    /// holding an active cancel count in `PHASE_CANCEL_MASK`.
    ///
    /// On drop (strictly AFTER `cancel_fn` returns and `PHASE_CANCEL_MASK` hits `0`),
    /// completes the Dekker handoff for `neutral_after_cancel` by invoking
    /// `on_neutral_after_cancel` if a neutral sender was overlapped by this delayed cancel.
    #[cfg(any(windows, test))]
    fn cancel_io_if_no_neutral_in_flight(
        &self,
        handle: usize,
        cancel_fn: impl FnOnce(usize),
        on_neutral_after_cancel: impl FnOnce(),
    ) -> bool {
        if handle == 0 {
            return false;
        }
        loop {
            let cur = self.io_phase.load(Ordering::SeqCst);
            if cur & (PHASE_PRIMARY_NEUTRAL | PHASE_FALLBACK) != 0 {
                return false;
            }
            if (cur & PHASE_CANCEL_MASK) == PHASE_CANCEL_MASK {
                return false;
            }
            if self
                .io_phase
                .compare_exchange_weak(cur, cur + 1, Ordering::SeqCst, Ordering::SeqCst)
                .is_ok()
            {
                break;
            }
        }
        struct CancelIoGuard<'a, F: FnOnce()> {
            slot: &'a EmergencySlot,
            on_neutral_after_cancel: Option<F>,
        }
        impl<F: FnOnce()> Drop for CancelIoGuard<'_, F> {
            fn drop(&mut self) {
                // Dekker C1: record completion and decrement `PHASE_CANCEL_MASK`.
                self.slot.cancel_completed_seq.fetch_add(1, Ordering::SeqCst);
                let prev = self.slot.io_phase.fetch_sub(1, Ordering::SeqCst);
                let remaining = (prev & PHASE_CANCEL_MASK).saturating_sub(1);
                // Dekker C2: when the last active cancel finishes, honor any
                // post-cancel neutral handed off while we were delayed.
                if remaining == 0
                    && self.slot.neutral_after_cancel.swap(false, Ordering::SeqCst)
                    && !self.slot.is_hung()
                    && let Some(f) = self.on_neutral_after_cancel.take()
                {
                    f();
                }
            }
        }
        let _guard = CancelIoGuard {
            slot: self,
            on_neutral_after_cancel: Some(on_neutral_after_cancel),
        };
        cancel_fn(handle);
        true
    }

    /// Waits boundedly up to `budget` for any pre-existing `CancelIoEx` calls
    /// in `PHASE_CANCEL_MASK` to finish. Because `PHASE_PRIMARY_NEUTRAL` or
    /// `PHASE_FALLBACK` is already held, no NEW `CancelIoEx` can enter while
    /// waiting. Returns `true` once `PHASE_CANCEL_MASK == 0`.
    #[cfg(any(windows, test))]
    fn wait_active_cancels(&self, budget: Duration) -> bool {
        let deadline = Instant::now() + budget;
        let mut spins = 0u32;
        while (self.io_phase.load(Ordering::SeqCst) & PHASE_CANCEL_MASK) != 0 {
            if Instant::now() >= deadline {
                return false;
            }
            spins += 1;
            if spins < 64 {
                std::thread::yield_now();
            } else {
                std::thread::sleep(Duration::from_millis(1));
            }
        }
        true
    }

    /// Atomically acquires `bit` (`PHASE_PRIMARY_NEUTRAL` or `PHASE_FALLBACK`)
    /// in `io_phase` (`SeqCst`) ONLY while `!is_hung()` and neither neutral
    /// phase bit is held (`io_phase & (PHASE_PRIMARY_NEUTRAL | PHASE_FALLBACK) == 0`).
    /// Zero spinning/timeouts — callers pair this with the deterministic `SeqCst`
    /// Dekker handoff in [`Self::deliver_or_handoff_neutral`].
    #[cfg(any(windows, test))]
    fn try_acquire_neutral_phase_immediate(&self, bit: u32) -> Option<NeutralPhaseGuard<'_>> {
        debug_assert!(bit == PHASE_PRIMARY_NEUTRAL || bit == PHASE_FALLBACK);
        loop {
            if self.is_hung() {
                return None;
            }
            let cur = self.io_phase.load(Ordering::SeqCst);
            if cur & (PHASE_PRIMARY_NEUTRAL | PHASE_FALLBACK) != 0 {
                return None;
            }
            if self
                .io_phase
                .compare_exchange_weak(cur, cur | bit, Ordering::SeqCst, Ordering::SeqCst)
                .is_ok()
            {
                return Some(NeutralPhaseGuard {
                    slot: self,
                    bit,
                    hung: false,
                });
            }
        }
    }

    /// Exclusively acquires `PHASE_PRIMARY_NEUTRAL` immediately via `SeqCst` CAS
    /// only while `PHASE_FALLBACK == 0` and `PHASE_PRIMARY_NEUTRAL == 0`.
    #[cfg(any(windows, test))]
    fn try_acquire_primary_neutral(&self) -> Option<NeutralPhaseGuard<'_>> {
        self.try_acquire_neutral_phase_immediate(PHASE_PRIMARY_NEUTRAL)
    }

    /// Exclusively acquires `PHASE_FALLBACK` immediately via `SeqCst` CAS only
    /// while `PHASE_PRIMARY_NEUTRAL == 0` and `PHASE_FALLBACK == 0`.
    #[cfg(any(windows, test))]
    fn try_acquire_fallback_neutral(&self) -> Option<NeutralPhaseGuard<'_>> {
        self.try_acquire_neutral_phase_immediate(PHASE_FALLBACK)
    }

    /// Shared deterministic `SeqCst` Dekker neutral-delivery / handoff helper
    /// backed by the 4 monotonic ticket counters (`neutral_requested_seq`,
    /// `neutral_completed_seq`, `release_leds_requested_seq`,
    /// `release_leds_completed_seq`), used by `cancel_all_open_handles`,
    /// `run_emergency_reset` post-cancel callbacks, `run_serialized_write`'s
    /// preempted-write trailing neutral, and `run_device_worker`'s shutdown neutral.
    ///
    /// Zero timeout blind spots:
    /// 1. Attempts immediate non-blocking CAS acquisition of `phase_bit`.
    /// 2. If acquired: issues a ticket if needed, delivers neutral via
    ///    [`Self::send_neutral_with_cancel_retry_bounded`] (dynamically
    ///    snapshotting `want_release_leds` before EVERY send attempt), drains
    ///    any newly arriving tickets while held (aborting immediately on
    ///    exhausted `Failed` or `Hung`), drops `NeutralPhaseGuard` (`SeqCst`),
    ///    and performs a post-release `SeqCst` Dekker recheck.
    /// 3. If NOT acquired: publishes its monotonic ticket (`request_neutral`)
    ///    BEFORE checking `io_phase` / `emergency_in_flight` (`SeqCst`). If
    ///    all active owners already exited, loops back to acquire; otherwise
    ///    returns cleanly (the live owner's post-release check sees the ticket).
    #[cfg(any(windows, test))]
    fn deliver_or_handoff_neutral(
        &self,
        phase_bit: u32,
        release_leds: bool,
        send_fn: impl FnMut(bool, Option<Duration>) -> NeutralSendOutcome,
    ) -> NeutralSendOutcome {
        let mut attempts = MAX_NEUTRAL_ATTEMPTS;
        self.deliver_or_handoff_neutral_bounded(
            phase_bit,
            release_leds,
            None,
            None,
            &mut attempts,
            || {},
            send_fn,
        )
    }

    #[cfg(any(windows, test))]
    fn deliver_or_handoff_neutral_with_hook(
        &self,
        phase_bit: u32,
        release_leds: bool,
        existing_ticket: Option<u64>,
        before_dekker_store_hook: impl FnMut(),
        send_fn: impl FnMut(bool, Option<Duration>) -> NeutralSendOutcome,
    ) -> NeutralSendOutcome {
        let mut attempts = MAX_NEUTRAL_ATTEMPTS;
        self.deliver_or_handoff_neutral_bounded(
            phase_bit,
            release_leds,
            existing_ticket,
            None,
            &mut attempts,
            before_dekker_store_hook,
            send_fn,
        )
    }

    #[cfg(any(windows, test))]
    #[allow(clippy::too_many_arguments)]
    fn deliver_or_handoff_neutral_bounded(
        &self,
        phase_bit: u32,
        release_leds: bool,
        existing_ticket: Option<u64>,
        deadline: Option<Instant>,
        attempts_remaining: &mut u32,
        mut before_dekker_store_hook: impl FnMut(),
        mut send_fn: impl FnMut(bool, Option<Duration>) -> NeutralSendOutcome,
    ) -> NeutralSendOutcome {
        let mut ticket = existing_ticket;
        loop {
            if self.is_hung() {
                return NeutralSendOutcome::Hung;
            }
            if let Some(d) = deadline
                && Instant::now() >= d
            {
                return NeutralSendOutcome::Failed;
            }
            if let Some(mut guard) = self.try_acquire_neutral_phase_immediate(phase_bit) {
                let my_req = match ticket {
                    Some(req) => {
                        let satisfied = if release_leds {
                            self.release_leds_completed_seq.load(Ordering::SeqCst) >= req
                        } else {
                            self.neutral_completed_seq.load(Ordering::SeqCst) >= req
                        };
                        if satisfied && !self.has_pending_neutral() {
                            drop(guard);
                            if !self.is_hung() && self.has_pending_neutral() {
                                continue;
                            }
                            return NeutralSendOutcome::Delivered;
                        }
                        req
                    }
                    None => {
                        let req = self.request_neutral(release_leds);
                        ticket = Some(req);
                        req
                    }
                };
                let (last_outcome, seq_snap, led_snap) = loop {
                    if *attempts_remaining == 0 {
                        break (
                            SendAttemptOutcome::Failed,
                            self.neutral_requested_seq.load(Ordering::SeqCst),
                            self.release_leds_requested_seq.load(Ordering::SeqCst),
                        );
                    }
                    let seq_before = self.neutral_requested_seq.load(Ordering::SeqCst);
                    let led_before = self.release_leds_requested_seq.load(Ordering::SeqCst);
                    let outcome = self.send_neutral_with_cancel_retry_bounded(
                        deadline,
                        false,
                        attempts_remaining,
                        &mut send_fn,
                    );
                    if outcome == SendAttemptOutcome::Hung {
                        guard.hung = true;
                        break (outcome, seq_before, led_before);
                    }
                    let newer = (self.neutral_requested_seq.load(Ordering::SeqCst) > seq_before
                        || self.release_leds_requested_seq.load(Ordering::SeqCst) > led_before)
                        && self.has_pending_neutral();
                    let post_cancel_ticket_ready = outcome == SendAttemptOutcome::HandedOffToCancel
                        && (self.io_phase.load(Ordering::SeqCst) & PHASE_CANCEL_MASK) == 0
                        && newer
                        && *attempts_remaining > 0;
                    if post_cancel_ticket_ready {
                        continue;
                    }
                    if outcome != SendAttemptOutcome::Delivered || !newer {
                        break (outcome, seq_before, led_before);
                    }
                    if deadline.is_none() {
                        *attempts_remaining = MAX_NEUTRAL_ATTEMPTS;
                    }
                };
                drop(guard);
                if last_outcome == SendAttemptOutcome::Hung {
                    return NeutralSendOutcome::Hung;
                }
                // Post-release Dekker recheck (`SeqCst` strictly AFTER clearing `phase_bit`):
                // re-loop for newly published unsatisfied tickets after a successful send OR
                // when `CancelIoGuard::drop` finished right while we held `phase_bit` and
                // re-requested a post-cancel ticket (`PHASE_CANCEL_MASK == 0`).
                let newer_after_drop = (self.neutral_requested_seq.load(Ordering::SeqCst)
                    > seq_snap
                    || self.release_leds_requested_seq.load(Ordering::SeqCst) > led_snap)
                    && self.has_pending_neutral();
                let post_cancel_ready_after_drop = last_outcome
                    == SendAttemptOutcome::HandedOffToCancel
                    && (self.io_phase.load(Ordering::SeqCst) & PHASE_CANCEL_MASK) == 0
                    && newer_after_drop
                    && *attempts_remaining > 0;
                if (last_outcome == SendAttemptOutcome::Delivered && newer_after_drop)
                    || post_cancel_ready_after_drop
                {
                    if deadline.is_none() && last_outcome == SendAttemptOutcome::Delivered {
                        *attempts_remaining = MAX_NEUTRAL_ATTEMPTS;
                    }
                    continue;
                }
                let _ = my_req;
                return last_outcome.into_public();
            }

            before_dekker_store_hook();

            // Dekker Step 1: publish monotonic ticket BEFORE observing active state.
            let my_req = match ticket {
                Some(req) => {
                    if release_leds {
                        self.release_leds_requested_seq
                            .fetch_max(req, Ordering::SeqCst);
                    }
                    req
                }
                None => {
                    let req = self.request_neutral(release_leds);
                    ticket = Some(req);
                    req
                }
            };

            // Dekker Step 2: check if all active neutral owners cleared right before Step 1.
            let busy = (self.io_phase.load(Ordering::SeqCst)
                & (PHASE_PRIMARY_NEUTRAL | PHASE_FALLBACK))
                != 0
                || self.emergency_in_flight.load(Ordering::SeqCst);
            if !busy && !self.is_hung() {
                continue;
            }
            let satisfied = if release_leds {
                self.release_leds_completed_seq.load(Ordering::SeqCst) >= my_req
            } else {
                self.neutral_completed_seq.load(Ordering::SeqCst) >= my_req
            };
            return if satisfied {
                NeutralSendOutcome::Delivered
            } else {
                NeutralSendOutcome::Failed
            };
        }
    }

    /// Routes the worker thread's final shutdown neutral packet (`release_leds = true`)
    /// through the shared `io_phase` / monotonic-ticket Dekker path before
    /// `RegisteredHidHandle::drop` closes the slot.
    #[cfg(any(windows, test))]
    fn run_worker_shutdown_neutral(
        &self,
        send_fn: impl FnMut(bool, Option<Duration>) -> NeutralSendOutcome,
    ) -> NeutralSendOutcome {
        self.deliver_or_handoff_neutral(PHASE_PRIMARY_NEUTRAL, true, send_fn)
    }

    /// Sends neutral reports while holding `PHASE_PRIMARY_NEUTRAL` or
    /// `PHASE_FALLBACK` (blocking new `CancelIoEx` calls), enforcing:
    /// - Dynamic `want_release_leds` snapshot (`release_leds_requested_seq > release_leds_completed_seq`)
    ///   and exact `target` snapshot BEFORE EVERY send attempt (never stale `release_leds = false`).
    /// - Strict `deadline` enforcement BEFORE EVERY `send_fn` call, passing
    ///   `remaining: Option<Duration>` directly to `send_fn(want_release_leds, remaining)`.
    /// - Strict attempt cap via `attempts_remaining: &mut u32`.
    /// - Immediate abort on `NeutralSendOutcome::Hung` (0 further I/O).
    /// - Delayed-cancel Dekker handoff via `neutral_after_cancel` when `PHASE_CANCEL_MASK != 0`.
    /// - Advances `neutral_completed_seq` (and `release_leds_completed_seq` when
    ///   `want_release_leds`) via `fetch_max(target, SeqCst)` ONLY on uncancelled
    ///   `NeutralSendOutcome::Delivered` with zero concurrent worker active write
    ///   (`worker_write_excluded || SLOT_WRITING == 0` both before and after `send_fn`).
    #[cfg(test)]
    fn send_neutral_with_cancel_retry(
        &self,
        send_fn: impl FnMut(bool, Option<Duration>) -> NeutralSendOutcome,
    ) -> NeutralSendOutcome {
        let mut attempts = MAX_NEUTRAL_ATTEMPTS;
        self.send_neutral_with_cancel_retry_bounded(None, false, &mut attempts, send_fn)
            .into_public()
    }

    #[cfg(any(windows, test))]
    fn send_neutral_with_cancel_retry_bounded(
        &self,
        deadline: Option<Instant>,
        worker_write_excluded: bool,
        attempts_remaining: &mut u32,
        mut send_fn: impl FnMut(bool, Option<Duration>) -> NeutralSendOutcome,
    ) -> SendAttemptOutcome {
        const CANCEL_DRAIN_BUDGET: Duration = Duration::from_millis(25);
        const FAILED_NEUTRAL_RETRY_DELAY: Duration = Duration::from_millis(2);

        let check_remaining = |deadline: Option<Instant>| -> Option<Option<Duration>> {
            match deadline {
                Some(d) => {
                    let rem = d.checked_duration_since(Instant::now())?;
                    if rem.is_zero() { None } else { Some(Some(rem)) }
                }
                None => Some(None),
            }
        };

        if self.is_hung() {
            return SendAttemptOutcome::Hung;
        }
        let Some(rem_for_drain) = check_remaining(deadline) else {
            return SendAttemptOutcome::Failed;
        };
        let drain_budget = match rem_for_drain {
            Some(rem) => CANCEL_DRAIN_BUDGET.min(rem),
            None => CANCEL_DRAIN_BUDGET,
        };
        let drained = self.wait_active_cancels(drain_budget);

        if *attempts_remaining == 0 {
            return SendAttemptOutcome::Failed;
        }
        let Some(rem) = check_remaining(deadline) else {
            return SendAttemptOutcome::Failed;
        };
        *attempts_remaining -= 1;

        let mut led_req = self.release_leds_requested_seq.load(Ordering::SeqCst);
        let mut target = self.neutral_requested_seq.load(Ordering::SeqCst).max(led_req);
        let mut want_release_leds =
            led_req > self.release_leds_completed_seq.load(Ordering::SeqCst);
        let clear_pre =
            worker_write_excluded || (self.state.load(Ordering::SeqCst) & SLOT_WRITING) == 0;
        let seq_before = self.cancel_completed_seq.load(Ordering::SeqCst);
        let mut outcome = send_fn(want_release_leds, rem);
        if outcome == NeutralSendOutcome::Hung {
            return SendAttemptOutcome::Hung;
        }

        let mut attempt_writing_clear = clear_pre
            && (worker_write_excluded || (self.state.load(Ordering::SeqCst) & SLOT_WRITING) == 0);
        let overlapped = !drained
            || self.cancel_completed_seq.load(Ordering::SeqCst) != seq_before
            || (self.io_phase.load(Ordering::SeqCst) & PHASE_CANCEL_MASK) != 0;
        if overlapped {
            // Only retry immediately when `PHASE_CANCEL_MASK == 0`.
            if (self.io_phase.load(Ordering::SeqCst) & PHASE_CANCEL_MASK) == 0 {
                if *attempts_remaining == 0 {
                    return SendAttemptOutcome::Failed;
                }
                let Some(rem_retry) = check_remaining(deadline) else {
                    return SendAttemptOutcome::Failed;
                };
                *attempts_remaining -= 1;
                led_req = self.release_leds_requested_seq.load(Ordering::SeqCst);
                target = self.neutral_requested_seq.load(Ordering::SeqCst).max(led_req);
                want_release_leds =
                    led_req > self.release_leds_completed_seq.load(Ordering::SeqCst);
                let retry_clear_pre = worker_write_excluded
                    || (self.state.load(Ordering::SeqCst) & SLOT_WRITING) == 0;
                outcome = send_fn(want_release_leds, rem_retry);
                if outcome == NeutralSendOutcome::Hung {
                    return SendAttemptOutcome::Hung;
                }
                attempt_writing_clear = retry_clear_pre
                    && (worker_write_excluded
                        || (self.state.load(Ordering::SeqCst) & SLOT_WRITING) == 0);
            } else {
                // Dekker N1/N2: hand off to the last active `CancelIoGuard::drop`
                // (or send immediately if `PHASE_CANCEL_MASK` just drained to 0).
                self.neutral_after_cancel.store(true, Ordering::SeqCst);
                if (self.io_phase.load(Ordering::SeqCst) & PHASE_CANCEL_MASK) == 0
                    && self.neutral_after_cancel.swap(false, Ordering::SeqCst)
                {
                    if *attempts_remaining == 0 {
                        return SendAttemptOutcome::Failed;
                    }
                    let Some(rem_retry) = check_remaining(deadline) else {
                        return SendAttemptOutcome::Failed;
                    };
                    *attempts_remaining -= 1;
                    led_req = self.release_leds_requested_seq.load(Ordering::SeqCst);
                    target = self.neutral_requested_seq.load(Ordering::SeqCst).max(led_req);
                    want_release_leds =
                        led_req > self.release_leds_completed_seq.load(Ordering::SeqCst);
                    let retry_clear_pre = worker_write_excluded
                        || (self.state.load(Ordering::SeqCst) & SLOT_WRITING) == 0;
                    outcome = send_fn(want_release_leds, rem_retry);
                    if outcome == NeutralSendOutcome::Hung {
                        return SendAttemptOutcome::Hung;
                    }
                    attempt_writing_clear = retry_clear_pre
                        && (worker_write_excluded
                            || (self.state.load(Ordering::SeqCst) & SLOT_WRITING) == 0);
                } else {
                    // Handed off to delayed `CancelIoGuard::drop` (distinct from
                    // exhausted settled failures so if `CancelIoGuard::drop`
                    // already completed while we still held `io_phase` and
                    // published a new post-cancel ticket, our outer loop will
                    // immediately service it once `PHASE_CANCEL_MASK == 0`).
                    return SendAttemptOutcome::HandedOffToCancel;
                }
            }
        }

        if outcome == NeutralSendOutcome::Delivered {
            if attempt_writing_clear {
                if want_release_leds {
                    self.release_leds_completed_seq
                        .fetch_max(target, Ordering::SeqCst);
                }
                self.neutral_completed_seq
                    .fetch_max(target, Ordering::SeqCst);
            }
            return SendAttemptOutcome::Delivered;
        }

        // Settled non-hung `Failed` with `PHASE_CANCEL_MASK == 0`: bounded retries
        // strictly capped by `*attempts_remaining` and `deadline`.
        while *attempts_remaining > 0 {
            if self.is_hung() {
                return SendAttemptOutcome::Hung;
            }
            let Some(rem_before_sleep) = check_remaining(deadline) else {
                break;
            };
            match rem_before_sleep {
                Some(rem) => std::thread::sleep(FAILED_NEUTRAL_RETRY_DELAY.min(rem)),
                None => std::thread::sleep(FAILED_NEUTRAL_RETRY_DELAY),
            }
            if self.is_hung() {
                return SendAttemptOutcome::Hung;
            }
            // Re-check `deadline` strictly AFTER sleeping and BEFORE calling `send_fn`.
            let Some(rem_after_sleep) = check_remaining(deadline) else {
                break;
            };
            *attempts_remaining -= 1;
            let retry_led_req = self.release_leds_requested_seq.load(Ordering::SeqCst);
            let retry_target = self
                .neutral_requested_seq
                .load(Ordering::SeqCst)
                .max(retry_led_req);
            let retry_want_leds =
                retry_led_req > self.release_leds_completed_seq.load(Ordering::SeqCst);
            let retry_clear_pre =
                worker_write_excluded || (self.state.load(Ordering::SeqCst) & SLOT_WRITING) == 0;
            match send_fn(retry_want_leds, rem_after_sleep) {
                NeutralSendOutcome::Delivered => {
                    if retry_clear_pre
                        && (worker_write_excluded
                            || (self.state.load(Ordering::SeqCst) & SLOT_WRITING) == 0)
                    {
                        if retry_want_leds {
                            self.release_leds_completed_seq
                                .fetch_max(retry_target, Ordering::SeqCst);
                        }
                        self.neutral_completed_seq
                            .fetch_max(retry_target, Ordering::SeqCst);
                    }
                    return SendAttemptOutcome::Delivered;
                }
                NeutralSendOutcome::Hung => {
                    return SendAttemptOutcome::Hung;
                }
                NeutralSendOutcome::Failed => {}
            }
        }
        SendAttemptOutcome::Failed
    }

    #[cfg(windows)]
    fn set_fallback_event(&self, fallback_event: usize) {
        self.fallback_event.store(fallback_event, Ordering::Relaxed);
    }

    #[cfg(windows)]
    fn set_firmware_version(&self, fw: u16) {
        self.fw_ver.store(fw, Ordering::Release);
    }

    #[cfg(windows)]
    fn next_bt_seq(&self) -> u8 {
        self.bt_seq.fetch_add(1, Ordering::Relaxed) & 0x0F
    }

    #[cfg(any(windows, test))]
    fn try_acquire_worker_write(&self, shutdown_flag: &AtomicBool) -> Option<u32> {
        loop {
            if shutdown_flag.load(Ordering::SeqCst)
                || self.emergency_in_flight.load(Ordering::SeqCst)
            {
                return None;
            }
            let cur = self.state.load(Ordering::Acquire);
            if cur & (SLOT_HUNG | SLOT_CLOSING | SLOT_EMERGENCY) != 0 {
                return None;
            }
            if cur & SLOT_WRITING == 0 {
                let epoch = self.reset_epoch.load(Ordering::SeqCst);
                if self
                    .state
                    .compare_exchange_weak(
                        cur,
                        cur | SLOT_WRITING,
                        Ordering::AcqRel,
                        Ordering::Acquire,
                    )
                    .is_ok()
                {
                    if shutdown_flag.load(Ordering::SeqCst)
                        || self.emergency_in_flight.load(Ordering::SeqCst)
                        || self.state.load(Ordering::Acquire)
                            & (SLOT_HUNG | SLOT_CLOSING | SLOT_EMERGENCY)
                            != 0
                    {
                        self.release_writing();
                        return None;
                    }
                    return Some(epoch);
                }
            } else {
                std::thread::yield_now();
            }
        }
    }

    /// Runs a normal worker write serialized with [`Self::run_emergency_reset`]
    /// via monotonic `SeqCst` ticket counters so whichever side finishes last
    /// ALWAYS transmits a final neutral packet with dynamic `release_leds`
    /// upgrade (never capturing a stale `release_leds = false` before handoff)
    /// and preserves `Delivered` / `Failed` / `Hung` outcome reporting.
    #[cfg(any(windows, test))]
    fn run_serialized_write(
        &self,
        shutdown_flag: &AtomicBool,
        before_io_hook: impl FnOnce(),
        write_fn: impl FnOnce() -> bool,
        mut neutral_fn: impl FnMut(bool) -> NeutralSendOutcome,
    ) -> SerializedWriteResult {
        let Some(snapped_epoch) = self.try_acquire_worker_write(shutdown_flag) else {
            return SerializedWriteResult {
                write_ok: false,
                preempted: true,
                trailing_neutral: None,
            };
        };

        before_io_hook();

        let still_clear = self.reset_epoch.load(Ordering::SeqCst) == snapped_epoch
            && !shutdown_flag.load(Ordering::SeqCst)
            && !self.emergency_in_flight.load(Ordering::SeqCst)
            && (self.state.load(Ordering::Acquire) & (SLOT_HUNG | SLOT_CLOSING | SLOT_EMERGENCY))
                == 0;
        if !still_clear {
            self.release_writing();
            return SerializedWriteResult {
                write_ok: false,
                preempted: true,
                trailing_neutral: None,
            };
        }

        let ok = write_fn();

        let preempted = self.reset_epoch.load(Ordering::SeqCst) != snapped_epoch
            || self.emergency_in_flight.load(Ordering::SeqCst)
            || shutdown_flag.load(Ordering::SeqCst);
        if preempted {
            self.release_writing();
            let mut trailing_neutral = None;
            if !self.is_hung() {
                let initial_release_leds = shutdown_flag.load(Ordering::SeqCst);
                // Dekker W1: publish monotonic ticket BEFORE checking `emergency_in_flight`.
                let req = self.request_neutral(initial_release_leds);
                // Dekker W2: if Emergency is still in flight, it will observe
                // `neutral_requested_seq >= req` after clearing `emergency_in_flight`.
                // Otherwise drive delivery through `deliver_or_handoff_neutral_with_hook`,
                // dynamically recomputing `release_leds` right before each send.
                if !self.emergency_in_flight.load(Ordering::SeqCst) {
                    trailing_neutral = Some(self.deliver_or_handoff_neutral_with_hook(
                        PHASE_PRIMARY_NEUTRAL,
                        initial_release_leds,
                        Some(req),
                        || {},
                        |want_release_leds, _rem| {
                            let release_leds =
                                want_release_leds || shutdown_flag.load(Ordering::SeqCst);
                            neutral_fn(release_leds)
                        },
                    ));
                }
            }
            return SerializedWriteResult {
                write_ok: false,
                preempted: true,
                trailing_neutral,
            };
        }

        self.release_writing();
        SerializedWriteResult {
            write_ok: ok,
            preempted: false,
            trailing_neutral: None,
        }
    }

    /// Executes an emergency neutral reset on this slot using the matching
    /// monotonic ticket Dekker half of [`Self::run_serialized_write`] and the
    /// default bounded fatal confirmation budget (`FATAL_RESET_BUDGET = 200 ms`).
    #[cfg(any(windows, test))]
    #[allow(clippy::too_many_arguments)]
    fn run_emergency_reset(
        &self,
        fatal: bool,
        claim_wait: Duration,
        write_wait: Duration,
        on_in_flight: impl FnMut(usize),
        send_neutral: impl FnMut(&EmergencyBorrow<'_>, bool, Option<Duration>) -> NeutralSendOutcome,
        send_fallback_neutral: impl FnMut(
            &CancelBorrow<'_>,
            bool,
            Option<Duration>,
        ) -> NeutralSendOutcome,
        after_first_neutral_hook: impl FnOnce(),
    ) -> ResetOutcome {
        self.run_emergency_reset_with_budget(
            fatal,
            claim_wait,
            write_wait,
            FATAL_RESET_BUDGET,
            on_in_flight,
            send_neutral,
            send_fallback_neutral,
            after_first_neutral_hook,
        )
    }

    /// Executes an emergency neutral reset with an explicit `fatal_budget`.
    ///
    /// When `fatal` is `true`:
    /// - Publishes a monotonic ticket `my_req = request_neutral(true)` (`SeqCst`).
    /// - Single-atomic confirmation check: `is_fatal_confirmed(my_req)`
    ///   (`!is_hung() && release_leds_completed_seq >= my_req`).
    /// - Strictly enforces `fatal_deadline` BEFORE EVERY `send_fn` call and
    ///   passes `Some(remaining)` into `send_fn` so overlapped HID waits never
    ///   overshoot `fatal_deadline`.
    /// - Strictly caps total HID write attempts across ALL outer/inner loops at
    ///   `MAX_NEUTRAL_ATTEMPTS = 3`.
    #[cfg(any(windows, test))]
    #[allow(clippy::too_many_arguments)]
    fn run_emergency_reset_with_budget(
        &self,
        fatal: bool,
        claim_wait: Duration,
        write_wait: Duration,
        fatal_budget: Duration,
        mut on_in_flight: impl FnMut(usize),
        mut send_neutral: impl FnMut(
            &EmergencyBorrow<'_>,
            bool,
            Option<Duration>,
        ) -> NeutralSendOutcome,
        mut send_fallback_neutral: impl FnMut(
            &CancelBorrow<'_>,
            bool,
            Option<Duration>,
        ) -> NeutralSendOutcome,
        after_first_neutral_hook: impl FnOnce(),
    ) -> ResetOutcome {
        if (self.state.load(Ordering::SeqCst) & SLOT_CLOSED) != 0 {
            return ResetOutcome::SlotIdle;
        }
        let my_req = self.request_neutral(fatal);
        let fatal_deadline = Instant::now() + fatal_budget;
        let effective_claim_wait = if fatal {
            claim_wait.min(fatal_budget / 2)
        } else {
            claim_wait
        };
        let mut attempts_remaining = MAX_NEUTRAL_ATTEMPTS;

        let Some(mut borrow) = self.try_claim_emergency(effective_claim_wait) else {
            if (self.state.load(Ordering::SeqCst) & SLOT_CLOSED) != 0 {
                return ResetOutcome::SlotIdle;
            }
            if !fatal {
                return if self.neutral_completed_seq.load(Ordering::SeqCst) >= my_req {
                    ResetOutcome::Confirmed
                } else {
                    ResetOutcome::Unconfirmed
                };
            }
            let Some(cancel_borrow) = self.try_borrow_handle_for_cancel() else {
                return if (self.state.load(Ordering::SeqCst) & SLOT_CLOSED) != 0 {
                    ResetOutcome::SlotIdle
                } else {
                    ResetOutcome::Unconfirmed
                };
            };
            if cancel_borrow.handle == 0 {
                return ResetOutcome::SlotIdle;
            }

            let mut issued_cancel = false;
            let mut sent_while_writing = false;
            loop {
                if self.is_hung() {
                    return ResetOutcome::Unconfirmed;
                }
                if self.is_fatal_confirmed(my_req) {
                    return ResetOutcome::Confirmed;
                }
                if Instant::now() >= fatal_deadline {
                    return ResetOutcome::Unconfirmed;
                }
                if let Some(mut guard) = self.try_acquire_fallback_neutral() {
                    if !issued_cancel {
                        issued_cancel = true;
                        let drain_rem = Duration::from_millis(25)
                            .min(fatal_deadline.saturating_duration_since(Instant::now()));
                        let _ = self.wait_active_cancels(drain_rem);
                        on_in_flight(cancel_borrow.handle);
                    }
                    // Wait boundedly within `fatal_deadline` for any active
                    // worker write (`SLOT_WRITING`) to clear before sending.
                    while (self.state.load(Ordering::SeqCst) & SLOT_WRITING) != 0 {
                        if self.is_hung() || Instant::now() >= fatal_deadline {
                            break;
                        }
                        let rem = fatal_deadline.saturating_duration_since(Instant::now());
                        std::thread::sleep(Duration::from_millis(1).min(rem));
                    }
                    // Strictly enforce `fatal_deadline` BEFORE calling `send_fn`.
                    if Instant::now() >= fatal_deadline || self.is_hung() {
                        drop(guard);
                        return if self.is_fatal_confirmed(my_req) {
                            ResetOutcome::Confirmed
                        } else {
                            ResetOutcome::Unconfirmed
                        };
                    }
                    let writing_clear_before_send =
                        (self.state.load(Ordering::SeqCst) & SLOT_WRITING) == 0;
                    if !writing_clear_before_send && sent_while_writing {
                        drop(guard);
                        let rem = fatal_deadline.saturating_duration_since(Instant::now());
                        if rem.is_zero() {
                            return ResetOutcome::Unconfirmed;
                        }
                        std::thread::sleep(Duration::from_millis(1).min(rem));
                        continue;
                    }
                    if attempts_remaining == 0 {
                        drop(guard);
                        return if self.is_fatal_confirmed(my_req) {
                            ResetOutcome::Confirmed
                        } else {
                            ResetOutcome::Unconfirmed
                        };
                    }
                    let last_outcome = loop {
                        let outcome = self.send_neutral_with_cancel_retry_bounded(
                            Some(fatal_deadline),
                            false,
                            &mut attempts_remaining,
                            |want_release_leds, rem| {
                                send_fallback_neutral(&cancel_borrow, want_release_leds, rem)
                            },
                        );
                        if !writing_clear_before_send {
                            sent_while_writing = true;
                        }
                        if outcome == SendAttemptOutcome::Hung {
                            guard.hung = true;
                            break outcome;
                        }
                        let post_cancel_ready = outcome == SendAttemptOutcome::HandedOffToCancel
                            && (self.io_phase.load(Ordering::SeqCst) & PHASE_CANCEL_MASK) == 0
                            && self.has_pending_neutral()
                            && attempts_remaining > 0
                            && Instant::now() < fatal_deadline;
                        if post_cancel_ready {
                            continue;
                        }
                        if outcome != SendAttemptOutcome::Delivered
                            || !self.has_pending_neutral()
                            || attempts_remaining == 0
                            || Instant::now() >= fatal_deadline
                        {
                            break outcome;
                        }
                    };
                    drop(guard);
                    if last_outcome == SendAttemptOutcome::Hung {
                        return ResetOutcome::Unconfirmed;
                    }
                    if self.is_fatal_confirmed(my_req) {
                        return ResetOutcome::Confirmed;
                    }
                    if attempts_remaining == 0
                        || last_outcome == SendAttemptOutcome::Failed
                        || Instant::now() >= fatal_deadline
                    {
                        return ResetOutcome::Unconfirmed;
                    }
                    let rem = fatal_deadline.saturating_duration_since(Instant::now());
                    if rem.is_zero() {
                        return ResetOutcome::Unconfirmed;
                    }
                    if !(last_outcome == SendAttemptOutcome::HandedOffToCancel
                        && (self.io_phase.load(Ordering::SeqCst) & PHASE_CANCEL_MASK) == 0)
                    {
                        std::thread::sleep(Duration::from_millis(1).min(rem));
                    }
                    continue;
                }

                // Another owner holds `PHASE_PRIMARY_NEUTRAL` or `PHASE_FALLBACK`;
                // `my_req` is already published in `neutral_requested_seq` and
                // `release_leds_requested_seq`. Wait boundedly for confirmation
                // or phase release.
                if self.is_fatal_confirmed(my_req) {
                    return ResetOutcome::Confirmed;
                }
                if Instant::now() >= fatal_deadline || self.is_hung() {
                    return ResetOutcome::Unconfirmed;
                }
                if (self.io_phase.load(Ordering::SeqCst)
                    & (PHASE_PRIMARY_NEUTRAL | PHASE_FALLBACK))
                    != 0
                {
                    let rem = fatal_deadline.saturating_duration_since(Instant::now());
                    std::thread::sleep(Duration::from_millis(1).min(rem));
                }
            }
        };

        if borrow.handle == 0 || borrow.emergency_event == 0 {
            return ResetOutcome::SlotIdle;
        }
        let handle_usize = borrow.handle;
        self.emergency_in_flight.store(true, Ordering::SeqCst);
        if fatal {
            self.cancel_io_if_no_neutral_in_flight(handle_usize, &mut on_in_flight, || {
                let _ = self.deliver_or_handoff_neutral_bounded(
                    PHASE_PRIMARY_NEUTRAL,
                    true,
                    Some(my_req),
                    Some(fatal_deadline),
                    &mut attempts_remaining,
                    || {},
                    |want_release_leds, rem| send_neutral(&borrow, want_release_leds, rem),
                );
            });
        }
        let effective_write_wait = if fatal {
            write_wait.min(fatal_deadline.saturating_duration_since(Instant::now()))
        } else {
            write_wait
        };
        borrow.acquired_writing = self.wait_and_acquire_writing(effective_write_wait, || {
            if !fatal {
                self.cancel_io_if_no_neutral_in_flight(handle_usize, &mut on_in_flight, || {
                    let _ = self.deliver_or_handoff_neutral(
                        PHASE_PRIMARY_NEUTRAL,
                        false,
                        |want_release_leds, rem| send_neutral(&borrow, want_release_leds, rem),
                    );
                });
            }
        });
        if !borrow.acquired_writing {
            self.cancel_io_if_no_neutral_in_flight(handle_usize, &mut on_in_flight, || {
                let deadline_opt = fatal.then_some(fatal_deadline);
                let mut nonfatal_cb_attempts = MAX_NEUTRAL_ATTEMPTS;
                let cb_attempts = if fatal {
                    &mut attempts_remaining
                } else {
                    &mut nonfatal_cb_attempts
                };
                let _ = self.deliver_or_handoff_neutral_bounded(
                    PHASE_PRIMARY_NEUTRAL,
                    fatal,
                    Some(my_req),
                    deadline_opt,
                    cb_attempts,
                    || {},
                    |want_release_leds, rem| send_neutral(&borrow, want_release_leds, rem),
                );
            });
        }

        let mut hook = Some(after_first_neutral_hook);
        let mut sent_at_least_once = false;
        loop {
            if self.is_hung() {
                borrow.release_writing_if_held();
                self.emergency_in_flight.store(false, Ordering::SeqCst);
                return ResetOutcome::Unconfirmed;
            }
            if fatal && self.is_fatal_confirmed(my_req) {
                borrow.release_writing_if_held();
                self.emergency_in_flight.store(false, Ordering::SeqCst);
                return ResetOutcome::Confirmed;
            }
            // On fatal retries after an active worker write was still in flight
            // on pass 1, wait boundedly for `SLOT_WRITING` to clear before sending
            // the confirming neutral strictly after the active write ends.
            if fatal && !borrow.acquired_writing && sent_at_least_once {
                let rem = fatal_deadline.saturating_duration_since(Instant::now());
                if rem.is_zero() {
                    self.emergency_in_flight.store(false, Ordering::SeqCst);
                    return if self.is_fatal_confirmed(my_req) {
                        ResetOutcome::Confirmed
                    } else {
                        ResetOutcome::Unconfirmed
                    };
                }
                borrow.acquired_writing =
                    self.wait_and_acquire_writing(Duration::from_millis(1).min(rem), || {});
                if !borrow.acquired_writing {
                    continue;
                }
            }
            if fatal && Instant::now() >= fatal_deadline {
                borrow.release_writing_if_held();
                self.emergency_in_flight.store(false, Ordering::SeqCst);
                return if self.is_fatal_confirmed(my_req) {
                    ResetOutcome::Confirmed
                } else {
                    ResetOutcome::Unconfirmed
                };
            }

            let Some(mut neutral_guard) = self.try_acquire_primary_neutral() else {
                borrow.release_writing_if_held();
                if fatal {
                    self.emergency_in_flight.store(false, Ordering::SeqCst);
                    if self.is_fatal_confirmed(my_req) {
                        return ResetOutcome::Confirmed;
                    }
                    if Instant::now() >= fatal_deadline || self.is_hung() {
                        return ResetOutcome::Unconfirmed;
                    }
                    if (self.io_phase.load(Ordering::SeqCst)
                        & (PHASE_PRIMARY_NEUTRAL | PHASE_FALLBACK))
                        != 0
                    {
                        let rem = fatal_deadline.saturating_duration_since(Instant::now());
                        std::thread::sleep(Duration::from_millis(1).min(rem));
                    }
                    self.emergency_in_flight.store(true, Ordering::SeqCst);
                    continue;
                }
                // Dekker E1: clear `emergency_in_flight` FIRST, THEN recheck
                // monotonic tickets and `io_phase` (`SeqCst`).
                self.emergency_in_flight.store(false, Ordering::SeqCst);
                if self.has_pending_neutral()
                    && (self.io_phase.load(Ordering::SeqCst)
                        & (PHASE_PRIMARY_NEUTRAL | PHASE_FALLBACK))
                        == 0
                    && !self.is_hung()
                {
                    self.emergency_in_flight.store(true, Ordering::SeqCst);
                    continue;
                }
                return if self.neutral_completed_seq.load(Ordering::SeqCst) >= my_req {
                    ResetOutcome::Confirmed
                } else {
                    ResetOutcome::Unconfirmed
                };
            };

            if fatal && (attempts_remaining == 0 || Instant::now() >= fatal_deadline) {
                drop(neutral_guard);
                borrow.release_writing_if_held();
                self.emergency_in_flight.store(false, Ordering::SeqCst);
                return if self.is_fatal_confirmed(my_req) {
                    ResetOutcome::Confirmed
                } else {
                    ResetOutcome::Unconfirmed
                };
            }

            let seq_before_pass = self.neutral_requested_seq.load(Ordering::SeqCst);
            let led_before_pass = self.release_leds_requested_seq.load(Ordering::SeqCst);
            let send_deadline = fatal.then_some(fatal_deadline);
            let mut nonfatal_attempts = MAX_NEUTRAL_ATTEMPTS;
            let attempts_ref = if fatal {
                &mut attempts_remaining
            } else {
                &mut nonfatal_attempts
            };
            let outcome = self.send_neutral_with_cancel_retry_bounded(
                send_deadline,
                borrow.acquired_writing,
                attempts_ref,
                |want_release_leds, rem| send_neutral(&borrow, want_release_leds, rem),
            );
            if outcome == SendAttemptOutcome::Hung {
                neutral_guard.hung = true;
            }
            drop(neutral_guard);
            if outcome == SendAttemptOutcome::Hung {
                borrow.release_writing_if_held();
                self.emergency_in_flight.store(false, Ordering::SeqCst);
                return ResetOutcome::Unconfirmed;
            }
            sent_at_least_once = true;
            if let Some(f) = hook.take() {
                f();
            }
            borrow.release_writing_if_held();

            // Dekker E1: mark emergency neutral finished strictly AFTER
            // `NeutralPhaseGuard::drop` cleared `PHASE_PRIMARY_NEUTRAL` (`SeqCst`).
            self.emergency_in_flight.store(false, Ordering::SeqCst);
            if outcome == SendAttemptOutcome::Failed {
                return if fatal && self.is_fatal_confirmed(my_req) {
                    ResetOutcome::Confirmed
                } else {
                    ResetOutcome::Unconfirmed
                };
            }
            if outcome == SendAttemptOutcome::HandedOffToCancel {
                if fatal && self.is_fatal_confirmed(my_req) {
                    return ResetOutcome::Confirmed;
                }
                let post_cancel_ready = (self.io_phase.load(Ordering::SeqCst) & PHASE_CANCEL_MASK)
                    == 0
                    && self.has_pending_neutral()
                    && (!fatal || (attempts_remaining > 0 && Instant::now() < fatal_deadline));
                if post_cancel_ready {
                    self.emergency_in_flight.store(true, Ordering::SeqCst);
                    continue;
                }
                if fatal
                    && self.neutral_after_cancel.load(Ordering::SeqCst)
                    && Instant::now() < fatal_deadline
                {
                    let rem = fatal_deadline.saturating_duration_since(Instant::now());
                    std::thread::sleep(Duration::from_millis(1).min(rem));
                    self.emergency_in_flight.store(true, Ordering::SeqCst);
                    continue;
                }
                return ResetOutcome::Unconfirmed;
            }

            // Dekker E2: recheck monotonic tickets strictly AFTER clearing `emergency_in_flight`.
            let newer_ticket = self.neutral_requested_seq.load(Ordering::SeqCst) > seq_before_pass
                || self.release_leds_requested_seq.load(Ordering::SeqCst) > led_before_pass
                || self.release_leds_requested_seq.load(Ordering::SeqCst)
                    > self.release_leds_completed_seq.load(Ordering::SeqCst);
            if newer_ticket {
                if fatal && (attempts_remaining == 0 || Instant::now() >= fatal_deadline) {
                    return if self.is_fatal_confirmed(my_req) {
                        ResetOutcome::Confirmed
                    } else {
                        ResetOutcome::Unconfirmed
                    };
                }
                self.emergency_in_flight.store(true, Ordering::SeqCst);
                continue;
            }
            if fatal {
                if self.is_fatal_confirmed(my_req) {
                    return ResetOutcome::Confirmed;
                }
                if attempts_remaining == 0 || Instant::now() >= fatal_deadline {
                    return ResetOutcome::Unconfirmed;
                }
                // Pass 1 ran while `SLOT_WRITING` was still held; loop to wait
                // boundedly for `SLOT_WRITING` to clear.
                self.emergency_in_flight.store(true, Ordering::SeqCst);
                continue;
            }
            return if self.neutral_completed_seq.load(Ordering::SeqCst) >= my_req {
                ResetOutcome::Confirmed
            } else {
                ResetOutcome::Unconfirmed
            };
        }
    }

    #[cfg(any(windows, test))]
    fn wait_and_acquire_writing(&self, write_wait: Duration, on_in_flight: impl FnOnce()) -> bool {
        let mut notified = false;
        let deadline = Instant::now() + write_wait;
        let mut on_in_flight = Some(on_in_flight);
        loop {
            let cur = self.state.load(Ordering::Acquire);
            if cur & SLOT_WRITING == 0 {
                if self
                    .state
                    .compare_exchange_weak(
                        cur,
                        cur | SLOT_WRITING,
                        Ordering::AcqRel,
                        Ordering::Acquire,
                    )
                    .is_ok()
                {
                    return true;
                }
            } else {
                if !notified {
                    notified = true;
                    if let Some(f) = on_in_flight.take() {
                        f();
                    }
                }
                if Instant::now() >= deadline {
                    return false;
                }
                std::thread::yield_now();
            }
        }
    }

    #[cfg(any(windows, test))]
    fn release_writing(&self) {
        self.state.fetch_and(!SLOT_WRITING, Ordering::AcqRel);
    }

    /// Marks the slot closing and waits up to `budget` for any active emergency
    /// or cancel borrow (and any in-flight `io_phase` cancel or neutral write)
    /// to drain, then atomically transitions to `SLOT_CLOSED` before clearing
    /// handles. Returns `true` if `SLOT_HUNG` is set OR a borrow/phase did not
    /// settle within `budget` (caller MUST leak handles rather than closing
    /// underneath a wedged kernel IRP or live borrower).
    #[cfg(any(windows, test))]
    fn begin_close_and_wait_bounded(&self, budget: Duration) -> bool {
        self.state.fetch_or(SLOT_CLOSING, Ordering::AcqRel);
        let deadline = Instant::now() + budget;
        loop {
            let cur = self.state.load(Ordering::Acquire);
            if cur & SLOT_HUNG != 0 {
                return true;
            }
            if cur & (SLOT_EMERGENCY | SLOT_BORROW_MASK) == 0
                && self.io_phase.load(Ordering::SeqCst) == 0
            {
                if self
                    .state
                    .compare_exchange_weak(
                        cur,
                        cur | SLOT_CLOSED,
                        Ordering::AcqRel,
                        Ordering::Acquire,
                    )
                    .is_ok()
                {
                    break;
                }
                continue;
            }
            if Instant::now() >= deadline {
                self.mark_hung();
                return true;
            }
            std::thread::yield_now();
        }
        let hung = self.is_hung();
        if !hung {
            self.handle.store(0, Ordering::Release);
            self.emergency_event.store(0, Ordering::Release);
            self.fallback_event.store(0, Ordering::Release);
        }
        hung
    }

    #[cfg(any(windows, test))]
    fn try_claim_emergency(&self, emergency_wait: Duration) -> Option<EmergencyBorrow<'_>> {
        let claim_deadline = Instant::now() + emergency_wait;
        loop {
            let cur = self.state.load(Ordering::Acquire);
            if cur & (SLOT_HUNG | SLOT_CLOSING | SLOT_CLOSED) != 0 {
                return None;
            }
            if cur & SLOT_EMERGENCY != 0 {
                if Instant::now() >= claim_deadline {
                    return None;
                }
                std::thread::yield_now();
                continue;
            }
            let next = (cur | SLOT_EMERGENCY) + 1;
            if self
                .state
                .compare_exchange_weak(cur, next, Ordering::AcqRel, Ordering::Acquire)
                .is_ok()
            {
                break;
            }
        }
        self.reset_epoch.fetch_add(1, Ordering::SeqCst);

        Some(EmergencyBorrow {
            slot: self,
            handle: self.handle.load(Ordering::Acquire),
            emergency_event: self.emergency_event.load(Ordering::Acquire),
            transport: if self.transport.load(Ordering::Acquire) == 1 {
                Transport::Bluetooth
            } else {
                Transport::Usb
            },
            out_len: usize::from(self.out_len.load(Ordering::Acquire)),
            fw_ver: self.fw_ver.load(Ordering::Acquire),
            acquired_writing: false,
        })
    }

    /// Acquires a lightweight borrow of the live HID handle (without needing
    /// exclusive `SLOT_EMERGENCY`) whenever the handle has not closed (`!SLOT_CLOSED`,
    /// including while `SLOT_EMERGENCY` keeps a closing slot's handle alive),
    /// so a fatal reset can still call `CancelIoEx` + fallback neutral.
    #[cfg(any(windows, test))]
    fn try_borrow_handle_for_cancel(&self) -> Option<CancelBorrow<'_>> {
        loop {
            let cur = self.state.load(Ordering::Acquire);
            if cur & (SLOT_HUNG | SLOT_CLOSED) != 0 {
                return None;
            }
            // Allow borrowing on an open slot OR a closing slot whose handle is
            // still kept alive by an active `SLOT_EMERGENCY` reset.
            if (cur & SLOT_CLOSING) != 0 && (cur & SLOT_EMERGENCY) == 0 {
                return None;
            }
            if (cur & SLOT_BORROW_MASK) == SLOT_BORROW_MASK {
                return None;
            }
            if self
                .state
                .compare_exchange_weak(cur, cur + 1, Ordering::AcqRel, Ordering::Acquire)
                .is_ok()
            {
                let handle = self.handle.load(Ordering::Acquire);
                let fallback_event = self.fallback_event.load(Ordering::Acquire);
                return Some(CancelBorrow {
                    slot: self,
                    handle,
                    fallback_event,
                    transport: if self.transport.load(Ordering::Acquire) == 1 {
                        Transport::Bluetooth
                    } else {
                        Transport::Usb
                    },
                    out_len: usize::from(self.out_len.load(Ordering::Acquire)),
                    fw_ver: self.fw_ver.load(Ordering::Acquire),
                });
            }
        }
    }
}

#[cfg(any(windows, test))]
struct NeutralPhaseGuard<'a> {
    slot: &'a EmergencySlot,
    bit: u32,
    hung: bool,
}

#[cfg(any(windows, test))]
impl Drop for NeutralPhaseGuard<'_> {
    fn drop(&mut self) {
        if self.hung {
            self.slot.mark_hung();
        } else {
            self.slot.io_phase.fetch_and(!self.bit, Ordering::SeqCst);
        }
    }
}

#[cfg(any(windows, test))]
struct CancelBorrow<'a> {
    slot: &'a EmergencySlot,
    handle: usize,
    fallback_event: usize,
    transport: Transport,
    out_len: usize,
    fw_ver: u16,
}

#[cfg(any(windows, test))]
impl Drop for CancelBorrow<'_> {
    fn drop(&mut self) {
        self.slot.state.fetch_sub(1, Ordering::AcqRel);
    }
}

#[cfg(any(windows, test))]
struct EmergencyBorrow<'a> {
    slot: &'a EmergencySlot,
    handle: usize,
    emergency_event: usize,
    transport: Transport,
    out_len: usize,
    fw_ver: u16,
    acquired_writing: bool,
}

#[cfg(any(windows, test))]
impl EmergencyBorrow<'_> {
    #[cfg(test)]
    fn wait_and_acquire_writing(&mut self, write_wait: Duration, on_in_flight: impl FnOnce()) {
        self.acquired_writing = self.slot.wait_and_acquire_writing(write_wait, on_in_flight);
    }

    fn release_writing_if_held(&mut self) {
        if self.acquired_writing {
            self.acquired_writing = false;
            self.slot.release_writing();
        }
    }
}

#[cfg(any(windows, test))]
impl Drop for EmergencyBorrow<'_> {
    fn drop(&mut self) {
        let release = if self.acquired_writing {
            SLOT_WRITING + SLOT_EMERGENCY + 1
        } else {
            SLOT_EMERGENCY + 1
        };
        self.slot.state.fetch_sub(release, Ordering::AcqRel);
    }
}

/// Default bounded confirmation deadline for fatal shutdown emergency resets
/// (`crash.rs` / Ctrl+C / process exit) before giving up and reporting
/// `ResetOutcome::Unconfirmed`.
#[cfg(any(windows, test))]
const FATAL_RESET_BUDGET: Duration = Duration::from_millis(200);

/// Strictly caps total HID write attempts across any single neutral delivery or
/// fatal emergency reset (initial attempt + up to 2 retries = 3 max attempts).
#[cfg(any(windows, test))]
const MAX_NEUTRAL_ATTEMPTS: u32 = 3;

/// Lock-free bitmask of slots whose fatal neutral reset ended `Unconfirmed`
/// (readable by diagnostics/tests even when `stderr` is a redirected pipe).
#[cfg(any(windows, test))]
static FATAL_UNCONFIRMED_SLOTS: AtomicU32 = AtomicU32::new(0);

/// Win32 `FILE_TYPE_CHAR` (`0x0002`) — the ONLY `GetFileType` value safe for
/// synchronous `WriteFile` to `STD_ERROR_HANDLE` inside a fatal crash handler
/// (never write to `FILE_TYPE_PIPE`, which can block indefinitely on a full pipe).
#[cfg(any(windows, test))]
const FILE_TYPE_CHAR_RAW: u32 = 0x0002;

#[cfg(any(windows, test))]
fn should_write_fatal_stderr(file_type: u32) -> bool {
    file_type == FILE_TYPE_CHAR_RAW
}

#[cfg(any(windows, test))]
fn record_fatal_unconfirmed_slot(slot_idx: usize) {
    if slot_idx < 32 {
        FATAL_UNCONFIRMED_SLOTS.fetch_or(1u32 << slot_idx, Ordering::SeqCst);
    }
}

/// Splits an optional remaining fatal budget into `(write_timeout_ms, cancel_timeout_ms)`
/// strictly satisfying `write_timeout_ms + cancel_timeout_ms <= remaining_ms`,
/// `write_timeout_ms <= default_write_ms`, and `cancel_timeout_ms <= default_cancel_ms`.
/// Returns `None` when `remaining` has expired (`0 ms`), preventing overlapped I/O
/// from starting past `fatal_deadline`.
#[cfg(any(windows, test))]
fn split_overlapped_budget(
    remaining: Option<Duration>,
    default_write_ms: u32,
    default_cancel_ms: u32,
) -> Option<(u32, u32)> {
    let Some(rem) = remaining else {
        return Some((default_write_ms, default_cancel_ms));
    };
    let total_ms = rem.as_millis().min(u32::MAX as u128) as u32;
    if total_ms == 0 {
        return None;
    }
    if total_ms == 1 {
        return Some((1.min(default_write_ms), 0));
    }
    let cancel_ms = (total_ms / 2).min(default_cancel_ms);
    let write_ms = (total_ms - cancel_ms).min(default_write_ms);
    Some((write_ms, cancel_ms))
}

/// Result of attempting to transmit a neutral reset report to a DualSense HID
/// device. Preserves the distinction between confirmed delivery, settled
/// non-hung failure, and a wedged/hung kernel IRP (`SLOT_HUNG`).
#[cfg(any(windows, test))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum NeutralSendOutcome {
    Delivered,
    Failed,
    Hung,
}

/// Internal outcome of [`EmergencySlot::send_neutral_with_cancel_retry_bounded`]
/// distinguishing an active `CancelIoEx` handoff (`HandedOffToCancel`) from an
/// exhausted settled failure (`Failed`) so callers never drop a post-cancel
/// ticket published by `CancelIoGuard::drop` while the sender still holds `io_phase`.
#[cfg(any(windows, test))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SendAttemptOutcome {
    Delivered,
    Failed,
    Hung,
    HandedOffToCancel,
}

#[cfg(any(windows, test))]
impl SendAttemptOutcome {
    fn into_public(self) -> NeutralSendOutcome {
        match self {
            Self::Delivered => NeutralSendOutcome::Delivered,
            Self::Failed | Self::HandedOffToCancel => NeutralSendOutcome::Failed,
            Self::Hung => NeutralSendOutcome::Hung,
        }
    }
}

/// Result of [`EmergencySlot::run_emergency_reset`].
#[cfg(any(windows, test))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ResetOutcome {
    /// No device handle was open on this slot (`SLOT_CLOSED` or null handle).
    SlotIdle,
    /// Neutral report delivery was confirmed (on fatal resets: strictly with
    /// `SLOT_WRITING == 0`, `io_phase == 0`, and no pending trailing/post-cancel handoff).
    Confirmed,
    /// Delivery could not be confirmed within the bounded budget or the slot hung.
    Unconfirmed,
}

#[cfg(any(windows, test))]
struct SerializedWriteResult {
    write_ok: bool,
    preempted: bool,
    trailing_neutral: Option<NeutralSendOutcome>,
}

/// Outcome of a strictly bounded overlapped emergency write attempt.
#[cfg(any(windows, test))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum EmergencyWriteOutcome {
    Completed,
    /// `WriteFile` failed immediately (not `ERROR_IO_PENDING`).
    ImmediateError,
    /// Timed out, cancelled cleanly via `CancelIoEx` + bounded wait.
    TimedOutSettled,
    /// Kernel driver ignored `CancelIoEx` past the bounded wait.
    Hung,
}

/// Classifies a strictly bounded overlapped emergency write outcome into
/// [`NeutralSendOutcome`] without EVER falling back to unbounded synchronous HID I/O.
#[cfg(any(windows, test))]
fn classify_emergency_write(outcome: EmergencyWriteOutcome) -> NeutralSendOutcome {
    match outcome {
        EmergencyWriteOutcome::Completed => NeutralSendOutcome::Delivered,
        EmergencyWriteOutcome::ImmediateError | EmergencyWriteOutcome::TimedOutSettled => {
            NeutralSendOutcome::Failed
        }
        EmergencyWriteOutcome::Hung => NeutralSendOutcome::Hung,
    }
}

#[cfg(windows)]
static EMERGENCY_SLOTS: [EmergencySlot; MAX_DUALSENSE] =
    [const { EmergencySlot::new() }; MAX_DUALSENSE];

/// Snapshot of host XInput HID interfaces enumerated during DualSense scanning.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct XInputTopology {
    pub virtual_present: bool,
    pub physical_present: bool,
}

impl XInputTopology {
    pub const fn only_virtual(self) -> bool {
        self.virtual_present && !self.physical_present
    }
}

/// Classifies a lowercased Windows HID interface path as `(is_virtual_xinput, is_physical_xinput)`.
#[cfg(any(windows, test))]
fn classify_xinput_hid_path(lower: &str) -> (bool, bool) {
    let is_virtual = (lower.contains("ig_") && lower.contains("vid_28de"))
        || lower.contains("vigem")
        || (lower.contains("ig_") && lower.contains("root#"));
    let is_physical = !is_virtual && lower.contains("ig_");
    (is_virtual, is_physical)
}

/// Returns the latest enumerated host XInput HID topology (whether virtual
/// wrappers like Steam Input / ViGEm and/or physical XInput controllers exist).
pub fn xinput_topology() -> XInputTopology {
    let bits = XINPUT_TOPOLOGY.load(Ordering::Acquire);
    XInputTopology {
        virtual_present: bits & XINPUT_VIRTUAL_BIT != 0,
        physical_present: bits & XINPUT_PHYSICAL_BIT != 0,
    }
}

/// Validates a HID collection's `OutputReportByteLength` for `transport`.
///
/// Rejects `0`, lengths below the DualSense protocol minimum (`48` USB / `78` BT),
/// and lengths exceeding `MAX_EMERGENCY_REPORT_LEN` (`128`), guaranteeing that
/// normal and emergency reports always transmit the exact same byte length with
/// zero truncation.
#[cfg(any(windows, test))]
fn validate_output_report_len(raw_out_len: u16, transport: Transport) -> Option<usize> {
    let len = usize::from(raw_out_len);
    let min_len = match transport {
        Transport::Usb => USB_OUTPUT_REPORT_LEN,
        Transport::Bluetooth => BT_REPORT_LEN,
    };
    if len == 0 || len < min_len || len > MAX_EMERGENCY_REPORT_LEN {
        return None;
    }
    Some(len)
}

/// Encodes an `OutputCommand` into `buf` sized to exact `out_len` (`<= MAX_EMERGENCY_REPORT_LEN`).
#[cfg(windows)]
fn encode_output_packet(
    transport: Transport,
    out_len: usize,
    bt_seq: u8,
    cmd: &OutputCommand,
    buf: &mut [u8; MAX_EMERGENCY_REPORT_LEN],
) -> usize {
    let len = out_len.min(MAX_EMERGENCY_REPORT_LEN);
    buf[..len].fill(0);
    match transport {
        Transport::Bluetooth => {
            let bt = cmd.encode_bt(bt_seq);
            let copy_len = BT_REPORT_LEN.min(len);
            buf[..copy_len].copy_from_slice(&bt[..copy_len]);
        }
        Transport::Usb => {
            buf[0] = USB_OUTPUT_REPORT_ID;
            buf[1..1 + OUTPUT_COMMON_LEN].copy_from_slice(&cmd.encode_common());
        }
    }
    len
}

/// Computes the exact buffer length for `HidD_GetFeature` given the device
/// descriptor's `HIDP_CAPS.FeatureReportByteLength` and the minimum expected
/// `report_len`.
#[cfg(any(windows, test))]
fn feature_buffer_len(feat_len: usize, report_len: usize) -> Option<usize> {
    if feat_len == 0 {
        Some(report_len)
    } else if feat_len < report_len {
        None
    } else {
        Some(feat_len)
    }
}

/// Decides whether outgoing HID reports should enable compatible HID rumble
/// (`FLAG0_HAPTICS_SELECT`), which hardware-mutes USB 4-channel voice-coil
/// audio haptics on the DualSense.
pub fn should_enable_hid_rumble(
    transport: Transport,
    mode: HapticsMode,
    audio_haptics_active: bool,
    multi_usb: bool,
) -> bool {
    match (transport, mode) {
        (_, HapticsMode::Off) => false,
        (_, HapticsMode::Hid) => true,
        (Transport::Usb, HapticsMode::Audio4Ch) => multi_usb,
        (Transport::Bluetooth, HapticsMode::Audio4Ch) => true,
        (Transport::Usb, HapticsMode::Hybrid) => multi_usb || !audio_haptics_active,
        (Transport::Bluetooth, HapticsMode::Hybrid) => true,
    }
}

/// Best-effort, lock-free, allocation-free hardware reset for crash, panic,
/// and Ctrl+C paths.
///
/// When `fatal` is `true`, permanently halts normal DualSense output writes
/// and 4-channel WASAPI audio haptics (lock-free), cancels pending HID I/O
/// unconditionally on every open slot, and sends strictly bounded overlapped
/// neutral reset packets (zero rumble, active `0x05` trigger release, LED
/// release; 50 ms write + 50 ms cancel wait, marking `SLOT_HUNG` and leaking
/// static I/O/handles on a wedged kernel driver) using a strict Dekker/`SeqCst`
/// handoff with worker writes. Never calls unbounded synchronous HID I/O.
pub fn emergency_reset_all(fatal: bool) {
    let is_fatal = fatal || SHUTDOWN_REQUESTED.load(Ordering::Acquire);
    if is_fatal {
        SHUTDOWN_REQUESTED.store(true, Ordering::SeqCst);
        audio_haptics::shutdown_global_lockfree();
    }
    #[cfg(windows)]
    imp::emergency_reset_all(is_fatal);
}

/// Clean shutdown path for normal title/process exit and `end_title`:
/// 1. Halts spawning and synchronizes global audio-haptics shutdown within a
///    bounded budget so zero non-zero WASAPI `ReleaseBuffer` can start or
///    remain in flight.
/// 2. Signals all HID and WASAPI workers to stop, cancelling pending HID I/O
///    so workers wake promptly, and waits up to `SHUTDOWN_JOIN_TIMEOUT` (250 ms)
///    for `live_workers == 0` plus bounded OS-thread join (never unbounded `.join()`).
/// 3. Runs `emergency_reset_all(true)` after zero non-hung workers remain alive.
pub fn shutdown_and_join() {
    static CANCELLED: AtomicBool = AtomicBool::new(false);
    let start = Instant::now();
    SHUTDOWN_REQUESTED.store(true, Ordering::SeqCst);
    let _ = audio_haptics::shutdown_global_bounded(Duration::from_millis(50));
    let remaining = SHUTDOWN_JOIN_TIMEOUT.saturating_sub(start.elapsed());
    let _ = WORKERS.shutdown_and_wait(remaining, || {
        #[cfg(windows)]
        if !CANCELLED.swap(true, Ordering::AcqRel) {
            imp::cancel_all_open_handles();
        }
    });
    emergency_reset_all(true);
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LivenessEvent {
    Unchanged,
    Connected,
    Disconnected,
}

/// Tracks whether a USB/Dongle/BT DualSense handle actually has a live
/// controller behind it (`sensor_timestamp` advancing within 1.5 s, or valid
/// basic Bluetooth `0x01` reports arriving within 1.5 s).
#[derive(Default)]
pub struct LivenessMonitor {
    connected: bool,
    last_sensor_ts: Option<u32>,
    last_advance_at: Option<Instant>,
}

impl LivenessMonitor {
    pub fn is_connected(&self) -> bool {
        self.connected
    }

    pub fn on_report(&mut self, raw: &RawReport, now: Instant) -> LivenessEvent {
        if !raw.has_live_telemetry() {
            return self.check_timeout(now);
        }
        if raw.basic_bt_report {
            // Basic Bluetooth `0x01` reports carry no hardware sensor clock, so
            // every valid arriving packet refreshes liveness.
            self.last_sensor_ts = None;
            self.last_advance_at = Some(now);
            if !self.connected {
                self.connected = true;
                return LivenessEvent::Connected;
            }
            return LivenessEvent::Unchanged;
        }
        let advanced = match self.last_sensor_ts {
            None => true,
            Some(prev) => prev != raw.sensor_timestamp,
        };
        if advanced {
            self.last_sensor_ts = Some(raw.sensor_timestamp);
            self.last_advance_at = Some(now);
            if !self.connected {
                self.connected = true;
                return LivenessEvent::Connected;
            }
            LivenessEvent::Unchanged
        } else {
            self.check_timeout(now)
        }
    }

    pub fn check_timeout(&mut self, now: Instant) -> LivenessEvent {
        if self.connected
            && self
                .last_advance_at
                .is_some_and(|at| now.saturating_duration_since(at) >= DONGLE_FREEZE_TIMEOUT)
        {
            self.connected = false;
            // Preserve `last_sensor_ts` so a frozen `DS5Dongle` repeating the
            // exact same stale timestamp stays disconnected until the sensor
            // clock genuinely advances again.
            self.last_advance_at = None;
            return LivenessEvent::Disconnected;
        }
        LivenessEvent::Unchanged
    }

    pub fn force_disconnect(&mut self) {
        self.connected = false;
        self.last_sensor_ts = None;
        self.last_advance_at = None;
    }
}

/// Parses an incoming raw HID frame and advances `liveness`, ensuring a stream
/// of corrupt/unparseable frames still times out after `DONGLE_FREEZE_TIMEOUT`.
fn on_raw_packet(
    packet: &[u8],
    transport: Transport,
    liveness: &mut LivenessMonitor,
    now: Instant,
) -> (Option<RawReport>, LivenessEvent) {
    let Some(raw) = parse_input_report(packet, transport) else {
        return (None, liveness.check_timeout(now));
    };
    let event = liveness.on_report(&raw, now);
    (Some(raw), event)
}

/// Stateful arbiter for PS short-press Guide events and the PS-first
/// `PS + Touchpad` gyro-recenter chord (with 60 ms simultaneous-press grace).
#[derive(Default)]
pub struct SystemButtonFilter {
    ps_down_since: Option<Instant>,
    ps_chorded: bool,
    recenter_latched: bool,
    guide_pulse_until: Option<Instant>,
    tp_chord_suppressed: bool,
    tp_down_since: Option<Instant>,
    pending_tp_button: u32,
    tp_pulse: Option<(u32, Instant)>,
}

pub struct FilterOutcome {
    pub recenter_gyro: bool,
    pub system_buttons: u32,
    pub touchpad_buttons: u32,
}

impl SystemButtonFilter {
    pub fn reset(&mut self) {
        *self = Self::default();
    }

    pub fn update(
        &mut self,
        raw: &RawReport,
        other_buttons: u32,
        cfg: &DualSenseConfig,
        now: Instant,
    ) -> FilterOutcome {
        let ps = raw.ps_pressed();
        let tp = raw.touchpad_clicked();
        let mut recenter_gyro = false;

        if !ps && !tp {
            self.tp_chord_suppressed = false;
        }

        if ps && tp {
            if !self.recenter_latched {
                self.recenter_latched = true;
                recenter_gyro = true;
            }
            self.ps_chorded = true;
            self.tp_chord_suppressed = true;
            self.tp_down_since = None;
            self.pending_tp_button = 0;
            self.tp_pulse = None;
        }

        if ps {
            let since = *self.ps_down_since.get_or_insert(now);
            if other_buttons != 0
                || raw.any_face_button_pressed()
                || now.saturating_duration_since(since) >= PS_SHORT_PRESS_MAX
            {
                self.ps_chorded = true;
            }
        } else {
            if let Some(since) = self.ps_down_since.take()
                && !self.ps_chorded
                && now.saturating_duration_since(since) < PS_SHORT_PRESS_MAX
            {
                // Hold the virtual Guide bit for 80 ms so the poll loop delivers press + release.
                self.guide_pulse_until = Some(now + BUTTON_PULSE_DURATION);
            }
            self.ps_chorded = false;
            self.recenter_latched = false;
        }

        let mut touchpad_buttons = 0u32;
        if tp {
            if !self.tp_chord_suppressed && !ps {
                if let Some(since) = self.tp_down_since {
                    if now.saturating_duration_since(since) >= TOUCHPAD_CHORD_GRACE {
                        touchpad_buttons = self.pending_tp_button;
                    }
                } else {
                    self.tp_down_since = Some(now);
                    self.pending_tp_button = touchpad_button(raw, cfg);
                    self.tp_pulse = None;
                }
            }
        } else if let Some(since) = self.tp_down_since.take() {
            let btn = std::mem::take(&mut self.pending_tp_button);
            if !self.tp_chord_suppressed
                && btn != 0
                && now.saturating_duration_since(since) < TOUCHPAD_CHORD_GRACE
            {
                self.tp_pulse = Some((btn, now + BUTTON_PULSE_DURATION));
            }
        }

        if touchpad_buttons == 0
            && let Some((btn, until)) = self.tp_pulse
        {
            if now < until {
                touchpad_buttons = btn;
            } else {
                self.tp_pulse = None;
            }
        }

        let system_buttons = if self.guide_pulse_until.is_some_and(|until| now < until) {
            SYSTEM_BUTTON_GUIDE
        } else {
            self.guide_pulse_until = None;
            0
        };

        FilterOutcome {
            recenter_gyro,
            system_buttons,
            touchpad_buttons,
        }
    }
}

#[derive(Clone, Copy, Default)]
pub struct SlotSnapshot {
    pub connected: bool,
    pub transport: Option<Transport>,
    pub pad: PadState,
    pub battery: BatteryState,
    pub system_buttons: u32,
}

pub(crate) struct SharedSlot {
    pub path: String,
    pub transport: Transport,
    pub container_id: Option<ContainerId>,
    pub stop: AtomicBool,
    pub state: Mutex<WorkerState>,
}

#[cfg(any(windows, test))]
fn clear_slot_state_on_disconnect(shared: &SharedSlot) -> u16 {
    let mut st = shared.state.lock().unwrap_or_else(|e| e.into_inner());
    st.snapshot = SlotSnapshot::default();
    st.raw_latest = None;
    st.rumble = [0.0; 4];
    st.audio_haptics_active = false;
    st.gyro.reset_stream();
    st.firmware_version
}

pub(crate) struct WorkerState {
    pub snapshot: SlotSnapshot,
    pub raw_latest: Option<RawReport>,
    pub gyro: GyroTracker,
    pub firmware_version: u16,
    pub rumble: [f32; 4],
    pub focused: bool,
    pub audio_haptics_active: bool,
    pub multi_usb: bool,
    pub leds_dirty: bool,
    pub leds_epoch: u64,
    pub host_slot: usize,
    pub config: DualSenseConfig,
}

impl WorkerState {
    fn new(host_slot: usize, config: DualSenseConfig) -> Self {
        Self {
            snapshot: SlotSnapshot::default(),
            raw_latest: None,
            gyro: GyroTracker::default(),
            firmware_version: FIRMWARE_VIBRATION2_MIN,
            rumble: [0.0; 4],
            focused: true,
            audio_haptics_active: false,
            multi_usb: false,
            leds_dirty: true,
            leds_epoch: 1,
            host_slot,
            config,
        }
    }

    fn mark_leds_dirty(&mut self) {
        self.leds_dirty = true;
        self.leds_epoch = self.leds_epoch.wrapping_add(1);
    }
}

pub struct DualSenseManager {
    slots: [Option<Arc<SharedSlot>>; MAX_DUALSENSE],
    config: DualSenseConfig,
    audio_haptics: AudioHapticsEngine,
    multi_usb_logged: bool,
    #[cfg(windows)]
    last_scan: Option<Instant>,
}

impl DualSenseManager {
    pub fn new(config: DualSenseConfig) -> Self {
        Self {
            slots: [const { None }; MAX_DUALSENSE],
            config,
            audio_haptics: AudioHapticsEngine::default(),
            multi_usb_logged: false,
            #[cfg(windows)]
            last_scan: None,
        }
    }

    pub fn set_config(&mut self, config: DualSenseConfig) {
        self.config = config.clone();
        for slot in self.slots.iter().flatten() {
            if let Ok(mut st) = slot.state.lock() {
                st.config = config.clone();
                st.mark_leds_dirty();
            }
        }
    }

    /// Scans for hot-plugged controllers (throttled internally) and reads the
    /// latest state of all active DualSense slots.
    pub fn poll(&mut self, focused: bool) -> [Option<SlotSnapshot>; MAX_DUALSENSE] {
        if !self.config.enabled || SHUTDOWN_REQUESTED.load(Ordering::Acquire) {
            self.audio_haptics.stop();
            return [None; MAX_DUALSENSE];
        }

        #[cfg(windows)]
        self.scan_if_due();

        let mut out = [None; MAX_DUALSENSE];
        let mut usb_open_count = 0usize;
        let mut usb_connected_count = 0usize;
        let mut primary_usb_slot: Option<usize> = None;
        let mut primary_container_id: Option<ContainerId> = None;
        let mut primary_rumble = [0.0f32; 4];

        for (idx, entry) in self.slots.iter_mut().enumerate() {
            let remove = entry
                .as_ref()
                .is_some_and(|s| s.stop.load(Ordering::Acquire));
            if remove {
                *entry = None;
                continue;
            }
            let Some(slot) = entry.as_ref() else {
                continue;
            };
            // Count ALL open USB DualSense HID devices (including an idle/unpaired
            // DS5Dongle) so 4ch WASAPI never routes ambiguously across >1 USB device.
            if slot.transport == Transport::Usb {
                usb_open_count += 1;
            }
            let mut st = slot.state.lock().unwrap_or_else(|e| e.into_inner());
            if st.focused != focused {
                st.focused = focused;
                if focused {
                    st.mark_leds_dirty();
                }
            }
            if !st.snapshot.connected {
                st.audio_haptics_active = false;
                continue;
            }
            if slot.transport == Transport::Usb {
                usb_connected_count += 1;
                if primary_usb_slot.is_none() {
                    primary_usb_slot = Some(idx);
                    primary_container_id = slot.container_id;
                    primary_rumble = st.rumble;
                }
            }

            let mut snap = st.snapshot;
            if let Some(raw) = st.raw_latest {
                let cfg = st.config.clone();
                st.gyro.apply_to_right_stick(
                    &raw,
                    &cfg,
                    &mut snap.pad.right_x,
                    &mut snap.pad.right_y,
                );
            }
            if !focused {
                snap.pad = PadState::default();
                snap.system_buttons = 0;
            }
            out[idx] = Some(snap);
        }

        let single_usb_ready = usb_open_count == 1 && usb_connected_count == 1;
        let multi_usb = usb_open_count > 1;
        let audio_mode_selected = matches!(
            self.config.haptics_mode,
            HapticsMode::Audio4Ch | HapticsMode::Hybrid
        );
        if multi_usb && audio_mode_selected {
            if !self.multi_usb_logged {
                tracing::info!(
                    open_usb_devices = usb_open_count,
                    "multiple open USB DualSense HID devices detected; 4ch WASAPI audio haptics requires a single unambiguous USB DualSense, falling back to HID rumble"
                );
                self.multi_usb_logged = true;
            }
        } else if !multi_usb {
            self.multi_usb_logged = false;
        }

        let target_container = if single_usb_ready {
            primary_container_id
        } else {
            None
        };
        self.audio_haptics
            .control()
            .set_target_container(target_container);

        let want_audio_haptics =
            single_usb_ready && target_container.is_some() && focused && audio_mode_selected;
        if want_audio_haptics {
            self.audio_haptics
                .control()
                .set_options(self.config.speaker_mirror, false);
            self.audio_haptics.control().set_levels(
                primary_rumble[0],
                primary_rumble[1],
                primary_rumble[2],
                primary_rumble[3],
            );
            self.audio_haptics.ensure_started();
        } else {
            // Immediately mute both voice-coil loopback synthesis and speaker_mirror
            // when unfocused, multi-USB, missing ContainerId, or disabled.
            self.audio_haptics.control().set_options(false, true);
            self.audio_haptics.control().clear_levels();
            if !single_usb_ready || target_container.is_none() || !audio_mode_selected {
                self.audio_haptics.stop();
            }
        }

        let primary_audio_active = want_audio_haptics && self.audio_haptics.is_active();
        for (idx, entry) in self.slots.iter().enumerate() {
            if let Some(slot) = entry {
                let mut st = slot.state.lock().unwrap_or_else(|e| e.into_inner());
                st.audio_haptics_active =
                    primary_audio_active && single_usb_ready && primary_usb_slot == Some(idx);
                st.multi_usb = multi_usb;
            }
        }

        out
    }

    pub fn set_rumble(&mut self, index: usize, motors: [f32; 4]) {
        if index >= MAX_DUALSENSE {
            return;
        }
        if let Some(slot) = &self.slots[index] {
            let mut st = slot.state.lock().unwrap_or_else(|e| e.into_inner());
            // Guard rumble updates with `snapshot.connected` under the same mutex
            // so a rumble call racing with disconnect cannot persist into reconnect.
            if st.snapshot.connected {
                st.rumble = motors;
            } else {
                st.rumble = [0.0; 4];
            }
        }
    }

    pub fn active_count(&self) -> usize {
        self.slots
            .iter()
            .flatten()
            .filter(|s| {
                s.state
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .snapshot
                    .connected
            })
            .count()
    }

    #[cfg(windows)]
    fn scan_if_due(&mut self) {
        const RESCAN: Duration = Duration::from_millis(1000);
        if self.last_scan.is_some_and(|t| t.elapsed() < RESCAN) {
            return;
        }
        self.last_scan = Some(Instant::now());
        imp::scan_and_spawn(&mut self.slots, &self.config);
    }
}

#[cfg(windows)]
mod imp {
    use std::ptr::{null, null_mut};
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::time::{Duration, Instant};

    use victory_core::config::DualSenseConfig;
    use windows_sys::Win32::Devices::DeviceAndDriverInstallation::{
        CM_GET_DEVICE_INTERFACE_LIST_PRESENT, CM_Get_DevNode_PropertyW,
        CM_Get_Device_Interface_List_SizeW, CM_Get_Device_Interface_ListW,
        CM_Get_Device_Interface_PropertyW, CM_LOCATE_DEVNODE_NORMAL, CM_Locate_DevNodeW,
        CR_SUCCESS,
    };
    use windows_sys::Win32::Devices::HumanInterfaceDevice::{
        HIDD_ATTRIBUTES, HIDP_CAPS, HIDP_STATUS_SUCCESS, HidD_FreePreparsedData,
        HidD_GetAttributes, HidD_GetFeature, HidD_GetHidGuid, HidD_GetPreparsedData, HidP_GetCaps,
        PHIDP_PREPARSED_DATA,
    };
    use windows_sys::Win32::Devices::Properties::{
        DEVPKEY_Device_ContainerId, DEVPKEY_Device_InstanceId, DEVPROP_TYPE_GUID,
        DEVPROP_TYPE_STRING,
    };
    use windows_sys::Win32::Foundation::{
        CloseHandle, ERROR_IO_INCOMPLETE, ERROR_IO_PENDING, ERROR_OPERATION_ABORTED, GENERIC_READ,
        GENERIC_WRITE, GetLastError, HANDLE, INVALID_HANDLE_VALUE, WAIT_OBJECT_0,
    };
    use windows_sys::Win32::Storage::FileSystem::{
        CreateFileW, FILE_FLAG_OVERLAPPED, FILE_SHARE_READ, FILE_SHARE_WRITE, FILE_TYPE_CHAR,
        GetFileType, OPEN_EXISTING, ReadFile, WriteFile,
    };
    use windows_sys::Win32::System::Console::{GetStdHandle, STD_ERROR_HANDLE};
    use windows_sys::Win32::System::IO::{CancelIoEx, GetOverlappedResult, OVERLAPPED};
    use windows_sys::Win32::System::Threading::{CreateEventW, ResetEvent, WaitForSingleObject};

    use super::{
        ContainerId, ControllerModel, EMERGENCY_SLOTS, EmergencySlot, EmergencyWriteOutcome,
        ImuCalibration, LivenessEvent, LivenessMonitor, MAX_DUALSENSE, MAX_EMERGENCY_REPORT_LEN,
        NeutralSendOutcome, OutputCommand, ReportLimiter, ResetOutcome, SHUTDOWN_REQUESTED,
        SharedSlot, SlotSnapshot, SystemButtonFilter, Transport, WorkerRegistration, WorkerState,
        XINPUT_PHYSICAL_BIT, XINPUT_TOPOLOGY, XINPUT_VIRTUAL_BIT, classify_emergency_write,
        classify_xinput_hid_path, clear_slot_state_on_disconnect, encode_output_packet,
        encode_trigger_effect, feature_buffer_len, on_raw_packet,
        protocol::{
            BT_REPORT_LEN, FEATURE_CALIBRATION_ID, FEATURE_CALIBRATION_LEN, FEATURE_FIRMWARE_ID,
            FEATURE_FIRMWARE_LEN, SONY_VID, USAGE_GAMEPAD, USAGE_PAGE_GENERIC_DESKTOP,
            parse_firmware_version, to_pad_state_without_touchpad,
        },
        should_enable_hid_rumble, validate_output_report_len, worker_coordinator,
    };

    const NORMAL_WRITE_TIMEOUT_MS: u32 = 25;
    const NORMAL_CANCEL_TIMEOUT_MS: u32 = 100;
    const EMERGENCY_WRITE_TIMEOUT_MS: u32 = 50;
    const EMERGENCY_CANCEL_TIMEOUT_MS: u32 = 50;
    const EMERGENCY_CLAIM_WAIT: Duration = Duration::from_millis(60);
    const EMERGENCY_WRITE_WAIT: Duration = Duration::from_millis(50);
    const CLOSE_BORROW_WAIT: Duration = Duration::from_millis(250);
    const DISCONNECT_RESET_RETRY: Duration = Duration::from_millis(50);

    /// RAII wrapper for an owned Win32 kernel handle (`HANDLE`) so failed
    /// thread spawns or early returns never leak kernel objects.
    struct OwnedHandle(HANDLE);

    // SAFETY: Win32 kernel handles are process-wide and safe to move across threads.
    unsafe impl Send for OwnedHandle {}

    impl OwnedHandle {
        fn new(h: HANDLE) -> Option<Self> {
            if h.is_null() || h == INVALID_HANDLE_VALUE {
                None
            } else {
                Some(Self(h))
            }
        }

        fn raw(&self) -> HANDLE {
            self.0
        }

        /// Intentionally leaks the handle when a hung kernel IRP or unreturned
        /// emergency borrow still holds it.
        fn leak(self) {
            std::mem::forget(self);
        }
    }

    impl Drop for OwnedHandle {
        fn drop(&mut self) {
            if !self.0.is_null() && self.0 != INVALID_HANDLE_VALUE {
                // SAFETY: valid owned kernel handle no longer referenced by any IRP.
                unsafe {
                    let _ = CloseHandle(self.0);
                }
            }
        }
    }

    /// Heap-allocated overlapped I/O context (`OVERLAPPED` + data buffer) so
    /// that if a buggy HID driver ignores `CancelIoEx` past our bounded wait,
    /// we can intentionally `Box::leak` it rather than freeing memory under a
    /// live kernel IRP.
    struct PendingIo {
        ov: OVERLAPPED,
        buf: Vec<u8>,
    }

    // SAFETY: `PendingIo` is only accessed by the single worker thread owning it.
    unsafe impl Send for PendingIo {}

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    enum WriteResult {
        Completed,
        Failed,
        Hung,
    }

    struct Discovered {
        path: String,
        path_wide: Vec<u16>,
        handle: OwnedHandle,
        container_id: Option<ContainerId>,
        model: ControllerModel,
        transport: Transport,
        in_len: usize,
        out_len: usize,
        feat_len: usize,
    }

    struct DisconnectReset {
        firmware_version: u16,
        last_attempt: Instant,
    }

    /// RAII guard binding an opened HID device, its Win32 events, its reusable
    /// overlapped read context, and its `EmergencySlot` registration.
    ///
    /// `Drop` ALWAYS runs on BOTH normal exit AND panic unwind:
    /// 1. Cancels any pending overlapped read with a bounded 100 ms wait
    ///    (leaking the heap `PendingIo` and marking `SLOT_HUNG` if unsettled).
    /// 2. Calls `EmergencySlot::begin_close_and_wait_bounded(250 ms)`.
    /// 3. Closes handles ONLY when zero pending I/O and zero emergency borrows
    ///    remain; otherwise intentionally leaks handles rather than closing
    ///    underneath a kernel IRP or active borrow.
    /// 4. Clears shared snapshot/rumble state and marks `shared.stop = true`.
    struct RegisteredHidHandle {
        slot_idx: usize,
        dev: Option<Discovered>,
        read_event: Option<OwnedHandle>,
        write_event: Option<OwnedHandle>,
        emergency_event: Option<OwnedHandle>,
        fallback_event: Option<OwnedHandle>,
        read_io: Option<Box<PendingIo>>,
        read_pending: bool,
        shared: Arc<SharedSlot>,
    }

    impl RegisteredHidHandle {
        fn register(
            slot_idx: usize,
            dev: Discovered,
            read_event: OwnedHandle,
            write_event: OwnedHandle,
            emergency_event: OwnedHandle,
            fallback_event: OwnedHandle,
            shared: Arc<SharedSlot>,
        ) -> Option<Self> {
            let slot = &EMERGENCY_SLOTS[slot_idx];
            slot.set_fallback_event(fallback_event.raw() as usize);
            if !slot.publish(
                dev.handle.raw() as usize,
                emergency_event.raw() as usize,
                dev.transport,
                dev.out_len,
            ) {
                return None;
            }
            // SAFETY: zero-initialized `OVERLAPPED` is valid.
            let mut ov: OVERLAPPED = unsafe { std::mem::zeroed() };
            ov.hEvent = read_event.raw();
            let read_io = Box::new(PendingIo {
                ov,
                buf: vec![0u8; dev.in_len.max(BT_REPORT_LEN)],
            });
            Some(Self {
                slot_idx,
                dev: Some(dev),
                read_event: Some(read_event),
                write_event: Some(write_event),
                emergency_event: Some(emergency_event),
                fallback_event: Some(fallback_event),
                read_io: Some(read_io),
                read_pending: false,
                shared,
            })
        }

        fn slot(&self) -> &'static EmergencySlot {
            &EMERGENCY_SLOTS[self.slot_idx]
        }

        fn dev(&self) -> &Discovered {
            self.dev.as_ref().expect("RegisteredHidHandle still active")
        }

        fn read_event_raw(&self) -> HANDLE {
            self.read_event
                .as_ref()
                .expect("RegisteredHidHandle still active")
                .raw()
        }

        fn write_event_raw(&self) -> HANDLE {
            self.write_event
                .as_ref()
                .expect("RegisteredHidHandle still active")
                .raw()
        }

        /// Cancels any pending overlapped read with a bounded 100 ms wait.
        /// If the kernel driver fails to settle within 100 ms, intentionally
        /// leaks `read_io` and marks the slot `SLOT_HUNG`.
        fn cancel_pending_read_bounded(&mut self) {
            if !self.read_pending {
                return;
            }
            self.read_pending = false;
            let (Some(dev), Some(read_event), Some(mut io)) = (
                self.dev.as_ref(),
                self.read_event.as_ref(),
                self.read_io.take(),
            ) else {
                return;
            };
            let handle = dev.handle.raw();
            let event = read_event.raw();
            // SAFETY: `handle`, `event`, and heap-allocated `io` remain valid
            // across cancellation and bounded wait.
            let settled = unsafe {
                let _ = CancelIoEx(handle, &io.ov);
                let mut dummy = 0u32;
                if WaitForSingleObject(event, NORMAL_CANCEL_TIMEOUT_MS) == WAIT_OBJECT_0
                    || GetOverlappedResult(handle, &io.ov, &mut dummy, 0) != 0
                {
                    let _ = GetOverlappedResult(handle, &io.ov, &mut dummy, 0);
                    true
                } else {
                    false
                }
            };
            if settled {
                // Re-attach reusable read context (or let it drop in `Drop`).
                io.ov.hEvent = event;
                self.read_io = Some(io);
            } else {
                // Kernel IRP is still pending: intentionally leak `OVERLAPPED` +
                // buffer and mark slot hung so handles are also leaked on close.
                let _ = Box::leak(io);
                self.slot().mark_hung();
            }
        }
    }

    impl Drop for RegisteredHidHandle {
        fn drop(&mut self) {
            self.cancel_pending_read_bounded();
            let must_leak = self.slot().begin_close_and_wait_bounded(CLOSE_BORROW_WAIT);
            if must_leak {
                if let Some(dev) = self.dev.take() {
                    dev.handle.leak();
                }
                if let Some(ev) = self.emergency_event.take() {
                    ev.leak();
                }
                if let Some(ev) = self.fallback_event.take() {
                    ev.leak();
                }
                if let Some(ev) = self.write_event.take() {
                    ev.leak();
                }
                if let Some(ev) = self.read_event.take() {
                    ev.leak();
                }
            } else {
                drop(self.dev.take());
                drop(self.emergency_event.take());
                drop(self.fallback_event.take());
                drop(self.write_event.take());
                drop(self.read_event.take());
            }
            let _ = clear_slot_state_on_disconnect(&self.shared);
            self.shared.stop.store(true, Ordering::Release);
        }
    }

    /// Issues a bounded overlapped HID `WriteFile` using heap-allocated
    /// `PendingIo`. Never waits `INFINITE` after `CancelIoEx`: if the kernel
    /// driver fails to settle within `cancel_timeout_ms`, intentionally leaks
    /// `PendingIo` (`Box::leak`) and returns `WriteResult::Hung`.
    unsafe fn write_overlapped_bounded(
        handle: HANDLE,
        event: HANDLE,
        packet: &[u8],
        write_timeout_ms: u32,
        cancel_timeout_ms: u32,
    ) -> WriteResult {
        // SAFETY: zero-initialized `OVERLAPPED` is valid.
        let mut ov: OVERLAPPED = unsafe { std::mem::zeroed() };
        ov.hEvent = event;
        let mut io = Box::new(PendingIo {
            ov,
            buf: packet.to_vec(),
        });
        // SAFETY: caller guarantees valid open overlapped handle and exclusively
        // owned manual-reset event; `io` lives on the heap across I/O.
        unsafe {
            let _ = ResetEvent(event);
            let ok = WriteFile(
                handle,
                io.buf.as_ptr(),
                io.buf.len() as u32,
                null_mut(),
                &mut io.ov,
            );
            if ok != 0 {
                return WriteResult::Completed;
            }
            let err = GetLastError();
            if err != ERROR_IO_PENDING {
                return WriteResult::Failed;
            }
            let mut written = 0u32;
            if WaitForSingleObject(event, write_timeout_ms) == WAIT_OBJECT_0 {
                let done = GetOverlappedResult(handle, &io.ov, &mut written, 0);
                return if done != 0 {
                    WriteResult::Completed
                } else {
                    WriteResult::Failed
                };
            }
            // Timeout: cancel IRP and wait bounded `cancel_timeout_ms`.
            let _ = CancelIoEx(handle, &io.ov);
            if WaitForSingleObject(event, cancel_timeout_ms) == WAIT_OBJECT_0
                || GetOverlappedResult(handle, &io.ov, &mut written, 0) != 0
            {
                let done = GetOverlappedResult(handle, &io.ov, &mut written, 0);
                return if done != 0 {
                    WriteResult::Completed
                } else {
                    WriteResult::Failed
                };
            }
            // Still pending even after `CancelIoEx` + bounded wait: leak heap
            // `OVERLAPPED` + buffer so the wedged IRP never touches freed memory.
            let _ = Box::leak(io);
            WriteResult::Hung
        }
    }

    /// Strictly time-bounded overlapped write using `'static` `OVERLAPPED` and
    /// buffer memory. Never falls back to unbounded synchronous HID I/O.
    unsafe fn emergency_write_bounded(
        handle: HANDLE,
        event: HANDLE,
        ov: &mut OVERLAPPED,
        packet: &[u8],
        write_timeout_ms: u32,
        cancel_timeout_ms: u32,
    ) -> EmergencyWriteOutcome {
        // SAFETY: caller holds `SLOT_EMERGENCY` with `!SLOT_HUNG`, guaranteeing
        // exclusive ownership of `'static` `ov`/`packet` and live `handle`/`event`.
        unsafe {
            let _ = ResetEvent(event);
            let ok = WriteFile(
                handle,
                packet.as_ptr(),
                packet.len() as u32,
                null_mut(),
                ov,
            );
            if ok != 0 {
                return EmergencyWriteOutcome::Completed;
            }
            if GetLastError() != ERROR_IO_PENDING {
                return EmergencyWriteOutcome::ImmediateError;
            }
            let mut written = 0u32;
            if WaitForSingleObject(event, write_timeout_ms) == WAIT_OBJECT_0 {
                let done = GetOverlappedResult(handle, ov, &mut written, 0);
                return if done != 0 {
                    EmergencyWriteOutcome::Completed
                } else {
                    EmergencyWriteOutcome::TimedOutSettled
                };
            }
            let _ = CancelIoEx(handle, ov);
            if WaitForSingleObject(event, cancel_timeout_ms) == WAIT_OBJECT_0
                || GetOverlappedResult(handle, ov, &mut written, 0) != 0
            {
                let _ = GetOverlappedResult(handle, ov, &mut written, 0);
                return EmergencyWriteOutcome::TimedOutSettled;
            }
            EmergencyWriteOutcome::Hung
        }
    }

    /// Best-effort cancel of pending overlapped reads/writes on all open slots
    /// to wake sleeping HID workers immediately during clean shutdown, skipping
    /// any slot that currently has an in-flight primary or fallback neutral write.
    /// If a delayed cancel overlaps a concurrent neutral reset, the `CancelIoGuard`
    /// Dekker handoff sends a guaranteed uncancelled follow-up neutral right after
    /// `CancelIoEx` returns.
    pub(super) fn cancel_all_open_handles() {
        for (slot_idx, slot) in EMERGENCY_SLOTS.iter().enumerate() {
            let Some(borrow) = slot.try_claim_emergency(Duration::from_millis(10)) else {
                continue;
            };
            if borrow.handle != 0 {
                let _ = slot.cancel_io_if_no_neutral_in_flight(
                    borrow.handle,
                    |h| {
                        // SAFETY: `borrow` prevents the worker from closing `handle`
                        // while `CancelIoEx` executes, and `io_phase` guarantees no
                        // primary or fallback neutral write is in flight.
                        unsafe {
                            let _ = CancelIoEx(h as HANDLE, null());
                        }
                    },
                    || {
                        let outcome = slot.deliver_or_handoff_neutral(
                            super::PHASE_PRIMARY_NEUTRAL,
                            true,
                            |release_leds, remaining| {
                                // SAFETY: `borrow` holds `SLOT_EMERGENCY` with `!SLOT_HUNG`,
                                // granting exclusive access to `'static` `slot.io` and
                                // keeping `handle`/`emergency_event` alive.
                                let io = unsafe { &mut *slot.io.get() };
                                // SAFETY: as above.
                                unsafe {
                                    send_neutral_static(
                                        slot,
                                        io,
                                        borrow.handle as HANDLE,
                                        borrow.emergency_event as HANDLE,
                                        borrow.transport,
                                        borrow.out_len,
                                        borrow.fw_ver,
                                        release_leds,
                                        remaining,
                                    )
                                }
                            },
                        );
                        let handed_off = (slot.io_phase.load(Ordering::SeqCst)
                            & (super::PHASE_PRIMARY_NEUTRAL | super::PHASE_FALLBACK))
                            != 0
                            || slot.emergency_in_flight.load(Ordering::SeqCst)
                            || slot.neutral_after_cancel.load(Ordering::SeqCst);
                        if outcome != NeutralSendOutcome::Delivered && !handed_off {
                            tracing::warn!(
                                slot = slot_idx,
                                ?outcome,
                                "DualSense post-cancel neutral release unconfirmed"
                            );
                        }
                    },
                );
            }
        }
    }

    /// Transmits one strictly bounded overlapped neutral packet using caller-owned
    /// `'static` `EmergencyIo` and event handle, honoring dynamic `release_leds`
    /// and strictly bounding overlapped write + cancel waits within `remaining`.
    #[allow(clippy::too_many_arguments)]
    unsafe fn send_neutral_static(
        slot: &EmergencySlot,
        io: &mut super::EmergencyIo,
        handle: HANDLE,
        event: HANDLE,
        transport: Transport,
        out_len: usize,
        fw_ver: u16,
        release_leds: bool,
        remaining: Option<Duration>,
    ) -> NeutralSendOutcome {
        if handle.is_null() || event.is_null() {
            return NeutralSendOutcome::Failed;
        }
        let Some((write_ms, cancel_ms)) = super::split_overlapped_budget(
            remaining,
            EMERGENCY_WRITE_TIMEOUT_MS,
            EMERGENCY_CANCEL_TIMEOUT_MS,
        ) else {
            return NeutralSendOutcome::Failed;
        };
        let cmd = OutputCommand::neutral(fw_ver, release_leds);
        // SAFETY: zero-initialized `OVERLAPPED` is valid.
        let ov = io.ov.write(unsafe { std::mem::zeroed() });
        ov.hEvent = event;
        let bt_seq = if transport == Transport::Bluetooth {
            slot.next_bt_seq()
        } else {
            0
        };
        let len = encode_output_packet(transport, out_len, bt_seq, &cmd, &mut io.buf);
        // SAFETY: `'static` `ov` and `io.buf` outlive any hung IRP.
        let outcome = unsafe {
            emergency_write_bounded(handle, event, ov, &io.buf[..len], write_ms, cancel_ms)
        };
        classify_emergency_write(outcome)
    }

    /// Compile-time proof that `FILE_TYPE_CHAR_RAW` matches Win32 `FILE_TYPE_CHAR`.
    const _: () = assert!(FILE_TYPE_CHAR == super::FILE_TYPE_CHAR_RAW);

    /// Crash-handler-safe (lock-free, allocation-free, pipe-safe) diagnostic:
    /// always records `slot_idx` in `FATAL_UNCONFIRMED_SLOTS`, and ONLY invokes
    /// `WriteFile` on `STD_ERROR_HANDLE` when `GetFileType(stderr) == FILE_TYPE_CHAR`
    /// so a full redirected stderr pipe can NEVER block fatal crash cleanup.
    fn write_fatal_unconfirmed_stderr(slot_idx: usize) {
        super::record_fatal_unconfirmed_slot(slot_idx);
        let mut msg = *b"victory: dualsense fatal neutral reset unconfirmed on slot 0\n";
        msg[msg.len() - 2] = b'0' + (slot_idx as u8).min(9);
        // SAFETY: `GetStdHandle` + `GetFileType` + `WriteFile` with a stack byte
        // array performs zero heap allocations, acquires no Rust/tracing locks,
        // and skips pipes/disk/unknown handles.
        unsafe {
            let stderr = GetStdHandle(STD_ERROR_HANDLE);
            if !stderr.is_null()
                && stderr != INVALID_HANDLE_VALUE
                && super::should_write_fatal_stderr(GetFileType(stderr))
            {
                let mut written = 0u32;
                let _ = WriteFile(
                    stderr,
                    msg.as_ptr(),
                    msg.len() as u32,
                    &mut written,
                    null_mut(),
                );
            }
        }
    }

    pub(super) fn emergency_reset_all(fatal: bool) {
        for (slot_idx, slot) in EMERGENCY_SLOTS.iter().enumerate() {
            let outcome = slot.run_emergency_reset(
                fatal,
                EMERGENCY_CLAIM_WAIT,
                EMERGENCY_WRITE_WAIT,
                |handle_usize| {
                    // SAFETY: `borrow`/`cancel_borrow` keeps `handle` alive across `CancelIoEx`.
                    unsafe {
                        let _ = CancelIoEx(handle_usize as HANDLE, null());
                    }
                },
                |borrow, release_leds, remaining| {
                    // SAFETY: we hold `SLOT_EMERGENCY` with `!SLOT_HUNG`, granting
                    // exclusive mutable access to this slot's `'static` `slot.io`.
                    let io = unsafe { &mut *slot.io.get() };
                    // SAFETY: `borrow` keeps `handle` and `emergency_event` alive.
                    unsafe {
                        send_neutral_static(
                            slot,
                            io,
                            borrow.handle as HANDLE,
                            borrow.emergency_event as HANDLE,
                            borrow.transport,
                            borrow.out_len,
                            borrow.fw_ver,
                            release_leds,
                            remaining,
                        )
                    }
                },
                |cancel_borrow, release_leds, remaining| {
                    // SAFETY: `run_emergency_reset` acquired `PHASE_FALLBACK` in
                    // `slot.io_phase` exclusively with `!SLOT_HUNG`, granting
                    // exclusive access to `'static` `slot.fallback_io` without
                    // racing `slot.io`.
                    let io = unsafe { &mut *slot.fallback_io.get() };
                    // SAFETY: `cancel_borrow` keeps `handle` and `fallback_event` alive.
                    unsafe {
                        send_neutral_static(
                            slot,
                            io,
                            cancel_borrow.handle as HANDLE,
                            cancel_borrow.fallback_event as HANDLE,
                            cancel_borrow.transport,
                            cancel_borrow.out_len,
                            cancel_borrow.fw_ver,
                            release_leds,
                            remaining,
                        )
                    }
                },
                || {},
            );
            if outcome == ResetOutcome::Unconfirmed {
                if fatal {
                    write_fatal_unconfirmed_stderr(slot_idx);
                } else {
                    tracing::warn!(
                        slot = slot_idx,
                        "DualSense non-fatal emergency neutral reset unconfirmed"
                    );
                }
            }
        }
    }

    pub(super) fn scan_and_spawn(
        slots: &mut [Option<Arc<SharedSlot>>; MAX_DUALSENSE],
        config: &DualSenseConfig,
    ) {
        if SHUTDOWN_REQUESTED.load(Ordering::Acquire) {
            return;
        }
        let paths = enumerate_hid_interfaces();
        let mut topology = 0u8;
        for (_, lower) in &paths {
            let (is_virtual, is_physical) = classify_xinput_hid_path(lower);
            if is_virtual {
                topology |= XINPUT_VIRTUAL_BIT;
            }
            if is_physical {
                topology |= XINPUT_PHYSICAL_BIT;
            }
        }
        XINPUT_TOPOLOGY.store(topology, Ordering::Release);

        for (wide, lower) in paths {
            if SHUTDOWN_REQUESTED.load(Ordering::Acquire) {
                break;
            }
            if !lower.contains("054c") {
                continue;
            }
            if slots
                .iter()
                .flatten()
                .any(|s| !s.stop.load(Ordering::Acquire) && s.path == lower)
            {
                continue;
            }
            let Some(free_slot) = slots
                .iter()
                .enumerate()
                .position(|(idx, s)| s.is_none() && !EMERGENCY_SLOTS[idx].is_hung())
            else {
                break;
            };
            let Some(dev) = try_open_dualsense(lower, wide) else {
                continue;
            };
            let Some(ticket) = worker_coordinator().begin_spawn() else {
                // Shutdown already started; `dev` drops cleanly closing `dev.handle`.
                break;
            };
            let shared = Arc::new(SharedSlot {
                path: dev.path.clone(),
                transport: dev.transport,
                container_id: dev.container_id,
                stop: AtomicBool::new(false),
                state: std::sync::Mutex::new(WorkerState::new(free_slot, config.clone())),
            });
            slots[free_slot] = Some(Arc::clone(&shared));
            let worker_shared = Arc::clone(&shared);
            if ticket
                .spawn(format!("victory dualsense {free_slot}"), move |worker_reg| {
                    run_device_worker(free_slot, dev, worker_shared, worker_reg)
                })
                .is_none()
            {
                // Both `dev` and `ticket` dropped on spawn error, closing
                // `dev.handle` and clearing `live_workers` + `pending_registrations`.
                slots[free_slot] = None;
            }
        }
    }

    fn enumerate_hid_interfaces() -> Vec<(Vec<u16>, String)> {
        let mut guid = windows_sys::core::GUID {
            data1: 0,
            data2: 0,
            data3: 0,
            data4: [0; 8],
        };
        // SAFETY: writes the system HID interface class GUID.
        unsafe { HidD_GetHidGuid(&mut guid) };

        let mut len = 0u32;
        // SAFETY: queries required UTF-16 multi-string length.
        let cr = unsafe {
            CM_Get_Device_Interface_List_SizeW(
                &mut len,
                &guid,
                null(),
                CM_GET_DEVICE_INTERFACE_LIST_PRESENT,
            )
        };
        if cr != CR_SUCCESS || len <= 1 {
            return Vec::new();
        }

        let mut buf = vec![0u16; len as usize + 16];
        // SAFETY: `buf` holds room for the queried multi-string list.
        let cr = unsafe {
            CM_Get_Device_Interface_ListW(
                &guid,
                null(),
                buf.as_mut_ptr(),
                buf.len() as u32,
                CM_GET_DEVICE_INTERFACE_LIST_PRESENT,
            )
        };
        if cr != CR_SUCCESS {
            return Vec::new();
        }

        let mut out = Vec::new();
        let mut start = 0usize;
        while start < buf.len() && buf[start] != 0 {
            let end = buf[start..]
                .iter()
                .position(|&c| c == 0)
                .map_or(buf.len(), |p| start + p);
            let mut wide = buf[start..end].to_vec();
            let lower = String::from_utf16_lossy(&wide).to_ascii_lowercase();
            wide.push(0);
            out.push((wide, lower));
            start = end + 1;
        }
        out
    }

    /// Queries the physical USB device's `DEVPKEY_Device_ContainerId` GUID for
    /// a HID interface path so WASAPI audio haptics can bind strictly 1-to-1
    /// to this exact physical controller.
    fn query_hid_container_id(path_wide: &[u16]) -> Option<ContainerId> {
        let mut prop_type = 0u32;
        let mut guid_bytes = [0u8; 16];
        let mut size = guid_bytes.len() as u32;
        // SAFETY: NUL-terminated UTF-16 device interface path and 16-byte GUID buffer.
        let cr = unsafe {
            CM_Get_Device_Interface_PropertyW(
                path_wide.as_ptr(),
                &DEVPKEY_Device_ContainerId,
                &mut prop_type,
                guid_bytes.as_mut_ptr(),
                &mut size,
                0,
            )
        };
        if cr == CR_SUCCESS && prop_type == DEVPROP_TYPE_GUID && size == 16 {
            return ContainerId::from_raw_guid_bytes(&guid_bytes);
        }

        // Standard Windows PnP path: interface -> DeviceInstanceId -> DevNode -> ContainerId.
        let mut inst_buf = [0u16; 260];
        let mut inst_size = (inst_buf.len() * size_of::<u16>()) as u32;
        let mut inst_type = 0u32;
        // SAFETY: queries UTF-16 DeviceInstanceId string for this device interface.
        let cr = unsafe {
            CM_Get_Device_Interface_PropertyW(
                path_wide.as_ptr(),
                &DEVPKEY_Device_InstanceId,
                &mut inst_type,
                inst_buf.as_mut_ptr().cast::<u8>(),
                &mut inst_size,
                0,
            )
        };
        if cr != CR_SUCCESS || inst_type != DEVPROP_TYPE_STRING || inst_size < 2 {
            return None;
        }
        let last = (inst_size as usize / size_of::<u16>()).min(inst_buf.len()) - 1;
        inst_buf[last] = 0;

        let mut devinst = 0u32;
        // SAFETY: NUL-terminated UTF-16 DeviceInstanceId string.
        let cr = unsafe {
            CM_Locate_DevNodeW(&mut devinst, inst_buf.as_ptr(), CM_LOCATE_DEVNODE_NORMAL)
        };
        if cr != CR_SUCCESS {
            return None;
        }

        prop_type = 0;
        guid_bytes = [0u8; 16];
        size = guid_bytes.len() as u32;
        // SAFETY: valid devinst from `CM_Locate_DevNodeW` and 16-byte GUID buffer.
        let cr = unsafe {
            CM_Get_DevNode_PropertyW(
                devinst,
                &DEVPKEY_Device_ContainerId,
                &mut prop_type,
                guid_bytes.as_mut_ptr(),
                &mut size,
                0,
            )
        };
        if cr == CR_SUCCESS && prop_type == DEVPROP_TYPE_GUID && size == 16 {
            ContainerId::from_raw_guid_bytes(&guid_bytes)
        } else {
            None
        }
    }

    fn try_open_dualsense(path: String, path_wide: Vec<u16>) -> Option<Discovered> {
        // Always open with FILE_SHARE_READ | FILE_SHARE_WRITE so DS5Dongle's
        // browser WebHID configurator and companion tools are never locked out.
        // SAFETY: NUL-terminated UTF-16 device interface path.
        let raw_handle = unsafe {
            CreateFileW(
                path_wide.as_ptr(),
                GENERIC_READ | GENERIC_WRITE,
                FILE_SHARE_READ | FILE_SHARE_WRITE,
                null(),
                OPEN_EXISTING,
                FILE_FLAG_OVERLAPPED,
                null_mut(),
            )
        };
        let handle = OwnedHandle::new(raw_handle)?;

        let mut attrs = HIDD_ATTRIBUTES {
            Size: size_of::<HIDD_ATTRIBUTES>() as u32,
            VendorID: 0,
            ProductID: 0,
            VersionNumber: 0,
        };
        // SAFETY: valid open HID handle and sized `HIDD_ATTRIBUTES`.
        let ok_attr = unsafe { HidD_GetAttributes(handle.raw(), &mut attrs) };
        if !ok_attr || attrs.VendorID != SONY_VID {
            return None;
        }
        let model = ControllerModel::from_pid(attrs.ProductID)?;

        let mut preparsed: PHIDP_PREPARSED_DATA = 0;
        // SAFETY: queries preparsed HID descriptor for `handle`.
        if !unsafe { HidD_GetPreparsedData(handle.raw(), &mut preparsed) } || preparsed == 0 {
            return None;
        }
        // SAFETY: `HIDP_CAPS` is a plain C integer struct valid when zero-initialized.
        let mut caps: HIDP_CAPS = unsafe { std::mem::zeroed() };
        // SAFETY: valid preparsed descriptor from `HidD_GetPreparsedData`.
        let status = unsafe { HidP_GetCaps(preparsed, &mut caps) };
        // SAFETY: frees preparsed descriptor allocated by `HidD_GetPreparsedData`.
        unsafe {
            let _ = HidD_FreePreparsedData(preparsed);
        }

        // Filter strictly for Generic Desktop (0x01) / Game Pad (0x05) so we never
        // bind DS5Dongle's vendor/WebHID config interface or audio HID controls.
        if status != HIDP_STATUS_SUCCESS
            || caps.UsagePage != USAGE_PAGE_GENERIC_DESKTOP
            || caps.Usage != USAGE_GAMEPAD
        {
            return None;
        }

        let transport = if caps.InputReportByteLength >= BT_REPORT_LEN as u16 {
            Transport::Bluetooth
        } else {
            Transport::Usb
        };
        // Reject unsupported OutputReportByteLength (0, short, or > 128) so
        // normal and emergency reports always use identical exact lengths.
        let out_len = validate_output_report_len(caps.OutputReportByteLength, transport)?;
        let in_len = usize::from(caps.InputReportByteLength).max(64);
        // Preserve exact descriptor `FeatureReportByteLength` for `HidD_GetFeature`.
        let feat_len = usize::from(caps.FeatureReportByteLength);
        let container_id = query_hid_container_id(&path_wide);

        Some(Discovered {
            path,
            path_wide,
            handle,
            container_id,
            model,
            transport,
            in_len,
            out_len,
            feat_len,
        })
    }

    fn query_features(dev: &Discovered) -> (Option<ImuCalibration>, Option<u16>) {
        // Pass exact `FeatureReportByteLength` (or report size only when the
        // descriptor reports 0); skip any report larger than `FeatureReportByteLength`.
        let cal = feature_buffer_len(dev.feat_len, FEATURE_CALIBRATION_LEN).and_then(|buf_len| {
            let mut cal_buf = vec![0u8; buf_len];
            cal_buf[0] = FEATURE_CALIBRATION_ID;
            // SAFETY: valid HID handle and buffer sized to descriptor `FeatureReportByteLength`
            // starting with report ID 0x05; on Bluetooth reading 0x05 also switches
            // the DualSense into enhanced 0x31 report mode.
            let got_cal = unsafe {
                HidD_GetFeature(
                    dev.handle.raw(),
                    cal_buf.as_mut_ptr().cast(),
                    cal_buf.len() as u32,
                )
            };
            got_cal
                .then(|| ImuCalibration::from_feature_report(&cal_buf, dev.transport))
                .flatten()
        });

        let fw = feature_buffer_len(dev.feat_len, FEATURE_FIRMWARE_LEN).and_then(|buf_len| {
            let mut fw_buf = vec![0u8; buf_len];
            fw_buf[0] = FEATURE_FIRMWARE_ID;
            // SAFETY: valid HID handle and buffer sized to descriptor `FeatureReportByteLength`
            // starting with report ID 0x20.
            let got_fw = unsafe {
                HidD_GetFeature(
                    dev.handle.raw(),
                    fw_buf.as_mut_ptr().cast(),
                    fw_buf.len() as u32,
                )
            };
            got_fw
                .then(|| parse_firmware_version(&fw_buf, dev.transport))
                .flatten()
        });

        (cal, fw)
    }

    fn send_output_outcome(
        slot_idx: usize,
        dev: &Discovered,
        write_event: HANDLE,
        cmd: &OutputCommand,
    ) -> NeutralSendOutcome {
        let slot = &EMERGENCY_SLOTS[slot_idx];
        let mut stack_buf = [0u8; MAX_EMERGENCY_REPORT_LEN];
        let bt_seq = if dev.transport == Transport::Bluetooth {
            slot.next_bt_seq()
        } else {
            0
        };
        let len = encode_output_packet(dev.transport, dev.out_len, bt_seq, cmd, &mut stack_buf);
        let packet = &stack_buf[..len];

        // Strictly bounded overlapped `WriteFile` only (never unbounded synchronous HID I/O).
        // SAFETY: `dev.handle` and `write_event` stay open for the entire
        // duration of this worker thread.
        match unsafe {
            write_overlapped_bounded(
                dev.handle.raw(),
                write_event,
                packet,
                NORMAL_WRITE_TIMEOUT_MS,
                NORMAL_CANCEL_TIMEOUT_MS,
            )
        } {
            WriteResult::Completed => NeutralSendOutcome::Delivered,
            WriteResult::Failed => NeutralSendOutcome::Failed,
            WriteResult::Hung => {
                slot.mark_hung();
                NeutralSendOutcome::Hung
            }
        }
    }

    fn send_output(
        slot_idx: usize,
        dev: &Discovered,
        write_event: HANDLE,
        cmd: &OutputCommand,
    ) -> bool {
        send_output_outcome(slot_idx, dev, write_event, cmd) == NeutralSendOutcome::Delivered
    }

    /// Sends an output report serialized against emergency resets via Dekker/`SeqCst`.
    /// Returns `(write_ok, preempted_by_emergency)`.
    fn send_output_serialized(
        slot_idx: usize,
        dev: &Discovered,
        write_event: HANDLE,
        cmd: &OutputCommand,
    ) -> (bool, bool) {
        let slot = &EMERGENCY_SLOTS[slot_idx];
        let res = slot.run_serialized_write(
            &SHUTDOWN_REQUESTED,
            || {},
            || send_output(slot_idx, dev, write_event, cmd),
            |release_leds| {
                send_output_outcome(
                    slot_idx,
                    dev,
                    write_event,
                    &OutputCommand::neutral(cmd.firmware_version, release_leds),
                )
            },
        );
        if let Some(outcome) = res.trailing_neutral
            && outcome != NeutralSendOutcome::Delivered
        {
            let handed_off = (slot.io_phase.load(Ordering::SeqCst)
                & (super::PHASE_PRIMARY_NEUTRAL | super::PHASE_FALLBACK))
                != 0
                || slot.emergency_in_flight.load(Ordering::SeqCst)
                || slot.neutral_after_cancel.load(Ordering::SeqCst);
            if !handed_off {
                tracing::warn!(
                    slot = slot_idx,
                    ?outcome,
                    "DualSense preempted-write trailing neutral release unconfirmed"
                );
            }
        }
        (res.write_ok, res.preempted)
    }

    #[allow(clippy::too_many_arguments)]
    fn on_disconnect(
        slot_idx: usize,
        dev: &Discovered,
        write_event: HANDLE,
        shared: &SharedSlot,
        sys_filter: &mut SystemButtonFilter,
        limiter: &mut ReportLimiter,
        pending_disconnect_reset: &mut Option<DisconnectReset>,
    ) {
        tracing::info!(
            slot = slot_idx,
            "DualSense wireless link idle/disconnected (restoring fallback)"
        );
        sys_filter.reset();
        limiter.reset();
        let fw = clear_slot_state_on_disconnect(shared);

        // Attempt to release active adaptive trigger resistance & rumble right
        // away in case only the HID IN pipe stalled while OUT still works.
        if !SHUTDOWN_REQUESTED.load(Ordering::Acquire) {
            let (ok, _) = send_output_serialized(
                slot_idx,
                dev,
                write_event,
                &OutputCommand::neutral(fw, false),
            );
            *pending_disconnect_reset = (!ok).then(|| DisconnectReset {
                firmware_version: fw,
                last_attempt: Instant::now(),
            });
        }
    }

    fn run_device_worker(
        slot_idx: usize,
        dev: Discovered,
        shared: Arc<SharedSlot>,
        worker_reg: WorkerRegistration,
    ) {
        // Declared FIRST so `_worker_reg` drops LAST (after `RegisteredHidHandle::drop`
        // finishes cancelling I/O and closing/leaking handles, even on panic unwind!).
        let _worker_reg = worker_reg;
        let _ = &dev.path_wide;

        // SAFETY: manual-reset unnamed Win32 events for overlapped ReadFile /
        // WriteFile, emergency reset I/O, and fatal-timeout fallback I/O.
        let read_event = OwnedHandle::new(unsafe { CreateEventW(null(), 1, 0, null()) });
        // SAFETY: as above.
        let write_event = OwnedHandle::new(unsafe { CreateEventW(null(), 1, 0, null()) });
        // SAFETY: as above.
        let emergency_event = OwnedHandle::new(unsafe { CreateEventW(null(), 1, 0, null()) });
        // SAFETY: as above.
        let fallback_event = OwnedHandle::new(unsafe { CreateEventW(null(), 1, 0, null()) });

        let (Some(read_event), Some(write_event), Some(emergency_event), Some(fallback_event)) =
            (read_event, write_event, emergency_event, fallback_event)
        else {
            shared.stop.store(true, Ordering::Release);
            return;
        };
        if SHUTDOWN_REQUESTED.load(Ordering::Acquire) {
            shared.stop.store(true, Ordering::Release);
            return;
        }

        let Some(mut hid) = RegisteredHidHandle::register(
            slot_idx,
            dev,
            read_event,
            write_event,
            emergency_event,
            fallback_event,
            Arc::clone(&shared),
        ) else {
            shared.stop.store(true, Ordering::Release);
            return;
        };

        // Initial feature probe (also triggers Bluetooth 0x01 -> 0x31 enhanced mode switch).
        let (init_cal, init_fw) = query_features(hid.dev());
        {
            let mut st = shared.state.lock().unwrap_or_else(|e| e.into_inner());
            if let Some(cal) = init_cal {
                st.gyro.set_calibration(cal);
            }
            if let Some(fw) = init_fw {
                st.firmware_version = fw;
                hid.slot().set_firmware_version(fw);
            }
        }

        let mut liveness = LivenessMonitor::default();
        let mut sys_filter = SystemButtonFilter::default();
        let mut limiter = ReportLimiter::default();
        let mut pending_disconnect_reset: Option<DisconnectReset> = None;

        loop {
            if shared.stop.load(Ordering::Acquire)
                || SHUTDOWN_REQUESTED.load(Ordering::Acquire)
                || hid.slot().is_hung()
            {
                break;
            }

            if !hid.read_pending {
                let handle = hid.dev().handle.raw();
                let read_event = hid.read_event_raw();
                let io = hid
                    .read_io
                    .as_mut()
                    .expect("read_io present while !read_pending");
                // SAFETY: resets manual-reset event and starts overlapped HID interrupt IN read.
                let read_start = unsafe {
                    let _ = ResetEvent(read_event);
                    io.ov = std::mem::zeroed();
                    io.ov.hEvent = read_event;
                    let mut got = 0u32;
                    let ok = ReadFile(
                        handle,
                        io.buf.as_mut_ptr(),
                        io.buf.len() as u32,
                        &mut got,
                        &mut io.ov,
                    );
                    if ok != 0 {
                        Ok(Some(got as usize))
                    } else {
                        let err = GetLastError();
                        if err == ERROR_IO_PENDING {
                            Ok(None)
                        } else {
                            Err(err)
                        }
                    }
                };
                match read_start {
                    Ok(Some(got)) => {
                        let write_event = hid.write_event_raw();
                        let dev = hid.dev.as_ref().expect("dev alive");
                        let packet = &hid.read_io.as_ref().expect("read_io alive").buf[..got];
                        handle_packet(
                            slot_idx,
                            dev,
                            write_event,
                            &shared,
                            packet,
                            &mut liveness,
                            &mut sys_filter,
                            &mut limiter,
                            &mut pending_disconnect_reset,
                        );
                        flush_pending_output(
                            slot_idx,
                            dev,
                            write_event,
                            &shared,
                            &liveness,
                            &mut limiter,
                            &mut pending_disconnect_reset,
                        );
                        continue;
                    }
                    Ok(None) => {
                        hid.read_pending = true;
                    }
                    Err(ERROR_OPERATION_ABORTED) => continue,
                    Err(_) => break,
                }
            }

            // Wait up to 20 ms so output commands and DS5Dongle 1.5s freeze checks
            // run promptly even when a wireless controller turns off mid-session.
            // SAFETY: valid event handle.
            let wait = unsafe { WaitForSingleObject(hid.read_event_raw(), 20) };
            if wait == WAIT_OBJECT_0 {
                let handle = hid.dev().handle.raw();
                let mut got = 0u32;
                // SAFETY: queries completed overlapped `ReadFile`.
                let ok = unsafe {
                    let io = hid.read_io.as_ref().expect("read_io alive");
                    GetOverlappedResult(handle, &io.ov, &mut got, 0)
                };
                hid.read_pending = false;
                if ok == 0 {
                    // SAFETY: plain GetLastError check.
                    let err = unsafe { GetLastError() };
                    if err == ERROR_IO_INCOMPLETE {
                        hid.read_pending = true;
                    } else if err == ERROR_OPERATION_ABORTED {
                        continue;
                    } else {
                        break;
                    }
                } else {
                    let write_event = hid.write_event_raw();
                    let dev = hid.dev.as_ref().expect("dev alive");
                    let packet = &hid.read_io.as_ref().expect("read_io alive").buf[..got as usize];
                    handle_packet(
                        slot_idx,
                        dev,
                        write_event,
                        &shared,
                        packet,
                        &mut liveness,
                        &mut sys_filter,
                        &mut limiter,
                        &mut pending_disconnect_reset,
                    );
                }
            } else if liveness.check_timeout(Instant::now()) == LivenessEvent::Disconnected {
                let write_event = hid.write_event_raw();
                on_disconnect(
                    slot_idx,
                    hid.dev(),
                    write_event,
                    &shared,
                    &mut sys_filter,
                    &mut limiter,
                    &mut pending_disconnect_reset,
                );
            }

            let write_event = hid.write_event_raw();
            flush_pending_output(
                slot_idx,
                hid.dev(),
                write_event,
                &shared,
                &liveness,
                &mut limiter,
                &mut pending_disconnect_reset,
            );
        }

        // Cancel pending read with bounded wait, then send a final neutral
        // release coordinated through `EmergencySlot` (`io_phase`,
        // `send_neutral_with_cancel_retry`, and `neutral_after_cancel`) BEFORE
        // `RegisteredHidHandle::drop` waits for active borrows/cancels and closes/leaks.
        hid.cancel_pending_read_bounded();
        if !hid.slot().is_hung() {
            let fw = shared
                .state
                .lock()
                .map_or(super::FIRMWARE_VIBRATION2_MIN, |s| s.firmware_version);
            let outcome = hid
                .slot()
                .run_worker_shutdown_neutral(|release_leds, _remaining| {
                    send_output_outcome(
                        slot_idx,
                        hid.dev(),
                        hid.write_event_raw(),
                        &OutputCommand::neutral(fw, release_leds),
                    )
                });
            let handed_off = (hid.slot().io_phase.load(Ordering::SeqCst)
                & (super::PHASE_PRIMARY_NEUTRAL | super::PHASE_FALLBACK))
                != 0
                || hid.slot().emergency_in_flight.load(Ordering::SeqCst)
                || hid.slot().neutral_after_cancel.load(Ordering::SeqCst);
            if outcome != NeutralSendOutcome::Delivered && !handed_off {
                tracing::warn!(
                    slot = slot_idx,
                    ?outcome,
                    "DualSense worker shutdown neutral release unconfirmed"
                );
            }
        }
        // `hid` drops first (closing/leaking handles safely after `begin_close_and_wait_bounded`),
        // then `_worker_reg` drops last (decrementing `live_workers` and waking shutdown).
    }

    #[allow(clippy::too_many_arguments)]
    fn handle_packet(
        slot_idx: usize,
        dev: &Discovered,
        write_event: HANDLE,
        shared: &SharedSlot,
        packet: &[u8],
        liveness: &mut LivenessMonitor,
        sys_filter: &mut SystemButtonFilter,
        limiter: &mut ReportLimiter,
        pending_disconnect_reset: &mut Option<DisconnectReset>,
    ) {
        let now = Instant::now();
        let (raw_opt, event) = on_raw_packet(packet, dev.transport, liveness, now);
        match event {
            LivenessEvent::Connected => {
                *pending_disconnect_reset = None;
                // Re-query calibration and firmware version when DS5Dongle's Bluetooth
                // controller wakes up after the dongle was already plugged into USB.
                let (cal, fw) = query_features(dev);
                let mut st = shared.state.lock().unwrap_or_else(|e| e.into_inner());
                st.rumble = [0.0; 4];
                if let Some(cal) = cal {
                    st.gyro.set_calibration(cal);
                }
                if let Some(fw) = fw {
                    st.firmware_version = fw;
                    EMERGENCY_SLOTS[slot_idx].set_firmware_version(fw);
                }
                st.mark_leds_dirty();
                limiter.reset();
                tracing::info!(
                    slot = slot_idx,
                    model = ?dev.model,
                    transport = ?dev.transport,
                    firmware = format_args!("{:#06x}", st.firmware_version),
                    "DualSense active"
                );
            }
            LivenessEvent::Disconnected => {
                on_disconnect(
                    slot_idx,
                    dev,
                    write_event,
                    shared,
                    sys_filter,
                    limiter,
                    pending_disconnect_reset,
                );
                return;
            }
            LivenessEvent::Unchanged => {
                if !liveness.is_connected() {
                    return;
                }
            }
        }
        let Some(raw) = raw_opt else {
            return;
        };

        let mut st = shared.state.lock().unwrap_or_else(|e| e.into_inner());
        if !st.snapshot.connected {
            // Ensure no stale rumble can leak across a fresh connect/reconnect boundary.
            st.rumble = [0.0; 4];
        }
        let mut pad = to_pad_state_without_touchpad(&raw, dev.model, &st.config);
        let filter = sys_filter.update(&raw, pad.buttons, &st.config, now);
        pad.buttons |= filter.touchpad_buttons;
        let space = st.config.gyro_space;
        st.gyro.push_sample(&raw, space);
        if filter.recenter_gyro {
            st.gyro.recenter();
            st.mark_leds_dirty();
            tracing::info!(slot = slot_idx, "DualSense gyroscope recentered");
        }
        st.raw_latest = Some(raw);
        st.snapshot = SlotSnapshot {
            connected: true,
            transport: Some(dev.transport),
            pad,
            battery: raw.battery_state(),
            system_buttons: filter.system_buttons,
        };
    }

    fn flush_pending_output(
        slot_idx: usize,
        dev: &Discovered,
        write_event: HANDLE,
        shared: &SharedSlot,
        liveness: &LivenessMonitor,
        limiter: &mut ReportLimiter,
        pending_disconnect_reset: &mut Option<DisconnectReset>,
    ) {
        if SHUTDOWN_REQUESTED.load(Ordering::Acquire) || EMERGENCY_SLOTS[slot_idx].is_hung() {
            return;
        }
        if !liveness.is_connected() {
            if let Some(pending) = pending_disconnect_reset.as_mut()
                && pending.last_attempt.elapsed() >= DISCONNECT_RESET_RETRY
            {
                let (ok, _) = send_output_serialized(
                    slot_idx,
                    dev,
                    write_event,
                    &OutputCommand::neutral(pending.firmware_version, false),
                );
                if ok {
                    *pending_disconnect_reset = None;
                } else {
                    pending.last_attempt = Instant::now();
                }
            }
            return;
        }
        let now = Instant::now();
        let (cmd, snapped_leds_epoch) = {
            let st = shared.state.lock().unwrap_or_else(|e| e.into_inner());
            if !st.focused {
                let neutral = OutputCommand::neutral(st.firmware_version, false);
                if !limiter.can_send(&neutral, now, false) {
                    return;
                }
                (neutral, None)
            } else {
                let cfg = &st.config;
                let enable_hid_rumble = should_enable_hid_rumble(
                    dev.transport,
                    cfg.haptics_mode,
                    st.audio_haptics_active,
                    st.multi_usb,
                );
                let to_u8 = |v: f32| {
                    if v.is_finite() {
                        (v.clamp(0.0, 1.0) * 255.0).round() as u8
                    } else {
                        0
                    }
                };
                let low_batt = st.raw_latest.is_some_and(|r| {
                    !r.basic_bt_report && r.charging_state == 0 && r.battery_level <= 1
                });
                let rgb = if low_batt {
                    [0xFF, 0x10, 0x00]
                } else {
                    cfg.lightbar_color
                };
                let cmd = OutputCommand {
                    firmware_version: st.firmware_version,
                    enable_hid_rumble,
                    motor_left: if enable_hid_rumble {
                        to_u8(st.rumble[0])
                    } else {
                        0
                    },
                    motor_right: if enable_hid_rumble {
                        to_u8(st.rumble[1])
                    } else {
                        0
                    },
                    left_trigger: encode_trigger_effect(
                        st.rumble[2],
                        cfg.adaptive_triggers,
                        cfg.trigger_strength,
                    ),
                    right_trigger: encode_trigger_effect(
                        st.rumble[3],
                        cfg.adaptive_triggers,
                        cfg.trigger_strength,
                    ),
                    update_leds: st.leds_dirty,
                    release_leds: false,
                    slot: st.host_slot,
                    lightbar_rgb: rgb,
                    brightness: cfg.led_brightness,
                };
                let force = st.leds_dirty;
                if !limiter.can_send(&cmd, now, force) {
                    return;
                }
                let epoch = st.leds_dirty.then_some(st.leds_epoch);
                (cmd, epoch)
            }
        };

        let (ok, preempted) = send_output_serialized(slot_idx, dev, write_event, &cmd);
        let done_at = Instant::now();
        if preempted {
            limiter.reset();
        } else if ok {
            limiter.record_sent(&cmd, done_at);
            if let Some(snapped) = snapped_leds_epoch {
                let mut st = shared.state.lock().unwrap_or_else(|e| e.into_inner());
                if st.leds_epoch == snapped {
                    st.leds_dirty = false;
                }
            }
        } else {
            limiter.record_failed(&cmd, done_at);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::modules::gameinput::{MENU, VIEW};
    use protocol::{
        BT_CRC_SEED_INPUT, BT_REPORT_ID, BT_REPORT_LEN, FEATURE_CALIBRATION_LEN,
        FEATURE_FIRMWARE_LEN, USB_INPUT_REPORT_ID, crc32_seeded,
    };

    /// RAII test harness mirroring `RegisteredHidHandle::drop` so panic-unwind
    /// safety can be tested deterministically without real Win32 HID devices.
    struct SlotCloseGuard<'a, F: FnOnce(bool)> {
        slot: &'a EmergencySlot,
        budget: Duration,
        on_close: Option<F>,
    }

    impl<F: FnOnce(bool)> Drop for SlotCloseGuard<'_, F> {
        fn drop(&mut self) {
            let must_leak = self.slot.begin_close_and_wait_bounded(self.budget);
            if let Some(f) = self.on_close.take() {
                f(must_leak);
            }
        }
    }

    #[test]
    fn dongle_liveness_ignores_standby_and_stays_disconnected_on_stale_timestamp() {
        let mut mon = LivenessMonitor::default();
        let t0 = Instant::now();

        let idle = RawReport::default();
        assert_eq!(mon.on_report(&idle, t0), LivenessEvent::Unchanged);
        assert!(!mon.is_connected());

        let mut live = RawReport {
            sensor_timestamp: 10_000,
            battery_level: 7,
            charging_state: 0,
            ..Default::default()
        };
        assert_eq!(mon.on_report(&live, t0), LivenessEvent::Connected);
        assert!(mon.is_connected());

        assert_eq!(
            mon.on_report(&live, t0 + Duration::from_millis(800)),
            LivenessEvent::Unchanged
        );
        assert_eq!(
            mon.on_report(&live, t0 + Duration::from_millis(1550)),
            LivenessEvent::Disconnected
        );
        assert!(!mon.is_connected());

        for extra_ms in [1600u64, 2000, 3500] {
            assert_eq!(
                mon.on_report(&live, t0 + Duration::from_millis(extra_ms)),
                LivenessEvent::Unchanged,
                "frozen DS5Dongle repeating stale sensor_timestamp must stay disconnected"
            );
            assert!(!mon.is_connected());
        }

        live.sensor_timestamp = 25_000;
        assert_eq!(
            mon.on_report(&live, t0 + Duration::from_millis(4000)),
            LivenessEvent::Connected
        );
        assert!(mon.is_connected());
    }

    #[test]
    fn corrupt_packet_stream_still_triggers_disconnect_timeout() {
        let mut mon = LivenessMonitor::default();
        let t0 = Instant::now();

        let mut bt = [0u8; BT_REPORT_LEN];
        bt[0] = BT_REPORT_ID;
        bt[1] = 0x02;
        bt[2 + 27..2 + 31].copy_from_slice(&42_000u32.to_le_bytes());
        bt[2 + 52] = 0x17;
        let crc = crc32_seeded(BT_CRC_SEED_INPUT, &bt[..BT_REPORT_LEN - 4]);
        bt[BT_REPORT_LEN - 4..].copy_from_slice(&crc.to_le_bytes());
        let (raw, ev) = on_raw_packet(&bt, Transport::Bluetooth, &mut mon, t0);
        assert!(raw.is_some());
        assert_eq!(ev, LivenessEvent::Connected);
        assert!(mon.is_connected());

        bt[BT_REPORT_LEN - 1] ^= 0xFF;
        let mut disconnected_at = None;
        for step in 1..=20u64 {
            let now = t0 + Duration::from_millis(step * 100);
            let (raw, ev) = on_raw_packet(&bt, Transport::Bluetooth, &mut mon, now);
            assert!(raw.is_none());
            if ev == LivenessEvent::Disconnected {
                disconnected_at = Some(step * 100);
                break;
            }
        }
        assert_eq!(disconnected_at, Some(1500));
        assert!(!mon.is_connected());
    }

    #[test]
    fn bt_basic_mode_stays_connected_past_timeout_then_disconnects_on_silence() {
        let mut mon = LivenessMonitor::default();
        let t0 = Instant::now();
        let basic_pkt = [USB_INPUT_REPORT_ID, 128, 128, 128, 128, 0, 0, 0, 0, 0];

        let (first, ev) = on_raw_packet(&basic_pkt, Transport::Bluetooth, &mut mon, t0);
        assert!(first.unwrap().basic_bt_report);
        assert_eq!(ev, LivenessEvent::Connected);

        for step in 1..=30u64 {
            let now = t0 + Duration::from_millis(step * 100);
            let (raw, ev) = on_raw_packet(&basic_pkt, Transport::Bluetooth, &mut mon, now);
            assert!(raw.is_some());
            assert_eq!(ev, LivenessEvent::Unchanged);
            assert!(mon.is_connected());
        }

        assert_eq!(
            mon.check_timeout(t0 + Duration::from_millis(4400)),
            LivenessEvent::Unchanged
        );
        assert_eq!(
            mon.check_timeout(t0 + Duration::from_millis(4500)),
            LivenessEvent::Disconnected
        );
        assert!(!mon.is_connected());
    }

    #[test]
    fn hid_rumble_enable_truth_table() {
        for transport in [Transport::Usb, Transport::Bluetooth] {
            for audio_active in [false, true] {
                for multi_usb in [false, true] {
                    assert!(!should_enable_hid_rumble(
                        transport,
                        HapticsMode::Off,
                        audio_active,
                        multi_usb
                    ));
                    assert!(should_enable_hid_rumble(
                        transport,
                        HapticsMode::Hid,
                        audio_active,
                        multi_usb
                    ));
                }
            }
        }
        assert!(!should_enable_hid_rumble(
            Transport::Usb,
            HapticsMode::Audio4Ch,
            false,
            false
        ));
        assert!(!should_enable_hid_rumble(
            Transport::Usb,
            HapticsMode::Audio4Ch,
            true,
            false
        ));
        assert!(should_enable_hid_rumble(
            Transport::Usb,
            HapticsMode::Audio4Ch,
            false,
            true
        ));
        assert!(should_enable_hid_rumble(
            Transport::Usb,
            HapticsMode::Audio4Ch,
            true,
            true
        ));
        assert!(should_enable_hid_rumble(
            Transport::Bluetooth,
            HapticsMode::Audio4Ch,
            false,
            false
        ));
        assert!(should_enable_hid_rumble(
            Transport::Bluetooth,
            HapticsMode::Audio4Ch,
            true,
            false
        ));
        assert!(should_enable_hid_rumble(
            Transport::Usb,
            HapticsMode::Hybrid,
            false,
            false
        ));
        assert!(!should_enable_hid_rumble(
            Transport::Usb,
            HapticsMode::Hybrid,
            true,
            false
        ));
        assert!(should_enable_hid_rumble(
            Transport::Usb,
            HapticsMode::Hybrid,
            true,
            true
        ));
        assert!(should_enable_hid_rumble(
            Transport::Bluetooth,
            HapticsMode::Hybrid,
            false,
            false
        ));
        assert!(should_enable_hid_rumble(
            Transport::Bluetooth,
            HapticsMode::Hybrid,
            true,
            false
        ));
    }

    #[test]
    fn feature_buffer_len_respects_descriptor_size() {
        assert_eq!(
            feature_buffer_len(0, FEATURE_CALIBRATION_LEN),
            Some(FEATURE_CALIBRATION_LEN)
        );
        assert_eq!(feature_buffer_len(64, FEATURE_CALIBRATION_LEN), Some(64));
        assert_eq!(feature_buffer_len(78, FEATURE_FIRMWARE_LEN), Some(78));
        assert_eq!(feature_buffer_len(41, FEATURE_FIRMWARE_LEN), None);
        assert_eq!(feature_buffer_len(16, FEATURE_CALIBRATION_LEN), None);
    }

    #[test]
    fn output_report_len_validation_rejects_zero_short_and_oversized() {
        // Real DualSense USB (48), DS5Dongle (64), Bluetooth (78), and up to 128 are accepted.
        assert_eq!(validate_output_report_len(48, Transport::Usb), Some(48));
        assert_eq!(validate_output_report_len(64, Transport::Usb), Some(64));
        assert_eq!(validate_output_report_len(128, Transport::Usb), Some(128));
        assert_eq!(
            validate_output_report_len(78, Transport::Bluetooth),
            Some(78)
        );
        assert_eq!(
            validate_output_report_len(128, Transport::Bluetooth),
            Some(128)
        );

        // Zero, too short, and > MAX_EMERGENCY_REPORT_LEN (128) MUST be rejected.
        assert_eq!(validate_output_report_len(0, Transport::Usb), None);
        assert_eq!(validate_output_report_len(0, Transport::Bluetooth), None);
        assert_eq!(validate_output_report_len(32, Transport::Usb), None);
        assert_eq!(validate_output_report_len(64, Transport::Bluetooth), None);
        assert_eq!(validate_output_report_len(129, Transport::Usb), None);
        assert_eq!(validate_output_report_len(256, Transport::Bluetooth), None);
    }

    #[test]
    fn ps_button_short_press_fires_guide_while_long_hold_and_chords_do_not() {
        let mut f = SystemButtonFilter::default();
        let cfg = DualSenseConfig::default();
        let t0 = Instant::now();
        let mut raw = RawReport::default();

        raw.buttons[2] = 0x01;
        assert_eq!(f.update(&raw, 0, &cfg, t0).system_buttons, 0);
        raw.buttons[2] = 0x00;
        assert_eq!(
            f.update(&raw, 0, &cfg, t0 + Duration::from_millis(150))
                .system_buttons,
            SYSTEM_BUTTON_GUIDE
        );
        assert_eq!(
            f.update(&raw, 0, &cfg, t0 + Duration::from_millis(300))
                .system_buttons,
            0
        );

        raw.buttons[2] = 0x01;
        let _ = f.update(&raw, 0, &cfg, t0 + Duration::from_secs(1));
        let _ = f.update(&raw, 0, &cfg, t0 + Duration::from_secs(3));
        raw.buttons[2] = 0x00;
        assert_eq!(
            f.update(&raw, 0, &cfg, t0 + Duration::from_millis(3100))
                .system_buttons,
            0
        );

        raw.buttons[2] = 0x01 | 0x02;
        let out = f.update(&raw, 0, &cfg, t0 + Duration::from_secs(4));
        assert!(out.recenter_gyro);
        assert_eq!(out.touchpad_buttons, 0);
        raw.buttons[2] = 0x00;
        let release = f.update(&raw, 0, &cfg, t0 + Duration::from_millis(4100));
        assert_eq!(release.system_buttons, 0);
        assert_eq!(release.touchpad_buttons, 0);
    }

    #[test]
    fn ps_touchpad_chord_arbitration_covers_release_order_grace_tap_and_hold() {
        let mut f = SystemButtonFilter::default();
        let cfg = DualSenseConfig::default();
        let t0 = Instant::now();
        let mut raw = RawReport::default();

        // 1. PS first, then Touchpad, then release PS FIRST while Touchpad is still held.
        raw.buttons[2] = 0x01;
        assert_eq!(f.update(&raw, 0, &cfg, t0).touchpad_buttons, 0);
        raw.buttons[2] = 0x01 | 0x02;
        let chord = f.update(&raw, 0, &cfg, t0 + Duration::from_millis(20));
        assert!(chord.recenter_gyro);
        assert_eq!(chord.touchpad_buttons, 0);
        assert_eq!(chord.system_buttons, 0);
        raw.buttons[2] = 0x02;
        for ms in [30u64, 80, 180] {
            let out = f.update(&raw, 0, &cfg, t0 + Duration::from_millis(ms));
            assert!(!out.recenter_gyro);
            assert_eq!(out.system_buttons, 0);
            assert_eq!(out.touchpad_buttons, 0);
        }
        raw.buttons[2] = 0x00;
        let after = f.update(&raw, 0, &cfg, t0 + Duration::from_millis(200));
        assert_eq!(after.touchpad_buttons, 0);
        assert_eq!(after.system_buttons, 0);

        // 2. Touchpad pressed FIRST, then PS arrives within 60 ms grace window.
        let t1 = t0 + Duration::from_secs(1);
        raw.buttons[2] = 0x02;
        assert_eq!(f.update(&raw, 0, &cfg, t1).touchpad_buttons, 0);
        raw.buttons[2] = 0x01 | 0x02;
        let chord2 = f.update(&raw, 0, &cfg, t1 + Duration::from_millis(45));
        assert!(chord2.recenter_gyro);
        assert_eq!(chord2.touchpad_buttons, 0);
        raw.buttons[2] = 0x02;
        assert_eq!(
            f.update(&raw, 0, &cfg, t1 + Duration::from_millis(120))
                .touchpad_buttons,
            0
        );
        raw.buttons[2] = 0x00;
        assert_eq!(
            f.update(&raw, 0, &cfg, t1 + Duration::from_millis(160))
                .touchpad_buttons,
            0
        );

        // 3. Fast Touchpad tap (< 60 ms) on right half emits an 80 ms MENU pulse.
        let t2 = t0 + Duration::from_secs(2);
        raw.touch[0] = protocol::TouchPoint {
            active: true,
            id: 1,
            x: 1500,
            y: 500,
        };
        raw.buttons[2] = 0x02;
        assert_eq!(f.update(&raw, 0, &cfg, t2).touchpad_buttons, 0);
        raw.buttons[2] = 0x00;
        raw.touch[0].active = false;
        assert_eq!(
            f.update(&raw, 0, &cfg, t2 + Duration::from_millis(30))
                .touchpad_buttons,
            MENU
        );
        assert_eq!(
            f.update(&raw, 0, &cfg, t2 + Duration::from_millis(80))
                .touchpad_buttons,
            MENU
        );
        assert_eq!(
            f.update(&raw, 0, &cfg, t2 + Duration::from_millis(120))
                .touchpad_buttons,
            0
        );

        // 4. Normal Touchpad hold (>= 60 ms) commits VIEW; PS afterwards aborts cleanly.
        let t3 = t0 + Duration::from_secs(3);
        raw.touch[0] = protocol::TouchPoint {
            active: true,
            id: 2,
            x: 200,
            y: 500,
        };
        raw.buttons[2] = 0x02;
        assert_eq!(f.update(&raw, 0, &cfg, t3).touchpad_buttons, 0);
        assert_eq!(
            f.update(&raw, 0, &cfg, t3 + Duration::from_millis(65))
                .touchpad_buttons,
            VIEW
        );
        raw.buttons[2] = 0x01 | 0x02;
        let abort = f.update(&raw, 0, &cfg, t3 + Duration::from_millis(120));
        assert!(abort.recenter_gyro);
        assert_eq!(abort.touchpad_buttons, 0);
        raw.buttons[2] = 0x02;
        assert_eq!(
            f.update(&raw, 0, &cfg, t3 + Duration::from_millis(180))
                .touchpad_buttons,
            0
        );
        raw.buttons[2] = 0x00;
        assert_eq!(
            f.update(&raw, 0, &cfg, t3 + Duration::from_millis(210))
                .touchpad_buttons,
            0
        );
    }

    #[test]
    fn xinput_hid_path_classification_distinguishes_virtual_and_physical() {
        assert_eq!(
            classify_xinput_hid_path(r"\\?\hid#vid_28de&pid_11ff&ig_00#7&123"),
            (true, false)
        );
        assert_eq!(
            classify_xinput_hid_path(r"\\?\hid#root#vigembus&ig_00#1&234"),
            (true, false)
        );
        assert_eq!(
            classify_xinput_hid_path(r"\\?\hid#vid_045e&pid_0b12&ig_00#7&456"),
            (false, true)
        );
        assert_eq!(
            classify_xinput_hid_path(r"\\?\hid#vid_054c&pid_0ce6#7&789"),
            (false, false)
        );
    }

    #[test]
    fn emergency_slot_serializes_worker_writes_blocks_close_and_tracks_hung_state() {
        let slot = Arc::new(EmergencySlot::new());
        assert!(
            slot.try_claim_emergency(Duration::from_millis(5))
                .is_none()
        );

        assert!(slot.publish(0x1234, 0x5678, Transport::Bluetooth, 78));

        let shutdown = AtomicBool::new(false);
        let mut borrow = slot
            .try_claim_emergency(Duration::from_millis(10))
            .expect("published slot must be borrowable");
        assert!(
            slot.try_acquire_worker_write(&shutdown).is_none(),
            "normal worker write must not start while emergency reset is active"
        );
        borrow.wait_and_acquire_writing(Duration::from_millis(10), || {});
        assert_eq!(borrow.handle, 0x1234);
        assert_eq!(borrow.emergency_event, 0x5678);
        assert!(borrow.acquired_writing);

        let closed = Arc::new(AtomicBool::new(false));
        let slot_bg = Arc::clone(&slot);
        let closed_bg = Arc::clone(&closed);
        let closer = std::thread::spawn(move || {
            let must_leak = slot_bg.begin_close_and_wait_bounded(Duration::from_millis(500));
            assert!(!must_leak);
            closed_bg.store(true, Ordering::Release);
        });

        std::thread::sleep(Duration::from_millis(15));
        assert!(
            !closed.load(Ordering::Acquire),
            "handle close must wait until active emergency borrow drops"
        );
        drop(borrow);
        closer.join().unwrap();
        assert!(closed.load(Ordering::Acquire));
        assert_eq!(slot.handle.load(Ordering::Acquire), 0);
        assert!(
            slot.try_claim_emergency(Duration::from_millis(5))
                .is_none()
        );

        // Verify SLOT_HUNG keeps handles/static I/O permanently frozen and un-republishable.
        let hung_slot = EmergencySlot::new();
        assert!(hung_slot.publish(0xAAAA, 0xBBBB, Transport::Usb, 48));
        {
            let _b = hung_slot
                .try_claim_emergency(Duration::from_millis(5))
                .unwrap();
            hung_slot.mark_hung();
        }
        assert!(hung_slot.begin_close_and_wait_bounded(Duration::from_millis(50)));
        assert_eq!(hung_slot.handle.load(Ordering::Acquire), 0xAAAA);
        assert!(!hung_slot.publish(0xCCCC, 0xDDDD, Transport::Usb, 48));
        assert!(
            hung_slot
                .try_claim_emergency(Duration::from_millis(5))
                .is_none()
        );
    }

    #[test]
    fn panic_unwind_runs_begin_close_and_waits_for_emergency_borrow() {
        let slot = Arc::new(EmergencySlot::new());
        assert!(slot.publish(0x7777, 0x8888, Transport::Usb, 48));
        let borrow = slot
            .try_claim_emergency(Duration::from_millis(10))
            .expect("borrow succeeds");

        let closed = Arc::new(AtomicBool::new(false));
        let leaked = Arc::new(AtomicBool::new(false));
        let (s_w, c_w, l_w) = (Arc::clone(&slot), Arc::clone(&closed), Arc::clone(&leaked));
        let worker = std::thread::spawn(move || {
            let res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let _guard = SlotCloseGuard {
                    slot: &s_w,
                    budget: Duration::from_millis(500),
                    on_close: Some(move |must_leak: bool| {
                        l_w.store(must_leak, Ordering::Release);
                        c_w.store(true, Ordering::Release);
                    }),
                };
                panic!("simulated HID worker panic while EmergencyBorrow is held");
            }));
            assert!(res.is_err());
        });

        // Wait while holding `borrow`: panic unwind MUST be blocked inside
        // `begin_close_and_wait_bounded` and MUST NOT close the handle yet!
        std::thread::sleep(Duration::from_millis(25));
        assert!(
            !closed.load(Ordering::Acquire),
            "panic unwind must not close handle while EmergencyBorrow is held"
        );
        drop(borrow);
        worker.join().unwrap();
        assert!(closed.load(Ordering::Acquire));
        assert!(!leaked.load(Ordering::Acquire));
        assert_eq!(slot.handle.load(Ordering::Acquire), 0);
    }

    #[test]
    fn concurrent_shutdown_waiters_all_block_until_workers_exit_and_reject_new_spawns() {
        use std::sync::Barrier;

        let coord = Arc::new(WorkerCoordinator::new());
        let ticket = coord.begin_spawn().expect("pre-shutdown spawn allowed");
        let pending_ticket = coord
            .begin_spawn()
            .expect("second pre-shutdown ticket allowed");
        let worker_done = Arc::new(AtomicBool::new(false));
        let release_worker = Arc::new(Barrier::new(2));

        let (wd_w, rw_w) = (Arc::clone(&worker_done), Arc::clone(&release_worker));
        let _shared_handle = ticket
            .spawn("test worker".into(), move |reg| {
                let _reg = reg;
                rw_w.wait();
                std::thread::sleep(Duration::from_millis(25));
                wd_w.store(true, Ordering::SeqCst);
            })
            .expect("spawn succeeded");

        // Launch TWO concurrent `shutdown_and_wait` callers; neither may return
        // early before `worker_done == true` AND `pending_ticket` finishes registering.
        let both_entered = Arc::new(Barrier::new(3));
        let mut callers = Vec::new();
        for _ in 0..2 {
            let (c, wd, be) = (
                Arc::clone(&coord),
                Arc::clone(&worker_done),
                Arc::clone(&both_entered),
            );
            callers.push(std::thread::spawn(move || {
                let ok = c.shutdown_and_wait(Duration::from_millis(500), || {
                    be.wait();
                });
                assert!(ok);
                assert!(
                    wd.load(Ordering::SeqCst),
                    "shutdown_and_wait returned while worker was still running"
                );
            }));
        }

        // Wait until both shutdown callers have entered `shutdown_and_wait`,
        // verify post-shutdown spawns are rejected immediately, then complete
        // the pending ticket (simulating a worker finishing registration after
        // shutdown began) and let the first worker exit.
        both_entered.wait();
        assert!(
            coord.begin_spawn().is_none(),
            "post-shutdown worker spawns must be rejected"
        );
        let _second_handle = pending_ticket
            .spawn("late registered worker".into(), |reg| {
                let _reg = reg;
            })
            .expect("reserved ticket completes spawn cleanly");
        release_worker.wait();
        for caller in callers {
            caller.join().unwrap();
        }

        // Subtest: When Caller 1 is actively joining a slow OS thread (so
        // `handles` is already drained and `joins_in_progress > 0`), a
        // concurrent Caller 2 MUST wait for Caller 1's OS join instead of
        // observing empty `handles` and returning `true` prematurely.
        {
            let coord2 = Arc::new(WorkerCoordinator::new());
            let gate = Arc::new((Mutex::new(false), Condvar::new()));
            let gate_bg = Arc::clone(&gate);
            let t = coord2.begin_spawn().expect("spawn allowed");
            let shared = t
                .spawn("slow os exit".into(), move |reg| {
                    // Drop `reg` BEFORE sleeping to simulate slow OS/TLS
                    // teardown after `live_workers` already hit `0`.
                    drop(reg);
                    let (m, cv) = &*gate_bg;
                    let mut done = m.lock().unwrap();
                    while !*done {
                        done = cv.wait(done).unwrap();
                    }
                })
                .expect("spawn ok");

            // Wait until `live_workers == 0` so `shutdown_and_wait` enters the
            // OS-join phase (`handles` drained, `joins_in_progress == 1`).
            while coord2.state.lock().unwrap().live_workers > 0 {
                std::thread::yield_now();
            }

            let c1 = Arc::clone(&coord2);
            let caller1 =
                std::thread::spawn(move || c1.shutdown_and_wait(Duration::from_millis(500), || {}));
            while coord2.state.lock().unwrap().joins_in_progress == 0 {
                std::thread::yield_now();
            }

            // Caller 2 with short budget MUST return `false` (never premature `true`)
            // while Caller 1 is still waiting on the OS thread handle.
            assert!(
                !coord2.shutdown_and_wait(Duration::from_millis(20), || {}),
                "concurrent shutdown_and_wait must not return true while Caller 1 is still joining"
            );
            // Concurrent `SharedJoinHandle::join_bounded` waiter also returns `false` on timeout.
            assert!(
                !shared.join_bounded(Instant::now() + Duration::from_millis(15)),
                "SharedJoinHandle::join_bounded must not return true while another caller is still joining"
            );

            // Launch Caller 3 with generous budget, then let the OS thread finish.
            let c3 = Arc::clone(&coord2);
            let caller3 =
                std::thread::spawn(move || c3.shutdown_and_wait(Duration::from_millis(500), || {}));
            std::thread::sleep(Duration::from_millis(15));
            {
                let (m, cv) = &*gate;
                *m.lock().unwrap() = true;
                cv.notify_all();
            }
            assert!(caller1.join().unwrap());
            assert!(caller3.join().unwrap());
            assert!(shared.join_bounded(Instant::now() + Duration::from_millis(50)));
        }
    }

    #[test]
    fn serialized_write_guarantees_neutral_is_always_last_across_preemption_races() {
        use std::sync::Barrier;

        // Case 1: Worker acquires SLOT_WRITING, pauses BEFORE `WriteFile` while
        // emergency reset runs. Resumed worker MUST skip `WriteFile` so emergency
        // neutral remains the sole/final report.
        {
            let slot = Arc::new(EmergencySlot::new());
            assert!(slot.publish(0x1111, 0x2222, Transport::Usb, 48));
            let shutdown = Arc::new(AtomicBool::new(false));
            let log = Arc::new(Mutex::new(Vec::<&'static str>::new()));
            let pause = Arc::new(Barrier::new(2));
            let resume = Arc::new(Barrier::new(2));

            let (s_w, sh_w, log_w, p_w, r_w) = (
                Arc::clone(&slot),
                Arc::clone(&shutdown),
                Arc::clone(&log),
                Arc::clone(&pause),
                Arc::clone(&resume),
            );
            let worker = std::thread::spawn(move || {
                s_w.run_serialized_write(
                    &sh_w,
                    || {
                        p_w.wait();
                        r_w.wait();
                    },
                    || {
                        log_w.lock().unwrap().push("worker_active");
                        true
                    },
                    |_| {
                        log_w.lock().unwrap().push("worker_trailing_neutral");
                        NeutralSendOutcome::Delivered
                    },
                )
            });

            pause.wait();
            assert_eq!(
                slot.run_emergency_reset(
                    false,
                    Duration::from_millis(20),
                    Duration::from_millis(10),
                    |_| {},
                    |_, _, _| {
                        log.lock().unwrap().push("emergency_neutral");
                        NeutralSendOutcome::Delivered
                    },
                    |_, _, _| NeutralSendOutcome::Delivered,
                    || {},
                ),
                ResetOutcome::Unconfirmed
            );
            resume.wait();
            let res = worker.join().unwrap();
            assert!(!res.write_ok && res.preempted);
            assert_eq!(*log.lock().unwrap(), ["emergency_neutral"]);
        }

        // Case 2: Emergency sends neutral and pauses BEFORE clearing
        // `emergency_in_flight`; Worker completes an active write during that
        // pause; then Emergency resumes. Dekker handoff MUST cause Emergency to
        // emit a trailing neutral so the final delivered report is strictly neutral.
        {
            let slot = Arc::new(EmergencySlot::new());
            assert!(slot.publish(0x2222, 0x3333, Transport::Usb, 48));
            let shutdown = Arc::new(AtomicBool::new(false));
            let log = Arc::new(Mutex::new(Vec::<&'static str>::new()));
            let worker_in_write = Arc::new(Barrier::new(2));
            let em_sent_first_neutral = Arc::new(Barrier::new(2));
            let worker_finished = Arc::new(Barrier::new(2));

            let (s_w, sh_w, log_w, b1_w, b2_w, b3_w) = (
                Arc::clone(&slot),
                Arc::clone(&shutdown),
                Arc::clone(&log),
                Arc::clone(&worker_in_write),
                Arc::clone(&em_sent_first_neutral),
                Arc::clone(&worker_finished),
            );
            let worker = std::thread::spawn(move || {
                let res = s_w.run_serialized_write(
                    &sh_w,
                    || {},
                    || {
                        b1_w.wait();
                        b2_w.wait();
                        log_w.lock().unwrap().push("worker_active");
                        true
                    },
                    |_| {
                        log_w.lock().unwrap().push("worker_trailing_neutral");
                        NeutralSendOutcome::Delivered
                    },
                );
                b3_w.wait();
                res
            });

            worker_in_write.wait();
            shutdown.store(true, Ordering::SeqCst);
            assert_eq!(
                slot.run_emergency_reset(
                    true,
                    Duration::from_millis(20),
                    Duration::from_millis(10),
                    |_| {},
                    |_, _, _| {
                        log.lock().unwrap().push("emergency_neutral");
                        NeutralSendOutcome::Delivered
                    },
                    |_, _, _| NeutralSendOutcome::Delivered,
                    || {
                        em_sent_first_neutral.wait();
                        worker_finished.wait();
                    },
                ),
                ResetOutcome::Confirmed
            );
            let res = worker.join().unwrap();
            assert!(!res.write_ok && res.preempted);
            assert_eq!(
                *log.lock().unwrap(),
                ["emergency_neutral", "worker_active", "emergency_neutral"]
            );
        }

        // Case 3 (Inverse ordering, both non-fatal & fatal): Emergency finishes
        // BEFORE Worker finishes its active write (`SLOT_WRITING` held until
        // `after_em.wait()`). On non-fatal, emergency returns `Confirmed` and
        // Worker emits a trailing neutral via Dekker handoff. On fatal, emergency
        // MUST NOT claim `Confirmed` while `SLOT_WRITING != 0` — it bounds at
        // `fatal_budget` and returns `Unconfirmed`, and once Worker unblocks,
        // Worker still emits its trailing neutral via Dekker handoff!
        for fatal in [false, true] {
            let slot = Arc::new(EmergencySlot::new());
            assert!(slot.publish(0x3333, 0x4444, Transport::Usb, 48));
            let shutdown = Arc::new(AtomicBool::new(false));
            let log = Arc::new(Mutex::new(Vec::<&'static str>::new()));
            let in_write = Arc::new(Barrier::new(2));
            let after_em = Arc::new(Barrier::new(2));

            let (s_w, sh_w, log_w, iw_w, ae_w) = (
                Arc::clone(&slot),
                Arc::clone(&shutdown),
                Arc::clone(&log),
                Arc::clone(&in_write),
                Arc::clone(&after_em),
            );
            let worker = std::thread::spawn(move || {
                s_w.run_serialized_write(
                    &sh_w,
                    || {},
                    || {
                        iw_w.wait();
                        ae_w.wait();
                        log_w.lock().unwrap().push("worker_active");
                        true
                    },
                    |release_leds| {
                        log_w.lock().unwrap().push(if release_leds {
                            "worker_trailing_neutral_fatal"
                        } else {
                            "worker_trailing_neutral_nonfatal"
                        });
                        NeutralSendOutcome::Delivered
                    },
                )
            });

            in_write.wait();
            if fatal {
                shutdown.store(true, Ordering::SeqCst);
            }
            let outcome = slot.run_emergency_reset_with_budget(
                fatal,
                Duration::from_millis(20),
                Duration::from_millis(10),
                Duration::from_millis(35),
                |_| {},
                |_, _, _| {
                    log.lock().unwrap().push("emergency_neutral");
                    NeutralSendOutcome::Delivered
                },
                |_, _, _| NeutralSendOutcome::Delivered,
                || {},
            );
            assert_eq!(
                outcome,
                ResetOutcome::Unconfirmed,
                "emergency reset must NOT claim Confirmed while SLOT_WRITING remains held (neutral_completed_seq < my_req)"
            );
            after_em.wait();
            let res = worker.join().unwrap();
            assert!(!res.write_ok && res.preempted);
            assert_eq!(res.trailing_neutral, Some(NeutralSendOutcome::Delivered));
            let expected_last = if fatal {
                "worker_trailing_neutral_fatal"
            } else {
                "worker_trailing_neutral_nonfatal"
            };
            assert_eq!(
                *log.lock().unwrap(),
                ["emergency_neutral", "worker_active", expected_last]
            );
        }

        // Case 4: On fatal emergency reset with NO write in flight (only pending
        // ReadFile), `on_in_flight` (`CancelIoEx`) MUST still run BEFORE `send_neutral`;
        // on non-fatal reset with no write in flight, it does not cancel idle reads.
        {
            let slot = EmergencySlot::new();
            assert!(slot.publish(0x4444, 0x5555, Transport::Usb, 48));
            let events = std::cell::RefCell::new(Vec::<&'static str>::new());
            assert_eq!(
                slot.run_emergency_reset(
                    false,
                    Duration::from_millis(10),
                    Duration::from_millis(10),
                    |_| events.borrow_mut().push("cancel_io"),
                    |_, _, _| {
                        events.borrow_mut().push("send_neutral");
                        NeutralSendOutcome::Delivered
                    },
                    |_, _, _| {
                        events.borrow_mut().push("fallback_neutral");
                        NeutralSendOutcome::Delivered
                    },
                    || {},
                ),
                ResetOutcome::Confirmed
            );
            assert_eq!(*events.borrow(), ["send_neutral"]);

            events.borrow_mut().clear();
            assert_eq!(
                slot.run_emergency_reset(
                    true,
                    Duration::from_millis(10),
                    Duration::from_millis(10),
                    |_| events.borrow_mut().push("cancel_io"),
                    |_, _, _| {
                        events.borrow_mut().push("send_neutral");
                        NeutralSendOutcome::Delivered
                    },
                    |_, _, _| {
                        events.borrow_mut().push("fallback_neutral");
                        NeutralSendOutcome::Delivered
                    },
                    || {},
                ),
                ResetOutcome::Confirmed
            );
            assert_eq!(*events.borrow(), ["cancel_io", "send_neutral"]);
        }

        // Case 5: Even when a non-fatal reset holds `SLOT_EMERGENCY` past
        // `claim_wait` AND a closing worker has already set `SLOT_CLOSING`, a
        // fatal emergency reset MUST still borrow the live handle, call
        // `CancelIoEx`, AND transmit a replacement neutral packet strictly
        // after `CancelIoEx`, returning `Confirmed` without corrupting borrow
        // refcounts or allowing premature close.
        {
            let slot = Arc::new(EmergencySlot::new());
            assert!(slot.publish(0x6666, 0x7777, Transport::Usb, 48));
            let nonfatal_borrow = slot
                .try_claim_emergency(Duration::from_millis(10))
                .expect("non-fatal claim succeeds");

            let closed = Arc::new(AtomicBool::new(false));
            let (s_c, c_c) = (Arc::clone(&slot), Arc::clone(&closed));
            let closer = std::thread::spawn(move || {
                let must_leak = s_c.begin_close_and_wait_bounded(Duration::from_millis(500));
                assert!(!must_leak);
                c_c.store(true, Ordering::Release);
            });
            while (slot.state.load(Ordering::Acquire) & SLOT_CLOSING) == 0 {
                std::thread::yield_now();
            }

            let events = std::cell::RefCell::new(Vec::<(&'static str, usize)>::new());
            let outcome = slot.run_emergency_reset(
                true,
                Duration::from_millis(5),
                Duration::from_millis(5),
                |h| events.borrow_mut().push(("cancel_io", h)),
                |_, _, _| {
                    events.borrow_mut().push(("send_neutral", 0));
                    NeutralSendOutcome::Delivered
                },
                |cb, release_leds, _| {
                    assert!(release_leds, "fatal fallback neutral must release LEDs");
                    // While our fallback neutral is in flight (`PHASE_FALLBACK` set),
                    // neither primary reset / `cancel_all_open_handles` nor a second
                    // fatal reset may issue `CancelIoEx` on this slot!
                    assert!(
                        !slot.cancel_io_if_no_neutral_in_flight(
                            cb.handle,
                            |_| {
                                events.borrow_mut().push(("spurious_cancel", cb.handle));
                            },
                            || {},
                        ),
                        "CancelIoEx must be suppressed while a fallback neutral is in flight"
                    );
                    events.borrow_mut().push(("fallback_neutral", cb.handle));
                    NeutralSendOutcome::Delivered
                },
                || {},
            );
            assert_eq!(outcome, ResetOutcome::Confirmed);
            assert_eq!(
                *events.borrow(),
                [("cancel_io", 0x6666), ("fallback_neutral", 0x6666)],
                "fatal timeout fallback must issue CancelIoEx followed by fallback neutral even while SLOT_CLOSING + SLOT_EMERGENCY are active"
            );
            assert!(
                !closed.load(Ordering::Acquire),
                "handle must not close while non-fatal borrow is held"
            );

            drop(nonfatal_borrow);
            closer.join().unwrap();
            assert!(closed.load(Ordering::Acquire));
            assert_eq!(slot.handle.load(Ordering::Acquire), 0);
            assert!(
                slot.try_borrow_handle_for_cancel().is_none(),
                "closed slot must refuse cancel borrows"
            );
        }
    }

    #[test]
    fn emergency_write_classification_is_strictly_bounded_overlapped() {
        assert_eq!(
            classify_emergency_write(EmergencyWriteOutcome::Completed),
            NeutralSendOutcome::Delivered
        );
        assert_eq!(
            classify_emergency_write(EmergencyWriteOutcome::ImmediateError),
            NeutralSendOutcome::Failed
        );
        assert_eq!(
            classify_emergency_write(EmergencyWriteOutcome::TimedOutSettled),
            NeutralSendOutcome::Failed
        );
        assert_eq!(
            classify_emergency_write(EmergencyWriteOutcome::Hung),
            NeutralSendOutcome::Hung
        );
    }

    #[test]
    fn io_phase_mutual_exclusion_and_delayed_cancel_retry() {
        use std::sync::Barrier;

        let slot = Arc::new(EmergencySlot::new());
        assert!(slot.publish(0x8888, 0x9999, Transport::Usb, 48));

        // 1. Strict mutual exclusion: `try_acquire_primary_neutral()` refuses
        // while `PHASE_FALLBACK` is held, and succeeds once cleared.
        slot.io_phase.fetch_or(PHASE_FALLBACK, Ordering::AcqRel);
        assert!(
            slot.try_acquire_primary_neutral().is_none(),
            "primary neutral must not overlap an active fatal fallback neutral"
        );
        slot.io_phase.fetch_and(!PHASE_FALLBACK, Ordering::AcqRel);
        let guard = slot
            .try_acquire_primary_neutral()
            .expect("primary neutral succeeds once PHASE_FALLBACK clears");
        assert!(slot.try_acquire_primary_neutral().is_none());
        drop(guard);

        // 2. Normal uncontested neutral sends exactly once and advances `neutral_completed_seq`.
        let req1 = slot.request_neutral(false);
        let _g = slot.try_acquire_primary_neutral().unwrap();
        let mut sends = 0usize;
        assert_eq!(
            slot.send_neutral_with_cancel_retry(|_, _| {
                sends += 1;
                NeutralSendOutcome::Delivered
            }),
            NeutralSendOutcome::Delivered
        );
        assert_eq!(sends, 1);
        assert_eq!(slot.neutral_completed_seq.load(Ordering::SeqCst), req1);
        drop(_g);

        // 3a. Delayed pre-existing `CancelIoEx` outliving the 25 ms initial drain
        // and completing DURING neutral send MUST trigger a follow-up neutral
        // running strictly after `PHASE_CANCEL_MASK == 0`.
        let entered_cancel = Arc::new(Barrier::new(2));
        let release_cancel = Arc::new(Barrier::new(2));
        let (s_c, ec_c, rc_c) = (
            Arc::clone(&slot),
            Arc::clone(&entered_cancel),
            Arc::clone(&release_cancel),
        );
        let canceller = std::thread::spawn(move || {
            assert!(s_c.cancel_io_if_no_neutral_in_flight(
                0x8888,
                |_| {
                    ec_c.wait();
                    rc_c.wait();
                },
                || {},
            ));
        });
        entered_cancel.wait();

        let _req2 = slot.request_neutral(false);
        let _g = slot.try_acquire_primary_neutral().unwrap();
        let mut attempts = 0usize;
        let mut retry_saw_zero_cancels = false;
        let mut rc_opt = Some(release_cancel);
        assert_eq!(
            slot.send_neutral_with_cancel_retry(|_, _| {
                attempts += 1;
                if let Some(b) = rc_opt.take() {
                    // Pre-existing CancelIoEx completes mid-send, incrementing `cancel_completed_seq`
                    // and clearing `PHASE_CANCEL_MASK`.
                    b.wait();
                    while (slot.io_phase.load(Ordering::SeqCst) & PHASE_CANCEL_MASK) != 0 {
                        std::thread::yield_now();
                    }
                } else {
                    retry_saw_zero_cancels =
                        (slot.io_phase.load(Ordering::SeqCst) & PHASE_CANCEL_MASK) == 0;
                }
                NeutralSendOutcome::Delivered
            }),
            NeutralSendOutcome::Delivered
        );
        assert_eq!(
            attempts, 2,
            "overlapping delayed CancelIoEx must trigger exactly one uncancelled follow-up neutral"
        );
        assert!(
            retry_saw_zero_cancels,
            "retry neutral must run strictly after PHASE_CANCEL_MASK == 0"
        );
        drop(_g);
        canceller.join().unwrap();

        // 3b. Delayed pre-existing `CancelIoGuard` preempted BEFORE `CancelIoEx`
        // past the 25 ms drain timeout AND past the initial neutral send:
        // neutral MUST NOT burn a retry while `PHASE_CANCEL_MASK != 0`, MUST NOT
        // advance `neutral_completed_seq` yet, and MUST hand off via `neutral_after_cancel`
        // so `CancelIoGuard::drop` fires the trailing neutral strictly AFTER `CancelIoEx`
        // completes and `PHASE_CANCEL_MASK == 0`.
        {
            let entered_preempt = Arc::new(Barrier::new(2));
            let release_preempt = Arc::new(Barrier::new(2));
            let trace = Arc::new(Mutex::new(Vec::<&'static str>::new()));
            let (s_p, ep_p, rp_p, tr_p) = (
                Arc::clone(&slot),
                Arc::clone(&entered_preempt),
                Arc::clone(&release_preempt),
                Arc::clone(&trace),
            );
            let preempted_canceller = std::thread::spawn(move || {
                let tr_drop = Arc::clone(&tr_p);
                let s_drop = Arc::clone(&s_p);
                assert!(s_p.cancel_io_if_no_neutral_in_flight(
                    0x8888,
                    |_| {
                        ep_p.wait();
                        rp_p.wait();
                        tr_p.lock().unwrap().push("delayed_cancel_io");
                    },
                    move || {
                        assert_eq!(
                            s_drop.io_phase.load(Ordering::SeqCst) & PHASE_CANCEL_MASK,
                            0,
                            "post-cancel neutral must run strictly after PHASE_CANCEL_MASK == 0"
                        );
                        tr_drop.lock().unwrap().push("post_cancel_neutral");
                    },
                ));
            });
            entered_preempt.wait();

            let completed_before = slot.neutral_completed_seq.load(Ordering::SeqCst);
            let _req3 = slot.request_neutral(true);
            let _g = slot.try_acquire_primary_neutral().unwrap();
            let mut initial_attempts = 0usize;
            assert_eq!(
                slot.send_neutral_with_cancel_retry(|_, _| {
                    initial_attempts += 1;
                    trace.lock().unwrap().push("initial_neutral");
                    NeutralSendOutcome::Delivered
                }),
                NeutralSendOutcome::Failed,
                "overlapped send handed off to CancelIoGuard must not report confirmed delivery yet"
            );
            assert_eq!(
                slot.neutral_completed_seq.load(Ordering::SeqCst),
                completed_before,
                "neutral_completed_seq must NOT advance when handed off to delayed CancelIoGuard"
            );
            assert_eq!(
                initial_attempts, 1,
                "must NOT burn a retry while PHASE_CANCEL_MASK != 0"
            );
            assert!(slot.neutral_after_cancel.load(Ordering::SeqCst));
            drop(_g);

            release_preempt.wait();
            preempted_canceller.join().unwrap();
            assert!(!slot.neutral_after_cancel.load(Ordering::SeqCst));
            assert_eq!(
                *trace.lock().unwrap(),
                [
                    "initial_neutral",
                    "delayed_cancel_io",
                    "post_cancel_neutral"
                ]
            );
        }

        // 4. Pending monotonic ticket is NEVER lost when `PHASE_FALLBACK` is
        // held during a non-fatal primary `run_emergency_reset` attempt, and
        // fallback honors any new ticket requested during fallback neutral.
        let _pending_req = slot.request_neutral(false);
        slot.io_phase.fetch_or(PHASE_FALLBACK, Ordering::SeqCst);
        let mut primary_sent = false;
        assert_eq!(
            slot.run_emergency_reset(
                false,
                Duration::from_millis(10),
                Duration::from_millis(10),
                |_| {},
                |_, _, _| {
                    primary_sent = true;
                    NeutralSendOutcome::Delivered
                },
                |_, _, _| NeutralSendOutcome::Delivered,
                || {},
            ),
            ResetOutcome::Unconfirmed
        );
        assert!(!primary_sent);
        assert!(
            slot.has_pending_neutral(),
            "primary bailout on PHASE_FALLBACK must preserve pending monotonic tickets"
        );
        slot.io_phase.fetch_and(!PHASE_FALLBACK, Ordering::SeqCst);

        // 5. Primary Dekker recheck after clearing `emergency_in_flight`: if
        // `PHASE_FALLBACK` clears right as primary gives up on `try_acquire_primary_neutral`,
        // either primary re-arms and sends neutral OR pending ticket remains preserved.
        {
            slot.io_phase.fetch_or(PHASE_FALLBACK, Ordering::SeqCst);
            let s_bg = Arc::clone(&slot);
            let releaser = std::thread::spawn(move || {
                std::thread::sleep(Duration::from_millis(6));
                let _ = s_bg.request_neutral(false);
                s_bg.io_phase.fetch_and(!PHASE_FALLBACK, Ordering::SeqCst);
            });
            let mut sends = 0usize;
            let _ = slot.run_emergency_reset(
                false,
                Duration::from_millis(10),
                Duration::from_millis(10),
                |_| {},
                |_, _, _| {
                    sends += 1;
                    NeutralSendOutcome::Delivered
                },
                |_, _, _| NeutralSendOutcome::Delivered,
                || {},
            );
            releaser.join().unwrap();
            assert!(
                sends >= 1 || slot.has_pending_neutral(),
                "trailing neutral ticket must either be sent or preserved for Dekker handoff"
            );
        }

        // Now verify fallback honors a new ticket requested during fallback send.
        let held_borrow = slot
            .try_claim_emergency(Duration::from_millis(10))
            .expect("hold SLOT_EMERGENCY");
        let fallback_sends = std::cell::Cell::new(0usize);
        assert_eq!(
            slot.run_emergency_reset(
                true,
                Duration::from_millis(5),
                Duration::from_millis(5),
                |_| {},
                |_, _, _| NeutralSendOutcome::Delivered,
                |_, _, _| {
                    let n = fallback_sends.get() + 1;
                    fallback_sends.set(n);
                    if n == 1 {
                        // Simulate worker requesting a trailing neutral during fallback write.
                        let _ = slot.request_neutral(true);
                    }
                    NeutralSendOutcome::Delivered
                },
                || {},
            ),
            ResetOutcome::Confirmed
        );
        assert_eq!(
            fallback_sends.get(),
            2,
            "fallback neutral must honor new ticket requested during fallback write"
        );
        assert!(!slot.has_pending_neutral());
        drop(held_borrow);
    }

    #[test]
    fn post_cancel_dekker_handoff_delivers_neutral_when_owner_exits_before_step_1() {
        use std::sync::Barrier;

        // 1. Exact interleaving: active owner holds `PHASE_FALLBACK` when
        // `deliver_or_handoff_neutral_with_hook` tries immediate CAS, then
        // releases `PHASE_FALLBACK` and finishes its post-release check RIGHT
        // BEFORE Dekker Step 1 (`request_neutral`). Dekker Step 2 MUST observe
        // idle, loop back, claim the neutral phase, and deliver the neutral!
        {
            let slot = Arc::new(EmergencySlot::new());
            assert!(slot.publish(0x9001, 0x9002, Transport::Usb, 48));
            let owner_locked = Arc::new(Barrier::new(2));
            let release_owner = Arc::new(Barrier::new(2));
            let owner_exited = Arc::new(Barrier::new(2));

            let (s_o, ol_o, ro_o, oe_o) = (
                Arc::clone(&slot),
                Arc::clone(&owner_locked),
                Arc::clone(&release_owner),
                Arc::clone(&owner_exited),
            );
            let owner = std::thread::spawn(move || {
                let guard = s_o
                    .try_acquire_fallback_neutral()
                    .expect("owner acquires PHASE_FALLBACK");
                ol_o.wait();
                ro_o.wait();
                drop(guard);
                // Simulate owner's post-release Dekker recheck completing
                // BEFORE post-cancel's Dekker Step 1 ticket publication!
                assert!(!s_o.has_pending_neutral());
                oe_o.wait();
            });

            owner_locked.wait();
            let mut sends = 0usize;
            let mut hook_runs = 0usize;
            let mut ro_opt = Some(release_owner);
            let mut oe_opt = Some(owner_exited);
            assert_eq!(
                slot.deliver_or_handoff_neutral_with_hook(
                    PHASE_PRIMARY_NEUTRAL,
                    true,
                    None,
                    || {
                        hook_runs += 1;
                        if let (Some(ro), Some(oe)) = (ro_opt.take(), oe_opt.take()) {
                            ro.wait();
                            oe.wait();
                        }
                    },
                    |release_leds, _| {
                        assert!(release_leds);
                        sends += 1;
                        NeutralSendOutcome::Delivered
                    },
                ),
                NeutralSendOutcome::Delivered
            );
            owner.join().unwrap();
            assert_eq!(hook_runs, 1);
            assert_eq!(
                sends, 1,
                "Dekker Step 2 must observe idle after owner exited before Step 1 and deliver neutral"
            );
            assert!(!slot.has_pending_neutral());
            assert_eq!(slot.io_phase.load(Ordering::SeqCst), 0);
            assert_eq!(slot.neutral_completed_seq.load(Ordering::SeqCst), 1);
            assert_eq!(slot.release_leds_completed_seq.load(Ordering::SeqCst), 1);
        }

        // 2. Inverse interleaving: active fatal fallback STILL holds `PHASE_FALLBACK`
        // across Dekker Step 1 & Step 2 of `deliver_or_handoff_neutral` ->
        // post-cancel hands off cleanly (0 direct sends), and fallback's
        // loop / post-release Dekker check delivers the trailing neutral!
        {
            let slot = Arc::new(EmergencySlot::new());
            assert!(slot.publish(0x9003, 0x9004, Transport::Usb, 48));
            let held_borrow = slot
                .try_claim_emergency(Duration::from_millis(10))
                .expect("hold SLOT_EMERGENCY so fatal reset takes fallback path");

            let in_fallback = Arc::new(Barrier::new(2));
            let release_fallback = Arc::new(Barrier::new(2));
            let fallback_sends = Arc::new(AtomicU32::new(0));

            let (s_f, if_f, rf_f, fs_f) = (
                Arc::clone(&slot),
                Arc::clone(&in_fallback),
                Arc::clone(&release_fallback),
                Arc::clone(&fallback_sends),
            );
            let fallback_thread = std::thread::spawn(move || {
                let mut if_opt = Some(if_f);
                let mut rf_opt = Some(rf_f);
                assert_eq!(
                    s_f.run_emergency_reset(
                        true,
                        Duration::from_millis(5),
                        Duration::from_millis(5),
                        |_| {},
                        |_, _, _| NeutralSendOutcome::Delivered,
                        |_, release_leds, _| {
                            assert!(release_leds);
                            let n = fs_f.fetch_add(1, Ordering::SeqCst) + 1;
                            if n == 1
                                && let (Some(b_in), Some(b_rel)) = (if_opt.take(), rf_opt.take())
                            {
                                b_in.wait();
                                b_rel.wait();
                            }
                            NeutralSendOutcome::Delivered
                        },
                        || {},
                    ),
                    ResetOutcome::Confirmed
                );
            });

            in_fallback.wait();
            let mut post_cancel_sends = 0usize;
            assert_eq!(
                slot.deliver_or_handoff_neutral(PHASE_PRIMARY_NEUTRAL, true, |_, _| {
                    post_cancel_sends += 1;
                    NeutralSendOutcome::Delivered
                }),
                NeutralSendOutcome::Failed
            );
            assert_eq!(
                post_cancel_sends, 0,
                "post-cancel helper must hand off while PHASE_FALLBACK is active"
            );
            assert!(slot.has_pending_neutral());

            release_fallback.wait();
            fallback_thread.join().unwrap();
            assert_eq!(
                fallback_sends.load(Ordering::SeqCst),
                2,
                "active fallback owner must observe new ticket and emit trailing neutral"
            );
            assert!(!slot.has_pending_neutral());
            drop(held_borrow);
        }
    }

    #[test]
    fn worker_shutdown_and_trailing_neutral_coordinate_with_delayed_cancel_before_handle_close() {
        use std::sync::Barrier;

        // Case A: Worker runs `run_worker_shutdown_neutral` (teardown path) while
        // a delayed `CancelIoGuard` from `cancel_all_open_handles` holds an
        // `EmergencyBorrow` and `PHASE_CANCEL_MASK == 1`. Worker hands off via
        // `neutral_after_cancel`, then enters `begin_close_and_wait_bounded`
        // (handle MUST stay valid and NOT close yet!). Delayed `CancelIoEx`
        // runs, drops `CancelIoGuard`, delivers the uncancelled post-cancel
        // neutral strictly after `PHASE_CANCEL_MASK == 0` while handle != 0,
        // and drops `EmergencyBorrow` -> ONLY THEN does the handle close!
        {
            let slot = Arc::new(EmergencySlot::new());
            assert!(slot.publish(0xA001, 0xA002, Transport::Usb, 48));
            let trace = Arc::new(Mutex::new(Vec::<&'static str>::new()));
            let cancel_entered = Arc::new(Barrier::new(2));
            let cancel_release = Arc::new(Barrier::new(2));
            let handle_closed = Arc::new(AtomicBool::new(false));

            let (s_c, tr_c, ce_c, cr_c) = (
                Arc::clone(&slot),
                Arc::clone(&trace),
                Arc::clone(&cancel_entered),
                Arc::clone(&cancel_release),
            );
            let cancel_thread = std::thread::spawn(move || {
                let borrow = s_c
                    .try_claim_emergency(Duration::from_millis(20))
                    .expect("cancel_all_open_handles claims EmergencyBorrow");
                let tr_post = Arc::clone(&tr_c);
                let s_post = Arc::clone(&s_c);
                assert!(s_c.cancel_io_if_no_neutral_in_flight(
                    borrow.handle,
                    |_| {
                        ce_c.wait();
                        cr_c.wait();
                        tr_c.lock().unwrap().push("delayed_cancel_io");
                    },
                    move || {
                        let outcome = s_post.deliver_or_handoff_neutral(
                            PHASE_PRIMARY_NEUTRAL,
                            true,
                            |release_leds, _| {
                                assert!(release_leds);
                                assert_eq!(
                                    s_post.io_phase.load(Ordering::SeqCst) & PHASE_CANCEL_MASK,
                                    0,
                                    "post-cancel neutral must run strictly after PHASE_CANCEL_MASK == 0"
                                );
                                assert_ne!(
                                    s_post.handle.load(Ordering::Acquire),
                                    0,
                                    "handle must stay open while post-cancel neutral runs"
                                );
                                tr_post.lock().unwrap().push("post_cancel_neutral");
                                NeutralSendOutcome::Delivered
                            },
                        );
                        assert_eq!(outcome, NeutralSendOutcome::Delivered);
                    },
                ));
                drop(borrow);
            });

            cancel_entered.wait();

            let (s_w, tr_w, hc_w) = (
                Arc::clone(&slot),
                Arc::clone(&trace),
                Arc::clone(&handle_closed),
            );
            let worker = std::thread::spawn(move || {
                let tr_close = Arc::clone(&tr_w);
                let _close_guard = SlotCloseGuard {
                    slot: &s_w,
                    budget: Duration::from_millis(500),
                    on_close: Some(move |must_leak: bool| {
                        assert!(!must_leak, "clean cancel handoff must not leak handle");
                        tr_close.lock().unwrap().push("handle_closed");
                        hc_w.store(true, Ordering::Release);
                    }),
                };
                assert_eq!(
                    s_w.run_worker_shutdown_neutral(|release_leds, _| {
                        assert!(release_leds);
                        tr_w.lock().unwrap().push("worker_shutdown_initial");
                        NeutralSendOutcome::Delivered
                    }),
                    NeutralSendOutcome::Failed
                );
                assert!(s_w.neutral_after_cancel.load(Ordering::SeqCst));
                // `_close_guard` drops here, blocking in `begin_close_and_wait_bounded`.
            });

            while (slot.state.load(Ordering::Acquire) & SLOT_CLOSING) == 0 {
                std::thread::yield_now();
            }
            assert!(
                !handle_closed.load(Ordering::Acquire),
                "handle must not close while delayed CancelIoGuard + EmergencyBorrow are active"
            );
            assert_eq!(
                slot.neutral_completed_seq.load(Ordering::SeqCst),
                0,
                "overlapped initial shutdown neutral must not advance neutral_completed_seq"
            );

            cancel_release.wait();
            cancel_thread.join().unwrap();
            worker.join().unwrap();

            assert!(handle_closed.load(Ordering::Acquire));
            assert_eq!(slot.handle.load(Ordering::Acquire), 0);
            assert!(slot.neutral_completed_seq.load(Ordering::SeqCst) >= 1);
            assert!(slot.release_leds_completed_seq.load(Ordering::SeqCst) >= 1);
            assert_eq!(
                *trace.lock().unwrap(),
                [
                    "worker_shutdown_initial",
                    "delayed_cancel_io",
                    "post_cancel_neutral",
                    "handle_closed",
                ]
            );
        }

        // Case B: Worker's preempted-write trailing neutral (`run_serialized_write`)
        // overlaps a delayed `CancelIoGuard` from `cancel_all_open_handles` before
        // `SlotCloseGuard` closes the handle -> `neutral_after_cancel` delivers
        // the uncancelled post-cancel neutral strictly after `PHASE_CANCEL_MASK == 0`
        // BEFORE the handle closes!
        {
            let slot = Arc::new(EmergencySlot::new());
            assert!(slot.publish(0xA003, 0xA004, Transport::Usb, 48));
            let shutdown = Arc::new(AtomicBool::new(false));
            let trace = Arc::new(Mutex::new(Vec::<&'static str>::new()));
            let worker_in_write = Arc::new(Barrier::new(2));
            let cancel_entered = Arc::new(Barrier::new(2));
            let cancel_release = Arc::new(Barrier::new(2));
            let handle_closed = Arc::new(AtomicBool::new(false));

            let (s_w, sh_w, tr_w, wi_w, ce_w, hc_w) = (
                Arc::clone(&slot),
                Arc::clone(&shutdown),
                Arc::clone(&trace),
                Arc::clone(&worker_in_write),
                Arc::clone(&cancel_entered),
                Arc::clone(&handle_closed),
            );
            let worker = std::thread::spawn(move || {
                let tr_close = Arc::clone(&tr_w);
                let _close_guard = SlotCloseGuard {
                    slot: &s_w,
                    budget: Duration::from_millis(500),
                    on_close: Some(move |must_leak: bool| {
                        assert!(!must_leak);
                        tr_close.lock().unwrap().push("handle_closed");
                        hc_w.store(true, Ordering::Release);
                    }),
                };
                let tr_write = Arc::clone(&tr_w);
                let tr_neutral = Arc::clone(&tr_w);
                let res = s_w.run_serialized_write(
                    &sh_w,
                    || {},
                    || {
                        wi_w.wait();
                        ce_w.wait();
                        tr_write.lock().unwrap().push("worker_active");
                        true
                    },
                    |release_leds| {
                        assert!(release_leds);
                        tr_neutral.lock().unwrap().push("worker_trailing_initial");
                        NeutralSendOutcome::Delivered
                    },
                );
                assert!(!res.write_ok && res.preempted);
                assert_eq!(res.trailing_neutral, Some(NeutralSendOutcome::Failed));
                assert!(s_w.neutral_after_cancel.load(Ordering::SeqCst));
            });

            worker_in_write.wait();
            shutdown.store(true, Ordering::SeqCst);

            let (s_c, tr_c, ce_c, cr_c) = (
                Arc::clone(&slot),
                Arc::clone(&trace),
                Arc::clone(&cancel_entered),
                Arc::clone(&cancel_release),
            );
            let cancel_thread = std::thread::spawn(move || {
                let borrow = s_c
                    .try_claim_emergency(Duration::from_millis(20))
                    .expect("cancel_all_open_handles claims EmergencyBorrow");
                let tr_post = Arc::clone(&tr_c);
                let s_post = Arc::clone(&s_c);
                assert!(s_c.cancel_io_if_no_neutral_in_flight(
                    borrow.handle,
                    |_| {
                        ce_c.wait();
                        cr_c.wait();
                        tr_c.lock().unwrap().push("delayed_cancel_io");
                    },
                    move || {
                        let outcome = s_post.deliver_or_handoff_neutral(
                            PHASE_PRIMARY_NEUTRAL,
                            true,
                            |release_leds, _| {
                                assert!(release_leds);
                                assert_eq!(
                                    s_post.io_phase.load(Ordering::SeqCst) & PHASE_CANCEL_MASK,
                                    0
                                );
                                assert_ne!(s_post.handle.load(Ordering::Acquire), 0);
                                tr_post.lock().unwrap().push("post_cancel_neutral");
                                NeutralSendOutcome::Delivered
                            },
                        );
                        assert_eq!(outcome, NeutralSendOutcome::Delivered);
                    },
                ));
                drop(borrow);
            });

            while (slot.state.load(Ordering::Acquire) & SLOT_CLOSING) == 0 {
                std::thread::yield_now();
            }
            assert!(
                !handle_closed.load(Ordering::Acquire),
                "handle must stay open until delayed cancel + post-cancel neutral complete"
            );

            cancel_release.wait();
            cancel_thread.join().unwrap();
            worker.join().unwrap();

            assert!(handle_closed.load(Ordering::Acquire));
            assert_eq!(slot.handle.load(Ordering::Acquire), 0);
            assert!(slot.neutral_completed_seq.load(Ordering::SeqCst) >= 1);
            assert_eq!(
                *trace.lock().unwrap(),
                [
                    "worker_active",
                    "worker_trailing_initial",
                    "delayed_cancel_io",
                    "post_cancel_neutral",
                    "handle_closed",
                ]
            );
        }
    }

    #[test]
    fn fatal_reset_waits_for_confirmed_delivery_and_bounds_stalled_owners() {
        use std::sync::Barrier;

        // Case 1: Fatal reset while non-fatal primary holds `SLOT_EMERGENCY + PHASE_PRIMARY_NEUTRAL`
        // MUST NOT return early on handoff; it waits until actual neutral delivery
        // with `release_leds == true` and `target >= my_req` completes and only then
        // returns `ResetOutcome::Confirmed`.
        {
            let slot = Arc::new(EmergencySlot::new());
            assert!(slot.publish(0xB001, 0xB002, Transport::Usb, 48));
            let in_nonfatal = Arc::new(Barrier::new(2));
            let release_nonfatal = Arc::new(Barrier::new(2));
            let nonfatal_sends = Arc::new(AtomicU32::new(0));
            let saw_led_upgrade = Arc::new(AtomicBool::new(false));
            let fatal_done = Arc::new(AtomicBool::new(false));

            let (s_nf, inf_nf, rel_nf, cnt_nf, led_nf) = (
                Arc::clone(&slot),
                Arc::clone(&in_nonfatal),
                Arc::clone(&release_nonfatal),
                Arc::clone(&nonfatal_sends),
                Arc::clone(&saw_led_upgrade),
            );
            let nonfatal = std::thread::spawn(move || {
                let mut inf_opt = Some(inf_nf);
                let mut rel_opt = Some(rel_nf);
                assert_eq!(
                    s_nf.run_emergency_reset(
                        false,
                        Duration::from_millis(10),
                        Duration::from_millis(10),
                        |_| {},
                        |_, release_leds, _| {
                            let n = cnt_nf.fetch_add(1, Ordering::SeqCst) + 1;
                            if n == 1
                                && let (Some(b_in), Some(b_rel)) = (inf_opt.take(), rel_opt.take())
                            {
                                b_in.wait();
                                b_rel.wait();
                            }
                            if release_leds {
                                led_nf.store(true, Ordering::SeqCst);
                            }
                            NeutralSendOutcome::Delivered
                        },
                        |_, _, _| NeutralSendOutcome::Delivered,
                        || {},
                    ),
                    ResetOutcome::Confirmed
                );
            });

            in_nonfatal.wait();
            let (s_f, fd_f, led_f) = (
                Arc::clone(&slot),
                Arc::clone(&fatal_done),
                Arc::clone(&saw_led_upgrade),
            );
            let fatal = std::thread::spawn(move || {
                let res = s_f.run_emergency_reset_with_budget(
                    true,
                    Duration::from_millis(5),
                    Duration::from_millis(5),
                    Duration::from_millis(250),
                    |_| {},
                    |_, _, _| NeutralSendOutcome::Delivered,
                    |_, release_leds, _| {
                        if release_leds {
                            led_f.store(true, Ordering::SeqCst);
                        }
                        NeutralSendOutcome::Delivered
                    },
                    || {},
                );
                fd_f.store(true, Ordering::SeqCst);
                res
            });

            // Wait past claim_wait (5 ms) while non-fatal primary still holds
            // `PHASE_PRIMARY_NEUTRAL`: fatal MUST NOT return early on handoff!
            std::thread::sleep(Duration::from_millis(25));
            assert!(
                !fatal_done.load(Ordering::SeqCst),
                "fatal reset must wait for actual delivery instead of returning early on handoff"
            );

            release_nonfatal.wait();
            nonfatal.join().unwrap();
            assert_eq!(fatal.join().unwrap(), ResetOutcome::Confirmed);
            assert!(saw_led_upgrade.load(Ordering::SeqCst));
            assert!(!slot.has_pending_neutral());
        }

        // Case 2: Fatal reset while non-fatal primary is permanently stalled
        // holding `SLOT_EMERGENCY + PHASE_PRIMARY_NEUTRAL` hits the bounded
        // `fatal_budget` promptly and explicitly returns `ResetOutcome::Unconfirmed`.
        {
            let slot = Arc::new(EmergencySlot::new());
            assert!(slot.publish(0xB003, 0xB004, Transport::Usb, 48));
            let held_borrow = slot
                .try_claim_emergency(Duration::from_millis(10))
                .expect("hold SLOT_EMERGENCY");
            let held_phase = slot
                .try_acquire_primary_neutral()
                .expect("hold PHASE_PRIMARY_NEUTRAL");

            let start = Instant::now();
            let outcome = slot.run_emergency_reset_with_budget(
                true,
                Duration::from_millis(5),
                Duration::from_millis(5),
                Duration::from_millis(40),
                |_| {},
                |_, _, _| NeutralSendOutcome::Delivered,
                |_, _, _| NeutralSendOutcome::Delivered,
                || {},
            );
            assert_eq!(
                outcome,
                ResetOutcome::Unconfirmed,
                "stalled non-fatal primary past fatal deadline must return Unconfirmed"
            );
            assert!(
                start.elapsed() < Duration::from_millis(200),
                "fatal reset must remain strictly bounded on stalled owner"
            );
            assert_eq!(slot.release_leds_completed_seq.load(Ordering::SeqCst), 0);

            drop(held_phase);
            drop(held_borrow);
        }

        // Case 3a: Fatal reset while a worker active write holds `SLOT_WRITING`
        // and settles within `fatal_budget`: waits until the active write settles
        // and delivers a confirming neutral strictly AFTER the active write.
        {
            let slot = Arc::new(EmergencySlot::new());
            assert!(slot.publish(0xB005, 0xB006, Transport::Usb, 48));
            let shutdown = Arc::new(AtomicBool::new(false));
            let trace = Arc::new(Mutex::new(Vec::<&'static str>::new()));
            let in_write = Arc::new(Barrier::new(2));

            let (s_w, sh_w, tr_w, iw_w) = (
                Arc::clone(&slot),
                Arc::clone(&shutdown),
                Arc::clone(&trace),
                Arc::clone(&in_write),
            );
            let worker = std::thread::spawn(move || {
                let tr_n = Arc::clone(&tr_w);
                s_w.run_serialized_write(
                    &sh_w,
                    || {},
                    || {
                        iw_w.wait();
                        // Hold `SLOT_WRITING` past `write_wait` (5 ms) but well within `fatal_budget` (250 ms).
                        std::thread::sleep(Duration::from_millis(25));
                        tr_w.lock().unwrap().push("worker_active");
                        true
                    },
                    |_| {
                        tr_n.lock().unwrap().push("worker_trailing_neutral");
                        NeutralSendOutcome::Delivered
                    },
                )
            });

            in_write.wait();
            shutdown.store(true, Ordering::SeqCst);
            let outcome = slot.run_emergency_reset_with_budget(
                true,
                Duration::from_millis(5),
                Duration::from_millis(5),
                Duration::from_millis(250),
                |_| {},
                |_, release_leds, _| {
                    assert!(release_leds);
                    trace.lock().unwrap().push("emergency_neutral");
                    NeutralSendOutcome::Delivered
                },
                |_, _, _| NeutralSendOutcome::Delivered,
                || {},
            );
            assert_eq!(outcome, ResetOutcome::Confirmed);
            let res = worker.join().unwrap();
            assert!(!res.write_ok && res.preempted);
            let events = trace.lock().unwrap();
            assert_eq!(
                events.last().copied(),
                Some("emergency_neutral"),
                "confirming fatal neutral must be strictly after worker_active: {events:?}"
            );
            assert!(events.contains(&"worker_active"));
        }

        // Case 3b: Fatal reset while a worker write never settles within
        // `fatal_budget` returns `ResetOutcome::Unconfirmed` promptly.
        {
            let slot = Arc::new(EmergencySlot::new());
            assert!(slot.publish(0xB007, 0xB008, Transport::Usb, 48));
            let shutdown = AtomicBool::new(false);
            let _epoch = slot
                .try_acquire_worker_write(&shutdown)
                .expect("worker holds SLOT_WRITING");

            let start = Instant::now();
            let outcome = slot.run_emergency_reset_with_budget(
                true,
                Duration::from_millis(5),
                Duration::from_millis(5),
                Duration::from_millis(40),
                |_| {},
                |_, _, _| NeutralSendOutcome::Delivered,
                |_, _, _| NeutralSendOutcome::Delivered,
                || {},
            );
            assert_eq!(
                outcome,
                ResetOutcome::Unconfirmed,
                "unsettled SLOT_WRITING past fatal_budget must return Unconfirmed"
            );
            assert!(start.elapsed() < Duration::from_millis(200));
            slot.release_writing();
        }
    }

    #[test]
    fn neutral_send_outcome_retries_settled_failure_and_reports_unconfirmed_exhaustion() {
        use std::sync::Barrier;

        // 1. Exact interleaving: emergency neutral finishes before worker
        // completes active write -> worker's trailing neutral runs -> first
        // attempt returns settled non-hung `Failed` -> bounded retry runs and
        // delivers `Delivered`, advancing `neutral_completed_seq`!
        {
            let slot = Arc::new(EmergencySlot::new());
            assert!(slot.publish(0xC001, 0xC002, Transport::Usb, 48));
            let shutdown = Arc::new(AtomicBool::new(false));
            let in_write = Arc::new(Barrier::new(2));
            let after_em = Arc::new(Barrier::new(2));
            let trailing_attempts = Arc::new(AtomicU32::new(0));

            let (s_w, sh_w, iw_w, ae_w, att_w) = (
                Arc::clone(&slot),
                Arc::clone(&shutdown),
                Arc::clone(&in_write),
                Arc::clone(&after_em),
                Arc::clone(&trailing_attempts),
            );
            let worker = std::thread::spawn(move || {
                s_w.run_serialized_write(
                    &sh_w,
                    || {},
                    || {
                        iw_w.wait();
                        ae_w.wait();
                        true
                    },
                    |_| {
                        let attempt = att_w.fetch_add(1, Ordering::SeqCst) + 1;
                        if attempt == 1 {
                            NeutralSendOutcome::Failed
                        } else {
                            NeutralSendOutcome::Delivered
                        }
                    },
                )
            });

            in_write.wait();
            assert_eq!(
                slot.run_emergency_reset(
                    false,
                    Duration::from_millis(20),
                    Duration::from_millis(10),
                    |_| {},
                    |_, _, _| NeutralSendOutcome::Delivered,
                    |_, _, _| NeutralSendOutcome::Delivered,
                    || {},
                ),
                ResetOutcome::Unconfirmed
            );
            let seq_after_em = slot.neutral_completed_seq.load(Ordering::SeqCst);
            assert_eq!(
                seq_after_em, 0,
                "neutral delivered while worker still held SLOT_WRITING must not advance neutral_completed_seq"
            );

            after_em.wait();
            let res = worker.join().unwrap();
            assert!(!res.write_ok && res.preempted);
            assert_eq!(res.trailing_neutral, Some(NeutralSendOutcome::Delivered));
            assert_eq!(
                trailing_attempts.load(Ordering::SeqCst),
                2,
                "settled Failed trailing neutral must retry boundedly and succeed on attempt 2"
            );
            assert_eq!(
                slot.neutral_completed_seq.load(Ordering::SeqCst),
                slot.neutral_requested_seq.load(Ordering::SeqCst)
            );
        }

        // 2. All retries return settled non-hung `Failed` -> helper exhausts
        // 3 attempts (initial + 2 retries), returns `Failed` (Unconfirmed), and
        // does NOT advance `neutral_completed_seq`.
        {
            let slot = EmergencySlot::new();
            assert!(slot.publish(0xC003, 0xC004, Transport::Usb, 48));
            let mut attempts = 0usize;
            let outcome = slot.deliver_or_handoff_neutral(PHASE_PRIMARY_NEUTRAL, true, |_, _| {
                attempts += 1;
                NeutralSendOutcome::Failed
            });
            assert_eq!(outcome, NeutralSendOutcome::Failed);
            assert_eq!(attempts, 3, "initial attempt + 2 bounded retries");
            assert_eq!(
                slot.neutral_completed_seq.load(Ordering::SeqCst),
                0,
                "exhausted Failed neutral must never advance neutral_completed_seq"
            );
            assert_eq!(slot.release_leds_completed_seq.load(Ordering::SeqCst), 0);
            assert!(!slot.is_hung());
        }

        // 3. `Hung` immediately aborts with 0 retries, marks `SLOT_HUNG`, and
        // never advances `neutral_completed_seq`.
        {
            let slot = EmergencySlot::new();
            assert!(slot.publish(0xC005, 0xC006, Transport::Usb, 48));
            let mut attempts = 0usize;
            let outcome = slot.deliver_or_handoff_neutral(PHASE_PRIMARY_NEUTRAL, true, |_, _| {
                attempts += 1;
                NeutralSendOutcome::Hung
            });
            assert_eq!(outcome, NeutralSendOutcome::Hung);
            assert_eq!(attempts, 1, "Hung IRP must immediately abort with zero retries");
            assert!(slot.is_hung());
            assert_eq!(slot.neutral_completed_seq.load(Ordering::SeqCst), 0);
        }
    }

    #[test]
    fn monotonic_tickets_enforce_fatal_confirmation_led_upgrade_budget_cap_and_pipe_safety() {
        use std::sync::Barrier;

        // P2 #1: Failed 2nd/trailing send after a 1st successful send MUST NOT
        // advance `completed_seq` to the 2nd ticket -> fatal reset returns `Unconfirmed`.
        {
            let slot = EmergencySlot::new();
            assert!(slot.publish(0xD001, 0xD002, Transport::Usb, 48));
            assert_eq!(
                slot.run_emergency_reset(
                    true,
                    Duration::from_millis(10),
                    Duration::from_millis(10),
                    |_| {},
                    |_, _, _| NeutralSendOutcome::Delivered,
                    |_, _, _| NeutralSendOutcome::Delivered,
                    || {},
                ),
                ResetOutcome::Confirmed
            );
            assert_eq!(slot.neutral_completed_seq.load(Ordering::SeqCst), 1);
            assert_eq!(slot.release_leds_completed_seq.load(Ordering::SeqCst), 1);

            let outcome2 = slot.run_emergency_reset_with_budget(
                true,
                Duration::from_millis(10),
                Duration::from_millis(10),
                Duration::from_millis(100),
                |_| {},
                |_, _, _| NeutralSendOutcome::Failed,
                |_, _, _| NeutralSendOutcome::Failed,
                || {},
            );
            assert_eq!(
                outcome2,
                ResetOutcome::Unconfirmed,
                "failed 2nd fatal send must not falsely confirm from earlier completed ticket 1"
            );
            assert_eq!(slot.neutral_requested_seq.load(Ordering::SeqCst), 2);
            assert_eq!(slot.neutral_completed_seq.load(Ordering::SeqCst), 1);
            assert_eq!(slot.release_leds_completed_seq.load(Ordering::SeqCst), 1);
        }

        // P2 #2: Concurrent owner delivering an earlier send (`target < my_req`,
        // or `release_leds == false`) MUST NOT satisfy `is_fatal_confirmed(my_req)`;
        // only a `Delivered` send with `target >= my_req` and `release_leds == true` does.
        {
            let slot = Arc::new(EmergencySlot::new());
            assert!(slot.publish(0xD003, 0xD004, Transport::Usb, 48));
            let in_earlier_send = Arc::new(Barrier::new(2));
            let finish_earlier_send = Arc::new(Barrier::new(2));
            let (s_o, in_o, fin_o) = (
                Arc::clone(&slot),
                Arc::clone(&in_earlier_send),
                Arc::clone(&finish_earlier_send),
            );
            let owner = std::thread::spawn(move || {
                let req1 = s_o.request_neutral(true);
                assert_eq!(req1, 1);
                let _g = s_o.try_acquire_primary_neutral().unwrap();
                assert_eq!(
                    s_o.send_neutral_with_cancel_retry(|want_leds, _| {
                        assert!(want_leds);
                        // Snapshot `target = 1` ALREADY happened before entering `send_fn`!
                        in_o.wait();
                        fin_o.wait();
                        NeutralSendOutcome::Delivered
                    }),
                    NeutralSendOutcome::Delivered
                );
                drop(_g);
            });

            in_earlier_send.wait();
            // Fatal requests ticket 2 AFTER owner already snapshotted `target = 1`.
            let my_req = slot.request_neutral(true);
            assert_eq!(my_req, 2);
            finish_earlier_send.wait();
            owner.join().unwrap();

            assert_eq!(slot.release_leds_completed_seq.load(Ordering::SeqCst), 1);
            assert!(
                !slot.is_fatal_confirmed(my_req),
                "earlier delivery with target=1 must NOT confirm fatal ticket my_req=2"
            );

            // A subsequent `Delivered` send with `target >= 2` and `want_release_leds == true`
            // satisfies `is_fatal_confirmed(my_req)` via single-atomic check.
            let _g = slot.try_acquire_primary_neutral().unwrap();
            assert_eq!(
                slot.send_neutral_with_cancel_retry(|want_leds, _| {
                    assert!(want_leds);
                    NeutralSendOutcome::Delivered
                }),
                NeutralSendOutcome::Delivered
            );
            drop(_g);
            assert!(slot.is_fatal_confirmed(my_req));
            assert_eq!(slot.release_leds_completed_seq.load(Ordering::SeqCst), 2);
        }

        // P2 #3: Worker initially computing `release_leds = false` dynamically
        // upgrades to `release_leds = true` inside its trailing send when fatal
        // requests `release_leds = true` before the worker send runs.
        {
            let slot = Arc::new(EmergencySlot::new());
            assert!(slot.publish(0xD005, 0xD006, Transport::Usb, 48));
            let owner_locked = Arc::new(Barrier::new(2));
            let release_owner = Arc::new(Barrier::new(2));
            let owner_exited = Arc::new(Barrier::new(2));

            let (s_o, ol_o, ro_o, oe_o) = (
                Arc::clone(&slot),
                Arc::clone(&owner_locked),
                Arc::clone(&release_owner),
                Arc::clone(&owner_exited),
            );
            let blocker = std::thread::spawn(move || {
                let g = s_o.try_acquire_fallback_neutral().unwrap();
                ol_o.wait();
                ro_o.wait();
                drop(g);
                oe_o.wait();
            });

            owner_locked.wait();
            let worker_req = slot.request_neutral(false);
            let mut fatal_req = 0u64;
            let mut observed_release_leds = false;
            let mut ro_opt = Some(release_owner);
            let mut oe_opt = Some(owner_exited);
            let outcome = slot.deliver_or_handoff_neutral_with_hook(
                PHASE_PRIMARY_NEUTRAL,
                false, // Worker originally computed `release_leds = false`!
                Some(worker_req),
                || {
                    // Fatal requests `release_leds = true` right before worker sends!
                    fatal_req = slot.request_neutral(true);
                    if let (Some(ro), Some(oe)) = (ro_opt.take(), oe_opt.take()) {
                        ro.wait();
                        oe.wait();
                    }
                },
                |want_release_leds, _| {
                    observed_release_leds = want_release_leds;
                    NeutralSendOutcome::Delivered
                },
            );
            blocker.join().unwrap();
            assert_eq!(outcome, NeutralSendOutcome::Delivered);
            assert!(
                observed_release_leds,
                "worker trailing neutral must dynamically upgrade release_leds from false to true"
            );
            assert!(slot.is_fatal_confirmed(fatal_req));
        }

        // P2 #4: `fatal_deadline` prevents post-deadline `send_fn` calls and
        // bounds `remaining` budget passed to `send_fn` + `split_overlapped_budget`.
        {
            assert_eq!(split_overlapped_budget(None, 50, 50), Some((50, 50)));
            assert_eq!(
                split_overlapped_budget(Some(Duration::ZERO), 50, 50),
                None
            );
            assert_eq!(
                split_overlapped_budget(Some(Duration::from_millis(1)), 50, 50),
                Some((1, 0))
            );
            let (w, c) = split_overlapped_budget(Some(Duration::from_millis(30)), 50, 50).unwrap();
            assert!(w + c <= 30 && w <= 50 && c <= 50);

            // (a) Worker holds `SLOT_WRITING` past `fatal_budget` while `SLOT_EMERGENCY` is held:
            // fallback waits for `SLOT_WRITING`, hits deadline, and performs ZERO post-deadline sends.
            let slot = EmergencySlot::new();
            assert!(slot.publish(0xD007, 0xD008, Transport::Usb, 48));
            let shutdown = AtomicBool::new(false);
            let _w = slot.try_acquire_worker_write(&shutdown).unwrap();
            let _b = slot
                .try_claim_emergency(Duration::from_millis(5))
                .expect("hold SLOT_EMERGENCY");
            let mut fallback_calls = 0usize;
            let outcome = slot.run_emergency_reset_with_budget(
                true,
                Duration::from_millis(5),
                Duration::from_millis(5),
                Duration::from_millis(20),
                |_| {},
                |_, _, _| NeutralSendOutcome::Delivered,
                |_, _, _| {
                    fallback_calls += 1;
                    NeutralSendOutcome::Delivered
                },
                || {},
            );
            assert_eq!(outcome, ResetOutcome::Unconfirmed);
            assert_eq!(
                fallback_calls, 0,
                "must NEVER invoke send_fn after SLOT_WRITING wait exhausts fatal_deadline"
            );
            slot.release_writing();
            drop(_b);

            // (b) First send consumes remaining fatal_budget (10 ms) and returns `Failed`:
            // retry sleep / post-sleep deadline check prevents any post-deadline retry call!
            let slot2 = EmergencySlot::new();
            assert!(slot2.publish(0xD009, 0xD00A, Transport::Usb, 48));
            let budget = Duration::from_millis(12);
            let mut calls = 0usize;
            let mut seen_rem = Vec::new();
            let outcome = slot2.run_emergency_reset_with_budget(
                true,
                Duration::from_millis(2),
                Duration::from_millis(2),
                budget,
                |_| {},
                |_, _, rem| {
                    calls += 1;
                    seen_rem.push(rem);
                    std::thread::sleep(Duration::from_millis(15));
                    NeutralSendOutcome::Failed
                },
                |_, _, _| NeutralSendOutcome::Failed,
                || {},
            );
            assert_eq!(outcome, ResetOutcome::Unconfirmed);
            assert_eq!(calls, 1, "must NOT invoke retry send_fn past fatal_deadline");
            let rem0 = seen_rem[0].expect("fatal send_fn must receive Some(remaining)");
            assert!(rem0 <= budget && !rem0.is_zero());
        }

        // P2 #5: Pipe-safe fatal diagnostic helper updates `FATAL_UNCONFIRMED_SLOTS`
        // and permits synchronous stderr WriteFile ONLY for `FILE_TYPE_CHAR`.
        {
            assert!(should_write_fatal_stderr(FILE_TYPE_CHAR_RAW));
            assert!(!should_write_fatal_stderr(0x0003)); // FILE_TYPE_PIPE
            assert!(!should_write_fatal_stderr(0x0001)); // FILE_TYPE_DISK
            assert!(!should_write_fatal_stderr(0x0000)); // FILE_TYPE_UNKNOWN
            record_fatal_unconfirmed_slot(2);
            assert_ne!(
                FATAL_UNCONFIRMED_SLOTS.load(Ordering::SeqCst) & (1u32 << 2),
                0
            );
        }

        // P2 #6: Fast-failing HID writes in BOTH fatal primary AND fatal fallback
        // outer loops are strictly capped at `MAX_NEUTRAL_ATTEMPTS = 3` total calls.
        {
            // (a) Fatal primary outer loop:
            let slot_p = EmergencySlot::new();
            assert!(slot_p.publish(0xD00B, 0xD00C, Transport::Usb, 48));
            let mut primary_attempts = 0u32;
            let outcome_p = slot_p.run_emergency_reset_with_budget(
                true,
                Duration::from_millis(5),
                Duration::from_millis(5),
                Duration::from_millis(200),
                |_| {},
                |_, _, _| {
                    primary_attempts += 1;
                    NeutralSendOutcome::Failed
                },
                |_, _, _| NeutralSendOutcome::Failed,
                || {},
            );
            assert_eq!(outcome_p, ResetOutcome::Unconfirmed);
            assert_eq!(
                primary_attempts, MAX_NEUTRAL_ATTEMPTS,
                "fatal primary outer loop must cap total HID writes at MAX_NEUTRAL_ATTEMPTS (3)"
            );

            // (b) Fatal fallback outer loop:
            let slot_f = EmergencySlot::new();
            assert!(slot_f.publish(0xD00D, 0xD00E, Transport::Usb, 48));
            let _held = slot_f
                .try_claim_emergency(Duration::from_millis(5))
                .expect("hold SLOT_EMERGENCY");
            let mut fallback_attempts = 0u32;
            let outcome_f = slot_f.run_emergency_reset_with_budget(
                true,
                Duration::from_millis(5),
                Duration::from_millis(5),
                Duration::from_millis(200),
                |_| {},
                |_, _, _| NeutralSendOutcome::Failed,
                |_, _, _| {
                    fallback_attempts += 1;
                    NeutralSendOutcome::Failed
                },
                || {},
            );
            assert_eq!(outcome_f, ResetOutcome::Unconfirmed);
            assert_eq!(
                fallback_attempts, MAX_NEUTRAL_ATTEMPTS,
                "fatal fallback outer loop must cap total HID writes at MAX_NEUTRAL_ATTEMPTS (3)"
            );
            drop(_held);
        }
    }

    #[test]
    fn cancel_guard_completing_while_sender_holds_phase_never_drops_post_cancel_ticket() {
        use std::sync::Barrier;

        // 1. Direct contract check: `send_neutral_with_cancel_retry_bounded`
        // returns `SendAttemptOutcome::HandedOffToCancel` (distinct from
        // exhausted `Failed`) when `PHASE_CANCEL_MASK != 0`.
        {
            let slot = EmergencySlot::new();
            assert!(slot.publish(0xE001, 0xE002, Transport::Usb, 48));
            let _req = slot.request_neutral(true);
            slot.io_phase.fetch_add(1, Ordering::SeqCst); // simulate active CancelIoGuard
            let _g = slot.try_acquire_primary_neutral().unwrap();
            let mut attempts = MAX_NEUTRAL_ATTEMPTS;
            let outcome = slot.send_neutral_with_cancel_retry_bounded(
                None,
                false,
                &mut attempts,
                |_, _| NeutralSendOutcome::Delivered,
            );
            assert_eq!(outcome, SendAttemptOutcome::HandedOffToCancel);
            assert!(slot.neutral_after_cancel.load(Ordering::SeqCst));
            assert_eq!(attempts, MAX_NEUTRAL_ATTEMPTS - 1);
            drop(_g);
            slot.io_phase.fetch_sub(1, Ordering::SeqCst);
        }

        // 2. Live two-thread race: delayed `CancelIoGuard` waits until sender
        // sets `neutral_after_cancel = true` while STILL holding `PHASE_PRIMARY_NEUTRAL`,
        // then completes `CancelIoGuard::drop` immediately (`on_neutral_after_cancel`
        // calls `deliver_or_handoff_neutral`, sees `PHASE_PRIMARY_NEUTRAL` busy,
        // and bumps `neutral_requested_seq`). Sender MUST NOT treat the cancel
        // handoff as an exhausted failure; it services the post-cancel ticket
        // once `PHASE_CANCEL_MASK == 0` and advances `neutral_completed_seq` cleanly!
        for _ in 0..3 {
            let slot = Arc::new(EmergencySlot::new());
            assert!(slot.publish(0xE003, 0xE004, Transport::Usb, 48));
            let cancel_entered = Arc::new(Barrier::new(2));
            let post_cancel_deliveries = Arc::new(AtomicU32::new(0));

            let (s_c, ce_c, pcd_c) = (
                Arc::clone(&slot),
                Arc::clone(&cancel_entered),
                Arc::clone(&post_cancel_deliveries),
            );
            let canceller = std::thread::spawn(move || {
                let s_cb = Arc::clone(&s_c);
                let pcd_cb = Arc::clone(&pcd_c);
                assert!(s_c.cancel_io_if_no_neutral_in_flight(
                    0xE003,
                    |_| {
                        ce_c.wait();
                        while !s_c.neutral_after_cancel.load(Ordering::SeqCst) {
                            std::thread::yield_now();
                        }
                    },
                    move || {
                        let _ = s_cb.deliver_or_handoff_neutral(
                            PHASE_PRIMARY_NEUTRAL,
                            true,
                            |want_leds, _| {
                                assert!(want_leds);
                                assert_eq!(
                                    s_cb.io_phase.load(Ordering::SeqCst) & PHASE_CANCEL_MASK,
                                    0
                                );
                                pcd_cb.fetch_add(1, Ordering::SeqCst);
                                NeutralSendOutcome::Delivered
                            },
                        );
                    },
                ));
            });

            cancel_entered.wait();
            let pcd_s = Arc::clone(&post_cancel_deliveries);
            let s_s = Arc::clone(&slot);
            let _ = slot.deliver_or_handoff_neutral(PHASE_PRIMARY_NEUTRAL, true, |want_leds, _| {
                assert!(want_leds);
                if (s_s.io_phase.load(Ordering::SeqCst) & PHASE_CANCEL_MASK) == 0 {
                    pcd_s.fetch_add(1, Ordering::SeqCst);
                }
                NeutralSendOutcome::Delivered
            });
            canceller.join().unwrap();

            assert!(
                post_cancel_deliveries.load(Ordering::SeqCst) >= 1,
                "at least one uncancelled post-cancel neutral must be delivered after PHASE_CANCEL_MASK == 0"
            );
            assert_eq!(
                slot.neutral_completed_seq.load(Ordering::SeqCst),
                slot.neutral_requested_seq.load(Ordering::SeqCst),
                "post-cancel ticket must never be dropped when CancelIoGuard::drop runs while sender holds phase"
            );
            assert_eq!(
                slot.release_leds_completed_seq.load(Ordering::SeqCst),
                slot.release_leds_requested_seq.load(Ordering::SeqCst)
            );
            assert!(!slot.has_pending_neutral());
            assert!(!slot.neutral_after_cancel.load(Ordering::SeqCst));
        }
    }

    #[test]
    fn bounded_join_returns_promptly_when_thread_exceeds_deadline() {
        use std::sync::Condvar;

        // 1. Thread that finishes promptly joins cleanly within deadline.
        let fast = std::thread::spawn(|| {});
        assert!(join_handle_bounded(
            fast,
            Instant::now() + Duration::from_millis(250)
        ));

        // 2. Wedged thread past deadline returns promptly without blocking;
        // `SharedJoinHandle` retains the live `JoinHandle` and keeps
        // `is_os_finished() == false` until the OS thread actually exits.
        let gate = Arc::new((Mutex::new(false), Condvar::new()));
        let gate_bg = Arc::clone(&gate);
        let slow = SharedJoinHandle::new(std::thread::spawn(move || {
            let (m, cv) = &*gate_bg;
            let mut done = m.lock().unwrap();
            while !*done {
                done = cv.wait(done).unwrap();
            }
        }));
        let start = Instant::now();
        let joined = slow.join_bounded(start + Duration::from_millis(25));
        assert!(!joined);
        assert!(
            start.elapsed() < Duration::from_millis(200),
            "bounded OS-thread join must return promptly on a wedged thread"
        );
        assert!(
            !slow.is_os_finished(),
            "timed-out SharedJoinHandle must retain live JoinHandle and not report finished"
        );
        // Cleanly wake the thread and verify a subsequent bounded join succeeds.
        {
            let (m, cv) = &*gate;
            *m.lock().unwrap() = true;
            cv.notify_all();
        }
        assert!(slow.join_bounded(Instant::now() + Duration::from_millis(250)));
        assert!(slow.is_os_finished());
    }

    #[test]
    fn set_rumble_after_disconnect_is_ignored_and_never_persists_into_reconnect() {
        let cfg = DualSenseConfig::default();
        let mut mgr = DualSenseManager::new(cfg.clone());
        let shared = Arc::new(SharedSlot {
            path: "test-ds".into(),
            transport: Transport::Usb,
            container_id: None,
            stop: AtomicBool::new(false),
            state: Mutex::new(WorkerState::new(0, cfg)),
        });
        shared.state.lock().unwrap().snapshot.connected = true;
        mgr.slots[0] = Some(Arc::clone(&shared));

        mgr.set_rumble(0, [0.8, 0.6, 0.9, 0.7]);
        assert_eq!(shared.state.lock().unwrap().rumble, [0.8, 0.6, 0.9, 0.7]);

        let _ = clear_slot_state_on_disconnect(&shared);
        assert_eq!(shared.state.lock().unwrap().rumble, [0.0; 4]);

        mgr.set_rumble(0, [1.0, 1.0, 1.0, 1.0]);
        assert_eq!(
            shared.state.lock().unwrap().rumble,
            [0.0; 4],
            "set_rumble while disconnected must not store stale rumble"
        );

        {
            let mut st = shared.state.lock().unwrap();
            if !st.snapshot.connected {
                st.rumble = [0.0; 4];
            }
            st.snapshot.connected = true;
        }
        assert_eq!(shared.state.lock().unwrap().rumble, [0.0; 4]);
        mgr.set_rumble(0, [0.25, 0.5, 0.0, 0.0]);
        assert_eq!(shared.state.lock().unwrap().rumble, [0.25, 0.5, 0.0, 0.0]);
    }

    #[test]
    fn disconnect_clears_stale_rumble_and_snapshot() {
        let shared = SharedSlot {
            path: "test-ds".into(),
            transport: Transport::Usb,
            container_id: None,
            stop: AtomicBool::new(false),
            state: Mutex::new(WorkerState::new(0, DualSenseConfig::default())),
        };
        {
            let mut st = shared.state.lock().unwrap();
            st.snapshot.connected = true;
            st.raw_latest = Some(RawReport::default());
            st.rumble = [0.8, 0.6, 0.9, 0.7];
            st.audio_haptics_active = true;
            st.firmware_version = 0x0402;
        }
        let fw = clear_slot_state_on_disconnect(&shared);
        assert_eq!(fw, 0x0402);
        let st = shared.state.lock().unwrap();
        assert_eq!(st.rumble, [0.0; 4]);
        assert!(!st.snapshot.connected);
        assert!(st.raw_latest.is_none());
        assert!(!st.audio_haptics_active);
    }

    #[test]
    fn unpaired_usb_dongle_plus_wired_dualsense_triggers_multi_usb_hid_fallback() {
        let cfg = DualSenseConfig {
            haptics_mode: HapticsMode::Hybrid,
            ..Default::default()
        };
        let mut mgr = DualSenseManager::new(cfg.clone());
        #[cfg(windows)]
        {
            mgr.last_scan = Some(Instant::now());
        }

        let cid_dongle = ContainerId::from_guid_fields(1, 2, 3, [4; 8]);
        let cid_wired = ContainerId::from_guid_fields(5, 6, 7, [8; 8]);

        let dongle = Arc::new(SharedSlot {
            path: "usb-dongle".into(),
            transport: Transport::Usb,
            container_id: cid_dongle,
            stop: AtomicBool::new(false),
            state: Mutex::new(WorkerState::new(0, cfg.clone())),
        });
        let wired = Arc::new(SharedSlot {
            path: "usb-wired".into(),
            transport: Transport::Usb,
            container_id: cid_wired,
            stop: AtomicBool::new(false),
            state: Mutex::new(WorkerState::new(1, cfg)),
        });
        wired.state.lock().unwrap().snapshot = SlotSnapshot {
            connected: true,
            transport: Some(Transport::Usb),
            ..Default::default()
        };

        mgr.slots[0] = Some(Arc::clone(&dongle));
        mgr.slots[1] = Some(Arc::clone(&wired));
        let snaps = mgr.poll(true);
        assert!(snaps[0].is_none());
        assert!(snaps[1].is_some());

        let wired_st = wired.state.lock().unwrap();
        assert!(
            wired_st.multi_usb,
            "open unpaired USB dongle + wired DualSense must count as multi-USB"
        );
        assert!(!wired_st.audio_haptics_active);
        assert!(should_enable_hid_rumble(
            Transport::Usb,
            HapticsMode::Hybrid,
            wired_st.audio_haptics_active,
            wired_st.multi_usb
        ));
    }
}
