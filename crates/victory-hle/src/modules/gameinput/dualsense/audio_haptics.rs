//! Optional 4-channel WASAPI DualSense voice-coil haptics synthesizer
//! (`haptics_mode = "audio_4ch" | "hybrid"`).
//!
//! Wired DualSense and Pico 2 W `DS5Dongle` expose a 4-channel 48 kHz audio
//! endpoint (`FL, FR, RL, RR`): channels 1-2 feed the controller speaker /
//! headset jack (kept silent unless `speaker_mirror` is enabled) and channels
//! 3-4 drive the left and right voice-coil actuators. Game rumble scalars are
//! modulated onto dual resonant carriers (`58 Hz` heavy + `162 Hz` crisp) and
//! blended with low-passed audio captured exclusively from our own process
//! tree (`VIRTUAL_AUDIO_DEVICE_PROCESS_LOOPBACK`), never system-wide audio.

use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::{Arc, Condvar, LazyLock, Mutex};
use std::time::{Duration, Instant};

#[cfg(any(windows, test))]
use super::SharedJoinHandle;

/// Non-zero Windows PnP `DEVPKEY_Device_ContainerId` / `PKEY_Device_ContainerId`
/// GUID uniquely identifying one physical multi-function USB DualSense device
/// (shared between its HID gamepad interface and its 4-channel WASAPI endpoint).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ContainerId(u128);

impl ContainerId {
    pub fn from_guid_fields(data1: u32, data2: u16, data3: u16, data4: [u8; 8]) -> Option<Self> {
        let v = (u128::from(data1) << 96)
            | (u128::from(data2) << 80)
            | (u128::from(data3) << 64)
            | u128::from(u64::from_be_bytes(data4));
        (v != 0).then_some(Self(v))
    }

    #[cfg(any(windows, test))]
    pub fn from_raw_guid_bytes(bytes: &[u8]) -> Option<Self> {
        if bytes.len() < 16 {
            return None;
        }
        let data1 = u32::from_le_bytes(bytes[0..4].try_into().ok()?);
        let data2 = u16::from_le_bytes(bytes[4..6].try_into().ok()?);
        let data3 = u16::from_le_bytes(bytes[6..8].try_into().ok()?);
        let data4: [u8; 8] = bytes[8..16].try_into().ok()?;
        Self::from_guid_fields(data1, data2, data3, data4)
    }
}

/// Process-wide shutdown latch set during exit/Ctrl+C/fatal crash so any live
/// WASAPI voice-coil render thread silences and exits immediately.
static AUDIO_SHUTDOWN: AtomicBool = AtomicBool::new(false);

#[derive(Default)]
struct GateState {
    closed: bool,
    in_flight: usize,
}

/// Synchronizes WASAPI `GetBuffer -> fill -> ReleaseBuffer` submission against
/// `stop()` and `shutdown_global_bounded()` without holding a `Mutex` across
/// COM calls:
/// - `try_begin()` rejects new submissions once `closed == true` and increments
///   `in_flight: usize` until its [`SubmitGuard`] drops (counting overlapping
///   workers so a newer worker exiting can never mask an older wedged worker).
/// - `close_bounded(deadline)` sets `closed = true` and waits on the `Condvar`
///   until `in_flight == 0` (or `deadline` expires, so a wedged WASAPI driver
///   can never freeze polling or exit).
pub(crate) struct SubmitGate {
    state: Mutex<GateState>,
    changed: Condvar,
}

impl SubmitGate {
    pub(crate) fn new() -> Self {
        Self {
            state: Mutex::new(GateState::default()),
            changed: Condvar::new(),
        }
    }

    #[cfg(any(windows, test))]
    pub(crate) fn try_begin(&self) -> Option<SubmitGuard<'_>> {
        let mut st = self.state.lock().unwrap_or_else(|e| e.into_inner());
        if st.closed {
            return None;
        }
        st.in_flight += 1;
        Some(SubmitGuard { gate: self })
    }

    /// Closes the gate and waits until all active `ReleaseBuffer` calls finish
    /// or `deadline` expires. Returns `true` if zero submissions remain in flight.
    pub(crate) fn close_bounded(&self, deadline: Instant) -> bool {
        let mut st = self.state.lock().unwrap_or_else(|e| e.into_inner());
        st.closed = true;
        while st.in_flight > 0 {
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
            if wait_res.timed_out() && st.in_flight > 0 {
                return false;
            }
        }
        true
    }

    #[cfg(any(windows, test))]
    pub(crate) fn in_flight(&self) -> usize {
        self.state.lock().unwrap_or_else(|e| e.into_inner()).in_flight
    }

    /// Lock-free / non-blocking close for crash and Ctrl+C handlers.
    pub(crate) fn close_nonblocking(&self) {
        if let Ok(mut st) = self.state.try_lock() {
            st.closed = true;
        }
    }
}

/// Non-blockingly reaps any previously timed-out audio workers whose OS thread
/// has exited AND whose `SubmitGate` has drained (`in_flight == 0`). Returns
/// `true` iff zero draining workers remain and it is safe to spawn a new one.
#[cfg(any(windows, test))]
pub(crate) fn prune_draining_workers(
    draining: &mut Vec<(SharedJoinHandle, Arc<SubmitGate>)>,
) -> bool {
    let mut i = 0;
    while i < draining.len() {
        let (ref handle, ref gate) = draining[i];
        if gate.in_flight() == 0 && handle.is_os_finished() {
            let _ = handle.try_join_if_os_finished();
            draining.swap_remove(i);
        } else {
            i += 1;
        }
    }
    draining.is_empty()
}

#[cfg(any(windows, test))]
pub(crate) struct SubmitGuard<'a> {
    gate: &'a SubmitGate,
}

#[cfg(any(windows, test))]
impl Drop for SubmitGuard<'_> {
    fn drop(&mut self) {
        let mut st = self.gate.state.lock().unwrap_or_else(|e| e.into_inner());
        st.in_flight = st.in_flight.saturating_sub(1);
        if st.in_flight == 0 {
            self.gate.changed.notify_all();
        }
    }
}

static GLOBAL_SUBMIT_GATE: LazyLock<SubmitGate> = LazyLock::new(SubmitGate::new);

/// Lock-free global shutdown for crash and Ctrl+C handlers (never blocks on a mutex).
pub fn shutdown_global_lockfree() {
    AUDIO_SHUTDOWN.store(true, Ordering::SeqCst);
    GLOBAL_SUBMIT_GATE.close_nonblocking();
}

/// Synchronized, time-bounded global shutdown for normal exit / `end_title`.
/// Once this returns `true`, no non-zero `ReleaseBuffer` is in progress or can start.
pub fn shutdown_global_bounded(budget: Duration) -> bool {
    AUDIO_SHUTDOWN.store(true, Ordering::SeqCst);
    GLOBAL_SUBMIT_GATE.close_bounded(Instant::now() + budget)
}

/// Returns `true` iff the audio worker's bound `active_container` still matches
/// `current_target` (checked on every audio loop iteration before emitting samples).
#[cfg(any(windows, test))]
pub(crate) fn target_matches_active(
    current_target: Option<ContainerId>,
    active_container: ContainerId,
) -> bool {
    current_target == Some(active_container)
}

#[cfg(any(windows, test))]
const INITIAL_BACKOFF_SECS: u64 = 2;
#[cfg(any(windows, test))]
const MAX_BACKOFF_SECS: u64 = 10;

#[cfg(any(windows, test))]
fn next_backoff_secs(current: u64) -> u64 {
    if current == 0 {
        INITIAL_BACKOFF_SECS
    } else {
        current.saturating_mul(2).min(MAX_BACKOFF_SECS)
    }
}

/// Per-`ContainerId` exponential retry backoff so startup failures on controller
/// A never delay binding a newly switched controller B.
#[cfg(any(windows, test))]
#[derive(Default)]
pub(crate) struct RetryBackoff {
    next_retry_at: Option<Instant>,
    backoff_secs: u64,
    backoff_container: Option<ContainerId>,
}

#[cfg(any(windows, test))]
impl RetryBackoff {
    pub(crate) fn clear(&mut self) {
        self.next_retry_at = None;
        self.backoff_secs = 0;
        self.backoff_container = None;
    }

    /// Returns `true` if startup should defer for `current_target` at `now`.
    /// Clears stale backoff immediately whenever `current_target != backoff_container`.
    pub(crate) fn should_defer(&mut self, current_target: Option<ContainerId>, now: Instant) -> bool {
        if self.backoff_container != current_target {
            self.clear();
            return false;
        }
        self.next_retry_at.is_some_and(|at| now < at)
    }

    pub(crate) fn record_exit(
        &mut self,
        attempted_container: Option<ContainerId>,
        current_target: Option<ContainerId>,
        stream_started: bool,
        target_changed: bool,
        now: Instant,
    ) {
        if target_changed || current_target != attempted_container {
            self.clear();
            return;
        }
        let prev_secs = if self.backoff_container == attempted_container {
            self.backoff_secs
        } else {
            0
        };
        let next_secs = if stream_started {
            INITIAL_BACKOFF_SECS
        } else {
            next_backoff_secs(prev_secs)
        };
        self.backoff_container = attempted_container;
        self.backoff_secs = next_secs;
        self.next_retry_at = Some(now + Duration::from_secs(next_secs));
    }
}

#[derive(Default)]
struct BindingState {
    generation: u64,
    target_container: Option<ContainerId>,
    active: bool,
}

/// Shared control block updated by `gameinput` and read by the WASAPI render thread.
pub struct AudioHapticsControl {
    /// Protects `(generation, target_container, active)` as ONE atomic snapshot
    /// so an `A -> B -> A` switch can never slip between reading `generation`
    /// and reading `target_container`, and a stale worker can never clobber a
    /// newer worker's `active` flag.
    binding: Mutex<BindingState>,
    speaker_mirror: AtomicBool,
    muted: AtomicBool,
    low: AtomicU32,
    high: AtomicU32,
    left_trigger: AtomicU32,
    right_trigger: AtomicU32,
}

impl Default for AudioHapticsControl {
    fn default() -> Self {
        Self {
            binding: Mutex::new(BindingState::default()),
            speaker_mirror: AtomicBool::new(false),
            muted: AtomicBool::new(false),
            low: AtomicU32::new(0.0f32.to_bits()),
            high: AtomicU32::new(0.0f32.to_bits()),
            left_trigger: AtomicU32::new(0.0f32.to_bits()),
            right_trigger: AtomicU32::new(0.0f32.to_bits()),
        }
    }
}

impl AudioHapticsControl {
    pub fn is_active(&self) -> bool {
        self.binding.lock().unwrap_or_else(|e| e.into_inner()).active
    }

    /// Advances the monotonic worker generation token and clears the active bit,
    /// invalidating any stale/timed-out audio worker from earlier generations.
    pub(crate) fn claim_generation(&self) -> u64 {
        let mut st = self.binding.lock().unwrap_or_else(|e| e.into_inner());
        st.generation = st.generation.wrapping_add(1);
        st.active = false;
        st.generation
    }

    /// Returns `true` iff `generation` is still the active worker generation
    /// (prevents ABA stale-worker survival across `A -> B -> A` switches).
    #[cfg(test)]
    pub(crate) fn is_current_generation(&self, generation: u64) -> bool {
        self.binding
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .generation
            == generation
    }

    /// Sets or clears the `active` bit ONLY if `generation` is still the current
    /// generation token, returning `true` if the update took effect.
    #[cfg(any(windows, test))]
    pub(crate) fn set_active_if_current(&self, generation: u64, active: bool) -> bool {
        let mut st = self.binding.lock().unwrap_or_else(|e| e.into_inner());
        if st.generation == generation {
            st.active = active;
            true
        } else {
            false
        }
    }

    /// Atomically verifies in ONE snapshot that `generation` is still current
    /// AND `active_container` still matches `target_container` (checked before
    /// Start, per-iteration, before GetBuffer, and right before ReleaseBuffer).
    #[cfg(any(windows, test))]
    pub(crate) fn worker_binding_valid(
        &self,
        generation: u64,
        active_container: ContainerId,
    ) -> bool {
        let st = self.binding.lock().unwrap_or_else(|e| e.into_inner());
        st.generation == generation && target_matches_active(st.target_container, active_container)
    }

    pub fn set_target_container(&self, container_id: Option<ContainerId>) {
        let mut st = self.binding.lock().unwrap_or_else(|e| e.into_inner());
        if st.target_container != container_id {
            st.target_container = container_id;
            // Immediately invalidate the previous controller's worker generation
            // and clear `active` atomically in the same critical section so Hybrid
            // mode enables HID rumble fallback right away even if the old audio
            // thread is slow or wedged.
            st.generation = st.generation.wrapping_add(1);
            st.active = false;
        }
    }

    #[cfg(any(windows, test))]
    pub(crate) fn target_container(&self) -> Option<ContainerId> {
        self.binding
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .target_container
    }

    pub fn set_levels(&self, low: f32, high: f32, lt: f32, rt: f32) {
        let norm = |v: f32| if v.is_finite() { v.clamp(0.0, 1.0) } else { 0.0 };
        self.low.store(norm(low).to_bits(), Ordering::Release);
        self.high.store(norm(high).to_bits(), Ordering::Release);
        self.left_trigger
            .store(norm(lt).to_bits(), Ordering::Release);
        self.right_trigger
            .store(norm(rt).to_bits(), Ordering::Release);
    }

    pub fn clear_levels(&self) {
        self.set_levels(0.0, 0.0, 0.0, 0.0);
    }

    pub fn set_options(&self, speaker_mirror: bool, muted: bool) {
        self.speaker_mirror.store(speaker_mirror, Ordering::Release);
        self.muted.store(muted, Ordering::Release);
    }

    #[cfg(any(windows, test))]
    fn levels(&self) -> [f32; 4] {
        [
            f32::from_bits(self.low.load(Ordering::Acquire)),
            f32::from_bits(self.high.load(Ordering::Acquire)),
            f32::from_bits(self.left_trigger.load(Ordering::Acquire)),
            f32::from_bits(self.right_trigger.load(Ordering::Acquire)),
        ]
    }
}

/// Pure DSP synthesizer (unit-testable without audio hardware).
#[derive(Default)]
pub struct TactileSynth {
    phase_low: f32,
    phase_high: f32,
    lp_l1: f32,
    lp_l2: f32,
    lp_r1: f32,
    lp_r2: f32,
}

impl TactileSynth {
    /// Generates one 4-channel frame `[FL, FR, HapticL, HapticR]` at `rate` Hz.
    pub fn next_frame(
        &mut self,
        rate: f32,
        levels: [f32; 4],
        game_stereo: [f32; 2],
        speaker_mirror: bool,
        muted: bool,
    ) -> [f32; 4] {
        if muted {
            // Reset low-pass state so stale pre-mute loopback audio never leaks on resume.
            self.lp_l1 = 0.0;
            self.lp_l2 = 0.0;
            self.lp_r1 = 0.0;
            self.lp_r2 = 0.0;
            return [0.0; 4];
        }
        let [low, high, lt, rt] = levels;
        let tau = std::f32::consts::TAU;
        self.phase_low = (self.phase_low + 58.0 / rate).fract();
        self.phase_high = (self.phase_high + 162.0 / rate).fract();

        let carrier_low = (self.phase_low * tau).sin();
        let carrier_high = (self.phase_high * tau).sin();

        // Cascaded 1-pole low-pass (~135 Hz @ 48 kHz) extracting tactile sub-bass
        // from the title's own process-loopback stereo stream.
        let alpha = (tau * 135.0 / rate).clamp(0.005, 0.25);
        self.lp_l1 += alpha * (game_stereo[0] - self.lp_l1);
        self.lp_l2 += alpha * (self.lp_l1 - self.lp_l2);
        self.lp_r1 += alpha * (game_stereo[1] - self.lp_r1);
        self.lp_r2 += alpha * (self.lp_r1 - self.lp_r2);

        let rumble_l = (low * 0.75 + lt * 0.35) * carrier_low + (high * 0.30) * carrier_high;
        let rumble_r = (low * 0.45) * carrier_low + (high * 0.75 + rt * 0.35) * carrier_high;

        let soft_clip = |x: f32| (x / (1.0 + 0.5 * x.abs())).clamp(-0.92, 0.92);
        let haptic_l = soft_clip(rumble_l + self.lp_l2 * 0.55);
        let haptic_r = soft_clip(rumble_r + self.lp_r2 * 0.55);

        let (fl, fr) = if speaker_mirror {
            (
                (game_stereo[0] * 0.6).clamp(-0.9, 0.9),
                (game_stereo[1] * 0.6).clamp(-0.9, 0.9),
            )
        } else {
            (0.0, 0.0)
        };
        [fl, fr, haptic_l, haptic_r]
    }
}

pub struct AudioHapticsEngine {
    control: Arc<AudioHapticsControl>,
    #[cfg(windows)]
    worker: Mutex<imp::WorkerSlot>,
}

impl Default for AudioHapticsEngine {
    fn default() -> Self {
        Self {
            control: Arc::new(AudioHapticsControl::default()),
            #[cfg(windows)]
            worker: Mutex::new(imp::WorkerSlot::default()),
        }
    }
}

impl AudioHapticsEngine {
    pub fn control(&self) -> &Arc<AudioHapticsControl> {
        &self.control
    }

    pub fn is_active(&self) -> bool {
        self.control.is_active()
    }

    pub fn ensure_started(&self) {
        if AUDIO_SHUTDOWN.load(Ordering::Acquire) {
            return;
        }
        #[cfg(windows)]
        imp::ensure_started(&self.control, &self.worker);
    }

    pub fn stop(&self) {
        self.control.clear_levels();
        self.control.set_options(false, true);
        #[cfg(windows)]
        imp::stop(&self.control, &self.worker);
    }
}

impl Drop for AudioHapticsEngine {
    fn drop(&mut self) {
        self.stop();
    }
}

/// Pure-Rust representation of one active audio render endpoint observation,
/// kept platform-independent so fail-closed ambiguity resolution is directly
/// unit-testable.
#[cfg(any(windows, test))]
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum EndpointObservation<T> {
    Named {
        id: String,
        name_lower: String,
        container_id: Option<ContainerId>,
        device: T,
    },
    Unreadable,
}

#[cfg(any(windows, test))]
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum EndpointResolution<T> {
    None,
    Unique {
        id: String,
        name: String,
        device: T,
    },
    Ambiguous(&'static str),
}

#[cfg(any(windows, test))]
pub(crate) fn is_dualsense_audio_endpoint_name(lower: &str) -> bool {
    lower.contains("wireless controller") || lower.contains("dualsense") || lower.contains("ds5")
}

/// Fail-closed 1-to-1 physical DualSense render endpoint selector:
/// - Requires a valid non-zero `target_container` (`DEVPKEY_Device_ContainerId`
///   from the opened USB HID gamepad) so audio haptics can ONLY bind to the
///   exact same physical USB DualSense device.
/// - Returns `Unique` ONLY when every active render endpoint's friendly name
///   was read cleanly, no DualSense-named endpoint lacks a `ContainerId`, and
///   exactly one active DualSense endpoint matches `target_container`.
#[cfg(any(windows, test))]
pub(crate) fn resolve_unique_dualsense_endpoint<T>(
    target_container: Option<ContainerId>,
    observations: impl IntoIterator<Item = EndpointObservation<T>>,
) -> EndpointResolution<T> {
    let Some(target) = target_container else {
        return EndpointResolution::Ambiguous(
            "HID controller ContainerId unavailable; refusing unverified 4ch audio routing",
        );
    };
    let mut first: Option<(String, String, T)> = None;
    let mut match_count = 0usize;
    let mut unreadable_count = 0usize;
    for obs in observations {
        match obs {
            EndpointObservation::Named {
                id,
                name_lower,
                container_id,
                device,
            } => {
                let is_ds_name = is_dualsense_audio_endpoint_name(&name_lower);
                match container_id {
                    Some(cid) if cid == target && is_ds_name => {
                        match_count += 1;
                        if first.is_none() {
                            first = Some((id, name_lower, device));
                        }
                    }
                    None if is_ds_name => {
                        unreadable_count += 1;
                    }
                    _ => {}
                }
            }
            EndpointObservation::Unreadable => {
                unreadable_count += 1;
            }
        }
    }
    if match_count > 1 {
        EndpointResolution::Ambiguous(
            "multiple active DualSense audio endpoints share the same ContainerId; refusing ambiguous routing",
        )
    } else if unreadable_count > 0 {
        EndpointResolution::Ambiguous(
            "failed to read friendly name or ContainerId for an active audio endpoint; refusing ambiguous routing",
        )
    } else if let Some((id, name, device)) = first {
        EndpointResolution::Unique { id, name, device }
    } else {
        EndpointResolution::None
    }
}

#[cfg(windows)]
mod imp {
    use std::collections::VecDeque;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::{Arc, Mutex};
    use std::time::{Duration, Instant};

    use windows::Win32::Devices::FunctionDiscovery::{
        PKEY_Device_ContainerId, PKEY_Device_FriendlyName,
    };
    use windows::Win32::Media::Audio::{
        AUDCLNT_BUFFERFLAGS_SILENT, AUDCLNT_SHAREMODE_SHARED, AUDCLNT_STREAMFLAGS_AUTOCONVERTPCM,
        AUDCLNT_STREAMFLAGS_LOOPBACK, AUDIOCLIENT_ACTIVATION_PARAMS,
        AUDIOCLIENT_ACTIVATION_PARAMS_0, AUDIOCLIENT_ACTIVATION_TYPE_PROCESS_LOOPBACK,
        AUDIOCLIENT_PROCESS_LOOPBACK_PARAMS, ActivateAudioInterfaceAsync, DEVICE_STATE_ACTIVE,
        IActivateAudioInterfaceAsyncOperation, IActivateAudioInterfaceCompletionHandler,
        IActivateAudioInterfaceCompletionHandler_Impl, IAudioCaptureClient, IAudioClient,
        IAudioRenderClient, IMMDevice, IMMDeviceEnumerator,
        PROCESS_LOOPBACK_MODE_INCLUDE_TARGET_PROCESS_TREE, VIRTUAL_AUDIO_DEVICE_PROCESS_LOOPBACK,
        WAVE_FORMAT_PCM, WAVEFORMATEX, WAVEFORMATEXTENSIBLE, eRender,
    };
    use windows::Win32::Media::KernelStreaming::WAVE_FORMAT_EXTENSIBLE;
    use windows::Win32::Media::Multimedia::KSDATAFORMAT_SUBTYPE_IEEE_FLOAT;
    use windows::Win32::System::Com::StructuredStorage::{
        PROPVARIANT, PropVariantClear, PropVariantToGUID, PropVariantToString,
    };
    use windows::Win32::System::Com::{
        CLSCTX_ALL, COINIT_MULTITHREADED, CoCreateInstance, CoInitializeEx, CoTaskMemFree,
        CoUninitialize, STGM_READ,
    };
    use windows::core::{GUID, HRESULT, IUnknown, Interface, PWSTR, Ref, implement};

    use super::super::{SharedJoinHandle, worker_coordinator};
    use super::{
        AUDIO_SHUTDOWN, AudioHapticsControl, ContainerId, EndpointObservation, EndpointResolution,
        GLOBAL_SUBMIT_GATE, INITIAL_BACKOFF_SECS, RetryBackoff, SubmitGate, TactileSynth,
        next_backoff_secs, prune_draining_workers, resolve_unique_dualsense_endpoint,
        target_matches_active,
    };

    const CLSID_MMDEVICE_ENUMERATOR: GUID = GUID::from_u128(0xbcde0395_e52f_467c_8e3d_c4579291692e);
    const RATE: u32 = 48_000;
    const ENDPOINT_VERIFY_INTERVAL: Duration = Duration::from_secs(1);
    const AUDIO_STOP_TIMEOUT: Duration = Duration::from_millis(50);

    pub(super) struct WorkerSlot {
        handle: Option<SharedJoinHandle>,
        stop_flag: Option<Arc<AtomicBool>>,
        submit_gate: Option<Arc<SubmitGate>>,
        /// Previous workers whose `stop()` timed out before their OS thread
        /// exited or `SubmitGate` drained (`in_flight == 0`). `ensure_started`
        /// refuses to spawn a replacement until `draining` is empty.
        draining: Vec<(SharedJoinHandle, Arc<SubmitGate>)>,
        retry: Arc<Mutex<RetryBackoff>>,
    }

    impl Default for WorkerSlot {
        fn default() -> Self {
            Self {
                handle: None,
                stop_flag: None,
                submit_gate: None,
                draining: Vec::new(),
                retry: Arc::new(Mutex::new(RetryBackoff::default())),
            }
        }
    }

    /// Balances `CoInitializeEx` with `CoUninitialize` on the same worker thread.
    struct ComGuard(bool);

    impl ComGuard {
        fn init() -> Self {
            // SAFETY: initializes multithreaded COM for this dedicated audio thread.
            let hr = unsafe { CoInitializeEx(None, COINIT_MULTITHREADED) };
            // Both S_OK and S_FALSE increment the thread's COM reference count.
            Self(hr.is_ok())
        }
    }

    impl Drop for ComGuard {
        fn drop(&mut self) {
            if self.0 {
                // SAFETY: matches the successful `CoInitializeEx` on this thread;
                // declared before all COM locals so interfaces release first.
                unsafe { CoUninitialize() };
            }
        }
    }

    /// Frees the `PROPVARIANT` returned by `IPropertyStore::GetValue` on drop.
    struct PropVarGuard(PROPVARIANT);

    impl Drop for PropVarGuard {
        fn drop(&mut self) {
            // SAFETY: clears a valid initialized `PROPVARIANT` owned by this guard.
            unsafe {
                let _ = PropVariantClear(&mut self.0);
            }
        }
    }

    /// Clears `control.active` on worker exit ONLY if this worker's generation
    /// token is still current (so a stale timed-out worker cannot clobber a
    /// newer worker's `active` state).
    struct ActiveGuard<'a> {
        control: &'a AudioHapticsControl,
        generation: u64,
    }

    impl Drop for ActiveGuard<'_> {
        fn drop(&mut self) {
            let _ = self.control.set_active_if_current(self.generation, false);
        }
    }

    #[implement(IActivateAudioInterfaceCompletionHandler)]
    struct LoopbackActivated;

    impl IActivateAudioInterfaceCompletionHandler_Impl for LoopbackActivated_Impl {
        fn ActivateCompleted(
            &self,
            _: Ref<IActivateAudioInterfaceAsyncOperation>,
        ) -> windows::core::Result<()> {
            Ok(())
        }
    }

    pub(super) fn ensure_started(
        control: &Arc<AudioHapticsControl>,
        slot_lock: &Mutex<WorkerSlot>,
    ) {
        let mut slot = slot_lock.lock().unwrap_or_else(|e| e.into_inner());
        // Non-blockingly reap any previously timed-out workers; refuse to spawn
        // a replacement audio worker until every old worker has exited at the OS
        // level AND drained its `SubmitGate` (`in_flight == 0`).
        if !prune_draining_workers(&mut slot.draining) {
            return;
        }
        if let Some(handle) = &slot.handle {
            let gate_clear = slot.submit_gate.as_ref().is_none_or(|g| g.in_flight() == 0);
            if !handle.is_os_finished() || !gate_clear {
                return;
            }
            if let Some(done) = slot.handle.take() {
                let _ = done.try_join_if_os_finished();
            }
            slot.stop_flag = None;
            slot.submit_gate = None;
        }
        let current_target = control.target_container();
        {
            let mut retry = slot.retry.lock().unwrap_or_else(|e| e.into_inner());
            if retry.should_defer(current_target, Instant::now()) {
                return;
            }
        }

        let Some(ticket) = worker_coordinator().begin_spawn() else {
            return;
        };

        let generation = control.claim_generation();
        let stop_flag = Arc::new(AtomicBool::new(false));
        let worker_stop = Arc::clone(&stop_flag);
        let submit_gate = Arc::new(SubmitGate::new());
        let worker_gate = Arc::clone(&submit_gate);
        let ctrl = Arc::clone(control);
        let retry = Arc::clone(&slot.retry);
        let spawned = ticket.spawn("victory dualsense haptics".into(), move |worker_reg| {
            // Declared first so `worker_reg` drops last (even on panic),
            // notifying `shutdown_and_join` only after `client.Stop()` finishes.
            let _worker_reg = worker_reg;
            let attempted_container = ctrl.target_container();
            let mut stream_started = false;
            let mut target_changed = false;
            let res = run_worker(
                &ctrl,
                generation,
                &worker_stop,
                &worker_gate,
                &mut stream_started,
                &mut target_changed,
            );
            if let Err(e) = res {
                tracing::debug!("DualSense 4ch audio haptics inactive: {e}");
            }
            if !worker_stop.load(Ordering::Acquire) && !AUDIO_SHUTDOWN.load(Ordering::Acquire) {
                let mut r = retry.lock().unwrap_or_else(|e| e.into_inner());
                r.record_exit(
                    attempted_container,
                    ctrl.target_container(),
                    stream_started,
                    target_changed,
                    Instant::now(),
                );
            }
        });
        if let Some(handle) = spawned {
            slot.handle = Some(handle);
            slot.stop_flag = Some(stop_flag);
            slot.submit_gate = Some(submit_gate);
        }
    }

    pub(super) fn stop(control: &AudioHapticsControl, slot_lock: &Mutex<WorkerSlot>) {
        let deadline = Instant::now() + AUDIO_STOP_TIMEOUT;
        // Advance generation immediately so even a wedged/detached worker can
        // never flip `active` back on after `stop()` returns.
        let _ = control.claim_generation();
        let (handle, gate) = {
            let mut slot = slot_lock.lock().unwrap_or_else(|e| e.into_inner());
            if let Some(flag) = slot.stop_flag.take() {
                flag.store(true, Ordering::SeqCst);
            }
            {
                let mut r = slot.retry.lock().unwrap_or_else(|e| e.into_inner());
                r.clear();
            }
            (slot.handle.take(), slot.submit_gate.take())
        };
        // Synchronize with any in-flight `ReleaseBuffer` within our bounded
        // 50 ms budget so no non-zero buffer can be submitted after `stop()`.
        let gate_drained = gate.as_ref().is_none_or(|g| g.close_bounded(deadline));
        // Bounded OS-thread join so a wedged WASAPI driver never freezes polling.
        let thread_joined = handle.as_ref().is_none_or(|h| h.join_bounded(deadline));
        // If `stop()` timed out before the old worker drained AND exited, retain
        // its `(handle, gate)` in `draining` so `ensure_started()` refuses to
        // spawn a second concurrent audio worker until it finishes cleanly.
        if (!gate_drained || !thread_joined)
            && let (Some(h), Some(g)) = (handle, gate)
        {
            let mut slot = slot_lock.lock().unwrap_or_else(|e| e.into_inner());
            slot.draining.push((h, g));
        }
    }

    /// Frees a COM task-allocated UTF-16 string returned by `IMMDevice::GetId`.
    struct CoTaskMemGuard(PWSTR);

    impl Drop for CoTaskMemGuard {
        fn drop(&mut self) {
            if !self.0.is_null() {
                // SAFETY: frees valid COM task memory allocated by `IMMDevice::GetId`.
                unsafe { CoTaskMemFree(Some(self.0.0.cast())) };
            }
        }
    }

    fn find_dualsense_endpoint_with(
        enumerator: &IMMDeviceEnumerator,
        target_container: Option<ContainerId>,
    ) -> windows::core::Result<EndpointResolution<IMMDevice>> {
        // SAFETY: COM initialized on this worker thread before calling here.
        unsafe {
            let collection = enumerator.EnumAudioEndpoints(eRender, DEVICE_STATE_ACTIVE)?;
            let count = collection.GetCount()?;
            let mut observations = Vec::with_capacity(count as usize);
            for i in 0..count {
                let Ok(device) = collection.Item(i) else {
                    observations.push(EndpointObservation::Unreadable);
                    continue;
                };
                let Ok(raw_id) = device.GetId() else {
                    observations.push(EndpointObservation::Unreadable);
                    continue;
                };
                let id_guard = CoTaskMemGuard(raw_id);
                let Ok(id) = id_guard.0.to_string() else {
                    observations.push(EndpointObservation::Unreadable);
                    continue;
                };
                let Ok(store) = device.OpenPropertyStore(STGM_READ) else {
                    observations.push(EndpointObservation::Unreadable);
                    continue;
                };
                let Ok(prop) = store.GetValue(&PKEY_Device_FriendlyName) else {
                    observations.push(EndpointObservation::Unreadable);
                    continue;
                };
                let prop = PropVarGuard(prop);
                let mut chars = [0u16; 256];
                if PropVariantToString(&prop.0, &mut chars).is_err() {
                    observations.push(EndpointObservation::Unreadable);
                    continue;
                }
                let end = chars.iter().position(|&c| c == 0).unwrap_or(chars.len());
                let name_lower = String::from_utf16_lossy(&chars[..end]).to_ascii_lowercase();
                if name_lower.is_empty() {
                    observations.push(EndpointObservation::Unreadable);
                    continue;
                }
                let container_id = match store.GetValue(&PKEY_Device_ContainerId) {
                    Ok(cid_prop) => {
                        let cid_guard = PropVarGuard(cid_prop);
                        PropVariantToGUID(&cid_guard.0).ok().and_then(|g| {
                            ContainerId::from_guid_fields(g.data1, g.data2, g.data3, g.data4)
                        })
                    }
                    Err(_) => None,
                };
                observations.push(EndpointObservation::Named {
                    id,
                    name_lower,
                    container_id,
                    device,
                });
            }
            let resolution = resolve_unique_dualsense_endpoint(target_container, observations);
            static AMBIGUOUS_LOGGED: AtomicBool = AtomicBool::new(false);
            match &resolution {
                EndpointResolution::None | EndpointResolution::Unique { .. } => {
                    AMBIGUOUS_LOGGED.store(false, Ordering::Relaxed);
                }
                EndpointResolution::Ambiguous(reason) => {
                    if !AMBIGUOUS_LOGGED.swap(true, Ordering::Relaxed) {
                        tracing::warn!(
                            reason,
                            "refusing ambiguous 4ch DualSense audio routing (use haptics_mode = \"hybrid\" for automatic HID rumble fallback)"
                        );
                    }
                }
            }
            Ok(resolution)
        }
    }

    fn open_process_loopback(stop_flag: &AtomicBool) -> Option<(IAudioClient, IAudioCaptureClient)> {
        // SAFETY: activates a process-scoped loopback capture client on our own PID.
        unsafe {
            let params = AUDIOCLIENT_ACTIVATION_PARAMS {
                ActivationType: AUDIOCLIENT_ACTIVATION_TYPE_PROCESS_LOOPBACK,
                Anonymous: AUDIOCLIENT_ACTIVATION_PARAMS_0 {
                    ProcessLoopbackParams: AUDIOCLIENT_PROCESS_LOOPBACK_PARAMS {
                        TargetProcessId: std::process::id(),
                        ProcessLoopbackMode: PROCESS_LOOPBACK_MODE_INCLUDE_TARGET_PROCESS_TREE,
                    },
                },
            };
            #[repr(C)]
            struct Blob {
                vt: u16,
                reserved: [u16; 3],
                size: u32,
                data: *const AUDIOCLIENT_ACTIVATION_PARAMS,
            }
            let blob = Blob {
                vt: 0x41,
                reserved: [0; 3],
                size: size_of_val(&params) as u32,
                data: &params,
            };
            let handler: IActivateAudioInterfaceCompletionHandler = LoopbackActivated.into();
            let op = ActivateAudioInterfaceAsync(
                VIRTUAL_AUDIO_DEVICE_PROCESS_LOOPBACK,
                &IAudioClient::IID,
                Some((&raw const blob).cast()),
                &handler,
            )
            .ok()?;
            let mut client = None;
            for _ in 0..50 {
                if stop_flag.load(Ordering::Acquire) || AUDIO_SHUTDOWN.load(Ordering::Acquire) {
                    return None;
                }
                let (mut hr, mut unk) = (HRESULT(0), None::<IUnknown>);
                if op.GetActivateResult(&mut hr, &mut unk).is_ok() && hr.is_ok() {
                    client = unk.and_then(|u| u.cast::<IAudioClient>().ok());
                    break;
                }
                std::thread::sleep(Duration::from_millis(5));
            }
            let client = client?;
            let fmt = WAVEFORMATEX {
                wFormatTag: WAVE_FORMAT_PCM as u16,
                nChannels: 2,
                nSamplesPerSec: RATE,
                nAvgBytesPerSec: RATE * 4,
                nBlockAlign: 4,
                wBitsPerSample: 16,
                cbSize: 0,
            };
            client
                .Initialize(
                    AUDCLNT_SHAREMODE_SHARED,
                    AUDCLNT_STREAMFLAGS_LOOPBACK | AUDCLNT_STREAMFLAGS_AUTOCONVERTPCM,
                    400_000, // 40 ms
                    0,
                    &fmt,
                    None,
                )
                .ok()?;
            let capture: IAudioCaptureClient = client.GetService().ok()?;
            client.Start().ok()?;
            Some((client, capture))
        }
    }

    fn run_worker(
        control: &AudioHapticsControl,
        generation: u64,
        stop_flag: &AtomicBool,
        worker_gate: &SubmitGate,
        stream_started: &mut bool,
        target_changed: &mut bool,
    ) -> Result<(), Box<dyn std::error::Error>> {
        // Declared first so all COM interfaces drop before `CoUninitialize`.
        let _com = ComGuard::init();

        // SAFETY: COM is initialized on this dedicated audio worker thread.
        unsafe {
            let Some(active_container) = control.target_container() else {
                return Err("HID controller ContainerId unavailable".into());
            };
            let enumerator: IMMDeviceEnumerator =
                CoCreateInstance(&CLSID_MMDEVICE_ENUMERATOR, None, CLSCTX_ALL)?;
            let (endpoint_id, device) =
                match find_dualsense_endpoint_with(&enumerator, Some(active_container))? {
                    EndpointResolution::Unique { id, name, device } => {
                        tracing::info!(
                            endpoint = %name,
                            "found ContainerId-verified DualSense 4ch audio endpoint"
                        );
                        (id, device)
                    }
                    EndpointResolution::None => {
                        if !target_matches_active(control.target_container(), active_container) {
                            *target_changed = true;
                        }
                        return Err("no matching DualSense USB audio endpoint found".into());
                    }
                    EndpointResolution::Ambiguous(reason) => {
                        if !target_matches_active(control.target_container(), active_container) {
                            *target_changed = true;
                        }
                        return Err(reason.into());
                    }
                };
            let client: IAudioClient = device.Activate(CLSCTX_ALL, None)?;
            // Quadraphonic 48 kHz 32-bit float (`FL | FR | BL | BR` = 0x33).
            let quad = WAVEFORMATEXTENSIBLE {
                Format: WAVEFORMATEX {
                    wFormatTag: WAVE_FORMAT_EXTENSIBLE as u16,
                    nChannels: 4,
                    nSamplesPerSec: RATE,
                    nAvgBytesPerSec: RATE * 16,
                    nBlockAlign: 16,
                    wBitsPerSample: 32,
                    cbSize: 22,
                },
                Samples: windows::Win32::Media::Audio::WAVEFORMATEXTENSIBLE_0 {
                    wValidBitsPerSample: 32,
                },
                dwChannelMask: 0x33,
                SubFormat: KSDATAFORMAT_SUBTYPE_IEEE_FLOAT,
            };
            client.Initialize(
                AUDCLNT_SHAREMODE_SHARED,
                AUDCLNT_STREAMFLAGS_AUTOCONVERTPCM,
                300_000, // 30 ms shared buffer for low latency
                0,
                &quad.Format,
                None,
            )?;
            let buffer_frames = client.GetBufferSize()?;
            let render: IAudioRenderClient = client.GetService()?;
            let mut loopback = open_process_loopback(stop_flag);
            let mut lb_backoff_secs: u64 = 0;
            let mut next_loopback_retry: Option<Instant> = if loopback.is_none() {
                lb_backoff_secs = INITIAL_BACKOFF_SECS;
                Some(Instant::now() + Duration::from_secs(lb_backoff_secs))
            } else {
                None
            };
            if !control.worker_binding_valid(generation, active_container) {
                *target_changed = true;
                if let Some((lb_client, _)) = loopback {
                    let _ = lb_client.Stop();
                }
                return Ok(());
            }
            if stop_flag.load(Ordering::Acquire) || AUDIO_SHUTDOWN.load(Ordering::Acquire) {
                if let Some((lb_client, _)) = loopback {
                    let _ = lb_client.Stop();
                }
                return Ok(());
            }
            client.Start()?;
            *stream_started = true;
            if !control.set_active_if_current(generation, true) {
                // A newer generation superseded this worker during startup.
                let _ = client.Stop();
                let _ = client.Reset();
                if let Some((lb_client, _)) = loopback {
                    let _ = lb_client.Stop();
                }
                return Ok(());
            }
            let _active = ActiveGuard {
                control,
                generation,
            };

            let mut synth = TactileSynth::default();
            let mut loop_ring: VecDeque<[f32; 2]> = VecDeque::with_capacity(4096);
            let mut next_endpoint_verify = Instant::now() + ENDPOINT_VERIFY_INTERVAL;

            let mut loop_err: Option<windows::core::Error> = None;
            while !stop_flag.load(Ordering::Acquire) && !AUDIO_SHUTDOWN.load(Ordering::Acquire) {
                // Verify generation + target ContainerId on EVERY iteration BEFORE
                // generating or submitting samples (prevents both A -> B routing
                // and A -> B -> A ABA stale-worker survival).
                if !control.worker_binding_valid(generation, active_container) {
                    *target_changed = true;
                    tracing::info!(
                        "DualSense HID ContainerId target or worker generation changed; silencing and stopping 4ch stream"
                    );
                    break;
                }
                if Instant::now() >= next_endpoint_verify {
                    next_endpoint_verify = Instant::now() + ENDPOINT_VERIFY_INTERVAL;
                    match find_dualsense_endpoint_with(&enumerator, Some(active_container)) {
                        Ok(EndpointResolution::Unique { id, .. }) if id == endpoint_id => {}
                        Ok(_) => {
                            tracing::info!(
                                "DualSense 4ch audio endpoint is no longer uniquely matched to ContainerId; silencing and stopping stream"
                            );
                            break;
                        }
                        Err(e) => {
                            loop_err = Some(e);
                            break;
                        }
                    }
                }

                if loopback.is_none()
                    && next_loopback_retry.is_some_and(|at| Instant::now() >= at)
                {
                    if let Some(lb) = open_process_loopback(stop_flag) {
                        tracing::debug!("DualSense 4ch process loopback capture connected");
                        loopback = Some(lb);
                        lb_backoff_secs = 0;
                        next_loopback_retry = None;
                    } else {
                        lb_backoff_secs = next_backoff_secs(lb_backoff_secs);
                        next_loopback_retry =
                            Some(Instant::now() + Duration::from_secs(lb_backoff_secs));
                    }
                }

                let muted = control.muted.load(Ordering::Acquire);
                let mut loopback_broken = false;
                if let Some((_, ref capture)) = loopback {
                    loop {
                        match capture.GetNextPacketSize() {
                            Ok(0) => break,
                            Ok(_) => {
                                let (mut ptr, mut n, mut flags) = (std::ptr::null_mut(), 0u32, 0u32);
                                if capture
                                    .GetBuffer(&mut ptr, &mut n, &mut flags, None, None)
                                    .is_ok()
                                {
                                    if !muted {
                                        let silent =
                                            flags & AUDCLNT_BUFFERFLAGS_SILENT.0 as u32 != 0;
                                        if silent || ptr.is_null() {
                                            for _ in 0..n {
                                                if loop_ring.len() < 4096 {
                                                    loop_ring.push_back([0.0, 0.0]);
                                                }
                                            }
                                        } else {
                                            let samples = std::slice::from_raw_parts(
                                                ptr.cast::<i16>(),
                                                n as usize * 2,
                                            );
                                            for pair in samples.as_chunks::<2>().0 {
                                                if loop_ring.len() == 4096 {
                                                    loop_ring.pop_front();
                                                }
                                                loop_ring.push_back([
                                                    f32::from(pair[0]) / 32768.0,
                                                    f32::from(pair[1]) / 32768.0,
                                                ]);
                                            }
                                        }
                                    }
                                    if capture.ReleaseBuffer(n).is_err() {
                                        loopback_broken = true;
                                        break;
                                    }
                                } else {
                                    loopback_broken = true;
                                    break;
                                }
                            }
                            Err(_) => {
                                loopback_broken = true;
                                break;
                            }
                        }
                    }
                }
                if loopback_broken {
                    if let Some((lb_client, _)) = loopback.take() {
                        let _ = lb_client.Stop();
                    }
                    loop_ring.clear();
                    lb_backoff_secs = INITIAL_BACKOFF_SECS;
                    next_loopback_retry =
                        Some(Instant::now() + Duration::from_secs(lb_backoff_secs));
                }
                if muted {
                    loop_ring.clear();
                }

                let padding = match client.GetCurrentPadding() {
                    Ok(p) => p,
                    Err(e) => {
                        loop_err = Some(e);
                        break;
                    }
                };
                let avail = buffer_frames.saturating_sub(padding);
                if avail > 0 {
                    // Synchronize `GetBuffer -> fill -> ReleaseBuffer` with
                    // `shutdown_global_bounded` and `stop()` via `SubmitGate`.
                    let Some(_global_submit) = GLOBAL_SUBMIT_GATE.try_begin() else {
                        break;
                    };
                    let Some(_worker_submit) = worker_gate.try_begin() else {
                        break;
                    };
                    if stop_flag.load(Ordering::Acquire) || AUDIO_SHUTDOWN.load(Ordering::Acquire) {
                        break;
                    }
                    if !control.worker_binding_valid(generation, active_container) {
                        *target_changed = true;
                        break;
                    }
                    let dst = match render.GetBuffer(avail) {
                        Ok(ptr) => ptr,
                        Err(e) => {
                            loop_err = Some(e);
                            break;
                        }
                    };
                    let out = std::slice::from_raw_parts_mut(dst.cast::<f32>(), avail as usize * 4);
                    let levels = control.levels();
                    let mirror = !muted && control.speaker_mirror.load(Ordering::Acquire);
                    for chunk in out.as_chunks_mut::<4>().0 {
                        let game = if muted {
                            [0.0, 0.0]
                        } else {
                            loop_ring.pop_front().unwrap_or([0.0, 0.0])
                        };
                        let frame = synth.next_frame(RATE as f32, levels, game, mirror, muted);
                        chunk.copy_from_slice(&frame);
                    }
                    // Final check after reading levels and before `ReleaseBuffer`:
                    // if shutdown/stop, controller ContainerId switch, or A->B->A
                    // generation change arrived mid-fill, submit explicit silence
                    // and exit immediately.
                    let binding_invalid =
                        !control.worker_binding_valid(generation, active_container);
                    let abort_now = binding_invalid
                        || stop_flag.load(Ordering::Acquire)
                        || AUDIO_SHUTDOWN.load(Ordering::Acquire);
                    let release_flags = if abort_now || muted {
                        out.fill(0.0);
                        AUDCLNT_BUFFERFLAGS_SILENT.0 as u32
                    } else {
                        0
                    };
                    if let Err(e) = render.ReleaseBuffer(avail, release_flags) {
                        loop_err = Some(e);
                        break;
                    }
                    if abort_now {
                        if binding_invalid {
                            *target_changed = true;
                        }
                        break;
                    }
                }
                std::thread::sleep(Duration::from_millis(4));
            }

            let _ = client.Stop();
            let _ = client.Reset();
            if let Some((lb_client, _)) = loopback {
                let _ = lb_client.Stop();
            }
            if let Some(e) = loop_err {
                return Err(e.into());
            }
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn synth_keeps_speaker_silent_by_default_and_modulates_voice_coils() {
        let mut synth = TactileSynth::default();
        let mut peak_haptic = 0.0f32;
        for _ in 0..480 {
            let f = synth.next_frame(48_000.0, [0.8, 0.5, 0.4, 0.0], [0.5, -0.5], false, false);
            assert_eq!(
                (f[0], f[1]),
                (0.0, 0.0),
                "Ch1/2 must stay silent when speaker_mirror=false"
            );
            peak_haptic = peak_haptic.max(f[2].abs()).max(f[3].abs());
        }
        assert!(peak_haptic > 0.2 && peak_haptic <= 0.92);

        // Muting silences all 4 channels (including speaker_mirror & loopback) and clears filter state.
        let muted = synth.next_frame(48_000.0, [1.0, 1.0, 1.0, 1.0], [1.0, 1.0], true, true);
        assert_eq!(muted, [0.0; 4]);
        let after_unmute = synth.next_frame(48_000.0, [0.0; 4], [0.0, 0.0], true, false);
        assert_eq!(
            after_unmute,
            [0.0; 4],
            "unmuting with silent input must not leak stale pre-mute loopback filter energy"
        );
    }

    #[test]
    fn endpoint_resolver_matches_exact_container_id_and_fails_closed() {
        let cid_a = ContainerId::from_guid_fields(0x1111_2222, 0x3333, 0x4444, [1, 2, 3, 4, 5, 6, 7, 8])
            .unwrap();
        let cid_b = ContainerId::from_guid_fields(0x9999_8888, 0x7777, 0x6666, [8, 7, 6, 5, 4, 3, 2, 1])
            .unwrap();
        let cid_c = ContainerId::from_guid_fields(0xAAAA_BBBB, 0xCCCC, 0xDDDD, [9; 8]).unwrap();

        // All-zero GUID is rejected as invalid, and raw Win32 GUID bytes match field encoding.
        assert!(ContainerId::from_guid_fields(0, 0, 0, [0; 8]).is_none());
        assert!(ContainerId::from_raw_guid_bytes(&[0u8; 16]).is_none());
        let mut raw_a = [0u8; 16];
        raw_a[0..4].copy_from_slice(&0x1111_2222u32.to_le_bytes());
        raw_a[4..6].copy_from_slice(&0x3333u16.to_le_bytes());
        raw_a[6..8].copy_from_slice(&0x4444u16.to_le_bytes());
        raw_a[8..16].copy_from_slice(&[1, 2, 3, 4, 5, 6, 7, 8]);
        assert_eq!(ContainerId::from_raw_guid_bytes(&raw_a), Some(cid_a));

        let ds_a = EndpointObservation::Named {
            id: "ep-a".into(),
            name_lower: "speakers (wireless controller)".into(),
            container_id: Some(cid_a),
            device: 1u32,
        };
        let ds_b = EndpointObservation::Named {
            id: "ep-b".into(),
            name_lower: "dualsense wireless controller".into(),
            container_id: Some(cid_b),
            device: 2u32,
        };
        let ds_no_cid = EndpointObservation::Named {
            id: "ep-nocid".into(),
            name_lower: "wireless controller".into(),
            container_id: None,
            device: 3u32,
        };
        let speaker = EndpointObservation::Named {
            id: "ep-spk".into(),
            name_lower: "speakers (realtek high definition audio)".into(),
            container_id: Some(cid_c),
            device: 4u32,
        };

        // Exact ContainerId match succeeds even when another DualSense audio endpoint
        // exists with a different ContainerId!
        assert_eq!(
            resolve_unique_dualsense_endpoint(Some(cid_a), [speaker.clone(), ds_b.clone(), ds_a.clone()]),
            EndpointResolution::Unique {
                id: "ep-a".into(),
                name: "speakers (wireless controller)".into(),
                device: 1u32,
            }
        );

        // Mismatched ContainerId returns None (never routes to a different physical controller).
        assert_eq!(
            resolve_unique_dualsense_endpoint(Some(cid_a), [speaker.clone(), ds_b.clone()]),
            EndpointResolution::None
        );

        // Missing HID ContainerId fails closed.
        assert!(matches!(
            resolve_unique_dualsense_endpoint(None, [ds_a.clone()]),
            EndpointResolution::Ambiguous(_)
        ));

        // DualSense-named endpoint with missing ContainerId fails closed.
        assert!(matches!(
            resolve_unique_dualsense_endpoint(Some(cid_a), [ds_a.clone(), ds_no_cid]),
            EndpointResolution::Ambiguous(_)
        ));

        // Unreadable active endpoint alongside a matching DualSense MUST fail closed.
        assert!(matches!(
            resolve_unique_dualsense_endpoint(Some(cid_a), [ds_a, EndpointObservation::Unreadable]),
            EndpointResolution::Ambiguous(_)
        ));

        // Only unrelated readable endpoints -> None.
        assert_eq!(
            resolve_unique_dualsense_endpoint(Some(cid_a), [speaker]),
            EndpointResolution::None
        );
    }

    #[test]
    fn target_container_switch_a_to_b_rejects_emission_on_very_next_check() {
        let cid_a =
            ContainerId::from_guid_fields(0x1111_2222, 0x3333, 0x4444, [1, 2, 3, 4, 5, 6, 7, 8])
                .unwrap();
        let cid_b =
            ContainerId::from_guid_fields(0x9999_8888, 0x7777, 0x6666, [8, 7, 6, 5, 4, 3, 2, 1])
                .unwrap();

        let ctrl = AudioHapticsControl::default();
        ctrl.set_target_container(Some(cid_a));
        ctrl.set_levels(0.7, 0.5, 0.0, 0.0);
        assert!(target_matches_active(ctrl.target_container(), cid_a));
        assert_eq!(ctrl.levels(), [0.7, 0.5, 0.0, 0.0]);

        // Switching A -> B immediately invalidates A on the very next iteration check.
        ctrl.set_target_container(Some(cid_b));
        assert!(
            !target_matches_active(ctrl.target_container(), cid_a),
            "audio worker bound to A must stop immediately on A -> B switch"
        );
        assert!(target_matches_active(ctrl.target_container(), cid_b));

        // Disconnecting (B -> None) immediately invalidates B as well.
        ctrl.set_target_container(None);
        assert!(!target_matches_active(ctrl.target_container(), cid_b));
    }

    #[test]
    fn submit_gate_synchronizes_in_flight_release_buffer_and_bounds_wedged_driver() {
        use std::sync::Barrier;

        // 1. Normal path: `close_bounded` waits until an in-flight submit finishes,
        // then rejects every subsequent `try_begin`.
        let gate = Arc::new(SubmitGate::new());
        let entered = Arc::new(Barrier::new(2));
        let release = Arc::new(Barrier::new(2));
        let finished = Arc::new(AtomicBool::new(false));

        let (g_w, e_w, r_w, f_w) = (
            Arc::clone(&gate),
            Arc::clone(&entered),
            Arc::clone(&release),
            Arc::clone(&finished),
        );
        let worker = std::thread::spawn(move || {
            let _guard = g_w.try_begin().expect("gate open before close");
            e_w.wait();
            r_w.wait();
            std::thread::sleep(Duration::from_millis(15));
            f_w.store(true, Ordering::SeqCst);
        });

        entered.wait();
        let g_c = Arc::clone(&gate);
        let f_c = Arc::clone(&finished);
        let closer = std::thread::spawn(move || {
            let clean = g_c.close_bounded(Instant::now() + Duration::from_millis(500));
            assert!(clean);
            assert!(
                f_c.load(Ordering::SeqCst),
                "close_bounded must not return before in-flight ReleaseBuffer finishes"
            );
        });
        release.wait();
        closer.join().unwrap();
        worker.join().unwrap();
        assert!(
            gate.try_begin().is_none(),
            "post-close try_begin must always fail"
        );

        // 2. Overlapping workers: `in_flight: usize` stays non-zero until ALL
        // overlapping guards drop (a newer worker finishing cannot mask an older one).
        {
            let overlap = Arc::new(SubmitGate::new());
            let g1 = overlap.try_begin().expect("first guard");
            let g2 = overlap.try_begin().expect("second overlapping guard");
            drop(g2);
            let start = Instant::now();
            assert!(
                !overlap.close_bounded(start + Duration::from_millis(20)),
                "gate must stay in-flight while g1 is still held after g2 dropped"
            );
            drop(g1);
            assert!(overlap.close_bounded(Instant::now() + Duration::from_millis(100)));
        }

        // 3. Wedged WASAPI path: `close_bounded` returns false promptly at deadline.
        let wedged = SubmitGate::new();
        let _stuck_guard = wedged.try_begin().expect("gate open");
        let start = Instant::now();
        let clean = wedged.close_bounded(start + Duration::from_millis(25));
        assert!(!clean);
        assert!(
            start.elapsed() < Duration::from_millis(200),
            "bounded close must not hang on a wedged WASAPI submission"
        );
        assert!(wedged.try_begin().is_none());
    }

    #[test]
    fn active_generation_token_prevents_stale_worker_from_clobbering_newer_worker() {
        let ctrl = AudioHapticsControl::default();
        assert!(!ctrl.is_active());

        let gen1 = ctrl.claim_generation();
        assert!(ctrl.set_active_if_current(gen1, true));
        assert!(ctrl.is_active());

        // A newer worker supersedes gen1 (e.g. after a bounded stop() timeout).
        let gen2 = ctrl.claim_generation();
        assert!(gen2 > gen1);
        assert!(!ctrl.is_active(), "claiming a new generation clears active");
        assert!(ctrl.set_active_if_current(gen2, true));
        assert!(ctrl.is_active());

        // Stale gen1 worker exiting or trying to set/clear active MUST be ignored.
        assert!(!ctrl.set_active_if_current(gen1, false));
        assert!(
            ctrl.is_active(),
            "stale gen1 ActiveGuard drop must not clobber gen2 active flag"
        );
        assert!(!ctrl.set_active_if_current(gen1, true));

        // Current gen2 worker clears active cleanly on exit.
        assert!(ctrl.set_active_if_current(gen2, false));
        assert!(!ctrl.is_active());

        // Switching target ContainerId A -> B immediately clears `active` set by A's
        // worker (even if A's thread is wedged) so Hybrid mode falls back to HID rumble.
        let cid_a =
            ContainerId::from_guid_fields(0x1111_2222, 0x3333, 0x4444, [1, 2, 3, 4, 5, 6, 7, 8])
                .unwrap();
        let cid_b =
            ContainerId::from_guid_fields(0x9999_8888, 0x7777, 0x6666, [8, 7, 6, 5, 4, 3, 2, 1])
                .unwrap();
        ctrl.set_target_container(Some(cid_a));
        let gen_a = ctrl.claim_generation();
        assert!(ctrl.set_active_if_current(gen_a, true));
        assert!(ctrl.is_active());

        // Setting the SAME target ContainerId keeps A active.
        ctrl.set_target_container(Some(cid_a));
        assert!(ctrl.is_active());

        // Switching A -> B immediately clears active AND invalidates gen_a.
        ctrl.set_target_container(Some(cid_b));
        assert!(
            !ctrl.is_active(),
            "switching A -> B must immediately clear active set by A's worker"
        );
        assert!(
            !ctrl.set_active_if_current(gen_a, true),
            "stale A worker must not re-enable active after A -> B switch"
        );

        // Switching back A -> B -> A (ABA): even though target_container is A
        // again, stale gen_a MUST fail `is_current_generation` / `worker_binding_valid`.
        ctrl.set_target_container(Some(cid_a));
        assert!(
            !ctrl.is_current_generation(gen_a) && !ctrl.worker_binding_valid(gen_a, cid_a),
            "A -> B -> A switch must keep stale gen_a invalidated so it silences and exits"
        );
        let gen_a2 = ctrl.claim_generation();
        assert!(ctrl.worker_binding_valid(gen_a2, cid_a));
    }

    #[test]
    fn retry_backoff_is_scoped_per_container_id_and_clears_on_switch() {
        let cid_a =
            ContainerId::from_guid_fields(0x1111_2222, 0x3333, 0x4444, [1, 2, 3, 4, 5, 6, 7, 8])
                .unwrap();
        let cid_b =
            ContainerId::from_guid_fields(0x9999_8888, 0x7777, 0x6666, [8, 7, 6, 5, 4, 3, 2, 1])
                .unwrap();
        let t0 = Instant::now();
        let mut backoff = RetryBackoff::default();

        // Repeated failures on controller A build up to 10 s backoff.
        for _ in 0..4 {
            backoff.record_exit(Some(cid_a), Some(cid_a), false, false, t0);
        }
        assert!(backoff.should_defer(Some(cid_a), t0 + Duration::from_secs(5)));

        // Switching to controller B immediately clears A's backoff with zero wait.
        assert!(
            !backoff.should_defer(Some(cid_b), t0 + Duration::from_millis(1)),
            "backoff from controller A must not delay newly switched controller B"
        );
        assert!(!backoff.should_defer(Some(cid_a), t0 + Duration::from_millis(2)));
    }

    #[test]
    fn timed_out_stop_blocks_new_worker_spawn_until_draining_worker_finishes() {
        use std::sync::Barrier;

        let gate = Arc::new(SubmitGate::new());
        let entered = Arc::new(Barrier::new(2));
        let release = Arc::new(Barrier::new(2));

        let (g_w, e_w, r_w) = (Arc::clone(&gate), Arc::clone(&entered), Arc::clone(&release));
        let handle = SharedJoinHandle::new(std::thread::spawn(move || {
            let _submit = g_w.try_begin().expect("gate open");
            e_w.wait();
            r_w.wait();
        }));
        entered.wait();

        // Simulate `stop()` timing out within 20 ms while the worker still holds
        // an in-flight `ReleaseBuffer` (`SubmitGuard`).
        let deadline = Instant::now() + Duration::from_millis(20);
        assert!(!gate.close_bounded(deadline));
        assert!(!handle.join_bounded(deadline));

        // Retaining `(handle, gate)` in `draining` MUST block spawning a second
        // concurrent audio worker until the old worker drains AND exits.
        let mut draining = vec![(handle.clone(), Arc::clone(&gate))];
        assert!(
            !prune_draining_workers(&mut draining),
            "must refuse new audio worker spawn while timed-out worker is still draining"
        );

        // Release the old worker and wait for it to exit cleanly; now pruning succeeds.
        release.wait();
        assert!(gate.close_bounded(Instant::now() + Duration::from_millis(250)));
        assert!(handle.join_bounded(Instant::now() + Duration::from_millis(250)));
        assert!(
            prune_draining_workers(&mut draining),
            "once draining worker has finished and in_flight == 0, spawning may resume"
        );
    }
}
