//! What changed on the CPU's side of copies in the card's memory goes up on
//! a copy queue of its own (`VICTORY_UP_QUEUE=1`; off by default; A/B-able
//! with `VICTORY_AB=<s>,VICTORY_UP_QUEUE`).
//!
//! Without it the drawing queue copies the changed ranges from the PC's
//! memory to the card's side before a submission's lists: in a replay of
//! capture replay5, 269 copies and 76.6 MB, about 5 ms in which the drawing
//! queue does nothing else (PCIe-bound). With it the copy queue brings the
//! same bytes into a buffer of ours in the card's memory as soon as the
//! submission is prepared, while the drawing queue still runs earlier
//! submissions; the drawing queue waits for that copy (a fence) and then
//! copies from our buffer to the card's side, in the card's memory (about
//! 0.2 ms for those 76 MB), at the same place in its order as before. So
//! every list of the submission and of later ones finds the new bytes, and
//! no earlier one does: the card's side is still written only by the
//! drawing queue.
//!
//! With the sends of `send` (`VICTORY_SEND_OWN`, on by default) the bytes are
//! in the submission's own memory in the PC's: that comes up whole, in one
//! copy, and the sends (copies and pairs of offset and word) read it from our
//! buffer instead.
//!
//! The copy queue reads the CPU's side when the drawing queue would have,
//! or earlier, and never before the bytes were put there (it is sent after
//! that). Our buffer and lists belong to the submission's plan
//! (`CardPlan::up`) and come back with it when its submission has run
//! (`Host::card_done`), so they are not written again while in use.

use std::collections::HashSet;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering, Ordering::Relaxed};

use windows::Win32::Graphics::Direct3D12::*;

/// Whether it is on now (`VICTORY_UP_QUEUE=1`, or its phase of `VICTORY_AB`).
pub(super) fn on() -> bool {
    #[cfg(feature = "diagnostics")]
    if let Some(on) = crate::ab::pick("VICTORY_UP_QUEUE") {
        return on;
    }
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_UP_QUEUE").is_some_and(|v| v == "1"))
}

/// The most bytes one submission sends this way; more go as before.
const MOST: u64 = 512 << 20;

/// Our buffers are made in sizes of powers of two from this on.
const LEAST: u64 = 4 << 20;

/// The size of a new buffer for `total` bytes, or `None` if the one we
/// have (`have` bytes) holds them.
/// (GPL-3.0 provenance: Belobug/VIctory-Belobug-AMD#401.)
fn bounce_size(have: Option<u64>, total: u64) -> Option<u64> {
    if have.is_some_and(|have| have >= total) {
        return None;
    }
    Some(total.next_power_of_two().max(LEAST))
}

/// Whether a fence check confirms that `target` completed on the copy queue
/// (`u64::MAX` is D3D12's device-removed sentinel).
fn fence_confirmed(wait_ok: bool, completed: u64, target: u64) -> bool {
    wait_ok && completed != u64::MAX && completed >= target
}

/// What to do with a slot after a submission attempt or completion check.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SlotOutcome {
    Recycle,
    RetireUnconfirmed,
}

/// Decides whether a slot may be returned to the idle pool (`Recycle`) or must
/// be kept out of the pool (`RetireUnconfirmed`) so an in-flight command
/// allocator is never `Reset()` or prematurely dropped.
fn slot_outcome(submitted: bool, signal_ok: bool, wait_ok: bool, completed: u64, target: u64) -> SlotOutcome {
    if !submitted || (signal_ok && fence_confirmed(wait_ok, completed, target)) {
        SlotOutcome::Recycle
    } else {
        SlotOutcome::RetireUnconfirmed
    }
}

/// One submission's: a copy list and our buffer in the card's memory.
pub(super) struct Slot {
    allocator: ID3D12CommandAllocator,
    list: ID3D12GraphicsCommandList,
    bounce: Option<(ID3D12Resource, u64)>,
    /// Copy-queue fence value of its last submission (`0`: idle/never sent).
    last_sent: u64,
}

// SAFETY: free-threaded D3D12 objects, used by one submission at a time.
unsafe impl Send for Slot {}

impl Slot {
    /// Our buffer, which holds what went up.
    pub(super) fn bounce(&self) -> Option<&ID3D12Resource> {
        self.bounce.as_ref().map(|b| &b.0)
    }
}

/// Source kind of a range in a submission batch. Only owned immutable
/// `send::Memory` snapshots are eligible for the copy queue; shared
/// `Card::staging` buffers must stay on the drawing queue so an unconfirmed
/// copy-queue submission can never race later CPU writes into live staging
/// memory.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum SourceKind {
    OwnedSendMemory,
    #[cfg_attr(not(test), allow(dead_code))]
    SharedCardStaging,
}

/// Whether a batch may be submitted to `upq`: `VICTORY_SEND_OWN` must be on,
/// `upcheck` must not be active (`looks_none`), and every range in the batch
/// must come from an owned immutable `send::Memory` snapshot (non-empty and
/// containing zero `SharedCardStaging` sources).
pub(super) fn batch_eligible(send_own: bool, looks_none: bool, sources: &[SourceKind]) -> bool {
    send_own
        && looks_none
        && !sources.is_empty()
        && sources.iter().all(|&s| s == SourceKind::OwnedSendMemory)
}

/// Serialized poison flag guarded by `QUEUE`, paired with the lock-free
/// fast-path latch [`UpqState`]. Checked inside `QUEUE` immediately before
/// `ExecuteCommandLists` so a concurrent poison from `give_back` can never race
/// a submission.
struct QueuePoison {
    poisoned: bool,
}

impl QueuePoison {
    const fn new() -> Self {
        Self { poisoned: false }
    }

    /// Marks both the lock-protected flag and the atomic fast-path latch;
    /// returns `true` on the first transition.
    fn mark_poisoned(&mut self, state: &UpqState) -> bool {
        self.poisoned = true;
        state.trip()
    }

    /// Checked inside `QUEUE` immediately before `ExecuteCommandLists`.
    fn allow_submit(&self, state: &UpqState) -> bool {
        !self.poisoned && state.allow_send()
    }
}

struct Queue {
    device: ID3D12Device,
    queue: ID3D12CommandQueue,
    fence: ID3D12Fence,
    sent: u64,
    free: Vec<Slot>,
    poison: QueuePoison,
}

// SAFETY: as `Slot`.
unsafe impl Send for Queue {}

/// `None` inside: none could be made (then nothing goes this way).
static QUEUE: Mutex<Option<Option<Queue>>> = Mutex::new(None);

/// Maximum unconfirmed copy-queue slots kept pinned before intentionally
/// leaking overflow entries on a stuck/broken device.
const UNCONFIRMED_MAX: usize = 64;

/// An unconfirmed copy-queue submission keeping `slot` (with `slot.bounce`,
/// `allocator`, and `list` intact) and every source `ID3D12Resource` read by
/// the copy alive until the copy-queue fence confirms completion or the device
/// is removed. Never handed to graphics-queue `super::retire_resource`/`retire`.
struct UnconfirmedSlot<S = Slot, R = ID3D12Resource, F = ID3D12Fence> {
    #[allow(dead_code)]
    slot: S,
    #[allow(dead_code)]
    pinned_sources: Vec<R>,
    /// `Some((fence, target))` when `Queue::Signal` succeeded; `None` when
    /// `Signal` failed (no copy-queue fence value exists).
    signaled: Option<(F, u64)>,
}

// SAFETY: free-threaded D3D12 COM interfaces.
unsafe impl Send for UnconfirmedSlot {}

struct UnconfirmedPool<S, R, F> {
    entries: Vec<UnconfirmedSlot<S, R, F>>,
}

impl<S, R, F> UnconfirmedPool<S, R, F> {
    const fn new() -> Self {
        Self { entries: Vec::new() }
    }

    /// Removes and returns entries whose copy-queue fence confirmed completion
    /// (`completed != u64::MAX && completed >= target`), or all entries when
    /// `device_removed` is true.
    fn take_ready(
        &mut self,
        device_removed: bool,
        mut completed_of: impl FnMut(&F) -> u64,
    ) -> Vec<UnconfirmedSlot<S, R, F>> {
        let mut ready = Vec::new();
        let mut i = 0;
        while i < self.entries.len() {
            let is_ready = device_removed
                || self.entries[i]
                    .signaled
                    .as_ref()
                    .is_some_and(|(fence, target)| fence_confirmed(true, completed_of(fence), *target));
            if is_ready {
                ready.push(self.entries.swap_remove(i));
            } else {
                i += 1;
            }
        }
        ready
    }

    /// Pins `entry` if it has a copy-queue fence and capacity remains; returns
    /// `Err(entry)` if it must be intentionally leaked (`Signal` failed or
    /// pool is full at `UNCONFIRMED_MAX`).
    fn try_push(&mut self, entry: UnconfirmedSlot<S, R, F>) -> Result<(), UnconfirmedSlot<S, R, F>> {
        if entry.signaled.is_none() || self.entries.len() >= UNCONFIRMED_MAX {
            return Err(entry);
        }
        self.entries.push(entry);
        Ok(())
    }
}

/// Outcome of [`send`], distinguishing a pre-submission fallback from a
/// submission that already executed on the copy queue but could not be
/// confirmed complete (in which case callers must keep source upload memory
/// out of reusable pools).
pub(super) enum Outcome {
    /// Submitted on the copy queue and ordered ahead of the drawing queue.
    Sent(Slot),
    /// Nothing was submitted to the copy queue; safe to fall back and reuse
    /// source memory normally.
    NotSubmitted,
    /// `ExecuteCommandLists` already ran on the copy queue, but completion
    /// could not be confirmed: the copy queue is now poisoned, `slot` +
    /// sources are pinned to the copy-queue fence (or leaked if unsignaled),
    /// and the caller must NOT recycle source upload memory back into a pool.
    SubmittedUnconfirmed,
}

/// Process-wide poison latch for the up-queue copy path. Tripped on the first
/// unconfirmed copy-queue submission failure so all subsequent calls fall back
/// directly to the graphics queue, capping unconfirmed resource retention to
/// the single failing submission(s) already in flight.
struct UpqState {
    poisoned: AtomicBool,
}

impl UpqState {
    const fn new() -> Self {
        Self { poisoned: AtomicBool::new(false) }
    }

    fn is_poisoned(&self) -> bool {
        self.poisoned.load(Ordering::Acquire)
    }

    fn allow_send(&self) -> bool {
        !self.is_poisoned()
    }

    /// Trips poisoned state; returns `true` on the first transition.
    fn trip(&self) -> bool {
        !self.poisoned.swap(true, Ordering::AcqRel)
    }
}

static STATE: UpqState = UpqState::new();

/// Sets `q.poison` and `STATE` while the caller already holds `QUEUE`.
/// Never acquires `UNCONFIRMED`.
fn poison_under_queue(q: &mut Queue, why: &'static str) {
    if q.poison.mark_poisoned(&STATE) {
        tracing::warn!(
            why,
            "disabling up-queue copy path after unconfirmed submission failure; falling back to drawing queue"
        );
    }
}

/// Acquires `QUEUE` briefly to set `q.poison` and `STATE`, then drops `QUEUE`.
/// Must NEVER be called while holding `QUEUE` or `UNCONFIRMED`.
fn poison_queue(why: &'static str) {
    let first = {
        let mut guard = QUEUE.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(Some(q)) = guard.as_mut() {
            q.poison.mark_poisoned(&STATE)
        } else {
            STATE.trip()
        }
    };
    if first {
        tracing::warn!(
            why,
            "disabling up-queue copy path after unconfirmed submission failure; falling back to drawing queue"
        );
    }
}

/// Sweeps any completed/device-removed unconfirmed copy-queue slots (safe to
/// call even when `upq` is bypassed or poisoned).
pub(super) fn sweep(device: &ID3D12Device) {
    sweep_unconfirmed(Some(device));
}

/// Sweeps any completed/device-removed unconfirmed copy-queue slots and
/// reports whether the copy queue is still usable (not poisoned).
pub(super) fn usable(device: &ID3D12Device) -> bool {
    sweep_unconfirmed(Some(device));
    STATE.allow_send()
}

/// Copy-queue slots whose GPU completion could not be confirmed: pinned to
/// their copy-queue fence (never graphics-queue bury lag).
static UNCONFIRMED: Mutex<UnconfirmedPool<Slot, ID3D12Resource, ID3D12Fence>> =
    Mutex::new(UnconfirmedPool::new());
static UNCONFIRMED_COUNT: AtomicUsize = AtomicUsize::new(0);

fn device_removed(device: Option<&ID3D12Device>) -> bool {
    // SAFETY: plain device status query on a live device.
    device.is_some_and(|d| unsafe { d.GetDeviceRemovedReason() }.is_err())
}

/// Sweeps any pinned copy-queue slots whose copy-queue fence has reached its
/// target or whose device has been removed. Runs outside `QUEUE` and drops
/// released entries outside `UNCONFIRMED`.
fn sweep_unconfirmed(device: Option<&ID3D12Device>) {
    if UNCONFIRMED_COUNT.load(Relaxed) == 0 {
        return;
    }
    let removed = device_removed(device);
    let ready = {
        let mut pool = UNCONFIRMED.lock().unwrap_or_else(|e| e.into_inner());
        let ready = pool.take_ready(removed, |fence| {
            // SAFETY: plain fence query on a live fence.
            unsafe { fence.GetCompletedValue() }
        });
        UNCONFIRMED_COUNT.store(pool.entries.len(), Relaxed);
        ready
    };
    drop(ready);
}

/// Clones one COM reference per distinct copy-source `ID3D12Resource` in `ranges`.
fn pin_sources(ranges: &[Range<'_>]) -> Vec<ID3D12Resource> {
    let mut seen = HashSet::new();
    let mut pinned = Vec::new();
    for &(_, _, cpu, _) in ranges {
        if seen.insert(windows::core::Interface::as_raw(cpu) as usize) {
            pinned.push(cpu.clone());
        }
    }
    pinned
}

/// Pins an unconfirmed copy-queue slot (with `slot.bounce`, `allocator`, `list`,
/// and `pinned_sources` intact) until the copy-queue fence confirms completion
/// or the device is removed, and poisons the copy queue under `QUEUE` BEFORE
/// touching `UNCONFIRMED` (strict zero lock nesting between `QUEUE` and
/// `UNCONFIRMED`). If `Signal` failed (no copy-queue fence value) or
/// `UNCONFIRMED` is full at `UNCONFIRMED_MAX`, intentionally leaks via
/// `std::mem::forget` rather than freeing GPU-in-use COM objects.
fn pin_unconfirmed(entry: UnconfirmedSlot, device: Option<&ID3D12Device>, why: &'static str) {
    poison_queue(why);
    if entry.signaled.is_none() {
        std::mem::forget(entry);
        return;
    }
    let removed = device_removed(device);
    let (ready, overflow) = {
        let mut pool = UNCONFIRMED.lock().unwrap_or_else(|e| e.into_inner());
        let ready = pool.take_ready(removed, |fence| {
            // SAFETY: plain fence query on a live fence.
            unsafe { fence.GetCompletedValue() }
        });
        let overflow = pool.try_push(entry).err();
        UNCONFIRMED_COUNT.store(pool.entries.len(), Relaxed);
        (ready, overflow)
    };
    drop(ready);
    if let Some(entry) = overflow {
        std::mem::forget(entry);
    }
}

/// Guards a slot after `ExecuteCommandLists` so that any error return or panic
/// unwind never drops an in-flight allocator/bounce/source or recycles an
/// unconfirmed slot back into `Queue::free`. Declared outside `QUEUE` lock and
/// local `bounce` clone scopes so drop always runs with `QUEUE` unlocked.
struct Submitted<'a, 'b, 'c> {
    slot: Option<Slot>,
    fence: Option<(ID3D12Fence, u64)>,
    signal_ok: bool,
    wait_failed: bool,
    device: &'a ID3D12Device,
    ranges: &'b [Range<'c>],
}

impl<'a, 'b, 'c> Submitted<'a, 'b, 'c> {
    fn new(device: &'a ID3D12Device, ranges: &'b [Range<'c>]) -> Self {
        Self { slot: None, fence: None, signal_ok: false, wait_failed: false, device, ranges }
    }

    fn arm(&mut self, slot: Slot, fence: ID3D12Fence, target: u64) {
        self.slot = Some(slot);
        self.fence = Some((fence, target));
        self.signal_ok = false;
        self.wait_failed = false;
    }

    fn mark_signaled(&mut self) {
        self.signal_ok = true;
    }

    fn mark_wait_failed(&mut self) {
        self.wait_failed = true;
    }

    fn take(&mut self) -> Option<Slot> {
        self.slot.take()
    }
}

impl Drop for Submitted<'_, '_, '_> {
    fn drop(&mut self) {
        let Some(slot) = self.slot.take() else {
            return;
        };
        let signaled = if self.signal_ok { self.fence.take() } else { None };
        let (completed, target) = match &signaled {
            Some((fence, target)) => {
                // SAFETY: plain fence query on a live fence.
                (unsafe { fence.GetCompletedValue() }, *target)
            }
            None => (0, 1),
        };
        match slot_outcome(true, signaled.is_some(), !self.wait_failed, completed, target) {
            SlotOutcome::Recycle => {
                let _ = give_back(slot, || pin_sources(self.ranges));
            }
            SlotOutcome::RetireUnconfirmed => {
                let why = if signaled.is_none() {
                    "copy-queue Signal failed"
                } else if self.wait_failed {
                    "copy-queue CPU wait did not confirm completion"
                } else {
                    "copy-queue submission unwound before fence confirmation"
                };
                let pinned_sources = pin_sources(self.ranges);
                pin_unconfirmed(UnconfirmedSlot { slot, pinned_sources, signaled }, Some(self.device), why);
            }
        }
    }
}

/// Since the last frame-rate line: submissions, bytes, copies sent this
/// way; submissions that went as before because they were too big or
/// something could not be made.
pub(super) static STATS: [AtomicU64; 4] = [const { AtomicU64::new(0) }; 4];

fn make_queue(device: &ID3D12Device) -> Option<Queue> {
    // SAFETY: plain object creation.
    let made = unsafe {
        (|| -> windows::core::Result<Queue> {
            let queue: ID3D12CommandQueue = device.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC {
                Type: D3D12_COMMAND_LIST_TYPE_COPY,
                Priority: D3D12_COMMAND_QUEUE_PRIORITY_HIGH.0,
                ..Default::default()
            })?;
            let fence: ID3D12Fence = device.CreateFence(0, D3D12_FENCE_FLAG_NONE)?;
            Ok(Queue { device: device.clone(), queue, fence, sent: 0, free: Vec::new(), poison: QueuePoison::new() })
        })()
    };
    match made {
        Ok(q) => {
            super::name(&q.queue, || "copies in the card's memory: up queue".into());
            Some(q)
        }
        Err(e) => {
            tracing::warn!(error = %e, "no copy queue for what goes up to copies in the card's memory: it goes on the drawing queue");
            None
        }
    }
}

fn make_slot(device: &ID3D12Device) -> Option<Slot> {
    // SAFETY: plain object creation.
    unsafe {
        (|| -> windows::core::Result<Slot> {
            let allocator: ID3D12CommandAllocator = device.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COPY)?;
            let list: ID3D12GraphicsCommandList = device.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_COPY, &allocator, None)?;
            list.Close()?;
            Ok(Slot { allocator, list, bounce: None, last_sent: 0 })
        })()
        .ok()
    }
}

/// One range: the card side, its offset, the CPU side, the bytes.
pub(super) type Range<'a> = (&'a ID3D12Resource, u64, &'a ID3D12Resource, u64);

/// Sends `ranges` (card side, offset, CPU side, bytes; the same offset on
/// both sides) up on the copy queue into our buffer and makes the drawing
/// queue wait for that; then `record` gets, for each range, our buffer and
/// the range's offset in it, to copy from there on the drawing queue.
/// `None`: nothing was sent (too many bytes, or nothing could be made); the
/// caller copies as before.
///
/// # Safety
/// Live buffers, the ranges lie in them; the CPU's sides hold the bytes to
/// send; `drawing` is the queue the lists that read the card sides go to.
pub(super) unsafe fn send(
    host: &super::Host,
    drawing: &ID3D12CommandQueue,
    ranges: &[Range<'_>],
    mut record: impl FnMut(usize, &ID3D12Resource, u64),
) -> Outcome {
    sweep_unconfirmed(Some(&host.device));
    if !STATE.allow_send() {
        STATS[3].fetch_add(1, Relaxed);
        return Outcome::NotSubmitted;
    }
    let total: u64 = ranges.iter().map(|r| r.3.next_multiple_of(256)).sum();
    if total == 0 || total > MOST {
        STATS[3].fetch_add(1, Relaxed);
        return Outcome::NotSubmitted;
    }
    let popped = {
        let mut guard = QUEUE.lock().unwrap_or_else(|e| e.into_inner());
        let Some(q) = guard.get_or_insert_with(|| make_queue(&host.device)).as_mut() else {
            STATS[3].fetch_add(1, Relaxed);
            return Outcome::NotSubmitted;
        };
        if !q.poison.allow_submit(&STATE) {
            STATS[3].fetch_add(1, Relaxed);
            return Outcome::NotSubmitted;
        }
        q.free.pop()
    };
    let Some(mut slot) = popped.or_else(|| make_slot(&host.device)) else {
        STATS[3].fetch_add(1, Relaxed);
        return Outcome::NotSubmitted;
    };
    if let Some(size) = bounce_size(slot.bounce.as_ref().map(|b| b.1), total) {
        // The slot's last submission has run, but a buffer let go goes
        // through the bury lag like every other object of ours: the device
        // losses of 2026-11-25 were memory freed too early. Retired outside
        // `QUEUE` so no lock is held across retire/drop or buffer creation
        // (GPL-3.0 provenance: Belobug/VIctory-Belobug-AMD#401).
        if let Some((old, _)) = slot.bounce.take() {
            super::retire_resource(old);
        }
        let Some(made) = host.uav_buffer(size) else {
            let _ = give_back(slot, Vec::new);
            STATS[3].fetch_add(1, Relaxed);
            return Outcome::NotSubmitted;
        };
        super::name(&made, || format!("copies in the card's memory: what goes up, {} MB", size >> 20));
        slot.bounce = Some((made, size));
    }
    // Declared before `bounce` and before the `QUEUE` lock scope so on error
    // or unwind, `QUEUE` and the local `bounce` clone always drop BEFORE
    // `Submitted::drop` pins or recycles the slot.
    let mut submitted_guard = Submitted::new(&host.device, ranges);
    let Some((bounce, _)) = slot.bounce.clone() else {
        let _ = give_back(slot, Vec::new);
        STATS[3].fetch_add(1, Relaxed);
        return Outcome::NotSubmitted;
    };
    // SAFETY: the slot's last submission has run (it came back with its
    // plan); the caller's for the buffers.
    let list_to_run = unsafe {
        (|| -> windows::core::Result<ID3D12CommandList> {
            slot.allocator.Reset()?;
            slot.list.Reset(&slot.allocator, None)?;
            let mut at = 0u64;
            for &(_, offset, cpu, bytes) in ranges {
                slot.list.CopyBufferRegion(&bounce, at, cpu, offset, bytes);
                at += bytes.next_multiple_of(256);
            }
            slot.list.Close()?;
            windows::core::Interface::cast(&slot.list)
        })()
    };
    let list_to_run = match list_to_run {
        Ok(list) => list,
        Err(e) => {
            // (Nothing waits for a copy that was not sent.)
            tracing::warn!(error = %e, "what goes up to copies in the card's memory could not go on the copy queue: as before");
            let _ = give_back(slot, Vec::new);
            STATS[3].fetch_add(1, Relaxed);
            return Outcome::NotSubmitted;
        }
    };
    let cpu_wait_fence = {
        let mut guard = QUEUE.lock().unwrap_or_else(|e| e.into_inner());
        let Some(q) = guard.as_mut().and_then(Option::as_mut) else {
            drop(guard);
            let _ = give_back(slot, Vec::new);
            STATS[3].fetch_add(1, Relaxed);
            return Outcome::NotSubmitted;
        };
        if !q.poison.allow_submit(&STATE) {
            // Another thread (e.g. `give_back` on the relay thread) poisoned
            // the copy queue after we passed the pre-lock fast check: return
            // the unused slot cleanly without calling `ExecuteCommandLists`.
            q.free.push(slot);
            STATS[3].fetch_add(1, Relaxed);
            return Outcome::NotSubmitted;
        }
        // SAFETY: `list_to_run` was closed above and belongs to `slot`.
        unsafe { q.queue.ExecuteCommandLists(&[Some(list_to_run)]) };
        q.sent += 1;
        let target = q.sent;
        slot.last_sent = target;
        submitted_guard.arm(slot, q.fence.clone(), target);
        // SAFETY: `q.fence` is a live fence owned by the copy queue.
        if let Err(e) = unsafe { q.queue.Signal(&q.fence, target) } {
            // Poison under `QUEUE` BEFORE unlocking so no concurrent `send()`
            // can execute on this queue; `submitted_guard` then pins/leaks
            // the bounce + allocator + sources after `QUEUE` unlocks.
            poison_under_queue(q, "copy-queue Signal failed");
            tracing::warn!(error = %e, "what goes up to copies in the card's memory could not signal its fence: as before");
            STATS[3].fetch_add(1, Relaxed);
            return Outcome::SubmittedUnconfirmed;
        }
        submitted_guard.mark_signaled();
        // SAFETY: a fence of ours that the copy queue signals.
        if let Err(e) = unsafe { drawing.Wait(&q.fence, target) } {
            // The drawing queue cannot be made to wait: wait on the CPU after
            // releasing `QUEUE`, and only proceed if completion is confirmed.
            tracing::warn!(error = %e, "the drawing queue could not wait for the up queue: waited on the CPU");
            Some((q.fence.clone(), target))
        } else {
            None
        }
    };
    if let Some((fence, target)) = cpu_wait_fence {
        // SAFETY: `fence` was signalled above; a null event waits until `target` is reached.
        let wait_ok = unsafe {
            fence.SetEventOnCompletion(target, windows::Win32::Foundation::HANDLE::default())
        }
        .is_ok();
        // SAFETY: plain fence query on a live fence.
        let completed = unsafe { fence.GetCompletedValue() };
        if !fence_confirmed(wait_ok, completed, target) {
            tracing::warn!(
                wait_ok,
                completed,
                target,
                "up queue CPU wait did not confirm copy completion: as before"
            );
            submitted_guard.mark_wait_failed();
            STATS[3].fetch_add(1, Relaxed);
            return Outcome::SubmittedUnconfirmed;
        }
    }
    let mut at = 0u64;
    for (i, r) in ranges.iter().enumerate() {
        record(i, &bounce, at);
        at += r.3.next_multiple_of(256);
    }
    STATS[0].fetch_add(1, Relaxed);
    STATS[1].fetch_add(ranges.iter().map(|r| r.3).sum::<u64>(), Relaxed);
    STATS[2].fetch_add(ranges.len() as u64, Relaxed);
    match submitted_guard.take() {
        Some(slot) => Outcome::Sent(slot),
        None => Outcome::NotSubmitted,
    }
}

/// A plan's slot, back once its submission has run. Returns `true` if the copy
/// queue's fence confirmed completion (safe for the caller to recycle source
/// upload memory), or `false` if completion could not be confirmed (copy queue
/// is poisoned, `slot` + `pin_extra()` sources stay pinned to the copy-queue
/// fence outside `QUEUE`, and caller must NOT recycle source upload memory).
pub(super) fn give_back(mut slot: Slot, pin_extra: impl FnOnce() -> Vec<ID3D12Resource>) -> bool {
    enum Back {
        Stored(ID3D12Device),
        PinUnconfirmed(Slot, Option<(ID3D12Fence, u64)>, Option<ID3D12Device>),
        Orphan(Slot),
    }
    let action = {
        let mut guard = QUEUE.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(Some(q)) = guard.as_mut() {
            let reusable = if slot.last_sent == 0 {
                true
            } else {
                // SAFETY: plain fence query on a live fence.
                let completed = unsafe { q.fence.GetCompletedValue() };
                slot_outcome(true, true, true, completed, slot.last_sent) == SlotOutcome::Recycle
            };
            if reusable {
                slot.last_sent = 0;
                q.free.push(slot);
                Back::Stored(q.device.clone())
            } else {
                // Poison under `QUEUE` BEFORE unlocking so any concurrent
                // `send()` serialized on `QUEUE` bails out before
                // `ExecuteCommandLists`, then pin in `UNCONFIRMED` after
                // dropping `QUEUE`.
                poison_under_queue(q, "copy-queue fence not confirmed at card_done");
                let signaled = Some((q.fence.clone(), slot.last_sent));
                Back::PinUnconfirmed(slot, signaled, Some(q.device.clone()))
            }
        } else if slot.last_sent != 0 {
            Back::PinUnconfirmed(slot, None, None)
        } else {
            Back::Orphan(slot)
        }
    };
    match action {
        Back::Stored(device) => {
            sweep_unconfirmed(Some(&device));
            true
        }
        Back::PinUnconfirmed(slot, signaled, device) => {
            pin_unconfirmed(
                UnconfirmedSlot { slot, pinned_sources: pin_extra(), signaled },
                device.as_ref(),
                "copy-queue fence not confirmed at card_done",
            );
            false
        }
        Back::Orphan(mut slot) => {
            if let Some((old, _)) = slot.bounce.take() {
                super::retire_resource(old);
            }
            drop(slot);
            true
        }
    }
}

/// For the frame-rate line.
pub(super) fn line(frames: f64) {
    let [subs, bytes, copies, before] = [0, 1, 2, 3].map(|i| STATS[i].swap(0, Relaxed) as f64 / frames);
    if subs + before > 0.0 {
        tracing::info!(
            "copies in the card's memory brought up on a copy queue per frame (VICTORY_UP_QUEUE): {subs:.1} submissions, \
             {copies:.0} ranges, {:.1} MB; {before:.1} submissions sent up on the drawing queue (too big, or nothing to send with)",
            bytes / 1048576.0
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    #[test]
    fn a_buffer_is_made_only_when_the_one_we_have_is_too_small() {
        assert_eq!(bounce_size(None, 1), Some(LEAST));
        assert_eq!(bounce_size(Some(LEAST), LEAST), None);
        assert_eq!(bounce_size(Some(LEAST), LEAST + 256), Some(2 * LEAST));
        assert_eq!(bounce_size(Some(64 << 20), 5 << 20), None);
        assert_eq!(bounce_size(None, (5 << 20) + 1), Some(8 << 20));
        assert_eq!(bounce_size(None, MOST), Some(MOST));
        assert_eq!(bounce_size(Some(MOST), MOST), None);
    }

    #[test]
    fn unconfirmed_submitted_slots_are_never_recycled() {
        // Never submitted: always reusable.
        assert_eq!(slot_outcome(false, false, false, 0, 1), SlotOutcome::Recycle);
        // Submitted + signaled + confirmed completion: reusable.
        assert_eq!(slot_outcome(true, true, true, 7, 7), SlotOutcome::Recycle);
        assert_eq!(slot_outcome(true, true, true, 9, 7), SlotOutcome::Recycle);
        // Signal failed after ExecuteCommandLists: never recycle.
        assert_eq!(slot_outcome(true, false, true, 9, 7), SlotOutcome::RetireUnconfirmed);
        // Wait failed (e.g. SetEventOnCompletion error): never recycle.
        assert_eq!(slot_outcome(true, true, false, 7, 7), SlotOutcome::RetireUnconfirmed);
        // Fence still behind target: never recycle.
        assert_eq!(slot_outcome(true, true, true, 6, 7), SlotOutcome::RetireUnconfirmed);
        // D3D12 device-removed sentinel (`u64::MAX`): never recycle.
        assert_eq!(slot_outcome(true, true, true, u64::MAX, 7), SlotOutcome::RetireUnconfirmed);
        assert_eq!(slot_outcome(true, true, true, u64::MAX, u64::MAX), SlotOutcome::RetireUnconfirmed);
    }

    #[test]
    fn unconfirmed_slot_retains_bounce_and_sources_until_copy_fence_confirms() {
        struct FakeSlot {
            _allocator: Arc<()>,
            _bounce: Arc<()>,
        }

        let alloc = Arc::new(());
        let bounce = Arc::new(());
        let src1 = Arc::new(());
        let src2 = Arc::new(());
        let fence_value = Arc::new(AtomicU64::new(4));

        let mut pool: UnconfirmedPool<FakeSlot, Arc<()>, Arc<AtomicU64>> = UnconfirmedPool::new();
        assert!(
            pool.try_push(UnconfirmedSlot {
                slot: FakeSlot { _allocator: Arc::clone(&alloc), _bounce: Arc::clone(&bounce) },
                pinned_sources: vec![Arc::clone(&src1), Arc::clone(&src2)],
                signaled: Some((Arc::clone(&fence_value), 7)),
            })
            .is_ok()
        );

        // Below target (4 < 7): slot, bounce, and sources all stay pinned.
        assert!(pool.take_ready(false, |f| f.load(Relaxed)).is_empty());
        assert_eq!(Arc::strong_count(&alloc), 2);
        assert_eq!(Arc::strong_count(&bounce), 2);
        assert_eq!(Arc::strong_count(&src1), 2);
        assert_eq!(Arc::strong_count(&src2), 2);

        // Device-removed sentinel (`u64::MAX`) without `device_removed` flag: still pinned.
        fence_value.store(u64::MAX, Relaxed);
        assert!(pool.take_ready(false, |f| f.load(Relaxed)).is_empty());
        assert_eq!(Arc::strong_count(&bounce), 2);
        assert_eq!(Arc::strong_count(&src1), 2);

        // Once copy-queue fence confirms >= target (7): swept and released.
        fence_value.store(7, Relaxed);
        let ready = pool.take_ready(false, |f| f.load(Relaxed));
        assert_eq!(ready.len(), 1);
        drop(ready);
        assert_eq!(Arc::strong_count(&alloc), 1);
        assert_eq!(Arc::strong_count(&bounce), 1);
        assert_eq!(Arc::strong_count(&src1), 1);
        assert_eq!(Arc::strong_count(&src2), 1);

        // Device removal sweeps stuck signaled entries too.
        fence_value.store(0, Relaxed);
        assert!(
            pool.try_push(UnconfirmedSlot {
                slot: FakeSlot { _allocator: Arc::clone(&alloc), _bounce: Arc::clone(&bounce) },
                pinned_sources: vec![Arc::clone(&src1)],
                signaled: Some((Arc::clone(&fence_value), 9)),
            })
            .is_ok()
        );
        let ready = pool.take_ready(true, |f| f.load(Relaxed));
        assert_eq!(ready.len(), 1);
        drop(ready);
        assert_eq!(Arc::strong_count(&bounce), 1);
        assert_eq!(Arc::strong_count(&src1), 1);

        // Unsignaled (`Signal` failed) or pool overflow (> UNCONFIRMED_MAX) is
        // rejected so caller intentionally `std::mem::forget`s instead of dropping.
        assert!(
            pool.try_push(UnconfirmedSlot {
                slot: FakeSlot { _allocator: Arc::clone(&alloc), _bounce: Arc::clone(&bounce) },
                pinned_sources: vec![Arc::clone(&src1)],
                signaled: None,
            })
            .is_err()
        );
        for target in 1..=(UNCONFIRMED_MAX as u64) {
            assert!(
                pool.try_push(UnconfirmedSlot {
                    slot: FakeSlot { _allocator: Arc::clone(&alloc), _bounce: Arc::clone(&bounce) },
                    pinned_sources: vec![Arc::clone(&src1)],
                    signaled: Some((Arc::clone(&fence_value), target)),
                })
                .is_ok()
            );
        }
        assert!(
            pool.try_push(UnconfirmedSlot {
                slot: FakeSlot { _allocator: Arc::clone(&alloc), _bounce: Arc::clone(&bounce) },
                pinned_sources: vec![Arc::clone(&src1)],
                signaled: Some((Arc::clone(&fence_value), 999)),
            })
            .is_err()
        );
    }

    #[test]
    fn poisoning_stops_subsequent_upq_submissions() {
        let state = UpqState::new();
        assert!(state.allow_send());
        assert!(!state.is_poisoned());
        // First unconfirmed failure trips the latch once.
        assert!(state.trip());
        assert!(state.is_poisoned());
        assert!(!state.allow_send());
        // Subsequent trips are idempotent and keep submissions disabled.
        assert!(!state.trip());
        assert!(!state.allow_send());
    }

    #[test]
    fn only_all_owned_send_memory_batches_are_eligible_for_upq() {
        assert!(batch_eligible(true, true, &[SourceKind::OwnedSendMemory]));
        assert!(batch_eligible(true, true, &[SourceKind::OwnedSendMemory, SourceKind::OwnedSendMemory]));
        // Any shared `Card::staging` source in the batch disqualifies `upq`.
        assert!(!batch_eligible(true, true, &[SourceKind::SharedCardStaging]));
        assert!(!batch_eligible(true, true, &[SourceKind::OwnedSendMemory, SourceKind::SharedCardStaging]));
        // Disabling `VICTORY_SEND_OWN`, enabling `upcheck`, or empty batches disqualify `upq`.
        assert!(!batch_eligible(false, true, &[SourceKind::OwnedSendMemory]));
        assert!(!batch_eligible(true, false, &[SourceKind::OwnedSendMemory]));
        assert!(!batch_eligible(true, true, &[]));
    }

    #[test]
    fn serialized_queue_poison_rejects_inflight_caller_before_submit() {
        let state = UpqState::new();
        let queue = Mutex::new(QueuePoison::new());

        // Caller A passes the pre-lock fast check before any failure occurs.
        assert!(state.allow_send());

        // Concurrently, the relay thread detects an unconfirmed slot in
        // `give_back` and poisons under the `QUEUE` lock before releasing it.
        {
            let mut q = queue.lock().unwrap();
            assert!(q.allow_submit(&state));
            assert!(q.mark_poisoned(&state));
            assert!(!q.allow_submit(&state));
            // Idempotent if called again under `QUEUE`.
            assert!(!q.mark_poisoned(&state));
        }

        // Caller A now acquires `QUEUE` right before `ExecuteCommandLists`:
        // the serialized check rejects submission cleanly (`Outcome::NotSubmitted`).
        let submitted = {
            let q = queue.lock().unwrap();
            q.allow_submit(&state)
        };
        assert!(!submitted);
        assert!(state.is_poisoned());
    }
}
