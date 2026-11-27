//! A host list's D3D12 list, with barrier calls joined
//! (`VICTORY_JOIN_BARRIERS=1`; off by default; A/B-able with
//! `VICTORY_AB=<s>,VICTORY_JOIN_BARRIERS`).
//!
//! The state changes the host asks for (`crate::list::ensure_state`, the
//! aliasing before a texture's first upload) and the title's own barriers
//! wait here instead of going to the list at once; they go, in the order
//! they were asked for, in one call, at the next use of the list for
//! anything else. Every use goes through [`Rec`]'s `Deref`, so nothing is
//! recorded before a barrier asked for earlier. In a replay of capture
//! replay5 joining the calls that had no work between them saved 2.4 ms of
//! 79.5 (2,490 calls joined, 1,619 of them a texture's state change after
//! its upload followed by the next texture's aliasing).

use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering::Relaxed};

use windows::Win32::Graphics::Direct3D12::*;

/// Whether barriers are held now (`VICTORY_JOIN_BARRIERS=1`, or its phase
/// of `VICTORY_AB`).
pub(crate) fn on() -> bool {
    #[cfg(feature = "diagnostics")]
    if let Some(on) = crate::ab::pick("VICTORY_JOIN_BARRIERS") {
        return on;
    }
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_JOIN_BARRIERS").is_some_and(|v| v == "1"))
}

/// Barriers held. Their resources are borrowed: the list's states hold the
/// textures (`HostList::transition`), and the title's barriers name what
/// the title keeps alive until the list has run.
struct Held(Vec<D3D12_RESOURCE_BARRIER>);

// SAFETY: borrowed interface pointers of free-threaded objects.
unsafe impl Send for Held {}

/// Since the last frame-rate line: barrier calls asked for while held, and
/// calls made of them.
pub(crate) static STATS: [std::sync::atomic::AtomicU64; 2] = [const { std::sync::atomic::AtomicU64::new(0) }; 2];

/// `VICTORY_CLAMP_WAITS` (on by default; `0` off): the argument clamps
/// (`crate::indirect`) omit their pre-clamp global UAV barrier when a global
/// UAV barrier is already active with no intervening commands recorded on this
/// list (`Rec::clamp_wait`), and omit the post-clamp global UAV barrier (the
/// scratch buffer transition to `INDIRECT_ARGUMENT` orders the clamped records
/// and `budget` is only updated via atomics).
pub(crate) fn clamp_waits() -> bool {
    #[cfg(feature = "diagnostics")]
    if let Some(on) = crate::ab::pick("VICTORY_CLAMP_WAITS") {
        return on;
    }
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_CLAMP_WAITS").is_none_or(|v| v != "0"))
}

/// Since the last frame-rate line: argument-clamp pre-waits made, and elided.
pub(crate) static CLAMP_WAITS: [std::sync::atomic::AtomicU64; 2] =
    [const { std::sync::atomic::AtomicU64::new(0) }; 2];

pub(crate) struct Rec {
    list: ID3D12GraphicsCommandList,
    any: AtomicBool,
    /// True when a global NULL UAV barrier has already been emitted or held
    /// since the last non-barrier command on this list, so another UAV barrier
    /// before any intervening GPU work is redundant.
    uav_synced: AtomicBool,
    held: Mutex<Held>,
}

/// True if `b` is a UAV barrier with `pResource == None` (synchronizes all UAVs).
fn is_global_uav(b: &D3D12_RESOURCE_BARRIER) -> bool {
    if b.Type != D3D12_RESOURCE_BARRIER_TYPE_UAV {
        return false;
    }
    // SAFETY: `Type == UAV` selects the `UAV` union arm.
    unsafe { b.Anonymous.UAV.pResource.is_none() }
}

/// Resource pointer named by a per-resource UAV barrier (`None` for global UAV
/// or non-UAV barriers).
fn uav_resource(b: &D3D12_RESOURCE_BARRIER) -> Option<usize> {
    if b.Type != D3D12_RESOURCE_BARRIER_TYPE_UAV {
        return None;
    }
    // SAFETY: `Type == UAV` selects the `UAV` union arm.
    unsafe {
        b.Anonymous
            .UAV
            .pResource
            .as_ref()
            .map(|r| windows::core::Interface::as_raw(r) as usize)
    }
}

/// Filters/merges `incoming` into `out`, collapsing redundant UAV barriers
/// without altering any transition or aliasing barrier order. Returns whether
/// `out` (or a previously emitted barrier via `already_uav_synced`) now covers
/// all UAVs globally.
fn push_deduped(
    out: &mut Vec<D3D12_RESOURCE_BARRIER>,
    incoming: &[D3D12_RESOURCE_BARRIER],
    already_uav_synced: bool,
) -> bool {
    let mut has_global = already_uav_synced || out.iter().any(is_global_uav);
    if !has_global && incoming.iter().any(is_global_uav) {
        // A global NULL UAV barrier in this contiguous batch subsumes any
        // per-resource UAV barriers already queued in `out`.
        out.retain(|b| b.Type != D3D12_RESOURCE_BARRIER_TYPE_UAV);
    }
    for b in incoming {
        if b.Type == D3D12_RESOURCE_BARRIER_TYPE_UAV {
            if has_global {
                continue;
            }
            if is_global_uav(b) {
                has_global = true;
            } else if let Some(res) = uav_resource(b)
                && out.iter().any(|existing| uav_resource(existing) == Some(res))
            {
                continue;
            }
        }
        // SAFETY: plain copy of D3D12_RESOURCE_BARRIER with borrowed ManuallyDrop pointers.
        out.push(unsafe { std::ptr::read(b) });
    }
    has_global
}

impl Rec {
    pub(crate) fn new(list: ID3D12GraphicsCommandList) -> Self {
        Self {
            list,
            any: AtomicBool::new(false),
            uav_synced: AtomicBool::new(false),
            held: Mutex::new(Held(Vec::new())),
        }
    }

    /// Marks the list as having unsynchronized GPU writes.
    #[allow(dead_code)]
    pub(crate) fn writes(&self) {
        self.uav_synced.store(false, Relaxed);
    }

    /// Whether an argument clamp must emit a pre-clamp global UAV barrier
    /// before reading argument/count buffers. Must be called BEFORE derefing
    /// `&self` (`Deref` flushes held barriers and clears `uav_synced`).
    pub(crate) fn clamp_wait(&self) -> bool {
        if clamp_waits() && self.uav_synced.load(Relaxed) {
            CLAMP_WAITS[1].fetch_add(1, Relaxed);
            return false;
        }
        CLAMP_WAITS[0].fetch_add(1, Relaxed);
        true
    }

    /// Records `barriers` now, or holds them for the next use of the list,
    /// collapsing redundant UAV barriers when no GPU work intervened.
    ///
    /// # Safety
    /// As `ResourceBarrier`'s: an open list, live resources.
    pub(crate) unsafe fn barriers(&self, barriers: &[D3D12_RESOURCE_BARRIER]) {
        if barriers.is_empty() {
            return;
        }
        let already_synced = self.uav_synced.load(Relaxed);
        if !on() {
            // (What was held while it was on goes first.)
            if self.any.load(Relaxed) {
                self.flush();
            }
            let mut filtered = Vec::with_capacity(barriers.len());
            let now_synced = push_deduped(&mut filtered, barriers, already_synced);
            if !filtered.is_empty() {
                // SAFETY: the caller's.
                unsafe { self.list.ResourceBarrier(&filtered) };
            }
            if now_synced {
                self.uav_synced.store(true, Relaxed);
            }
            return;
        }
        let mut held = self.held.lock().unwrap_or_else(|e| e.into_inner());
        // A resource whose state changes again: what is held goes first, so
        // that no call holds two changes of one resource.
        if barriers.iter().filter_map(named).any(|r| held.0.iter().filter_map(named).any(|h| h == r)) {
            // SAFETY: as in `flush`.
            unsafe { self.list.ResourceBarrier(&held.0) };
            held.0.clear();
            STATS[1].fetch_add(1, Relaxed);
        }
        let now_synced = push_deduped(&mut held.0, barriers, already_synced);
        if now_synced {
            self.uav_synced.store(true, Relaxed);
        }
        STATS[0].fetch_add(1, Relaxed);
        self.any.store(!held.0.is_empty(), Relaxed);
    }

    /// What is held goes to the list, in one call.
    fn flush(&self) {
        let mut held = self.held.lock().unwrap_or_else(|e| e.into_inner());
        self.any.store(false, Relaxed);
        if held.0.is_empty() {
            return;
        }
        // SAFETY: the barriers were asked for on this open list, in order;
        // their resources are alive (`Held`).
        unsafe { self.list.ResourceBarrier(&held.0) };
        held.0.clear();
        STATS[1].fetch_add(1, Relaxed);
    }
}

/// The resource a state change or aliasing names (its interface pointer).
fn named(b: &D3D12_RESOURCE_BARRIER) -> Option<usize> {
    // SAFETY: the union's member is the one `Type` says; a pointer read.
    unsafe {
        let p: Option<&ID3D12Resource> = match b.Type {
            D3D12_RESOURCE_BARRIER_TYPE_TRANSITION => b.Anonymous.Transition.pResource.as_ref(),
            D3D12_RESOURCE_BARRIER_TYPE_ALIASING => b.Anonymous.Aliasing.pResourceAfter.as_ref(),
            _ => None,
        };
        p.map(|r| windows::core::Interface::as_raw(r) as usize)
    }
}

impl std::ops::Deref for Rec {
    type Target = ID3D12GraphicsCommandList;

    /// The list, with what is held recorded first. Any command recorded
    /// through `Deref` may perform GPU writes, so `uav_synced` resets.
    fn deref(&self) -> &ID3D12GraphicsCommandList {
        if self.any.load(Relaxed) {
            self.flush();
        }
        self.uav_synced.store(false, Relaxed);
        &self.list
    }
}

/// For the frame-rate line.
pub(crate) fn line(frames: f64) {
    let [asked, made] = [0, 1].map(|i| STATS[i].swap(0, Relaxed) as f64 / frames);
    if asked > 0.0 {
        tracing::info!(
            "barrier calls per frame (VICTORY_JOIN_BARRIERS): {asked:.0} asked for, {made:.0} made of them"
        );
    }
    let [made, left] = [0, 1].map(|i| CLAMP_WAITS[i].swap(0, Relaxed) as f64 / frames);
    if made + left > 0.0 {
        tracing::info!(
            "argument clamps' waits per frame (VICTORY_CLAMP_WAITS): {made:.0} made, {left:.0} left out"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn global_uav() -> D3D12_RESOURCE_BARRIER {
        D3D12_RESOURCE_BARRIER {
            Type: D3D12_RESOURCE_BARRIER_TYPE_UAV,
            Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
            Anonymous: D3D12_RESOURCE_BARRIER_0 {
                UAV: std::mem::ManuallyDrop::new(D3D12_RESOURCE_UAV_BARRIER {
                    pResource: std::mem::ManuallyDrop::new(None),
                }),
            },
        }
    }

    fn transition_dummy() -> D3D12_RESOURCE_BARRIER {
        D3D12_RESOURCE_BARRIER {
            Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
            Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
            Anonymous: D3D12_RESOURCE_BARRIER_0 {
                Transition: std::mem::ManuallyDrop::new(D3D12_RESOURCE_TRANSITION_BARRIER {
                    pResource: std::mem::ManuallyDrop::new(None),
                    Subresource: D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                    StateBefore: D3D12_RESOURCE_STATE_COMMON,
                    StateAfter: D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                }),
            },
        }
    }

    #[test]
    fn consecutive_global_uav_barriers_collapse_while_keeping_transitions() {
        let mut out = Vec::new();
        assert!(push_deduped(&mut out, &[global_uav(), global_uav()], false));
        assert_eq!(out.len(), 1);
        assert!(is_global_uav(&out[0]));

        let mut next = Vec::new();
        assert!(push_deduped(&mut next, &[global_uav(), transition_dummy()], true));
        assert_eq!(next.len(), 1);
        assert_eq!(next[0].Type, D3D12_RESOURCE_BARRIER_TYPE_TRANSITION);
    }
}
