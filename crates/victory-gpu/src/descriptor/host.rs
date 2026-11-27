//! Host descriptor heaps behind the title's shader-visible heaps (decision
//! 0007, step 3). Translated shaders index the heaps directly at the
//! title's descriptor positions, so each host heap keeps the title heap's
//! order: slot `i` is made from what the title wrote at slot `i`, when it
//! writes it (view creation and descriptor copies) and again when a draw
//! reads it through a table and finds the slot changed.

use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};

use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;

use super::{VIEW_TAG, View, view_at};
use crate::host::host;

/// A set of a heap's slots, a bit for each: no lock (a pass that went through
/// a set of 70,000 under its lock kept everyone who writes slots waiting for
/// 5 ms of a frame, timeline thin-6).
struct Slots(Vec<std::sync::atomic::AtomicU64>);

impl Slots {
    fn new(count: usize) -> Self {
        let mut bits = Vec::new();
        bits.resize_with(count.div_ceil(64), Default::default);
        Slots(bits)
    }

    fn put(&self, slot: u32) {
        if let Some(bits) = self.0.get(slot as usize / 64) {
            bits.fetch_or(1 << (slot % 64), std::sync::atomic::Ordering::AcqRel);
        }
    }

    /// Whether it was there.
    fn remove(&self, slot: u32) -> bool {
        use std::sync::atomic::Ordering::{AcqRel, Acquire};
        let bit = 1 << (slot % 64);
        self.0.get(slot as usize / 64).is_some_and(|bits| {
            bits.load(Acquire) & bit != 0 && bits.fetch_and(!bit, AcqRel) & bit != 0
        })
    }

    fn of(n: usize, mut bits: u64, out: &mut Vec<u32>) {
        while bits != 0 {
            out.push(n as u32 * 64 + bits.trailing_zeros());
            bits &= bits - 1;
        }
    }

    /// The set slots selected by the same mask in each 64-slot word, in order.
    fn all(&self, mask: u64) -> Vec<u32> {
        let mut out = Vec::new();
        for (n, bits) in self.0.iter().enumerate() {
            Self::of(n, bits.load(std::sync::atomic::Ordering::Acquire) & mask, &mut out);
        }
        out
    }

    /// One part of a round-robin scan. The pattern repeats at every word boundary.
    fn part<const PARTS: u32>(&self, part: u32) -> Vec<u32> {
        const { assert!(PARTS > 0 && u64::BITS % PARTS == 0) };
        if part >= PARTS {
            return Vec::new();
        }
        let mask = (part..u64::BITS)
            .step_by(PARTS as usize)
            .fold(0, |mask, bit| mask | (1 << bit));
        self.all(mask)
    }

    /// All of them, and none is left.
    fn take(&self) -> Vec<u32> {
        use std::sync::atomic::Ordering::{AcqRel, Acquire};
        let mut out = Vec::new();
        for (n, bits) in self.0.iter().enumerate() {
            if bits.load(Acquire) != 0 {
                Self::of(n, bits.swap(0, AcqRel), &mut out);
            }
        }
        out
    }
}

/// The record each slot of a heap was last made from, and a lock for every
/// few thousandth of the slots: whoever makes a slot holds its lock, and no
/// other.
///
/// One lock for the heap was what everything of a frame waited for once the
/// lists were translated by threads of ours: half of what those did, two
/// thirds of the heaps' records and half of the pass over the buffer views
/// (profile thin-2).
struct Made {
    keys: Vec<std::sync::atomic::AtomicU64>,
    locks: Box<[Mutex<()>]>,
}

impl Made {
    const LOCKS: usize = 1 << 14;

    fn new(count: usize) -> Self {
        let mut keys = Vec::new();
        keys.resize_with(count, Default::default);
        Made { keys, locks: (0..Self::LOCKS).map(|_| Mutex::new(())).collect() }
    }

    /// The lock slot `index` is made under. Neighbours have different ones,
    /// and so have the slots that threads which go through parts of a heap in
    /// step are at (with one of 512 by the slot's lowest bits, they stood 2 to
    /// 4 ms a frame of each in each other's way, run b5-small).
    fn lock(&self, index: u32) -> std::sync::MutexGuard<'_, ()> {
        let lock = index.wrapping_mul(0x9E37_79B1) as usize >> (32 - 14);
        self.locks[lock].lock().unwrap_or_else(|e| e.into_inner())
    }

    fn get(&self, index: u32) -> Option<u64> {
        Some(self.keys.get(index as usize)?.load(std::sync::atomic::Ordering::Acquire))
    }

    /// Under the slot's lock.
    fn set(&self, index: u32, key: u64) {
        if let Some(k) = self.keys.get(index as usize) {
            k.store(key, std::sync::atomic::Ordering::Release);
        }
    }
}

mod native;

/// The title's slots past a host heap that could not hold them (the NVIDIA
/// cap): how many times a slot was skipped, the highest index among them,
/// and the size of the heap they were past (`Marvisak #3475`).
pub(crate) static PAST_HEAP: PastHeap = PastHeap::new();

pub(crate) struct PastHeap {
    skips: std::sync::atomic::AtomicU64,
    highest: std::sync::atomic::AtomicU32,
    count: std::sync::atomic::AtomicU32,
}

impl PastHeap {
    const fn new() -> Self {
        Self {
            skips: std::sync::atomic::AtomicU64::new(0),
            highest: std::sync::atomic::AtomicU32::new(0),
            count: std::sync::atomic::AtomicU32::new(0),
        }
    }

    /// Counts every skip, returning the highest index seen and whether this
    /// observation first reaches a new power-of-two warning range.
    fn note(&self, index: u32, count: u32) -> (u32, bool) {
        self.skips.fetch_add(1, Ordering::Relaxed);
        let previous = self.highest.fetch_max(index, Ordering::Relaxed);
        let highest = previous.max(index);
        self.count.store(count, Ordering::Relaxed);
        let warn = worth_a_warning(slots_past(count, previous), slots_past(count, highest));
        (highest, warn)
    }

    #[cfg(test)]
    fn get(&self) -> (u64, u32, u32) {
        (
            self.skips.load(Ordering::Relaxed),
            self.highest.load(Ordering::Relaxed),
            self.count.load(Ordering::Relaxed),
        )
    }
}

/// How many of the title's slots are past a heap of `count` slots when the
/// title's slots reach `highest`.
pub(crate) fn slots_past(count: u32, highest: u32) -> u32 {
    highest.saturating_sub(count.saturating_sub(1))
}

fn worth_a_warning(previous: u32, past: u32) -> bool {
    past > previous && (previous == 0 || past.ilog2() > previous.ilog2())
}

/// Each host heap's identity, handed out in order (`fxceless #1324`, GPL-3.0):
/// dynamic slots are returned only to the heap that allocated them.
static NEXT_ID: AtomicU64 = AtomicU64::new(1);

pub(crate) struct HostHeap {
    id: u64,
    native_slots: std::sync::OnceLock<native::Slots>,
    pub(crate) buffer_sweep: Mutex<crate::draw::BufferSweep>,
    pub(crate) buffer_frame: std::sync::atomic::AtomicU64,
    pub(crate) heap: ID3D12DescriptorHeap,
    cpu: usize,
    gpu: u64,
    step: u32,
    /// Slots mirroring the title's; view heaps have `dynamic` more.
    count: u32,
    /// Dynamic slots past `count`: [`DYNAMIC`] in view heaps, none in
    /// sampler heaps. (Upstream falls back to 2^16 on a capped host heap;
    /// ours keeps 2^18 there: 2^16 and 2^17 both ran out in this title.)
    dynamic: u32,
    sampler: bool,
    /// Where the raw twins of the title's slots start
    /// (`victory_shader::emit::raw_views`; 0: there are none), and which of
    /// them are not null.
    raw: u32,
    twins: Vec<std::sync::atomic::AtomicBool>,
    /// When each slot was written last (with `crate::book`; else empty).
    #[cfg(feature = "diagnostics")]
    written: Vec<std::sync::atomic::AtomicU32>,
    /// The record (view index and generation) each slot was last made from.
    made: Made,
    /// The same for a slot whose descriptor is made and stays (else 0): what
    /// [`HostHeap::mirror`] asks first, without the lock. Changed under it.
    quick: Vec<std::sync::atomic::AtomicU64>,
    /// The records of the title's heap as the pass over records copied in
    /// as bytes last saw them ([`Copied::changes`]). Made on first
    /// use.
    copied: Mutex<Vec<u8>>,
    /// New ones of them that wait for a draw ([`HostHeap::copied_wait`]),
    /// by the frame they were found in.
    waiting: Mutex<std::collections::VecDeque<(u64, Vec<u32>)>>,
    /// The slots made from a texture whose memory was found away, with what
    /// they were made from and when that was
    /// ([`HostHeap::texture_slot_gone`]).
    away: Mutex<std::collections::HashMap<u32, (u64, u64)>>,
    /// The host resource each slot views, held while the slot does: shaders
    /// index heaps directly, so a slot the title has not rewritten since it
    /// released the resource may still be read, and a freed host resource
    /// read there is a GPU page fault (the console reads stale memory).
    ///
    /// A lock for each slot: one for the heap was what the recording threads
    /// waited for most (5.6 % of their time, profile waits-1: each descriptor
    /// made for a draw takes it, and lets go of what the slot held under it).
    kept: Box<[Mutex<Option<ID3D12Resource>>]>,
    /// Whether each slot of `kept` holds a resource: written with the slot's
    /// lock held, read without by [`HostHeap::keep_resource`] (`ReZOnGithub 685b830`).
    kept_some: Box<[std::sync::atomic::AtomicBool]>,
    /// Where the search for free dynamic slots starts, and which are taken
    /// (by lists not yet reset).
    next: Mutex<(u32, Taken)>,
    /// How many dynamic slots are free (changed under `next`'s lock): with
    /// fewer than a table needs, the search is not made. Once the world
    /// had taken them all, every later table of the frame searched every
    /// slot before it was refused.
    free: std::sync::atomic::AtomicU32,
    /// The shortest table that found no run of free slots since slots were
    /// last given back (changed under `next`'s lock; `u32::MAX`: none).
    /// Until something is given back the free runs only shrink, so a table
    /// of that length or longer is refused without the search. Enough
    /// slots were free, but in pieces, and every such table walked the
    /// whole ring before it was refused: 0.33 ms each with 131,072 slots,
    /// 1.0 s of the recording threads in a frame that refused 3,143 (run
    /// 45), under the lock all of them need. (A free run that lies across
    /// the place the search starts from is seen by neither.)
    too_long: std::sync::atomic::AtomicU32,
    /// View heaps: the bytes per element of the buffer view at each title
    /// slot, for shaders that pick a buffer by an index they compute
    /// (`victory_shader::emit::SLOT_STRIDES`): an upload buffer of
    /// `SLOT_STRIDES_COUNT` dwords, its GPU address and its mapping.
    strides: Option<(ID3D12Resource, u64, usize)>,
    /// CPU-RAM shadow of `strides` dwords to skip redundant Write-Combined writes.
    stride_shadow: Box<[std::sync::atomic::AtomicU32]>,
    /// Title slots holding buffer views. Such a view gets its host slot
    /// (and its memory a host copy) only when something reads it: a draw
    /// through a table, or [`HostHeap::buffer_slots`] for shaders that pick
    /// buffers from the whole heap.
    buffers: Slots,
    /// Of those and of the slots that left the set: the ones written since
    /// [`HostHeap::buffer_changes`] last took them.
    changed: Slots,
    /// Title slots holding a view of a texture the title may fill itself (a
    /// placed texture, a texture object), with the frame from which they
    /// are due: such a texture is uploaded when a draw reads it through a
    /// table; one a shader picks by a computed index is brought up to date
    /// from here ([`HostHeap::texture_slots_due`]).
    textures: std::sync::Arc<Mutex<TextureQueue>>,
    /// Its slots ([`Queued`]).
    queued: std::sync::Arc<Queued>,
    /// Per title slot, the frame (plus one; 0: never) in which a draw last
    /// read it through a table, or made a table it lies in (with
    /// [`TABLED_READ`] set if the draw read it). The draw that
    /// reads such a slot brings its texture up to date itself and the slot
    /// then leaves `textures`; a write of a slot that was a table's within
    /// the last [`TABLED_FRAMES`] frames is due only [`TABLED_GRACE`]
    /// frames later, by when its draw has read it. A title that builds a
    /// table per draw writes thousands of texture views a frame into slots
    /// only tables read (the title: some 2,500 a frame in its
    /// opening cutscene, a few hundred textures over and over), and every
    /// one of them stood in the queue of textures picked by index, ahead
    /// of or behind the ones shaders do pick by index.
    tabled: Vec<std::sync::atomic::AtomicU32>,
    /// The title memory each placed texture view of this heap describes,
    /// by slot and by address ([`Spans`]): which slots a newer view of
    /// other memory there makes stale, and which to look at again when the
    /// runtime maps, unmaps or writes memory ([`remapped`]).
    placed: std::sync::Arc<Mutex<Spans>>,
    /// The slots that may be in `placed` (a slot noted there is put here
    /// under its lock, and taken out when it is forgotten there; a slot
    /// another's view made stale stays, which says only "may"): a write of
    /// any other view needs nothing of `placed` ([`spans_quick`]).
    spanned: Slots,
    /// Dynamic writable placed-texture slots (`resource[12] & 4 != 0`) tracked
    /// for bindless owner-switch invalidation: slot -> `(record_key, owner_id, host_raw)`.
    /// Updated under `self.made.lock(index)` when the slot is built/cleared;
    /// scanned only when `resource::all_target_changes()` advances past
    /// `last_target_gen` or `placed_recheck_needed` is set.
    placed_owners: Mutex<std::collections::HashMap<u32, (u64, u64, usize)>>,
    /// Last observed [`crate::resource::all_target_changes`] generation on this heap.
    last_target_gen: std::sync::atomic::AtomicU64,
    /// Set whenever a slot publishes an owner resolved at an older target
    /// generation than `all_target_changes()` or `last_target_gen`, closing
    /// the concurrent publication race against `texture_slots_target_owners`.
    placed_recheck_needed: std::sync::atomic::AtomicBool,
    /// Serializes per-heap draining of [`UAV_WRITTEN`] so `uav_cursor` only
    /// advances after `slots_uav_written` finishes queueing affected slots.
    uav_drain: Mutex<()>,
    /// Count of in-progress per-heap [`UAV_WRITTEN`] drains; nonzero blocks
    /// [`HostHeap::bake_covered`] until invalidation has landed in `textures`.
    uav_draining: std::sync::atomic::AtomicUsize,
    /// Per-heap read cursor into [`UAV_WRITTEN`] (`VICTORY_UAV_RECHECK=1`).
    uav_cursor: std::sync::atomic::AtomicU64,
    /// Per-heap last frame that drained [`UAV_WRITTEN`].
    uav_frame: std::sync::atomic::AtomicU64,
}

/// Whether a slot's write of no placed texture view leaves `placed` alone
/// when the slot is not in it ([`HostHeap::spanned`]; `VICTORY_SPANS_QUICK=0`:
/// its lock is taken for every write, as before).
///
/// Every record the pass over records copied as bytes mirrors went by that
/// one lock of the heap, from six threads, with the title's own threads and
/// the texture passes: 4 of the 16 ms a frame the submitting thread waited
/// for that pass were its threads waiting there (run city4, timeline 113).
/// What is done is the same: forgetting a slot that is not there does
/// nothing.
///
/// Run heapsync-ab3 (A/B in the street): 13,300 of 13,500 such writes a
/// frame left it alone; making views 1.7 ms a frame against 2.4, the pass
/// 11.7-12.0 ms against 13.2-13.8; the picture the same.
/// Since the last frame line: writes that left `placed` alone
/// ([`spans_quick`]), writes that took its lock.
pub(crate) static SPANS_QUICK: [std::sync::atomic::AtomicU64; 2] = [const { std::sync::atomic::AtomicU64::new(0) }; 2];

fn spans_quick() -> bool {
    #[cfg(feature = "diagnostics")]
    if let Some(on) = crate::ab::pick("VICTORY_SPANS_QUICK") {
        return on;
    }
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_SPANS_QUICK").is_none_or(|v| v != "0"))
}

/// The spans of every view heap made ([`HostHeap::placed`]), for the
/// texture code, which has no heap at hand ([`next_start`],
/// [`first_viewed`]).
static PLACED_SPANS: Mutex<Vec<std::sync::Weak<Mutex<Spans>>>> = Mutex::new(Vec::new());

/// Where the next texture placed inside `[start, end)` starts, past the
/// first bytes of the one at `start` ([`HEAD`]): what is there of the
/// texture at `start` ends there, whatever its full size. `None`: no view
/// in a slot shaders pick by index describes memory in that range.
pub(crate) fn next_start(start: u64, end: u64) -> Option<u64> {
    let heaps = PLACED_SPANS.lock().unwrap_or_else(|e| e.into_inner());
    heaps
        .iter()
        .filter_map(std::sync::Weak::upgrade)
        .filter_map(|spans| spans.lock().unwrap_or_else(|e| e.into_inner()).next_start(start, end))
        .min()
}

/// The lowest level that a view of the placed texture at `start` shows,
/// over the slots shaders pick by index that hold one now. The title
/// views a texture from the first level it holds. `None`: no such slot.
pub(crate) fn first_viewed(start: u64) -> Option<u32> {
    let heaps = PLACED_SPANS.lock().unwrap_or_else(|e| e.into_inner());
    heaps
        .iter()
        .filter_map(std::sync::Weak::upgrade)
        .filter_map(|spans| spans.lock().unwrap_or_else(|e| e.into_inner()).first_viewed(start))
        .min()
}

/// Levels on demand (`crate::tiles`): where the placed textures start
/// whose first level with memory changed since a texture pass last took
/// them ([`HostHeap::texture_views_remade`]).
static RESHOWN: Mutex<Vec<u64>> = Mutex::new(Vec::new());

/// The first level with memory of the placed texture at `start` changed:
/// its slots are made again at the next texture pass.
pub(crate) fn levels_changed(start: u64) {
    RESHOWN.lock().unwrap_or_else(|e| e.into_inner()).push(start);
}

/// [`levels_changed`] once the submissions begun so far (that of the list
/// whose step calls this among them) have run.
pub(crate) fn levels_changed_after(start: u64) {
    let after = crate::host::submissions().0;
    RESHOWN_AFTER.lock().unwrap_or_else(|e| e.into_inner()).push((after, start));
}

static RESHOWN_AFTER: Mutex<Vec<(u64, u64)>> = Mutex::new(Vec::new());

/// Slots that show a placed texture from another level than the title's view
/// in them says now, and are made from that view when the upload it needs
/// has run ([`HostHeap::texture_slot_flip`]).
struct Flip {
    /// The submission to wait for.
    after: u64,
    /// The heap (its address), the slot, where its record is and which
    /// record it was.
    heap: usize,
    index: u32,
    handle: u64,
    key: u64,
    /// The frame it was asked in, of a slot that shows nothing of the
    /// texture until then ([`HostHeap::texture_slot_show_after`]).
    late: Option<u64>,
    /// When it was asked (`crate::host::now_ms`), for `crate::texlate`.
    asked: u64,
}

/// Whether a title slot shows a texture that is new to it only once the
/// submission with its upload has finished (`VICTORY_SHOW_AFTER=1`; off).
/// Shown at once, the lists already on the card read it by index while it is
/// made: all three losses whose report had the address book (n08, n11, n15)
/// were page faults in a texture made and shown in the last tenth of a
/// second.
pub(crate) fn show_after() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_SHOW_AFTER").is_some_and(|v| v != "0"))
}

/// Whether a title slot that would show a texture made while submissions
/// begun before it may still run shows it only once those have finished
/// (`VICTORY_SHOW_NEW_LATER`, on; 0: at once, as before; A/B-able).
///
/// [`show_after`] left out slots made "at once" from a draw's table, the
/// texture pass's paths that are no upload of theirs, and the heap's own
/// look at buffer views: the device losses of steps-ab2, start-3 and the
/// user's play-1125-2107 (no ray tracing) were all page faults at the first
/// byte of a texture in a block made that millisecond, its slot changed "at
/// once" while lists sent before read that slot by index (its neighbours,
/// shown after their upload, did not fault).
pub(crate) fn show_new_later() -> bool {
    #[cfg(feature = "diagnostics")]
    if let Some(on) = crate::ab::pick("VICTORY_SHOW_NEW_LATER") {
        return on;
    }
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_SHOW_NEW_LATER").is_none_or(|v| v != "0"))
}

/// Slots [`show_new_later`] holds back now: heap, slot, record.
static NEW_PENDING: std::sync::LazyLock<Mutex<std::collections::HashSet<(usize, u32, u64)>>> =
    std::sync::LazyLock::new(Default::default);

/// Slots held back by [`show_new_later`] since the start.
pub(crate) static NEW_LATER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// The submissions begun when the host texture `view` shows was made, if
/// one begun before that may still run ([`crate::pool::begun_before`]). The
/// host texture is made here if it is not yet, as [`make`] would.
fn begun_before(view: &View) -> Option<u64> {
    // (Resolved first: the texture made here, a moment ago, is the case.)
    // Keep `_placed` (and its `ResourceRef` owner) alive across `host_resource`.
    let _placed;
    let shader = match view {
        View::Placed(false, resource, _) if resource[0] != 1 => {
            _placed = placed_texture(view)?;
            &_placed.view
        }
        View::Shader(..) => view,
        _ => return None,
    };
    let &View::Shader(resource, _) = shader else {
        return None;
    };
    let texture = crate::resource::host_resource(resource as crate::object::This)?;
    crate::pool::begun_before(texture)
}

/// Slots shown late ([`show_after`]), the frames they waited, the longest;
/// those that were given the neutral texture meanwhile; those not shown any
/// more when their time came.
static LATE: [std::sync::atomic::AtomicU64; 5] = [const { std::sync::atomic::AtomicU64::new(0) }; 5];

/// What a texture slot showed before its last write: its record (0: null or
/// the neutral texture), when (ms), and the submissions begun and finished.
#[cfg(feature = "diagnostics")]
static BEFORE: Mutex<std::collections::BTreeMap<u32, (u64, u64, (u64, u64), &'static str)>> =
    Mutex::new(std::collections::BTreeMap::new());

#[cfg(feature = "diagnostics")]
fn note_before(index: u32, was: u64, by: &'static str) {
    if crate::book::on() {
        let entry = (was, crate::host::now_ms(), crate::host::submissions(), by);
        BEFORE.lock().unwrap_or_else(|e| e.into_inner()).insert(index, entry);
    }
}

/// A 1x1 gray texture, in the place of one that is not there yet.
fn neutral() -> Option<&'static ID3D12Resource> {
    static TEXTURE: std::sync::OnceLock<Option<ID3D12Resource>> = std::sync::OnceLock::new();
    TEXTURE
        .get_or_init(|| {
            let host = host()?;
            let device = &host.device;
            let (from, data) = host.upload_buffer(256)?;
            // SAFETY: the buffer's mapping; plain creation and recording,
            // waited for before what it names goes.
            unsafe {
                std::ptr::write_bytes(data, 0x80, 256);
                let mut made: Option<ID3D12Resource> = None;
                device
                    .CreateCommittedResource(
                        &D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() },
                        D3D12_HEAP_FLAG_NONE,
                        &D3D12_RESOURCE_DESC {
                            Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
                            Width: 1,
                            Height: 1,
                            DepthOrArraySize: 1,
                            MipLevels: 1,
                            Format: DXGI_FORMAT_R8G8B8A8_UNORM,
                            SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                            ..Default::default()
                        },
                        D3D12_RESOURCE_STATE_COMMON,
                        None,
                        &mut made,
                    )
                    .ok()?;
                let made = made?;
                let queue: ID3D12CommandQueue = device
                    .CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC { Type: D3D12_COMMAND_LIST_TYPE_COPY, ..Default::default() })
                    .ok()?;
                let allocator: ID3D12CommandAllocator = device.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COPY).ok()?;
                let list: ID3D12GraphicsCommandList =
                    device.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_COPY, &allocator, None).ok()?;
                let to = D3D12_TEXTURE_COPY_LOCATION {
                    pResource: std::mem::transmute_copy(&made),
                    Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
                    Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: 0 },
                };
                let footprint = D3D12_PLACED_SUBRESOURCE_FOOTPRINT {
                    Offset: 0,
                    Footprint: D3D12_SUBRESOURCE_FOOTPRINT {
                        Format: DXGI_FORMAT_R8G8B8A8_UNORM,
                        Width: 1,
                        Height: 1,
                        Depth: 1,
                        RowPitch: 256,
                    },
                };
                let source = D3D12_TEXTURE_COPY_LOCATION {
                    pResource: std::mem::transmute_copy(&from),
                    Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
                    Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { PlacedFootprint: footprint },
                };
                list.CopyTextureRegion(&to, 0, 0, 0, &source, None);
                list.Close().ok()?;
                let fence: ID3D12Fence = device.CreateFence(0, D3D12_FENCE_FLAG_NONE).ok()?;
                queue.ExecuteCommandLists(&[Some(windows::core::Interface::cast(&list).ok()?)]);
                queue.Signal(&fence, 1).ok()?;
                // (Without an event: this waits.)
                fence.SetEventOnCompletion(1, windows::Win32::Foundation::HANDLE::default()).ok()?;
                Some(made)
            }
        })
        .as_ref()
}

/// Makes the descriptor of [`neutral`] for a slot of `view` at `at`, if the
/// view is a 2D texture's or an array's; else a null one of its kind.
fn neutral_for(view: &View, at: D3D12_CPU_DESCRIPTOR_HANDLE) -> Option<ID3D12Resource> {
    const TEXTURE2D: u32 = 4;
    const TEXTURE2D_ARRAY: u32 = 5;
    let dimension = match view {
        View::Shader(_, Some(desc)) => desc[1],
        View::Placed(false, resource, desc) if resource[0] != 1 => desc[1],
        _ => 0,
    };
    let desc = match dimension {
        TEXTURE2D => D3D12_SHADER_RESOURCE_VIEW_DESC {
            Format: DXGI_FORMAT_R8G8B8A8_UNORM,
            ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2D,
            Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
            Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                Texture2D: D3D12_TEX2D_SRV { MipLevels: 1, ..Default::default() },
            },
        },
        TEXTURE2D_ARRAY => D3D12_SHADER_RESOURCE_VIEW_DESC {
            Format: DXGI_FORMAT_R8G8B8A8_UNORM,
            ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2DARRAY,
            Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
            Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                Texture2DArray: D3D12_TEX2D_ARRAY_SRV { MipLevels: 1, ArraySize: 1, ..Default::default() },
            },
        },
        _ => {
            null_for(view, at);
            return None;
        }
    };
    let (Some(texture), Some(host)) = (neutral(), host()) else {
        null_for(view, at);
        return None;
    };
    // SAFETY: a slot of our heap, a live texture.
    unsafe { host.device.CreateShaderResourceView(texture, Some(&desc), at) };
    Some(texture.clone())
}

static FLIPS: Mutex<Vec<Flip>> = Mutex::new(Vec::new());

/// Makes the slots whose new texture's upload has run
/// ([`HostHeap::texture_slots_flip`]); on any thread. Not if the heap is at
/// something else: the next texture pass does.
pub(crate) fn flip_now() {
    let _ = super::with_view_heap(HostHeap::texture_slots_flip);
}

/// Since the start: slots made from their new view after its upload had run,
/// and those the title had written again by then.
pub(crate) static FLIPPED: [std::sync::atomic::AtomicU64; 2] = [const { std::sync::atomic::AtomicU64::new(0) }; 2];

/// The key of the record at `handle`, a view's descriptor slot; 0: none.
fn record_key(handle: u64) -> u64 {
    // SAFETY: the handle is a descriptor slot in title memory.
    let [tag, i, generation] = unsafe { (handle as *const [u32; 3]).read_unaligned() };
    if tag == VIEW_TAG { u64::from(generation) << 32 | u64::from(i) } else { 0 }
}

/// Levels on demand (`crate::tiles`): where the placed textures start
/// whose last upload left a held level without tiles (the card had no
/// room, or no heap could be had), and when (`crate::host::moment`). A
/// texture pass makes their slots due again once the card has room
/// ([`HostHeap::texture_slots_waiting`]).
static WAITING: Mutex<std::collections::BTreeMap<u64, u64>> = Mutex::new(std::collections::BTreeMap::new());

/// The last upload of the placed texture at `start` left a held level
/// without tiles. A texture picked by index is uploaded once and then
/// only when its memory is found changed, and its staging buffer goes
/// with the upload: without this the level stayed empty (the texture
/// blurred, or black if no level had tiles) for as long as the title left
/// its memory alone, where a texture refused for want of room is tried
/// again.
pub(crate) fn levels_waiting(start: u64) {
    WAITING.lock().unwrap_or_else(|e| e.into_inner()).insert(start, crate::host::moment());
}

/// Placed textures (by the start of their memory) filled while the runtime
/// was remapping with less than was mapped of them
/// (`crate::texture::fill_unlocked`), whose slots the next texture pass
/// makes due again; and how often in a row that was, which ends it: memory
/// that is shown and cannot be read would be uploaded again every frame.
static SHORT: Mutex<(std::collections::BTreeSet<u64>, std::collections::BTreeMap<u64, u32>)> =
    Mutex::new((std::collections::BTreeSet::new(), std::collections::BTreeMap::new()));

/// Fillings in a row of one texture that got less than was mapped, after
/// which its slots wait for the next notice of its memory.
const SHORT_IN_A_ROW: u32 = 8;

/// The placed texture at `start` was filled with less than was mapped.
pub(crate) fn read_short(start: u64) {
    let mut short = SHORT.lock().unwrap_or_else(|e| e.into_inner());
    let times = short.1.entry(start).or_insert(0);
    *times += 1;
    if *times <= SHORT_IN_A_ROW {
        short.0.insert(start);
    }
}

/// The placed texture at `start` was filled with all that was mapped.
pub(crate) fn read_all(start: u64) {
    let mut short = SHORT.lock().unwrap_or_else(|e| e.into_inner());
    if !short.1.is_empty() {
        short.1.remove(&start);
    }
}

/// Title memory by the placed texture views that describe it: per span of
/// memory (its start, its end at full size) the title slots holding a
/// view of a texture that takes exactly that span. A console title need
/// not clear a descriptor when it streams a texture out, so a newer view
/// that cannot be there together with an older one is all that says the
/// older one is stale ([`Rule`] says when that is). Views of one span with
/// different descriptions (the formats of one family, say) are one
/// texture's.
///
/// Spans may overlap: a texture's smallest levels lie at the start of its
/// memory, and a title that holds only those puts the next texture where
/// the larger ones would be (the title: three 512x256 textures
/// of 0x2c000 bytes each whose starts are 0x2000 and 0x4000 apart, all
/// three read by the same frame's draws, run 188). What is there of a
/// texture ends where the next one starts.
/// No host heap is needed, so it is tested without one.
#[derive(Default)]
pub(crate) struct Spans {
    /// When a newer view makes an older one stale.
    rule: Rule,
    by_start: std::collections::BTreeMap<u64, (u64, std::collections::BTreeSet<u32>)>,
    by_slot: std::collections::HashMap<u32, u64>,
    /// The level each slot's view starts at, and where what it is read from
    /// ends ([`Spans::shows`]).
    first: std::collections::HashMap<u32, (u32, u64)>,
    /// The bytes a placed texture takes, by resource description: worked
    /// out once for each (a title writes the views of a few thousand
    /// descriptions, thousands of times a frame).
    sizes: std::collections::HashMap<[u32; 14], Option<u64>>,
}

/// When a view the title writes makes an older view of other memory stale.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) enum Rule {
    /// When it starts where the older one starts: at the same address
    /// with another length, or within the older one's first [`HEAD`]
    /// bytes (or the older within its own). No two textures share the
    /// block of their smallest levels. A view that only reaches into the
    /// address range of an older texture's larger levels says nothing.
    #[default]
    Start,
    /// Whenever their spans at full size overlap (the rule until run
    /// 188, where it made null the slots of textures packed as above:
    /// each view written took its neighbours out of the queue and off the
    /// card, for good). `VICTORY_STALE_TEXTURES=overlap`.
    Overlap,
    /// Never; a view that would be stale under [`Rule::Overlap`] is left
    /// out of the spans instead. `VICTORY_STALE_TEXTURES=0`, as before.
    Never,
}

/// The first bytes of a placed texture that are its own under
/// [`Rule::Start`]: the 4 KiB block its smallest levels share. (Of 129
/// pairs of overlapping neighbours in the listings of runs 188 and 189,
/// none starts closer than 0x2000 to the one before it.)
const HEAD: u64 = 4096;

impl Spans {
    /// The bytes of title memory a placed texture with the resource
    /// description `desc` takes (`crate::resource::placed_bytes`).
    fn bytes(&mut self, desc: &[u32; 14]) -> Option<u64> {
        if let Some(&known) = self.sizes.get(desc) {
            return known;
        }
        let bytes = crate::resource::placed_bytes(desc);
        self.sizes.insert(*desc, bytes);
        bytes
    }

    /// The starts of the spans that overlap `[start, end)`, each taken to
    /// end where the next one starts: the last span that starts before
    /// `start`, if it reaches past it at full size, and every span that
    /// starts inside.
    fn overlapping(&self, start: u64, end: u64) -> Vec<u64> {
        let before = self
            .by_start
            .range(..start)
            .next_back()
            .filter(|(_, span)| span.0 > start)
            .map(|(at, _)| *at);
        before.into_iter().chain(self.by_start.range(start..end).map(|(at, _)| *at)).collect()
    }

    /// Slot `index` holds a view of a texture in `[start, end)`. `newer`:
    /// the title wrote it now, so the slots whose views cannot be there
    /// with it ([`Rule`]) are stale: they are returned, and forgotten.
    /// Otherwise (a draw made the slot from what it held), and under
    /// [`Rule::Never`], a view that clashes with other spans is left out.
    fn note(&mut self, index: u32, start: u64, end: u64, newer: bool) -> Vec<u32> {
        let same = |span: Option<&(u64, std::collections::BTreeSet<u32>)>| span.is_some_and(|s| s.0 == end);
        if self.by_slot.get(&index) == Some(&start) && same(self.by_start.get(&start)) {
            return Vec::new();
        }
        self.forget(index);
        if end <= start {
            return Vec::new();
        }
        let rule = self.rule;
        // The spans the view cannot be there with.
        let clashing: Vec<u64> = match rule {
            Rule::Start => self
                .by_start
                .range(start.saturating_sub(HEAD - 1)..start.saturating_add(HEAD))
                .map(|(at, _)| *at)
                .collect(),
            Rule::Overlap | Rule::Never => self.overlapping(start, end),
        };
        let mut stale = Vec::new();
        for at in clashing {
            if at == start && same(self.by_start.get(&at)) {
                continue;
            }
            if !newer || rule == Rule::Never {
                return stale;
            }
            if let Some((old_end, slots)) = self.by_start.remove(&at) {
                // The first of them are said with both spans, to tell a
                // replacement from a neighbour in one launch's log.
                static SAID: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
                if SAID.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 48 {
                    tracing::info!(
                        slot = index,
                        start = format_args!("{start:#x}"),
                        end = format_args!("{end:#x}"),
                        stale_start = format_args!("{at:#x}"),
                        stale_end = format_args!("{old_end:#x}"),
                        stale_slots = slots.len(),
                        rule = ?rule,
                        "a placed texture view makes older views of other memory stale"
                    );
                }
                for slot in slots {
                    self.by_slot.remove(&slot);
                    stale.push(slot);
                }
            }
        }
        self.by_start.entry(start).or_insert_with(|| (end, Default::default())).1.insert(index);
        self.by_slot.insert(index, start);
        stale
    }

    /// The view in slot `index` starts at level `first` of its texture; the
    /// memory of that and the smaller levels ends at `end`.
    fn shows(&mut self, index: u32, first: u32, end: u64) {
        self.first.insert(index, (first, end));
    }

    /// Whether slot `index`'s view is read from memory at or past `start`.
    fn reaches(&self, index: u32, start: u64) -> bool {
        self.first.get(&index).is_none_or(|&(_, end)| end > start)
    }

    /// The lowest level a view of the texture at `start` starts at.
    fn first_viewed(&self, start: u64) -> Option<u32> {
        self.by_start.get(&start)?.1.iter().filter_map(|slot| self.first.get(slot)).map(|&(first, _)| first).min()
    }

    /// The first span that starts inside `[start, end)` past the first
    /// [`HEAD`] bytes of it.
    fn next_start(&self, start: u64, end: u64) -> Option<u64> {
        self.by_start.range(start.saturating_add(HEAD)..end).next().map(|(at, _)| *at)
    }

    /// The slots that hold a view of the texture at `start`.
    fn slots_at(&self, start: u64) -> Vec<u32> {
        self.by_start.get(&start).map(|span| span.1.iter().copied().collect()).unwrap_or_default()
    }

    /// Slot `index` holds no placed texture view any more.
    fn forget(&mut self, index: u32) {
        self.first.remove(&index);
        let Some(start) = self.by_slot.remove(&index) else {
            return;
        };
        if let Some(span) = self.by_start.get_mut(&start) {
            span.1.remove(&index);
            if span.1.is_empty() {
                self.by_start.remove(&start);
            }
        }
    }

    /// The slots whose texture's memory overlaps `[start, end)`.
    fn slots_in(&self, start: u64, end: u64) -> Vec<u32> {
        self.overlapping(start, end)
            .iter()
            .filter_map(|at| self.by_start.get(at))
            .flat_map(|span| span.1.iter().copied())
            .filter(|&slot| self.reaches(slot, start))
            .collect()
    }

    /// The slots of the textures that hold bytes of `[start, end)`, for a
    /// write there: as [`Spans::slots_in`], but when a texture starts at
    /// `start` itself the one before it is left out (it ends where that
    /// one starts, and a request that fills a texture begins at its first
    /// byte: `slots_in` would name its neighbour with every such write).
    fn slots_holding(&self, start: u64, end: u64) -> Vec<u32> {
        if end <= start {
            return Vec::new();
        }
        let before = if self.by_start.contains_key(&start) {
            None
        } else {
            self.by_start.range(..start).next_back().filter(|(_, span)| span.0 > start).map(|(_, span)| span)
        };
        before
            .into_iter()
            .chain(self.by_start.range(start..end).map(|(_, span)| span))
            .flat_map(|span| span.1.iter().copied())
            .filter(|&slot| self.reaches(slot, start))
            .collect()
    }
}

/// Title memory written behind the guards' back whose bytes are there
/// (DirectStorage's reads into physical pages, at the addresses the title
/// has them mapped; `crate::watch::title_memory_written`), since a texture
/// pass last took the ranges ([`page_written`]). Kept apart from
/// [`REMAPPED`] so that what such writes do is counted by itself.
static PAGE_WRITTEN: Mutex<Vec<(u64, u64)>> = Mutex::new(Vec::new());

/// Since the start: such ranges kept for a pass; ranges dropped because no
/// pass took them; slots of placed textures that held bytes of them when a
/// pass took them; of those, the slots whose texture had been uploaded
/// before the write (out of the queue, or in it after an upload): they are
/// the ones that kept the bytes from before, and the ones made due when
/// `VICTORY_PAGE_WRITES` asks for it.
pub(crate) static PAGE_WRITES: [std::sync::atomic::AtomicU64; 4] =
    [const { std::sync::atomic::AtomicU64::new(0) }; 4];

/// Frames from the pass that takes a written range to the look at its
/// slots: the levels of one texture arrive in a few requests, and a look
/// two frames on takes those of the next frame with it.
const PAGE_WRITE_AFTER: u64 = 2;

/// The bytes of a write the guards do not see are in title memory `[at,
/// at + len)`: the placed textures there changed with no descriptor
/// written and nothing mapped. Kept for the next texture pass
/// ([`HostHeap::texture_slots_page_written`]), as [`remapped`] keeps its
/// ranges; locks one mutex and calls nothing.
pub(crate) fn page_written(at: usize, len: usize) {
    use std::sync::atomic::Ordering::Relaxed;
    /// Ranges kept until a pass takes them.
    const MOST: usize = 1 << 15;
    let (start, end) = (at as u64, at.saturating_add(len) as u64);
    if end <= SPANNED[0].load(Relaxed) || start >= SPANNED[1].load(Relaxed) {
        return;
    }
    let mut ranges = PAGE_WRITTEN.lock().unwrap_or_else(|e| e.into_inner());
    PAGE_WRITES[0].fetch_add(1, Relaxed);
    if let Some(last) = ranges.last_mut()
        && last.1 == start
    {
        last.1 = end;
        return;
    }
    if ranges.len() >= MOST {
        PAGE_WRITES[1].fetch_add(ranges.len() as u64, Relaxed);
        ranges.clear();
    }
    ranges.push((start, end));
}

/// Bounded monotonic log of title-memory spans written on the GPU
/// ([`gpu_written`]), read independently by each heap via its own cursor
/// (`HostHeap::uav_cursor`; opt-in `VICTORY_UAV_RECHECK=1`).
struct UavLog {
    next: u64,
    entries: std::collections::VecDeque<(u64, (u64, u64))>,
}

const UAV_LOG_CAP: usize = 1 << 15;

static UAV_SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

static UAV_WRITTEN: Mutex<UavLog> = Mutex::new(UavLog {
    next: 1,
    entries: std::collections::VecDeque::new(),
});

/// Since the start: GPU writes of title textures noted, ranges dropped,
/// slots of placed textures showing them, slots made due, and peak per frame.
#[cfg_attr(not(feature = "diagnostics"), allow(dead_code))]
pub(crate) static UAV_RECHECK: [std::sync::atomic::AtomicU64; 5] =
    [const { std::sync::atomic::AtomicU64::new(0) }; 5];

/// Slots one frame may make due for GPU-written title textures.
const UAV_DUE_A_FRAME: u64 = 32;

/// Whether a title texture written on the GPU is rechecked by the texture
/// pass (`VICTORY_UAV_RECHECK=1`; opt-in, off by default).
fn uav_recheck() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_UAV_RECHECK").is_some_and(|v| v == "1"))
}

fn push_uav_range(range: (u64, u64)) {
    use std::sync::atomic::Ordering::{Relaxed, Release};
    let mut log = UAV_WRITTEN.lock().unwrap_or_else(|e| e.into_inner());
    if log.entries.len() >= UAV_LOG_CAP {
        log.entries.pop_front();
        UAV_RECHECK[1].fetch_add(1, Relaxed);
    }
    let seq = log.next;
    log.next += 1;
    log.entries.push_back((seq, range));
    UAV_SEQ.store(seq, Release);
}

/// Title texture `resource` was written on the GPU (unordered access or GPU
/// copy): always marks its host copy via `crate::texture::host_written` so
/// placed views alias it (`crate::resource::alias_of`), and when
/// `VICTORY_UAV_RECHECK=1` appends its title-memory span to [`UAV_WRITTEN`]
/// for per-heap bounded recheck.
pub(crate) fn gpu_written(resource: crate::object::This) {
    use std::sync::atomic::Ordering::Relaxed;
    if !crate::resource::title_address(resource) {
        return;
    }
    if !crate::resource::host_written_mark(resource) {
        crate::texture::host_written(resource);
    }
    if !uav_recheck() {
        return;
    }
    let Some((at, len)) = crate::texture::title_span(resource, 0) else {
        return;
    };
    if len == 0 {
        return;
    }
    UAV_RECHECK[0].fetch_add(1, Relaxed);
    push_uav_range((at, at.saturating_add(len)));
}

/// Title memory the runtime mapped, unmapped or is about to write since a
/// texture pass last took the ranges ([`remapped`]).
static REMAPPED: Mutex<Vec<(u64, u64)>> = Mutex::new(Vec::new());

/// The lowest start and the highest end of any span of [`Spans`] so far
/// (the start is `u64::MAX` until there is one): what the runtime does to
/// memory outside is none of the texture pass's business.
static SPANNED: [std::sync::atomic::AtomicU64; 2] = [
    std::sync::atomic::AtomicU64::new(u64::MAX),
    std::sync::atomic::AtomicU64::new(0),
];

/// Since the start: slots whose view a newer view of other memory there
/// made stale; ranges of memory noted as mapped, unmapped or written;
/// ranges dropped because no pass took them; slots made due by such
/// ranges; and made slots whose memory was found unmapped (made null).
pub(crate) static STALE: [std::sync::atomic::AtomicU64; 5] =
    [const { std::sync::atomic::AtomicU64::new(0) }; 5];

/// When a view the title writes makes older views of other memory stale
/// ([`Rule`]): where it starts in the block of their smallest levels, by
/// default; `VICTORY_STALE_TEXTURES=0`: never, as before;
/// `VICTORY_STALE_TEXTURES=overlap`: wherever their spans overlap, the rule
/// until run 188.
fn stale_rule() -> Rule {
    static RULE: std::sync::OnceLock<Rule> = std::sync::OnceLock::new();
    *RULE.get_or_init(|| match std::env::var("VICTORY_STALE_TEXTURES").as_deref() {
        Ok("0") => Rule::Never,
        Ok("overlap") => Rule::Overlap,
        _ => Rule::Start,
    })
}

/// The runtime mapped or unmapped title memory `[at, at + len)`, or is
/// about to write it through the kernel (`crate::watch`): the placed
/// textures there may have arrived, gone, or changed with no descriptor
/// written. The range is kept for the next texture pass
/// ([`HostHeap::texture_slots_remapped`]). Called with the runtime's own
/// locks held: this locks one mutex of its own and calls nothing.
pub(crate) fn remapped(at: usize, len: usize) {
    use std::sync::atomic::Ordering::Relaxed;
    /// Ranges kept until a pass takes them.
    const MOST: usize = 1 << 15;
    let (start, end) = (at as u64, at.saturating_add(len) as u64);
    if end <= SPANNED[0].load(Relaxed) || start >= SPANNED[1].load(Relaxed) {
        return;
    }
    let mut ranges = REMAPPED.lock().unwrap_or_else(|e| e.into_inner());
    STALE[1].fetch_add(1, Relaxed);
    // Pages are mapped one after the other.
    if let Some(last) = ranges.last_mut()
        && last.1 == start
    {
        last.1 = end;
        return;
    }
    if ranges.len() >= MOST {
        // No pass runs (no shader that picks textures by index is bound):
        // the slots there are looked at when they are due, as before.
        STALE[2].fetch_add(ranges.len() as u64, Relaxed);
        ranges.clear();
    }
    ranges.push((start, end));
}

/// Ranges of memory, those that touch made one (DexNom/VIctory 921d8d3b:
/// several threads map and write, some 5,000 ranges a frame, the same
/// textures' over and over, and each was looked up).
fn join(ranges: &mut Vec<(u64, u64)>) {
    ranges.sort_unstable();
    ranges.dedup_by(|next, last| {
        if next.0 > last.1 {
            return false;
        }
        last.1 = last.1.max(next.1);
        true
    });
}

/// Frames for which a slot a draw read through a table counts as a table's
/// slot ([`HostHeap::tabled`]).
const TABLED_FRAMES: u64 = 240;

/// Frames a write of a table's slot waits in the queue for the draw that
/// reads it, before the slot is taken for one shaders pick by index.
/// Frames as `crate::frame::current` counts them: in its opening cutscene
/// The title begins two or three of them per frame it presents
/// (435 begun in the 353 s of run 52's, for some 140 presented).
const TABLED_GRACE: u64 = 8;

/// Frames for which a record copied into a slot a draw read through a
/// table is left to such a draw ([`HostHeap::copied_tabled`]).
const COPIED_FRAMES: u32 = 8;

/// Frames a record copied into any other slot waits for one, before it is
/// taken for one shaders pick by index ([`HostHeap::copied_wait`]).
const COPIED_GRACE: u64 = 4;

/// In [`HostHeap::tabled`]: a draw read the slot in that frame.
const TABLED_READ: u32 = 1 << 31;

/// Since the start: writes of texture views into slots that count as a
/// table's (due later), and slots that left the queue because a draw read
/// them through a table.
pub(crate) static TABLED_WRITES: [std::sync::atomic::AtomicU64; 2] =
    [const { std::sync::atomic::AtomicU64::new(0) }; 2];

/// Whether slots draws read through tables leave the queue of textures
/// picked by index; `VICTORY_TABLED_TEXTURES=1` keeps them in it as
/// before.
fn tabled_out() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_TABLED_TEXTURES").is_none_or(|v| v != "1"))
}

/// What the queue of textures picked by index knows of a title slot.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Waiting {
    /// The frame from which the slot is due.
    due: u64,
    /// The frame it was written in.
    written: u64,
    /// Which write of the queue put it here: the pass hands it back with
    /// what it did, so a slot written again meanwhile stays as written.
    write: u64,
    /// Looks in a row that found nothing to upload (its memory not mapped,
    /// or no more of it than at the last upload).
    looks: u8,
    /// Bytes of its memory that were mapped when it was last uploaded
    /// without all of it there; 0: not uploaded yet.
    mapped: u64,
    /// For a slot uploaded whole and looked at again
    /// ([`TextureQueue::recheck`]): where its memory is (address, bytes),
    /// the fingerprint it had at the upload, how many looks found it
    /// unchanged, and the frame of the upload. A fingerprint of 0: none.
    checked: (u64, u64, u64, u8, u64),
    /// Whole/dense baseline fingerprint captured alongside a `Plain` 3-strip
    /// baseline so the final `Plain` probe can verify the entire texture
    /// before retiring (0 if none or already dense in `checked.2`).
    dense_print: u64,
    /// Non-zero monotonic FIFO sequence assigned on the first budget deferral
    /// (`TextureQueue::defer_recheck`) so deferred `Plain`/`Pending` slots
    /// jump ahead of newer arriving streams on the next pass (starvation-free).
    defer_seq: u64,
    /// When the slot first waited for this (`crate::host::now_ms`), for
    /// `crate::texlate`: kept over writes until it is brought up to date;
    /// and whether it waited for its memory meanwhile.
    since: u64,
    waited: bool,
    /// When a look last found its memory away (or no more of it): 0 none.
    away: u64,
}

/// Recheck policy for a texture slot uploaded whole (`crate::draw`,
/// `VICTORY_LIVE_TEXTURES`; GPL-3.0 base: avatarmanz/VIctory-7900XTX #390).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) enum LiveState {
    /// Never changed after upload: cheap 3-strip fingerprint along
    /// `RECHECK_FRAMES`, then exits the queue.
    #[default]
    Plain,
    /// Changed once after upload (one-shot streaming arrival or 1st video
    /// frame): dense/whole fingerprint (governed by the pass budget) for a
    /// short 4-frame window so a video updating between the 3 strips promotes
    /// to [`Self::Live`], while a one-shot texture exits cleanly to `Done`.
    ChangedOnce,
    /// Changed at least twice: confirmed in-place video/dynamic texture,
    /// polled every frame with dense/whole fingerprinting (budgeted).
    Live,
    /// Previously [`Self::Live`], stayed unchanged for `LIVE_LOOKS` frames
    /// (paused video / static screen): polled periodically with dense/whole
    /// fingerprinting (budgeted) until overwritten or removed; any change
    /// immediately returns it to [`Self::Live`].
    Paused,
}

impl LiveState {
    /// Whether this state uses dense/whole fingerprinting (full up to 16 MiB,
    /// uniform 64×4 KiB grid above 16 MiB) governed by the per-pass live hash
    /// budget, instead of the plain three 16 KiB strips.
    pub(crate) const fn whole_fingerprint(self) -> bool {
        matches!(self, Self::ChangedOnce | Self::Live | Self::Paused)
    }
}

/// Strictly monotonic upload-attempt token generator so two uploads of the
/// same slot even within the exact same frame carry distinct
/// [`LiveBaseline::Pending`] tokens.
static UPLOAD_ATTEMPTS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);

/// Allocates the next non-zero monotonic upload-attempt token.
pub(crate) fn next_upload_attempt() -> u64 {
    UPLOAD_ATTEMPTS.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
}

/// Which fingerprint representation is stored in `Waiting::checked.2`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) enum LiveBaseline {
    /// Plain 3-strip [`crate::texture::fingerprint`].
    #[default]
    Strip,
    /// Dense/whole [`crate::texture::fingerprint_for`] (`dense = true`).
    Dense,
    /// No verified baseline is committed yet (awaiting submit-time
    /// `commit_baseline` for this monotonic `attempt` token, or deferred when
    /// pass hash budget was full); the next pass with budget conservatively
    /// forces an upload and captures a verified baseline (`Strip` or `Dense`).
    Pending(u64),
}

/// The queue of title slots whose texture view waits to be brought up to
/// date for shaders that pick textures by index: which are due, and in
/// which order (no host heap needed, so it is tested without one).
#[derive(Default)]
pub(crate) struct TextureQueue {
    slots: std::collections::BTreeMap<u32, Waiting>,
    /// Each slot write, requeue, async retry or new pass gets a later revision (`TaynanGT #85`).
    revision: u64,
    active_passes: usize,
    /// `(revision, covered_frame)` whose due slots a bake finished serving.
    baked: Option<(u64, u64)>,
    /// Slots held back because the card had no room for the host copy
    /// they need ([`TextureQueue::hold`]), each with the write it was
    /// held as.
    held: std::collections::BTreeMap<u32, u64>,
    /// Which slots are in `slots`, for those who only want one out
    /// ([`Queued`]).
    queued: std::sync::Arc<Queued>,
    /// Recheck state, stored baseline kind, and last-poll sequence for slots
    /// past [`LiveState::Plain`] (`crate::draw`, `VICTORY_LIVE_TEXTURES`;
    /// GPL-3.0 provenance: avatarmanz/VIctory-7900XTX #390).
    live: std::collections::HashMap<u32, (LiveState, LiveBaseline, u64)>,
    /// Monotonic sequence bumped whenever a dense/live (`ChangedOnce`/`Live`/
    /// `Paused` or retiring final-dense `Plain`) slot is updated, so deferred
    /// over-budget slots rotate fairly ahead of recently hashed slots on the
    /// next pass.
    live_polls: u64,
    /// Monotonic FIFO sequence bumped when a `Plain`/`Pending` slot is budget-
    /// deferred (`defer_recheck`), preventing starvation under continuous
    /// streaming of newer higher-write-ID slots.
    defer_ticks: u64,
}

/// Only a pass that did not overlap another can certify a bake. Ordinary
/// passes can defer slots while the bake is reading its snapshot.
pub(crate) struct TexturePass<'a> {
    queue: &'a Mutex<TextureQueue>,
    pub(crate) revision: Option<u64>,
}

impl<'a> TexturePass<'a> {
    fn begin(queue: &'a Mutex<TextureQueue>) -> Self {
        let mut textures = queue.lock().unwrap_or_else(|e| e.into_inner());
        let revision = textures.next_revision();
        textures.active_passes += 1;
        Self { queue, revision: (textures.active_passes == 1).then_some(revision) }
    }
}

impl Drop for TexturePass<'_> {
    fn drop(&mut self) {
        self.queue.lock().unwrap_or_else(|e| e.into_inner()).active_passes -= 1;
    }
}

/// A bit for each slot of a heap's [`TextureQueue`], changed under its lock
/// and read without it: most of the 50,000 descriptors a frame written,
/// looked at and made are not textures, and all of them took that lock to
/// take nothing out of the queue (12 ms a frame of waiting, run city1).
/// Empty: not known.
#[derive(Default)]
pub(crate) struct Queued(Box<[std::sync::atomic::AtomicU64]>);

impl Queued {
    fn new(slots: u32) -> Queued {
        Queued((0..slots.div_ceil(64)).map(|_| std::sync::atomic::AtomicU64::new(0)).collect())
    }

    fn set(&self, slot: u32, there: bool) {
        use std::sync::atomic::Ordering::SeqCst;
        if let Some(word) = self.0.get(slot as usize / 64) {
            let bit = 1 << (slot % 64);
            if there {
                word.fetch_or(bit, SeqCst);
            } else {
                word.fetch_and(!bit, SeqCst);
            }
        }
    }

    /// Whether `slot` may be in the queue.
    fn may(&self, slot: u32) -> bool {
        let word = self.0.get(slot as usize / 64);
        word.is_none_or(|w| w.load(std::sync::atomic::Ordering::SeqCst) >> (slot % 64) & 1 != 0)
    }
}

/// Slots due of a [`TextureQueue`]: `[0]` dense/live (`ChangedOnce`/`Live`/
/// `Paused` and retiring final-dense `Plain` checks), `[1]` plain rechecks &
/// `later`/`partly`, `[2]` fresh writes.
/// Each entry holds `(priority_tier, rank, slot, write)`.
type Due = [Vec<(u8, u64, u32, u64)>; 3];

/// Number of `Plain` probes in `draw::RECHECK_FRAMES`; the final look
/// (`PLAIN_FINAL_LOOK = 4`) runs a dense check against `Waiting::dense_print`
/// before retiring `Plain` when `len > 48 KiB`.
const PLAIN_FINAL_LOOK: u8 = 4;

/// Minimum texture byte size for which 3 × 16 KiB strips leave unprobed gaps.
const PLAIN_STRIP_BYTES: u64 = 3 * (16 << 10);

impl TextureQueue {
    fn next_revision(&mut self) -> u64 {
        self.revision += 1;
        self.revision
    }

    fn bake_covered(&self, frame: u64) -> bool {
        self.baked == Some((self.revision, frame))
    }

    fn finish_bake(&mut self, revision: u64, frame: u64) {
        if self.revision == revision {
            self.baked = Some((revision, frame));
        }
    }

    fn insert(&mut self, index: u32, waiting: Waiting) {
        if self.slots.insert(index, waiting).is_none() {
            self.queued.set(index, true);
        }
        self.live.remove(&index);
    }

    fn take_out(&mut self, index: u32) -> bool {
        let was = self.slots.remove(&index).is_some();
        if was {
            self.queued.set(index, false);
        }
        self.live.remove(&index);
        was
    }

    /// A texture view was written into slot `index` in `frame`, due from
    /// frame `due`. `seen`: it was not written, a draw or the pass made
    /// the host slot from a view the queue may hold already, which then
    /// stays as it is.
    fn written(&mut self, index: u32, frame: u64, due: u64, seen: bool) {
        if seen && self.slots.contains_key(&index) {
            return;
        }
        let write = self.next_revision();
        let (since, waited, away) = self.kept(index);
        self.insert(
            index,
            Waiting {
                due,
                written: frame,
                write,
                looks: 0,
                mapped: 0,
                checked: (0, 0, 0, 0, 0),
                dense_print: 0,
                defer_seq: 0,
                since,
                waited,
                away,
            },
        );
    }

    /// What a slot written again keeps of its wait.
    fn kept(&self, index: u32) -> (u64, bool, u64) {
        self.slots.get(&index).map_or_else(
            || (crate::host::now_ms(), false, 0),
            |w| (w.since, w.waited || w.looks > 0 || w.mapped > 0, w.away),
        )
    }

    /// When slot `index` began to wait, whether it waited for its memory, and
    /// when a look last found that away.
    fn since(&self, index: u32) -> Option<(u64, bool, u64)> {
        self.slots.get(&index).map(|w| (w.since, w.waited || w.looks > 0 || w.mapped > 0, w.away))
    }

    fn remove(&mut self, index: u32) {
        self.take_out(index);
    }

    /// Up to `max` slots due at `frame`, each with its write number, in
    /// the order to serve them:
    /// - first the ones looked at before (their memory was not mapped, or
    ///   only part of it): the look is a query, and memory that has
    ///   arrived since is what the title streamed in last;
    /// - then the others, the newest write first and the oldest next, in
    ///   turns: slot order served the low slots of a backlog of
    ///   25,000-40,000 while the textures of the shot on screen waited
    ///   (run 52: 98% of the drawn pixels had no colour texture in the
    ///   first 3D frames).
    #[cfg(test)]
    fn due(&self, frame: u64, max: usize) -> Vec<(u32, u64)> {
        Self::in_order(self.all_due(frame), max)
    }

    /// The slots due at `frame`, for [`TextureQueue::in_order`].
    fn all_due(&self, frame: u64) -> Due {
        let mut due = Due::default();
        for (&index, w) in self.slots.iter().filter(|(_, w)| w.due <= frame) {
            if let Some(&(state, baseline, poll)) = self.live.get(&index)
                && (state.whole_fingerprint()
                    || (state == LiveState::Plain && baseline == LiveBaseline::Strip))
            {
                due[0].push((0, poll, index, w.write));
            } else if w.looks == 0 && w.mapped == 0 {
                due[2].push((0, 0, index, w.write));
            } else if w.defer_seq != 0 {
                // Tier 0: budget-deferred slots drain strictly FIFO ahead of
                // newer arriving streams (starvation-free).
                due[1].push((0, w.defer_seq, index, w.write));
            } else if matches!(self.live.get(&index), Some(&(_, LiveBaseline::Pending(_), _))) {
                // Tier 1: older `Pending` slots run FIFO by `write` ahead of
                // newer higher-ID streams and routine probes.
                due[1].push((1, w.write, index, w.write));
            } else {
                // Tier 2: regular `Plain` / `later` / `partly` slots, earliest
                // due frame first, oldest write first within the same frame.
                due[1].push((2, w.due, index, w.write));
            }
        }
        due
    }

    /// What [`TextureQueue::due`] says of `due` (with nothing held: some
    /// ten thousand slots are sorted).
    ///
    /// Interleaves all 3 buckets (`again` = `Plain` recheck / `later` /
    /// `partly`, `live` = `ChangedOnce` / `Live` / `Paused` + retiring
    /// final-dense `Plain`, and `fresh` = newest/oldest alternating)
    /// 1-for-1-for-1 (`[again, live, fresh, ...]`), preserving the upstream
    /// rule that an `again` slot leads at index 0 while running `live` ahead
    /// of `fresh` and guaranteeing neither 4,096 `again` nor 4,096 `live`
    /// slots can starve `fresh`.
    ///
    /// Inside `again`, all 3 tiers are sorted strictly oldest-first — Tier 0
    /// (budget-deferred, `defer_seq ASC`), Tier 1 (`Pending`, `write ASC`),
    /// and Tier 2 (regular `Plain`/`later`/`partly`, `(due ASC, write ASC)`) —
    /// and merged in a 4-slot cadence `[T0, T2, T1, T0]` (50% Tier 0, 25%
    /// Tier 1, 25% Tier 2, falling back to non-empty tiers) so even an
    /// unbounded flood past `max` in any tier never starves the other tiers.
    fn in_order([mut live, again, mut fresh]: Due, max: usize) -> Vec<(u32, u64)> {
        live.sort_unstable_by(|a, b| a.1.cmp(&b.1).then_with(|| a.3.cmp(&b.3)));
        let mut tier0: Vec<(u8, u64, u32, u64)> = Vec::new();
        let mut tier1: Vec<(u8, u64, u32, u64)> = Vec::new();
        let mut tier2: Vec<(u8, u64, u32, u64)> = Vec::new();
        for entry in again {
            match entry.0 {
                0 => tier0.push(entry),
                1 => tier1.push(entry),
                _ => tier2.push(entry),
            }
        }
        // Sort each tier strictly oldest-first so newer higher-`write`
        // arrivals can never jump ahead of older entries in the same tier.
        tier0.sort_unstable_by(|a, b| a.1.cmp(&b.1).then_with(|| a.3.cmp(&b.3)));
        tier1.sort_unstable_by(|a, b| a.1.cmp(&b.1).then_with(|| a.3.cmp(&b.3)));
        tier2.sort_unstable_by(|a, b| a.1.cmp(&b.1).then_with(|| a.3.cmp(&b.3)));
        let mut again_order: Vec<(u32, u64)> =
            Vec::with_capacity((tier0.len() + tier1.len() + tier2.len()).min(max));
        let (mut i0, mut i1, mut i2) = (0, 0, 0);
        while again_order.len() < max && (i0 < tier0.len() || i1 < tier1.len() || i2 < tier2.len())
        {
            let pick = match again_order.len() % 4 {
                1 => {
                    if i2 < tier2.len() {
                        2
                    } else if i0 < tier0.len() {
                        0
                    } else {
                        1
                    }
                }
                2 => {
                    if i1 < tier1.len() {
                        1
                    } else if i0 < tier0.len() {
                        0
                    } else {
                        2
                    }
                }
                _ => {
                    if i0 < tier0.len() {
                        0
                    } else if i1 < tier1.len() {
                        1
                    } else {
                        2
                    }
                }
            };
            match pick {
                0 => {
                    again_order.push((tier0[i0].2, tier0[i0].3));
                    i0 += 1;
                }
                1 => {
                    again_order.push((tier1[i1].2, tier1[i1].3));
                    i1 += 1;
                }
                _ => {
                    again_order.push((tier2[i2].2, tier2[i2].3));
                    i2 += 1;
                }
            }
        }
        fresh.sort_unstable_by_key(|a| std::cmp::Reverse(a.3));
        let mut fresh_order: Vec<(u32, u64)> = Vec::with_capacity(fresh.len().min(max));
        let (mut new, mut old) = (0, fresh.len());
        while new < old && fresh_order.len() < max {
            fresh_order.push((fresh[new].2, fresh[new].3));
            new += 1;
            if new < old && fresh_order.len() < max {
                old -= 1;
                fresh_order.push((fresh[old].2, fresh[old].3));
            }
        }
        let mut order: Vec<(u32, u64)> =
            Vec::with_capacity(max.min(again_order.len() + live.len() + fresh_order.len()));
        let (mut ag, mut lv, mut fr) = (0, 0, 0);
        while order.len() < max
            && (ag < again_order.len() || lv < live.len() || fr < fresh_order.len())
        {
            if ag < again_order.len() {
                order.push(again_order[ag]);
                ag += 1;
            }
            if order.len() < max && lv < live.len() {
                order.push((live[lv].2, live[lv].3));
                lv += 1;
            }
            if order.len() < max && fr < fresh_order.len() {
                order.push(fresh_order[fr]);
                fr += 1;
            }
        }
        order
    }

    /// The entry of `index` if it is still the write the pass was handed.
    fn same(&mut self, index: u32, write: u64) -> Option<&mut Waiting> {
        self.slots.get_mut(&index).filter(|w| w.write == write)
    }

    /// Slot `index` is up to date: out of the queue until written again.
    fn done(&mut self, index: u32, write: u64) {
        if self.same(index, write).is_some() {
            self.take_out(index);
        }
    }

    /// Nothing to upload at slot `index` now: looked at again after 1, 2,
    /// 4, ... frames, at most `most`.
    fn later(&mut self, index: u32, write: u64, frame: u64, most: u64) {
        if let Some(w) = self.same(index, write) {
            w.due = frame + (1u64 << w.looks.min(16)).min(most.max(1));
            w.looks = w.looks.saturating_add(1);
            w.away = crate::host::now_ms();
            w.checked = (0, 0, 0, 0, 0);
            w.dense_print = 0;
            w.defer_seq = 0;
            self.live.remove(&index);
        }
    }

    /// The card has no room for the host copy slot `index` needs: it is
    /// not handed out again before frame `due`, or before the card has
    /// room again ([`TextureQueue::room`]). Nothing else of it changes: it
    /// is not a slot "waiting for its memory", and when it is due it has
    /// its old place among the slots never looked at. (It stayed due, so
    /// every pass on a full card looked at the same few hundred slots
    /// again and stopped there, and the slots behind them were never
    /// reached: the ones whose host copy exists and only wants bringing up
    /// to date, and the ones whose memory went, which is what lets host
    /// copies go and makes room.)
    fn hold(&mut self, index: u32, write: u64, due: u64) {
        if let Some(w) = self.same(index, write) {
            w.due = w.due.max(due);
            self.held.insert(index, write);
        }
    }

    /// The card has room again at `frame`: the slots held back for lack of
    /// it are due now, as if they had never been held. Returns how many.
    fn room(&mut self, frame: u64) -> usize {
        let held = std::mem::take(&mut self.held);
        let mut back = 0;
        for (index, write) in held {
            if let Some(w) = self.same(index, write) {
                w.due = w.due.min(frame);
                back += 1;
            }
        }
        if back > 0 {
            self.next_revision();
        }
        back
    }

    /// Slot `index` was uploaded with `mapped` bytes of its memory there,
    /// not all of it: looked at again from frame `due`.
    fn partly(&mut self, index: u32, write: u64, due: u64, mapped: u64) {
        if let Some(w) = self.same(index, write) {
            (w.due, w.looks, w.mapped) = (due, 0, mapped.max(1));
            w.checked = (0, 0, 0, 0, 0);
            w.dense_print = 0;
            w.defer_seq = 0;
            self.live.remove(&index);
        }
    }

    /// Slot `index` was uploaded whole from memory `at` (address, bytes)
    /// with fingerprint `print` (and optional `dense_print` for final `Plain`
    /// verification): looked at again from frame `due`, for a title that fills
    /// a texture's memory after it wrote the view (the `looks`-th such look;
    /// `uploaded` is the upload's frame; `state`: [`LiveState`]; `baseline`:
    /// [`LiveBaseline`]).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn recheck(
        &mut self,
        index: u32,
        write: u64,
        due: u64,
        at: (u64, u64),
        print: u64,
        dense_print: u64,
        looks: u8,
        uploaded: u64,
        state: LiveState,
        baseline: LiveBaseline,
    ) {
        if let Some(w) = self.same(index, write) {
            (w.due, w.looks, w.mapped) = (due, 1, 0);
            w.checked = (at.0, at.1, print, looks, uploaded);
            w.dense_print = dense_print;
            w.defer_seq = 0;
            let final_plain_dense = state == LiveState::Plain
                && baseline == LiveBaseline::Strip
                && looks >= PLAIN_FINAL_LOOK
                && at.1 > PLAIN_STRIP_BYTES;
            if state == LiveState::Plain && baseline == LiveBaseline::Strip && !final_plain_dense {
                self.live.remove(&index);
            } else {
                self.live_polls += 1;
                let poll = self.live_polls;
                self.live.insert(index, (state, baseline, poll));
            }
        }
    }

    /// Defers a recheck slot to `due` when the pass's recheck byte budget is
    /// exhausted, preserving its `checked` state, `looks`, `baseline`, and
    /// older `live_polls` rank, and stamping a FIFO `defer_seq` so deferred
    /// `Plain`/`Pending` slots drain ahead of newer arriving streams.
    pub(crate) fn defer_recheck(&mut self, index: u32, write: u64, due: u64) {
        if let Some(w) = self.slots.get_mut(&index).filter(|w| w.write == write) {
            w.due = w.due.max(due);
            if w.defer_seq == 0 {
                self.defer_ticks += 1;
                w.defer_seq = self.defer_ticks;
            }
        }
    }

    /// Commits a submit-time baseline fingerprint computed directly from the
    /// exact `tiled` guest bytes uploaded to the GPU (`Staging::on_uploaded`),
    /// upgrading the record-time `Pending(attempt)` placeholder ONLY if slot
    /// `index` is still on `write` and its current baseline matches
    /// `LiveBaseline::Pending(attempt)` identically.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn commit_baseline(
        &mut self,
        index: u32,
        write: u64,
        attempt: u64,
        due: u64,
        print: u64,
        dense_print: u64,
        state: LiveState,
        baseline: LiveBaseline,
    ) {
        if !self
            .live
            .get(&index)
            .is_some_and(|&(_, b, _)| b == LiveBaseline::Pending(attempt))
        {
            return;
        }
        if let Some(w) = self.same(index, write)
            && w.checked.2 != 0
        {
            let pulled_earlier = due < w.due;
            w.due = due;
            w.checked.2 = print;
            w.dense_print = dense_print;
            w.defer_seq = 0;
            let final_plain_dense = state == LiveState::Plain
                && baseline == LiveBaseline::Strip
                && w.checked.3 >= PLAIN_FINAL_LOOK
                && w.checked.1 > PLAIN_STRIP_BYTES;
            if state == LiveState::Plain && baseline == LiveBaseline::Strip && !final_plain_dense {
                self.live.remove(&index);
            } else {
                let poll = self.live.get(&index).map_or_else(
                    || {
                        self.live_polls += 1;
                        self.live_polls
                    },
                    |&(_, _, poll)| poll,
                );
                self.live.insert(index, (state, baseline, poll));
            }
            if pulled_earlier {
                self.next_revision();
            }
        }
    }

    /// Pulls a still-`Pending(attempt)` slot forward to `due` when an async
    /// ahead upload fails before submission or misses its upload snapshot.
    pub(crate) fn retry_pending(&mut self, index: u32, write: u64, attempt: u64, due: u64) {
        if self
            .live
            .get(&index)
            .is_some_and(|&(_, b, _)| b == LiveBaseline::Pending(attempt))
            && let Some(w) = self.same(index, write)
        {
            w.due = w.due.min(due);
            self.next_revision();
        }
    }

    /// Completes a non-recheck (`VICTORY_HEAP_RECHECK=0` or partial/unmapped)
    /// ahead upload once the copy queue finishes: marks the slot done/partly on
    /// `ok == true`, or pulls its fallback `due` forward to `retry_due` on
    /// pre-submit failure (strictly guarded by `write`).
    pub(crate) fn finish_ahead(
        &mut self,
        index: u32,
        write: u64,
        ok: bool,
        again: Option<(u64, u64)>,
        retry_due: u64,
    ) {
        if self.same(index, write).is_none() {
            return;
        }
        if ok {
            match again {
                Some((frame, mapped)) => self.partly(index, write, frame, mapped),
                None => self.done(index, write),
            }
        } else if let Some(w) = self.same(index, write) {
            w.due = w.due.min(retry_due);
        }
        self.next_revision();
    }

    /// What [`TextureQueue::recheck`] noted of slot `index`, if it is
    /// waiting for such a look, plus its `dense_print`, [`LiveState`], and
    /// [`LiveBaseline`].
    pub(crate) fn checked(
        &self,
        index: u32,
    ) -> Option<(u64, u64, u64, u64, u8, u64, LiveState, LiveBaseline)> {
        let w = self.slots.get(&index).filter(|w| w.checked.2 != 0)?;
        let (at, len, print, looks, uploaded) = w.checked;
        let (state, baseline) = self
            .live
            .get(&index)
            .map_or((LiveState::Plain, LiveBaseline::Strip), |&(s, b, _)| (s, b));
        Some((at, len, print, w.dense_print, looks, uploaded, state, baseline))
    }

    /// The memory slot `index`'s texture lies in was mapped, unmapped or
    /// written in `frame`: the slot is looked at from frame `due` as one
    /// written now, whatever was known of it (a pass at work on it does
    /// not take it out).
    fn changed(&mut self, index: u32, frame: u64, due: u64) {
        let write = self.next_revision();
        let due = self.slots.get(&index).map_or(due, |w| w.due.min(due));
        let (since, waited, away) = self.kept(index);
        self.insert(
            index,
            Waiting {
                due,
                written: frame,
                write,
                looks: 0,
                mapped: 0,
                checked: (0, 0, 0, 0, 0),
                dense_print: 0,
                defer_seq: 0,
                since,
                waited,
                away,
            },
        );
    }

    /// Bytes of slot `index`'s memory that were mapped at its last upload
    /// (0: not uploaded, or all of it was).
    fn mapped(&self, index: u32) -> u64 {
        self.slots.get(&index).map_or(0, |w| w.mapped)
    }

    /// For the frame line: slots in the queue; of them due at `frame`;
    /// waiting for their memory (looked at, never uploaded); uploaded in
    /// part; and the frames the oldest due one has waited since its write.
    fn census(&self, frame: u64) -> [u64; 5] {
        let mut c = [self.slots.len() as u64, 0, 0, 0, 0];
        for w in self.slots.values() {
            if w.mapped > 0 {
                c[3] += 1;
            } else if w.looks > 0 {
                c[2] += 1;
            }
            if w.due <= frame {
                c[1] += 1;
                c[4] = c[4].max(frame.saturating_sub(w.written));
            }
        }
        c
    }
}

/// Whether `view` is of a texture the title may fill itself: a texture it
/// placed in memory, or a texture object.
pub(crate) fn is_texture_view(view: &View) -> bool {
    match view {
        v if native_tlas(v).is_some() => false,
        View::Placed(false, resource, _) => resource[0] != 1,
        View::Shader(r, _) => crate::resource::buffer_bytes(*r as crate::object::This).is_none(),
        _ => false,
    }
}

/// Buffer views left null because the host would refuse them
/// ([`buffer_view_ok`]), since the start.
pub(crate) static REFUSED_VIEWS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Whether the host device is gone, and why (the first time it is asked
/// after that): for passes that make many views, to name the one that did
/// it.
pub(crate) fn device_removed() -> Option<String> {
    let host = host()?;
    // SAFETY: a plain query.
    let reason = unsafe { host.device.GetDeviceRemovedReason() };
    reason.err().map(|e| e.to_string())
}

/// Writes of title view-heap slots with a placed texture view and with a
/// view of a texture object, since the start (diagnostic counts).
pub(crate) static PLACED_TEXTURE_SLOTS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub(crate) static OBJECT_TEXTURE_SLOTS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Bytes per element and typed integer format bits of a buffer view, decoded
/// in a single pass over the view descriptor; `(None, 0)` if not a buffer view.
fn view_stride_and_kind(view: &View) -> (Option<u32>, u32) {
    use victory_shader::emit::{VIEW_INTEGER, VIEW_SIGNED};
    let Some((stride, raw, format)) = view_elements(view) else {
        return (None, 0);
    };
    let bytes = if stride > 0 {
        stride
    } else if raw {
        1
    } else {
        crate::resource::element(format).map_or(0, |(bytes, _)| bytes as u32)
    };
    let kind = if stride == 0 && !raw {
        match format {
            3 | 7 | 12 | 17 | 25 | 30 | 36 | 42 | 50 | 57 | 62 => VIEW_INTEGER,
            4 | 8 | 14 | 18 | 32 | 38 | 43 | 52 | 59 | 64 => VIEW_INTEGER | VIEW_SIGNED,
            _ => 0,
        }
    } else {
        0
    };
    (Some(bytes), kind)
}

/// Packed [`victory_shader::emit::SLOT_STRIDES`] word for `view`.
fn view_stride_word(view: &View) -> u32 {
    let (stride, kind) = view_stride_and_kind(view);
    stride.unwrap_or(0) | kind
}

/// Bytes per element of a buffer view, as the console's descriptor holds
/// them for indexed reads (structured: the stride; raw: 1, the index is the
/// byte; typed: the format's); `None` if `view` is not a buffer view.
#[cfg(test)]
fn view_stride(view: &View) -> Option<u32> {
    view_stride_and_kind(view).0
}

/// What else the table of strides says of a typed buffer view of an integer
/// format (`victory_shader::emit::VIEW_INTEGER`); 0 for any other view.
#[cfg(test)]
fn view_kind(view: &View) -> u32 {
    view_stride_and_kind(view).1
}

/// A buffer view's structure stride, whether it is raw, and its format;
/// `None` if `view` is not a buffer view.
fn view_elements(view: &View) -> Option<(u32, bool, u32)> {
    const BUFFER: u32 = 1;
    let (words, unordered): (&[u32; 10], bool) = match view {
        View::Shader(_, Some(d)) if d[1] == BUFFER => (d, false),
        View::Unordered(_, _, Some(d)) if d[1] == BUFFER => (d, true),
        View::Placed(unordered, resource, d) if resource[0] == BUFFER => {
            (d[..10].try_into().ok()?, *unordered)
        }
        _ => return None,
    };
    // SAFETY: buffer views' union members.
    Some(unsafe {
        if unordered {
            let d = uav_desc(words);
            let b = d.Anonymous.Buffer;
            (b.StructureByteStride, b.Flags.0 & D3D12_BUFFER_UAV_FLAG_RAW.0 != 0, d.Format.0 as u32)
        } else {
            let d = srv_desc(words);
            let b = d.Anonymous.Buffer;
            (b.StructureByteStride, b.Flags.0 & D3D12_BUFFER_SRV_FLAG_RAW.0 != 0, d.Format.0 as u32)
        }
    })
}

/// Whether a buffer view that begins `offset` bytes into its host copy begins
/// where no host view can: raw views begin on 16 bytes, others on a multiple
/// of their elements (`stride` of a structured one, which is made raw where it
/// can be, `element` of a typed one). The console's begin on any dword, and
/// the title puts parameter blocks at ring positions: a raw view of 128 bytes
/// at 8 (the colour table bakes'), structured ones of elements that are no
/// divisor of their place (the culling's 456-byte constants). Such a view gets
/// a copy of its own bytes ([`crate::shadow::exact`]). With no raw twin,
/// an untyped structured view also needs that copy to be 16-byte aligned.
/// A typed view of 12-byte elements also needs its own copy when its start
/// is a multiple of 16 but not a multiple of 12.
fn off_elements(offset: u64, stride: u32, raw: bool, element: u64, has_raw_twins: bool) -> bool {
    let needs_raw = stride > 0 && stride.is_multiple_of(4) && !has_raw_twins;
    if raw || needs_raw {
        !offset.is_multiple_of(16)
    } else if stride > 0 {
        !offset.is_multiple_of(16) && !offset.is_multiple_of(u64::from(stride))
    } else {
        element > 0 && !offset.is_multiple_of(element)
    }
}

/// Whether `view` has a raw twin when its host slot is made ([`raw_twins`]):
/// a buffer view or a constant buffer's.
fn twinned(view: &View) -> bool {
    matches!(view, View::Constant(..)) || view_elements(view).is_some()
}

/// Whether views off their copy's elements get a copy of their own bytes. On by
/// default; `VICTORY_VIEW_BYTES=0` leaves them null, as before.
fn view_bytes() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_VIEW_BYTES").is_none_or(|v| v != "0"))
}

/// The copy the buffer view of `[start, start + len)` is made on: `shadow`, the one
/// its buffer has, or one of the view's own bytes ([`off_elements`]).
fn view_home(
    shadow: std::sync::Arc<crate::shadow::Shadow>,
    start: u64,
    len: u64,
    (stride, raw, format): (u32, bool, u32),
) -> std::sync::Arc<crate::shadow::Shadow> {
    let Some(offset) = shadow.offset_in(start).map(|_| shadow.offset(start)) else {
        return shadow;
    };
    if !view_bytes() {
        return shadow;
    }
    let element = crate::resource::element(format).map_or(0, |(bytes, _)| bytes);
    // Pick the same copy when creating the descriptor and when refreshing it,
    // even if the latter lookup comes before the title creates its view heap.
    if stride > 0 {
        probe_view_limit();
    }
    let has_raw_twins = victory_shader::emit::raw_views() != 0;
    if !off_elements(offset, stride, raw, element, has_raw_twins) {
        return shadow;
    }
    crate::shadow::exact(start, len).unwrap_or(shadow)
}

/// For device-removed reports: what the slots of the title's view heap that
/// the words of user data may name hold now ([`crate::host::indirects`]).
#[cfg(feature = "diagnostics")]
pub(crate) fn describe_slots(words: &[u32]) -> Vec<(usize, u32, String)> {
    super::with_view_heap(|host| {
        crate::host::indirects::slots(words, host.count, 6)
            .into_iter()
            .take(24)
            .map(|(word, slot)| (word, slot, host.describe_slot(slot).unwrap_or_default()))
            .collect()
    })
    .unwrap_or_default()
}

/// For `crate::book`: the bytes of up to `n` descriptors of the view heap from
/// slot `at` on (from the first twin on, where `at` lies before it and past
/// the others), as the card's heap has them: the first slot, the bytes, and
/// the slot behind the last one worth reading.
#[cfg(feature = "diagnostics")]
pub(crate) fn shown_bytes(at: u32, n: u32) -> Option<(u32, Vec<u8>, u32)> {
    super::with_view_heap(|host| {
        let own = host.count + host.dynamic;
        let end = if host.raw == 0 { own } else { host.raw + own };
        let at = if at >= own && at < host.raw { host.raw } else { at };
        let stop = if at < own { own } else { end };
        let n = n.min(stop.saturating_sub(at));
        if n == 0 || host.step != 32 {
            return None;
        }
        let mut bytes = vec![0u8; n as usize * 32];
        crate::host::read_into(host.cpu(at).ptr as *const u8, &mut bytes).then_some((at, bytes, end))
    })?
}

/// For `crate::book`: what our books say of a slot of the view heap.
#[cfg(feature = "diagnostics")]
pub(crate) fn slot_story(slot: u32) -> String {
    use std::sync::atomic::Ordering::Relaxed;
    use windows::core::Interface;
    super::with_view_heap(|host| {
        let (index, twin) = if host.raw != 0 && slot >= host.raw { (slot - host.raw, true) } else { (slot, false) };
        let mut text = if index < host.count {
            format!("title slot {index}, made from {}", host.describe_slot(index).unwrap_or_default())
        } else {
            let taken = host.next.lock().unwrap_or_else(|e| e.into_inner()).1.get((index - host.count) as usize);
            format!("dynamic slot {} (taken {taken:?})", index - host.count)
        };
        if twin {
            let made = host.twins.get(index as usize).map(|t| t.load(Relaxed));
            text.push_str(&format!(", its raw twin (made {made:?})"));
        }
        if let Some(at) = host.written.get(slot as usize) {
            text.push_str(&format!(", written last at {:.1} s", f64::from(at.load(Relaxed)) / 1e3));
        }
        let before = BEFORE.lock().unwrap_or_else(|e| e.into_inner()).get(&index).copied();
        if let Some((was, at, (started, finished), by)) = before {
            let what = match view_at(was as u32, (was >> 32) as u32) {
                _ if was == 0 => "nothing (or the neutral texture)".into(),
                None => "a view that is gone".into(),
                Some(view) => format!("{view:?}").chars().take(160).collect::<String>(),
            };
            text.push_str(&format!(
                ", changed ({by}) at {:.3} s with submissions {started} begun and {finished} finished, showed before: {what}",
                at as f64 / 1e3
            ));
        }
        let kept = host.kept_in(index);
        match kept {
            None => text.push_str(", holds nothing"),
            Some(resource) => {
                // SAFETY: a resource the slot holds.
                let d = unsafe { resource.GetDesc() };
                let gpu = if d.Dimension == D3D12_RESOURCE_DIMENSION_BUFFER {
                    // SAFETY: as above.
                    unsafe { resource.GetGPUVirtualAddress() }
                } else {
                    0
                };
                text.push_str(&format!(
                    ", holds {:#x}: dimension {} {}x{}x{} levels {} f{} at {gpu:#x}, {} references",
                    resource.as_raw() as usize,
                    d.Dimension.0,
                    d.Width,
                    d.Height,
                    d.DepthOrArraySize,
                    d.MipLevels,
                    d.Format.0,
                    crate::host::references(&resource)
                ));
            }
        }
        text
    })
    .unwrap_or_default()
}

/// Host heaps' GPU ranges (start, bytes, bytes per slot, sampler), for
/// device-removed reports: a shader indexing a heap past its end faults.
static GPU_RANGES: Mutex<Vec<(u64, u64, u64, bool)>> = Mutex::new(Vec::new());

/// For device-removed reports: where host GPU address `va` lies against
/// the host descriptor heaps.
pub(crate) fn near_heaps(va: u64) -> String {
    let ranges = GPU_RANGES.lock().unwrap_or_else(|e| e.into_inner());
    let mut text = String::new();
    for &(start, bytes, step, sampler) in ranges.iter() {
        let kind = if sampler { "sampler" } else { "view" };
        let place = if va < start {
            format!("{:#x} below", start - va)
        } else if va < start + bytes {
            format!("inside, slot {}", (va - start) / step)
        } else {
            format!("{:#x} past its end (slot {})", va - start - bytes, (va - start) / step)
        };
        text.push_str(&format!("; {kind} heap {start:#x}+{bytes:#x}: {place}"));
    }
    text
}

/// Dynamic slots given out and given back since the start, for the log of
/// a failed search (their rate over time is what a frame takes).
static GIVEN: [std::sync::atomic::AtomicU64; 2] =
    [std::sync::atomic::AtomicU64::new(0), std::sync::atomic::AtomicU64::new(0)];

/// Which dynamic slots are taken, a bit each, searched a word at a time.
///
/// Why: a slot at a time, a table that found no run walked the whole ring
/// under the lock every recording thread needs: on a host whose view heaps
/// stop at [`PORTABLE_VIEWS`] (NVIDIA), searching a word at a time avoids
/// linear scan lock contention.
pub(crate) struct Taken {
    words: Vec<u64>,
    len: usize,
}

impl Taken {
    fn new(len: usize) -> Self {
        Self { words: vec![0; len.div_ceil(64)], len }
    }

    /// Whether slot `at` is taken; `None` past the end.
    #[cfg_attr(not(feature = "diagnostics"), allow(dead_code))]
    fn get(&self, at: usize) -> Option<bool> {
        (at < self.len).then(|| self.words[at / 64] >> (at % 64) & 1 != 0)
    }

    /// The first of `n` free slots in a row within `[from, to)`.
    fn find(&self, from: usize, to: usize, n: usize) -> Option<usize> {
        let to = to.min(self.len);
        let mut run = 0usize;
        let mut at = from;
        while at < to {
            let bit = at % 64;
            let span = (64 - bit).min(to - at);
            let mask = if span == 64 { u64::MAX } else { (1u64 << span) - 1 };
            let word = (self.words[at / 64] >> bit) & mask;
            if word == 0 {
                run += span;
                if run >= n {
                    return Some(at + span - run);
                }
                at += span;
            } else if word == mask {
                run = 0;
                at += span;
            } else {
                let free = word.trailing_zeros() as usize;
                run += free;
                if run >= n {
                    return Some(at + free - run);
                }
                let held = (!(word >> free)).trailing_zeros() as usize;
                run = 0;
                at += free + held;
            }
        }
        None
    }

    /// Marks `[from, to)` taken or free; how many changed.
    fn set(&mut self, from: usize, to: usize, taken: bool) -> usize {
        let to = to.min(self.len);
        let mut changed = 0;
        let mut at = from;
        while at < to {
            let bit = at % 64;
            let span = (64 - bit).min(to - at);
            let mask = (if span == 64 { u64::MAX } else { (1u64 << span) - 1 }) << bit;
            let word = &mut self.words[at / 64];
            if taken {
                changed += (!*word & mask).count_ones() as usize;
                *word |= mask;
            } else {
                changed += (*word & mask).count_ones() as usize;
                *word &= !mask;
            }
            at += span;
        }
        changed
    }

    /// The longest run of free slots (for the log of a failed search).
    fn longest_free(&self) -> usize {
        let (mut best, mut run) = (0, 0);
        for at in 0..self.len {
            if self.words[at / 64] >> (at % 64) & 1 != 0 {
                run = 0;
            } else {
                run += 1;
                best = best.max(run);
            }
        }
        best
    }
}

/// The host's limit for shader-visible view heaps on resource binding
/// tiers 1 and 2; tier 3 hardware may take more.
const PORTABLE_VIEWS: u32 = 1_000_000;

/// The size of every host view heap on a host that takes more than
/// [`PORTABLE_VIEWS`] (AMD): the title's heap (a little over 1,000,000
/// views in the retail game) plus [`DYNAMIC`] fits with room to spare, and
/// the last slot is the null slot every view index at or past it reads.
const UNCAPPED_VIEWS: u32 = 1 << 21;

/// Where the raw twins of the title's slots start in a heap that has them
/// ([`raw_twins`]): past every slot a title's heap and the dynamic ones take.
const RAW_TWINS: u32 = 1 << 21;

/// Whether view heaps keep a raw twin of every buffer view of the title's
/// slots, which shaders read that pick buffers by an index they compute
/// (`victory_shader::emit::raw_views` says why; `VICTORY_RAW_VIEWS=0`: they
/// read the title's slots themselves, as before). Needs a host that takes
/// heaps of 2^22 views.
fn raw_twins() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_RAW_VIEWS").is_none_or(|v| v != "0"))
}

/// Picks, once, the null slot at the end of every host view heap and tells
/// the translator now (`set_view_heap_limit`) instead of when the title's
/// heap is made: a shader keeps the limit it was translated with, and the
/// title makes 25 pipelines (their shaders translated at once) before it
/// makes its view heap. Called before a shader is translated and before a
/// view heap is made.
///
/// Every host clamps. A host that caps view heaps at [`PORTABLE_VIEWS`]
/// (NVIDIA) hung on a view index past the heap; one that does not (AMD
/// RDNA3, RX 7900 XT) took a page fault and removed the device on entering
/// the world: a compute program read view slot 16,777,088 (0xffff80) of a
/// heap of 1,507,328, past its end. Unclamped, that index is undefined on
/// any host.
pub(crate) fn probe_view_limit() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        let Some(host) = host() else { return };
        // Query D3D12_OPTIONS19 unconditionally instead of probing oversized
        // 4M/2M heaps, keeping D3D12 Debug Layer clean on NVIDIA 1M heaps in
        // both RT-off and native DXR modes.
        let max_views = {
            let mut limits = D3D12_FEATURE_DATA_D3D12_OPTIONS19::default();
            let supported = unsafe {
                host.device.CheckFeatureSupport(
                    D3D12_FEATURE_D3D12_OPTIONS19, (&raw mut limits).cast(), size_of_val(&limits) as u32,
                )
            };
            (supported.is_ok() && limits.MaxViewDescriptorHeapSize != 0)
                .then_some(limits.MaxViewDescriptorHeapSize)
        };
        let fits = |count: u32| {
            if let Some(max) = max_views { return count <= max; }
            // SAFETY: plain heap creation; the heap is released at once.
            let made: windows::core::Result<ID3D12DescriptorHeap> = unsafe {
                host.device
                    .CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                        Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                        NumDescriptors: count,
                        Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
                        NodeMask: 0,
                    })
            };
            made.is_ok()
        };
        let twins = raw_twins() && fits(2 * RAW_TWINS);
        if twins {
            victory_shader::emit::set_raw_views(RAW_TWINS);
        }
        // (The twins lie past the null slot.)
        let views = if twins {
            RAW_TWINS
        } else if fits(UNCAPPED_VIEWS) {
            UNCAPPED_VIEWS
        } else {
            PORTABLE_VIEWS
        };
        tracing::info!(
            views,
            raw_twins = twins,
            "host view heaps hold {views} views; view indices are clamped to the null slot at the end from the first shader on"
        );
        victory_shader::emit::set_view_heap_limit(views - 1);
    });
}

/// Slots past the title's in a view heap, for tables a draw needs made
/// afresh (constant buffers copied per draw). A list holds the slots it
/// took until it is reset (its submissions have run) or released.
/// 2^16 ran out in the world ("no free dynamic descriptor slots", run 11;
/// upstream PR #336 takes 2^18). On a host heap capped at 1,000,000 views
/// (NVIDIA) these come out of the title's slots, whose highest seen is
/// about 720,000: 2^17 left 868,000 of them, and ran out in the opening
/// cutscene (a frame asks for up to 16,000 tables while the frame before
/// still holds its own; a draw refused its table read slots never made
/// for it, and the card stopped within frames). 2^18 leaves 737,000; a
/// table past them is made afresh in these slots like any other.
pub(crate) const DYNAMIC: u32 = 1 << 18;

/// [`DYNAMIC`] for a heap of `count` title slots whose null slot is `null`
/// (0: none): up to four times as many where the host heap has the room (a
/// host that takes more than [`PORTABLE_VIEWS`]; `VICTORY_DYNAMIC_SLOTS`
/// sets the most). 2^18 ran out in the title's bigger places (session
/// play-1125-0127: 27,000 to 35,000 tables a frame, up to 3,285 of them
/// refused, with 2,639 indirect draws and dispatches).
fn dynamic_slots(count: u32, null: u32) -> u32 {
    static MOST: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
    let most = *MOST.get_or_init(|| {
        std::env::var("VICTORY_DYNAMIC_SLOTS")
            .ok()
            .and_then(|v| v.trim().parse().ok())
            .unwrap_or(DYNAMIC << 2)
    });
    if null < PORTABLE_VIEWS {
        // NVIDIA caps the shader-visible view heap at 1,000,000 views; allow
        // `VICTORY_DYNAMIC_SLOTS` to borrow up to half the heap from the
        // title's kept-in-place slots (trustno1-coder/VIctory-NVIDIA a9632bc).
        static CAPPED: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
        return *CAPPED.get_or_init(|| {
            let asked = std::env::var("VICTORY_DYNAMIC_SLOTS")
                .ok()
                .and_then(|v| v.trim().parse::<u32>().ok());
            let slots = asked.unwrap_or(DYNAMIC).clamp(DYNAMIC, null / 2);
            if slots != DYNAMIC {
                tracing::info!(
                    slots,
                    title_in_place = null - slots,
                    "dynamic descriptor slots on a capped view heap"
                );
            }
            slots
        });
    }
    null.saturating_sub(count).min(most).max(DYNAMIC)
}

impl HostHeap {
    pub(crate) fn new(sampler: bool, count: u32) -> Option<Self> {
        let host = host()?;
        let kind = if sampler {
            D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER
        } else {
            D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV
        };
        let make = |count: u32| -> windows::core::Result<ID3D12DescriptorHeap> {
            // SAFETY: plain heap creation.
            unsafe {
                host.device
                    .CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                        Type: kind,
                        NumDescriptors: count,
                        Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
                        NodeMask: 0,
                    })
            }
        };
        // A view heap ends at the null slot shaders clamp view indices to
        // (`probe_view_limit`), whatever the title's own count: the title's
        // slots, then the dynamic ones, then null up to and with that slot.
        // Sampler heaps always back the full 2,048 hardware limit and reserve
        // slot 2,047 as a permanent null point sampler so `umin(index, 2047)`
        // in `victory-shader` never reads past smaller title sampler heaps.
        const SAMPLERS: u32 = 2048;
        if !sampler {
            probe_view_limit();
        }
        let null = if sampler {
            SAMPLERS - 1
        } else {
            victory_shader::emit::view_heap_limit()
        };
        let raw = if sampler { 0 } else { victory_shader::emit::raw_views() };
        let extra = if sampler { 0 } else { dynamic_slots(count, null) };
        let (heap, count, size) = if sampler {
            let fitted = count.min(SAMPLERS - 1);
            match make(SAMPLERS) {
                Ok(heap) => (heap, fitted, SAMPLERS),
                Err(e) => {
                    tracing::warn!(error = %e, count, sampler, "no host descriptor heap");
                    return None;
                }
            }
        } else if null == 0 {
            match make(count + extra) {
                Ok(heap) => (heap, count, count + extra),
                Err(e) => {
                    tracing::warn!(error = %e, count, sampler, "no host descriptor heap");
                    return None;
                }
            }
        } else {
            let size = if raw == 0 { null + 1 } else { 2 * raw };
            let fitted = if count + extra <= null {
                count
            } else {
                // shortcut: descriptors past the host's limit are left out; a title that reaches
                // them needs remapping. They read the null slot instead.
                tracing::warn!(count, "host view heap capped at {size}; views past it read a null slot");
                null - extra
            };
            match make(size) {
                Ok(heap) => (heap, fitted, size),
                Err(e) => {
                    tracing::warn!(error = %e, count, size, "no host view heap");
                    return None;
                }
            }
        };
        // SAFETY: a live heap.
        let (cpu, step) = unsafe {
            (
                heap.GetCPUDescriptorHandleForHeapStart().ptr,
                host.device.GetDescriptorHandleIncrementSize(kind),
            )
        };
        tracing::debug!(count, sampler, "host descriptor heap");
        // SAFETY: a live shader-visible heap.
        let gpu = unsafe { heap.GetGPUDescriptorHandleForHeapStart().ptr };
        GPU_RANGES.lock().unwrap_or_else(|e| e.into_inner()).push((
            gpu,
            u64::from(size) * u64::from(step),
            u64::from(step),
            sampler,
        ));
        let strides = (!sampler)
            .then(|| {
                // The strides, then the views' title addresses
                // (`victory_shader::emit::VIEW_ADDRESSES`).
                let bytes = u64::from(victory_shader::emit::SLOT_STRIDES_BYTES);
                let (buffer, data) = host.upload_buffer(bytes)?;
                // SAFETY: the buffer's whole mapping; a live buffer.
                let gpu = unsafe {
                    std::ptr::write_bytes(data, 0, bytes as usize);
                    buffer.GetGPUVirtualAddress()
                };
                crate::host::name(&buffer, || "element strides by view heap slot".to_owned());
                Some((buffer, gpu, data as usize))
            })
            .flatten();
        let stride_shadow: Box<[std::sync::atomic::AtomicU32]> = if strides.is_some() {
            (0..victory_shader::emit::SLOT_STRIDES_COUNT)
                .map(|_| std::sync::atomic::AtomicU32::new(0))
                .collect()
        } else {
            Box::new([])
        };
        let queued = std::sync::Arc::new(Queued::new(if sampler { 0 } else { count }));
        let h = HostHeap {
            id: NEXT_ID.fetch_add(1, Ordering::Relaxed),
            buffer_sweep: Mutex::default(),
            buffer_frame: std::sync::atomic::AtomicU64::new(u64::MAX),
            heap,
            cpu,
            gpu,
            step,
            count,
            dynamic: extra,
            sampler,
            raw,
            twins: (0..if raw == 0 { 0 } else { count + extra }).map(|_| std::sync::atomic::AtomicBool::new(false)).collect(),
            #[cfg(feature = "diagnostics")]
            written: (0..if sampler || !crate::book::on() { 0 } else { size })
                .map(|_| std::sync::atomic::AtomicU32::new(0))
                .collect(),
            made: Made::new(count as usize),
            quick: (0..count).map(|_| std::sync::atomic::AtomicU64::new(0)).collect(),
            copied: Mutex::new(Vec::new()),
            waiting: Mutex::default(),
            away: Mutex::default(),
            native_slots: Default::default(),
            kept: (0..count + extra).map(|_| Mutex::new(None)).collect(),
            kept_some: (0..count + extra)
                .map(|_| std::sync::atomic::AtomicBool::new(false))
                .collect(),
            next: Mutex::new((0, Taken::new(extra as usize))),
            free: std::sync::atomic::AtomicU32::new(extra),
            too_long: std::sync::atomic::AtomicU32::new(u32::MAX),
            strides,
            stride_shadow,
            buffers: Slots::new(count as usize),
            changed: Slots::new(count as usize),
            textures: std::sync::Arc::new(Mutex::new(TextureQueue {
                queued: queued.clone(),
                ..Default::default()
            })),
            queued,
            tabled: (0..if sampler { 0 } else { count })
                .map(|_| std::sync::atomic::AtomicU32::new(0))
                .collect(),
            placed: {
                let spans = std::sync::Arc::<Mutex<Spans>>::default();
                if !sampler {
                    PLACED_SPANS
                        .lock()
                        .unwrap_or_else(|e| e.into_inner())
                        .push(std::sync::Arc::downgrade(&spans));
                }
                spans
            },
            spanned: Slots::new((count + extra) as usize),
            placed_owners: Mutex::default(),
            last_target_gen: std::sync::atomic::AtomicU64::new(crate::resource::all_target_changes()),
            placed_recheck_needed: std::sync::atomic::AtomicBool::new(false),
            uav_drain: Mutex::new(()),
            uav_draining: std::sync::atomic::AtomicUsize::new(0),
            uav_cursor: std::sync::atomic::AtomicU64::new(UAV_SEQ.load(Ordering::Acquire)),
            uav_frame: std::sync::atomic::AtomicU64::new(u64::MAX),
        };
        // A slot never written is undefined: shaders reading one (a slot
        // the title wrote in a way we do not know, or a table entry no
        // used-resource entry names) read a wild address. Every slot starts
        // null instead.
        let started = std::time::Instant::now();
        if let Err(e) = h.fill_null(kind, size) {
            tracing::warn!(error = %e, "host descriptor heap not made null");
        }
        tracing::debug!(ms = started.elapsed().as_millis(), "host descriptor heap made null");
        Some(h)
    }

    /// Makes the first `total` slots null: one null slot in a CPU-side heap,
    /// doubled by copies to a block, the block copied over the heap.
    fn fill_null(&self, kind: D3D12_DESCRIPTOR_HEAP_TYPE, total: u32) -> windows::core::Result<()> {
        const BLOCK: u32 = 1 << 16;
        let Some(host) = host() else { return Ok(()) };
        // SAFETY: plain heap creation and copies inside live heaps.
        unsafe {
            let block: ID3D12DescriptorHeap =
                host.device.CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                    Type: kind,
                    NumDescriptors: BLOCK,
                    ..Default::default()
                })?;
            let start = block.GetCPUDescriptorHandleForHeapStart();
            let at = |i: u32| D3D12_CPU_DESCRIPTOR_HANDLE {
                ptr: start.ptr + (i * self.step) as usize,
            };
            // (The twins are null buffer views.)
            let ends = if self.raw == 0 { [total, total] } else { [self.raw.min(total), total] };
            let mut done = 0;
            for (twin, end) in ends.into_iter().enumerate() {
                if done == end {
                    continue;
                }
                if self.sampler {
                    null_sampler(start);
                } else if twin == 1 {
                    null_buffer_view(start);
                } else {
                    null_view(start);
                }
                let mut filled = 1;
                while filled < BLOCK {
                    let n = filled.min(BLOCK - filled);
                    host.device.CopyDescriptorsSimple(n, at(filled), start, kind);
                    filled += n;
                }
                while done < end {
                    let n = BLOCK.min(end - done);
                    host.device.CopyDescriptorsSimple(n, self.cpu(done), start, kind);
                    done += n;
                }
            }
        }
        Ok(())
    }

    /// A plain point sampler at slot `index`, for sampler slots without a
    /// view.
    fn null_sampler(&self, index: u32) {
        null_sampler(self.at(index));
        self.put(index);
    }
}

/// A plain point sampler at `at`.
fn null_sampler(at: D3D12_CPU_DESCRIPTOR_HANDLE) {
    let Some(host) = host() else { return };
    let desc = D3D12_SAMPLER_DESC {
        Filter: D3D12_FILTER_MIN_MAG_MIP_POINT,
        AddressU: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
        AddressV: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
        AddressW: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
        MaxAnisotropy: 1,
        ComparisonFunc: D3D12_COMPARISON_FUNC_NEVER,
        ..Default::default()
    };
    // SAFETY: a slot of a sampler heap of ours.
    unsafe { host.device.CreateSampler(&desc, at) };
}

impl HostHeap {
    /// `n` consecutive free dynamic slots (view heaps only), taken until
    /// [`HostHeap::release`]: the first's index.
    pub(crate) fn dynamic(&self, n: u32) -> Option<u32> {
        if self.sampler || n == 0 || n > self.dynamic {
            return None;
        }
        #[cfg(feature = "diagnostics")]
        let started = std::time::Instant::now();
        #[cfg(feature = "diagnostics")]
        crate::copystat::add(crate::copystat::C::DynamicAsked, 1);
        let mut guard = self.next.lock().unwrap_or_else(|e| e.into_inner());
        let (next, taken) = &mut *guard;
        let enough = self.free.load(std::sync::atomic::Ordering::Relaxed) >= n
            && n < self.too_long.load(std::sync::atomic::Ordering::Relaxed);
        let n = n as usize;
        // Search from `next` on, then from the start (extending past cursor by
        // n - 1 so contiguous runs crossing the cursor are not missed).
        let cursor = *next as usize;
        let end = self.dynamic as usize;
        let second_end = cursor.saturating_add(n.saturating_sub(1)).min(end);
        // (A run cannot wrap past the end: the two passes are searched apart.)
        let found = if enough { taken.find(cursor, end, n).or_else(|| taken.find(0, second_end, n)) } else { None };
        let Some(first) = found else {
            if enough {
                self.too_long.fetch_min(n as u32, std::sync::atomic::Ordering::Relaxed);
                static FAILED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
                let failed = FAILED.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
                if failed.is_power_of_two() {
                    tracing::info!(
                        failed,
                        n,
                        free = self.free.load(std::sync::atomic::Ordering::Relaxed),
                        longest = taken.longest_free(),
                        slots = end,
                        given_out = GIVEN[0].load(std::sync::atomic::Ordering::Relaxed),
                        given_back = GIVEN[1].load(std::sync::atomic::Ordering::Relaxed),
                        "dynamic descriptor slots: a table found no run of free slots"
                    );
                }
            }
            static ONCE: std::sync::Once = std::sync::Once::new();
            ONCE.call_once(|| tracing::warn!(n, "no free dynamic descriptor slots: a table is not made"));
            #[cfg(feature = "diagnostics")]
            {
                use crate::copystat::{C, add};
                add(C::DynamicRefused, 1);
                add(C::DynamicNs, started.elapsed().as_nanos() as u64);
            }
            return None;
        };
        #[cfg(feature = "diagnostics")]
        crate::copystat::add(crate::copystat::C::DynamicNs, started.elapsed().as_nanos() as u64);
        taken.set(first, first + n, true);
        self.free.fetch_sub(n as u32, std::sync::atomic::Ordering::Relaxed);
        GIVEN[0].fetch_add(n as u64, std::sync::atomic::Ordering::Relaxed);
        *next = ((first + n) % end) as u32;
        Some(self.count + first as u32)
    }

    /// The title slots this heap has host slots for; fewer than the title's
    /// when the host caps its size.
    pub(crate) fn count(&self) -> u32 {
        self.count
    }

    /// This heap's identity ([`NEXT_ID`]): dynamic slots are given back only
    /// to it, never to a heap made later at the same address.
    pub(crate) fn id(&self) -> u64 {
        self.id
    }

    /// The GPU address of the table of element strides by title slot (view
    /// heaps).
    pub(crate) fn strides_gpu(&self) -> Option<u64> {
        self.strides.as_ref().map(|s| s.1)
    }

    /// The table's buffer, for a list to hold.
    pub(crate) fn strides_buffer(&self) -> Option<&ID3D12Resource> {
        self.strides.as_ref().map(|s| &s.0)
    }

    /// Up to `max` title slots whose texture view is due at `frame`, each
    /// with its write number, in the order to serve them
    /// ([`TextureQueue::due`]), and the queue's counts
    /// ([`TextureQueue::census`]).
    pub(crate) fn texture_slots_due(&self, frame: u64, max: usize) -> (Vec<(u32, u64)>, [u64; 5]) {
        let (due, census) = {
            let textures = self.textures.lock().unwrap_or_else(|e| e.into_inner());
            (if max == 0 { Due::default() } else { textures.all_due(frame) }, textures.census(frame))
        };
        (TextureQueue::in_order(due, max), census)
    }

    /// A new pass invalidates earlier coverage before gathering the due slots.
    pub(crate) fn begin_texture_pass(&self) -> TexturePass<'_> {
        TexturePass::begin(&self.textures)
    }

    pub(crate) fn finish_bake(&self, revision: u64, frame: u64) {
        self.textures
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .finish_bake(revision, frame);
    }

    fn has_unread_uav_written(&self) -> bool {
        self.uav_draining.load(Ordering::Acquire) != 0
            || UAV_SEQ.load(Ordering::Acquire) > self.uav_cursor.load(Ordering::Acquire)
    }

    fn has_stale_dynamic_placed(&self) -> bool {
        if self.placed_recheck_needed.load(Ordering::SeqCst) {
            return true;
        }
        let generation = crate::resource::all_target_changes();
        if self.last_target_gen.load(Ordering::Acquire) == generation {
            return false;
        }
        !self.placed_owners.lock().unwrap_or_else(|e| e.into_inner()).is_empty()
    }

    /// Re-checks tracked bindless dynamic writable placed-texture slots when
    /// `resource::all_target_changes()` has advanced since this heap last
    /// checked (or a concurrent publisher armed `placed_recheck_needed`): any
    /// slot whose backing target owner or host resource changed is immediately
    /// requeued in `TextureQueue` (bumping revision) so bindless shaders never
    /// keep reading a stale resource A after B replaces it.
    pub(crate) fn texture_slots_target_owners(&self, frame: u64) -> usize {
        if self.sampler {
            return 0;
        }
        let recheck = self.placed_recheck_needed.load(Ordering::SeqCst);
        let generation = crate::resource::all_target_changes();
        let prev_gen = self.last_target_gen.load(Ordering::SeqCst);
        if !recheck && prev_gen == generation {
            return 0;
        }
        // Clear `placed_recheck_needed` BEFORE snapshotting `placed_owners`
        // (paired with the `SeqCst` fence in `set_placed_owner` AFTER inserting
        // into `placed_owners`), so a concurrent mirror either becomes visible
        // here or observes the newer generation/seen_gen and re-arms recheck.
        self.placed_recheck_needed.store(false, Ordering::SeqCst);
        let tracked: Vec<(u32, u64, u64, usize)> = {
            let owners = self.placed_owners.lock().unwrap_or_else(|e| e.into_inner());
            if owners.is_empty() {
                let _ = self.last_target_gen.compare_exchange(
                    prev_gen,
                    generation,
                    Ordering::SeqCst,
                    Ordering::Acquire,
                );
                return 0;
            }
            owners
                .iter()
                .map(|(&slot, &(key, owner_id, host_raw))| (slot, key, owner_id, host_raw))
                .collect()
        };
        let mut stale = Vec::new();
        for (slot, key, recorded_id, recorded_host) in tracked {
            let Some(view) = view_at(key as u32, (key >> 32) as u32) else {
                continue;
            };
            if !is_dynamic_placed_texture(&view) {
                continue;
            }
            let current = placed_texture(&view);
            let current_id = current.as_ref().map_or(0, PlacedTexture::owner_id);
            let current_host = current.as_ref().map_or(0, PlacedTexture::host_raw_if_made);
            let changed = recorded_id != current_id
                || (current_id != 0 && current_host != 0 && current_host != recorded_host)
                || (current_id == 0 && recorded_host != 0);
            if changed {
                stale.push(slot);
            }
        }
        if !stale.is_empty() {
            let mut textures = self.textures.lock().unwrap_or_else(|e| e.into_inner());
            for &slot in &stale {
                textures.changed(slot, frame, frame);
            }
        }
        let _ = self.last_target_gen.compare_exchange(
            prev_gen,
            generation,
            Ordering::SeqCst,
            Ordering::Acquire,
        );
        stale.len()
    }

    /// Whether a completed bake covered this heap for `frame` and there are no
    /// pending memory notes, target-owner changes, or unread/draining GPU UAV
    /// writes (`TaynanGT #85`).
    pub(crate) fn bake_covered(&self, frame: u64) -> bool {
        let uav_seq0 = UAV_SEQ.load(Ordering::Acquire);
        let target_gen0 = crate::resource::all_target_changes();
        if self.has_unread_uav_written() || self.has_stale_dynamic_placed() {
            return false;
        }
        let covered = self
            .textures
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .bake_covered(frame);
        let noted = |ranges: &Mutex<Vec<(u64, u64)>>| {
            !ranges.lock().unwrap_or_else(|e| e.into_inner()).is_empty()
        };
        covered
            && !noted(&REMAPPED)
            && !noted(&PAGE_WRITTEN)
            && UAV_SEQ.load(Ordering::Acquire) == uav_seq0
            && crate::resource::all_target_changes() == target_gen0
            && !self.has_unread_uav_written()
            && !self.has_stale_dynamic_placed()
    }

    /// Every title slot of the queue that is due at `frame`, and whether
    /// it was uploaded before (in part, or whole and waiting for another
    /// look): for the card accounting line
    /// (`crate::resource::card_accounting`).
    pub(crate) fn texture_slots_all_due(&self, frame: u64) -> Vec<(u32, bool)> {
        let textures = self.textures.lock().unwrap_or_else(|e| e.into_inner());
        textures
            .slots
            .iter()
            .filter(|(_, w)| w.due <= frame)
            .map(|(&index, w)| (index, w.mapped > 0 || w.checked.2 != 0))
            .collect()
    }

    /// The texture view of title slot `index` (as handed out with `write`)
    /// was brought up to date: due again from frame `again` (uploaded with
    /// that many bytes of its memory there, not all), or not until the
    /// slot is written again.
    pub(crate) fn texture_slot_since(&self, index: u32) -> Option<(u64, bool, u64)> {
        self.textures.lock().unwrap_or_else(|e| e.into_inner()).since(index)
    }

    /// The texture view of title slot `index` (as handed out with `write`)
    /// was brought up to date: see [`HostHeap::texture_slot_done`].
    pub(crate) fn texture_slot_done(&self, index: u32, write: u64, again: Option<(u64, u64)>) {
        let mut textures = self.textures.lock().unwrap_or_else(|e| e.into_inner());
        match again {
            Some((frame, mapped)) => textures.partly(index, write, frame, mapped),
            None => textures.done(index, write),
        }
    }

    /// Title slot `index` was uploaded whole: see [`TextureQueue::recheck`].
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn texture_slot_recheck(
        &self,
        index: u32,
        write: u64,
        due: u64,
        at: (u64, u64),
        print: u64,
        dense_print: u64,
        looks: u8,
        uploaded: u64,
        state: LiveState,
        baseline: LiveBaseline,
    ) {
        let mut textures = self.textures.lock().unwrap_or_else(|e| e.into_inner());
        textures.recheck(index, write, due, at, print, dense_print, looks, uploaded, state, baseline);
    }

    /// See [`TextureQueue::defer_recheck`].
    pub(crate) fn texture_slot_defer_recheck(&self, index: u32, write: u64, due: u64) {
        self.textures
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .defer_recheck(index, write, due);
    }

    /// Shared handle to this heap's [`TextureQueue`] for submit-time upload
    /// baseline commits (`Staging::on_uploaded`).
    pub(crate) fn texture_queue(&self) -> std::sync::Arc<Mutex<TextureQueue>> {
        self.textures.clone()
    }

    /// See [`TextureQueue::checked`].
    pub(crate) fn texture_slot_checked(
        &self,
        index: u32,
    ) -> Option<(u64, u64, u64, u64, u8, u64, LiveState, LiveBaseline)> {
        self.textures.lock().unwrap_or_else(|e| e.into_inner()).checked(index)
    }

    /// Nothing to upload at title slot `index` now (its memory is not
    /// mapped, or no more of it than before): looked at again after 1, 2,
    /// 4, ... frames, at most `most`.
    pub(crate) fn texture_slot_later(&self, index: u32, write: u64, frame: u64, most: u64) {
        let mut textures = self.textures.lock().unwrap_or_else(|e| e.into_inner());
        textures.later(index, write, frame, most);
    }

    /// See [`TextureQueue::hold`].
    pub(crate) fn texture_slot_held(&self, index: u32, write: u64, due: u64) {
        self.textures.lock().unwrap_or_else(|e| e.into_inner()).hold(index, write, due);
    }

    /// See [`TextureQueue::room`].
    pub(crate) fn texture_slots_room(&self, frame: u64) -> usize {
        self.textures.lock().unwrap_or_else(|e| e.into_inner()).room(frame)
    }

    /// Bytes of title slot `index`'s memory that were mapped when it was
    /// last uploaded in part (0: not uploaded, or whole).
    pub(crate) fn texture_slot_mapped(&self, index: u32) -> u64 {
        self.textures.lock().unwrap_or_else(|e| e.into_inner()).mapped(index)
    }

    /// Takes the ranges of title memory the runtime mapped, unmapped or
    /// wrote since the last call ([`remapped`]) and makes the slots whose
    /// placed texture lies there due from the next frame (by when a write
    /// under way has landed): memory that arrived is uploaded, memory that
    /// went makes the slot null ([`HostHeap::texture_slot_gone`]), and a
    /// texture streamed into the place of another of the same description
    /// is uploaded although no descriptor says so. Slots tables read are
    /// left to their draws. Returns how many slots it made due.
    /// shortcut: the ranges are kept in one place for all heaps; the heap
    /// whose pass runs first takes them (a title has one view heap).
    pub(crate) fn texture_slots_remapped(&self, frame: u64) -> usize {
        use std::sync::atomic::Ordering::Relaxed;
        self.texture_slots_page_written(frame);
        self.texture_slots_uav_written(frame);
        self.texture_slots_target_owners(frame);
        let mut ranges = std::mem::take(&mut *REMAPPED.lock().unwrap_or_else(|e| e.into_inner()));
        if ranges.is_empty() {
            return 0;
        }
        join(&mut ranges);
        let mut slots: Vec<u32> = {
            let placed = self.placed.lock().unwrap_or_else(|e| e.into_inner());
            ranges.iter().flat_map(|&(start, end)| placed.slots_in(start, end)).collect()
        };
        slots.sort_unstable();
        slots.dedup();
        slots.retain(|&slot| !self.tabled_lately(slot, frame));
        let mut textures = self.textures.lock().unwrap_or_else(|e| e.into_inner());
        for &slot in &slots {
            textures.changed(slot, frame, frame + 1);
        }
        STALE[3].fetch_add(slots.len() as u64, Relaxed);
        slots.len()
    }

    /// Makes the slots of the textures filled with less than was mapped
    /// ([`read_short`]) due from the next frame on. Returns how many.
    pub(crate) fn texture_slots_short(&self, frame: u64) -> usize {
        if self.sampler {
            return 0;
        }
        let starts = std::mem::take(&mut SHORT.lock().unwrap_or_else(|e| e.into_inner()).0);
        if starts.is_empty() {
            return 0;
        }
        let mut slots: Vec<u32> = {
            let placed = self.placed.lock().unwrap_or_else(|e| e.into_inner());
            starts.iter().flat_map(|&start| placed.slots_at(start)).collect()
        };
        slots.sort_unstable();
        slots.dedup();
        slots.retain(|&slot| !self.tabled_lately(slot, frame));
        let mut textures = self.textures.lock().unwrap_or_else(|e| e.into_inner());
        for &slot in &slots {
            textures.changed(slot, frame, frame + 1);
        }
        slots.len()
    }

    /// Takes the ranges of title memory written behind the guards' back
    /// since the last call ([`page_written`]) and finds the slots whose
    /// placed texture holds bytes of them ([`Spans::slots_holding`]). A
    /// slot whose texture was uploaded before the write shows the bytes
    /// from before: with `VICTORY_PAGE_WRITES` (`crate::watch::page_writes`,
    /// bit 2) it is due again [`PAGE_WRITE_AFTER`] frames on, as memory
    /// mapped under it would make it ([`HostHeap::texture_slots_remapped`]);
    /// its host copy exists, so the upload needs no room on the card. A
    /// slot still waiting for its first upload is left as it is: that
    /// upload takes the bytes that are there by then. Slots tables read
    /// are left to their draws. Counted either way; returns how many slots
    /// it made due.
    fn texture_slots_page_written(&self, frame: u64) -> usize {
        let ranges = std::mem::take(&mut *PAGE_WRITTEN.lock().unwrap_or_else(|e| e.into_inner()));
        if ranges.is_empty() {
            return 0;
        }
        self.slots_written_under(&ranges, frame, crate::watch::page_writes() & 2 != 0)[2]
    }

    /// [`HostHeap::texture_slots_page_written`] for `ranges` (start, end):
    /// the slots that held bytes of them, those of them uploaded before,
    /// and those made due (the latter, if `due`).
    fn slots_written_under(&self, ranges: &[(u64, u64)], frame: u64, due: bool) -> [usize; 3] {
        use std::sync::atomic::Ordering::Relaxed;
        let mut slots: Vec<u32> = {
            let placed = self.placed.lock().unwrap_or_else(|e| e.into_inner());
            ranges.iter().flat_map(|&(start, end)| placed.slots_holding(start, end)).collect()
        };
        slots.sort_unstable();
        slots.dedup();
        slots.retain(|&slot| !self.tabled_lately(slot, frame));
        let held = slots.len();
        let mut textures = self.textures.lock().unwrap_or_else(|e| e.into_inner());
        // Uploaded before: out of the queue, or in it after an upload (in
        // part, or whole and being looked at again).
        slots.retain(|slot| textures.slots.get(slot).is_none_or(|w| w.mapped > 0 || w.checked.2 != 0));
        PAGE_WRITES[2].fetch_add(held as u64, Relaxed);
        PAGE_WRITES[3].fetch_add(slots.len() as u64, Relaxed);
        if !due {
            return [held, slots.len(), 0];
        }
        for &slot in &slots {
            textures.changed(slot, frame, frame + PAGE_WRITE_AFTER);
        }
        [held, slots.len(), slots.len()]
    }

    /// Reads the title textures written on the GPU since this heap's cursor
    /// ([`gpu_written`]) and makes placed-texture slots showing their memory
    /// due, bounded to [`UAV_DUE_A_FRAME`] per frame, without draining other
    /// heaps' unread entries. Advances `uav_cursor` ONLY AFTER `slots_uav_written`
    /// finishes updating `self.textures` while holding `uav_drain` and an
    /// active `uav_draining` guard so concurrent `bake_covered` calls cannot
    /// certify against pre-invalidation queue state.
    fn texture_slots_uav_written(&self, frame: u64) -> usize {
        use std::sync::atomic::Ordering::{AcqRel, Acquire, Release};
        if self.uav_frame.load(Acquire) == frame {
            return 0;
        }
        if UAV_SEQ.load(Acquire) <= self.uav_cursor.load(Acquire) {
            return 0;
        }
        let _drain = self.uav_drain.lock().unwrap_or_else(|e| e.into_inner());
        struct DrainGuard<'a>(&'a std::sync::atomic::AtomicUsize);
        impl Drop for DrainGuard<'_> {
            fn drop(&mut self) {
                self.0.fetch_sub(1, Release);
            }
        }
        self.uav_draining.fetch_add(1, AcqRel);
        let _guard = DrainGuard(&self.uav_draining);
        if self.uav_frame.load(Acquire) == frame {
            return 0;
        }
        let cursor = self.uav_cursor.load(Acquire);
        if UAV_SEQ.load(Acquire) <= cursor {
            return 0;
        }
        let (ranges, overflow, latest) = {
            let log = UAV_WRITTEN.lock().unwrap_or_else(|e| e.into_inner());
            let latest = log.next.saturating_sub(1);
            if latest <= cursor {
                return 0;
            }
            let overflow = log
                .entries
                .front()
                .is_some_and(|&(first, _)| first > cursor.saturating_add(1));
            let ranges = if overflow {
                Vec::new()
            } else {
                log.entries
                    .iter()
                    .filter(|&&(seq, _)| seq > cursor)
                    .map(|&(_, range)| range)
                    .collect()
            };
            (ranges, overflow, latest)
        };
        let queued = if overflow {
            self.slots_uav_written(&[(0, u64::MAX)], frame)
        } else if ranges.is_empty() {
            0
        } else {
            self.slots_uav_written(&ranges, frame)
        };
        self.uav_cursor.store(latest, Release);
        self.uav_frame.store(frame, Release);
        queued
    }

    fn slots_uav_written(&self, ranges: &[(u64, u64)], frame: u64) -> usize {
        use std::sync::atomic::Ordering::Relaxed;
        let mut ranges = ranges.to_vec();
        join(&mut ranges);
        let mut slots: Vec<u32> = {
            let placed = self.placed.lock().unwrap_or_else(|e| e.into_inner());
            ranges
                .iter()
                .flat_map(|&(start, end)| placed.slots_in(start, end))
                .collect()
        };
        slots.sort_unstable();
        slots.dedup();
        slots.retain(|&slot| !self.tabled_lately(slot, frame));
        UAV_RECHECK[2].fetch_add(slots.len() as u64, Relaxed);
        let mut textures = self.textures.lock().unwrap_or_else(|e| e.into_inner());
        slots.retain(|slot| {
            textures
                .slots
                .get(slot)
                .is_none_or(|w| w.mapped > 0 || w.checked.2 != 0)
        });
        let due = slots.len() as u64;
        for (i, &slot) in slots.iter().enumerate() {
            let when = frame + 1 + i as u64 / UAV_DUE_A_FRAME;
            textures.changed(slot, frame, when);
        }
        UAV_RECHECK[3].fetch_add(due, Relaxed);
        UAV_RECHECK[4].fetch_max(due, Relaxed);
        slots.len()
    }

    /// Whether host slot `index` is made from the very record the title
    /// has at `handle` now. Its host texture then exists (a slot keeps what
    /// it views), so bringing that texture up to date takes no memory of
    /// the card.
    pub(crate) fn texture_slot_made_from(&self, index: u32, handle: u64) -> bool {
        if index >= self.count {
            return false;
        }
        // SAFETY: the handle is a descriptor slot in title memory.
        let [tag, i, generation] = unsafe { (handle as *const [u32; 3]).read_unaligned() };
        let key = u64::from(generation) << 32 | u64::from(i);
        if tag != VIEW_TAG {
            return false;
        }
        // And the slot holds a host resource: a slot is also noted as made
        // from a view the host could not make (`make` gave nothing, the
        // slot is null and keeps nothing). The host copy of such a
        // texture is held by no slot, may have been let go, and would be
        // made anew here, with the card full.
        if key == 0 || self.made.get(index) != Some(key) {
            return false;
        }
        let _made = self.made.lock(index);
        self.made.get(index) == Some(key) && self.kept_in(index).is_some()
    }

    /// Levels on demand (`crate::tiles`): makes again the host slots of the
    /// placed textures whose first level with memory changed since the
    /// last call ([`levels_changed`]), each from the view it was made
    /// from, so that it shows the texture from that level ([`make`]). Only
    /// slots that are made; a slot draws read through a table is made by
    /// its draw. Returns how many it made.
    /// shortcut: the starts are kept in one place for all heaps; the heap
    /// whose pass runs first takes them (a title has one view heap).
    /// Levels on demand (`crate::tiles`): the card has room, so the slots
    /// of up to [`Self::WAITING_A_PASS`] placed textures whose upload left
    /// a held level without tiles at least a second ago
    /// ([`levels_waiting`]) are due again from the next frame, as memory
    /// mapped under them would make them
    /// ([`HostHeap::texture_slots_remapped`]): the upload then gives the
    /// level its tiles. To be called only while the card has room: without
    /// it the upload would give nothing again. Slots tables read are left
    /// to their draws (`texture.rs` `upload` asks again for those). Returns
    /// how many slots it made due.
    /// shortcut: kept in one place for all heaps, as the starts below.
    pub(crate) fn texture_slots_waiting(&self, frame: u64) -> usize {
        if self.sampler {
            return 0;
        }
        let starts: Vec<u64> = {
            let mut waiting = WAITING.lock().unwrap_or_else(|e| e.into_inner());
            if waiting.is_empty() {
                return 0;
            }
            let starts: Vec<u64> = waiting
                .iter()
                .filter(|&(_, &since)| crate::host::long_ago(since, 1000, 2))
                .map(|(&start, _)| start)
                .take(Self::WAITING_A_PASS)
                .collect();
            for start in &starts {
                waiting.remove(start);
            }
            starts
        };
        if starts.is_empty() {
            return 0;
        }
        let mut slots: Vec<u32> = {
            let placed = self.placed.lock().unwrap_or_else(|e| e.into_inner());
            starts.iter().flat_map(|&start| placed.slots_at(start)).collect()
        };
        slots.sort_unstable();
        slots.dedup();
        slots.retain(|&slot| !self.tabled_lately(slot, frame));
        let mut textures = self.textures.lock().unwrap_or_else(|e| e.into_inner());
        for &slot in &slots {
            textures.changed(slot, frame, frame + 1);
        }
        crate::tiles::STATS[10].fetch_add(slots.len() as u64, std::sync::atomic::Ordering::Relaxed);
        slots.len()
    }

    /// Textures a pass asks again for their levels
    /// ([`HostHeap::texture_slots_waiting`]): the first ones given tiles
    /// may take the room the others were asked for, and an upload that
    /// gives nothing is a whole texture staged and copied for nothing.
    const WAITING_A_PASS: usize = 128;

    pub(crate) fn texture_views_remade(&self) -> usize {
        if self.sampler {
            return 0;
        }
        {
            let mut after = RESHOWN_AFTER.lock().unwrap_or_else(|e| e.into_inner());
            if !after.is_empty() {
                let finished = crate::host::submissions().1;
                let mut now = RESHOWN.lock().unwrap_or_else(|e| e.into_inner());
                after.retain(|&(at, start)| {
                    if at <= finished {
                        now.push(start);
                    }
                    at > finished
                });
            }
        }
        let starts = std::mem::take(&mut *RESHOWN.lock().unwrap_or_else(|e| e.into_inner()));
        if starts.is_empty() {
            return 0;
        }
        let slots: Vec<u32> = {
            let placed = self.placed.lock().unwrap_or_else(|e| e.into_inner());
            starts.iter().flat_map(|&start| placed.slots_at(start)).collect()
        };
        let mut remade = 0;
        for slot in slots {
            let _made = self.made.lock(slot);
            let Some(key) = self.made.get(slot).filter(|&key| key != 0) else {
                continue;
            };
            let Some(view) = view_at(key as u32, (key >> 32) as u32) else {
                continue;
            };
            if let Some(kept) = make(&view, self.at(slot)) {
                self.put(slot);
                self.keep(slot, kept);
                remade += 1;
            }
        }
        remade
    }

    /// The memory of the texture title slot `index` views is not mapped
    /// (any more): a host slot made from it becomes null again, so that
    /// the host copy can be let go (`crate::resource::let_go_idle`; a slot
    /// keeps what it views). True if the slot was made.
    ///
    /// Not at the first look: a title that moves a texture in its streaming
    /// memory unmaps it before it maps it again, and what was shown went
    /// black for the pictures in between (session play-1125-0127: faces,
    /// books, posters and sleeves for 0.3 to 2 s). Only when it is still
    /// away `AWAY_MS` after it was first found away
    /// (`VICTORY_TEXTURE_AWAY_MS`; 0: at once, as before).
    pub(crate) fn texture_slot_gone(&self, index: u32) -> bool {
        const AWAY_MS: u64 = 3000;
        static WAIT: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
        let wait = *WAIT.get_or_init(|| {
            std::env::var("VICTORY_TEXTURE_AWAY_MS").ok().and_then(|v| v.trim().parse().ok()).unwrap_or(AWAY_MS)
        });
        let _made = self.made.lock(index);
        let Some(key) = self.made.get(index).filter(|&key| key != 0) else {
            return false;
        };
        let now = crate::host::now_ms();
        {
            let mut away = self.away.lock().unwrap_or_else(|e| e.into_inner());
            match away.get(&index) {
                Some(&(was, since)) if was == key && now.saturating_sub(since) < wait => return false,
                Some(&(was, _)) if was == key => {
                    away.remove(&index);
                }
                _ if wait == 0 => {}
                _ => {
                    away.insert(index, (key, now));
                    return false;
                }
            }
        }
        self.unmake(index);
        STALE[4].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        true
    }

    /// Whether host slot `index` shows the placed texture the title's record
    /// at `handle` views, by an earlier view of it.
    pub(crate) fn texture_slot_shows_other(&self, index: u32, handle: u64) -> bool {
        let key = record_key(handle);
        let Some(shown) = self.made.get(index) else {
            return false;
        };
        if key == 0 || shown == 0 || shown == key {
            return false;
        }
        let view = |key: u64| view_at(key as u32, (key >> 32) as u32);
        matches!((view(shown), view(key)), (Some(a), Some(b)) if same_placed_texture(&a, &b))
    }

    /// Slot `index` is made from the record at `handle` (what it is now)
    /// when `list`, which holds the upload of what that view shows, has run:
    /// until then it shows what it showed. (Made at once, the frames on the
    /// card read the new view before its levels were filled.)
    pub(crate) fn texture_slot_flip(&self, list: &crate::host::HostList, index: u32, handle: u64) {
        self.flip(list, index, handle, None);
    }

    /// The same of a slot that shows nothing of the texture the record at
    /// `handle` views, or something else: until then it shows the neutral
    /// texture ([`show_after`]).
    pub(crate) fn texture_slot_show_after(&self, list: &crate::host::HostList, index: u32, handle: u64) {
        if self.show_later(index, handle) {
            self.flip(list, index, handle, Some(crate::frame::current()));
        }
    }

    /// What slot `index` shows until the texture the record at `handle` views
    /// is shown. False: it shows it, or there is none.
    fn show_later(&self, index: u32, handle: u64) -> bool {
        use std::sync::atomic::Ordering::Relaxed;
        let key = record_key(handle);
        let view = view_at(key as u32, (key >> 32) as u32);
        let (true, Some(view)) = (key != 0, view) else {
            return false;
        };
        let stale = {
            let _made = self.made.lock(index);
            let Some(shown) = self.made.get(index) else {
                return false;
            };
            if shown == key {
                return false;
            }
            // (As `mirror`.)
            let stale = self.note_buffer(index, Some(&view), true);
            let old = view_at(shown as u32, (shown >> 32) as u32);
            let kept = old.is_some_and(|old| match kin(&old, &view) {
                Kin::Same => true,
                kin => kin.kept(keep_shown()),
            });
            if shown == 0 || !kept {
                #[cfg(feature = "diagnostics")]
                note_before(index, shown, "neutral");
                self.quick_is(index, 0);
                self.made.set(index, 0);
                self.set_placed_owner(index, None);
                let kept = neutral_for(&view, self.at(index));
                self.put(index);
                self.clear_twin(index);
                LATE[3].fetch_add(u64::from(kept.is_some()), Relaxed);
                self.keep(index, kept);
            }
            stale
        };
        self.unmake_stale(index, stale);
        true
    }

    /// [`HostHeap::texture_slot_show_after`] of a texture whose upload `job`
    /// makes: shown when that has run.
    pub(crate) fn texture_slot_show_ahead(&self, job: &crate::texture::ahead::Job, index: u32, handle: u64) {
        if !self.show_later(index, handle) {
            return;
        }
        let (heap, key) = (std::ptr::from_ref(self) as usize, record_key(handle));
        let late = Some(crate::frame::current());
        let asked = crate::host::now_ms();
        job.then(move || {
            let flip = Flip { after: 0, heap, index, handle, key, late, asked };
            FLIPS.lock().unwrap_or_else(|e| e.into_inner()).push(flip);
        });
    }

    /// [`HostHeap::mirror`] of a slot whose table a draw read: a texture made
    /// a moment ago is shown after the draw's list ([`show_after`]; the draw
    /// reads the table's dynamic slots).
    pub(crate) fn mirror_after(&self, list: &crate::host::HostList, index: u32, handle: u64) {
        if show_after()
            && index < self.count
            && !self.texture_slot_made_from(index, handle)
            && let Some(view) = super::view(handle)
        {
            // Keep `placed` (and its `ResourceRef` owner) alive across every
            // dereference of `resource`.
            let placed = placed_texture(&view);
            let resolved = placed.as_ref().map_or(&view, |p| &p.view);
            if let &View::Shader(resource, _) = resolved
                && crate::resource::buffer_bytes(resource as crate::object::This).is_none()
                && crate::resource::host_resource(resource as crate::object::This).is_some_and(crate::pool::fresh)
                && !crate::texture::ahead::there(resource as crate::object::This)
            {
                return self.texture_slot_show_after(list, index, handle);
            }
        }
        self.mirror(index, handle, true);
    }

    fn flip(&self, list: &crate::host::HostList, index: u32, handle: u64, late: Option<u64>) {
        let (heap, key) = (std::ptr::from_ref(self) as usize, record_key(handle));
        let asked = crate::host::now_ms();
        list.defer_host(move || {
            let after = crate::host::submissions().0;
            let flip = Flip { after, heap, index, handle, key, late, asked };
            FLIPS.lock().unwrap_or_else(|e| e.into_inner()).push(flip);
        });
    }

    /// Makes the slots whose new view's upload has run
    /// ([`HostHeap::texture_slot_flip`]); returns how many.
    pub(crate) fn texture_slots_flip(&self) -> usize {
        use std::sync::atomic::Ordering::Relaxed;
        let finished = crate::host::submissions().1;
        let ready: Vec<Flip> = {
            let mut flips = FLIPS.lock().unwrap_or_else(|e| e.into_inner());
            if flips.is_empty() {
                return 0;
            }
            let own = std::ptr::from_ref(self) as usize;
            let (ready, wait): (Vec<Flip>, Vec<Flip>) =
                flips.drain(..).partition(|f| f.after <= finished && f.heap == own);
            *flips = wait;
            ready
        };
        {
            let mut pending = NEW_PENDING.lock().unwrap_or_else(|e| e.into_inner());
            if !pending.is_empty() {
                for f in &ready {
                    pending.remove(&(f.heap, f.index, f.key));
                }
            }
        }
        let mut flipped = 0;
        let mut requeue = Vec::new();
        for f in ready {
            let _made = self.made.lock(f.index);
            let Some(shown) = self.made.get(f.index) else {
                continue;
            };
            if shown == f.key {
                continue;
            }
            // Written again since, or made null: that write is in the queue.
            // (A slot shown late is null until now.)
            let view = view_at(f.key as u32, (f.key >> 32) as u32);
            let there = shown != 0 || f.late.is_some();
            let (true, true, Some(view)) = (f.key != 0 && record_key(f.handle) == f.key, there, view) else {
                FLIPPED[1].fetch_add(1, Relaxed);
                LATE[4].fetch_add(u64::from(f.late.is_some()), Relaxed);
                continue;
            };
            // Its memory went meanwhile: the texture pass finds that.
            if f.late.is_some()
                && let View::Placed(_, resource, desc) = &view
            {
                let address = u64::from(desc[10]) | u64::from(desc[11]) << 32;
                let len = crate::resource::placed_bytes(resource).unwrap_or(1);
                if crate::texture::mapped_bytes(address as *const u8, len as usize) == 0 {
                    LATE[4].fetch_add(1, Relaxed);
                    continue;
                }
            }
            let dynamic_placed = is_dynamic_placed_texture(&view);
            let gen_before = crate::resource::all_target_changes();
            let placed = dynamic_placed.then(|| placed_texture(&view)).flatten();
            let resolved_gen = placed.as_ref().map_or(gen_before, |p| p.resolved_gen);
            let (kept, deferred) = crate::resource::allocation_deferred_during(|| {
                make_twinned_with_placed(
                    &view,
                    dynamic_placed.then_some(placed.as_ref()),
                    self.at(f.index),
                    None,
                )
            });
            if let Some(kept) = kept {
                #[cfg(feature = "diagnostics")]
                note_before(f.index, shown, if f.late.is_some() { "shown after its upload" } else { "flip" });
                self.put(f.index);
                #[cfg(feature = "diagnostics")]
                let was = shown;
                let kept_raw = kept.as_ref().map_or(0, |r| windows::core::Interface::as_raw(r) as usize);
                self.quick_is(f.index, 0);
                self.made.set(f.index, f.key);
                self.set_placed_owner(
                    f.index,
                    dynamic_placed.then_some((
                        f.key,
                        placed.as_ref().map_or(0, PlacedTexture::owner_id),
                        kept_raw,
                        resolved_gen,
                    )),
                );
                self.keep(f.index, kept);
                #[cfg(feature = "diagnostics")]
                if f.late.is_some() && was != 0 {
                    same_bytes(was, f.key);
                }
                crate::texlate::note(
                    if f.late.is_some() { crate::texlate::Kind::New } else { crate::texlate::Kind::Sharper },
                    f.asked,
                );
                if let Some(since) = f.late {
                    let frames = crate::frame::current().saturating_sub(since);
                    let n = LATE[0].fetch_add(1, Relaxed) + 1;
                    let all = LATE[1].fetch_add(frames, Relaxed) + frames;
                    let most = LATE[2].fetch_max(frames, Relaxed).max(frames);
                    if n == 1 || n.is_multiple_of(5000) {
                        tracing::info!(
                            "slots shown after their texture's upload (VICTORY_SHOW_AFTER=1), since the start: {n}, {:.2} frames after they were asked for, the longest {most}; the neutral texture meanwhile in {}; {} not shown any more by then",
                            all as f64 / n as f64,
                            LATE[3].load(Relaxed),
                            LATE[4].load(Relaxed)
                        );
                    }
                } else {
                    flipped += 1;
                }
            } else if deferred {
                // The backing can be evicted while a flip is waiting; requeue
                // for the texture pass after releasing `made`.
                requeue.push(f.index);
            } else {
                // Creation failed without allocation deferral (for example an
                // out-of-bounds view or blocked owner): replace any prior or
                // neutral descriptor with a matching null view so stale data
                // cannot remain bound.
                if !self.sampler {
                    null_for(&view, self.at(f.index));
                }
                self.put(f.index);
                self.quick_is(f.index, 0);
                self.made.set(f.index, f.key);
                self.set_placed_owner(
                    f.index,
                    dynamic_placed.then_some((f.key, 0, 0, resolved_gen)),
                );
                self.twin_made(f.index, false);
                self.keep(f.index, None);
            }
        }
        if !requeue.is_empty() {
            let frame = crate::frame::current();
            let mut textures = self.textures.lock().unwrap_or_else(|e| e.into_inner());
            for index in requeue {
                textures.changed(index, frame, frame + 1);
            }
        }
        FLIPPED[0].fetch_add(flipped as u64, Relaxed);
        flipped
    }

    /// The memory of the texture title slot `index` views is there (again):
    /// see [`HostHeap::texture_slot_gone`].
    pub(crate) fn texture_slot_here(&self, index: u32) {
        let mut away = self.away.lock().unwrap_or_else(|e| e.into_inner());
        if !away.is_empty() {
            away.remove(&index);
        }
    }

    /// Makes host slot `index` null again, as it is before its view is
    /// made: a null view of the kind it held. Under its lock ([`Made`]).
    fn unmake(&self, index: u32) {
        self.quick_is(index, 0);
        let Some(key) = self.made.get(index) else {
            return;
        };
        self.made.set(index, 0);
        self.set_placed_owner(index, None);
        if key == 0 {
            return;
        }
        match view_at(key as u32, (key >> 32) as u32) {
            Some(view) => null_for(&view, self.at(index)),
            None => null_view(self.at(index)),
        }
        self.put(index);
        self.clear_twin(index);
        self.write_stride(index, 0);
        self.keep(index, None);
    }

    /// The slots [`HostHeap::note_buffer`] found stale for slot `index`, with
    /// what they were made from then, become null. None of their locks is
    /// held here; one made from another record meanwhile is left as it is.
    fn unmake_stale(&self, index: u32, stale: Vec<(u32, u64)>) {
        for (slot, key) in stale {
            if slot == index || key == 0 {
                continue;
            }
            let _made = self.made.lock(slot);
            if self.made.get(slot) == Some(key) {
                self.unmake(slot);
            }
        }
    }

    /// Title slot `index` is a table's ([`HostHeap::tabled`]). `read`: a
    /// draw read it through the table and brought what it views up to
    /// date, so the slot leaves the queue of textures picked by index; the
    /// caller makes the host slot in place ([`HostHeap::mirror`]), as the
    /// pass would have, for a shader that picks it by index all the same.
    pub(crate) fn texture_slot_tabled(&self, index: u32, read: bool) {
        self.tabled_in(index, read, crate::frame::current());
    }

    /// [`HostHeap::texture_slot_tabled`] in frame `frame`.
    fn tabled_in(&self, index: u32, read: bool, frame: u64) {
        use std::sync::atomic::Ordering::Relaxed;
        let Some(stamp) = self.tabled.get(index as usize) else {
            return;
        };
        let now = (frame as u32).wrapping_add(1) & !TABLED_READ;
        let old = stamp.load(Relaxed);
        if !read {
            if old & !TABLED_READ != now {
                stamp.store(now, Relaxed);
            }
            return;
        }
        // Once per slot and frame: tables are read by every draw.
        if old != now | TABLED_READ {
            stamp.store(now | TABLED_READ, Relaxed);
            if tabled_out() && self.queued.may(index) {
                let mut textures = self.textures.lock().unwrap_or_else(|e| e.into_inner());
                if textures.take_out(index) {
                    TABLED_WRITES[1].fetch_add(1, Relaxed);
                }
            }
        }
    }

    /// Whether title slot `index` was a table's within the last
    /// [`TABLED_FRAMES`] frames before `frame` (never, if such slots are
    /// to be queued as any other).
    fn tabled_lately(&self, index: u32, frame: u64) -> bool {
        let stamp = self.tabled_stamp(index) & !TABLED_READ;
        let now = (frame as u32).wrapping_add(1) & !TABLED_READ;
        tabled_out() && stamp != 0 && u64::from(now.wrapping_sub(stamp)) <= TABLED_FRAMES
    }

    /// What `tabled` holds for title slot `index`; 0: never a table's.
    fn tabled_stamp(&self, index: u32) -> u32 {
        self.tabled
            .get(index as usize)
            .map_or(0, |s| s.load(std::sync::atomic::Ordering::Relaxed))
    }

    /// The queue's part of a write of title slot `index` in `frame`:
    /// `texture`, whether it now holds a view of a texture the title may
    /// fill; `seen` as for [`HostHeap::note_buffer`].
    fn queue_texture(&self, index: u32, texture: bool, seen: bool, frame: u64) {
        let stamp = if tabled_out() { self.tabled_stamp(index) } else { 0 };
        let now = (frame as u32).wrapping_add(1) & !TABLED_READ;
        // A table's slot lately: the draw that reads it uploads its
        // texture, so its write waits for that draw.
        let lately = stamp != 0 && u64::from(now.wrapping_sub(stamp & !TABLED_READ)) <= TABLED_FRAMES;
        // Made now by a draw of this frame that read it through a table:
        // up to date.
        let read_now = seen && stamp == now | TABLED_READ;
        // (A slot is written by one thread at a time.)
        if !(texture && !read_now) && !self.queued.may(index) {
            return;
        }
        let mut textures = self.textures.lock().unwrap_or_else(|e| e.into_inner());
        if texture && !read_now {
            let due = if lately && !seen {
                TABLED_WRITES[0].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                frame + TABLED_GRACE
            } else {
                0
            };
            textures.written(index, frame, due, seen);
        } else {
            textures.remove(index);
        }
    }

    /// Title slot `index` lies in a table a draw reads or had made afresh:
    /// what it holds is the draw's to make ([`HostHeap::copied_tabled`]).
    pub(crate) fn slot_tabled(&self, index: u32) {
        self.tabled_in(index, false, crate::frame::current());
    }

    /// The title slots that hold buffer views.
    pub(crate) fn buffer_slots(&self) -> Vec<u32> {
        self.buffers.all(u64::MAX)
    }

    /// Of [`HostHeap::buffer_slots`], those of one of `PARTS` parts: every
    /// slot is in the same part each time.
    pub(crate) fn buffer_slots_of<const PARTS: u32>(&self, part: u32) -> Vec<u32> {
        self.buffers.part::<PARTS>(part)
    }

    /// The title slots whose buffer view was written, or went, since the
    /// last call.
    pub(crate) fn buffer_changes(&self) -> Vec<u32> {
        self.changed.take()
    }

    /// Records what title slot `index` now holds for shaders that pick
    /// buffers from the heap: its element stride in the table, and whether
    /// it is a buffer view at all.
    /// `seen`: the slot was not written now, a draw or a pass made its
    /// host slot from what it held.
    /// Returns the title slots whose placed texture view the write made
    /// stale ([`Spans`]), each with the record its host slot is made from:
    /// out of the queue of textures already, their host slots are the
    /// caller's to make null when it has let go of its own slot's lock
    /// ([`HostHeap::unmake_stale`]).
    fn note_buffer(&self, index: u32, view: Option<&View>, seen: bool) -> Vec<(u32, u64)> {
        if self.sampler {
            return Vec::new();
        }
        // Measured for the texture side of the same gap (textures a shader
        // picks by a computed index are neither made, if placed, nor
        // uploaded): slots written with a placed texture view, and with a
        // view of a texture object.
        let texture = match view {
            Some(View::Placed(false, resource, _)) if resource[0] != 1 => {
                PLACED_TEXTURE_SLOTS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                true
            }
            Some(v) if native_tlas(v).is_some() => false,
            Some(View::Shader(r, _)) if crate::resource::buffer_bytes(*r as crate::object::This).is_none() => {
                OBJECT_TEXTURE_SLOTS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                true
            }
            _ => false,
        };
        self.queue_texture(index, texture, seen, crate::frame::current());
        // The memory a placed texture view describes. Not for a slot
        // tables read (thousands of writes a frame, each read by its
        // draw): no pass looks at it, and the same texture's view in a
        // slot shaders pick by index says what it says.
        let placed_view = match view {
            Some(View::Placed(false, resource, desc))
                if resource[0] != 1 && !self.tabled_lately(index, crate::frame::current()) =>
            {
                // (And the level the view starts at: word 4 of a texture
                // view's description, after format, dimension, mapping
                // and a pad.)
                let at = u64::from(desc[10]) | u64::from(desc[11]) << 32;
                let held = crate::resource::placed_bytes_from(resource, crate::resource::first_mip(desc));
                Some((resource, at, desc[4], held.map_or(u64::MAX, |held| at.saturating_add(held))))
            }
            _ => None,
        };
        // (Under the slot's lock, as every write of it: whether it is in
        // `spanned` changes only here.)
        let in_spans = |index: u32| {
            let bit = 1u64 << (index % 64);
            self.spanned.0.get(index as usize / 64).is_none_or(|w| w.load(std::sync::atomic::Ordering::Acquire) & bit != 0)
        };
        let stale = if placed_view.is_none() && !in_spans(index) && spans_quick() {
            SPANS_QUICK[0].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            Vec::new()
        } else {
            use std::sync::atomic::Ordering::Relaxed;
            SPANS_QUICK[1].fetch_add(1, Relaxed);
            let mut placed = self.placed.lock().unwrap_or_else(|e| e.into_inner());
            let span = placed_view.and_then(|(resource, start, first, held)| {
                Some((start, start.saturating_add(placed.bytes(resource)?), first, held))
            });
            match span {
                Some((start, end, first, held)) => {
                    SPANNED[0].fetch_min(start, Relaxed);
                    SPANNED[1].fetch_max(end, Relaxed);
                    placed.rule = stale_rule();
                    self.spanned.put(index);
                    let stale = placed.note(index, start, end, !seen);
                    placed.shows(index, first, held);
                    // (Under the lock of `placed`: a later write of one of
                    // them comes by here before it makes the slot.)
                    stale.into_iter().map(|slot| (slot, self.made.get(slot).unwrap_or(0))).collect()
                }
                None => {
                    placed.forget(index);
                    self.spanned.remove(index);
                    Vec::new()
                }
            }
        };
        if !stale.is_empty() {
            STALE[0].fetch_add(stale.len() as u64, std::sync::atomic::Ordering::Relaxed);
            let mut textures = self.textures.lock().unwrap_or_else(|e| e.into_inner());
            for &(slot, _) in &stale {
                textures.remove(slot);
            }
        }
        let (stride, kind) = view.map_or((None, 0), view_stride_and_kind);
        self.write_stride(index, stride.unwrap_or(0) | kind);
        if let Some((_, _, data)) = &self.strides
            && index < victory_shader::emit::SLOT_STRIDES_COUNT
        {
            write_view_address(*data, index, view);
        }
        let was = if stride.is_some() {
            self.buffers.put(index);
            true
        } else {
            self.buffers.remove(index)
        };
        if was {
            self.changed.put(index);
        }
        stale
    }

    /// Writes `word` (stride | format kind) for view-heap slot `index` into
    /// the host's [`victory_shader::emit::SLOT_STRIDES`] table.
    #[inline]
    fn write_stride(&self, index: u32, word: u32) {
        if let Some((_, _, data)) = &self.strides
            && let Some(shadow) = self.stride_shadow.get(index as usize)
        {
            if shadow.load(std::sync::atomic::Ordering::Relaxed) == word {
                return;
            }
            shadow.store(word, std::sync::atomic::Ordering::Relaxed);
            // SAFETY: inside the mapping; shaders read it while we write (a
            // single aligned dword: either value is one the slot had, and
            // dynamic slots are exclusively owned until submission finishes).
            unsafe { (*data as *mut u32).add(index as usize).write_volatile(word) };
        }
    }

    /// Gives back `n` dynamic slots from `first` (an index [`dynamic`]
    /// returned), if they belong to this heap (`id == self.id`).
    ///
    /// [`dynamic`]: HostHeap::dynamic
    pub(crate) fn release(&self, id: u64, first: u32, n: u32) {
        if id != self.id {
            return;
        }
        let Some(at) = first.checked_sub(self.count) else {
            return;
        };
        let mut guard = self.next.lock().unwrap_or_else(|e| e.into_inner());
        let end = (at + n).min(self.dynamic) as usize;
        if (at as usize) < end {
            if let Some(slots) = self.native_slots.get() {
                slots.release(first, self.count + end as u32);
            }
            let given = guard.1.set(at as usize, end, false) as u32;
            GIVEN[1].fetch_add(u64::from(given), std::sync::atomic::Ordering::Relaxed);
            self.free.fetch_add(given, std::sync::atomic::Ordering::Relaxed);
            self.too_long.store(u32::MAX, std::sync::atomic::Ordering::Relaxed);
        }
    }

    /// Makes slot `index` (a dynamic one) from `view` reusing an
    /// already-resolved [`PlacedTexture`] so descriptor creation and resource
    /// state transitions operate on the exact same pinned owner; a null view
    /// if the host cannot.
    pub(crate) fn make_at_resolved(
        &self,
        index: u32,
        view: &View,
        placed: Option<&PlacedTexture>,
    ) {
        let kept = make_twinned_with_placed(view, Some(placed), self.at(index), self.twin(index));
        if kept.is_none() {
            null_for(view, self.at(index));
        }
        self.put(index);
        self.twin_made(index, kept.is_some() && twinned(view));
        self.write_stride(index, if kept.is_some() { view_stride_word(view) } else { 0 });
        self.keep_view(index, view, kept.flatten());
    }

    /// Makes slot `index` (a dynamic one) the view the record `words` names,
    /// from this thread's descriptor of it if the recording `recording` has
    /// made one, else as [`HostHeap::make_at`] does, and keeps that
    /// ([`ListViews`]). What is known of the record in this recording;
    /// `None` (the slot is not made): not a view kept so.
    pub(crate) fn make_listed(&self, recording: u64, words: [u32; 3], index: u32) -> Option<Listed> {
        if self.sampler {
            return None;
        }
        LIST_VIEWS.with(|views| {
            let mut views = views.borrow_mut();
            if views.is_none() {
                *views = Some(ListViews::new()?);
            }
            let views = views.as_mut()?;
            let twins = self.twins.get(index as usize).is_some();
            if views.recording != recording || views.twins != twins {
                views.recording = recording;
                views.twins = twins;
                views.made.clear();
            }
            if !views.made.contains_key(&words) {
                let view = super::view_at(words[1], words[2])?;
                let step = if twins { 2 } else { 1 };
                let at = step * views.made.len() as u32;
                // (A constant buffer's bytes are copied for each draw.)
                if at + step > ListViews::SLOTS
                    || matches!(view, View::Constant(..))
                    || native_tlas(&view).is_some()
                    || is_dynamic_placed_texture(&view)
                {
                    return None;
                }
                let twin = twins && twinned(&view);
                let kept = make_twinned(&view, views.slot(at), twin.then(|| views.slot(at + 1)))?;
                let stride_word = view_stride_word(&view);
                let listed = Listed {
                    kind: crate::draw::view_kind(&view),
                    texture: is_texture_view(&view),
                    ready: (0, 0, false),
                    tabled: None,
                };
                views.made.insert(words, ListView { view, at, twin, stride_word, kept, listed });
            }
            let made = &views.made[&words];
            self.copy_from(index, views.slot(made.at));
            if made.twin && twins {
                self.copy_from(self.raw + index, views.slot(made.at + 1));
                self.twins[index as usize].store(true, std::sync::atomic::Ordering::Relaxed);
            } else {
                self.clear_twin(index);
            }
            self.write_stride(index, made.stride_word);
            // Most draws find their slot holding it already: avoid redundant COM
            // AddRef/Release (provenance: avatarmanz/VIctory-7900XTX #376, GPL-3.0).
            if !self.holds(index, made.kept.as_ref()) {
                self.keep(index, made.kept.clone());
            }
            Some(made.listed)
        })
    }

    /// As [`HostHeap::copy_to`], from the slot `from`, which the card does not
    /// see.
    fn copy_from(&self, to: u32, from: D3D12_CPU_DESCRIPTOR_HANDLE) {
        #[cfg(feature = "diagnostics")]
        if let Some(written) = self.written.get(to as usize) {
            written.store(crate::host::coarse_ms() as u32, std::sync::atomic::Ordering::Relaxed);
        }
        let Some(host) = host() else { return };
        // SAFETY: a slot of our heap, and one of this thread's heap.
        unsafe { host.device.CopyDescriptorsSimple(1, self.cpu(to), from, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV) };
    }

    /// Makes slot `index` read nothing: a null view, or a plain sampler.
    pub(crate) fn make_null(&self, index: u32) {
        if self.sampler {
            self.null_sampler(index);
        } else {
            null_view(self.at(index));
            self.put(index);
            self.write_stride(index, 0);
        }
        self.clear_twin(index);
        self.set_placed_owner(index, None);
        self.keep(index, None);
    }

    fn placed_owner(&self, index: u32) -> (u64, usize) {
        let owners = self.placed_owners.lock().unwrap_or_else(|e| e.into_inner());
        owners.get(&index).map_or((0, 0), |&(_, id, raw)| (id, raw))
    }

    fn set_placed_owner(&self, index: u32, entry: Option<(u64, u64, usize, u64)>) {
        if index >= self.count {
            return;
        }
        let published_gen = {
            let mut owners = self.placed_owners.lock().unwrap_or_else(|e| e.into_inner());
            match entry {
                Some((key, owner_id, host_raw, resolved_gen)) => {
                    owners.insert(index, (key, owner_id, host_raw));
                    Some(resolved_gen)
                }
                None => {
                    if !owners.is_empty() {
                        owners.remove(&index);
                    }
                    None
                }
            }
        };
        // After publishing into `placed_owners` and releasing its lock, check
        // whether `ALL_TARGET_CHANGES` or `last_target_gen` moved past the
        // generation at which this owner was resolved; if so, pull
        // `last_target_gen` back and arm `placed_recheck_needed` with `SeqCst`
        // so a concurrent scan that missed our insert must re-scan.
        if let Some(resolved_gen) = published_gen {
            std::sync::atomic::fence(Ordering::SeqCst);
            let current_gen = crate::resource::all_target_changes();
            let seen_gen = self.last_target_gen.load(Ordering::SeqCst);
            if current_gen != resolved_gen || seen_gen > resolved_gen {
                self.last_target_gen.fetch_min(resolved_gen, Ordering::SeqCst);
                self.placed_recheck_needed.store(true, Ordering::SeqCst);
            }
        }
    }

    fn kept_raw(&self, index: u32) -> usize {
        use windows::core::Interface;
        if !self
            .kept_some
            .get(index as usize)
            .is_some_and(|s| s.load(Ordering::Acquire))
        {
            return 0;
        }
        self.kept
            .get(index as usize)
            .and_then(|slot| {
                slot.lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .as_ref()
                    .map(|r| Interface::as_raw(r) as usize)
            })
            .unwrap_or(0)
    }

    /// Where the raw twin of slot `index` is made.
    fn twin(&self, index: u32) -> Option<D3D12_CPU_DESCRIPTOR_HANDLE> {
        (index < self.twins.len() as u32).then(|| staged(self.sampler, 1).unwrap_or_else(|| self.cpu(self.raw + index)))
    }

    /// The raw twin of slot `index` was made by this thread ([`make_twinned`]),
    /// or is a null view if it was not.
    fn twin_made(&self, index: u32, made: bool) {
        if !made {
            return self.clear_twin(index);
        }
        if let Some(shown) = self.twins.get(index as usize) {
            self.put_twin(index);
            shown.store(true, std::sync::atomic::Ordering::Relaxed);
        }
    }

    /// The raw twin of slot `index` is a null view again.
    fn clear_twin(&self, index: u32) {
        if let Some(made) = self.twins.get(index as usize)
            && made.swap(false, std::sync::atomic::Ordering::Relaxed)
        {
            null_buffer_view(self.at(self.raw + index));
            self.put(self.raw + index);
        }
    }

    /// Slot `index` now views `resource` (or nothing of the host's).
    fn keep(&self, index: u32, resource: Option<ID3D12Resource>) {
        if let Some(slots) = self.native_slots.get() {
            slots.forget(index);
        }
        self.keep_resource(index, resource);
    }

    fn keep_resource(&self, index: u32, resource: Option<ID3D12Resource>) {
        let Some(slot) = self.kept.get(index as usize) else {
            return;
        };
        let old = {
            use windows::core::Interface;
            // Nothing kept and nothing to keep (constant buffers, null
            // views): skip the slot lock (`ReZOnGithub 685b830`).
            if resource.is_none()
                && self
                    .kept_some
                    .get(index as usize)
                    .is_some_and(|s| !s.load(Ordering::Acquire))
            {
                return;
            }
            let mut slot = slot.lock().unwrap_or_else(|e| e.into_inner());
            // The same again: nothing to retire.
            if slot.as_ref().map(Interface::as_raw) == resource.as_ref().map(Interface::as_raw) {
                return;
            }
            if let Some(some) = self.kept_some.get(index as usize) {
                some.store(resource.is_some(), Ordering::Release);
            }
            std::mem::replace(&mut *slot, resource)
        };
        // A running submission may still read the old resource through
        // this slot.
        if let Some(old) = old {
            crate::host::retire_resource(old);
        }
    }

    /// Whether slot `index` views `resource` (or nothing, with `None`).
    fn holds(&self, index: u32, resource: Option<&ID3D12Resource>) -> bool {
        use windows::core::Interface;
        self.kept.get(index as usize).is_some_and(|slot| {
            slot.lock().unwrap_or_else(|e| e.into_inner()).as_ref().map(Interface::as_raw)
                == resource.map(Interface::as_raw)
        })
    }

    /// What slot `index` views.
    pub(crate) fn kept_in(&self, index: u32) -> Option<ID3D12Resource> {
        self.kept.get(index as usize)?.lock().unwrap_or_else(|e| e.into_inner()).clone()
    }

    /// Makes slot `index` (a dynamic one) a raw buffer view of `len` bytes
    /// of `buffer` from byte `offset` (a multiple of 16): a root SRV with a
    /// size, whose reads past it return zeros instead of reaching whatever
    /// the host placed after the buffer.
    pub(crate) fn raw_at(&self, index: u32, buffer: &ID3D12Resource, offset: u64, len: u64) {
        let Some(host) = host() else { return };
        let elements = (len / 4).clamp(1, u64::from(u32::MAX)) as u32;
        let desc = D3D12_SHADER_RESOURCE_VIEW_DESC {
            Format: DXGI_FORMAT_R32_TYPELESS,
            ViewDimension: D3D12_SRV_DIMENSION_BUFFER,
            Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
            Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                Buffer: D3D12_BUFFER_SRV {
                    FirstElement: offset / 4,
                    NumElements: elements,
                    StructureByteStride: 0,
                    Flags: D3D12_BUFFER_SRV_FLAG_RAW,
                },
            },
        };
        // SAFETY: a slot of our heap and a live buffer covering the range.
        unsafe {
            host.device
                .CreateShaderResourceView(buffer, Some(&desc), self.at(index))
        };
        self.put(index);
        // (Bound by its handle, not read by index.)
        self.clear_twin(index);
        self.write_stride(index, 1);
        self.keep(index, Some(buffer.clone()));
    }

    /// Makes slot `index` (a dynamic one) a raw buffer view that reads
    /// zeros.
    pub(crate) fn null_raw_at(&self, index: u32) {
        null_buffer_view(self.at(index));
        self.put(index);
        self.clear_twin(index);
        self.write_stride(index, 0);
        self.keep(index, None);
    }

    /// Makes slot `index` (a dynamic one) a raw unordered access view of
    /// `len` bytes of `buffer` from byte `offset` (a multiple of 16): a
    /// root UAV with a size, whose accesses past it do nothing instead of
    /// reaching whatever the host placed after the buffer.
    pub(crate) fn raw_uav_at(&self, index: u32, buffer: &ID3D12Resource, offset: u64, len: u64) {
        let Some(host) = host() else { return };
        let elements = (len / 4).clamp(1, u64::from(u32::MAX)) as u32;
        let desc = D3D12_UNORDERED_ACCESS_VIEW_DESC {
            Format: DXGI_FORMAT_R32_TYPELESS,
            ViewDimension: D3D12_UAV_DIMENSION_BUFFER,
            Anonymous: D3D12_UNORDERED_ACCESS_VIEW_DESC_0 {
                Buffer: D3D12_BUFFER_UAV {
                    FirstElement: offset / 4,
                    NumElements: elements,
                    StructureByteStride: 0,
                    CounterOffsetInBytes: 0,
                    Flags: D3D12_BUFFER_UAV_FLAG_RAW,
                },
            },
        };
        // SAFETY: a slot of our heap and a live buffer covering the range.
        unsafe {
            host.device.CreateUnorderedAccessView(
                buffer,
                None::<&ID3D12Resource>,
                Some(&desc),
                self.at(index),
            )
        };
        self.put(index);
        self.clear_twin(index);
        self.write_stride(index, 1);
        self.keep(index, Some(buffer.clone()));
    }

    /// Makes slot `index` (a dynamic one) a raw unordered access view of
    /// nothing: reads give zeros, writes are dropped.
    pub(crate) fn null_raw_uav_at(&self, index: u32) {
        let Some(host) = host() else { return };
        let desc = D3D12_UNORDERED_ACCESS_VIEW_DESC {
            Format: DXGI_FORMAT_R32_TYPELESS,
            ViewDimension: D3D12_UAV_DIMENSION_BUFFER,
            Anonymous: D3D12_UNORDERED_ACCESS_VIEW_DESC_0 {
                Buffer: D3D12_BUFFER_UAV {
                    NumElements: 1,
                    Flags: D3D12_BUFFER_UAV_FLAG_RAW,
                    ..Default::default()
                },
            },
        };
        // SAFETY: a slot of our heap; a null resource is allowed with a desc.
        unsafe {
            host.device.CreateUnorderedAccessView(
                None::<&ID3D12Resource>,
                None::<&ID3D12Resource>,
                Some(&desc),
                self.at(index),
            )
        };
        self.put(index);
        self.clear_twin(index);
        self.write_stride(index, 0);
        self.keep(index, None);
    }

    /// The shader-visible handle of slot `index`.
    pub(crate) fn gpu_handle(&self, index: u32) -> D3D12_GPU_DESCRIPTOR_HANDLE {
        D3D12_GPU_DESCRIPTOR_HANDLE {
            ptr: self.gpu + u64::from(index) * u64::from(self.step),
        }
    }

    /// Makes slot `index` (a dynamic one) a constant buffer view of host
    /// memory, `size` bytes at `gpu` of `buffer`, which the slot holds.
    ///
    /// The list holds it too, until it is reset; the slot shows it until it
    /// is made again, and a shader that picks the slot by an index it
    /// computes from wrong data reads it through there. The list's scratch
    /// chunks go back to the pool and are freed past its limit: session n02
    /// lost the device to a page fault in the memory of three freed ones.
    /// `VICTORY_SLOT_HOLDS=0`: only the list holds it, as before.
    pub(crate) fn constant_at(&self, index: u32, buffer: &ID3D12Resource, gpu: u64, size: u32) {
        static HOLDS: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let holds = *HOLDS.get_or_init(|| std::env::var_os("VICTORY_SLOT_HOLDS").is_none_or(|v| v != "0"));
        let Some(host) = host() else { return };
        let desc = D3D12_CONSTANT_BUFFER_VIEW_DESC {
            BufferLocation: gpu,
            SizeInBytes: constant_bytes(size),
        };
        // SAFETY: a slot of our heap; the memory is the caller's.
        unsafe {
            host.device
                .CreateConstantBufferView(Some(&desc), self.at(index))
        };
        self.put(index);
        // A program that reads it as raw bytes reads its twin.
        let twin = self.twin(index).filter(|_| desc.SizeInBytes > 0);
        if let Some(twin) = twin {
            // SAFETY: a live buffer that holds the range.
            let first = (gpu - unsafe { buffer.GetGPUVirtualAddress() }) / 4;
            let raw = D3D12_SHADER_RESOURCE_VIEW_DESC {
                Format: DXGI_FORMAT_R32_TYPELESS,
                ViewDimension: D3D12_SRV_DIMENSION_BUFFER,
                Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
                Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                    Buffer: D3D12_BUFFER_SRV {
                        FirstElement: first,
                        NumElements: desc.SizeInBytes / 4,
                        StructureByteStride: 0,
                        Flags: D3D12_BUFFER_SRV_FLAG_RAW,
                    },
                },
            };
            // SAFETY: as above, and a slot of our heaps.
            unsafe { host.device.CreateShaderResourceView(buffer, Some(&raw), twin) };
        }
        self.twin_made(index, twin.is_some());
        self.write_stride(index, 0);
        // After the descriptor: what it showed is let go when nothing shows
        // it any more.
        self.keep(index, holds.then(|| buffer.clone()));
    }

    /// Renderer port 5 (`crate::port_cbv`): makes slot `index` (a dynamic
    /// one) show the constant buffer of `size` bytes at `gpu` of `buffer`
    /// (title memory in place) with nothing copied: its raw twin a raw view
    /// of those bytes, which translated shaders read; the slot itself a
    /// constant buffer view of them where that can be (on 256 bytes), else a
    /// null one. Whether the twin was made (`false`: the caller copies the
    /// bytes as before).
    pub(crate) fn constant_raw_at(&self, index: u32, buffer: &ID3D12Resource, gpu: u64, size: u32) -> bool {
        let Some(host) = host() else { return false };
        let Some(twin) = self.twin(index) else { return false };
        // SAFETY: a live buffer.
        let (base, width) = unsafe { (buffer.GetGPUVirtualAddress(), buffer.GetDesc().Width) };
        let Some(offset) = gpu.checked_sub(base).filter(|&o| o < width && crate::port_cbv::raw_ok(o)) else {
            return false;
        };
        let bytes = u64::from(constant_bytes(size)).min(width - offset) & !3;
        if bytes == 0 {
            return false;
        }
        let raw = D3D12_SHADER_RESOURCE_VIEW_DESC {
            Format: DXGI_FORMAT_R32_TYPELESS,
            ViewDimension: D3D12_SRV_DIMENSION_BUFFER,
            Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
            Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                Buffer: D3D12_BUFFER_SRV {
                    FirstElement: offset / 4,
                    NumElements: (bytes / 4) as u32,
                    StructureByteStride: 0,
                    Flags: D3D12_BUFFER_SRV_FLAG_RAW,
                },
            },
        };
        let cbv = D3D12_CONSTANT_BUFFER_VIEW_DESC {
            BufferLocation: if gpu.is_multiple_of(256) { gpu } else { 0 },
            SizeInBytes: if gpu.is_multiple_of(256) { constant_bytes(size).min(((width - offset) & !255) as u32) } else { 0 },
        };
        let cbv = if cbv.SizeInBytes == 0 { D3D12_CONSTANT_BUFFER_VIEW_DESC::default() } else { cbv };
        // SAFETY: slots of our heaps; the ranges lie in the live buffer (a
        // zero location makes a null constant buffer view).
        unsafe {
            host.device.CreateConstantBufferView(Some(&cbv), self.at(index));
            self.put(index);
            host.device.CreateShaderResourceView(buffer, Some(&raw), twin);
        }
        self.twin_made(index, true);
        self.write_stride(index, 0);
        self.keep(index, Some(buffer.clone()));
        true
    }

    /// Where a descriptor for slot `index` is made, then [`HostHeap::put`]:
    /// a slot of this thread's in a heap the card does not see
    /// ([`staged`]), or the slot itself.
    fn at(&self, index: u32) -> D3D12_CPU_DESCRIPTOR_HANDLE {
        staged(self.sampler, 0).unwrap_or_else(|| self.cpu(index))
    }

    /// The descriptor this thread made last ([`HostHeap::at`]) goes to slot
    /// `index`.
    fn put(&self, index: u32) {
        self.copy_to(index, 0);
    }

    /// The raw twin this thread made last goes to that of slot `index`.
    fn put_twin(&self, index: u32) {
        if self.twin(index).is_some() {
            self.copy_to(self.raw + index, 1);
        }
    }

    fn copy_to(&self, to: u32, which: u32) {
        #[cfg(feature = "diagnostics")]
        if let Some(written) = self.written.get(to as usize) {
            written.store(crate::host::coarse_ms() as u32, std::sync::atomic::Ordering::Relaxed);
        }
        let (Some(from), Some(host)) = (staged(self.sampler, which), host()) else {
            return; // made in place
        };
        let kind = if self.sampler {
            D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER
        } else {
            D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV
        };
        // SAFETY: a slot of our heap, and one of this thread's heap slots.
        unsafe { host.device.CopyDescriptorsSimple(1, self.cpu(to), from, kind) };
    }

    fn cpu(&self, index: u32) -> D3D12_CPU_DESCRIPTOR_HANDLE {
        D3D12_CPU_DESCRIPTOR_HANDLE {
            ptr: self.cpu + index as usize * self.step as usize,
        }
    }

    /// The records last seen of the title's heap, `step` bytes each, held for
    /// threads that compare what they read of its changed pages beside one
    /// another ([`Copied::changes`]; `super::sync_heaps`: the title copies
    /// records into its heap as bytes).
    ///
    /// The page says only that something on it was written, and `mirror`
    /// by itself would take every buffer or placed-texture record on it for
    /// newly written each time (it leaves such a slot unmade until a draw
    /// or a pass uses it): the texture queue's entry for the slot would
    /// start again, and the buffer pass would look at 2,048 slots for one
    /// that changed.
    pub(crate) fn copied_begin(&self, step: usize) -> Copied<'_> {
        let mut copied = self.copied.lock().unwrap_or_else(|e| e.into_inner());
        if copied.is_empty() {
            // Nothing there: a slot never looked at that holds nothing is
            // as it was.
            copied.resize(self.count as usize * step, 0);
        }
        let (at, len) = (copied.as_mut_ptr(), copied.len());
        Copied { at, len, step, _held: copied }
    }

    /// Whether new records of draws' tables are left to the draws
    /// (`VICTORY_HEAP_SYNC=all`: every new record is mirrored at once, as
    /// it first was).
    ///
    /// The title builds each draw's tables in its heap by copying records
    /// there as bytes: 60,000 to 210,000 a frame in its world (session
    /// play-1125-0127), 0.9 microseconds each on the submitting thread and
    /// 1.4 more in the buffer pass they were all handed to, 120 to 600 ms
    /// of every frame, and the draws make what they read of them anyway.
    pub(crate) fn copied_waits(&self) -> bool {
        static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *ON.get_or_init(|| std::env::var_os("VICTORY_HEAP_SYNC").is_none_or(|v| v != "all"))
    }

    /// Whether a draw read title slot `index` through a table within the
    /// last few frames before `frame` ([`HostHeap::tabled`]): what it holds
    /// is such a draw's to make.
    pub(crate) fn copied_tabled(&self, index: u32, frame: u64) -> bool {
        if !self.copied_waits() {
            return false;
        }
        let stamp = self.tabled_stamp(index) & !TABLED_READ;
        let now = (frame as u32).wrapping_add(1) & !TABLED_READ;
        stamp != 0 && now.wrapping_sub(stamp) & !TABLED_READ <= COPIED_FRAMES
    }

    /// New records the title copied into slots `slots` in `frame` wait for
    /// the draw that may read them through a table.
    pub(crate) fn copied_wait(&self, frame: u64, slots: Vec<u32>) {
        self.waiting.lock().unwrap_or_else(|e| e.into_inner()).push_back((frame, slots));
    }

    /// The slots that have waited long enough at `frame`
    /// ([`HostHeap::copied_wait`]), each once.
    pub(crate) fn copied_before(&self, frame: u64) -> Vec<u32> {
        let mut waiting = self.waiting.lock().unwrap_or_else(|e| e.into_inner());
        let mut out = Vec::new();
        while waiting.front().is_some_and(|(written, _)| written + COPIED_GRACE <= frame) {
            out.extend(waiting.pop_front().into_iter().flat_map(|(_, slots)| slots));
        }
        drop(waiting);
        out.sort_unstable();
        out.dedup();
        out
    }

    /// Makes host slot `index` from the record the title has at `handle`
    /// if it changed since, and returns the view recorded there. Buffer
    /// views are made only for draws (`for_draw`), which also copy their
    /// data: a host copy of a buffer is made on first use.
    pub(crate) fn mirror(&self, index: u32, handle: u64, for_draw: bool) -> Option<View> {
        self.mirror_with_placed(index, handle, for_draw, None)
            .map(|(view, _)| view)
    }

    /// [`HostHeap::mirror`] also returning the exact pinned [`PlacedTexture`]
    /// used to build or validate the slot so callers (`table_entry`,
    /// `heap_textures`) pass the exact same `ResourceRef` into `readable_resolved`.
    pub(crate) fn mirror_resolved(
        &self,
        index: u32,
        handle: u64,
        for_draw: bool,
    ) -> Option<(View, Option<PlacedTexture>)> {
        self.mirror_with_placed(index, handle, for_draw, None)
    }

    /// [`HostHeap::mirror_resolved`] reusing an already-pinned [`PlacedTexture`]
    /// when supplied (`Some((pinned_key, resolved_gen, placed))`) AND the guest
    /// slot's record key under `self.made.lock(index)` still equals `pinned_key`.
    /// If the guest overwrote the slot meanwhile, or if `placed` is `None` from
    /// an older `resolved_gen` than `all_target_changes()` / `last_target_gen`,
    /// ignores the stale input, resolves fresh for the current key, builds from
    /// the fresh owner, and returns it; otherwise preserves `resolved_gen` into
    /// `set_placed_owner` so even a concurrent arrival after a `None` lookup
    /// arms `placed_recheck_needed`.
    pub(crate) fn mirror_with_placed(
        &self,
        index: u32,
        handle: u64,
        for_draw: bool,
        supplied_placed: Option<(u64, u64, Option<&PlacedTexture>)>,
    ) -> Option<(View, Option<PlacedTexture>)> {
        if index >= self.count {
            // Diagnostic (NVIDIA heap cap): power-of-two rate-limited warning
            // (`Marvisak #3475`).
            let (highest, warn) = PAST_HEAP.note(index, self.count);
            let past = slots_past(self.count, highest);
            if warn {
                tracing::warn!(
                    past,
                    index,
                    highest,
                    count = self.count,
                    for_draw,
                    "title slot past the host heap"
                );
            }
            return None;
        }
        // SAFETY: the handle is a descriptor slot in title memory.
        let [tag, i, generation] = unsafe { (handle as *const [u32; 3]).read_unaligned() };
        // Generations start at 1, so a made slot's key is never 0.
        let key = u64::from(generation) << 32 | u64::from(i);
        // Made from this record, and nothing has changed it: lockless fast path.
        // (`dynamic_placed` slots never arm `quick`, so they always enter the
        // locked owner-check path below.)
        if tag == VIEW_TAG
            && quick_mirror()
            && self.quick.get(index as usize).is_some_and(|k| k.load(Ordering::Acquire) == key)
            && let Some(view) = view_at(i, generation)
        {
            let supplied_matches =
                supplied_placed.is_some_and(|(pinned_key, _, _)| pinned_key == key);
            let placed = if supplied_matches { None } else { placed_texture(&view) };
            return Some((view, placed));
        }
        let held = self.made.lock(index);
        // Re-read the title slot record under `made`'s lock so a late mirror
        // cannot overwrite a newer view written by another thread (`Machad3x 60518f1`).
        // SAFETY: the handle is a descriptor slot in title memory.
        let [tag, i, generation] = unsafe { (handle as *const [u32; 3]).read_unaligned() };
        let key = u64::from(generation) << 32 | u64::from(i);
        let supplied = supplied_placed.and_then(|(pinned_key, supplied_gen, placed)| {
            if pinned_key != key {
                return None;
            }
            if placed.is_none()
                && (crate::resource::all_target_changes() != supplied_gen
                    || self.last_target_gen.load(Ordering::Acquire) > supplied_gen)
            {
                return None;
            }
            Some((supplied_gen, placed))
        });
        let shown = self.made.get(index).unwrap_or(0);
        let view = (tag == VIEW_TAG).then(|| view_at(i, generation)).flatten();
        let Some(view) = view else {
            // Not a view of ours (never written, or of a resource the
            // title released): nothing to read.
            if shown != 0 {
                self.quick_is(index, 0);
                self.made.set(index, 0);
                self.make_null(index);
                let _ = self.note_buffer(index, None, false);
            } else if for_draw {
                // Left for later and stale since: out of the sets and the stride table.
                let _ = self.note_buffer(index, None, false);
            }
            return None;
        };
        let dynamic_placed = is_dynamic_placed_texture(&view);
        if shown == key && dynamic_placed {
            // Re-verify the live placed-texture owner so freeing resource A and
            // creating resource B at the same address never leaves slot `index`
            // pointing at A while draw-time state transitions operate on B.
            let gen_before = crate::resource::all_target_changes();
            let (owned_placed, deferred_resolve) = if supplied.is_some() {
                (None, false)
            } else {
                crate::resource::allocation_deferred_during(|| placed_texture(&view))
            };
            let (supplied_gen, active_placed) = match supplied {
                Some((g, p)) => (g, p),
                None => (gen_before, owned_placed.as_ref()),
            };
            let resolved_gen = active_placed.map_or(supplied_gen, |p| p.resolved_gen.min(supplied_gen));
            let current_id = active_placed.map_or(0, PlacedTexture::owner_id);
            let (recorded_id, _) = self.placed_owner(index);
            let current_host = active_placed.map_or(0, PlacedTexture::host_raw_if_made);
            let kept_host = self.kept_raw(index);
            let changed = recorded_id != current_id
                || (current_id != 0 && current_host != 0 && current_host != kept_host)
                || (current_id == 0 && kept_host != 0);
            if changed {
                self.quick_is(index, 0);
                let (kept, deferred_make) = if deferred_resolve {
                    (None, true)
                } else {
                    crate::resource::allocation_deferred_during(|| {
                        make_twinned_with_placed(
                            &view,
                            Some(active_placed),
                            self.at(index),
                            self.twin(index),
                        )
                    })
                };
                let made_some = kept.is_some();
                let ready_key = if (deferred_resolve || deferred_make) && !made_some {
                    0
                } else {
                    key
                };
                let flat_kept = kept.flatten();
                let kept_raw = flat_kept
                    .as_ref()
                    .map_or(0, |r| windows::core::Interface::as_raw(r) as usize);
                self.made.set(index, ready_key);
                self.set_placed_owner(
                    index,
                    (ready_key != 0).then_some((
                        ready_key,
                        if made_some { current_id } else { 0 },
                        kept_raw,
                        resolved_gen,
                    )),
                );
                if !made_some && !self.sampler {
                    null_for(&view, self.at(index));
                }
                if made_some || !self.sampler {
                    self.put(index);
                }
                self.twin_made(index, made_some && twinned(&view));
                self.keep_view(index, &view, flat_kept);
            }
            return Some((view, owned_placed));
        }
        let mut pinned_placed: Option<PlacedTexture> = None;
        if shown != key {
            // Views of other textures in the memory this one describes
            // are stale: their host slots read nothing from here on, and
            // let go of the host copies they held.
            // (When this slot's lock is let go: a lock at a time.)
            let stale = self.note_buffer(index, Some(&view), for_draw);
            // A view of a texture object that has no host copy yet waits
            // for it, as buffer views wait for a draw: the slot is made
            // when the texture is first used (`super::make_waiting`), or
            // here when a draw reads the slot through a table.
            let waiting = if for_draw { None } else { view.waiting_texture() };
            if let Some(texture) = waiting {
                super::wait_for(texture, handle);
            }
            let later = !for_draw && (view.is_buffer() || waiting.is_some());
            // Another view of the placed texture the slot shows already (a
            // title that streams a texture in writes its view again with
            // more mips each time some arrive): the slot goes on showing
            // the old view until a draw or the texture pass makes the new
            // one and uploads what it shows. Made null here, the texture
            // read as nothing for as long as it waited in the queue, every
            // time it got sharper.
            if later
                && keep_views()
                && let Some(old) = view_at(shown as u32, (shown >> 32) as u32)
            {
                let kin = kin(&old, &view);
                if kin == Kin::Same {
                    KEPT_VIEWS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    drop(held);
                    self.unmake_stale(index, stale);
                    return Some((view, None));
                }
                // And of another texture that can stand for it
                // ([`keep_shown`]): black, then grey for a frame or two is
                // what popped.
                if is_texture_view(&old) {
                    let kept = kin.kept(keep_shown());
                    rewritten(kin, kept);
                    if kept {
                        drop(held);
                        self.unmake_stale(index, stale);
                        return Some((view, None));
                    }
                }
            }
            // A texture made while lists sent before it may still run: those
            // read this slot by index, and must not find it ([`show_new_later`]).
            // The slot shows what it showed (or the neutral texture), and the
            // new view when they have all run.
            let (after, allocation_deferred) = if !later
                && !self.sampler
                && show_new_later()
                && is_texture_view(&view)
            {
                crate::resource::allocation_deferred_during(|| begun_before(&view))
            } else {
                (None, false)
            };
            if let Some(after) = after {
                // Asked again (each draw that reads the slot, each pass)
                // while it waits: nothing more to do.
                let own = (std::ptr::from_ref(self) as usize, index, key);
                if !NEW_PENDING.lock().unwrap_or_else(|e| e.into_inner()).insert(own) {
                    drop(held);
                    self.unmake_stale(index, stale);
                    return Some((view, None));
                }
                let old = view_at(shown as u32, (shown >> 32) as u32);
                let kept = old.is_some_and(|old| match kin(&old, &view) {
                    Kin::Same => true,
                    kin => kin.kept(keep_shown()),
                });
                if shown == 0 || !kept {
                    #[cfg(feature = "diagnostics")]
                    note_before(index, shown, "neutral, new texture");
                    self.quick_is(index, 0);
                    self.made.set(index, 0);
                    self.set_placed_owner(index, None);
                    let kept = neutral_for(&view, self.at(index));
                    self.put(index);
                    self.clear_twin(index);
                    self.keep(index, kept);
                }
                let late = Some(crate::frame::current());
                let flip = Flip { after, heap: std::ptr::from_ref(self) as usize, index, handle, key, late, asked: crate::host::now_ms() };
                FLIPS.lock().unwrap_or_else(|e| e.into_inner()).push(flip);
                NEW_LATER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                drop(held);
                self.unmake_stale(index, stale);
                return Some((view, None));
            }
            #[cfg(feature = "diagnostics")]
            if !self.sampler && is_texture_view(&view) {
                note_before(index, shown, if later { "null, waits" } else { "at once" });
            }
            self.quick_is(index, 0);
            let mut built_owner = 0u64;
            let mut built_gen = match supplied {
                Some((g, _)) => g,
                None => crate::resource::all_target_changes(),
            };
            let (kept, deferred) = if later || allocation_deferred {
                (None, allocation_deferred)
            } else if matches!(view, View::Placed(..)) {
                crate::resource::allocation_deferred_during(|| {
                    if supplied.is_none() {
                        pinned_placed = placed_texture(&view);
                    }
                    let active = match supplied {
                        Some((_, p)) => p,
                        None => pinned_placed.as_ref(),
                    };
                    built_owner = active.map_or(0, PlacedTexture::owner_id);
                    if let Some(p) = active {
                        built_gen = built_gen.min(p.resolved_gen);
                    }
                    make_twinned_with_placed(
                        &view,
                        Some(active),
                        self.at(index),
                        self.twin(index),
                    )
                })
            } else {
                crate::resource::allocation_deferred_during(|| {
                    make_twinned(&view, self.at(index), self.twin(index))
                })
            };
            // A null descriptor during an allocation cooldown is still due (`TaynanGT #161`).
            let ready_key = if later || (deferred && kept.is_none()) { 0 } else { key };
            let made_some = kept.is_some();
            let flat_kept = kept.flatten();
            let kept_raw = flat_kept
                .as_ref()
                .map_or(0, |r| windows::core::Interface::as_raw(r) as usize);
            self.made.set(index, ready_key);
            self.set_placed_owner(
                index,
                (dynamic_placed && ready_key != 0).then_some((
                    ready_key,
                    if made_some { built_owner } else { 0 },
                    kept_raw,
                    built_gen,
                )),
            );
            if !made_some && !self.sampler {
                null_for(&view, self.at(index));
            }
            if made_some || !self.sampler {
                self.put(index);
            }
            self.twin_made(index, made_some && twinned(&view));
            self.keep_view(index, &view, flat_kept);
            // Never arm `quick` for dynamic writable placed textures so owner
            // switches (A -> B at the same address) are re-verified under lock.
            self.quick_is(index, if dynamic_placed { 0 } else { ready_key });
            drop(held);
            self.unmake_stale(index, stale);
        } else if supplied.is_none() && matches!(view, View::Placed(..)) {
            pinned_placed = placed_texture(&view);
        }
        Some((view, pinned_placed))
    }

    /// Whether slot `index` was made from no record: what [`HostHeap::mirror`]
    /// would find before it did nothing for one that holds no view.
    pub(crate) fn shows_none(&self, index: u32) -> bool {
        self.made.get(index).unwrap_or(0) == 0
    }

    /// Slot `index` is made from record `key` and stays so (0: not). Under
    /// the lock of `made`.
    fn quick_is(&self, index: u32, key: u64) {
        if let Some(quick) = self.quick.get(index as usize) {
            quick.store(key, std::sync::atomic::Ordering::Release);
        }
    }

    /// Whether slot `index` is backed by a made host descriptor (tests).
    #[cfg(test)]
    pub(crate) fn made(&self, index: u32) -> bool {
        self.made.get(index).is_some_and(|key| key != 0)
    }

    /// Whether host slot `index` shows `buffer`; `None` if it shows no
    /// resource (diagnostics: a buffer view's slot must show the host copy
    /// that is brought up to date for it).
    #[cfg(feature = "diagnostics")]
    pub(crate) fn shows(&self, index: u32, buffer: &ID3D12Resource) -> Option<bool> {
        let shown = self.kept_in(index)?;
        Some(windows::core::Interface::as_raw(&shown) == windows::core::Interface::as_raw(buffer))
    }

    /// What the queue of textures picked by index knows of title slot
    /// `index` at `frame`, in a few words (diagnostics).
    #[cfg(feature = "diagnostics")]
    pub(crate) fn texture_slot_state(&self, index: u32, frame: u64) -> String {
        let textures = self.textures.lock().unwrap_or_else(|e| e.into_inner());
        match textures.slots.get(&index) {
            None => "not queued".into(),
            Some(w) => format!(
                "queued due={} written@{} looks={} part_mapped={} recheck={}",
                if w.due <= frame { "now".to_string() } else { format!("@{}", w.due) },
                w.written,
                w.looks,
                w.mapped,
                w.checked.2 != 0
            ),
        }
    }

    /// The view host slot `index` was made from, in a few words, with the
    /// title and host addresses of a buffer view (diagnostics). "null" for
    /// a slot never made or left null.
    #[cfg(feature = "diagnostics")]
    pub(crate) fn describe_slot(&self, index: u32) -> Option<String> {
        let key = self.made.get(index)?;
        if key == 0 {
            return Some("null".into());
        }
        let view = view_at(key as u32, (key >> 32) as u32)?;
        let mut text: String = format!("{view:?}").chars().take(120).collect();
        if let Some((address, len)) = buffer_range(&view) {
            let host = crate::shadow::find(address, len.max(1)).map(|s| s.gpu(address));
            text.push_str(&format!(" title {address:#x}+{len:#x} host {host:#x?}"));
        }
        Some(text)
    }
}

/// Slots of this thread's own in heaps the card does not see, two of views
/// and two of samplers (`which` 0 or 1): descriptors are made there and
/// copied whole into the heaps it does see. Made in place they are seen half
/// written: an RX 7900 XTX saw a texture of no size in 4 % of its reads of
/// slots whose views a thread made over and over (the driver clears the
/// descriptor and fills it step by step), and none in 8,500 million with the
/// same descriptors copied, three times as fast
/// (`indirect::tests::descriptors_written_under_readers`). Title threads and
/// our passes rewrite slots while submissions run whose shaders pick them;
/// three device losses of 2026-11-25 (play-1125-0203, n02, n03) were reads of
/// memory no view names. `VICTORY_VIEW_COPY=0`: made in place, as before.
fn staged(sampler: bool, which: u32) -> Option<D3D12_CPU_DESCRIPTOR_HANDLE> {
    /// Threads at a time that have slots; more make views in place.
    const THREADS: u32 = 4096;
    struct Stage {
        starts: [usize; 2],
        steps: [usize; 2],
        _heaps: [ID3D12DescriptorHeap; 2],
    }
    // SAFETY: descriptor heaps are free-threaded; each thread has slots of
    // its own.
    unsafe impl Send for Stage {}
    // SAFETY: as above.
    unsafe impl Sync for Stage {}
    static STAGE: std::sync::OnceLock<Option<Stage>> = std::sync::OnceLock::new();
    /// Places given back, and how many were ever given out.
    static PLACES: Mutex<(Vec<u32>, u32)> = Mutex::new((Vec::new(), 0));
    struct Place(Option<u32>);
    impl Drop for Place {
        fn drop(&mut self) {
            if let Some(place) = self.0 {
                PLACES.lock().unwrap_or_else(|e| e.into_inner()).0.push(place);
            }
        }
    }
    thread_local! {
        static PLACE: Place = {
            let mut places = PLACES.lock().unwrap_or_else(|e| e.into_inner());
            Place(places.0.pop().or_else(|| {
                (places.1 < THREADS).then(|| {
                    places.1 += 1;
                    places.1 - 1
                })
            }))
        };
    }
    let stage = STAGE
        .get_or_init(|| {
            if std::env::var_os("VICTORY_VIEW_COPY").is_some_and(|v| v == "0") {
                return None;
            }
            let host = host()?;
            let make = |kind: D3D12_DESCRIPTOR_HEAP_TYPE| -> Option<ID3D12DescriptorHeap> {
                // SAFETY: plain heap creation.
                unsafe {
                    host.device.CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                        Type: kind,
                        NumDescriptors: 2 * THREADS,
                        ..Default::default()
                    })
                }
                .inspect_err(|e| tracing::warn!(error = %e, "no heap to make views in: they are made in place"))
                .ok()
            };
            let kinds = [D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER];
            let heaps = [make(kinds[0])?, make(kinds[1])?];
            // SAFETY: live heaps, plain queries.
            let (starts, steps) = unsafe {
                (
                    [0, 1].map(|i: usize| heaps[i].GetCPUDescriptorHandleForHeapStart().ptr),
                    kinds.map(|kind| host.device.GetDescriptorHandleIncrementSize(kind) as usize),
                )
            };
            Some(Stage { starts, steps, _heaps: heaps })
        })
        .as_ref()?;
    // (A thread that is ending has none any more.)
    let place = PLACE.try_with(|p| p.0).ok().flatten()?;
    let kind = usize::from(sampler);
    Some(D3D12_CPU_DESCRIPTOR_HANDLE {
        ptr: stage.starts[kind] + (2 * place + which) as usize * stage.steps[kind],
    })
}

/// Logs `what` the first time it happens.
macro_rules! once {
    ($($arg:tt)*) => {{
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| tracing::warn!($($arg)*));
    }};
}

/// A view description from its words, with the format as the host has it.
fn srv_desc(words: &[u32; 10]) -> D3D12_SHADER_RESOURCE_VIEW_DESC {
    let mut words = *words;
    words[0] = crate::resource::pc_format(words[0]);
    // SAFETY: 40 bytes in the host's layout.
    unsafe {
        std::ptr::from_ref(&words)
            .cast::<D3D12_SHADER_RESOURCE_VIEW_DESC>()
            .read_unaligned()
    }
}

fn uav_desc(words: &[u32; 10]) -> D3D12_UNORDERED_ACCESS_VIEW_DESC {
    let mut words = *words;
    words[0] = crate::resource::pc_format(words[0]);
    // SAFETY: 40 bytes in the host's layout.
    unsafe {
        std::ptr::from_ref(&words)
            .cast::<D3D12_UNORDERED_ACCESS_VIEW_DESC>()
            .read_unaligned()
    }
}

/// A buffer view of title memory at `address` on its host copy: the first
/// element moves by the copy's offset.
/// Bytes per element of a buffer view.
fn element_size(desc: &D3D12_SHADER_RESOURCE_VIEW_DESC) -> u64 {
    // SAFETY: a buffer view's union member.
    let buffer = unsafe { desc.Anonymous.Buffer };
    if buffer.StructureByteStride > 0 {
        u64::from(buffer.StructureByteStride)
    } else if buffer.Flags.0 & D3D12_BUFFER_SRV_FLAG_RAW.0 != 0 {
        4
    } else {
        crate::resource::element(desc.Format.0 as u32).map_or(0, |(bytes, _)| bytes)
    }
}

/// The title memory a buffer shader or unordered access view reads: its
/// address and length.
pub(crate) fn buffer_range(view: &View) -> Option<(u64, u64)> {
    let (base, words): (u64, &[u32; 10]) = match view {
        View::Shader(r, Some(desc)) => (
            crate::resource::buffer_bytes(*r as crate::object::This)?.0 as u64,
            desc,
        ),
        View::Placed(false, resource, desc) if resource[0] == 1 => (
            u64::from(desc[10]) | u64::from(desc[11]) << 32,
            desc[..10].try_into().ok()?,
        ),
        View::Unordered(r, _, Some(desc)) => {
            let base = crate::resource::buffer_bytes(*r as crate::object::This)?.0 as u64;
            return unordered_range(base, desc);
        }
        View::Placed(true, resource, desc) if resource[0] == 1 => {
            let base = u64::from(desc[10]) | u64::from(desc[11]) << 32;
            return unordered_range(base, desc[..10].try_into().ok()?);
        }
        _ => return None,
    };
    let desc = srv_desc(words);
    let element = element_size(&desc);
    // SAFETY: a buffer view's union member.
    let buffer = unsafe { desc.Anonymous.Buffer };
    Some((
        base + buffer.FirstElement * element,
        u64::from(buffer.NumElements) * element,
    ))
}

/// The address a shader finds in a descriptor of title memory: where a
/// buffer view starts, or the ray-tracing structure of a structure's view
/// (no resource: the view's description holds the address, and the shader
/// walks the structure from there itself).
pub(crate) fn view_address(view: &View) -> Option<u64> {
    const STRUCTURE: u32 = D3D12_SRV_DIMENSION_RAYTRACING_ACCELERATION_STRUCTURE.0 as u32;
    let words = match view {
        View::Shader(_, Some(desc)) => Some(&desc[..]),
        View::Placed(false, _, desc) => Some(&desc[..10]),
        _ => None,
    };
    if let Some(w) = words
        && w[1] == STRUCTURE
    {
        let at = u64::from(w[4]) | u64::from(w[5]) << 32;
        return (at != 0).then_some(at);
    }
    buffer_range(view).map(|range| range.0)
}

/// Enters the title address of `view` (0 for none, or a view that is no
/// buffer's or structure's) at title slot `index` of the view addresses
/// after the strides in the mapping at `data`
/// (`victory_shader::emit::VIEW_ADDRESSES`): shaders that pick a view at a
/// computed index and read its address as a pointer read it there.
fn write_view_address(data: usize, index: u32, view: Option<&View>) {
    use victory_shader::emit::{SLOT_STRIDES_COUNT, VIEW_ADDRESSES};
    if index >= SLOT_STRIDES_COUNT {
        return;
    }
    let address = view.and_then(view_address).unwrap_or(0);
    let at = (data + VIEW_ADDRESSES as usize) as *mut u32;
    // SAFETY: inside the mapping (`SLOT_STRIDES_BYTES`); shaders read it
    // while we write (each dword is one the slot had or has).
    unsafe {
        at.add(2 * index as usize).write_volatile(address as u32);
        at.add(2 * index as usize + 1).write_volatile((address >> 32) as u32);
    }
}

/// The title memory the host slot of buffer view `view` is made on
/// (address, bytes): the whole buffer the view is of, which is what
/// [`make`] finds the host copy by. [`buffer_range`] is the part of it the
/// view reads.
pub(crate) fn buffer_extent(view: &View) -> Option<(u64, u64)> {
    match view {
        View::Shader(r, Some(_)) | View::Unordered(r, _, Some(_)) => {
            let (data, len) = crate::resource::buffer_bytes(*r as crate::object::This)?;
            Some((data as u64, len))
        }
        View::Placed(_, resource, desc) if resource[0] == 1 => Some((
            u64::from(desc[10]) | u64::from(desc[11]) << 32,
            u64::from(resource[4]) | u64::from(resource[5]) << 32,
        )),
        _ => None,
    }
}

/// Whether a buffer view is brought up to date in the host copy its slot
/// shows ([`buffer_copy`]). On by default; `VICTORY_VIEW_EXTENT=0` looks the
/// copy up by the view's range alone, as before.
fn view_extent() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_VIEW_EXTENT").is_none_or(|v| v != "0"))
}

/// The host copy buffer view `view` reads, and the title range (address,
/// bytes) it reads there; `None` for other views and empty ranges.
///
/// The copy is the one the view's slot shows: [`make`] finds it by the
/// whole buffer the view is of ([`buffer_extent`]). Looked up by the range
/// the view reads, as it was, the copy is another one wherever the two
/// lookups differ: for memory outside buffer objects, a view of a part of a
/// buffer that crosses a window line (the buffer's window is two cells, the
/// part's one), and for buffer objects, a part that also lies in a smaller
/// buffer object. The slot then showed a copy nothing brought up to date,
/// and what shaders wrote through it was looked for in the other copy.
pub(crate) fn buffer_copy(view: &View) -> Option<(std::sync::Arc<crate::shadow::Shadow>, u64, u64)> {
    let (address, len) = buffer_range(view)?;
    if len == 0 {
        return None;
    }
    if view_extent()
        && let Some((base, width)) = buffer_extent(view)
        // (The console is lax: a view may begin past its buffer's end.)
        && address >= base
        && address - base < width.max(1)
        && let Some(shadow) = crate::shadow::find(base, width)
    {
        return Some((view_home(shadow, address, len, view_elements(view)?), address, len));
    }
    let shadow = crate::shadow::find(address, len)?;
    Some((view_home(shadow, address, len, view_elements(view)?), address, len))
}

/// [`buffer_range`] of an unordered access view of memory at `base`.
fn unordered_range(base: u64, words: &[u32; 10]) -> Option<(u64, u64)> {
    let desc = uav_desc(words);
    // SAFETY: a buffer view's union member.
    let buffer = unsafe { desc.Anonymous.Buffer };
    let element = if buffer.StructureByteStride > 0 {
        u64::from(buffer.StructureByteStride)
    } else if buffer.Flags.0 & D3D12_BUFFER_UAV_FLAG_RAW.0 != 0 {
        4
    } else {
        crate::resource::element(desc.Format.0 as u32)?.0
    };
    Some((
        base + buffer.FirstElement * element,
        u64::from(buffer.NumElements) * element,
    ))
}

/// The host resource a slot made by [`make`] views, if any.
type Kept = Option<ID3D12Resource>;

/// A structured buffer view (`first`, `count` elements of `stride` bytes,
/// `offset` bytes into its host copy) as a raw view of the same bytes:
/// its first dword and its dwords. Translated shaders read untyped buffers
/// as raw bytes at `element * stride + offset` (the stride from the
/// used-resource entry, or from the table by heap slot for a buffer picked
/// by index), and a raw read through a structured view gives zeros on
/// NVIDIA (reported upstream with PR #193; structured reads through raw
/// views were right there). So the host's view is raw whenever it can be:
/// the bytes must start on 16 (the host's raw view alignment) and be whole
/// dwords. A view that cannot stays structured.
fn raw_window(offset: u64, first: u64, count: u32, stride: u32) -> Option<(u64, u32)> {
    if stride == 0 {
        return None;
    }
    let start = first.checked_mul(u64::from(stride))?.checked_add(offset)?;
    let bytes = u64::from(count) * u64::from(stride);
    (start % 16 == 0 && bytes % 4 == 0)
        .then(|| u32::try_from(bytes / 4).ok().map(|dwords| (start / 4, dwords)))
        .flatten()
}

/// Whether the host accepts a buffer view with these fields of a buffer of
/// `width` bytes. It removes the device for one it does not (view creation
/// returns nothing), and the console is laxer: the title's heap holds
/// views past their memory's end, of no elements, of typeless formats
/// without the raw flag, with strides the host has no structured buffers
/// for. Since every buffer view of the heap is made for shaders that pick
/// them by index, one such view ended the run (2026-11-24, run 10).
fn buffer_view_ok(format: DXGI_FORMAT, first: u64, count: u32, stride: u32, raw: bool, width: u64) -> bool {
    use crate::resource::family;
    let f = format.0 as u32;
    let element = if raw {
        // raw: R32 typeless, counted in dwords, starting on 16 bytes
        if format != DXGI_FORMAT_R32_TYPELESS || stride != 0 || !first.is_multiple_of(4) {
            return false;
        }
        4
    } else if stride > 0 {
        // structured: no format, a stride the host takes
        if format != DXGI_FORMAT_UNKNOWN || stride > 2048 || !stride.is_multiple_of(4) {
            return false;
        }
        u64::from(stride)
    } else {
        // typed: a format with a size that is not typeless
        let typeless = family(f) == f && family(f + 1) == f;
        match crate::resource::element(f) {
            Some((bytes, 1)) if f != 0 && !typeless && bytes > 0 => bytes,
            _ => return false,
        }
    };
    count > 0
        && first
            .checked_add(u64::from(count))
            .and_then(|end| end.checked_mul(element))
            .is_some_and(|end| end <= width)
}

/// Raw twins ([`raw_twins`]) made since the last frame line, and those of them that
/// do not begin on 16 bytes, as a raw view must ([`move_raw`]), and those of
/// these that could not be moved there (null).
pub(crate) static TWINS: [std::sync::atomic::AtomicU64; 3] = [const { std::sync::atomic::AtomicU64::new(0) }; 3];

/// Moves the raw view at `at`, a slot of our heaps the card does not see,
/// which shows `bytes` from `start` on (GPU address), `by` bytes on: for a
/// twin whose view begins between two multiples of 16, where the host makes
/// no raw view. An AMD descriptor of a raw buffer is its address (48 bits),
/// its bytes and flags; only one that says just that is changed, and only
/// where what the handle points at is the descriptor. False: not moved.
pub(crate) fn move_raw(at: D3D12_CPU_DESCRIPTOR_HANDLE, start: u64, bytes: u32, by: u32) -> bool {
    // (Off dwords too, as the typed view it stands for begins there: a view
    // of 16 bit values at an odd one.)
    if by >= bytes {
        return false;
    }
    let mut words = [0u8; 16];
    if !crate::host::read_into(at.ptr as *const u8, &mut words) {
        return false;
    }
    let word = |i: usize| u32::from_le_bytes([words[4 * i], words[4 * i + 1], words[4 * i + 2], words[4 * i + 3]]);
    let address = u64::from(word(0)) | u64::from(word(1)) << 32;
    if address != start || word(2) != bytes || start >> 48 != 0 {
        return false;
    }
    let moved = start + u64::from(by);
    // SAFETY: this thread's own slot (it has slots of its own: the caller
    // asks only for those), read just now.
    unsafe {
        let words = at.ptr as *mut u32;
        words.write(moved as u32);
        words.add(1).write((moved >> 32) as u32);
        words.add(2).write(bytes - by);
    }
    true
}

/// The bytes a buffer view the host takes ([`buffer_view_ok`]) shows as a raw
/// view (first dword, dwords, and the bytes past that dword it begins: a raw
/// view begins on 16 bytes); of a buffer of `width` bytes.
fn twin_window(format: DXGI_FORMAT, first: u64, count: u32, stride: u32, raw: bool, width: u64) -> Option<(u64, u32, u32)> {
    use std::sync::atomic::Ordering::Relaxed;
    TWINS[0].fetch_add(1, Relaxed);
    let element = if raw {
        4
    } else if stride > 0 {
        u64::from(stride)
    } else {
        crate::resource::element(format.0 as u32).map_or(0, |(bytes, _)| bytes)
    };
    let (start, end) = (first * element, ((first + u64::from(count)) * element).next_multiple_of(4));
    let end = end.min(width & !3);
    let from = start & !15;
    if end <= start {
        return None;
    }
    let dwords = u32::try_from((end - from) / 4).ok()?;
    if from != start {
        TWINS[1].fetch_add(1, Relaxed);
    }
    Some((from / 4, dwords, (start - from) as u32))
}

fn buffer_srv(
    address: u64,
    len: u64,
    mut desc: D3D12_SHADER_RESOURCE_VIEW_DESC,
    at: D3D12_CPU_DESCRIPTOR_HANDLE,
    twin: Option<D3D12_CPU_DESCRIPTOR_HANDLE>,
) -> Option<Kept> {
    let mut shadow = crate::shadow::find(address, len)?;
    let mut offset = shadow.offset(address);
    let unit = element_size(&desc);
    // SAFETY: a buffer view's union member.
    let own = unsafe { &mut desc.Anonymous.Buffer };
    let start = address + own.FirstElement * unit;
    let kind = (own.StructureByteStride, own.Flags.0 & D3D12_BUFFER_SRV_FLAG_RAW.0 != 0, desc.Format.0 as u32);
    let home = view_home(shadow.clone(), start, u64::from(own.NumElements) * unit, kind);
    if !std::sync::Arc::ptr_eq(&home, &shadow) {
        // Its own bytes: it begins where they do.
        (shadow, offset) = (home, 0);
        own.FirstElement = 0;
    }
    // A structured view as a raw view of the same bytes (see `raw_window`).
    // SAFETY: a buffer view's union member.
    let structured = unsafe { desc.Anonymous.Buffer };
    let window = (desc.Format == DXGI_FORMAT_UNKNOWN)
        .then(|| raw_window(offset, structured.FirstElement, structured.NumElements, structured.StructureByteStride))
        .flatten();
    if let Some((first, count)) = window {
        desc.Format = DXGI_FORMAT_R32_TYPELESS;
        desc.Anonymous.Buffer = D3D12_BUFFER_SRV {
            FirstElement: first,
            NumElements: count,
            StructureByteStride: 0,
            Flags: D3D12_BUFFER_SRV_FLAG_RAW,
        };
    }
    let element = element_size(&desc);
    // SAFETY: a buffer view's union member.
    let buffer = unsafe { &mut desc.Anonymous.Buffer };
    if window.is_none() {
        if element == 0 || offset % element != 0 {
            once!(
                offset,
                element,
                "buffer view not on an element of its host copy"
            );
            return None;
        }
        buffer.FirstElement += offset / element;
    }
    let (first, count, stride) = (buffer.FirstElement, buffer.NumElements, buffer.StructureByteStride);
    let raw = buffer.Flags.0 & D3D12_BUFFER_SRV_FLAG_RAW.0 != 0;
    let width = shadow.width();
    if !buffer_view_ok(desc.Format, first, count, stride, raw, width) {
        REFUSED_VIEWS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        once!(
            format = desc.Format.0,
            first,
            count,
            stride,
            raw,
            width,
            "buffer view the host would refuse left null"
        );
        return None;
    }
    // The console has no component mapping; one the host does not know is the default's.
    if desc.Shader4ComponentMapping != D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING {
        desc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    }
    let host = host()?;
    // SAFETY: a live buffer and a slot of our heap.
    unsafe {
        host.device
            .CreateShaderResourceView(&shadow.buffer, Some(&desc), at)
    };
    if let Some(twin) = twin {
        let window = twin_window(desc.Format, first, count, stride, raw, width);
        if let Some((first, count, by)) = window {
            desc.Format = DXGI_FORMAT_R32_TYPELESS;
            desc.Anonymous.Buffer = D3D12_BUFFER_SRV {
                FirstElement: first,
                NumElements: count,
                StructureByteStride: 0,
                Flags: D3D12_BUFFER_SRV_FLAG_RAW,
            };
            // SAFETY: a live buffer and a slot of our heap.
            let start = unsafe {
                host.device
                    .CreateShaderResourceView(&shadow.buffer, Some(&desc), twin);
                shadow.buffer.GetGPUVirtualAddress() + first * 4
            };
            if by > 0 && !(staged(false, 1).is_some_and(|s| s.ptr == twin.ptr) && move_raw(twin, start, count.saturating_mul(4), by)) {
                TWINS[2].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                null_buffer_view(twin);
            }
        } else {
            null_buffer_view(twin);
        }
    }
    Some(Some(shadow.buffer.clone()))
}

/// An unordered access view of title buffer memory at `address` (`len`
/// bytes) on its host copy, as [`buffer_srv`]; false if the copy cannot be
/// written by shaders (upload memory).
/// shortcut: what shaders write stays in the host copy; title memory (and
/// so the CPU, and copies run on the CPU) does not see it.
fn buffer_uav(
    address: u64,
    len: u64,
    mut desc: D3D12_UNORDERED_ACCESS_VIEW_DESC,
    at: D3D12_CPU_DESCRIPTOR_HANDLE,
    twin: Option<D3D12_CPU_DESCRIPTOR_HANDLE>,
) -> Option<Kept> {
    let mut shadow = crate::shadow::find(address, len)?;
    let unit = {
        // SAFETY: a buffer view's union member.
        let b = unsafe { desc.Anonymous.Buffer };
        if b.StructureByteStride > 0 {
            u64::from(b.StructureByteStride)
        } else if b.Flags.0 & D3D12_BUFFER_UAV_FLAG_RAW.0 != 0 {
            4
        } else {
            crate::resource::element(desc.Format.0 as u32).map_or(0, |(bytes, _)| bytes)
        }
    };
    // SAFETY: a buffer view's union member.
    let own = unsafe { &mut desc.Anonymous.Buffer };
    let start = address + own.FirstElement * unit;
    let kind = (own.StructureByteStride, own.Flags.0 & D3D12_BUFFER_UAV_FLAG_RAW.0 != 0, desc.Format.0 as u32);
    let home = view_home(shadow.clone(), start, u64::from(own.NumElements) * unit, kind);
    let mut address = address;
    if !std::sync::Arc::ptr_eq(&home, &shadow) {
        // Its own bytes: it begins where they do.
        (shadow, address) = (home, start);
        own.FirstElement = 0;
    }
    // SAFETY: a live buffer.
    let flags = unsafe { shadow.buffer.GetDesc() }.Flags;
    if flags.0 & D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS.0 == 0 {
        once!("unordered access to a buffer whose host copy shaders cannot write");
        return None;
    }
    // SAFETY: a buffer view's union member.
    let buffer = unsafe { &mut desc.Anonymous.Buffer };
    let element = if buffer.StructureByteStride > 0 {
        u64::from(buffer.StructureByteStride)
    } else if buffer.Flags.0 & D3D12_BUFFER_UAV_FLAG_RAW.0 != 0 {
        4
    } else {
        crate::resource::element(desc.Format.0 as u32).map_or(0, |(bytes, _)| bytes)
    };
    let offset = shadow.offset(address);
    // A structured view as a raw view of the same bytes (see `raw_window`).
    let window = (desc.Format == DXGI_FORMAT_UNKNOWN)
        .then(|| raw_window(offset, buffer.FirstElement, buffer.NumElements, buffer.StructureByteStride))
        .flatten();
    if let Some((first, count)) = window {
        buffer.FirstElement = first;
        buffer.NumElements = count;
        buffer.StructureByteStride = 0;
        buffer.Flags = D3D12_BUFFER_UAV_FLAG_RAW;
    } else {
        if element == 0 || offset % element != 0 {
            once!(offset, element, "unordered buffer view not on an element of its host copy");
            return None;
        }
        buffer.FirstElement += offset / element;
    }
    buffer.CounterOffsetInBytes = 0;
    let (first, count, stride) = (buffer.FirstElement, buffer.NumElements, buffer.StructureByteStride);
    let raw = buffer.Flags.0 & D3D12_BUFFER_UAV_FLAG_RAW.0 != 0;
    if window.is_some() {
        // (Left out at first: every structured unordered access view was then
        // refused by `buffer_view_ok` and left null, runs 15 and 16.)
        desc.Format = DXGI_FORMAT_R32_TYPELESS;
    }
    // SAFETY: a live buffer.
    let width = unsafe { shadow.buffer.GetDesc() }.Width;
    if !buffer_view_ok(desc.Format, first, count, stride, raw, width) {
        REFUSED_VIEWS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        once!(
            format = desc.Format.0,
            first,
            count,
            stride,
            raw,
            width,
            "unordered buffer view the host would refuse left null"
        );
        return None;
    }
    let host = host()?;
    // SAFETY: a live buffer and a slot of our heap.
    unsafe {
        host.device
            .CreateUnorderedAccessView(&shadow.buffer, None::<&ID3D12Resource>, Some(&desc), at)
    };
    if let Some(twin) = twin {
        let window = twin_window(desc.Format, first, count, stride, raw, width);
        if let Some((first, count, by)) = window {
            desc.Format = DXGI_FORMAT_R32_TYPELESS;
            desc.Anonymous.Buffer = D3D12_BUFFER_UAV {
                FirstElement: first,
                NumElements: count,
                StructureByteStride: 0,
                CounterOffsetInBytes: 0,
                Flags: D3D12_BUFFER_UAV_FLAG_RAW,
            };
            // SAFETY: a live buffer and a slot of our heap.
            let start = unsafe {
                host.device
                    .CreateUnorderedAccessView(&shadow.buffer, None::<&ID3D12Resource>, Some(&desc), twin);
                shadow.buffer.GetGPUVirtualAddress() + first * 4
            };
            if by > 0 && !(staged(false, 1).is_some_and(|s| s.ptr == twin.ptr) && move_raw(twin, start, count.saturating_mul(4), by)) {
                TWINS[2].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                null_buffer_view(twin);
            }
        } else {
            null_buffer_view(twin);
        }
    }
    Some(Some(shadow.buffer.clone()))
}

/// Whether the host can make a shader (or unordered access) view of
/// `texture` in `format` (none: the texture's own). The console allows any
/// view; the host refuses (and removes the device for) views without the
/// resource's permission, of typed depth, and in another format than a
/// typed resource's. A view of a typed texture in another format of its
/// family takes the texture's format.
/// shortcut: such a view (sRGB of UNORM, say) reads without its
/// conversion; cast with castable formats if colors come out wrong.
/// Whether the host D3D12 device supports typed UAV stores in DXGI `format`
/// (`slim-shady110 517ba14`: prevents NVIDIA device removal on `B5G5R5A1_UNORM`
/// etc.). Cached per format id.
fn typed_store(format: u32) -> bool {
    use std::sync::atomic::{AtomicU8, Ordering::Relaxed};
    static KNOWN: [AtomicU8; 256] = [const { AtomicU8::new(0) }; 256];
    if let Some(slot) = KNOWN.get(format as usize) {
        match slot.load(Relaxed) {
            1 => return false,
            2 => return true,
            _ => {}
        }
    }
    let Some(host) = host() else {
        return true;
    };
    let mut data = D3D12_FEATURE_DATA_FORMAT_SUPPORT {
        Format: DXGI_FORMAT(format as i32),
        ..Default::default()
    };
    // SAFETY: plain format-support query on our live D3D12 device.
    let supported = unsafe {
        host.device.CheckFeatureSupport(
            D3D12_FEATURE_FORMAT_SUPPORT,
            (&raw mut data).cast(),
            size_of::<D3D12_FEATURE_DATA_FORMAT_SUPPORT>() as u32,
        )
    }
    .is_ok()
        && (data.Support2 & D3D12_FORMAT_SUPPORT2_UAV_TYPED_STORE) != D3D12_FORMAT_SUPPORT2_NONE;
    if let Some(slot) = KNOWN.get(format as usize) {
        slot.store(if supported { 2 } else { 1 }, Relaxed);
    }
    supported
}

fn is_typeless_format(format: u32) -> bool {
    format != 0
        && crate::resource::family(format) == format
        && format
            .checked_add(1)
            .is_some_and(|next| crate::resource::family(next) == format)
}

fn is_planar_yuv_format(format: DXGI_FORMAT) -> bool {
    matches!(
        format,
        DXGI_FORMAT_NV12 | DXGI_FORMAT_P010 | DXGI_FORMAT_P016 | DXGI_FORMAT_NV11
    )
}

fn planar_yuv_plane_matches(
    res_format: DXGI_FORMAT,
    view_format: DXGI_FORMAT,
    plane_slice: u32,
    unordered: bool,
) -> bool {
    match res_format {
        DXGI_FORMAT_NV12 | DXGI_FORMAT_NV11 => match plane_slice {
            0 => matches!(view_format, DXGI_FORMAT_R8_UNORM | DXGI_FORMAT_R8_UINT),
            1 => matches!(view_format, DXGI_FORMAT_R8G8_UNORM | DXGI_FORMAT_R8G8_UINT),
            _ => false,
        },
        DXGI_FORMAT_P010 | DXGI_FORMAT_P016 => match plane_slice {
            0 => matches!(view_format, DXGI_FORMAT_R16_UNORM | DXGI_FORMAT_R16_UINT),
            1 => {
                matches!(
                    view_format,
                    DXGI_FORMAT_R16G16_UNORM | DXGI_FORMAT_R16G16_UINT
                ) || (unordered && view_format == DXGI_FORMAT_R32_UINT)
            }
            _ => false,
        },
        _ => false,
    }
}

fn planar_yuv_view_compatible(
    res_format: DXGI_FORMAT,
    view_format: DXGI_FORMAT,
    unordered: bool,
) -> bool {
    planar_yuv_plane_matches(res_format, view_format, 0, unordered)
        || planar_yuv_plane_matches(res_format, view_format, 1, unordered)
}

fn fits(texture: &ID3D12Resource, mut format: Option<&mut DXGI_FORMAT>, unordered: bool) -> bool {
    use crate::resource::{DEPTH_FORMATS, family};
    // SAFETY: a live resource.
    let d = unsafe { texture.GetDesc() };
    let allowed = if unordered {
        d.Flags.0 & D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS.0 != 0
    } else {
        d.Flags.0 & D3D12_RESOURCE_FLAG_DENY_SHADER_RESOURCE.0 == 0
    };
    let own = d.Format.0 as u32;
    if !allowed || DEPTH_FORMATS.iter().any(|&(typed, _)| typed == own) {
        if unordered {
            crate::resource::DEPTH_UNORDERED[3].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
        once!(
            unordered,
            format = own,
            flags = d.Flags.0,
            "view the host does not allow"
        );
        return false;
    }
    // Every family has more than one format, its typeless one first.
    let typeless = is_typeless_format(own);
    let planar_yuv = is_planar_yuv_format(d.Format);
    if let Some(format) = format.as_deref_mut()
        && unordered
        && let Some(&(_, linear)) = crate::resource::SRGB
            .iter()
            .find(|&&(s, _)| s == format.0 as u32)
    {
        format.0 = linear as i32;
    }
    let view = format.as_ref().map_or(0, |f| f.0 as u32);
    // D3D12 forbids typeless and depth-only formats as SRV/UAV view formats.
    if is_typeless_format(view) || DEPTH_FORMATS.iter().any(|&(typed, _)| typed == view) {
        once!(view, own, unordered, "typeless or depth-only view format rejected");
        return false;
    }
    // A typed unordered view in a format the host cannot store to
    // (`B5G5R5A1_UNORM` on NVIDIA) removes the device when made: keep the slot
    // null instead (`slim-shady110 517ba14`).
    let typed = if view == 0
        || (!typeless && !planar_yuv)
        || (view != own && crate::resource::target_format(view) == own)
    {
        own
    } else {
        view
    };
    if unordered && !(view == 0 && (typeless || planar_yuv)) && !typed_store(typed) {
        once!(view = typed, own, "an unordered view in a format the host cannot store");
        return false;
    }
    if planar_yuv {
        let ok = view != 0 && planar_yuv_view_compatible(d.Format, DXGI_FORMAT(view as i32), unordered);
        if !ok {
            once!(view, own, unordered, "incompatible planar YUV view format");
        }
        return ok;
    }
    if view == 0 {
        return !typeless;
    }
    // A view of a target whose format the host refuses, in the title's
    // format: in the stand-in the host copy has.
    if view != own && crate::resource::target_format(view) == own {
        if let Some(format) = format {
            *format = d.Format;
        }
        return true;
    }
    if family(view) != family(own) {
        once!(view, own, "view format of another family");
        return false;
    }
    if let Some(format) = format
        && !typeless
    {
        *format = d.Format;
    }
    true
}

/// Queries the host device's D3D12 plane count for `format` (`0` if unsupported).
fn device_format_plane_count(device: &ID3D12Device, format: DXGI_FORMAT) -> u8 {
    let mut info = D3D12_FEATURE_DATA_FORMAT_INFO {
        Format: format,
        PlaneCount: 0,
    };
    // SAFETY: plain format-info query on a live D3D12 device.
    let ok = unsafe {
        device.CheckFeatureSupport(
            D3D12_FEATURE_FORMAT_INFO,
            (&raw mut info).cast(),
            size_of::<D3D12_FEATURE_DATA_FORMAT_INFO>() as u32,
        )
    }
    .is_ok();
    if ok { info.PlaneCount } else { 0 }
}

/// Fallback plane count when no live D3D12 device is available (unit tests).
fn default_format_plane_count(format: DXGI_FORMAT) -> u8 {
    match format {
        DXGI_FORMAT_UNKNOWN => 0,
        DXGI_FORMAT_R32G8X24_TYPELESS
        | DXGI_FORMAT_D32_FLOAT_S8X24_UINT
        | DXGI_FORMAT_R24G8_TYPELESS
        | DXGI_FORMAT_D24_UNORM_S8_UINT
        | DXGI_FORMAT_NV12
        | DXGI_FORMAT_P010
        | DXGI_FORMAT_P016
        | DXGI_FORMAT_NV11
        | DXGI_FORMAT_420_OPAQUE => 2,
        _ => 1,
    }
}

/// Cached plane count for `format` on the active host device.
fn host_format_plane_count(format: DXGI_FORMAT) -> u8 {
    use std::sync::atomic::{AtomicU8, Ordering::Relaxed};
    static KNOWN: [AtomicU8; 256] = [const { AtomicU8::new(0) }; 256];
    let idx = format.0 as usize;
    if let Some(slot) = KNOWN.get(idx) {
        let cached = slot.load(Relaxed);
        if cached != 0 {
            return cached - 1;
        }
    }
    let Some(host) = host() else {
        return default_format_plane_count(format);
    };
    let planes = device_format_plane_count(&host.device, format);
    if let Some(slot) = KNOWN.get(idx) {
        slot.store(planes.saturating_add(1), Relaxed);
    }
    planes
}

/// Queries whether `device` supports multisampled UAVs (`WriteableMSAATexturesSupported`).
fn device_writeable_msaa(device: &ID3D12Device) -> bool {
    let mut o14 = D3D12_FEATURE_DATA_D3D12_OPTIONS14::default();
    // SAFETY: plain feature-support query on a live D3D12 device.
    unsafe {
        device.CheckFeatureSupport(
            D3D12_FEATURE_D3D12_OPTIONS14,
            (&raw mut o14).cast(),
            size_of::<D3D12_FEATURE_DATA_D3D12_OPTIONS14>() as u32,
        )
    }
    .is_ok()
        && o14.WriteableMSAATexturesSupported.as_bool()
}

/// Cached `WriteableMSAATexturesSupported` capability for the active host device.
fn host_writeable_msaa() -> bool {
    use std::sync::atomic::{AtomicU8, Ordering::Relaxed};
    static KNOWN: AtomicU8 = AtomicU8::new(0);
    match KNOWN.load(Relaxed) {
        1 => return false,
        2 => return true,
        _ => {}
    }
    let Some(host) = host() else {
        return false;
    };
    let supported = device_writeable_msaa(&host.device);
    KNOWN.store(if supported { 2 } else { 1 }, Relaxed);
    supported
}

/// Validates `plane_slice` against both the resource's D3D12 plane count and
/// planar format/view-format pairing rules (depth/stencil and YUV planar formats).
fn plane_slice_valid(
    res_format: DXGI_FORMAT,
    view_format: DXGI_FORMAT,
    plane_count: u8,
    plane_slice: u32,
    unordered: bool,
) -> bool {
    if plane_count == 0 || plane_slice >= u32::from(plane_count) {
        return false;
    }
    let vf = if view_format == DXGI_FORMAT_UNKNOWN {
        res_format
    } else {
        view_format
    };
    if is_typeless_format(vf.0 as u32)
        || crate::resource::DEPTH_FORMATS
            .iter()
            .any(|&(typed, _)| typed == vf.0 as u32)
    {
        return false;
    }
    match res_format {
        DXGI_FORMAT_R32G8X24_TYPELESS | DXGI_FORMAT_D32_FLOAT_S8X24_UINT => {
            (plane_slice == 0 && vf == DXGI_FORMAT_R32_FLOAT_X8X24_TYPELESS)
                || (plane_slice == 1 && vf == DXGI_FORMAT_X32_TYPELESS_G8X24_UINT)
        }
        DXGI_FORMAT_R24G8_TYPELESS | DXGI_FORMAT_D24_UNORM_S8_UINT => {
            (plane_slice == 0 && vf == DXGI_FORMAT_R24_UNORM_X8_TYPELESS)
                || (plane_slice == 1 && vf == DXGI_FORMAT_X24_TYPELESS_G8_UINT)
        }
        DXGI_FORMAT_NV12 | DXGI_FORMAT_NV11 | DXGI_FORMAT_P010 | DXGI_FORMAT_P016 => {
            planar_yuv_plane_matches(res_format, vf, plane_slice, unordered)
        }
        DXGI_FORMAT_420_OPAQUE => false,
        _ => !matches!(
            vf,
            DXGI_FORMAT_R32_FLOAT_X8X24_TYPELESS
                | DXGI_FORMAT_X32_TYPELESS_G8X24_UINT
                | DXGI_FORMAT_R24_UNORM_X8_TYPELESS
                | DXGI_FORMAT_X24_TYPELESS_G8_UINT
        ),
    }
}

/// Rejects view formats that D3D12 never permits on SRV/UAV descriptors
/// (typeless resource formats, depth-only formats, or `DXGI_FORMAT_UNKNOWN`
/// on a typeless/planar-YUV resource).
fn view_format_typed_valid(own_format: DXGI_FORMAT, view_format: DXGI_FORMAT) -> bool {
    let eff = if view_format == DXGI_FORMAT_UNKNOWN {
        own_format
    } else {
        view_format
    };
    !is_typeless_format(eff.0 as u32)
        && !is_planar_yuv_format(eff)
        && eff != DXGI_FORMAT_420_OPAQUE
        && !crate::resource::DEPTH_FORMATS
            .iter()
            .any(|&(typed, _)| typed == eff.0 as u32)
}

/// Validates and clamps a texture SRV description against the host resource's
/// actual dimensions, sample count, mip count, array/depth extent, and planar
/// format rules so console descriptors (e.g. Xbox 4-bit max-mip `MipLevels = 15`
/// on a 7-mip texture) never trigger `RemoveDevice(DXGI_ERROR_INVALID_CALL)` on
/// PC D3D12.
fn clamp_srv_desc_for_resource(
    own: &D3D12_RESOURCE_DESC,
    desc: &mut D3D12_SHADER_RESOURCE_VIEW_DESC,
    plane_count: u8,
) -> bool {
    if !view_format_typed_valid(own.Format, desc.Format) {
        return false;
    }
    let own_mips = u32::from(own.MipLevels).max(1);
    let own_array = u32::from(own.DepthOrArraySize).max(1);
    let fix_mips = |most: &mut u32, mips: &mut u32| -> bool {
        if *most >= own_mips || *mips == 0 {
            return false;
        }
        let avail = own_mips - *most;
        if *mips != u32::MAX && *mips > avail {
            *mips = avail;
        }
        true
    };
    let fix_array = |first: &mut u32, size: &mut u32| -> bool {
        if *first >= own_array || *size == 0 {
            return false;
        }
        let avail = own_array - *first;
        if *size != u32::MAX && *size > avail {
            *size = avail;
        }
        true
    };
    // SAFETY: union arm matches `desc.ViewDimension`.
    unsafe {
        match desc.ViewDimension {
            D3D12_SRV_DIMENSION_TEXTURE1D => {
                if own.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE1D {
                    return false;
                }
                let mut t = desc.Anonymous.Texture1D;
                if !fix_mips(&mut t.MostDetailedMip, &mut t.MipLevels) {
                    return false;
                }
                desc.Anonymous.Texture1D = t;
                true
            }
            D3D12_SRV_DIMENSION_TEXTURE1DARRAY => {
                if own.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE1D {
                    return false;
                }
                let mut t = desc.Anonymous.Texture1DArray;
                if !fix_mips(&mut t.MostDetailedMip, &mut t.MipLevels)
                    || !fix_array(&mut t.FirstArraySlice, &mut t.ArraySize)
                {
                    return false;
                }
                desc.Anonymous.Texture1DArray = t;
                true
            }
            D3D12_SRV_DIMENSION_TEXTURE2D => {
                if own.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D || own.SampleDesc.Count > 1 {
                    return false;
                }
                let mut t = desc.Anonymous.Texture2D;
                if !fix_mips(&mut t.MostDetailedMip, &mut t.MipLevels)
                    || !plane_slice_valid(own.Format, desc.Format, plane_count, t.PlaneSlice, false)
                {
                    return false;
                }
                desc.Anonymous.Texture2D = t;
                true
            }
            D3D12_SRV_DIMENSION_TEXTURE2DARRAY => {
                if own.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D || own.SampleDesc.Count > 1 {
                    return false;
                }
                let mut t = desc.Anonymous.Texture2DArray;
                if !fix_mips(&mut t.MostDetailedMip, &mut t.MipLevels)
                    || !fix_array(&mut t.FirstArraySlice, &mut t.ArraySize)
                    || !plane_slice_valid(own.Format, desc.Format, plane_count, t.PlaneSlice, false)
                {
                    return false;
                }
                desc.Anonymous.Texture2DArray = t;
                true
            }
            D3D12_SRV_DIMENSION_TEXTURE2DMS => {
                own.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE2D && own.SampleDesc.Count >= 1
            }
            D3D12_SRV_DIMENSION_TEXTURE2DMSARRAY => {
                if own.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D || own.SampleDesc.Count == 0
                {
                    return false;
                }
                let mut t = desc.Anonymous.Texture2DMSArray;
                if !fix_array(&mut t.FirstArraySlice, &mut t.ArraySize) {
                    return false;
                }
                desc.Anonymous.Texture2DMSArray = t;
                true
            }
            D3D12_SRV_DIMENSION_TEXTURE3D => {
                if own.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE3D {
                    return false;
                }
                let mut t = desc.Anonymous.Texture3D;
                if !fix_mips(&mut t.MostDetailedMip, &mut t.MipLevels) {
                    return false;
                }
                desc.Anonymous.Texture3D = t;
                true
            }
            D3D12_SRV_DIMENSION_TEXTURECUBE => {
                if own.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D
                    || own.SampleDesc.Count > 1
                    || own_array < 6
                {
                    return false;
                }
                let mut t = desc.Anonymous.TextureCube;
                if !fix_mips(&mut t.MostDetailedMip, &mut t.MipLevels) {
                    return false;
                }
                desc.Anonymous.TextureCube = t;
                true
            }
            D3D12_SRV_DIMENSION_TEXTURECUBEARRAY => {
                if own.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D || own.SampleDesc.Count > 1 {
                    return false;
                }
                let mut t = desc.Anonymous.TextureCubeArray;
                if !fix_mips(&mut t.MostDetailedMip, &mut t.MipLevels)
                    || t.First2DArrayFace >= own_array
                    || t.NumCubes == 0
                {
                    return false;
                }
                let avail_cubes = (own_array - t.First2DArrayFace) / 6;
                if avail_cubes == 0 {
                    return false;
                }
                if t.NumCubes != u32::MAX && t.NumCubes > avail_cubes {
                    t.NumCubes = avail_cubes;
                }
                desc.Anonymous.TextureCubeArray = t;
                true
            }
            _ => false,
        }
    }
}

/// Validates and clamps a texture UAV description against the host resource's
/// actual dimensions, mip count, array/W-slice extent, planar format rules,
/// and MSAA UAV capability so out-of-range console views never trigger
/// `RemoveDevice(DXGI_ERROR_INVALID_CALL)`.
fn clamp_uav_desc_for_resource(
    own: &D3D12_RESOURCE_DESC,
    desc: &mut D3D12_UNORDERED_ACCESS_VIEW_DESC,
    plane_count: u8,
    writeable_msaa: bool,
) -> bool {
    if !view_format_typed_valid(own.Format, desc.Format) {
        return false;
    }
    let own_mips = u32::from(own.MipLevels).max(1);
    let own_array = u32::from(own.DepthOrArraySize).max(1);
    let fix_array = |first: &mut u32, size: &mut u32| -> bool {
        if *first >= own_array || *size == 0 {
            return false;
        }
        let avail = own_array - *first;
        if *size != u32::MAX && *size > avail {
            *size = avail;
        }
        true
    };
    // SAFETY: union arm matches `desc.ViewDimension`.
    unsafe {
        match desc.ViewDimension {
            D3D12_UAV_DIMENSION_TEXTURE1D => {
                own.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE1D
                    && desc.Anonymous.Texture1D.MipSlice < own_mips
            }
            D3D12_UAV_DIMENSION_TEXTURE1DARRAY => {
                if own.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE1D {
                    return false;
                }
                let mut t = desc.Anonymous.Texture1DArray;
                if t.MipSlice >= own_mips || !fix_array(&mut t.FirstArraySlice, &mut t.ArraySize) {
                    return false;
                }
                desc.Anonymous.Texture1DArray = t;
                true
            }
            D3D12_UAV_DIMENSION_TEXTURE2D => {
                let t = desc.Anonymous.Texture2D;
                own.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE2D
                    && own.SampleDesc.Count <= 1
                    && t.MipSlice < own_mips
                    && plane_slice_valid(own.Format, desc.Format, plane_count, t.PlaneSlice, true)
            }
            D3D12_UAV_DIMENSION_TEXTURE2DARRAY => {
                if own.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D || own.SampleDesc.Count > 1 {
                    return false;
                }
                let mut t = desc.Anonymous.Texture2DArray;
                if t.MipSlice >= own_mips
                    || !fix_array(&mut t.FirstArraySlice, &mut t.ArraySize)
                    || !plane_slice_valid(own.Format, desc.Format, plane_count, t.PlaneSlice, true)
                {
                    return false;
                }
                desc.Anonymous.Texture2DArray = t;
                true
            }
            D3D12_UAV_DIMENSION_TEXTURE2DMS => {
                writeable_msaa
                    && own.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE2D
                    && own.SampleDesc.Count >= 1
            }
            D3D12_UAV_DIMENSION_TEXTURE2DMSARRAY => {
                if !writeable_msaa
                    || own.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D
                    || own.SampleDesc.Count == 0
                {
                    return false;
                }
                let mut t = desc.Anonymous.Texture2DMSArray;
                if !fix_array(&mut t.FirstArraySlice, &mut t.ArraySize) {
                    return false;
                }
                desc.Anonymous.Texture2DMSArray = t;
                true
            }
            D3D12_UAV_DIMENSION_TEXTURE3D => {
                if own.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE3D {
                    return false;
                }
                let mut t = desc.Anonymous.Texture3D;
                if t.MipSlice >= own_mips {
                    return false;
                }
                let mip_depth = (own_array >> t.MipSlice.min(31)).max(1);
                if t.FirstWSlice >= mip_depth || t.WSize == 0 {
                    return false;
                }
                let avail = mip_depth - t.FirstWSlice;
                if t.WSize != u32::MAX && t.WSize > avail {
                    t.WSize = avail;
                }
                desc.Anonymous.Texture3D = t;
                true
            }
            _ => false,
        }
    }
}

fn clamp_srv_bounds(texture: &ID3D12Resource, desc: &mut D3D12_SHADER_RESOURCE_VIEW_DESC) -> bool {
    // SAFETY: a live texture resource.
    let own = unsafe { texture.GetDesc() };
    let plane_count = host_format_plane_count(own.Format);
    let ok = clamp_srv_desc_for_resource(&own, desc, plane_count);
    if !ok {
        once!(
            dim = desc.ViewDimension.0,
            res_dim = own.Dimension.0,
            res_fmt = own.Format.0,
            view_fmt = desc.Format.0,
            res_mips = own.MipLevels,
            res_array = own.DepthOrArraySize,
            "SRV texture view out of host resource bounds"
        );
    }
    ok
}

fn clamp_uav_bounds(
    texture: &ID3D12Resource,
    desc: &mut D3D12_UNORDERED_ACCESS_VIEW_DESC,
) -> bool {
    // SAFETY: a live texture resource.
    let own = unsafe { texture.GetDesc() };
    let plane_count = host_format_plane_count(own.Format);
    let writeable_msaa = host_writeable_msaa();
    let ok = clamp_uav_desc_for_resource(&own, desc, plane_count, writeable_msaa);
    if !ok {
        once!(
            dim = desc.ViewDimension.0,
            res_dim = own.Dimension.0,
            res_fmt = own.Format.0,
            view_fmt = desc.Format.0,
            res_mips = own.MipLevels,
            res_array = own.DepthOrArraySize,
            "UAV texture view out of host resource bounds"
        );
    }
    ok
}

/// The title's unordered access view of a depth texture, as a view of the
/// texture's colour twin `twin` ([`crate::resource::Twin`]): the same mip
/// and slices, in the colour format of the twin's family for a depth or
/// typeless format (R16_UNORM for 16-bit depth, R32_FLOAT for 32-bit; the
/// bits are the same). Without a description: mip 0 of every slice.
/// The twin of a stencil plane (R8) stands for the views of that plane
/// only, which become R8_UINT views of its one plane; `None` for a view of
/// the depth plane, which the twin does not stand for. The depth plane has
/// a twin of its own, viewed as a depth texture's without stencil: a view
/// in the format of the depth plane of a depth-stencil format (21, 46)
/// takes the twin's colour format too.
fn twin_uav(
    twin: &ID3D12Resource,
    desc: Option<&D3D12_UNORDERED_ACCESS_VIEW_DESC>,
) -> Option<D3D12_UNORDERED_ACCESS_VIEW_DESC> {
    // SAFETY: a live resource.
    let d = unsafe { twin.GetDesc() };
    let mut desc = desc.cloned().unwrap_or_else(|| {
        let mut all = D3D12_UNORDERED_ACCESS_VIEW_DESC {
            ViewDimension: D3D12_UAV_DIMENSION_TEXTURE2D,
            ..Default::default()
        };
        if d.DepthOrArraySize > 1 {
            all.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2DARRAY;
            all.Anonymous.Texture2DArray = D3D12_TEX2D_ARRAY_UAV {
                ArraySize: u32::from(d.DepthOrArraySize),
                ..Default::default()
            };
        }
        all
    });
    // A stencil plane's twin (R8 typeless, 60): the title's view is of the
    // stencil plane if it names plane 1 or a stencil view format (22, 47).
    if d.Format.0 == 60 {
        let mut plane = 0;
        match desc.ViewDimension {
            D3D12_UAV_DIMENSION_TEXTURE2D => {
                // SAFETY: the member the dimension names.
                let mut view = unsafe { desc.Anonymous.Texture2D };
                plane = std::mem::take(&mut view.PlaneSlice);
                desc.Anonymous.Texture2D = view;
            }
            D3D12_UAV_DIMENSION_TEXTURE2DARRAY => {
                // SAFETY: the member the dimension names.
                let mut view = unsafe { desc.Anonymous.Texture2DArray };
                plane = std::mem::take(&mut view.PlaneSlice);
                desc.Anonymous.Texture2DArray = view;
            }
            _ => {}
        }
        if plane != 1 && !matches!(desc.Format.0, 22 | 47) {
            return None;
        }
        desc.Format.0 = 62;
        return Some(desc);
    }
    // None, typeless and depth formats (R16: 0, 53, 55; R32: 0, 39, 40),
    // and those of a depth-stencil format's depth plane (19 to 21, 44 to
    // 46), for the twin of such a plane.
    desc.Format.0 = match (desc.Format.0, d.Format.0) {
        (0 | 53 | 55 | 19..=21 | 44..=46, 53) => 56,
        (0 | 39 | 40 | 19..=21 | 44..=46, 39) => 41,
        (other, _) => other,
    };
    Some(desc)
}

/// The bytes of the host's constant buffer view for a title's of `size`:
/// whole 256s, and at most the 64 KiB a shader can index. The host removes
/// the device for a larger view, and the title's heap holds such sizes in
/// slots it does not use.
fn constant_bytes(size: u32) -> u32 {
    const LARGEST: u32 = 64 << 10;
    size.min(LARGEST).next_multiple_of(256)
}

/// Renderer port 5 ([`crate::port_cbv`]): the constant buffer of `size` bytes
/// at `gpu` of `buffer` as a constant buffer view at `at` where it can be (on
/// 256 bytes; else a null one) and a raw view of its bytes at `twin`, which
/// translated shaders read. `false` (nothing made): not on 16 bytes.
fn constant_twinned(
    buffer: &ID3D12Resource,
    gpu: u64,
    size: u32,
    at: D3D12_CPU_DESCRIPTOR_HANDLE,
    twin: Option<D3D12_CPU_DESCRIPTOR_HANDLE>,
) -> bool {
    let Some(host) = host() else { return false };
    // SAFETY: a live buffer.
    let (base, width) = unsafe { (buffer.GetGPUVirtualAddress(), buffer.GetDesc().Width) };
    let Some(offset) = gpu.checked_sub(base).filter(|&o| o < width && crate::port_cbv::raw_ok(o)) else {
        return false;
    };
    let room = width - offset;
    let cbv = if gpu.is_multiple_of(256) && room >= 256 {
        D3D12_CONSTANT_BUFFER_VIEW_DESC {
            BufferLocation: gpu,
            SizeInBytes: constant_bytes(size).min((room & !255) as u32),
        }
    } else {
        D3D12_CONSTANT_BUFFER_VIEW_DESC::default()
    };
    let bytes = u64::from(constant_bytes(size)).min(room) & !3;
    // SAFETY: slots of our heaps; the ranges lie in the live buffer.
    unsafe {
        host.device.CreateConstantBufferView(Some(&cbv), at);
        if let Some(twin) = twin {
            let raw = D3D12_SHADER_RESOURCE_VIEW_DESC {
                Format: DXGI_FORMAT_R32_TYPELESS,
                ViewDimension: D3D12_SRV_DIMENSION_BUFFER,
                Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
                Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                    Buffer: D3D12_BUFFER_SRV {
                        FirstElement: offset / 4,
                        NumElements: (bytes / 4) as u32,
                        StructureByteStride: 0,
                        Flags: D3D12_BUFFER_SRV_FLAG_RAW,
                    },
                },
            };
            if bytes == 0 {
                null_buffer_view(twin);
            } else {
                host.device.CreateShaderResourceView(buffer, Some(&raw), twin);
            }
            crate::port_cbv::twinned();
        }
    }
    true
}

/// Native RTAS descriptors have no pResource. The address is carried exclusively
/// by the SRV union; keep the resource (and therefore its BLAS pins) in the slot.
pub(crate) fn tlas_srv(
    dev: &ID3D12Device,
    target: &ID3D12Resource,
    at: D3D12_CPU_DESCRIPTOR_HANDLE,
) {
    let desc = D3D12_SHADER_RESOURCE_VIEW_DESC {
        Format: DXGI_FORMAT_UNKNOWN,
        ViewDimension: D3D12_SRV_DIMENSION_RAYTRACING_ACCELERATION_STRUCTURE,
        Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
        Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
            RaytracingAccelerationStructure: D3D12_RAYTRACING_ACCELERATION_STRUCTURE_SRV {
                Location: unsafe { target.GetGPUVirtualAddress() },
            },
        },
    };
    unsafe { dev.CreateShaderResourceView(None::<&ID3D12Resource>, Some(&desc), at) };
}

fn native_tlas(view: &View) -> Option<u64> {
    match view {
        View::Shader(0, Some(d))
            if d[1] == D3D12_SRV_DIMENSION_RAYTRACING_ACCELERATION_STRUCTURE.0 as u32
                && crate::dxr::requested() =>
        {
            Some(u64::from(d[4]) | u64::from(d[5]) << 32)
        }
        _ => None,
    }
}

fn empty_tlas_srv(
    host: &crate::host::Host,
    at: D3D12_CPU_DESCRIPTOR_HANDLE,
) -> Option<ID3D12Resource> {
    let empty = host.dxr.as_ref()?.srv(0)?;
    tlas_srv(&host.device, &empty, at);
    #[cfg(feature = "diagnostics")]
    crate::replay::record::dxr_slot(at.ptr, 0);
    Some(empty)
}

/// Makes the host descriptor for `view` at `at`: the host resource it
/// views, or `None` if it cannot.
fn make(view: &View, at: D3D12_CPU_DESCRIPTOR_HANDLE) -> Option<Kept> {
    make_twinned(view, at, None)
}

/// [`make`]; `twin`: where the raw twin of a buffer view goes
/// ([`raw_twins`]; a constant buffer's is the same view again, other views
/// have none).
fn make_twinned(
    view: &View,
    at: D3D12_CPU_DESCRIPTOR_HANDLE,
    twin: Option<D3D12_CPU_DESCRIPTOR_HANDLE>,
) -> Option<Kept> {
    make_twinned_with_placed(view, None, at, twin)
}

fn make_twinned_with_placed(
    view: &View,
    resolved_placed: Option<Option<&PlacedTexture>>,
    at: D3D12_CPU_DESCRIPTOR_HANDLE,
    twin: Option<D3D12_CPU_DESCRIPTOR_HANDLE>,
) -> Option<Kept> {
    let host = host()?;
    let device = &host.device;
    if let Some(key) = native_tlas(view) {
        let target = if key != 0
            && let Some(target) = host.dxr.as_ref()?.srv(key)
        {
            tlas_srv(device, &target, at);
            target
        } else {
            if key != 0 {
                crate::dxr::note_empty_tlas_fallback(key);
            }
            empty_tlas_srv(host, at)?
        };
        #[cfg(feature = "diagnostics")]
        crate::replay::record::dxr_slot(at.ptr, key);
        return Some(Some(target));
    }
    match view {
        &View::Constant(address, size) => {
            let shadow = crate::shadow::find(address, u64::from(size))?;
            let gpu = shadow.gpu(address);
            if crate::port_cbv::on() && twin.is_some() {
                // Renderer port 5: shaders read constant buffers as raw bytes
                // through the twin; on 1M-capped NVIDIA heaps (`twin == None`)
                // shaders read regular CBVs at `at`, so never write null CBVs
                // there (`Bebrik229 79680c0`).
                return constant_twinned(&shadow.buffer, gpu, size, at, twin).then(|| Some(shadow.buffer.clone()));
            }
            if gpu % 256 != 0 {
                once!(
                    address = format_args!("{address:#x}"),
                    "constant buffer not 256-byte aligned on the host"
                );
                return None;
            }
            let desc = D3D12_CONSTANT_BUFFER_VIEW_DESC {
                BufferLocation: gpu,
                SizeInBytes: constant_bytes(size),
            };
            // SAFETY: slots of our heap; the range lies in the host copy.
            unsafe {
                device.CreateConstantBufferView(Some(&desc), at);
                if let Some(twin) = twin {
                    device.CreateConstantBufferView(Some(&desc), twin);
                }
            }
            Some(Some(shadow.buffer.clone()))
        }
        View::Shader(resource, desc) => {
            let resource = *resource as crate::object::This;
            if let Some(copy) = crate::resource::host_copy(resource) {
                let texture = &copy.texture;
                // The title's views of textures with a stencil plane, as it
                // describes them (format, dimension, component mapping, a
                // pad, then first level, levels, plane): each distinct one
                // once. They say how its shaders read the stencil plane
                // (the lighting takes each pixel's material class and its
                // "inside a room" bit from it).
                #[cfg(feature = "diagnostics")]
                {
                    // SAFETY: a live resource.
                    let own = unsafe { texture.GetDesc() }.Format.0;
                    if matches!(own, 19..=22 | 44..=47) {
                        static SEEN: Mutex<Vec<(i32, Option<[u32; 10]>)>> = Mutex::new(Vec::new());
                        let mut seen = SEEN.lock().unwrap_or_else(|e| e.into_inner());
                        if seen.len() < 64 && !seen.contains(&(own, *desc)) {
                            seen.push((own, *desc));
                            tracing::info!(
                                ?resource,
                                texture_format = own,
                                view = ?desc,
                                "shader view of a depth-stencil texture"
                            );
                        }
                    }
                }
                let mut desc = desc.as_ref().map(srv_desc);
                if !fits(texture, desc.as_mut().map(|d| &mut d.Format), false) {
                    return None;
                }
                // Levels on demand (`crate::tiles`): a texture made level by
                // level is shown from the first level that is the title's
                // and has memory, if the title's view starts before it (a
                // level without memory reads zeros: a black texture). The
                // title's own view mostly starts there already.
                if let Some(first) = crate::resource::first_shown(resource)
                    && first > 0
                {
                    // SAFETY: a live resource.
                    let own = unsafe { texture.GetDesc() };
                    // The texture's own view (the title gave none): all of it.
                    let whole = D3D12_SHADER_RESOURCE_VIEW_DESC {
                        Format: own.Format,
                        ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2D,
                        Shader4ComponentMapping: 5768,
                        Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                            Texture2D: D3D12_TEX2D_SRV {
                                MostDetailedMip: 0,
                                MipLevels: u32::MAX,
                                PlaneSlice: 0,
                                ResourceMinLODClamp: 0.0,
                            },
                        },
                    };
                    crate::tiles::show_from(desc.get_or_insert(whole), first, u32::from(own.MipLevels));
                }
                if copy.base > 0
                    && let Some(desc) = &mut desc
                {
                    from_base(desc, copy.base);
                }
                // Made whole from the store: all its levels.
                let prefilled = copy.prefilled.load(std::sync::atomic::Ordering::Relaxed);
                if prefilled > 0
                    && let Some(desc) = &mut desc
                {
                    show_prefilled(desc, prefilled);
                }
                if let Some(desc) = &mut desc
                    && !clamp_srv_bounds(texture, desc)
                {
                    return None;
                }
                // SAFETY: a live texture and a slot of our heap.
                unsafe {
                    device.CreateShaderResourceView(
                        texture,
                        desc.as_ref().map(std::ptr::from_ref),
                        at,
                    )
                };
                Some(Some(texture.clone()))
            } else if let (Some((data, len)), Some(desc)) =
                (crate::resource::buffer_bytes(resource), desc)
            {
                buffer_srv(data as u64, len, srv_desc(desc), at, twin)
            } else {
                None
            }
        }
        View::Unordered(resource, _counter, desc) => {
            let resource = *resource as crate::object::This;
            let Some(texture) = crate::resource::host_resource(resource) else {
                return match (crate::resource::buffer_bytes(resource), desc) {
                    (Some((data, len)), Some(desc)) => {
                        buffer_uav(data as u64, len, uav_desc(desc), at, twin)
                    }
                    _ => None,
                };
            };
            // The first such views of depth textures are logged as the
            // title describes them (format, dimension, then mip, slices and
            // plane): they say which planes shaders write, and as what.
            if crate::resource::depth_and_unordered(resource) {
                static LOGGED: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
                if LOGGED.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 512 {
                    tracing::info!(
                        ?resource,
                        view = ?desc,
                        "unordered access view of a depth texture"
                    );
                }
            }
            let mut desc = desc.as_ref().map(uav_desc);
            // A depth texture's unordered access is its colour twin's: the
            // host gives a depth texture none.
            let twin = crate::resource::twin_resource(resource)
                .and_then(|twin| Some((twin, twin_uav(twin, desc.as_ref())?)))
                // A view of the depth plane of a texture with a stencil
                // plane: of that plane's own twin.
                .or_else(|| {
                    let twin = crate::resource::twin_plane_resource(resource)?;
                    Some((twin, twin_uav(twin, desc.as_ref())?))
                });
            let texture = match twin {
                Some((twin, view)) => {
                    desc = Some(view);
                    twin
                }
                None => texture,
            };
            if !fits(texture, desc.as_mut().map(|d| &mut d.Format), true) {
                return None;
            }
            if let Some(desc) = &mut desc
                && !clamp_uav_bounds(texture, desc)
            {
                return None;
            }
            // SAFETY: a live texture and a slot of our heap.
            unsafe {
                device.CreateUnorderedAccessView(
                    texture,
                    None::<&ID3D12Resource>,
                    desc.as_ref().map(std::ptr::from_ref),
                    at,
                )
            };
            Some(Some(texture.clone()))
        }
        View::Sampler(words) => {
            // SAFETY: 13 words in the host's layout.
            let desc = unsafe {
                std::ptr::from_ref(words)
                    .cast::<D3D12_SAMPLER_DESC>()
                    .read_unaligned()
            };
            // SAFETY: a slot of our sampler heap.
            unsafe { device.CreateSampler(&desc, at) };
            Some(None)
        }
        View::ConsoleSampler(words) => {
            let desc = crate::sampler::console(words);
            // SAFETY: a slot of our sampler heap.
            unsafe { device.CreateSampler(&desc, at) };
            Some(None)
        }
        View::Placed(unordered, resource, desc) => {
            let owned_placed;
            let resolved = match resolved_placed {
                Some(placed) => placed,
                None => {
                    owned_placed = placed_texture(view);
                    owned_placed.as_ref()
                }
            };
            if let Some(placed) = resolved {
                let made = make(&placed.view, at);
                #[cfg(feature = "diagnostics")]
                if *unordered && made.as_ref().is_some_and(Option::is_some) {
                    static FIRST: std::sync::Once = std::sync::Once::new();
                    FIRST.call_once(|| {
                        tracing::info!(
                            dimension = resource[0],
                            resource_format = resource[8],
                            view_format = desc[0],
                            flags = resource[12],
                            address = format_args!(
                                "{:#x}",
                                u64::from(desc[10]) | u64::from(desc[11]) << 32
                            ),
                            "mirrored a placed texture UAV"
                        )
                    });
                }
                return made;
            }
            const BUFFER: u32 = 1;
            if resource[0] != BUFFER {
                static SAID: Mutex<std::collections::BTreeSet<u64>> = Mutex::new(std::collections::BTreeSet::new());
                static TIMES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
                let times = TIMES.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
                let at = u64::from(desc[10]) | u64::from(desc[11]) << 32;
                let key = {
                    use std::hash::{Hash, Hasher};
                    let mut h = std::collections::hash_map::DefaultHasher::new();
                    resource.hash(&mut h);
                    desc.hash(&mut h);
                    h.finish()
                };
                let mut said = SAID.lock().unwrap_or_else(|e| e.into_inner());
                if (said.len() < 512 && said.insert(key)) || times.is_power_of_two() {
                    #[cfg(all(windows, feature = "diagnostics"))]
                    let (keys, addresses) = crate::resource::placed_textures_like(resource);
                    #[cfg(not(all(windows, feature = "diagnostics")))]
                    let (keys, addresses): (usize, Vec<u64>) = (0, Vec::new());
                    tracing::warn!(
                        address = format_args!("{at:#x}"),
                        times,
                        "placed unordered texture view not mirrored (null: writes through it are lost): unordered {unordered}, resource {resource:?}, desc {desc:?}, matches {keys}, addresses {addresses:?}"
                    );
                }
                return None;
            }
            // The resource description's width, after the alignment.
            let len = u64::from(resource[4]) | u64::from(resource[5]) << 32;
            let address = u64::from(desc[10]) | u64::from(desc[11]) << 32;
            let words: &[u32; 10] = desc[..10].try_into().unwrap_or(&[0; 10]);
            // An empty title buffer: nothing to view, and no host copy to
            // look for (#70: such views were in flight at device removals).
            if len == 0 {
                return None;
            }
            if *unordered {
                buffer_uav(address, len, uav_desc(words), at, twin)
            } else {
                buffer_srv(address, len, srv_desc(words), at, twin)
            }
        }
    }
}

/// The view `desc` of a host copy whose levels below `levels` the store
/// filled (`crate::texstore`): from level 0, if it showed from one of them.
fn show_prefilled(desc: &mut D3D12_SHADER_RESOURCE_VIEW_DESC, levels: u32) {
    if desc.ViewDimension != D3D12_SRV_DIMENSION_TEXTURE2D {
        return;
    }
    // SAFETY: the member its dimension says.
    let t = unsafe { &mut desc.Anonymous.Texture2D };
    if t.MostDetailedMip == 0 || t.MostDetailedMip > levels {
        return;
    }
    if t.MipLevels != u32::MAX {
        t.MipLevels += t.MostDetailedMip;
    }
    t.MostDetailedMip = 0;
    t.ResourceMinLODClamp = 0.0;
}

/// The view `desc` of a texture, of a host copy of its levels from `base`
/// on (`crate::resource::held_only`). What it shows of larger ones, if any,
/// is left out.
fn from_base(desc: &mut D3D12_SHADER_RESOURCE_VIEW_DESC, base: u32) {
    if desc.ViewDimension != D3D12_SRV_DIMENSION_TEXTURE2D {
        return;
    }
    // SAFETY: the member its dimension says.
    let t = unsafe { &mut desc.Anonymous.Texture2D };
    let lost = base.saturating_sub(t.MostDetailedMip);
    t.MostDetailedMip = t.MostDetailedMip.saturating_sub(base);
    if t.MipLevels != u32::MAX {
        t.MipLevels = t.MipLevels.saturating_sub(lost).max(1);
    }
    t.ResourceMinLODClamp = (t.ResourceMinLODClamp - base as f32).max(0.0);
}

/// Writes of a slot with another view of the placed texture it showed,
/// since the start: the slot kept showing the old view.
pub(crate) static KEPT_VIEWS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Whether a slot written with another view of the placed texture it
/// shows keeps showing the old one until the new is made;
/// `VICTORY_KEEP_VIEWS=0` makes it null at once, as before.
fn keep_views() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_KEEP_VIEWS").is_none_or(|v| v != "0"))
}

/// What the texture view a slot is written with has to do with the one it
/// shows.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Kin {
    /// A view of the same placed texture ([`same_placed_texture`]).
    Same,
    /// A texture of the same description placed elsewhere (moved, or one of
    /// its kind).
    Moved,
    /// One of the same format and shape in another size: the same texture
    /// with more or fewer levels, as a title streams it.
    Sized,
    /// A view of the same dimension, of another texture.
    Other,
    /// A view no shader that reads the one can read.
    None,
}

impl Kin {
    /// Whether the slot goes on showing what it shows, by [`keep_shown`].
    fn kept(self, keep: u8) -> bool {
        match self {
            Kin::Same => true,
            Kin::Moved | Kin::Sized => keep >= 1,
            Kin::Other => keep >= 2,
            Kin::None => false,
        }
    }
}

/// [`Kin`] of the view `old`, which a slot shows, and `new`.
fn kin(old: &View, new: &View) -> Kin {
    if same_placed_texture(old, new) {
        return Kin::Same;
    }
    if let (View::Placed(false, a, from), View::Placed(false, b, to)) = (old, new) {
        if a[0] == 1 || b[0] == 1 || from[1] != to[1] {
            return Kin::None;
        }
        // (Width, 64 bits, height, depth and levels, format: words 4 to 8.)
        if a[0] == b[0] && a[2..13] == b[2..13] {
            return Kin::Moved;
        }
        let (wide, high) = ([a[4], b[4]].map(u64::from), [a[6], b[6]].map(u64::from));
        return if from[0] == to[0] && a[0] == b[0] && wide[0] * high[1] == wide[1] * high[0] {
            Kin::Sized
        } else {
            Kin::Other
        };
    }
    let dimension = |view: &View| match view {
        View::Placed(false, resource, desc) if resource[0] != 1 => desc[1],
        View::Shader(r, Some(desc)) if crate::resource::buffer_bytes(*r as crate::object::This).is_none() => desc[1],
        _ => 0,
    };
    match (dimension(old), dimension(new)) {
        (0, _) | (_, 0) => Kin::None,
        (a, b) if a == b => Kin::Other,
        _ => Kin::None,
    }
}

/// Whether a slot that is written with a view of another texture goes on
/// showing the one it shows until the new one is made and its upload has run
/// (`VICTORY_KEEP_SHOWN`): 0: never, it reads as zeros, then as the neutral
/// texture; 1: one of the same kind in another place or size; 2: any of the
/// same dimension.
fn keep_shown() -> u8 {
    #[cfg(feature = "diagnostics")]
    if let Some(on) = crate::ab::pick("VICTORY_KEEP_SHOWN") {
        return if on { 2 } else { 0 };
    }
    static KEEP: std::sync::OnceLock<u8> = std::sync::OnceLock::new();
    *KEEP.get_or_init(|| std::env::var("VICTORY_KEEP_SHOWN").ok().and_then(|v| v.trim().parse().ok()).unwrap_or(1))
}

/// Slots that showed a texture and were written with a view of another,
/// since the start, by [`Kin`] (moved, sized, other, none), and those of
/// them that kept what they showed.
static REWRITTEN: [std::sync::atomic::AtomicU64; 5] = [const { std::sync::atomic::AtomicU64::new(0) }; 5];

fn rewritten(kin: Kin, kept: bool) {
    use std::sync::atomic::Ordering::Relaxed;
    let class = match kin {
        Kin::Moved | Kin::Same => 0,
        Kin::Sized => 1,
        Kin::Other => 2,
        Kin::None => 3,
    };
    REWRITTEN[class].fetch_add(1, Relaxed);
    REWRITTEN[4].fetch_add(u64::from(kept), Relaxed);
    let n: u64 = REWRITTEN[..4].iter().map(|n| n.load(Relaxed)).sum();
    if n.is_multiple_of(5000) {
        tracing::info!(
            "slots that showed a texture and were written with another, since the start: {} of the same kind elsewhere, {} of another size, {} others, {} of another dimension; {} of them kept what they showed meanwhile (VICTORY_KEEP_SHOWN)",
            REWRITTEN[0].load(Relaxed),
            REWRITTEN[1].load(Relaxed),
            REWRITTEN[2].load(Relaxed),
            REWRITTEN[3].load(Relaxed),
            REWRITTEN[4].load(Relaxed)
        );
    }
}

/// Diagnostics: whether a texture that a slot showed until its new one came
/// holds the new one's bytes (the title moved it), by the bytes their host
/// copies were filled from: equal, different, not known.
#[cfg(feature = "diagnostics")]
fn same_bytes(old: u64, new: u64) {
    use std::sync::atomic::Ordering::Relaxed;
    static SEEN: [std::sync::atomic::AtomicU64; 6] = [const { std::sync::atomic::AtomicU64::new(0) }; 6];
    let view = |key: u64| view_at(key as u32, (key >> 32) as u32);
    let (Some(old), Some(new)) = (view(old), view(new)) else {
        return;
    };
    let held = |view: &View| {
        let View::Placed(false, resource, desc) = view else {
            return None;
        };
        let address = u64::from(desc[10]) | u64::from(desc[11]) << 32;
        crate::resource::texture_held(crate::resource::texture_at_made(address, resource)?)
    };
    let class = match kin(&old, &new) {
        Kin::Moved => 0,
        Kin::Same | Kin::None => return,
        _ => 3,
    };
    let which = match (held(&old), held(&new)) {
        (Some(a), Some(b)) if a == b => 0,
        (Some(a), Some(b)) if a.0 == b.0 => 1,
        _ => 2,
    };
    SEEN[class + which].fetch_add(1, Relaxed);
    let n: u64 = SEEN.iter().map(|n| n.load(Relaxed)).sum();
    if n.is_multiple_of(1000) {
        let seen: Vec<u64> = SEEN.iter().map(|n| n.load(Relaxed)).collect();
        tracing::info!(
            "slots that went on showing another texture until the new one came, since the start: of one of the same kind elsewhere {} had the new one's bytes, {} others, {} not known; of others {}, {}, {}",
            seen[0], seen[1], seen[2], seen[3], seen[4], seen[5]
        );
    }
}

/// For `crate::popwatch`, since the start: slots made null as they were
/// written with another texture, such that kept the old one, slots given the
/// neutral texture, slots made null as their memory was another's.
#[cfg(feature = "diagnostics")]
pub(crate) fn pop_causes() -> [u64; 4] {
    use std::sync::atomic::Ordering::Relaxed;
    let all: u64 = REWRITTEN[..4].iter().map(|n| n.load(Relaxed)).sum();
    let kept = REWRITTEN[4].load(Relaxed);
    [all.saturating_sub(kept), kept, LATE[3].load(Relaxed), STALE[0].load(Relaxed)]
}

/// Whether a slot already made from its record is told so without the lock
/// (`VICTORY_QUICK_MIRROR=0`: with it, as before).
fn quick_mirror() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_QUICK_MIRROR").is_none_or(|v| v != "0"))
}

/// The records last seen of a heap, held ([`HostHeap::copied_begin`]).
pub(crate) struct Copied<'a> {
    at: *mut u8,
    len: usize,
    step: usize,
    _held: std::sync::MutexGuard<'a, Vec<u8>>,
}

// SAFETY: the bytes are reached through `changes` alone, whose callers take
// parts apart from one another; the guard stays with the thread that has it.
unsafe impl Sync for Copied<'_> {}

impl Copied<'_> {
    /// The slots of `records` (as the title has them now, from slot `first`
    /// on) that are not the ones last seen, which they are from here on.
    /// (`records`: as the caller read them, through the kernel, which fails on
    /// a page the title unmapped meanwhile where a read in place would fault:
    /// the title unmaps heap pages too.)
    ///
    /// # Safety
    /// No other call beside this one has any of these slots.
    pub(crate) unsafe fn changes(&self, first: u32, records: &[u8], out: &mut Vec<u32>) {
        let at = first as usize * self.step;
        let end = (at + records.len()).min(self.len);
        if at >= end {
            return;
        }
        // SAFETY: within the records held; the caller's slots alone.
        let last = unsafe { std::slice::from_raw_parts_mut(self.at.add(at), end - at) };
        changed_records(last, &records[..end - at], self.step, first, out);
    }

    /// As [`Copied::changes`] for the `count` records of title memory at
    /// `title` (slot `first` on), 32 bytes each, read where they are, each
    /// once ([`crate::safecopy::changed`]); `bits` is room. False when a page
    /// of them could not be read: those before it are done, the others
    /// stay as last seen.
    ///
    /// # Safety
    /// As [`Copied::changes`]; `title` is title memory (read by a copy that
    /// fails where a page is not there).
    pub(crate) unsafe fn changes_in_place(&self, first: u32, title: *const u8, count: usize, bits: &mut Vec<u64>, out: &mut Vec<u32>) -> bool {
        let at = first as usize * self.step;
        let count = count.min(self.len.saturating_sub(at) / 32);
        if self.step != 32 || count == 0 {
            return true;
        }
        bits.clear();
        bits.resize(count.div_ceil(64), 0);
        // SAFETY: within the records held; the caller's slots alone.
        let whole = unsafe { crate::safecopy::changed(title, self.at.add(at), count, bits) };
        for (n, &word) in bits.iter().enumerate() {
            let mut word = word;
            while word != 0 {
                out.push(first + (n * 64) as u32 + word.trailing_zeros());
                word &= word - 1;
            }
        }
        whole
    }

    /// The `count` records last seen from slot `first` on (fewer where the
    /// heap ends).
    ///
    /// # Safety
    /// As [`Copied::changes`]: no other call beside this one has any of
    /// these slots.
    pub(crate) unsafe fn seen(&self, first: u32, count: usize) -> &[u8] {
        let at = (first as usize * self.step).min(self.len);
        let len = (count * self.step).min(self.len - at);
        // SAFETY: within the records held; the caller's slots alone.
        unsafe { std::slice::from_raw_parts(self.at.add(at), len) }
    }
}

/// The records (`step` bytes each, the first is slot `first`) that differ
/// between `last` and `records` (of one length) go into `out` and `last`.
/// Most of a changed page is as it was: it is looked at sixteen records at a
/// time.
fn changed_records(last: &mut [u8], records: &[u8], step: usize, first: u32, out: &mut Vec<u32>) {
    const AT_A_TIME: usize = 16;
    let block = AT_A_TIME * step;
    for (b, (old, new)) in last.chunks_mut(block).zip(records.chunks(block)).enumerate() {
        if old == new {
            continue;
        }
        for (n, (old, new)) in old.chunks_exact_mut(step).zip(new.chunks_exact(step)).enumerate() {
            if old != new {
                old.copy_from_slice(new);
                out.push(first + (b * AT_A_TIME + n) as u32);
            }
        }
    }
}

#[cfg(test)]
mod changed_records_tests {
    use super::changed_records;

    #[test]
    fn changed_records_are_those_that_differ_one_by_one() {
        for (step, count) in [(32usize, 128usize), (32, 2048), (16, 77), (32, 5), (32, 17)] {
            let mut seed = 7u64;
            let mut random = move || {
                seed = seed.wrapping_mul(0x5851_F42D_4C95_7F2D).wrapping_add(0x1405_7B7E_F767_814F);
                (seed >> 33) as usize
            };
            let mut last: Vec<u8> = (0..step * count).map(|_| random() as u8).collect();
            let mut now = last.clone();
            let mut differ = Vec::new();
            for slot in 0..count {
                if random() % 9 == 0 {
                    now[slot * step + random() % step] ^= 1 + (random() % 255) as u8;
                    differ.push(1000 + slot as u32);
                }
            }
            let mut out = Vec::new();
            changed_records(&mut last, &now, step, 1000, &mut out);
            assert_eq!(out, differ);
            assert_eq!(last, now);
            out.clear();
            changed_records(&mut last, &now, step, 1000, &mut out);
            assert!(out.is_empty());
        }
    }
}

/// Whether both are shader views of one texture the title placed in
/// memory: the same address and the same resource description, whatever
/// part of it each shows.
fn same_placed_texture(a: &View, b: &View) -> bool {
    match (a, b) {
        (View::Placed(false, resource, desc), View::Placed(false, other, other_desc)) => {
            // (Words 1 and 13 are the filler of the description's layout, not
            // part of what a texture is: `crate::resource::texture_at`.)
            let same = |from: usize, to: usize| resource[from..to] == other[from..to];
            resource[0] != 1 && resource[0] == other[0] && same(2, 13) && desc[10..12] == other_desc[10..12]
        }
        _ => false,
    }
}

/// Whether `view` is a placed texture whose backing resource can change
/// dynamically at the same guest address (`resource[12] & 4 != 0`: UAV/writable
/// target lookup via `TARGETS_AT`).
fn is_dynamic_placed_texture(view: &View) -> bool {
    matches!(view, View::Placed(_, resource, _) if resource[0] != 1 && resource[12] & 4 != 0)
}

/// Reads the guest descriptor slot record once, returning `(record_key, View)`
/// so callers deferring a mirror can verify under the slot lock that the guest
/// slot still holds the exact same record.
pub(crate) fn view_with_key(handle: u64) -> Option<(u64, View)> {
    // SAFETY: the handle is a descriptor slot in title memory.
    let [tag, i, generation] = unsafe { (handle as *const [u32; 3]).read_unaligned() };
    if tag != VIEW_TAG {
        return None;
    }
    let key = u64::from(generation) << 32 | u64::from(i);
    view_at(i, generation).map(|view| (key, view))
}

/// A placed texture view's host view, and the title resource it holds (when
/// the memory belongs to one; `avatarmanz #97`), along with the
/// [`crate::resource::all_target_changes`] generation observed right before
/// resolution so concurrent owner publication never misses a target change.
pub(crate) struct PlacedTexture {
    pub(crate) view: View,
    _owner: Option<crate::resource::ResourceRef>,
    resolved_gen: u64,
}

impl PlacedTexture {
    /// Monotonic [`crate::resource::resource_id`] of the resolved backing
    /// texture resource.
    pub(crate) fn owner_id(&self) -> u64 {
        match self.view {
            View::Shader(r, _) | View::Unordered(r, ..) => {
                crate::resource::resource_id(r as crate::object::This)
            }
            _ => 0,
        }
    }

    fn host_raw_if_made(&self) -> usize {
        use windows::core::Interface;
        match self.view {
            View::Shader(r, _) | View::Unordered(r, ..) => {
                crate::resource::made_copy(r as crate::object::This)
                    .map_or(0, |t| Interface::as_raw(&t.texture) as usize)
            }
            _ => 0,
        }
    }
}

/// Resolves [`placed_texture`] and also returns the pre-lookup
/// [`crate::resource::all_target_changes`] generation so callers deferring a
/// mirror retain the pre-lookup generation even when resolution returned `None`.
pub(crate) fn placed_texture_with_gen(view: &View) -> (u64, Option<PlacedTexture>) {
    let gen_before = crate::resource::all_target_changes();
    let placed = placed_texture(view);
    let resolved_gen = placed.as_ref().map_or(gen_before, |p| p.resolved_gen);
    (resolved_gen, placed)
}

/// A placed texture's view on its host resource: of the title resource that
/// writes the memory when there is one, else of a synthetic copy. UAV
/// writes must land on a writable resource (a synthetic non-UAV copy would
/// drop writes), and an incompatible live real title owner blocks synthetic
/// fallback for both placed UAVs and placed SRVs with `ALLOW_UNORDERED_ACCESS`.
pub(crate) fn placed_texture(view: &View) -> Option<PlacedTexture> {
    const BUFFER: u32 = 1;
    let View::Placed(unordered, resource, desc) = view else {
        return None;
    };
    if resource[0] == BUFFER || (*unordered && resource[12] & 4 == 0) {
        return None;
    }
    let resolved_gen = crate::resource::all_target_changes();
    let address = u64::from(desc[10]) | u64::from(desc[11]) << 32;
    let owner = if resource[12] & 4 != 0 {
        match crate::resource::writable_texture_at(address, resource) {
            crate::resource::WritableTextureAt::Match(owner) => Some(owner),
            crate::resource::WritableTextureAt::IncompatibleRealOwner => return None,
            crate::resource::WritableTextureAt::None => None,
        }
    } else {
        None
    };
    let texture = match &owner {
        Some(owner) => owner.as_ptr(),
        // Unordered needed a writable resource; a synthetic one with no
        // writable flag would drop writes.
        None if *unordered => return None,
        None => crate::resource::texture_at(address, resource)?,
    };
    crate::resource::want_level(texture, crate::resource::first_mip(desc));
    let words = desc[..10].try_into().ok();
    let view = if *unordered {
        View::Unordered(texture as usize, 0, words)
    } else {
        View::Shader(texture as usize, words)
    };
    Some(PlacedTexture { view, _owner: owner, resolved_gen })
}

/// A null view of the kind `view` is, for a slot whose view the host
/// cannot make: unordered access (to a buffer or a 2D texture) or a shader
/// view. Writes to it are dropped, reads give zeros.
fn null_for(view: &View, at: D3D12_CPU_DESCRIPTOR_HANDLE) {
    let Some(host) = host() else { return };
    let is_buffer = |r: &usize| crate::resource::buffer_bytes(*r as crate::object::This).is_some();
    // Unordered views: whether a buffer, and the title's view dimension.
    let (buffer, dimension) = match view {
        View::Unordered(r, _, desc) => (
            is_buffer(r) || desc.is_some_and(|d| d[1] == 1),
            desc.map(|d| d[1]),
        ),
        View::Placed(true, resource, desc) => (resource[0] == 1, Some(desc[1])),
        // Shader views of buffers read nothing through a null buffer view:
        // a null 2D view where the shader expects a buffer is a mismatch
        // the hardware does not survive in general (GPU-based validation
        // reports it).
        View::Shader(r, desc) if is_buffer(r) || desc.is_some_and(|d| d[1] == 1) => {
            return null_buffer_view(at);
        }
        View::Placed(false, resource, _) if resource[0] == 1 => return null_buffer_view(at),
        // Textures: a null view of the title's dimension (1D, arrays, 3D,
        // cubes), not always 2D.
        View::Shader(_, Some(desc)) => return null_like(desc, at),
        View::Placed(false, _, desc) => {
            return null_like(desc[..10].try_into().unwrap_or(&[0; 10]), at);
        }
        View::Constant(..) => {
            let desc = D3D12_CONSTANT_BUFFER_VIEW_DESC::default();
            // SAFETY: a slot of our heap; a zero location makes a null view.
            return unsafe { host.device.CreateConstantBufferView(Some(&desc), at) };
        }
        _ => return null_view(at),
    };
    let desc = if buffer {
        D3D12_UNORDERED_ACCESS_VIEW_DESC {
            Format: DXGI_FORMAT_R32_UINT,
            ViewDimension: D3D12_UAV_DIMENSION_BUFFER,
            ..Default::default()
        }
    } else {
        null_uav_desc_with_caps(dimension, host_writeable_msaa())
    };
    // SAFETY: a slot of our heap; a null resource is allowed with a desc.
    unsafe {
        host.device.CreateUnorderedAccessView(
            None::<&ID3D12Resource>,
            None::<&ID3D12Resource>,
            Some(&desc),
            at,
        )
    };
}

/// A null texture UAV of the title's view dimension (1D, 1D or 2D arrays,
/// 3D, or multisampled 2D / 2D array when `writeable_msaa` is supported),
/// falling back to 2D if it names none or lacks capability.
fn null_uav_desc_with_caps(
    dimension: Option<u32>,
    writeable_msaa: bool,
) -> D3D12_UNORDERED_ACCESS_VIEW_DESC {
    let mut desc = D3D12_UNORDERED_ACCESS_VIEW_DESC {
        Format: DXGI_FORMAT_R32_UINT,
        ViewDimension: D3D12_UAV_DIMENSION_TEXTURE2D,
        ..Default::default()
    };
    match dimension.map(|d| D3D12_UAV_DIMENSION(d as i32)) {
        Some(D3D12_UAV_DIMENSION_TEXTURE1D) => {
            desc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE1D;
        }
        Some(D3D12_UAV_DIMENSION_TEXTURE1DARRAY) => {
            desc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE1DARRAY;
            desc.Anonymous.Texture1DArray = D3D12_TEX1D_ARRAY_UAV {
                ArraySize: 1,
                ..Default::default()
            };
        }
        Some(D3D12_UAV_DIMENSION_TEXTURE2DARRAY) => {
            desc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2DARRAY;
            desc.Anonymous.Texture2DArray = D3D12_TEX2D_ARRAY_UAV {
                ArraySize: 1,
                ..Default::default()
            };
        }
        Some(D3D12_UAV_DIMENSION_TEXTURE2DMS) if writeable_msaa => {
            desc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2DMS;
        }
        Some(D3D12_UAV_DIMENSION_TEXTURE2DMSARRAY) if writeable_msaa => {
            desc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2DMSARRAY;
            desc.Anonymous.Texture2DMSArray = D3D12_TEX2DMS_ARRAY_UAV {
                ArraySize: 1,
                ..Default::default()
            };
        }
        Some(D3D12_UAV_DIMENSION_TEXTURE3D) => {
            desc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE3D;
            desc.Anonymous.Texture3D = D3D12_TEX3D_UAV {
                WSize: u32::MAX,
                ..Default::default()
            };
        }
        _ => {}
    }
    desc
}

#[cfg(test)]
fn null_uav_desc(dimension: Option<u32>) -> D3D12_UNORDERED_ACCESS_VIEW_DESC {
    null_uav_desc_with_caps(dimension, false)
}

/// A buffer view that reads zeros (raw, as translated shaders mostly read
/// buffers).
fn null_buffer_view(at: D3D12_CPU_DESCRIPTOR_HANDLE) {
    let Some(host) = host() else { return };
    let desc = D3D12_SHADER_RESOURCE_VIEW_DESC {
        Format: DXGI_FORMAT_R32_TYPELESS,
        ViewDimension: D3D12_SRV_DIMENSION_BUFFER,
        Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
        Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
            Buffer: D3D12_BUFFER_SRV {
                NumElements: 1,
                Flags: D3D12_BUFFER_SRV_FLAG_RAW,
                ..Default::default()
            },
        },
    };
    // SAFETY: a slot of our heap; a null resource is allowed with a desc.
    unsafe {
        host.device
            .CreateShaderResourceView(None::<&ID3D12Resource>, Some(&desc), at)
    };
}

/// A texture view that reads zeros with the dimension (and mip and array
/// counts) of the title's description `words`; 2D if it names none.
fn null_like(words: &[u32; 10], at: D3D12_CPU_DESCRIPTOR_HANDLE) {
    if words[1] == D3D12_SRV_DIMENSION_RAYTRACING_ACCELERATION_STRUCTURE.0 as u32
        && crate::dxr::requested()
        && let Some(host) = host()
        && empty_tlas_srv(host, at).is_some()
    {
        return;
    }
    const TEXTURES: std::ops::RangeInclusive<u32> = 2..=10;
    if !TEXTURES.contains(&words[1]) {
        return null_view(at);
    }
    let Some(host) = host() else { return };
    // One mip, one slice or cube, plane 0: fields a null view still checks.
    let mut w = [0u32; 10];
    w[1] = words[1];
    match words[1] {
        6 => {}
        7 => w[5] = 1,
        3 | 5 | 10 => (w[5], w[7]) = (1, 1),
        _ => w[5] = 1,
    }
    let mut desc = srv_desc(&w);
    desc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    desc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    // SAFETY: a slot of our heap; a null resource is allowed with a desc.
    unsafe {
        host.device
            .CreateShaderResourceView(None::<&ID3D12Resource>, Some(&desc), at)
    };
}

/// A view that reads zeros, for slots whose view the host cannot make.
/// shortcut: slots with no view at all (kind unknown) get this null 2D
/// view; null buffer views there did not stop the world's low-address
/// page faults (2026-11-24, run w10), so they are not the cause.
fn null_view(at: D3D12_CPU_DESCRIPTOR_HANDLE) {
    let Some(host) = host() else { return };
    let desc = D3D12_SHADER_RESOURCE_VIEW_DESC {
        Format: DXGI_FORMAT_R8G8B8A8_UNORM,
        ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2D,
        Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
        Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
            Texture2D: D3D12_TEX2D_SRV {
                MipLevels: 1,
                ..Default::default()
            },
        },
    };
    // SAFETY: a slot of our heap; a null resource is allowed with a desc.
    unsafe {
        host.device
            .CreateShaderResourceView(None::<&ID3D12Resource>, Some(&desc), at)
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every host sampler heap backs all 2,048 hardware descriptors and
    /// reserves slot 2,047 as a null sentinel so `umin(index, 2047)` in
    /// `victory-shader` never reads past smaller title sampler heaps.
    #[test]
    fn sampler_heaps_back_2048_slots_and_reserve_null_sentinel() {
        let Some(small) = HostHeap::new(true, 16) else { return };
        // SAFETY: live heap query.
        assert_eq!(unsafe { small.heap.GetDesc().NumDescriptors }, 2048);
        assert_eq!(small.count, 16);
        let Some(full) = HostHeap::new(true, 2048) else { return };
        // SAFETY: live heap query.
        assert_eq!(unsafe { full.heap.GetDesc().NumDescriptors }, 2048);
        assert_eq!(full.count, 2047);
    }

    /// A view of a host copy that begins at a later level shows the same
    /// levels of it.
    #[test]
    fn views_count_from_the_first_level_of_the_host_copy() {
        let view = |first, levels, clamp| {
            let mut desc = srv_desc(&[98, 4, 5768, 0, first, levels, 0, 0, 0, 0]);
            desc.Anonymous.Texture2D.ResourceMinLODClamp = clamp;
            from_base(&mut desc, 3);
            // SAFETY: the member its dimension says.
            let t = unsafe { desc.Anonymous.Texture2D };
            (t.MostDetailedMip, t.MipLevels, t.ResourceMinLODClamp)
        };
        assert_eq!(view(3, 4, 0.0), (0, 4, 0.0));
        assert_eq!(view(5, u32::MAX, 5.5), (2, u32::MAX, 2.5));
        // (Two levels the copy has not.)
        assert_eq!(view(1, 4, 0.0), (0, 2, 0.0));
    }

    /// 2D textures and arrays have a neutral stand-in, others a null view.
    #[test]
    fn a_texture_not_there_yet_has_a_neutral_stand_in() {
        let Some(host) = host() else { return };
        // SAFETY: plain heap creation and queries.
        unsafe {
            let heap: ID3D12DescriptorHeap = host
                .device
                .CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                    Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                    NumDescriptors: 1,
                    ..Default::default()
                })
                .unwrap();
            let at = heap.GetCPUDescriptorHandleForHeapStart();
            let view = |dimension| View::Shader(0, Some([98, dimension, 5768, 0, 0, 1, 0, 1, 0, 0]));
            assert!(neutral_for(&view(4), at).is_some());
            assert!(neutral_for(&view(5), at).is_some());
            assert!(neutral_for(&view(9), at).is_none());
            assert!(neutral_for(&view(8), at).is_none());
            assert!(host.device.GetDeviceRemovedReason().is_ok());
        }
    }

    /// Memory past the levels a slot's view shows is not its texture's.
    #[test]
    fn a_view_of_smaller_levels_is_not_told_of_memory_past_them() {
        let mut spans = Spans::default();
        spans.note(1, 0x10000, 0x50000, true);
        spans.shows(1, 3, 0x12000);
        spans.note(2, 0x10000, 0x50000, true);
        spans.shows(2, 0, 0x50000);
        assert_eq!(spans.slots_holding(0x11000, 0x11800), vec![1, 2]);
        assert_eq!(spans.slots_holding(0x20000, 0x21000), vec![2]);
        assert_eq!(spans.slots_in(0x20000, 0x21000), vec![2]);
        spans.forget(2);
        assert!(spans.slots_in(0x20000, 0x21000).is_empty());
        assert_eq!(spans.slots_in(0x0, 0x10001), vec![1]);
    }

    /// A texture filled with less than was mapped is looked at again, but
    /// not for ever: memory that is shown and cannot be read ends it, and a
    /// filling that got everything starts the count anew.
    #[test]
    fn a_short_filling_is_looked_at_again_a_few_times() {
        // An address no other test names.
        let start = 0x7fee_1234_0000u64;
        let pending = || super::SHORT.lock().unwrap().0.remove(&start);
        for _ in 0..super::SHORT_IN_A_ROW {
            super::read_short(start);
            assert!(pending());
        }
        super::read_short(start);
        assert!(!pending());
        super::read_all(start);
        super::read_short(start);
        assert!(pending());
        super::read_all(start);
    }

    /// A null texture UAV keeps the title's dimension (3D for the world's
    /// wind field), and is 2D when the title names none.
    #[test]
    fn null_uavs_keep_the_titles_dimension() {
        let d = null_uav_desc(Some(8));
        assert_eq!(d.ViewDimension, D3D12_UAV_DIMENSION_TEXTURE3D);
        // SAFETY: the member the dimension names.
        assert_eq!(unsafe { d.Anonymous.Texture3D.WSize }, u32::MAX);
        let d = null_uav_desc(Some(5));
        assert_eq!(d.ViewDimension, D3D12_UAV_DIMENSION_TEXTURE2DARRAY);
        // SAFETY: the member the dimension names.
        assert_eq!(unsafe { d.Anonymous.Texture2DArray.ArraySize }, 1);
        let dimension = |d| null_uav_desc(d).ViewDimension;
        assert_eq!(dimension(Some(2)), D3D12_UAV_DIMENSION_TEXTURE1D);
        assert_eq!(dimension(Some(4)), D3D12_UAV_DIMENSION_TEXTURE2D);
        assert_eq!(dimension(None), D3D12_UAV_DIMENSION_TEXTURE2D);
        // Multisampled UAVs need a cap; they stay 2D.
        assert_eq!(dimension(Some(6)), D3D12_UAV_DIMENSION_TEXTURE2D);
    }

    /// What the texture code asks of the spans (levels on demand): where
    /// the next texture starts inside a span, the lowest level a view of a
    /// texture shows, and a texture's slots.
    #[test]
    fn spans_say_the_next_start_and_the_lowest_level_viewed() {
        let mut spans = Spans::default();
        // A 512x512 texture of 0x56000 bytes viewed from levels 3 and 1,
        // and the next one 0x6000 after it (run 188).
        spans.note(1, 0x1_0000, 0x1_0000 + 0x5_6000, true);
        spans.shows(1, 3, u64::MAX);
        spans.note(2, 0x1_0000, 0x1_0000 + 0x5_6000, true);
        spans.shows(2, 1, u64::MAX);
        spans.note(3, 0x1_6000, 0x1_6000 + 0x5_6000, true);
        assert_eq!(spans.next_start(0x1_0000, 0x1_0000 + 0x5_6000), Some(0x1_6000));
        assert_eq!(spans.next_start(0x1_6000, 0x1_6000 + 0x5_6000), None);
        assert_eq!(spans.first_viewed(0x1_0000), Some(1));
        assert_eq!(spans.first_viewed(0x1_6000), None);
        assert_eq!(spans.slots_at(0x1_0000), vec![1, 2]);
        // The view from level 1 goes: the texture is viewed from level 3.
        spans.forget(2);
        assert_eq!(spans.first_viewed(0x1_0000), Some(3));
    }

    /// Views of one placed texture are told from views of others by the
    /// address and the resource description, not by what they show.
    #[test]
    fn views_of_one_placed_texture_are_told_by_address_and_description() {
        let texture = [3, 0, 0, 0, 512, 0, 512, 0x8_0001, 98, 1, 0, 0x105, 0, 0];
        let view = |resource: [u32; 14], first_mip: u32, address: u32| {
            let mut desc = [0u32; 16];
            (desc[0], desc[1], desc[4], desc[5], desc[10], desc[11]) = (98, 4, first_mip, 8 - first_mip, address, 0xb);
            View::Placed(false, resource, desc)
        };
        let low = view(texture, 3, 0x1000_0000);
        // more mips of the same texture
        assert!(same_placed_texture(&low, &view(texture, 0, 0x1000_0000)));
        // another address, another description, a buffer, an unordered view
        assert!(!same_placed_texture(&low, &view(texture, 3, 0x2000_0000)));
        let mut wider = texture;
        wider[4] = 1024;
        assert!(!same_placed_texture(&low, &view(wider, 3, 0x1000_0000)));
        let mut buffer = texture;
        buffer[0] = 1;
        assert!(!same_placed_texture(&view(buffer, 3, 0x1000_0000), &view(buffer, 3, 0x1000_0000)));
        let View::Placed(_, resource, desc) = low.clone() else { unreachable!() };
        assert!(!same_placed_texture(&low, &View::Placed(true, resource, desc)));
        assert!(!same_placed_texture(&low, &View::Shader(0, None)));
    }

    #[test]
    fn ranges_that_touch_are_one() {
        let mut ranges = vec![(40, 50), (30, 40), (35, 40), (10, 20), (45, 60), (60, 61), (1, 1)];
        join(&mut ranges);
        assert_eq!(ranges, [(1, 1), (10, 20), (30, 61)]);
        join(&mut ranges);
        assert_eq!(ranges.len(), 3);
    }

    /// A view's title address goes after the strides, at two dwords per
    /// slot; a view that is no buffer's, and none, enter 0.
    #[test]
    fn view_addresses_follow_the_strides() {
        use victory_shader::emit::{SLOT_STRIDES_BYTES, SLOT_STRIDES_COUNT, VIEW_ADDRESSES};
        let mut table = vec![0xdead_beefu32; SLOT_STRIDES_BYTES as usize / 4];
        let data = table.as_mut_ptr() as usize;
        // A ray-tracing structure's view: its address in words 4 and 5.
        const STRUCTURE: u32 = D3D12_SRV_DIMENSION_RAYTRACING_ACCELERATION_STRUCTURE.0 as u32;
        let mut desc = [0u32; 10];
        (desc[1], desc[4], desc[5]) = (STRUCTURE, 0x1234_5600, 0x42);
        let view = View::Shader(0, Some(desc));
        assert_eq!(view_address(&view), Some(0x42_1234_5600));
        // Placed acceleration-structure view carries the address in words 4 and 5 too;
        // zero addresses are rejected on both paths (`fxceless 9a9091d`).
        let mut placed_desc = [0u32; 16];
        (placed_desc[1], placed_desc[4], placed_desc[5]) = (STRUCTURE, 0x9876_5400, 0x7);
        assert_eq!(view_address(&View::Placed(false, [0; 14], placed_desc)), Some(0x7_9876_5400));
        let mut zero_desc = [0u32; 10];
        zero_desc[1] = STRUCTURE;
        assert_eq!(view_address(&View::Shader(0, Some(zero_desc))), None);
        let mut zero_placed = [0u32; 16];
        zero_placed[1] = STRUCTURE;
        assert_eq!(view_address(&View::Placed(false, [0; 14], zero_placed)), None);
        write_view_address(data, 7, Some(&view));
        write_view_address(data, 8, None);
        write_view_address(data, SLOT_STRIDES_COUNT - 1, Some(&view));
        // Past the table: nothing written.
        write_view_address(data, SLOT_STRIDES_COUNT, Some(&view));
        let at = VIEW_ADDRESSES as usize / 4;
        assert_eq!(table[at + 14..at + 18], [0x1234_5600, 0x42, 0, 0]);
        assert_eq!(table[at + 12..at + 14], [0xdead_beef; 2], "slot 6 untouched");
        let last = at + 2 * (SLOT_STRIDES_COUNT as usize - 1);
        assert_eq!(table[last..], [0x1234_5600, 0x42]);
        assert_eq!(table[..at].iter().filter(|&&w| w != 0xdead_beef).count(), 0, "the strides untouched");
    }

    /// Typed buffer views of integer formats are marked in the table of
    /// strides; float, normalized, raw and structured ones are not.
    #[test]
    fn integer_typed_buffer_views_are_marked_in_the_table_of_strides() {
        use victory_shader::emit::{VIEW_INTEGER, VIEW_SIGNED};
        // Format, dimension (buffer), first element, count, stride, counter, flags.
        let uav = |format, stride, flags| View::Unordered(0, 0, Some([format, 1, 0, 0, 64, stride, 0, 0, flags, 0]));
        assert_eq!(view_kind(&uav(12, 0, 0)), VIEW_INTEGER, "R16G16B16A16_UINT");
        assert_eq!(view_kind(&uav(59, 0, 0)), VIEW_INTEGER | VIEW_SIGNED, "R16_SINT");
        assert_eq!(view_kind(&uav(10, 0, 0)), 0, "R16G16B16A16_FLOAT");
        assert_eq!(view_kind(&uav(58, 0, 0)), 0, "R16_SNORM");
        assert_eq!(view_kind(&uav(39, 0, 1)), 0, "raw");
        assert_eq!(view_kind(&uav(0, 16, 0)), 0, "structured");
        assert_eq!(view_stride(&uav(12, 0, 0)), Some(8));
    }

    /// What may stand for a texture that is not there yet: one of its kind in
    /// another place or size; another one only if asked; nothing a shader
    /// cannot read in its place.
    #[test]
    fn a_slot_shows_on_what_can_stand_for_its_new_texture() {
        let texture = [3, 0, 0, 0, 512, 0, 256, 0x8_0001, 98, 1, 0, 0x105, 0, 0];
        let view = |resource: [u32; 14], format: u32, dimension: u32, address: u32| {
            let mut desc = [0u32; 16];
            (desc[0], desc[1], desc[10], desc[11]) = (format, dimension, address, 0xb);
            View::Placed(false, resource, desc)
        };
        let old = view(texture, 98, 4, 0x1000_0000);
        let mut twice = texture;
        (twice[4], twice[6], twice[7]) = (1024, 512, 0x9_0001);
        let mut square = texture;
        square[6] = 512;
        let mut buffer = texture;
        buffer[0] = 1;
        assert_eq!(kin(&old, &view(texture, 98, 4, 0x1000_0000)), Kin::Same);
        assert_eq!(kin(&old, &view(texture, 98, 4, 0x2000_0000)), Kin::Moved);
        assert_eq!(kin(&old, &view(twice, 98, 4, 0x2000_0000)), Kin::Sized);
        assert_eq!(kin(&old, &view(square, 98, 4, 0x2000_0000)), Kin::Other);
        assert_eq!(kin(&old, &view(twice, 83, 4, 0x2000_0000)), Kin::Other);
        assert_eq!(kin(&old, &view(texture, 98, 9, 0x2000_0000)), Kin::None);
        assert_eq!(kin(&old, &view(buffer, 98, 1, 0x2000_0000)), Kin::None);
        for (kin, kept) in [(Kin::Same, [true; 3]), (Kin::Moved, [false, true, true]), (Kin::Sized, [false, true, true]), (Kin::Other, [false, false, true]), (Kin::None, [false; 3])] {
            assert_eq!([0, 1, 2].map(|keep| kin.kept(keep)), kept, "{kin:?}");
        }
    }

    /// A constant buffer view larger than the host takes is cut to what it
    /// takes: the device, which such a view removes, stays.
    #[test]
    fn an_oversized_constant_buffer_view_leaves_the_device_alive() {
        let Some(host) = host() else {
            return; // no host GPU
        };
        let Some(heap) = HostHeap::new(false, 4) else {
            return;
        };
        // SAFETY: a plain device query.
        let alive = || unsafe { host.device.GetDeviceRemovedReason() }.is_ok();
        // Made first: an allocation that fails on a removed device must not
        // end the test early.
        let (buffer, _) = host.upload_buffer(1 << 20).expect("an upload buffer");
        let memory = vec![0u8; 2 << 20];
        let at = (memory.as_ptr() as u64).next_multiple_of(64 << 10);
        for size in [969_472u32, 896_256] {
            assert!(make(&View::Constant(at, size), heap.cpu(0)).is_some());
            assert!(alive(), "a view of {size} bytes");
        }
        // SAFETY: a live upload buffer.
        heap.constant_at(1, &buffer, unsafe { buffer.GetGPUVirtualAddress() }, 969_472);
        assert!(alive(), "a view of the list's own memory");
        assert_eq!(constant_bytes(480), 512);
        assert_eq!(constant_bytes(64 << 10), 64 << 10);
        assert_eq!(constant_bytes(u32::MAX), 64 << 10);
    }

    /// A null 3D UAV made on the real device leaves it alive.
    #[test]
    fn null_3d_uav_is_accepted_by_the_device() {
        let Some(host) = host() else {
            return; // no host GPU
        };
        let Some(heap) = HostHeap::new(false, 4) else {
            return;
        };
        let view = View::Unordered(0, 0, Some([28, 8, 0, 0, 0, 0, 0, 0, 0, 0]));
        null_for(&view, heap.cpu(0));
        // SAFETY: a plain device query.
        assert!(unsafe { host.device.GetDeviceRemovedReason() }.is_ok());
    }

    /// Dynamic slots stay taken until given back, and runs do not wrap.
    #[test]
    fn a_structured_view_made_raw_passes_the_host_check() {
        // 100 elements of 20 bytes from element 8, in a copy that starts 32
        // bytes before the view's memory: bytes 192..2192.
        let (first, count) = raw_window(32, 8, 100, 20).unwrap();
        assert_eq!((first, count), (48, 500));
        assert!(buffer_view_ok(DXGI_FORMAT_R32_TYPELESS, first, count, 0, true, 4096));
        // the format a raw view must name; the structured one's is refused
        assert!(!buffer_view_ok(DXGI_FORMAT_UNKNOWN, first, count, 0, true, 4096));
        // past the buffer, not on 16 bytes, no elements
        assert!(!buffer_view_ok(DXGI_FORMAT_R32_TYPELESS, first, count, 0, true, 2000));
        assert!(raw_window(4, 0, 100, 20).is_none());
        assert!(!buffer_view_ok(DXGI_FORMAT_R32_TYPELESS, 0, 0, 0, true, 4096));
        // a structured view the host takes as it is
        assert!(buffer_view_ok(DXGI_FORMAT_UNKNOWN, 1, 10, 12, false, 4096));
    }

    /// The raw twin of a view shows the same bytes: typed and structured ones
    /// that begin on 16 bytes, whole dwords, none past the buffer.
    #[test]
    fn raw_twins_show_the_views_bytes() {
        let (u16s, u32s, none) = (DXGI_FORMAT(57), DXGI_FORMAT(42), DXGI_FORMAT_UNKNOWN);
        assert_eq!(twin_window(u32s, 4, 10, 0, false, 4096), Some((4, 10, 0)));
        assert_eq!(twin_window(u16s, 8, 7, 0, false, 4096), Some((4, 4, 0)));
        // the last dword would reach past the buffer
        assert_eq!(twin_window(u16s, 8, 7, 0, false, 30), Some((4, 3, 0)));
        assert_eq!(twin_window(none, 2, 3, 24, false, 4096), Some((12, 18, 0)));
        assert_eq!(twin_window(DXGI_FORMAT_R32_TYPELESS, 8, 5, 0, true, 4096), Some((8, 5, 0)));
        // where no raw view begins: from the 16 bytes below
        assert_eq!(twin_window(none, 1, 3, 24, false, 4096), Some((4, 20, 8)));
        assert_eq!(twin_window(u32s, 1, 10, 0, false, 4096), Some((0, 11, 4)));
        assert_eq!(twin_window(u32s, 1, 10, 0, false, 4), None);
    }

    /// What a slot is made as is made beside the heap and copied in whole: the
    /// slot's descriptor is the one the view has, and each thread makes its
    /// own.
    #[test]
    fn views_are_made_beside_the_heap_and_copied_in() {
        let Some(host) = host() else {
            return; // no host GPU here
        };
        let Some(heap) = HostHeap::new(false, 4) else {
            return;
        };
        let (Some(a), Some(b)) = (staged(false, 0), staged(false, 1)) else {
            return;
        };
        assert_ne!(a.ptr, b.ptr);
        assert_ne!(Some(a.ptr), std::thread::spawn(|| staged(false, 0).map(|h| h.ptr)).join().unwrap());
        let (buffer, _) = host.shared_buffer(1 << 16).unwrap();
        let index = heap.dynamic(1).unwrap();
        let bytes = |at: D3D12_CPU_DESCRIPTOR_HANDLE| {
            let mut bytes = [0u8; 32];
            crate::host::read_into(at.ptr as *const u8, &mut bytes).then_some(bytes)
        };
        heap.raw_at(index, &buffer, 64, 1024);
        let (slot, beside) = (bytes(heap.cpu(index)), bytes(staged(false, 0).unwrap()));
        assert_eq!(slot, beside, "copied as it was made");
        heap.null_raw_at(index);
        assert_ne!(bytes(heap.cpu(index)), slot, "another view since");
        // SAFETY: a plain query.
        assert!(unsafe { host.device.GetDeviceRemovedReason() }.is_ok());
    }

    /// A raw view moved on by some dwords says so in its descriptor, on a
    /// card whose descriptors are what we take them for; on another nothing
    /// is changed.
    #[test]
    fn a_raw_view_is_moved_only_where_its_descriptor_says_where_it_is() {
        let (Some(host), Some(at)) = (host(), staged(false, 1)) else {
            return; // no host GPU here
        };
        let (buffer, _) = host.shared_buffer(1 << 16).unwrap();
        let desc = D3D12_SHADER_RESOURCE_VIEW_DESC {
            Format: DXGI_FORMAT_R32_TYPELESS,
            ViewDimension: D3D12_SRV_DIMENSION_BUFFER,
            Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
            Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                Buffer: D3D12_BUFFER_SRV {
                    FirstElement: 8,
                    NumElements: 100,
                    StructureByteStride: 0,
                    Flags: D3D12_BUFFER_SRV_FLAG_RAW,
                },
            },
        };
        // SAFETY: this thread's slot, a live buffer.
        let start = unsafe {
            host.device.CreateShaderResourceView(&buffer, Some(&desc), at);
            buffer.GetGPUVirtualAddress() + 32
        };
        let words = || {
            let mut bytes = [0u8; 32];
            assert!(crate::host::read_into(at.ptr as *const u8, &mut bytes));
            bytes
        };
        let before = words();
        assert!(!move_raw(at, start + 16, 400, 8), "another address");
        assert!(!move_raw(at, start, 404, 8), "another size");
        assert!(!move_raw(at, start, 400, 400), "nothing left");
        assert_eq!(words(), before);
        if move_raw(at, start, 400, 8) {
            let moved = words();
            assert_eq!(moved[..8], (start + 8).to_le_bytes());
            assert_eq!(moved[8..12], 392u32.to_le_bytes());
            assert_eq!(moved[12..], before[12..]);
        } else {
            assert_eq!(words(), before);
        }
    }

    /// Views that begin where no host view can: raw ones off 16 bytes,
    /// structured ones off their elements (and off 16 bytes, where they are
    /// made raw), typed ones off theirs.
    #[test]
    fn views_off_their_elements_are_told() {
        // the colour table bakes' parameter block: raw, 8 bytes into its window
        assert!(off_elements(8, 0, true, 4, true));
        assert!(!off_elements(0, 0, true, 4, true));
        assert!(!off_elements(48, 0, true, 4, true));
        // 456-byte elements: at one of them, on 16 bytes, at neither
        assert!(!off_elements(456, 456, false, 0, true));
        assert!(!off_elements(32, 456, false, 0, true));
        assert!(off_elements(100, 456, false, 0, true));
        // A typed view of 12-byte elements on 16 bytes and not on an element.
        assert!(off_elements(23_702_896, 0, false, 12, true));
        assert!(!off_elements(48, 0, false, 12, true));
        assert!(!off_elements(12, 0, false, 4, true));
        assert!(off_elements(12, 0, false, 8, true));
        // With no raw twin, a structured view is read as raw at its own slot.
        assert!(off_elements(56, 56, false, 0, false));
        assert!(!off_elements(56, 56, false, 0, true));
        assert!(!off_elements(112, 56, false, 0, false));
    }

    /// Such a view is made on a copy of its own bytes, and that copy is the one
    /// brought up to date for it; another's is its buffer's.
    #[test]
    fn views_off_their_elements_get_a_copy_of_their_bytes() {
        let Some(host) = host() else {
            return; // no host GPU here
        };
        let memory = vec![0u64; 1 << 13];
        let start = (memory.as_ptr() as u64).next_multiple_of(4096);
        // SAFETY: a plain CPU-side heap of one view.
        let cpu: ID3D12DescriptorHeap = unsafe {
            host.device.CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                NumDescriptors: 1,
                ..Default::default()
            })
        }
        .unwrap();
        // SAFETY: a live heap.
        let at = unsafe { cpu.GetCPUDescriptorHandleForHeapStart() };
        // Memory placed at `address`, seen as the title passes it: raw, 32
        // dwords, or four 456-byte elements.
        let view = |address: u64, raw: bool| {
            let resource: [u32; 14] = [1, 0, 0, 0, 4 * 456, 0, 1, 1 | 1 << 16, 0, 1, 0, 1, 0, 0];
            let mut desc = [0u32; 16];
            let words = if raw {
                [39, 1, 5768, 0, 0, 0, 32, 0, 1, 0]
            } else {
                [0, 1, 5768, 0, 0, 0, 4, 456, 0, 0]
            };
            desc[..10].copy_from_slice(&words);
            (desc[10], desc[11]) = (address as u32, (address >> 32) as u32);
            View::Placed(false, resource, desc)
        };
        let width = |v: &View| {
            let made = make(v, at).expect("a view").expect("a buffer");
            // SAFETY: a live buffer.
            unsafe { made.GetDesc() }.Width
        };
        for (off, raw, len) in [(8, true, 128u64), (8, false, 4 * 456)] {
            let v = view(start + off, raw);
            assert_eq!(width(&v), len.next_multiple_of(256));
            let (copy, address, bytes) = buffer_copy(&v).unwrap();
            assert_eq!((copy.span(), address, bytes), ((start + off, len), start + off, len));
        }
        // Register the backing range so the host copy starts at `start`,
        // independent of the loose memory window that contains the allocation.
        let mut v = view(start, false);
        let View::Placed(_, _, desc) = &mut v else { unreachable!() };
        desc[4] = 1; // FirstElement, at byte offset 456 in the copy
        desc[6] = 3; // NumElements
        let key = start as usize;
        crate::shadow::register(key, start, 4 * 456);
        let made = make(&v, at);
        let refreshed = buffer_copy(&v);
        crate::shadow::unregister(key, start);
        let made = made.expect("a view").expect("a buffer");
        let (copy, address, bytes) = refreshed.unwrap();
        assert_eq!((address, bytes), (start + 456, 3 * 456));
        assert_eq!(made, copy.buffer, "the descriptor and refresh share a copy");
        if victory_shader::emit::raw_views() == 0 {
            assert_eq!(copy.span(), (address, bytes));
            eprintln!("the structured slot without a raw twin uses a view-relative copy");
        } else {
            assert_eq!(copy.span(), (start, 4 * 456));
        }
        for (off, raw) in [(16, true), (32, false)] {
            let v = view(start + off, raw);
            let (copy, ..) = buffer_copy(&v).unwrap();
            assert_ne!(copy.span().0, start + off, "its buffer's copy");
            // SAFETY: a live buffer.
            assert_eq!(width(&v), unsafe { copy.buffer.GetDesc() }.Width);
        }
    }

    /// A slot rechecked with a live state says so when it is looked at, a plain
    /// recheck clears it, leaving the queue or overwriting clears it, and
    /// deferred over-budget live slots rotate ahead of recently hashed ones.
    #[test]
    fn live_slots_say_so_until_they_are_rechecked_plain_or_taken_out() {
        let mut q = TextureQueue::default();
        q.written(7, 1, 0, false);
        let write = q.due(1, 10)[0].1;
        q.recheck(7, write, 2, (0x1000, 0x2000), 5, 0, 0, 1, LiveState::Live, LiveBaseline::Dense);
        assert_eq!(
            q.checked(7),
            Some((0x1000, 0x2000, 5, 0, 0, 1, LiveState::Live, LiveBaseline::Dense))
        );
        assert!(q.due(1, 10).is_empty(), "due again from frame 2");
        assert_eq!(q.due(2, 10), [(7, write)], "a recheck is served");
        q.recheck(7, write, 3, (0x1000, 0x2000), 6, 0, 1, 1, LiveState::Plain, LiveBaseline::Strip);
        assert_eq!(q.checked(7).map(|c| (c.6, c.7)), Some((LiveState::Plain, LiveBaseline::Strip)));
        assert!(q.live.is_empty(), "Plain removes map entry");
        q.recheck(7, write, 3, (0x1000, 0x2000), 6, 0, 0, 1, LiveState::Paused, LiveBaseline::Dense);
        assert_eq!(q.checked(7).map(|c| (c.6, c.7)), Some((LiveState::Paused, LiveBaseline::Dense)));
        q.done(7, write);
        assert_eq!(q.checked(7), None);
        assert!(q.live.is_empty(), "a slot out of the queue is not live");
        // Another write: a fresh slot, not live, whatever it was.
        q.written(7, 4, 4, false);
        let write = q.due(4, 10)[0].1;
        q.recheck(7, write, 5, (0x1000, 0x2000), 6, 0, 0, 4, LiveState::Live, LiveBaseline::Dense);
        q.written(7, 5, 5, false);
        assert!(q.live.is_empty(), "overwritten slot clears stale live entry");
        q.recheck(7, q.due(5, 10)[0].1, 6, (0x1000, 0x2000), 6, 0, 0, 5, LiveState::Plain, LiveBaseline::Strip);
        assert_eq!(q.checked(7).map(|c| c.6), Some(LiveState::Plain));

        // Round-robin fairness across budget deferrals: deferred slot 12 keeps
        // its older poll rank and comes before newly hashed slots 10 and 11 next frame.
        let mut rr = TextureQueue::default();
        for slot in [10u32, 11, 12] {
            rr.written(slot, 1, 0, false);
        }
        let writes: Vec<(u32, u64)> = rr.due(1, 10);
        let w = |slot: u32| writes.iter().find(|&&(s, _)| s == slot).unwrap().1;
        rr.recheck(10, w(10), 2, (0x1000, 0x2000), 1, 0, 0, 1, LiveState::Live, LiveBaseline::Dense);
        rr.recheck(11, w(11), 2, (0x3000, 0x2000), 2, 0, 0, 1, LiveState::Live, LiveBaseline::Dense);
        rr.recheck(12, w(12), 2, (0x5000, 0x2000), 3, 0, 0, 1, LiveState::Live, LiveBaseline::Dense);
        assert_eq!(rr.due(2, 10), [(10, w(10)), (11, w(11)), (12, w(12))]);
        // Frame 2: hash 10 and 11, defer 12 for budget.
        rr.recheck(10, w(10), 3, (0x1000, 0x2000), 1, 0, 1, 1, LiveState::Live, LiveBaseline::Dense);
        rr.recheck(11, w(11), 3, (0x3000, 0x2000), 2, 0, 1, 1, LiveState::Live, LiveBaseline::Dense);
        rr.defer_recheck(12, w(12), 3);
        assert_eq!(rr.due(3, 10), [(12, w(12)), (10, w(10)), (11, w(11))]);
        assert_eq!(rr.checked(12).map(|c| c.4), Some(0), "deferred slot keeps looks count");
    }

    /// `TextureQueue::commit_baseline` upgrades a record-time `Pending(attempt)`
    /// slot with the exact submit-time fingerprint of uploaded bytes, clears
    /// `live` for `(Plain, Strip)`, and safely rejects stale callbacks —
    /// including two uploads of the SAME slot within the EXACT SAME frame.
    #[test]
    fn commit_baseline_updates_pending_and_ignores_stale_callbacks() {
        let mut q = TextureQueue::default();
        // 1) Live slot: Pending(a1) -> Dense with exact uploaded print and due.
        q.written(42, 10, 10, false);
        let w42 = q.due(10, 10)[0].1;
        let a1 = next_upload_attempt();
        q.recheck(42, w42, 11, (0x1000, 0x2000), 1, 0, 0, 10, LiveState::Live, LiveBaseline::Pending(a1));
        q.commit_baseline(42, w42, a1, 11, 0xbeef_cafe, 0, LiveState::Live, LiveBaseline::Dense);
        assert_eq!(
            q.checked(42),
            Some((0x1000, 0x2000, 0xbeef_cafe, 0, 0, 10, LiveState::Live, LiveBaseline::Dense))
        );
        // Repeat/stale callback once already Dense must no-op.
        q.commit_baseline(42, w42, a1, 99, 0x1111, 0, LiveState::Live, LiveBaseline::Dense);
        assert_eq!(q.checked(42).map(|c| c.2), Some(0xbeef_cafe));

        // 2) Plain slot: Pending(a2) -> Strip commits print + dense_print, updates due to frame + 4, and clears `live` map.
        q.written(43, 10, 10, false);
        let w43 = q.due(10, 10)[0].1;
        let a2 = next_upload_attempt();
        assert!(a2 > a1);
        q.recheck(43, w43, 11, (0x2000, 0x1000), 1, 0, 0, 10, LiveState::Plain, LiveBaseline::Pending(a2));
        assert!(q.live.contains_key(&43), "Pending Plain stays tracked until submit commits Strip");
        q.commit_baseline(43, w43, a2, 14, 0xface_b00c, 0x9999_8888, LiveState::Plain, LiveBaseline::Strip);
        assert!(!q.live.contains_key(&43), "Committed Plain Strip leaves `live` map");
        assert_eq!(
            q.checked(43),
            Some((0x2000, 0x1000, 0xface_b00c, 0x9999_8888, 0, 10, LiveState::Plain, LiveBaseline::Strip))
        );
        assert!(!q.due(13, 10).iter().any(|&(s, _)| s == 43));
        assert!(q.due(14, 10).iter().any(|&(s, _)| s == 43));

        // 3) Two uploads of the SAME slot in the EXACT SAME frame (`10`) with
        //    the SAME `write` get distinct monotonic tokens; the stale earlier
        //    callback is rejected while the newer callback commits cleanly.
        let same_frame_first = next_upload_attempt();
        let same_frame_second = next_upload_attempt();
        assert!(same_frame_second > same_frame_first);
        q.recheck(
            42,
            w42,
            11,
            (0x1000, 0x2000),
            1,
            0,
            0,
            10,
            LiveState::Live,
            LiveBaseline::Pending(same_frame_first),
        );
        q.recheck(
            42,
            w42,
            11,
            (0x1000, 0x2000),
            1,
            0,
            0,
            10,
            LiveState::Live,
            LiveBaseline::Pending(same_frame_second),
        );
        q.commit_baseline(42, w42, same_frame_first, 11, 0xdead, 0, LiveState::Live, LiveBaseline::Dense);
        assert_eq!(
            q.checked(42).map(|c| (c.2, c.7)),
            Some((1, LiveBaseline::Pending(same_frame_second))),
            "stale same-frame upload callback must not overwrite newer Pending(attempt)"
        );

        // 4) `retry_pending` pulls a distant ahead fallback (`due = 40`) forward to `11`
        //    only when both `write` and `Pending(attempt)` match, ignoring stale attempts
        //    or already-committed slots.
        let ahead_attempt = next_upload_attempt();
        q.written(44, 10, 10, false);
        let w44 = q.slots[&44].write;
        q.recheck(
            44,
            w44,
            40,
            (0x3000, 0x1000),
            1,
            0,
            0,
            10,
            LiveState::Plain,
            LiveBaseline::Pending(ahead_attempt),
        );
        q.retry_pending(44, w44, same_frame_first, 11);
        assert!(!q.due(11, 10).iter().any(|&(s, _)| s == 44), "stale attempt must not move due");
        q.retry_pending(44, w44.wrapping_add(1), ahead_attempt, 11);
        assert!(!q.due(11, 10).iter().any(|&(s, _)| s == 44), "stale write must not move due");
        q.retry_pending(44, w44, ahead_attempt, 11);
        assert!(q.due(11, 10).iter().any(|&(s, _)| s == 44), "matching Pending(attempt) must pull due to 11");
        // After baseline commits (e.g. due = 15), a stale `retry_pending` must not pull due back.
        q.commit_baseline(44, w44, ahead_attempt, 15, 0xaaaa, 0, LiveState::Plain, LiveBaseline::Strip);
        q.retry_pending(44, w44, ahead_attempt, 11);
        assert!(!q.due(11, 10).iter().any(|&(s, _)| s == 44), "committed baseline must ignore retry_pending");
        assert!(q.due(15, 10).iter().any(|&(s, _)| s == 44));

        // 5) `finish_ahead` handles non-recheck / partial ahead completion & pre-submit failure.
        q.written(45, 10, 10, false);
        let w45 = q.slots[&45].write;
        q.defer_recheck(45, w45, 40);
        q.finish_ahead(45, w45.wrapping_add(1), false, None, 11);
        assert!(!q.due(11, 10).iter().any(|&(s, _)| s == 45), "stale write must not move due");
        q.finish_ahead(45, w45, false, None, 11);
        assert!(q.due(11, 10).iter().any(|&(s, _)| s == 45), "failed ahead must schedule retry at frame+1");
        q.finish_ahead(45, w45, true, Some((20, 4096)), 11);
        assert!(!q.due(19, 10).iter().any(|&(s, _)| s == 45));
        assert!(q.due(20, 10).iter().any(|&(s, _)| s == 45));
        q.finish_ahead(45, w45, true, None, 11);
        assert!(!q.slots.contains_key(&45), "successful whole ahead without recheck retires slot");

        // 6) Remap while an older `Taken::Known` job is in flight: newer `write`
        //    calls `finish_ahead(..., false, None, frame + 1)` and stays queued
        //    at `frame + 1` instead of retiring against the older snapshot.
        q.written(46, 10, 10, false);
        let w46_old = q.slots[&46].write;
        q.changed(46, 10, 10);
        let w46_new = q.slots[&46].write;
        assert!(w46_new > w46_old);
        q.defer_recheck(46, w46_new, 40);
        q.finish_ahead(46, w46_old, true, None, 11);
        assert!(q.slots.contains_key(&46), "old write callback must not retire remapped slot");
        q.finish_ahead(46, w46_new, false, None, 11);
        assert!(
            q.due(11, 10).iter().any(|&(s, _)| s == 46),
            "Known ahead job finishing after remap must keep slot queued at frame + 1"
        );
        q.commit_baseline(
            42,
            w42,
            same_frame_second,
            11,
            0x2222_3333,
            0,
            LiveState::Live,
            LiveBaseline::Dense,
        );
        assert_eq!(
            q.checked(42).map(|c| (c.2, c.7)),
            Some((0x2222_3333, LiveBaseline::Dense))
        );

        // 4) Stale callback after slot overwrite (`written`) must be ignored.
        let a_over = next_upload_attempt();
        q.recheck(
            42,
            w42,
            13,
            (0x1000, 0x2000),
            1,
            0,
            0,
            12,
            LiveState::Live,
            LiveBaseline::Pending(a_over),
        );
        q.written(42, 13, 13, false);
        q.commit_baseline(42, w42, a_over, 13, 0xdead, 0, LiveState::Live, LiveBaseline::Dense);
        assert_eq!(q.checked(42), None, "stale write must not revive overwritten slot");
    }

    /// Under continuous streaming of newer higher-write-ID `Plain` slots
    /// exceeding the 2 MiB plain hash budget every frame, older `Pending` and
    /// deferred `Plain` slots are never starved and are probed within bounded
    /// frames.
    #[test]
    fn older_pending_and_deferred_plain_slots_never_starve_under_streaming() {
        let mut q = TextureQueue::default();
        // Older Pending slot (slot 1, low write ID) and older Plain recheck slot (slot 2).
        q.written(1, 1, 0, false);
        q.written(2, 1, 0, false);
        let init = q.due(1, 10);
        let w1 = init.iter().find(|&&(s, _)| s == 1).unwrap().1;
        let w2 = init.iter().find(|&&(s, _)| s == 2).unwrap().1;
        q.recheck(1, w1, 10, (0x1000, 0x40000), 1, 0, 0, 1, LiveState::Plain, LiveBaseline::Pending(next_upload_attempt()));
        q.recheck(2, w2, 10, (0x50000, 0x40000), 7, 9, 0, 1, LiveState::Plain, LiveBaseline::Strip);

        let mut next_slot = 100u32;
        let mut slot1_served_frame = None;
        let mut slot2_served_frame = None;
        const PLAIN_CAP_PER_FRAME: usize = 42; // 42 × 48 KiB <= 2 MiB

        for frame in 10..15u64 {
            // Stream 60 brand-new higher-ID Plain slots every single frame (> 42 budget).
            for _ in 0..60 {
                let s = next_slot;
                next_slot += 1;
                q.written(s, frame, frame, false);
                let w = q.slots.get(&s).unwrap().write;
                q.recheck(s, w, frame, (0x10_0000, 0x40000), 5, 6, 0, frame, LiveState::Plain, LiveBaseline::Strip);
            }
            let batch = q.due(frame, 4096);
            let mut serviced = 0usize;
            for (slot, write) in batch {
                if serviced < PLAIN_CAP_PER_FRAME {
                    serviced += 1;
                    if slot == 1 && slot1_served_frame.is_none() {
                        slot1_served_frame = Some(frame);
                    }
                    if slot == 2 && slot2_served_frame.is_none() {
                        slot2_served_frame = Some(frame);
                    }
                    q.done(slot, write);
                } else {
                    q.defer_recheck(slot, write, frame + 1);
                }
            }
        }
        assert_eq!(
            slot1_served_frame,
            Some(10),
            "older Pending slot must be prioritized ahead of newer higher-ID streams on frame 10"
        );
        assert!(
            slot2_served_frame.is_some_and(|f| f <= 12),
            "older deferred Plain slot must be serviced within bounded frames under continuous streaming, got {slot2_served_frame:?}"
        );
    }

    /// Even when a continuous flood keeps `> limit` entries in Tier 0 + Tier 1
    /// AND injects newer Tier-2 entries every frame, older unselected Tier-1
    /// (`Pending`) and Tier-2 (`Plain`) slots are both guaranteed to be
    /// selected within bounded frames (never starved behind `limit`).
    #[test]
    fn unselected_tier2_slot_makes_bounded_progress_under_limit_flood() {
        let mut q = TextureQueue::default();
        const LIMIT: usize = 4;

        // Seed 3 older Tier-2 Plain slots (slots 1, 2, 3; slot 3 sits behind 1 & 2)
        // and 2 older Tier-1 Pending slots (slots 11, 12; slot 12 sits behind 11).
        for slot in 1..=3u32 {
            q.written(slot, 1, 0, false);
            let w = q.slots.get(&slot).unwrap().write;
            q.recheck(slot, w, 10, (0x1000, 0x10000), 7, 9, 0, 1, LiveState::Plain, LiveBaseline::Strip);
        }
        for slot in 11..=12u32 {
            q.written(slot, 1, 0, false);
            let w = q.slots.get(&slot).unwrap().write;
            q.recheck(
                slot,
                w,
                10,
                (0x1800, 0x10000),
                1,
                0,
                0,
                1,
                LiveState::Plain,
                LiveBaseline::Pending(next_upload_attempt()),
            );
        }

        let mut next_slot = 100u32;
        let mut slot3_selected_frame = None;
        let mut slot12_selected_frame = None;
        for frame in 10..20u64 {
            // Flood 15 brand-new Tier-0 (deferred) + Tier-1 (Pending) + newer Tier-2 slots EVERY frame!
            for _ in 0..5 {
                let s_t0 = next_slot;
                next_slot += 1;
                q.written(s_t0, frame, frame, false);
                let w = q.slots.get(&s_t0).unwrap().write;
                q.recheck(s_t0, w, frame, (0x2000, 0x10000), 3, 4, 0, frame, LiveState::Plain, LiveBaseline::Strip);
                q.defer_recheck(s_t0, w, frame);
            }
            for _ in 0..5 {
                let s_t1 = next_slot;
                next_slot += 1;
                q.written(s_t1, frame, frame, false);
                let w = q.slots.get(&s_t1).unwrap().write;
                q.recheck(
                    s_t1,
                    w,
                    frame,
                    (0x3000, 0x10000),
                    1,
                    0,
                    0,
                    frame,
                    LiveState::Plain,
                    LiveBaseline::Pending(next_upload_attempt()),
                );
            }
            for _ in 0..5 {
                let s_t2 = next_slot;
                next_slot += 1;
                q.written(s_t2, frame, frame, false);
                let w = q.slots.get(&s_t2).unwrap().write;
                q.recheck(s_t2, w, frame, (0x4000, 0x10000), 5, 6, 0, frame, LiveState::Plain, LiveBaseline::Strip);
            }

            // Select ONLY `LIMIT = 4` entries (remaining entries stay unselected and never get defer_seq).
            let batch = q.due(frame, LIMIT);
            assert_eq!(batch.len(), LIMIT);
            for (slot, write) in batch {
                if slot == 3 {
                    slot3_selected_frame = Some(frame);
                }
                if slot == 12 {
                    slot12_selected_frame = Some(frame);
                }
                q.done(slot, write);
            }
            if slot3_selected_frame.is_some() && slot12_selected_frame.is_some() {
                break;
            }
        }
        assert_eq!(
            slot12_selected_frame,
            Some(11),
            "older Tier-1 Pending slot 12 behind slot 11 must be selected on frame 11 despite >LIMIT Tier-0 flood"
        );
        assert_eq!(
            slot3_selected_frame,
            Some(12),
            "older Tier-2 slot 3 behind 2 older Tier-2 slots must be selected on frame 12 despite >LIMIT Tier 0/1/2 flood"
        );
    }

    /// Even when 5,000 ready `again` (`Plain` recheck) slots AND 5,000 ready
    /// `Live` slots both exceed the 4,096 per-pass cap, `TextureQueue::in_order`
    /// interleaves all 3 buckets 1-for-1-for-1 (`[again, live, fresh, ...]`),
    /// so `batch[..3]` includes all three buckets on the very first pass, `live`
    /// runs ahead of `fresh`, and neither 4,096 `again` nor 4,096 `live` can
    /// ever starve `fresh`.
    #[test]
    fn plain_slots_never_starve_behind_thousands_of_live_slots() {
        let mut q = TextureQueue::default();
        // 5,000 Live slots (0..5_000)
        for slot in 0..5000u32 {
            q.written(slot, 1, 0, false);
        }
        for (slot, write) in q.due(1, 6000) {
            q.recheck(slot, write, 2, (0x1000, 0x2000), 7, 0, 0, 1, LiveState::Live, LiveBaseline::Dense);
        }
        // 5,000 Plain recheck (`again`) slots (10_000..15_000)
        for slot in 10_000..15_000u32 {
            q.written(slot, 1, 0, false);
        }
        for (slot, write) in q.due(1, 6000) {
            q.recheck(slot, write, 2, (0x4000, 0x2000), 9, 0, 0, 1, LiveState::Plain, LiveBaseline::Strip);
        }
        // 20 fresh newly created slots (20_000..20_020)
        for slot in 20_000..20_020u32 {
            q.written(slot, 2, 2, false);
        }
        let batch = q.due(2, 4096);
        assert_eq!(batch.len(), 4096);
        // Even breaking after just the first 3 items serves again, live, AND fresh (live ahead of fresh)!
        let first3: Vec<u32> = batch.iter().take(3).map(|&(s, _)| s).collect();
        assert!(
            (10_000..15_000).contains(&first3[0]),
            "slot 0 must be an again/Plain slot, got {}",
            first3[0]
        );
        assert!(first3[1] < 5000, "slot 1 must be a Live slot (ahead of fresh), got {}", first3[1]);
        assert!(
            (20_000..20_020).contains(&first3[2]),
            "slot 2 must be a fresh slot, got {}",
            first3[2]
        );

        let returned: std::collections::HashSet<u32> = batch.iter().map(|&(s, _)| s).collect();
        for slot in 20_000..20_020u32 {
            assert!(
                returned.contains(&slot),
                "Fresh slot {slot} starved behind 5,000 again + 5,000 Live slots"
            );
        }

        // Heavy 3-way contention (5,000 Live + 5,000 again + 5,000 fresh, cap 4,096): fair ~1/3 split.
        for slot in 20_020..25_000u32 {
            q.written(slot, 2, 2, false);
        }
        let split = q.due(2, 4096);
        assert_eq!(split.len(), 4096);
        let live_count = split.iter().filter(|&&(s, _)| s < 5000).count();
        let again_count = split.iter().filter(|&&(s, _)| (10_000..15_000).contains(&s)).count();
        let fresh_count = split.iter().filter(|&&(s, _)| (20_000..25_000).contains(&s)).count();
        assert_eq!(again_count, 1366);
        assert_eq!(fresh_count, 1365);
        assert_eq!(live_count, 1365);
    }

    /// The queue of textures picked by index serves the slots looked at
    /// before first, then the newest write and the oldest in turns; a
    /// slot whose memory is not there is looked at again after
    /// 1, 2, 4 frames up to the limit; and what the pass reports about a
    /// slot written again meanwhile is dropped.
    #[test]
    fn the_texture_queue_serves_the_newest_writes_first() {
        let mut q = TextureQueue::default();
        // Slots 10..18 written in frames 1..9, slot 5 last.
        for i in 0..8u32 {
            q.written(10 + i, u64::from(i) + 1, 0, false);
        }
        q.written(5, 9, 0, false);
        let order: Vec<u32> = q.due(9, 100).into_iter().map(|d| d.0).collect();
        // newest, oldest, newest, oldest, and so on
        assert_eq!(order, [5, 10, 17, 11, 16, 12, 15, 13, 14]);
        assert_eq!(q.due(9, 2).len(), 2);
        assert_eq!(q.census(9), [9, 9, 0, 0, 8]);
        // Slot 17's memory is not mapped: again next frame, then in 2, 4, 4.
        let write = |q: &TextureQueue, index: u32| q.due(u64::MAX, 100).iter().find(|d| d.0 == index).unwrap().1;
        let w17 = write(&q, 17);
        let mut frame = 9;
        for wait in [1, 2, 4, 4] {
            q.later(17, w17, frame, 4);
            assert!(!q.due(frame + wait - 1, 100).iter().any(|d| d.0 == 17));
            frame += wait;
            // Due again, and ahead of every slot never looked at.
            assert_eq!(q.due(frame, 100)[0].0, 17);
        }
        assert_eq!(q.census(9)[2], 1);
        q.done(17, w17);
        assert!(!q.due(u64::MAX, 100).iter().any(|d| d.0 == 17));
        // Uploaded with part of its memory there: kept, with what was there.
        let w16 = write(&q, 16);
        q.partly(16, w16, 12, 4096);
        assert_eq!(q.mapped(16), 4096);
        assert_eq!(q.due(12, 100)[0].0, 16);
        // The title writes slot 16 again while the pass works on it: what
        // the pass then says about the old write changes nothing.
        q.written(16, 13, 0, false);
        q.done(16, w16);
        q.later(16, w16, 13, 4);
        assert_eq!(q.mapped(16), 0);
        assert_eq!(q.due(13, 100)[0].0, 16);
        // A draw that makes the host slot (`seen`) keeps what the queue knows,
        let w15 = write(&q, 15);
        q.partly(15, w15, 20, 8192);
        q.written(15, 14, 0, true);
        assert_eq!(q.mapped(15), 8192);
        // and queues a slot the queue did not hold.
        q.written(99, 14, 0, true);
        assert!(q.due(14, 100).iter().any(|d| d.0 == 99));
        // A write due later (a table's slot) is not handed out before.
        q.written(98, 14, 16, false);
        assert!(!q.due(15, 100).iter().any(|d| d.0 == 98));
        assert!(q.due(16, 100).iter().any(|d| d.0 == 98));
    }

    /// Slots held back for lack of room on the card are not handed out
    /// until their frame, keep their place, are not counted as waiting
    /// for their memory, and all come back at once when the card has room;
    /// a slot written again meanwhile is the title's new write.
    #[test]
    fn slots_held_for_lack_of_room_come_back_when_there_is_room() {
        let mut q = TextureQueue::default();
        for i in 0..4u32 {
            q.written(20 + i, 1, 0, false);
        }
        let due = q.due(1, 100);
        assert_eq!(due.len(), 4);
        // The card is full: slots 20 and 21 are held back until frame 101.
        for &(index, write) in due.iter().filter(|d| d.0 < 22) {
            q.hold(index, write, 101);
        }
        let now: Vec<u32> = q.due(2, 100).into_iter().map(|d| d.0).collect();
        assert!(!now.contains(&20) && !now.contains(&21));
        assert!(now.contains(&22) && now.contains(&23));
        assert_eq!(q.census(2)[2], 0, "held slots do not wait for their memory");
        // Due again at their frame, in the order they had.
        assert_eq!(q.due(101, 100), due);
        // Slot 21 is written again while held: due as written.
        q.written(21, 3, 0, false);
        assert!(q.due(3, 100).iter().any(|d| d.0 == 21));
        // Room again at frame 5: slot 20 is due at once, and only it was
        // still held as it was.
        assert_eq!(q.room(5), 1);
        assert!(q.due(5, 100).iter().any(|d| d.0 == 20));
        assert_eq!(q.room(6), 0);
    }

    /// Textures of which only the smallest levels are held, packed into
    /// one another's address range (the three of run 188: 0x2c000 bytes
    /// each at full size, their starts 0x2000 and 0x4000 apart): writing
    /// any of them, in any order and again, leaves the others.
    #[test]
    fn packed_neighbours_stay() {
        let mut spans = Spans::default();
        assert_eq!(spans.rule, Rule::Start);
        let at = 0xb_86bf_8000u64;
        let (a, b, c) = (at, at + 0x2000, at + 0x6000);
        assert!(spans.note(105_998, a, a + 0x2c000, true).is_empty());
        assert!(spans.note(106_004, b, b + 0x2c000, true).is_empty());
        assert!(spans.note(108_521, c, c + 0x2c000, true).is_empty());
        // The first written again (as a title does when levels arrive),
        // from another slot too: its span reaches over the other two.
        assert!(spans.note(7, a, a + 0x2c000, true).is_empty());
        assert!(spans.note(106_004, b, b + 0x2c000, true).is_empty());
        let mut all = spans.slots_in(0, u64::MAX);
        all.sort_unstable();
        assert_eq!(all, [7, 105_998, 106_004, 108_521]);
        // A draw makes a slot of the second texture: it is one of them.
        assert!(spans.note(9, b, b + 0x2c000, false).is_empty());
        assert_eq!(spans.by_slot.get(&9), Some(&b));
    }

    /// A texture written where another starts replaces it: at the same
    /// address with another length, or inside the block of its smallest
    /// levels. The packed neighbours of the one replaced stay.
    #[test]
    fn a_texture_written_at_the_start_of_another_replaces_it() {
        let mut spans = Spans::default();
        let at = 0x7_0000_0000u64;
        assert!(spans.note(1, at, at + 0x2c000, true).is_empty());
        assert!(spans.note(2, at + 0x2000, at + 0x2e000, true).is_empty());
        // Another length from the same start: slot 1 is stale, slot 2 not.
        assert_eq!(spans.note(3, at, at + 0xb000, true), [1]);
        // A start inside the first 4 KiB of slot 3's texture: stale too.
        assert_eq!(spans.note(4, at + 0x800, at + 0x1800, true), [3]);
        // And the other way: a texture that starts just before one.
        assert_eq!(spans.note(5, at + 0x1800, at + 0x4800, true), [2]);
        let mut left = spans.slots_in(0, u64::MAX);
        left.sort_unstable();
        assert_eq!(left, [4, 5]);
        // Exactly 4 KiB apart: both there.
        assert!(spans.note(6, at + 0x2800, at + 0x3800, true).is_empty());
        // A draw's view that clashes is left out, and nothing is stale.
        assert!(spans.note(8, at + 0x900, at + 0x2900, false).is_empty());
        assert_eq!(spans.by_slot.get(&8), None);
        assert_eq!(spans.by_slot.len(), 3);
    }

    /// A slot written with another texture leaves its old span; the same
    /// view written again changes nothing.
    #[test]
    fn a_slot_rewritten_moves_to_its_new_span() {
        let mut spans = Spans::default();
        assert!(spans.note(1, 0x10000, 0x3c000, true).is_empty());
        assert!(spans.note(2, 0x12000, 0x3e000, true).is_empty());
        assert!(spans.note(1, 0x10000, 0x3c000, true).is_empty());
        assert_eq!(spans.slots_in(0x10000, 0x10001), [1]);
        // Slot 1 now views a texture far away.
        assert!(spans.note(1, 0x90000, 0x91000, true).is_empty());
        assert_eq!(spans.slots_in(0x10000, 0x10001), [] as [u32; 0]);
        assert_eq!(spans.slots_in(0x12000, 0x12001), [2]);
        assert_eq!(spans.slots_in(0x90000, 0x90001), [1]);
        spans.forget(1);
        spans.forget(2);
        assert!(spans.by_start.is_empty() && spans.by_slot.is_empty());
    }

    /// Memory the runtime maps, unmaps or writes names the textures that
    /// are there: among packed neighbours the one whose held levels the
    /// range lies in (it ends where the next one starts), and the ones
    /// that start inside the range.
    #[test]
    fn memory_changed_under_packed_neighbours_names_the_ones_there() {
        let mut spans = Spans::default();
        let at = 0x7_0000_0000u64;
        assert!(spans.note(1, at, at + 0x2c000, true).is_empty());
        assert!(spans.note(2, at + 0x2000, at + 0x2e000, true).is_empty());
        assert!(spans.note(3, at + 0x6000, at + 0x32000, true).is_empty());
        // Inside the first one's held levels.
        assert_eq!(spans.slots_in(at + 0x1000, at + 0x2000), [1]);
        // Inside the second's: not the first, which ends where it starts.
        assert_eq!(spans.slots_in(at + 0x3000, at + 0x4000), [2]);
        // Past the last start: the last one.
        assert_eq!(spans.slots_in(at + 0x20000, at + 0x21000), [3]);
        // A range over all of them.
        assert_eq!(spans.slots_in(at, at + 0x8000), [1, 2, 3]);
        // Past the end of all: none.
        assert_eq!(spans.slots_in(at + 0x32000, at + 0x33000), [] as [u32; 0]);
    }

    /// A write names the textures that hold its bytes: the one that
    /// starts where the write starts, not the neighbour before it, whose
    /// held levels end there.
    #[test]
    fn a_write_at_the_start_of_a_packed_texture_names_it_alone() {
        let mut spans = Spans::default();
        let at = 0x7_0000_0000u64;
        assert!(spans.note(1, at, at + 0x2c000, true).is_empty());
        assert!(spans.note(2, at + 0x2000, at + 0x2e000, true).is_empty());
        assert!(spans.note(3, at + 0x6000, at + 0x32000, true).is_empty());
        // A request that fills the second texture's held levels.
        assert_eq!(spans.slots_in(at + 0x2000, at + 0x6000), [1, 2]);
        assert_eq!(spans.slots_holding(at + 0x2000, at + 0x6000), [2]);
        // From inside the first one's held levels on: both.
        assert_eq!(spans.slots_holding(at + 0x1000, at + 0x3000), [1, 2]);
        // A whole 64 KiB page: every texture that starts in it.
        assert_eq!(spans.slots_holding(at, at + 0x10000), [1, 2, 3]);
        // Past the last start, the last one; past all, or no bytes, none.
        assert_eq!(spans.slots_holding(at + 0x20000, at + 0x21000), [3]);
        assert_eq!(spans.slots_holding(at + 0x32000, at + 0x33000), [] as [u32; 0]);
        assert_eq!(spans.slots_holding(at + 0x2000, at + 0x2000), [] as [u32; 0]);
    }

    /// A write behind the guards' back under a placed texture: a slot
    /// waiting for its first upload is left alone, one uploaded before is
    /// counted, and made due two frames on when that is asked for. (The
    /// ranges are handed over here, not through the list all heaps share.)
    #[test]
    fn a_page_write_makes_an_uploaded_texture_slot_due_when_asked() {
        let Some(heap) = HostHeap::new(false, 8) else {
            return; // no host GPU
        };
        let slots = |at: u64| -> Vec<u32> { heap.texture_slots_due(at, 8).0.into_iter().map(|d| d.0).collect() };
        // 64x64 BC7, one mip, at an address no test and no title uses.
        let at = 0x7fed_0000_0000u64;
        let mut desc = [0u32; 16];
        (desc[10], desc[11]) = (at as u32, (at >> 32) as u32);
        let view = View::Placed(false, [3, 0, 0, 0, 64, 0, 64, 0x1_0001, 98, 1, 0, 0x105, 0, 0], desc);
        assert!(heap.note_buffer(1, Some(&view), false).is_empty());
        let written = [(at, at + 0x1000)];
        // Waiting for its first upload: held, not uploaded, nothing done.
        assert_eq!(heap.slots_written_under(&written, 50, true), [1, 0, 0]);
        let (queued, write) = heap.texture_slots_due(u64::MAX, 8).0[0];
        assert_eq!(queued, 1);
        heap.texture_slot_done(1, write, None);
        assert_eq!(slots(u64::MAX), []);
        // Uploaded and out of the queue. Memory elsewhere: nothing.
        assert_eq!(heap.slots_written_under(&[(at + 0x10_0000, at + 0x10_1000)], 60, true), [0, 0, 0]);
        // Under it, counted only: it stays out of the queue.
        assert_eq!(heap.slots_written_under(&written, 60, false), [1, 1, 0]);
        assert_eq!(slots(u64::MAX), []);
        // Under it, asked for: due two frames on, not before.
        assert_eq!(heap.slots_written_under(&written, 60, true), [1, 1, 1]);
        assert_eq!(slots(61), []);
        assert_eq!(slots(62), [1]);
    }

    /// `VICTORY_STALE_TEXTURES=0`: nothing is ever stale, and a view that
    /// overlaps others is left out of the spans, as before.
    #[test]
    fn never_stale_leaves_overlapping_views_out() {
        let mut spans = Spans { rule: Rule::Never, ..Default::default() };
        assert!(spans.note(1, 0x1000, 0x3000, true).is_empty());
        assert!(spans.note(2, 0x2000, 0x4000, true).is_empty());
        assert!(spans.note(3, 0x1000, 0x2000, true).is_empty());
        assert_eq!(spans.slots_in(0, u64::MAX), [1]);
    }

    /// The rule until run 188 (`VICTORY_STALE_TEXTURES=overlap`): a view
    /// written over memory that older views describe otherwise makes
    /// those stale; views of the same span are one texture's.
    #[test]
    fn a_newer_view_of_other_memory_makes_the_older_stale() {
        let mut spans = Spans { rule: Rule::Overlap, ..Default::default() };
        // Three textures side by side, the first in two slots.
        assert!(spans.note(1, 0x1000, 0x3000, true).is_empty());
        assert!(spans.note(2, 0x1000, 0x3000, true).is_empty());
        assert!(spans.note(3, 0x3000, 0x4000, true).is_empty());
        assert!(spans.note(4, 0x8000, 0x9000, true).is_empty());
        // The same slot written with the same span again: nothing changes.
        assert!(spans.note(2, 0x1000, 0x3000, true).is_empty());
        assert_eq!(spans.slots_in(0x2000, 0x3800), [1, 2, 3]);
        assert_eq!(spans.slots_in(0x4000, 0x8000), [] as [u32; 0]);
        // A draw makes a slot whose view overlaps them: left out, nothing stale.
        assert!(spans.note(9, 0x2000, 0x3800, false).is_empty());
        assert_eq!(spans.slots_in(0x2000, 0x3800), [1, 2, 3]);
        // The title writes a texture over the end of the first and all of
        // the second: their three slots are stale.
        let mut stale = spans.note(5, 0x2000, 0x4000, true);
        stale.sort_unstable();
        assert_eq!(stale, [1, 2, 3]);
        assert_eq!(spans.slots_in(0, u64::MAX), [5, 4]);
        // Another start in the same memory: stale in turn. Another length
        // from the same start too.
        assert_eq!(spans.note(6, 0x2800, 0x3000, true), [5]);
        assert_eq!(spans.note(7, 0x2800, 0x2c00, true), [6]);
        // A slot that moves to another texture leaves its span behind.
        assert!(spans.note(7, 0x8000, 0x9000, true).is_empty());
        assert_eq!(spans.slots_in(0x2800, 0x2c00), [] as [u32; 0]);
        assert_eq!(spans.slots_in(0x8000, 0x8001), [4, 7]);
        spans.forget(4);
        spans.forget(7);
        assert!(spans.by_start.is_empty() && spans.by_slot.is_empty());
    }

    /// Memory the runtime maps under a slot's texture makes the slot due
    /// again from the next frame; a stale slot leaves the queue.
    #[test]
    fn remapped_memory_makes_its_texture_slots_due() {
        let Some(heap) = HostHeap::new(false, 8) else {
            return; // no host GPU
        };
        let slots = |at: u64| -> Vec<u32> { heap.texture_slots_due(at, 8).0.into_iter().map(|d| d.0).collect() };
        // 64x64 BC7, one mip, at an address no test and no title uses.
        let at = 0x7fee_0000_0000u64;
        let view = |address: u64, width: u32| {
            let mut desc = [0u32; 16];
            (desc[10], desc[11]) = (address as u32, (address >> 32) as u32);
            View::Placed(false, [3, 0, 0, 0, width, 0, 64, 0x1_0001, 98, 1, 0, 0x105, 0, 0], desc)
        };
        let len = crate::resource::placed_bytes(&[3, 0, 0, 0, 64, 0, 64, 0x1_0001, 98, 1, 0, 0x105, 0, 0]).unwrap();
        assert!(heap.note_buffer(1, Some(&view(at, 64)), false).is_empty());
        let (queued, write) = heap.texture_slots_due(u64::MAX, 8).0[0];
        assert_eq!(queued, 1);
        heap.texture_slot_done(1, write, None);
        assert_eq!(slots(u64::MAX), []);
        // Memory elsewhere: nothing. Under the texture: due from the next frame.
        remapped((at + 2 * len) as usize, 0x1000);
        remapped(at as usize, 0x1000);
        assert_eq!(heap.texture_slots_remapped(50), 1);
        assert_eq!(slots(50), []);
        assert_eq!(slots(51), [1]);
        // A wider texture written at the same address: slot 1 is stale.
        assert_eq!(heap.note_buffer(2, Some(&view(at, 128)), false), [(1, 0)]);
        assert_eq!(slots(u64::MAX), [2]);
    }

    /// A write of no placed texture view forgets the slot in the spans
    /// whether it goes by their lock or not (`spans_quick`): slots that are
    /// in them stay marked, also one another's view made stale.
    #[test]
    fn writes_of_other_views_leave_the_spans_as_under_their_lock() {
        let Some(heap) = HostHeap::new(false, 8) else {
            return; // no host GPU
        };
        let marked = |slot: u32| heap.spanned.all(u64::MAX).contains(&slot);
        let at = 0x7fed_0000_0000u64;
        let view = |address: u64, width: u32| {
            let mut desc = [0u32; 16];
            (desc[10], desc[11]) = (address as u32, (address >> 32) as u32);
            View::Placed(false, [3, 0, 0, 0, width, 0, 64, 0x1_0001, 98, 1, 0, 0x105, 0, 0], desc)
        };
        assert!(heap.note_buffer(1, Some(&view(at, 64)), false).is_empty());
        assert!(marked(1) && !marked(2));
        // Slot 1 made stale by slot 2: out of `by_slot`, still in `first`.
        assert_eq!(heap.note_buffer(2, Some(&view(at, 128)), false), [(1, 0)]);
        assert!(marked(1) && marked(2));
        assert!(heap.placed.lock().unwrap().first.contains_key(&1));
        // Written with no view: forgotten, under the lock, and unmarked.
        assert!(heap.note_buffer(1, None, false).is_empty());
        assert!(!marked(1) && !heap.placed.lock().unwrap().first.contains_key(&1));
        // A slot never in the spans: nothing there to change.
        assert!(heap.note_buffer(3, None, false).is_empty());
        assert!(heap.note_buffer(2, None, false).is_empty());
        let spans = heap.placed.lock().unwrap();
        assert!(spans.by_slot.is_empty() && spans.by_start.is_empty() && spans.first.is_empty());
        assert!(heap.spanned.all(u64::MAX).is_empty());
    }

    #[test]
    fn a_set_of_slots_holds_what_was_put_in() {
        let set = Slots::new(200);
        for slot in [199, 0, 64, 63, 130, 64] {
            set.put(slot);
        }
        assert_eq!(set.all(u64::MAX), [0, 63, 64, 130, 199]);
        assert_eq!(set.part::<2>(0), [0, 64, 130]);
        assert!(set.remove(64) && !set.remove(64) && !set.remove(5));
        assert_eq!(set.take(), [0, 63, 130, 199]);
        assert!(set.take().is_empty());
    }

    #[test]
    fn spread_scan_visits_each_live_slot_in_its_original_part() {
        let set = Slots::new(201);
        let live: Vec<u32> = (0..201).filter(|slot| slot % 5 != 2).collect();
        for &slot in &live {
            set.put(slot);
        }
        for part in 0..16 {
            let expected: Vec<_> = live.iter().copied().filter(|slot| slot % 16 == part).collect();
            assert_eq!(set.part::<16>(part), expected);
        }
        assert!(set.part::<16>(16).is_empty());
        assert_eq!(set.take(), live);
    }

    /// A slot a draw read through a table leaves the queue; the title's
    /// next write of it waits for its draw, and is due after the grace
    /// frames if none reads it.
    #[test]
    fn slots_read_through_tables_leave_the_texture_queue() {
        let Some(heap) = HostHeap::new(false, 8) else {
            return; // no host GPU
        };
        let frame = 100;
        let slots = |at: u64| -> Vec<u32> { heap.texture_slots_due(at, 8).0.into_iter().map(|d| d.0).collect() };
        let texture = View::Placed(false, [3, 0, 0, 0, 4, 0, 4, 0x1_0001, 28, 1, 0, 0x105, 0, 0], [0; 16]);
        assert!(is_texture_view(&texture));
        heap.queue_texture(2, true, false, frame);
        heap.queue_texture(3, true, false, frame);
        assert_eq!(slots(frame), [3, 2]);
        heap.tabled_in(3, true, frame);
        assert_eq!(slots(frame), [2]);
        // Made in place by the draw that read it: still out.
        heap.queue_texture(3, true, true, frame);
        assert_eq!(slots(frame), [2]);
        // Written again in the next frame: due once the grace frames have
        // passed without a draw that reads it.
        heap.queue_texture(3, true, false, frame + 1);
        assert_eq!(slots(frame + TABLED_GRACE), [2]);
        assert_eq!(slots(frame + 1 + TABLED_GRACE), [3, 2]);
        // The draw comes after all: out again.
        heap.tabled_in(3, true, frame + 1);
        assert_eq!(slots(u64::MAX), [2]);
        // A slot of a table no draw read stays queued; its next write waits.
        heap.tabled_in(2, false, frame + 1);
        assert_eq!(slots(frame + 1), [2]);
        heap.queue_texture(2, true, false, frame + 2);
        assert_eq!(slots(frame + 2), []);
        // Long after the last table, a write is due at once.
        heap.queue_texture(2, true, false, frame + 3 + TABLED_FRAMES);
        assert_eq!(slots(frame + 3 + TABLED_FRAMES), [2]);
    }

    #[test]
    fn dynamic_slots_are_held_until_released() {
        let Some(heap) = HostHeap::new(false, 4) else {
            return; // no host GPU
        };
        let a = heap.dynamic(heap.dynamic - 100).unwrap();
        let b = heap.dynamic(60).unwrap();
        assert_eq!((a, b), (4, 4 + heap.dynamic - 100));
        // 40 left at the end: 50 do not fit until something is given back.
        assert_eq!(heap.dynamic(50), None);
        heap.release(heap.id(), a, 50);
        assert_eq!(heap.dynamic(50), Some(4));
        assert_eq!(heap.dynamic(40), Some(4 + heap.dynamic - 40));
    }

    #[test]
    fn dynamic_search_can_find_a_run_across_the_cursor() {
        let Some(heap) = HostHeap::new(false, 4) else {
            eprintln!("no host descriptor heap; skipped");
            return;
        };
        let first = heap.dynamic(1).unwrap();
        let left = heap.dynamic(2).unwrap();
        let right = heap.dynamic(2).unwrap();
        let rest = heap.dynamic(heap.dynamic - 5).unwrap();
        assert_eq!((first, left, right, rest), (4, 5, 7, 9));

        // Reallocating the left half puts the cursor back in the middle.
        heap.release(heap.id(), left, 2);
        assert_eq!(heap.dynamic(2), Some(left));
        heap.release(heap.id(), left, 2);
        heap.release(heap.id(), right, 2);
        assert_eq!(heap.dynamic(4), Some(left));
        assert_eq!(heap.dynamic(1), None);

        // A free slot at each physical end does not form a two-slot range.
        heap.release(heap.id(), first, 1);
        let last = first + heap.dynamic - 1;
        heap.release(heap.id(), last, 1);
        assert_eq!(heap.dynamic(2), None);
    }

    #[test]
    fn past_heap_warns_only_on_power_of_two_expansion() {
        let past = PastHeap::new();
        let count = 1000;
        assert_eq!(past.note(1000, count), (1000, true)); // 1 slot past
        assert_eq!(past.note(1000, count), (1000, false));
        assert_eq!(past.note(1001, count), (1001, true)); // 2 slots past
        assert_eq!(past.note(1002, count), (1002, false)); // 3 slots past
        assert_eq!(past.note(1003, count), (1003, true)); // 4 slots past
        assert_eq!(past.note(1001, count), (1003, false)); // lower than highest
        assert_eq!(past.get(), (6, 1003, count));
    }

    #[test]
    fn bake_coverage_is_per_queue_and_rejects_overlapping_passes() {
        let a = std::sync::Arc::new(Mutex::new(TextureQueue::default()));
        let b = std::sync::Arc::new(Mutex::new(TextureQueue::default()));
        {
            let pass = TexturePass::begin(&a);
            let rev = pass.revision.expect("single pass gets a revision");
            drop(pass);
            a.lock().unwrap().finish_bake(rev, 10);
        }
        assert!(a.lock().unwrap().bake_covered(10));
        assert!(!a.lock().unwrap().bake_covered(11), "advancing frame must invalidate bake coverage");
        assert!(!b.lock().unwrap().bake_covered(10), "coverage must not leak across heaps");

        // Writing a new slot bumps revision and invalidates prior coverage.
        a.lock().unwrap().written(42, 10, 10, false);
        assert!(!a.lock().unwrap().bake_covered(10));

        // Async `retry_pending` invalidates bake coverage on the SAME frame.
        let write = {
            let mut q = a.lock().unwrap();
            let write = q.slots[&42].write;
            q.recheck(
                42,
                write,
                40,
                (0x1000, 0x1000),
                1,
                0,
                0,
                10,
                LiveState::Plain,
                LiveBaseline::Pending(7),
            );
            write
        };
        {
            let pass = TexturePass::begin(&a);
            let rev = pass.revision.expect("solo pass gets revision");
            drop(pass);
            a.lock().unwrap().finish_bake(rev, 10);
        }
        assert!(a.lock().unwrap().bake_covered(10));
        a.lock().unwrap().retry_pending(42, write, 7, 11);
        assert!(
            !a.lock().unwrap().bake_covered(10),
            "async retry_pending must invalidate same-frame bake coverage"
        );

        // Async `finish_ahead` failure also invalidates bake coverage.
        {
            let pass = TexturePass::begin(&a);
            let rev = pass.revision.expect("solo pass gets revision");
            drop(pass);
            a.lock().unwrap().finish_bake(rev, 10);
        }
        assert!(a.lock().unwrap().bake_covered(10));
        a.lock().unwrap().finish_ahead(42, write, false, None, 11);
        assert!(
            !a.lock().unwrap().bake_covered(10),
            "async finish_ahead must invalidate same-frame bake coverage"
        );

        // Overlapping passes cannot certify a bake: `p2` bumps revision so even
        // `p1`'s earlier revision cannot certify!
        let p1 = TexturePass::begin(&a);
        let p2 = TexturePass::begin(&a);
        let r1 = p1.revision.expect("first pass saw active_passes == 1");
        assert!(p2.revision.is_none());
        drop(p1);
        drop(p2);
        a.lock().unwrap().finish_bake(r1, 10);
        assert!(
            !a.lock().unwrap().bake_covered(10),
            "overlapping pass must prevent earlier pass from certifying"
        );

        // After both leave, a fresh solo pass can certify again.
        let solo = TexturePass::begin(&a);
        let rev = solo.revision.expect("solo pass after overlap gets revision");
        drop(solo);
        a.lock().unwrap().finish_bake(rev, 10);
        assert!(a.lock().unwrap().bake_covered(10));
    }

    #[test]
    fn uav_recheck_ring_delivers_ranges_to_every_heap_independently() {
        let Some(heap_a) = HostHeap::new(false, 64) else { return };
        let Some(heap_b) = HostHeap::new(false, 64) else { return };
        heap_a.placed.lock().unwrap().note(3, 0x2000_0000, 0x2001_0000, true);
        heap_b.placed.lock().unwrap().note(7, 0x2000_0000, 0x2001_0000, true);
        push_uav_range((0x2000_0000, 0x2001_0000));
        assert!(heap_a.has_unread_uav_written());
        assert!(heap_b.has_unread_uav_written());
        assert_eq!(heap_a.texture_slots_uav_written(200), 1);
        assert!(!heap_a.has_unread_uav_written());
        assert!(
            heap_b.has_unread_uav_written(),
            "draining heap A must not consume heap B's unread UAV ranges"
        );
        assert_eq!(heap_b.texture_slots_uav_written(200), 1);
        assert!(!heap_b.has_unread_uav_written());
        // Mark slot 3 uploaded/done on Heap A, then issue a second identical UAV
        // write to the exact same range: it must NOT be dropped and Heap A must
        // observe and requeue slot 3 again.
        let (due_a, _) = heap_a.texture_slots_due(201, 64);
        assert_eq!(due_a.len(), 1);
        heap_a.texture_slot_done(due_a[0].0, due_a[0].1, None);
        push_uav_range((0x2000_0000, 0x2001_0000));
        assert!(heap_a.has_unread_uav_written());
        assert_eq!(heap_a.texture_slots_uav_written(201), 1);
        assert!(!heap_a.has_unread_uav_written());
    }

    #[test]
    fn bindless_dynamic_placed_slot_requeues_and_rebuilds_on_owner_switch() {
        use windows::core::Interface;
        let Some(heap) = HostHeap::new(false, 64) else { return };
        let backing = vec![0u8; 64 << 10];
        let addr = backing.as_ptr() as u64;
        // 64x64 R8G8B8A8_UNORM (28) with UAV flag (4).
        let res_words = [3u32, 0, 0, 0, 64, 0, 64, 0x1_0001, 28, 1, 0, 0, 4, 0];
        let mut view_words = [0u32; 16];
        view_words[0] = 28;
        view_words[1] = 4; // Texture2D UAV
        view_words[10] = addr as u32;
        view_words[11] = (addr >> 32) as u32;

        // 1) Create real writable title texture A at `addr`.
        let mut res_a = std::ptr::null();
        // SAFETY: 56-byte ResourceDesc layout.
        let rdesc: crate::resource::ResourceDesc =
            unsafe { res_words.as_ptr().cast::<crate::resource::ResourceDesc>().read_unaligned() };
        assert_eq!(
            crate::resource::create_at_address(
                std::ptr::null(),
                addr,
                &rdesc,
                0,
                std::ptr::null(),
                &crate::resource::PAGEABLE_IIDS[0],
                &mut res_a
            ),
            crate::object::S_OK
        );

        // 2) Record a placed UAV descriptor in slot 0 and mirror it.
        let mut slot_mem = [0u32; 4];
        let handle = slot_mem.as_mut_ptr() as u64;
        crate::descriptor::create_placed_uav(
            std::ptr::null(),
            res_words.as_ptr().cast(),
            view_words.as_ptr().cast(),
            handle,
        );
        let (view, placed_a) = heap.mirror_resolved(0, handle, true).expect("mirror A");
        assert!(is_dynamic_placed_texture(&view));
        let host_a = heap.kept_in(0).expect("slot 0 holds host texture A");
        assert_eq!(placed_a.as_ref().unwrap().owner_id(), crate::resource::resource_id(res_a));
        drop(placed_a);
        // Drain queue and confirm slot 0 is no longer queued.
        let (due_before, _) = heap.texture_slots_due(10, 64);
        for (idx, w) in due_before {
            heap.texture_slot_done(idx, w, None);
        }
        assert!(heap.texture_slots_due(10, 64).0.is_empty());

        // 3) Free A and create real writable texture B at the EXACT SAME address
        //    without rewriting the guest descriptor slot!
        let release_a = unsafe {
            let vtbl = *(res_a as *const *const *const ());
            std::mem::transmute::<*const (), extern "system" fn(crate::object::This) -> u32>(*vtbl.add(2))
        };
        assert_eq!(release_a(res_a), 0);
        let mut res_b = std::ptr::null();
        assert_eq!(
            crate::resource::create_at_address(
                std::ptr::null(),
                addr,
                &rdesc,
                0,
                std::ptr::null(),
                &crate::resource::PAGEABLE_IIDS[0],
                &mut res_b
            ),
            crate::object::S_OK
        );

        // 4) Target-owner change scan automatically requeues bindless slot 0.
        assert!(heap.has_stale_dynamic_placed());
        assert_eq!(heap.texture_slots_target_owners(10), 1);
        let (due_after, _) = heap.texture_slots_due(10, 64);
        assert_eq!(due_after.len(), 1);
        assert_eq!(due_after[0].0, 0);

        // 5) Re-mirroring slot 0 with the unchanged guest handle switches kept
        //    host descriptor from A to B!
        let (_, placed_b) = heap.mirror_resolved(0, handle, true).expect("mirror B");
        let host_b = heap.kept_in(0).expect("slot 0 holds host texture B");
        assert_eq!(placed_b.as_ref().unwrap().owner_id(), crate::resource::resource_id(res_b));
        assert_ne!(host_a.as_raw(), host_b.as_raw());
        drop(placed_b);

        let release_b = unsafe {
            let vtbl = *(res_b as *const *const *const ());
            std::mem::transmute::<*const (), extern "system" fn(crate::object::This) -> u32>(*vtbl.add(2))
        };
        assert_eq!(release_b(res_b), 0);
    }

    #[test]
    fn dynamic_placed_publication_race_requeues_stale_owner() {
        let Some(heap) = HostHeap::new(false, 64) else { return };
        let backing = vec![0u8; 64 << 10];
        let addr = backing.as_ptr() as u64;
        let res_words = [3u32, 0, 0, 0, 64, 0, 64, 0x1_0001, 28, 1, 0, 0, 4, 0];
        let mut view_words = [0u32; 16];
        view_words[0] = 28;
        view_words[1] = 4; // Texture2D UAV
        view_words[10] = addr as u32;
        view_words[11] = (addr >> 32) as u32;
        // SAFETY: 56-byte ResourceDesc layout.
        let rdesc: crate::resource::ResourceDesc =
            unsafe { res_words.as_ptr().cast::<crate::resource::ResourceDesc>().read_unaligned() };
        let release = |res: crate::object::This| unsafe {
            let vtbl = *(res as *const *const *const ());
            std::mem::transmute::<*const (), extern "system" fn(crate::object::This) -> u32>(*vtbl.add(2))(res)
        };

        // 1) Create A and pin it in thread 1 (capturing `resolved_gen == g0`) BEFORE publishing.
        let mut res_a = std::ptr::null();
        assert_eq!(
            crate::resource::create_at_address(
                std::ptr::null(),
                addr,
                &rdesc,
                0,
                std::ptr::null(),
                &crate::resource::PAGEABLE_IIDS[0],
                &mut res_a
            ),
            crate::object::S_OK
        );
        let mut slot_mem = [0u32; 4];
        let handle = slot_mem.as_mut_ptr() as u64;
        crate::descriptor::create_placed_uav(
            std::ptr::null(),
            res_words.as_ptr().cast(),
            view_words.as_ptr().cast(),
            handle,
        );
        let (key, view_a) = view_with_key(handle).expect("record A");
        let placed_a = placed_texture(&view_a).expect("pin A");
        assert_eq!(placed_a.owner_id(), crate::resource::resource_id(res_a));

        // 2) Thread 2 releases title ref on A and creates B, advancing ALL_TARGET_CHANGES.
        assert_eq!(release(res_a), 1); // still kept alive by `placed_a`
        let mut res_b = std::ptr::null();
        assert_eq!(
            crate::resource::create_at_address(
                std::ptr::null(),
                addr,
                &rdesc,
                0,
                std::ptr::null(),
                &crate::resource::PAGEABLE_IIDS[0],
                &mut res_b
            ),
            crate::object::S_OK
        );

        // 3) Thread 3 runs owner scan while `placed_owners` is still empty, advancing `last_target_gen`.
        assert_eq!(heap.texture_slots_target_owners(10), 0);
        assert!(!heap.has_stale_dynamic_placed());

        // 4) Thread 1 resumes and publishes stale A into slot 0 via `mirror_with_placed`.
        //    Even BEFORE `placed_a` is dropped, `set_placed_owner` must detect that
        //    `all_target_changes` / `last_target_gen` advanced past `placed_a.resolved_gen`
        //    and arm `placed_recheck_needed`!
        let _ = heap
            .mirror_with_placed(0, handle, true, Some((key, placed_a.resolved_gen, Some(&placed_a))))
            .expect("publish stale A");
        assert!(
            heap.placed_recheck_needed.load(Ordering::Acquire),
            "publishing a stale resolved_gen must arm placed_recheck_needed"
        );
        assert!(heap.has_stale_dynamic_placed());
        // Drain any initial queue entry so we verify the owner scan itself requeues slot 0.
        let (due_init, _) = heap.texture_slots_due(10, 64);
        for (idx, w) in due_init {
            heap.texture_slot_done(idx, w, None);
        }
        drop(placed_a); // drops final ref to A

        // 5) Next owner scan observes `placed_recheck_needed`, detects A != B, and requeues slot 0.
        assert_eq!(heap.texture_slots_target_owners(10), 1);
        let (due_after, _) = heap.texture_slots_due(10, 64);
        assert_eq!(due_after.len(), 1);
        assert_eq!(due_after[0].0, 0);
        let (_, placed_b) = heap.mirror_resolved(0, handle, true).expect("rebuild B");
        assert_eq!(placed_b.as_ref().unwrap().owner_id(), crate::resource::resource_id(res_b));
        drop(placed_b);
        assert_eq!(release(res_b), 0);
    }

    #[test]
    fn deferred_mirror_with_placed_rejects_stale_key_on_guest_overwrite() {
        use windows::core::Interface;
        let Some(heap) = HostHeap::new(false, 64) else { return };
        let backing = vec![0u8; 128 << 10];
        let addr_a = backing.as_ptr() as u64;
        let addr_b = addr_a + (64 << 10);
        let res_words = [3u32, 0, 0, 0, 64, 0, 64, 0x1_0001, 28, 1, 0, 0, 4, 0];
        // SAFETY: 56-byte ResourceDesc layout.
        let rdesc: crate::resource::ResourceDesc =
            unsafe { res_words.as_ptr().cast::<crate::resource::ResourceDesc>().read_unaligned() };
        let release = |res: crate::object::This| unsafe {
            let vtbl = *(res as *const *const *const ());
            std::mem::transmute::<*const (), extern "system" fn(crate::object::This) -> u32>(*vtbl.add(2))(res)
        };
        let mut res_a = std::ptr::null();
        let mut res_b = std::ptr::null();
        assert_eq!(
            crate::resource::create_at_address(
                std::ptr::null(),
                addr_a,
                &rdesc,
                0,
                std::ptr::null(),
                &crate::resource::PAGEABLE_IIDS[0],
                &mut res_a
            ),
            crate::object::S_OK
        );
        assert_eq!(
            crate::resource::create_at_address(
                std::ptr::null(),
                addr_b,
                &rdesc,
                0,
                std::ptr::null(),
                &crate::resource::PAGEABLE_IIDS[0],
                &mut res_b
            ),
            crate::object::S_OK
        );

        let mut view_a_words = [0u32; 16];
        view_a_words[0] = 28;
        view_a_words[1] = 4;
        view_a_words[10] = addr_a as u32;
        view_a_words[11] = (addr_a >> 32) as u32;
        let mut view_b_words = view_a_words;
        view_b_words[10] = addr_b as u32;
        view_b_words[11] = (addr_b >> 32) as u32;

        // 1) Capture `(key_a, placed_a)` for guest slot 0 pointing to A.
        let mut slot_mem = [0u32; 4];
        let handle = slot_mem.as_mut_ptr() as u64;
        crate::descriptor::create_placed_uav(
            std::ptr::null(),
            res_words.as_ptr().cast(),
            view_a_words.as_ptr().cast(),
            handle,
        );
        let (key_a, view_a) = view_with_key(handle).expect("record A");
        let placed_a = placed_texture(&view_a).expect("pin A");
        assert_eq!(placed_a.owner_id(), crate::resource::resource_id(res_a));

        // 2) Guest overwrites slot 0 with B (`key_b != key_a`) BEFORE deferred mirror runs.
        crate::descriptor::create_placed_uav(
            std::ptr::null(),
            res_words.as_ptr().cast(),
            view_b_words.as_ptr().cast(),
            handle,
        );
        let (key_b, _) = view_with_key(handle).expect("record B");
        assert_ne!(key_a, key_b);

        // 3) `mirror_with_placed` must detect `key_a != key_b`, ignore stale `placed_a`,
        //    resolve B fresh, build the slot from B, and return B's `PlacedTexture`.
        let (_, returned_placed) = heap
            .mirror_with_placed(0, handle, true, Some((key_a, placed_a.resolved_gen, Some(&placed_a))))
            .expect("mirror B despite stale supplied key_a");
        let fresh_b = returned_placed.expect("freshly resolved placed B");
        assert_eq!(fresh_b.owner_id(), crate::resource::resource_id(res_b));
        assert_eq!(heap.placed_owner(0).0, crate::resource::resource_id(res_b));
        let host_kept = heap.kept_in(0).expect("kept host resource for B");
        let host_b = crate::resource::made_copy(res_b).expect("host texture B");
        assert_eq!(host_kept.as_raw(), host_b.texture.as_raw());

        drop(fresh_b);
        drop(placed_a);
        assert_eq!(release(res_a), 0);
        assert_eq!(release(res_b), 0);
    }

    #[test]
    fn placed_srv_with_uav_flag_blocks_synthetic_fallback_over_incompatible_real_owner() {
        let backing = vec![0u8; 256 << 10];
        let addr = backing.as_ptr() as u64;
        // Real title texture at `addr` is 128x128 with UAV flag.
        let real_words = [3u32, 0, 0, 0, 128, 0, 128, 0x1_0001, 28, 1, 0, 0, 4, 0];
        // SAFETY: 56-byte ResourceDesc layout.
        let real_desc: crate::resource::ResourceDesc =
            unsafe { real_words.as_ptr().cast::<crate::resource::ResourceDesc>().read_unaligned() };
        let mut real = std::ptr::null();
        assert_eq!(
            crate::resource::create_at_address(
                std::ptr::null(),
                addr,
                &real_desc,
                0,
                std::ptr::null(),
                &crate::resource::PAGEABLE_IIDS[0],
                &mut real
            ),
            crate::object::S_OK
        );

        // Placed SRV (`unordered == false`) at the SAME `addr` with UAV flag (4)
        // but incompatible shape (64x64) MUST return `None` instead of falling
        // through to a synthetic texture shadowing the real 128x128 title owner.
        let incompat_sr_words = [3u32, 0, 0, 0, 64, 0, 64, 0x1_0001, 28, 1, 0, 0, 4, 0];
        let mut view_words = [0u32; 16];
        view_words[0] = 28;
        view_words[1] = 4; // Texture2D SRV
        view_words[2] = 5768; // D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING
        view_words[5] = 1; // MipLevels
        view_words[10] = addr as u32;
        view_words[11] = (addr >> 32) as u32;
        let view = View::Placed(false, incompat_sr_words, view_words);
        assert!(
            placed_texture(&view).is_none(),
            "incompatible live real title owner must block synthetic fallback for placed SRV with UAV flag"
        );

        let release = unsafe {
            let vtbl = *(real as *const *const *const ());
            std::mem::transmute::<*const (), extern "system" fn(crate::object::This) -> u32>(*vtbl.add(2))
        };
        assert_eq!(release(real), 0);
    }

    #[test]
    fn deferred_mirror_with_stale_none_recovers_when_real_owner_arrives() {
        use windows::core::Interface;
        let Some(heap) = HostHeap::new(false, 64) else { return };
        let backing = vec![0u8; 256 << 10];
        let addr = backing.as_ptr() as u64;
        // Start with an incompatible real 128x128 UAV owner so a 64x64 placed SRV
        // with UAV flag resolves to `None` at `g0`.
        let incompat_words = [3u32, 0, 0, 0, 128, 0, 128, 0x1_0001, 28, 1, 0, 0, 4, 0];
        let compat_words = [3u32, 0, 0, 0, 64, 0, 64, 0x1_0001, 28, 1, 0, 0, 4, 0];
        let incompat_desc: crate::resource::ResourceDesc = unsafe {
            incompat_words
                .as_ptr()
                .cast::<crate::resource::ResourceDesc>()
                .read_unaligned()
        };
        let compat_desc: crate::resource::ResourceDesc = unsafe {
            compat_words
                .as_ptr()
                .cast::<crate::resource::ResourceDesc>()
                .read_unaligned()
        };
        let release = |res: crate::object::This| unsafe {
            let vtbl = *(res as *const *const *const ());
            std::mem::transmute::<*const (), extern "system" fn(crate::object::This) -> u32>(*vtbl.add(2))(res)
        };
        let mut res_incompat = std::ptr::null();
        assert_eq!(
            crate::resource::create_at_address(
                std::ptr::null(),
                addr,
                &incompat_desc,
                0,
                std::ptr::null(),
                &crate::resource::PAGEABLE_IIDS[0],
                &mut res_incompat
            ),
            crate::object::S_OK
        );

        let mut view_words = [0u32; 16];
        view_words[0] = 28;
        view_words[1] = 4; // Texture2D SRV
        view_words[2] = 5768; // D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING
        view_words[5] = 1; // MipLevels
        view_words[10] = addr as u32;
        view_words[11] = (addr >> 32) as u32;
        let mut slot_mem = [0u32; 4];
        let handle = slot_mem.as_mut_ptr() as u64;
        crate::descriptor::create_placed_srv(
            std::ptr::null(),
            compat_words.as_ptr().cast(),
            view_words.as_ptr().cast(),
            handle,
        );

        // 1) Pre-mirror capture resolves to `None` at `g0`.
        let (key, view) = view_with_key(handle).expect("record placed SRV");
        let (g0, placed_none) = placed_texture_with_gen(&view);
        assert!(placed_none.is_none());

        // 2) Incompatible owner is freed and compatible real owner B arrives at `g1`.
        assert_eq!(release(res_incompat), 0);
        let mut res_b = std::ptr::null();
        assert_eq!(
            crate::resource::create_at_address(
                std::ptr::null(),
                addr,
                &compat_desc,
                0,
                std::ptr::null(),
                &crate::resource::PAGEABLE_IIDS[0],
                &mut res_b
            ),
            crate::object::S_OK
        );

        // 3) Owner scan runs while `placed_owners` is still empty, advancing `last_target_gen` to `g1`.
        assert_eq!(heap.texture_slots_target_owners(10), 0);
        assert!(!heap.has_stale_dynamic_placed());

        // 4) Deferred `mirror_with_placed` runs with stale `(key, g0, None)`:
        //    must re-resolve fresh and bind B immediately (never leaving slot 0 permanently null).
        let (_, fresh) = heap
            .mirror_with_placed(0, handle, true, Some((key, g0, None)))
            .expect("mirror recovers B despite stale None");
        let fresh_b = fresh.expect("freshly resolved B");
        assert_eq!(fresh_b.owner_id(), crate::resource::resource_id(res_b));
        assert_eq!(heap.placed_owner(0).0, crate::resource::resource_id(res_b));
        let host_kept = heap.kept_in(0).expect("slot 0 bound to B");
        let host_b = crate::resource::made_copy(res_b).expect("host texture B");
        assert_eq!(host_kept.as_raw(), host_b.texture.as_raw());

        drop(fresh_b);
        assert_eq!(release(res_b), 0);
    }

    #[test]
    fn placed_srv_with_uav_flag_blocks_synthetic_fallback_over_flag_zero_real_texture() {
        let backing = vec![0u8; 256 << 10];
        let addr = backing.as_ptr() as u64;
        // Real plain title texture (`flags == 0`, excluded from `TARGETS_AT`) at `addr`.
        let plain_words = [3u32, 0, 0, 0, 64, 0, 64, 0x1_0001, 28, 1, 0, 0, 0, 0];
        let plain_desc: crate::resource::ResourceDesc = unsafe {
            plain_words
                .as_ptr()
                .cast::<crate::resource::ResourceDesc>()
                .read_unaligned()
        };
        let mut real_plain = std::ptr::null();
        assert_eq!(
            crate::resource::create_at_address(
                std::ptr::null(),
                addr,
                &plain_desc,
                0,
                std::ptr::null(),
                &crate::resource::PAGEABLE_IIDS[0],
                &mut real_plain
            ),
            crate::object::S_OK
        );

        // UAV-flagged (`flags == 4`) placed SRV at the exact same `addr` MUST see
        // `IncompatibleRealOwner` and return `None` from `placed_texture()` without
        // creating a synthetic texture shadowing `real_plain`.
        let uav_sr_words = [3u32, 0, 0, 0, 64, 0, 64, 0x1_0001, 28, 1, 0, 0, 4, 0];
        let mut view_words = [0u32; 16];
        view_words[0] = 28;
        view_words[1] = 4; // Texture2D SRV
        view_words[2] = 5768; // D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING
        view_words[5] = 1; // MipLevels
        view_words[10] = addr as u32;
        view_words[11] = (addr >> 32) as u32;
        let view = View::Placed(false, uav_sr_words, view_words);
        assert!(matches!(
            crate::resource::writable_texture_at(addr, &uav_sr_words),
            crate::resource::WritableTextureAt::IncompatibleRealOwner
        ));
        assert!(
            placed_texture(&view).is_none(),
            "live flag-0 real title texture must block synthetic fallback for UAV-flagged placed SRV"
        );

        // Once the flag-0 real title texture is released, resolution cleanly succeeds again.
        let release = unsafe {
            let vtbl = *(real_plain as *const *const *const ());
            std::mem::transmute::<*const (), extern "system" fn(crate::object::This) -> u32>(*vtbl.add(2))
        };
        assert_eq!(release(real_plain), 0);
        assert!(
            placed_texture(&view).is_some(),
            "after releasing flag-0 real title texture, placed SRV resolution must succeed"
        );
    }

    #[test]
    fn uav_drain_in_progress_blocks_bake_covered() {
        let Some(heap) = HostHeap::new(false, 16) else { return };
        let pass = heap.begin_texture_pass();
        let rev = pass.revision.expect("sole texture pass");
        heap.finish_bake(rev, 500);
        drop(pass);
        // Simulate an in-progress UAV drain before `slots_uav_written` finishes.
        heap.uav_draining.fetch_add(1, Ordering::AcqRel);
        assert!(heap.has_unread_uav_written());
        assert!(
            !heap.bake_covered(500),
            "concurrent bake must not certify while a UAV drain is in progress"
        );
        heap.uav_draining.fetch_sub(1, Ordering::Release);
        heap.uav_cursor
            .store(UAV_SEQ.load(Ordering::Acquire), Ordering::Release);
        assert!(!heap.has_unread_uav_written());
    }

    #[test]
    fn uav_writes_make_their_texture_slots_due_a_few_a_frame() {
        let total = (UAV_DUE_A_FRAME * 2 + 6) as u32;
        let Some(heap) = HostHeap::new(false, total + 8) else { return };
        {
            let mut placed = heap.placed.lock().unwrap();
            for slot in 0..total {
                placed.note(slot, 0x1000_0000, 0x1001_0000, true);
            }
        }
        let queued = heap.slots_uav_written(&[(0x1000_0000, 0x1001_0000)], 100);
        assert_eq!(queued, total as usize);
        let (due_next, _) = heap.texture_slots_due(101, 256);
        assert_eq!(due_next.len(), UAV_DUE_A_FRAME as usize);
        let (due_after, _) = heap.texture_slots_due(102, 256);
        assert_eq!(due_after.len(), (UAV_DUE_A_FRAME * 2) as usize);
        let (due_all, _) = heap.texture_slots_due(103, 256);
        assert_eq!(due_all.len(), total as usize);
    }

    #[test]
    fn srv_and_uav_bounds_clamp_excess_mips_and_reject_invalid_views() {
        let own_2d = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
            Width: 64,
            Height: 64,
            DepthOrArraySize: 4,
            MipLevels: 7,
            Format: DXGI_FORMAT_R8G8B8A8_UNORM,
            SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            ..Default::default()
        };
        // Exact Run 26 D3D12 Debug Layer failure: MostDetailedMip=0, MipLevels=15 on 7-mip texture.
        // Negative/NaN ResourceMinLODClamp must be preserved unchanged.
        let mut srv = D3D12_SHADER_RESOURCE_VIEW_DESC {
            Format: DXGI_FORMAT_R8G8B8A8_UNORM,
            ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2D,
            Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
            Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                Texture2D: D3D12_TEX2D_SRV {
                    MostDetailedMip: 0,
                    MipLevels: 15,
                    PlaneSlice: 0,
                    ResourceMinLODClamp: f32::NAN,
                },
            },
        };
        assert!(clamp_srv_desc_for_resource(&own_2d, &mut srv, 1));
        let t = unsafe { srv.Anonymous.Texture2D };
        assert_eq!((t.MostDetailedMip, t.MipLevels), (0, 7));
        assert!(t.ResourceMinLODClamp.is_nan());

        // Single-plane resource rejects PlaneSlice >= 1 for both SRV and UAV (D3D12 error 344).
        let mut bad_plane_srv = srv;
        bad_plane_srv.Anonymous.Texture2D = D3D12_TEX2D_SRV {
            MostDetailedMip: 0,
            MipLevels: 1,
            PlaneSlice: 1,
            ResourceMinLODClamp: 0.0,
        };
        assert!(!clamp_srv_desc_for_resource(&own_2d, &mut bad_plane_srv, 1));
        let mut bad_plane_uav = D3D12_UNORDERED_ACCESS_VIEW_DESC {
            Format: DXGI_FORMAT_R8G8B8A8_UNORM,
            ViewDimension: D3D12_UAV_DIMENSION_TEXTURE2D,
            Anonymous: D3D12_UNORDERED_ACCESS_VIEW_DESC_0 {
                Texture2D: D3D12_TEX2D_UAV { MipSlice: 0, PlaneSlice: 1 },
            },
        };
        assert!(!clamp_uav_desc_for_resource(&own_2d, &mut bad_plane_uav, 1, false));

        // Planar depth/stencil: R24G8_TYPELESS requires matching view format for plane 0 vs plane 1.
        let own_ds = D3D12_RESOURCE_DESC {
            Format: DXGI_FORMAT_R24G8_TYPELESS,
            MipLevels: 1,
            ..own_2d
        };
        let mut ds_depth_ok = D3D12_SHADER_RESOURCE_VIEW_DESC {
            Format: DXGI_FORMAT_R24_UNORM_X8_TYPELESS,
            ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2D,
            Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
            Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                Texture2D: D3D12_TEX2D_SRV { MostDetailedMip: 0, MipLevels: u32::MAX, PlaneSlice: 0, ResourceMinLODClamp: 0.0 },
            },
        };
        assert!(clamp_srv_desc_for_resource(&own_ds, &mut ds_depth_ok, 2));
        assert_eq!(unsafe { ds_depth_ok.Anonymous.Texture2D.MipLevels }, u32::MAX);
        let mut ds_depth_wrong_plane = ds_depth_ok;
        ds_depth_wrong_plane.Anonymous.Texture2D = D3D12_TEX2D_SRV { MostDetailedMip: 0, MipLevels: 1, PlaneSlice: 1, ResourceMinLODClamp: 0.0 };
        assert!(!clamp_srv_desc_for_resource(&own_ds, &mut ds_depth_wrong_plane, 2));
        let mut ds_stencil_ok = D3D12_SHADER_RESOURCE_VIEW_DESC {
            Format: DXGI_FORMAT_X24_TYPELESS_G8_UINT,
            ..ds_depth_wrong_plane
        };
        assert!(clamp_srv_desc_for_resource(&own_ds, &mut ds_stencil_ok, 2));
        let mut ds_stencil_wrong_plane = ds_stencil_ok;
        ds_stencil_wrong_plane.Anonymous.Texture2D = D3D12_TEX2D_SRV { MostDetailedMip: 0, MipLevels: 1, PlaneSlice: 0, ResourceMinLODClamp: 0.0 };
        assert!(!clamp_srv_desc_for_resource(&own_ds, &mut ds_stencil_wrong_plane, 2));

        // Planar YUV: NV12 plane 0 = R8_UNORM/R8_UINT, plane 1 = R8G8_UNORM/R8G8_UINT;
        // SNORM/SINT rejected; P010/P016 chroma UAV additionally allows R32_UINT.
        let own_nv12 = D3D12_RESOURCE_DESC { Format: DXGI_FORMAT_NV12, MipLevels: 1, ..own_2d };
        let mut nv12_p0 = D3D12_SHADER_RESOURCE_VIEW_DESC {
            Format: DXGI_FORMAT_R8_UNORM,
            ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2D,
            Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
            Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                Texture2D: D3D12_TEX2D_SRV { MostDetailedMip: 0, MipLevels: 1, PlaneSlice: 0, ResourceMinLODClamp: 0.0 },
            },
        };
        assert!(clamp_srv_desc_for_resource(&own_nv12, &mut nv12_p0, 2));
        let mut nv12_snorm = D3D12_SHADER_RESOURCE_VIEW_DESC { Format: DXGI_FORMAT_R8_SNORM, ..nv12_p0 };
        assert!(!clamp_srv_desc_for_resource(&own_nv12, &mut nv12_snorm, 2));
        let mut nv12_bad = nv12_p0;
        nv12_bad.Anonymous.Texture2D = D3D12_TEX2D_SRV { MostDetailedMip: 0, MipLevels: 1, PlaneSlice: 1, ResourceMinLODClamp: 0.0 };
        assert!(!clamp_srv_desc_for_resource(&own_nv12, &mut nv12_bad, 2));
        nv12_bad.Format = DXGI_FORMAT_R8G8_UNORM;
        assert!(clamp_srv_desc_for_resource(&own_nv12, &mut nv12_bad, 2));

        let own_p010 = D3D12_RESOURCE_DESC { Format: DXGI_FORMAT_P010, MipLevels: 1, ..own_2d };
        let mut p010_chroma_uav = D3D12_UNORDERED_ACCESS_VIEW_DESC {
            Format: DXGI_FORMAT_R32_UINT,
            ViewDimension: D3D12_UAV_DIMENSION_TEXTURE2D,
            Anonymous: D3D12_UNORDERED_ACCESS_VIEW_DESC_0 {
                Texture2D: D3D12_TEX2D_UAV { MipSlice: 0, PlaneSlice: 1 },
            },
        };
        assert!(clamp_uav_desc_for_resource(&own_p010, &mut p010_chroma_uav, 2, false));
        let mut p010_chroma_srv = D3D12_SHADER_RESOURCE_VIEW_DESC {
            Format: DXGI_FORMAT_R32_UINT,
            ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2D,
            Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
            Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                Texture2D: D3D12_TEX2D_SRV { MostDetailedMip: 0, MipLevels: 1, PlaneSlice: 1, ResourceMinLODClamp: 0.0 },
            },
        };
        assert!(!clamp_srv_desc_for_resource(&own_p010, &mut p010_chroma_srv, 2));

        // Typeless view format rejected even on matching typeless resource.
        let own_typeless = D3D12_RESOURCE_DESC { Format: DXGI_FORMAT_R8G8B8A8_TYPELESS, ..own_2d };
        let mut srv_typeless = D3D12_SHADER_RESOURCE_VIEW_DESC { Format: DXGI_FORMAT_R8G8B8A8_TYPELESS, ..srv };
        assert!(!clamp_srv_desc_for_resource(&own_typeless, &mut srv_typeless, 1));

        // Out-of-range MostDetailedMip and MipLevels=0 are rejected.
        let mut bad_most = srv;
        bad_most.Anonymous.Texture2D = D3D12_TEX2D_SRV { MostDetailedMip: 7, MipLevels: 1, ..Default::default() };
        assert!(!clamp_srv_desc_for_resource(&own_2d, &mut bad_most, 1));
        let mut bad_zero = srv;
        bad_zero.Anonymous.Texture2D = D3D12_TEX2D_SRV { MostDetailedMip: 0, MipLevels: 0, ..Default::default() };
        assert!(!clamp_srv_desc_for_resource(&own_2d, &mut bad_zero, 1));

        // Texture2DArray clamps both MipLevels and ArraySize while preserving negative ResourceMinLODClamp.
        let mut srv_arr = D3D12_SHADER_RESOURCE_VIEW_DESC {
            Format: DXGI_FORMAT_R8G8B8A8_UNORM,
            ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2DARRAY,
            Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
            Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                Texture2DArray: D3D12_TEX2D_ARRAY_SRV {
                    MostDetailedMip: 2,
                    MipLevels: 15,
                    FirstArraySlice: 1,
                    ArraySize: 10,
                    PlaneSlice: 0,
                    ResourceMinLODClamp: -1.0,
                },
            },
        };
        assert!(clamp_srv_desc_for_resource(&own_2d, &mut srv_arr, 1));
        let ta = unsafe { srv_arr.Anonymous.Texture2DArray };
        assert_eq!((ta.MostDetailedMip, ta.MipLevels, ta.FirstArraySlice, ta.ArraySize, ta.ResourceMinLODClamp), (2, 5, 1, 3, -1.0));

        // Single-sample multi-mip 2D texture is valid for TEXTURE2DMS and TEXTURE2DMSARRAY SRVs & UAVs.
        let mut srv_ms = D3D12_SHADER_RESOURCE_VIEW_DESC {
            Format: DXGI_FORMAT_R8G8B8A8_UNORM,
            ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2DMS,
            Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
            ..Default::default()
        };
        assert!(clamp_srv_desc_for_resource(&own_2d, &mut srv_ms, 1));
        let mut srv_ms_arr = D3D12_SHADER_RESOURCE_VIEW_DESC {
            Format: DXGI_FORMAT_R8G8B8A8_UNORM,
            ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2DMSARRAY,
            Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
            Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                Texture2DMSArray: D3D12_TEX2DMS_ARRAY_SRV { FirstArraySlice: 1, ArraySize: 8 },
            },
        };
        assert!(clamp_srv_desc_for_resource(&own_2d, &mut srv_ms_arr, 1));
        assert_eq!(unsafe { srv_ms_arr.Anonymous.Texture2DMSArray.ArraySize }, 3);

        // MS UAV dimensions require `writeable_msaa = true` and match `null_uav_desc_with_caps`.
        let mut uav_ms = D3D12_UNORDERED_ACCESS_VIEW_DESC {
            Format: DXGI_FORMAT_R8G8B8A8_UNORM,
            ViewDimension: D3D12_UAV_DIMENSION_TEXTURE2DMS,
            ..Default::default()
        };
        assert!(!clamp_uav_desc_for_resource(&own_2d, &mut uav_ms, 1, false));
        assert!(clamp_uav_desc_for_resource(&own_2d, &mut uav_ms, 1, true));
        let mut uav_ms_arr = D3D12_UNORDERED_ACCESS_VIEW_DESC {
            Format: DXGI_FORMAT_R8G8B8A8_UNORM,
            ViewDimension: D3D12_UAV_DIMENSION_TEXTURE2DMSARRAY,
            Anonymous: D3D12_UNORDERED_ACCESS_VIEW_DESC_0 {
                Texture2DMSArray: D3D12_TEX2DMS_ARRAY_UAV { FirstArraySlice: 2, ArraySize: 8 },
            },
        };
        assert!(clamp_uav_desc_for_resource(&own_2d, &mut uav_ms_arr, 1, true));
        assert_eq!(unsafe { uav_ms_arr.Anonymous.Texture2DMSArray.ArraySize }, 2);
        assert_eq!(null_uav_desc_with_caps(Some(6), true).ViewDimension, D3D12_UAV_DIMENSION_TEXTURE2DMS);
        assert_eq!(null_uav_desc_with_caps(Some(7), true).ViewDimension, D3D12_UAV_DIMENSION_TEXTURE2DMSARRAY);

        // Dimension mismatch rejected.
        let mut wrong_dim = srv;
        wrong_dim.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE3D;
        assert!(!clamp_srv_desc_for_resource(&own_2d, &mut wrong_dim, 1));

        // 3D UAV clamps WSize against mip depth (`8 >> 1 == 4`), preserves `u32::MAX`, and rejects out-of-range MipSlice.
        let own_3d = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE3D,
            Width: 16,
            Height: 16,
            DepthOrArraySize: 8,
            MipLevels: 4,
            Format: DXGI_FORMAT_R8G8B8A8_UNORM,
            SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            ..Default::default()
        };
        let mut uav_3d = D3D12_UNORDERED_ACCESS_VIEW_DESC {
            Format: DXGI_FORMAT_R8G8B8A8_UNORM,
            ViewDimension: D3D12_UAV_DIMENSION_TEXTURE3D,
            Anonymous: D3D12_UNORDERED_ACCESS_VIEW_DESC_0 {
                Texture3D: D3D12_TEX3D_UAV { MipSlice: 1, FirstWSlice: 1, WSize: 16 },
            },
        };
        assert!(clamp_uav_desc_for_resource(&own_3d, &mut uav_3d, 1, false));
        let u3 = unsafe { uav_3d.Anonymous.Texture3D };
        assert_eq!((u3.MipSlice, u3.FirstWSlice, u3.WSize), (1, 1, 3));

        let mut uav_3d_max = uav_3d;
        uav_3d_max.Anonymous.Texture3D = D3D12_TEX3D_UAV { MipSlice: 1, FirstWSlice: 1, WSize: u32::MAX };
        assert!(clamp_uav_desc_for_resource(&own_3d, &mut uav_3d_max, 1, false));
        assert_eq!(unsafe { uav_3d_max.Anonymous.Texture3D.WSize }, u32::MAX);

        let mut bad_uav_mip = uav_3d;
        bad_uav_mip.Anonymous.Texture3D = D3D12_TEX3D_UAV { MipSlice: 4, FirstWSlice: 0, WSize: 1 };
        assert!(!clamp_uav_desc_for_resource(&own_3d, &mut bad_uav_mip, 1, false));
    }

    #[test]
    fn srv_and_uav_bounds_match_d3d12_warp_debug_layer() {
        const CHILD_ENV: &str = "VICTORY_WARP_BOUNDS_CHILD";
        const NAME: &str = "descriptor::host::tests::srv_and_uav_bounds_match_d3d12_warp_debug_layer";
        // Run D3D12 Debug Layer activation in an isolated child process so
        // `EnableDebugLayer()` never removes a sibling test's live D3D12 device.
        if std::env::var_os(CHILD_ENV).is_none() {
            let out = std::process::Command::new(std::env::current_exe().expect("test exe"))
                .args(["--exact", NAME, "--nocapture"])
                .env(CHILD_ENV, "1")
                .output()
                .expect("spawn isolated WARP debug-layer child");
            assert!(
                out.status.success(),
                "WARP debug-layer child failed ({}):\nstdout:\n{}\nstderr:\n{}",
                out.status,
                String::from_utf8_lossy(&out.stdout),
                String::from_utf8_lossy(&out.stderr),
            );
            return;
        }

        use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_12_0;
        use windows::Win32::Graphics::Dxgi::{CreateDXGIFactory1, IDXGIAdapter, IDXGIFactory4};
        use windows::core::Interface;

        let require_debug = std::env::var_os("VICTORY_REQUIRE_D3D12_DEBUG_TESTS")
            .is_some_and(|v| v == "1")
            || std::env::var_os("VICTORY_REQUIRE_DXR_TESTS").is_some_and(|v| v == "1");

        // Isolated WARP D3D12 device with Debug Layer exercising the full
        // `fits() -> clamp_*_desc_for_resource() -> Create*View()` path.
        unsafe {
            let mut debug: Option<ID3D12Debug> = None;
            let debug_ok = D3D12GetDebugInterface(&mut debug).is_ok() && debug.is_some();
            if let Some(d) = debug {
                d.EnableDebugLayer();
            } else if require_debug {
                panic!("D3D12 Debug Layer required but unavailable");
            } else {
                eprintln!("NOTE: D3D12 Debug Layer unavailable on this host; running WARP device-removed checks only");
            }
            let factory = CreateDXGIFactory1::<IDXGIFactory4>().expect("DXGI factory");
            let adapter = factory.EnumWarpAdapter::<IDXGIAdapter>().expect("WARP adapter");
            let mut dev: Option<ID3D12Device> = None;
            D3D12CreateDevice(&adapter, D3D_FEATURE_LEVEL_12_0, &mut dev).expect("WARP D3D12 device");
            let dev = dev.expect("non-null WARP D3D12 device");
            let info: Option<ID3D12InfoQueue> = dev.cast().ok();
            if require_debug {
                assert!(
                    debug_ok && info.is_some(),
                    "ID3D12InfoQueue required when D3D12 Debug Layer verification is enforced"
                );
            }
            if let Some(iq) = &info {
                iq.ClearStoredMessages();
            }
            let heap: ID3D12DescriptorHeap = dev
                .CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                    Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                    NumDescriptors: 32,
                    ..Default::default()
                })
                .expect("WARP descriptor heap");
            let base = heap.GetCPUDescriptorHandleForHeapStart().ptr;
            let step = dev.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV) as usize;
            let slot = |i: usize| D3D12_CPU_DESCRIPTOR_HANDLE { ptr: base + i * step };
            let props = D3D12_HEAP_PROPERTIES {
                Type: D3D12_HEAP_TYPE_DEFAULT,
                ..Default::default()
            };
            let make_tex = |desc: &D3D12_RESOURCE_DESC| -> ID3D12Resource {
                let mut res: Option<ID3D12Resource> = None;
                dev.CreateCommittedResource(
                    &props,
                    D3D12_HEAP_FLAG_NONE,
                    desc,
                    D3D12_RESOURCE_STATE_COMMON,
                    None,
                    &mut res,
                )
                .expect("WARP committed texture");
                res.unwrap()
            };

            // 1. 7-mip 4-slice 2D array with UAV: clamp MipLevels=15 -> 7, preserve negative/NaN LOD clamp,
            //    reject PlaneSlice=1, accept PlaneSlice=0, AND accept 1-sample multi-mip MS SRV & UAV.
            let tex_7mip_desc = D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
                Width: 64,
                Height: 64,
                DepthOrArraySize: 4,
                MipLevels: 7,
                Format: DXGI_FORMAT_R8G8B8A8_UNORM,
                SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                Flags: D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                ..Default::default()
            };
            let tex_7mip = make_tex(&tex_7mip_desc);
            let planes_rgba = device_format_plane_count(&dev, DXGI_FORMAT_R8G8B8A8_UNORM);
            assert_eq!(planes_rgba, 1);

            let mut srv_clamped = D3D12_SHADER_RESOURCE_VIEW_DESC {
                Format: DXGI_FORMAT_R8G8B8A8_UNORM,
                ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2D,
                Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
                Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                    Texture2D: D3D12_TEX2D_SRV {
                        MostDetailedMip: 0,
                        MipLevels: 15,
                        PlaneSlice: 0,
                        ResourceMinLODClamp: -1.0,
                    },
                },
            };
            assert!(fits(&tex_7mip, Some(&mut srv_clamped.Format), false));
            assert!(clamp_srv_desc_for_resource(&tex_7mip_desc, &mut srv_clamped, planes_rgba));
            dev.CreateShaderResourceView(&tex_7mip, Some(&srv_clamped), slot(0));

            let mut srv_nan = srv_clamped;
            srv_nan.Anonymous.Texture2D = D3D12_TEX2D_SRV {
                MostDetailedMip: 0,
                MipLevels: u32::MAX,
                PlaneSlice: 0,
                ResourceMinLODClamp: f32::NAN,
            };
            assert!(fits(&tex_7mip, Some(&mut srv_nan.Format), false));
            assert!(clamp_srv_desc_for_resource(&tex_7mip_desc, &mut srv_nan, planes_rgba));
            dev.CreateShaderResourceView(&tex_7mip, Some(&srv_nan), slot(1));

            let mut uav_ok = D3D12_UNORDERED_ACCESS_VIEW_DESC {
                Format: DXGI_FORMAT_R8G8B8A8_UNORM,
                ViewDimension: D3D12_UAV_DIMENSION_TEXTURE2DARRAY,
                Anonymous: D3D12_UNORDERED_ACCESS_VIEW_DESC_0 {
                    Texture2DArray: D3D12_TEX2D_ARRAY_UAV {
                        MipSlice: 2,
                        FirstArraySlice: 1,
                        ArraySize: u32::MAX,
                        PlaneSlice: 0,
                    },
                },
            };
            assert!(fits(&tex_7mip, Some(&mut uav_ok.Format), true));
            assert!(clamp_uav_desc_for_resource(&tex_7mip_desc, &mut uav_ok, planes_rgba, false));
            dev.CreateUnorderedAccessView(&tex_7mip, None, Some(&uav_ok), slot(2));

            // 2. Typeless resource: typeless view format MUST be rejected by both `fits` and `clamp`,
            //    while compatible typed view format succeeds end-to-end.
            let tex_typeless_desc = D3D12_RESOURCE_DESC {
                Format: DXGI_FORMAT_R8G8B8A8_TYPELESS,
                ..tex_7mip_desc
            };
            let tex_typeless = make_tex(&tex_typeless_desc);
            let mut srv_typeless = D3D12_SHADER_RESOURCE_VIEW_DESC {
                Format: DXGI_FORMAT_R8G8B8A8_TYPELESS,
                ..srv_clamped
            };
            assert!(!fits(&tex_typeless, Some(&mut srv_typeless.Format), false));
            assert!(!clamp_srv_desc_for_resource(&tex_typeless_desc, &mut srv_typeless, planes_rgba));
            let mut srv_typed_on_typeless = D3D12_SHADER_RESOURCE_VIEW_DESC {
                Format: DXGI_FORMAT_R8G8B8A8_UNORM,
                ..srv_clamped
            };
            assert!(fits(&tex_typeless, Some(&mut srv_typed_on_typeless.Format), false));
            assert!(clamp_srv_desc_for_resource(&tex_typeless_desc, &mut srv_typed_on_typeless, planes_rgba));
            dev.CreateShaderResourceView(&tex_typeless, Some(&srv_typed_on_typeless), slot(3));

            // 3. Depth/stencil R24G8_TYPELESS: depth (plane 0) and stencil (plane 1) SRVs end-to-end.
            let tex_ds_desc = D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
                Width: 32,
                Height: 32,
                DepthOrArraySize: 1,
                MipLevels: 1,
                Format: DXGI_FORMAT_R24G8_TYPELESS,
                SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                Flags: D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL,
                ..Default::default()
            };
            let tex_ds = make_tex(&tex_ds_desc);
            let planes_ds = device_format_plane_count(&dev, DXGI_FORMAT_R24G8_TYPELESS);
            assert_eq!(planes_ds, 2);
            let mut srv_depth = D3D12_SHADER_RESOURCE_VIEW_DESC {
                Format: DXGI_FORMAT_R24_UNORM_X8_TYPELESS,
                ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2D,
                Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
                Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                    Texture2D: D3D12_TEX2D_SRV { MostDetailedMip: 0, MipLevels: 1, PlaneSlice: 0, ResourceMinLODClamp: 0.0 },
                },
            };
            assert!(fits(&tex_ds, Some(&mut srv_depth.Format), false));
            assert!(clamp_srv_desc_for_resource(&tex_ds_desc, &mut srv_depth, planes_ds));
            dev.CreateShaderResourceView(&tex_ds, Some(&srv_depth), slot(4));
            let mut srv_stencil = D3D12_SHADER_RESOURCE_VIEW_DESC {
                Format: DXGI_FORMAT_X24_TYPELESS_G8_UINT,
                ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2D,
                Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
                Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                    Texture2D: D3D12_TEX2D_SRV { MostDetailedMip: 0, MipLevels: 1, PlaneSlice: 1, ResourceMinLODClamp: 0.0 },
                },
            };
            assert!(fits(&tex_ds, Some(&mut srv_stencil.Format), false));
            assert!(clamp_srv_desc_for_resource(&tex_ds_desc, &mut srv_stencil, planes_ds));
            dev.CreateShaderResourceView(&tex_ds, Some(&srv_stencil), slot(5));

            // 4. Planar YUV NV12 end-to-end (`fits -> clamp -> CreateShaderResourceView`):
            //    R8_UNORM (plane 0) and R8G8_UNORM (plane 1) pass; R8_SNORM is rejected.
            let tex_nv12_desc = D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
                Width: 32,
                Height: 32,
                DepthOrArraySize: 1,
                MipLevels: 1,
                Format: DXGI_FORMAT_NV12,
                SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                Flags: D3D12_RESOURCE_FLAG_NONE,
                ..Default::default()
            };
            let tex_nv12 = make_tex(&tex_nv12_desc);
            let planes_nv12 = device_format_plane_count(&dev, DXGI_FORMAT_NV12);
            assert_eq!(planes_nv12, 2);
            let mut srv_nv12_y = D3D12_SHADER_RESOURCE_VIEW_DESC {
                Format: DXGI_FORMAT_R8_UNORM,
                ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2D,
                Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
                Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                    Texture2D: D3D12_TEX2D_SRV { MostDetailedMip: 0, MipLevels: 1, PlaneSlice: 0, ResourceMinLODClamp: 0.0 },
                },
            };
            assert!(fits(&tex_nv12, Some(&mut srv_nv12_y.Format), false));
            assert_eq!(srv_nv12_y.Format, DXGI_FORMAT_R8_UNORM);
            assert!(clamp_srv_desc_for_resource(&tex_nv12_desc, &mut srv_nv12_y, planes_nv12));
            dev.CreateShaderResourceView(&tex_nv12, Some(&srv_nv12_y), slot(6));
            let mut srv_nv12_uv = D3D12_SHADER_RESOURCE_VIEW_DESC {
                Format: DXGI_FORMAT_R8G8_UNORM,
                ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2D,
                Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
                Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                    Texture2D: D3D12_TEX2D_SRV { MostDetailedMip: 0, MipLevels: 1, PlaneSlice: 1, ResourceMinLODClamp: 0.0 },
                },
            };
            assert!(fits(&tex_nv12, Some(&mut srv_nv12_uv.Format), false));
            assert_eq!(srv_nv12_uv.Format, DXGI_FORMAT_R8G8_UNORM);
            assert!(clamp_srv_desc_for_resource(&tex_nv12_desc, &mut srv_nv12_uv, planes_nv12));
            dev.CreateShaderResourceView(&tex_nv12, Some(&srv_nv12_uv), slot(7));
            let mut srv_nv12_snorm = D3D12_SHADER_RESOURCE_VIEW_DESC { Format: DXGI_FORMAT_R8_SNORM, ..srv_nv12_y };
            assert!(!fits(&tex_nv12, Some(&mut srv_nv12_snorm.Format), false));
            assert!(!clamp_srv_desc_for_resource(&tex_nv12_desc, &mut srv_nv12_snorm, planes_nv12));

            // 5. Single-sample multi-mip 7-mip 4-slice texture viewed as TEXTURE2DMS & TEXTURE2DMSARRAY SRV/UAV.
            let mut srv_ms = D3D12_SHADER_RESOURCE_VIEW_DESC {
                Format: DXGI_FORMAT_R8G8B8A8_UNORM,
                ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2DMS,
                Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
                ..Default::default()
            };
            assert!(fits(&tex_7mip, Some(&mut srv_ms.Format), false));
            assert!(clamp_srv_desc_for_resource(&tex_7mip_desc, &mut srv_ms, planes_rgba));
            dev.CreateShaderResourceView(&tex_7mip, Some(&srv_ms), slot(8));
            let mut srv_ms_arr = D3D12_SHADER_RESOURCE_VIEW_DESC {
                Format: DXGI_FORMAT_R8G8B8A8_UNORM,
                ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2DMSARRAY,
                Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
                Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                    Texture2DMSArray: D3D12_TEX2DMS_ARRAY_SRV { FirstArraySlice: 0, ArraySize: u32::MAX },
                },
            };
            assert!(fits(&tex_7mip, Some(&mut srv_ms_arr.Format), false));
            assert!(clamp_srv_desc_for_resource(&tex_7mip_desc, &mut srv_ms_arr, planes_rgba));
            dev.CreateShaderResourceView(&tex_7mip, Some(&srv_ms_arr), slot(9));

            let writeable_msaa = device_writeable_msaa(&dev);
            let null_ms = null_uav_desc_with_caps(Some(6), writeable_msaa);
            dev.CreateUnorderedAccessView(None::<&ID3D12Resource>, None::<&ID3D12Resource>, Some(&null_ms), slot(10));
            let null_ms_arr = null_uav_desc_with_caps(Some(7), writeable_msaa);
            dev.CreateUnorderedAccessView(None::<&ID3D12Resource>, None::<&ID3D12Resource>, Some(&null_ms_arr), slot(11));
            if writeable_msaa {
                let mut uav_ms = D3D12_UNORDERED_ACCESS_VIEW_DESC {
                    Format: DXGI_FORMAT_R8G8B8A8_UNORM,
                    ViewDimension: D3D12_UAV_DIMENSION_TEXTURE2DMS,
                    ..Default::default()
                };
                assert!(fits(&tex_7mip, Some(&mut uav_ms.Format), true));
                assert!(clamp_uav_desc_for_resource(&tex_7mip_desc, &mut uav_ms, planes_rgba, true));
                dev.CreateUnorderedAccessView(&tex_7mip, None, Some(&uav_ms), slot(12));
                let mut uav_ms_arr = D3D12_UNORDERED_ACCESS_VIEW_DESC {
                    Format: DXGI_FORMAT_R8G8B8A8_UNORM,
                    ViewDimension: D3D12_UAV_DIMENSION_TEXTURE2DMSARRAY,
                    Anonymous: D3D12_UNORDERED_ACCESS_VIEW_DESC_0 {
                        Texture2DMSArray: D3D12_TEX2DMS_ARRAY_UAV { FirstArraySlice: 1, ArraySize: 8 },
                    },
                };
                assert!(fits(&tex_7mip, Some(&mut uav_ms_arr.Format), true));
                assert!(clamp_uav_desc_for_resource(&tex_7mip_desc, &mut uav_ms_arr, planes_rgba, true));
                dev.CreateUnorderedAccessView(&tex_7mip, None, Some(&uav_ms_arr), slot(13));
            }

            assert!(dev.GetDeviceRemovedReason().is_ok(), "WARP device must not be removed");
            if let Some(iq) = &info {
                let mut errs = Vec::new();
                for i in 0..iq.GetNumStoredMessages() {
                    let mut len = 0usize;
                    let _ = iq.GetMessage(i, None, &mut len);
                    if len == 0 {
                        continue;
                    }
                    let mut buf = vec![0u8; len];
                    if iq.GetMessage(i, Some(buf.as_mut_ptr().cast()), &mut len).is_ok() {
                        let msg = &*buf.as_ptr().cast::<D3D12_MESSAGE>();
                        if msg.Severity == D3D12_MESSAGE_SEVERITY_ERROR
                            || msg.Severity == D3D12_MESSAGE_SEVERITY_CORRUPTION
                        {
                            let desc = std::slice::from_raw_parts(msg.pDescription, msg.DescriptionByteLength);
                            errs.push(String::from_utf8_lossy(desc).into_owned());
                        }
                    }
                }
                assert!(errs.is_empty(), "WARP D3D12 InfoQueue errors: {errs:?}");
            }
        }
    }
}

/// What is known of a record a list's tables hold, in that recording of the
/// list ([`HostHeap::make_listed`]).
#[derive(Clone, Copy)]
pub(crate) struct Listed {
    pub(crate) kind: Option<victory_shader::native::ResourceType>,
    pub(crate) texture: bool,
    /// Read at this epoch of the list ([`crate::host::HostList::epoch`]) with
    /// this depth target tested: up to date; and whether it is that target's.
    pub(crate) ready: (u64, usize, bool),
    /// The title's slot last noted for it, and whether the draw read it.
    pub(crate) tabled: Option<(u32, bool)>,
}

struct ListView {
    view: View,
    /// Its descriptor's slot in [`ListViews`]; its raw twin's is the next.
    at: u32,
    twin: bool,
    stride_word: u32,
    kept: Kept,
    listed: Listed,
}

/// The descriptors a thread made for one recording of a list, by the record
/// they were made from, in a heap the card does not see.
///
/// The title's culling and lighting dispatches each bind a table of their own
/// that differs in one constant buffer, and the same thirty views beside it:
/// each was found, checked, made and brought up to date anew for every
/// dispatch, some 2,500 indirect dispatches in the one recording job the
/// frame waits for last (timeline waits-2: 24 of 65 ms). Here a list makes a
/// view once, as it makes a table once (`crate::draw::State::tables`), and
/// copies its descriptor.
struct ListViews {
    recording: u64,
    twins: bool,
    heap: ID3D12DescriptorHeap,
    step: usize,
    start: usize,
    made: std::collections::HashMap<[u32; 3], ListView, std::hash::BuildHasherDefault<crate::host::AddressHash>>,
}

impl ListViews {
    const SLOTS: u32 = 8192;

    fn new() -> Option<ListViews> {
        let host = host()?;
        // SAFETY: plain object creation.
        unsafe {
            let heap: ID3D12DescriptorHeap = host
                .device
                .CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                    Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                    NumDescriptors: Self::SLOTS,
                    ..Default::default()
                })
                .ok()?;
            let step = host.device.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV) as usize;
            let start = heap.GetCPUDescriptorHandleForHeapStart().ptr;
            Some(ListViews { recording: 0, twins: false, heap, step, start, made: Default::default() })
        }
    }

    fn slot(&self, i: u32) -> D3D12_CPU_DESCRIPTOR_HANDLE {
        debug_assert!(i < Self::SLOTS && !windows::core::Interface::as_raw(&self.heap).is_null());
        D3D12_CPU_DESCRIPTOR_HANDLE { ptr: self.start + i as usize * self.step }
    }
}

thread_local! {
    static LIST_VIEWS: std::cell::RefCell<Option<ListViews>> = const { std::cell::RefCell::new(None) };
}

/// The view the record `words` names, as this thread made it for `recording`.
pub(crate) fn listed_view(recording: u64, words: [u32; 3]) -> Option<View> {
    LIST_VIEWS.with(|views| {
        let views = views.borrow();
        let views = views.as_ref().filter(|v| v.recording == recording)?;
        views.made.get(&words).map(|m| m.view.clone())
    })
}

/// Notes what is known of the record `words` in `recording`.
pub(crate) fn note_listed(recording: u64, words: [u32; 3], listed: Listed) {
    LIST_VIEWS.with(|views| {
        if let Some(views) = views.borrow_mut().as_mut().filter(|v| v.recording == recording)
            && let Some(made) = views.made.get_mut(&words)
        {
            made.listed = listed;
        }
    });
}

/// Whether a list makes a view its tables hold once (`VICTORY_LIST_VIEWS=0`:
/// for every table).
pub(crate) fn list_views() -> bool {
    crate::toggled!("VICTORY_LIST_VIEWS")
}
