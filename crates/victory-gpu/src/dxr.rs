//! Opt-in, submission-ordered host DXR 1.1 sidecar. The console builder still
//! supplies all title-visible results. No shader translation depends on this module.
//! ABI and instance patching adapted from maren-halv/VIctory-DXR @ e1740b3, GPL-3.0.
mod abi;
mod capture;
mod commands;
mod geometry;
mod memory;
#[cfg(any(test, feature = "diagnostics"))]
mod packet;
#[cfg(test)]
mod tests;

pub use abi::{GuestBatchedDesc2, GuestBuildAsDesc};
pub use capture::{Captured, capture_build, capture_rehydrate};
pub(crate) use geometry::Work;
pub(crate) use memory::{Budget, slab_headroom};

use capture::{Checked, Op};
use commands::{Commands, PATCH_REPORT, Patcher, Table, barrier};
use geometry::Native;
use memory::{Memory, Storage};
use std::collections::{BTreeMap, HashMap, VecDeque};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering::Relaxed};
use std::sync::{Arc, Mutex, OnceLock};
use windows::Win32::Foundation::HANDLE;
use windows::Win32::Graphics::{Direct3D12::*, Dxgi::IDXGIAdapter3};
use windows::core::Interface;

pub fn requested() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_NATIVE_RT").is_some_and(|v| v == "1"))
}

fn strict_tlas() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_DXR_STRICT_TLAS").is_some_and(|v| v == "1"))
}

const AS_ALIGN: u64 = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BYTE_ALIGNMENT as u64;
const MIN_SCRATCH_BYTES: u64 = 16 << 20;
const MIN_TLAS_RESERVE_BYTES: u64 = 4 << 20;
const MIN_INST_POOL_BYTES: u64 = 64 * 1024;
const MAX_POOLED_INST_BYTES: u64 = 4 << 20;
const MAX_FREE_INST: usize = 8;
const MAX_FREE_TABLES: usize = 4;
const DEFAULT_REFIT_PERIOD: u32 = 64;

fn refit_dynamic_blas() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_NATIVE_RT_REFIT").is_some_and(|v| v == "1"))
}

fn parse_refit_period(raw: Option<&str>) -> u32 {
    raw.and_then(|v| v.trim().parse::<u32>().ok())
        .unwrap_or(DEFAULT_REFIT_PERIOD)
}

fn refit_period() -> u32 {
    static PERIOD: OnceLock<u32> = OnceLock::new();
    *PERIOD.get_or_init(|| {
        parse_refit_period(std::env::var("VICTORY_NATIVE_RT_REFIT_PERIOD").ok().as_deref())
    })
}

/// Bump-allocates 256B-aligned scratch slices across consecutive independent BLAS
/// builds within one submission so they run without per-BLAS `UAV` barriers,
/// flushing only on destination/source VA hazards, scratch exhaustion, or before
/// a dependent op (`CopyAs`, `BuildTlas`, or end of command list).
#[derive(Default)]
struct ScratchBatcher {
    buf_id: usize,
    cursor: u64,
    written_dests: Vec<u64>,
}

impl ScratchBatcher {
    fn has_pending(&self) -> bool {
        !self.written_dests.is_empty()
    }

    fn reset(&mut self) {
        self.cursor = 0;
        self.written_dests.clear();
    }

    /// Reserves a 256B-aligned scratch slice of `need` bytes within scratch buffer
    /// `buf_id` (`capacity >= aligned(need)`). Returns `(emit_uav_barrier_first, byte_offset)`.
    fn alloc(
        &mut self,
        buf_id: usize,
        capacity: u64,
        need: u64,
        dest_va: u64,
        src_va: u64,
    ) -> (bool, u64) {
        let aligned = need.max(AS_ALIGN).next_multiple_of(AS_ALIGN);
        if buf_id != self.buf_id {
            self.buf_id = buf_id;
            self.cursor = 0;
        }
        let dest_conflict = self.written_dests.contains(&dest_va);
        let src_conflict = src_va != 0 && self.written_dests.contains(&src_va);
        let fits = self
            .cursor
            .checked_add(aligned)
            .is_some_and(|end| end <= capacity);
        let need_barrier = self.has_pending() && (dest_conflict || src_conflict || !fits);
        if need_barrier {
            self.reset();
        }
        let offset = self.cursor;
        self.cursor = offset.saturating_add(aligned);
        self.written_dests.push(dest_va);
        (need_barrier, offset)
    }

    fn flush(&mut self, list: &ID3D12GraphicsCommandList4) {
        if self.has_pending() {
            barrier(list);
            self.reset();
        }
    }
}

static UNRECORDED: AtomicU64 = AtomicU64::new(0);
static EMPTY_TLAS_FALLBACKS: AtomicU64 = AtomicU64::new(0);

pub(crate) fn unrecorded(count: u64, reason: &str) {
    UNRECORDED.fetch_add(count, Relaxed);
    tracing::warn!(target:"victory_gpu::host", count,reason,"native RT capture failed");
}

pub(crate) fn note_empty_tlas_fallback(key: u64) {
    let n = EMPTY_TLAS_FALLBACKS.fetch_add(1, Relaxed) + 1;
    if n <= 8 || n.is_power_of_two() {
        tracing::warn!(
            target: "victory_gpu::host",
            key = format_args!("{key:#x}"),
            fallbacks = n,
            "native TLAS missing for non-zero guest VA; bound empty TLAS"
        );
    }
}

pub(crate) fn prepare(batch: Captured, list: &crate::host::HostList) -> Arc<[Work]> {
    batch
        .0
        .into_iter()
        .map(|op| Work::prepare(op, list))
        .collect()
}

type PinSlot = Arc<Mutex<Vec<(u64, Arc<Entry>)>>>;

/// Merges all candidate BLAS entries from `table` into `slot` (deduplicated by `Arc::ptr_eq`)
/// without dropping existing pins. Used on in-place TLAS failure paths so any GPU draw queued
/// after the failed build still keeps every potential BLAS slab slice alive until the TLAS
/// resource's `ID3DDestructionNotifier` fires (strictly after `finish_slot_readers()`).
fn retain_table_pins(slot: &PinSlot, table: &Table) {
    let mut guard = slot.lock().unwrap_or_else(|e| e.into_inner());
    for &(_, key, ref entry) in &table.slots {
        if !guard.iter().any(|(_, existing)| Arc::ptr_eq(existing, entry)) {
            guard.push((key, entry.clone()));
        }
    }
}

/// Preserves candidate BLAS pins for every unprocessed in-place TLAS observation in a
/// popped flight before returning an early error from `collect()`.
fn retain_in_place_pins<'a>(observations: impl IntoIterator<Item = &'a Observation>) {
    for obs in observations {
        if obs.in_place
            && let (Some(slot), Some(table)) = (&obs.entry.pins, &obs.table)
        {
            retain_table_pins(slot, table);
        }
    }
}

struct Entry {
    storage: Storage,
    top: bool,
    serial: u64,
    /// Monotonic generation stamped on creation and incremented on every in-place
    /// TLAS write so an earlier failed observation never evicts a later queued rebuild.
    write_serial: AtomicU64,
    flags: D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAGS,
    shape: Option<Vec<geometry::UpdateGeometry>>,
    update_scratch: u64,
    full_size: u64,
    consecutive_refits: u32,
    compact_size: AtomicU64,
    ready: AtomicBool,
    deps: Mutex<Option<Vec<(u64, u64)>>>,
    /// Single-registration pin slot attached once to a TLAS `storage.buffer` via
    /// `keep_on_resource`, storing `(guest_blas_va, Arc<Entry>)` pairs. Swapped
    /// in-place in `collect()` after the rebuilding flight's fence completes so old
    /// BLASes remain alive for all prior GPU readers and release once unreachable.
    pins: Option<PinSlot>,
}
struct Observation {
    key: u64,
    entry: Arc<Entry>,
    offset: u64,
    table: Option<Arc<Table>>,
    kind: Kind,
    compacted_from: Option<u64>,
    in_place: bool,
    write_serial: u64,
}
#[derive(Clone, Copy)]
enum Kind {
    Dynamic,
    Static,
    Refit,
    Top,
    Copy,
    AutoCompact,
    Empty,
}
#[derive(Clone, Copy, PartialEq, Eq)]
enum Origin {
    Guest,
    Empty,
    Maintenance,
}
#[derive(Default)]
struct Stats {
    input: u64,
    built: u64,
    skipped: u64,
    failed: u64,
    dynamic: u64,
    static_blas: u64,
    refits: u64,
    tlas: u64,
    tlas_reused: u64,
    blas_batched: u64,
    compacted: u64,
    missing_blas: u64,
    auto_compacted: u64,
    auto_refused: u64,
    stale: u64,
    gpu_ticks: u64,
    reasons: BTreeMap<&'static str, u64>,
}
impl Stats {
    fn reject(&mut self, n: u64, why: &'static str, skip: bool) {
        if skip {
            self.skipped += n
        } else {
            self.failed += n
        }
        *self.reasons.entry(why).or_default() += n;
    }
    fn built(&mut self, k: Kind) {
        match k {
            Kind::Empty => return,
            Kind::AutoCompact => {
                self.auto_compacted += 1;
                return;
            }
            Kind::Copy => self.compacted += 1,
            Kind::Top => self.tlas += 1,
            Kind::Dynamic => self.dynamic += 1,
            Kind::Refit => {
                self.refits += 1;
                self.dynamic += 1
            }
            Kind::Static => self.static_blas += 1,
        }
        self.built += 1;
    }
}

struct Flight {
    fence: u64,
    cmd: Commands,
    bytes: u64,
    observations: Vec<Observation>,
    // Inputs and destinations may be superseded several times before the signal.
    _work: Arc<[Work]>,
    resources: Vec<ID3D12Resource>,
    patched: Vec<ID3D12Resource>,
    table_uploads: Vec<ID3D12Resource>,
    sources: Vec<Arc<Entry>>,
}

struct Engine {
    mem: Memory,
    queue: ID3D12CommandQueue,
    fence: ID3D12Fence,
    next: u64,
    generation: u64,
    frequency: u64,
    compactions: VecDeque<(u64, u64)>,
    entries: HashMap<u64, Arc<Entry>>,
    table: Option<Arc<Table>>,
    table_words: Vec<[u32; 4]>,
    free_table_bufs: Vec<ID3D12Resource>,
    free_table_uploads: Vec<ID3D12Resource>,
    free_inst: Vec<ID3D12Resource>,
    patch: Patcher,
    scratch: Option<(u64, ID3D12Resource)>,
    pending: VecDeque<Flight>,
    pool: Vec<Commands>,
    stats: Stats,
    private_start: u64,
    last_log: std::time::Instant,
}

/// All GPU mutation runs under Host::submit's queue lock. Guest compute and direct
/// both execute on Host.queue (DIRECT); the renderer's readback COPY queue never
/// accesses native AS allocations. The private fence follows every batch on DIRECT.
pub(crate) struct Sidecar {
    engine: Mutex<Engine>,
    pub(crate) budget: Arc<Budget>,
    #[cfg(feature = "diagnostics")]
    captured_seed: AtomicBool,
}

impl Sidecar {
    pub(crate) fn new(
        dev: &ID3D12Device,
        queue: &ID3D12CommandQueue,
        adapter: Option<&IDXGIAdapter3>,
    ) -> Checked<Self> {
        #[cfg(feature = "diagnostics")]
        let _native_buffers = crate::replay::record::unrecorded();
        let dev: ID3D12Device5 = dev.cast().map_err(|_| "missing ID3D12Device5")?;
        let mut o = D3D12_FEATURE_DATA_D3D12_OPTIONS5::default();
        unsafe {
            dev.CheckFeatureSupport(
                D3D12_FEATURE_D3D12_OPTIONS5,
                (&raw mut o).cast(),
                size_of_val(&o) as u32,
            )
        }
        .map_err(|_| "DXR tier query")?;
        if o.RaytracingTier.0 < D3D12_RAYTRACING_TIER_1_1.0 {
            return Err("DXR tier 1.1 required");
        }
        let mut e = Engine {
            mem: Memory::new(
                dev.clone(),
                adapter.ok_or("DXGI adapter unavailable")?.clone(),
            )?,
            queue: queue.clone(),
            fence: unsafe { dev.CreateFence(0, D3D12_FENCE_FLAG_NONE) }.map_err(|_| "DXR fence")?,
            frequency: unsafe { queue.GetTimestampFrequency() }
                .map_err(|_| "DIRECT timestamp frequency")?,
            next: 1,
            generation: 0,
            compactions: VecDeque::new(),
            entries: HashMap::new(),
            table: None,
            table_words: Vec::new(),
            free_table_bufs: Vec::new(),
            free_table_uploads: Vec::new(),
            free_inst: Vec::new(),
            patch: Patcher::new(&dev)?,
            scratch: None,
            pending: VecDeque::new(),
            pool: vec![],
            stats: Stats::default(),
            private_start: crate::host::private_bytes(),
            last_log: std::time::Instant::now(),
        };
        // Legal null descriptors and unavailable/stale generations use a permanent
        // built, zero-instance TLAS. Never publish an unbuilt reservation as a TLAS.
        let w = Work {
            op: Op::Build(capture::Build {
                desc: GuestBuildAsDesc {
                    dest_va: 0,
                    inputs: abi::GuestBuildAsInputs {
                        kind: 0,
                        flags: 0,
                        num_descs: 0,
                        descs_layout: 0,
                        ptr_or_va: 0,
                    },
                    source_va: 0,
                    scratch_va: 0,
                },
                inputs: Ok(capture::Inputs::Tlas {
                    count: 0,
                    address: 0,
                }),
                postbuild: vec![],
                stream: vec![],
                rehydrate: None,
            }),
            input: Ok(Native::Tlas(0, 0)),
            buffers: vec![],
        };
        e.submit(Arc::from([w]), Origin::Empty)?;
        e.wait(1)?;
        if !e.entries.get(&0).is_some_and(|e| e.ready.load(Relaxed)) {
            return Err("empty TLAS build failed");
        }
        tracing::info!(target:"victory_gpu::host", cap_mb=e.mem.cap>>20, "native RT sidecar ready on DIRECT");
        let budget = e.mem.budget.clone();
        Ok(Self {
            engine: Mutex::new(e),
            budget,
            #[cfg(feature = "diagnostics")]
            captured_seed: AtomicBool::new(false),
        })
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, Engine> {
        self.engine.lock().unwrap_or_else(|e| e.into_inner())
    }

    pub(crate) fn collect(&self) {
        let mut e = self.lock();
        if let Err(why) = e.collect() {
            drop(e);
            crate::host::fail(format_args!("native RT completion: {why}"));
        }
        #[cfg(feature = "diagnostics")]
        let _native_buffers = crate::replay::record::unrecorded();
        if let Err(why) = e.maintain() {
            drop(e);
            crate::host::fail(format_args!("native RT compaction: {why}"));
        }
        if e.last_log.elapsed().as_secs() >= 2 {
            e.log();
            e.last_log = std::time::Instant::now();
        }
    }
    /// At the first recorded guest submit, freeze pre-existing ASes even if
    /// this frame contains no builder calls. The frame recorder has started
    /// its normal resource snapshot when ExecuteCommandLists returns.
    #[cfg(feature = "diagnostics")]
    pub(crate) fn capture_initial(&self) {
        if self.captured_seed.load(Relaxed) || !crate::replay::record::on() {
            return;
        }
        if let Some((folder, frame, after)) = crate::replay::record::dxr_boundary() {
            self.captured_seed.store(true, Relaxed);
            let _native_buffers = crate::replay::record::unrecorded();
            if let Err(err) = packet::save(&mut self.lock(), &[], folder, frame, after) {
                tracing::error!(target:"victory_gpu::host", %err, "DXR initial packet incomplete");
                let _ = std::fs::write(folder.join("dxr-failed.txt"), err.to_string());
            }
        }
    }

    pub(crate) fn execute(&self, work: Arc<[Work]>) {
        let mut e = self.lock();
        #[cfg(feature = "diagnostics")]
        let _native_buffers = crate::replay::record::unrecorded();
        #[cfg(feature = "diagnostics")]
        if let Some((folder, frame, after)) = crate::replay::record::dxr_boundary()
            && let Err(err) = packet::save(&mut e, &work, folder, frame, after)
        {
            tracing::error!(target:"victory_gpu::host", %err, "DXR packet incomplete");
            let _ = std::fs::write(folder.join("dxr-failed.txt"), err.to_string());
        }
        if let Err(why) = e.submit(work, Origin::Guest) {
            drop(e);
            crate::host::fail(format_args!("native RT submission: {why}"));
        }
    }

    /// Read-only lookup for descriptor writers. Incomplete or stale TLASes are
    /// never exposed. Each returned resource owns all its exact BLAS dependencies.
    pub(crate) fn slot_target(&self, key: u64) -> (ID3D12Resource, bool) {
        let e = self.lock();
        let t = e.entries.get(&key).filter(|t| t.top);
        let pending = t.is_some_and(|t| !t.ready.load(Relaxed));
        let selected = t
            .filter(|t| t.ready.load(Relaxed))
            .unwrap_or(&e.entries[&0]);
        (selected.storage.buffer.clone(), pending)
    }

    /// Before changing a shader-visible slot, finish *all* its previous readers,
    /// including guest work sent after the most recent sidecar command list.
    pub(crate) fn finish_slot_readers(&self) {
        let mut e = self.lock();
        let n = e.next;
        e.next += 1;
        if unsafe { e.queue.Signal(&e.fence, n) }.is_err() || e.wait(n).is_err() {
            drop(e);
            crate::host::fail(format_args!("native RT descriptor reader fence failed"));
        }
    }

    pub(crate) fn srv(&self, key: u64) -> Option<ID3D12Resource> {
        let e = self.lock();
        let t = e.entries.get(&key)?;
        (t.top && t.ready.load(Relaxed)).then(|| t.storage.buffer.clone())
    }
}

impl Engine {
    fn log(&self) {
        let s = &self.stats;
        let lost = UNRECORDED.load(Relaxed);
        let gpu_ms = s.gpu_ticks as f64 * 1000. / self.frequency as f64;
        let Ok(([budget, local, _, other], local_peak)) = self.mem.budget.usage() else {
            return;
        };
        let private = crate::host::private_bytes();
        // Count input at admission, independently of every outcome. Outstanding
        // GPU queries are the only guest items not classifiable at this instant.
        let pending_items = self
            .pending
            .iter()
            .flat_map(|f| &f.observations)
            .filter(|o| !matches!(o.kind, Kind::Empty | Kind::AutoCompact))
            .count() as u64;
        let input_items = (s.input + lost).saturating_sub(pending_items);
        let accounted = s.built + s.skipped + s.failed + lost;
        if input_items != accounted {
            tracing::error!(target: "victory_gpu::host", input_items, accounted, pending_items, "native RT item accounting mismatch");
        }
        tracing::info!(target:"victory_gpu::host", input_items, submitted_items=s.input+lost, pending_items,
            built=s.built, skipped=s.skipped, failed=s.failed+lost,
            dynamic_blas=s.dynamic, static_blas=s.static_blas,refits=s.refits,tlas=s.tlas,tlas_reused=s.tlas_reused,blas_batched=s.blas_batched,compacted=s.compacted+s.auto_compacted, auto_compacted=s.auto_compacted, auto_refused=s.auto_refused,
            missing_blas=s.missing_blas,stale_tlas=s.stale, empty_tlas_fallbacks=EMPTY_TLAS_FALLBACKS.load(Relaxed), pending_batches=self.pending.len(), reasons=?s.reasons,
            slab_mb=memory::slab_bytes()>>20, local_bytes=local, non_local_bytes=other, budget_bytes=budget,
            limit_bytes=memory::limit(self.mem.cap,budget), local_peak_bytes=local_peak,
            non_local_delta=other as i64-self.mem.start[3] as i64,private_commit_delta=private as i64-self.private_start as i64,
            gpu_ms,"native RT submission totals");
    }
    fn wait(&mut self, n: u64) -> Checked<()> {
        // A null event makes SetEventOnCompletion synchronous (no busy polling).
        if unsafe { self.fence.GetCompletedValue() } < n {
            unsafe { self.fence.SetEventOnCompletion(n, HANDLE::default()) }
                .map_err(|_| "DXR fence wait")?;
        }
        self.collect()
    }

    fn recycle_table(&mut self, table: Arc<Table>) {
        // SAFETY of state/lifetime: `Arc::try_unwrap` succeeds only when neither
        // `self.table` nor any remaining in-flight `Flight` holds `table`. All prior
        // SRV reads have completed on DIRECT and decayed `table.buf` back to COMMON.
        if let Ok(t) = Arc::try_unwrap(table)
            && self.free_table_bufs.len() < MAX_FREE_TABLES
        {
            self.free_table_bufs.push(t.buf);
        }
    }

    fn clear_table(&mut self) {
        if let Some(old_table) = self.table.take() {
            self.recycle_table(old_table);
        }
    }

    fn collect(&mut self) -> Checked<()> {
        let done = unsafe { self.fence.GetCompletedValue() };
        if done == u64::MAX {
            return Err("device removed from DXR fence");
        }
        let strict = strict_tlas();
        while self.pending.front().is_some_and(|p| p.fence <= done) {
            let f = self.pending.pop_front().unwrap();
            let bytes = match commands::read_bytes(&f.cmd.read, f.bytes as usize) {
                Ok(bytes) => bytes,
                Err(why) => {
                    retain_in_place_pins(&f.observations);
                    return Err(why);
                }
            };
            let u64_at = |at: usize| u64::from_le_bytes(bytes[at..at + 8].try_into().unwrap());
            self.stats.gpu_ticks += u64_at(8).saturating_sub(u64_at(0));
            let mut obs_iter = f.observations.into_iter();
            while let Some(obs) = obs_iter.next() {
                let is_latest_write = obs.entry.write_serial.load(Relaxed) == obs.write_serial;
                let size = u64_at(obs.offset as usize);
                if size == 0 || size > obs.entry.storage.bytes {
                    if matches!(obs.kind, Kind::AutoCompact) {
                        self.stats.auto_refused += 1;
                    } else if !matches!(obs.kind, Kind::Empty) {
                        self.stats
                            .reject(1, "invalid postbuild compacted size", false);
                    }
                    if obs.in_place
                        && let (Some(slot), Some(table)) = (&obs.entry.pins, &obs.table)
                    {
                        // Preserve every potential BLAS dependency on the TLAS resource's
                        // destruction notifier before dropping `obs.table` so draws queued
                        // after `f` cannot hit a freed BLAS slab slice before `finish_slot_readers()`.
                        retain_table_pins(slot, table);
                    }
                    if is_latest_write {
                        obs.entry.ready.store(false, Relaxed);
                        if self
                            .entries
                            .get(&obs.key)
                            .is_some_and(|e| Arc::ptr_eq(e, &obs.entry))
                        {
                            self.replace(obs.key, None);
                        }
                    }
                    if let Some(table) = obs.table {
                        self.recycle_table(table);
                    }
                    // A prebuild-validated in-place TLAS (already bound to a live shader-visible
                    // descriptor) or an exposed BLAS still referenced by a live/pending TLAS
                    // reporting `size == 0 || size > storage.bytes` indicates a fatal GPU/driver
                    // fault; fail closed instead of letting subsequent draws trace corrupted data.
                    if obs.in_place {
                        retain_in_place_pins(obs_iter.as_slice());
                        return Err("invalid in-place TLAS postbuild size");
                    }
                    if !obs.entry.top
                        && !matches!(obs.kind, Kind::AutoCompact | Kind::Empty)
                        && !strict
                        && Arc::strong_count(&obs.entry) > 1
                    {
                        retain_in_place_pins(obs_iter.as_slice());
                        return Err("invalid referenced BLAS postbuild size");
                    }
                    continue;
                }
                obs.entry.compact_size.store(size, Relaxed);
                if matches!(obs.kind, Kind::Static)
                    && obs.entry.storage.bytes.saturating_sub(size) >= 4096
                    && size <= obs.entry.storage.bytes - obs.entry.storage.bytes / 8
                    && self.compactions.len() < 4096
                {
                    self.compactions.push_back((obs.key, obs.entry.serial));
                }
                if let Some(serial) = obs.compacted_from {
                    // If the guest already rebuilt this BLAS while compaction was in flight,
                    // never overwrite it. In strict mode also refuse if a TLAS started using it.
                    if self
                        .entries
                        .get(&obs.key)
                        .is_none_or(|e| e.serial != serial)
                        || (strict && self.tlas_uses(obs.key, serial))
                    {
                        self.stats.auto_refused += 1;
                        if let Some(table) = obs.table {
                            self.recycle_table(table);
                        }
                        continue;
                    }
                    self.replace(obs.key, Some(obs.entry.clone()));
                }
                if let Some(table) = obs.table {
                    let base = obs.offset as usize + 16;
                    let missing = u32::from_le_bytes(bytes[base..base + 4].try_into().unwrap());
                    self.stats.missing_blas += u64::from(missing);
                    let mut deps = Vec::new();
                    let mut pins = Vec::new();
                    for (slot, key, entry) in &table.slots {
                        if bytes[base + 8 + slot / 8] & (1 << (slot % 8)) != 0 {
                            deps.push((*key, entry.serial));
                            pins.push((*key, entry.clone()));
                        }
                    }
                    // `patch.hlsl` already masks missing BLAS entries with `last.zw = 0`
                    // (legal null instance in DXR). Reject the whole TLAS only when zero
                    // valid instances survived or strict certification mode is requested.
                    if missing != 0 && (deps.is_empty() || strict) {
                        self.stats.reject(1, "TLAS references missing BLAS", false);
                        if obs.in_place
                            && let Some(slot) = &obs.entry.pins
                        {
                            // Unconditionally retain candidate BLAS pins on the resource's
                            // PinSlot (overwritten by any later queued rebuild on completion,
                            // or released when the resource is destroyed after `finish_slot_readers()`).
                            retain_table_pins(slot, &table);
                        }
                        if is_latest_write {
                            obs.entry.ready.store(false, Relaxed);
                            if self
                                .entries
                                .get(&obs.key)
                                .is_some_and(|e| Arc::ptr_eq(e, &obs.entry))
                            {
                                self.replace(obs.key, None);
                            }
                        }
                        self.recycle_table(table);
                        continue;
                    }
                    // In strict mode invalidate if any referenced BLAS changed generation.
                    // In live mode every BLAS slice is immutable and pinned in `obs.entry.pins`,
                    // so the TLAS remains completely valid until the next TLAS build completes.
                    let current = !strict
                        || deps
                            .iter()
                            .all(|(k, g)| self.entries.get(k).is_some_and(|e| e.serial == *g));
                    if let Some(slot) = &obs.entry.pins {
                        // Prior readers of the previous TLAS generation finished before
                        // `f.fence <= done`; swap in this generation's exact BLAS pins.
                        *slot.lock().unwrap_or_else(|e| e.into_inner()) = pins;
                    }
                    *obs.entry.deps.lock().unwrap_or_else(|e| e.into_inner()) = Some(deps);
                    if !current {
                        if is_latest_write {
                            obs.entry.ready.store(false, Relaxed);
                            self.stats.stale += 1;
                            if self
                                .entries
                                .get(&obs.key)
                                .is_some_and(|e| Arc::ptr_eq(e, &obs.entry))
                            {
                                self.replace(obs.key, None);
                            }
                        }
                    } else if self
                        .entries
                        .get(&obs.key)
                        .is_some_and(|e| Arc::ptr_eq(e, &obs.entry))
                    {
                        obs.entry.ready.store(true, Relaxed);
                    }
                    self.recycle_table(table);
                } else {
                    let valid = !strict
                        || obs
                            .entry
                            .deps
                            .lock()
                            .unwrap_or_else(|e| e.into_inner())
                            .as_ref()
                            .is_none_or(|ds| {
                                ds.iter().all(|(k, g)| {
                                    self.entries.get(k).is_some_and(|e| e.serial == *g)
                                })
                            });
                    obs.entry.ready.store(valid, Relaxed);
                    if !valid
                        && is_latest_write
                        && self
                            .entries
                            .get(&obs.key)
                            .is_some_and(|e| Arc::ptr_eq(e, &obs.entry))
                    {
                        self.stats.stale += 1;
                        self.replace(obs.key, None);
                    }
                }
                self.stats.built(obs.kind);
            }
            for buf in f.patched {
                // SAFETY: `f.fence <= done`; buffer ended in NON_PIXEL_SHADER_RESOURCE and
                // decayed to COMMON after ExecuteCommandLists.
                if self.free_inst.len() < MAX_FREE_INST
                    && unsafe { buf.GetDesc().Width } <= MAX_POOLED_INST_BYTES
                {
                    self.free_inst.push(buf);
                }
            }
            for upload in f.table_uploads {
                if self.free_table_uploads.len() < MAX_FREE_TABLES {
                    self.free_table_uploads.push(upload);
                }
            }
            if self.pool.len() < 8 && f.cmd.capacity <= 2 << 20 {
                self.pool.push(f.cmd);
            }
        }
        Ok(())
    }

    fn replace(&mut self, key: u64, new: Option<Arc<Entry>>) {
        let old = self.entries.remove(&key);
        if let Some(ref evicted) = old
            && evicted.top
        {
            evicted.ready.store(false, Relaxed);
        }
        let changed_blas = old.as_ref().is_some_and(|e| !e.top);
        if changed_blas || new.as_ref().is_some_and(|e| !e.top) {
            self.clear_table();
        }
        if changed_blas && strict_tlas() {
            self.entries.retain(|_, t| {
                let stale = t.top
                    && t.deps
                        .lock()
                        .unwrap_or_else(|e| e.into_inner())
                        .as_ref()
                        .is_some_and(|ds| ds.iter().any(|(k, _)| *k == key));
                if stale {
                    t.ready.store(false, Relaxed);
                    self.stats.stale += 1;
                }
                !stale
            });
        }
        if let Some(e) = new {
            self.entries.insert(key, e);
        }
    }

    fn tlas_uses(&self, key: u64, serial: u64) -> bool {
        self.pending
            .iter()
            .any(|f| f.observations.iter().any(|o| o.entry.top))
            || self.entries.values().any(|t| {
                t.top
                    && t.deps
                        .lock()
                        .unwrap_or_else(|e| e.into_inner())
                        .as_ref()
                        .is_some_and(|ds| ds.contains(&(key, serial)))
            })
    }

    fn maintain(&mut self) -> Checked<()> {
        // Avoid piling maintenance copies onto an active TLAS submission.
        if self
            .pending
            .iter()
            .any(|f| f.observations.iter().any(|o| o.entry.top))
        {
            return Ok(());
        }
        let strict = strict_tlas();
        let mut ops = Vec::new();
        while ops.len() < 8
            && let Some((key, serial)) = self.compactions.pop_front()
        {
            if self.entries.get(&key).is_none_or(|e| e.serial != serial)
                || (strict && self.tlas_uses(key, serial))
            {
                continue;
            }
            ops.push(Work {
                op: Op::Copy {
                    src: key,
                    dst: key,
                    mode: 1,
                },
                input: Ok(Native::Copy),
                buffers: vec![],
            });
        }
        if !ops.is_empty() {
            self.submit(ops.into(), Origin::Maintenance)?;
        }
        Ok(())
    }

    fn reject_work(&mut self, origin: Origin, w: &Work, why: &'static str) {
        if origin == Origin::Maintenance {
            self.stats.auto_refused += 1;
            return;
        }
        let n = w.op.item_count();
        let noop = matches!(w.op,Op::Copy{mode,..} if mode!=1);
        self.stats.reject(n, why, noop);
        // A rejected original HRESULT (and an ignored copy mode) has no guest write.
        let key = match &w.op {
            Op::Build(b) if b.rehydrate.as_ref().is_none_or(|r| r.hr >= 0) => b.desc.dest_va,
            Op::Copy { dst, mode: 1, .. } => *dst,
            _ => 0,
        };
        if key != 0 {
            self.replace(key, None);
        } else if matches!(&w.op, Op::Rejected { .. }) {
            // In strict certification mode clear everything; in live gameplay
            // drop TLAS entries so no stale TLAS persists while keeping built BLASes.
            if strict_tlas() {
                self.entries.retain(|&k, _| k == 0);
                self.clear_table();
            } else {
                self.entries.retain(|&k, e| k == 0 || !e.top);
                tracing::warn!(target: "victory_gpu::host", why, "native RT rejected batch without destination VA; invalidated TLAS entries");
            }
        }
    }

    fn submit(&mut self, work: Arc<[Work]>, origin: Origin) -> Checked<()> {
        self.collect()?;
        if origin == Origin::Guest {
            self.stats.input += work.iter().map(|w| w.op.item_count()).sum::<u64>();
        }
        if self.pending.len() >= 16 {
            self.wait(self.pending.front().unwrap().fence)?;
        }
        // Real Copy is a separate ABI call/boundary. If its source has any pending
        // build (initial build or in-place TLAS rebuild), wait for that producer so
        // `compact_size` and TLAS `pins`/`deps` match the exact copied generation.
        if let [w] = &*work
            && let Op::Copy { src, mode: 1, .. } = w.op
            && let Some(e) = self.entries.get(&src)
            && let Some(f) = self
                .pending
                .iter()
                .rev()
                .find(|f| f.observations.iter().any(|o| Arc::ptr_eq(&o.entry, e)))
        {
            self.wait(f.fence)?;
        }
        let needed = work
            .iter()
            .try_fold(16u64, |n, w| n.checked_add(report_size(w)))
            .ok_or("DXR report overflow")?;
        let cmd = if let Some(i) = self.pool.iter().position(|c| c.capacity >= needed) {
            self.pool.swap_remove(i)
        } else {
            match Commands::new(&mut self.mem, needed) {
                Ok(c) => c,
                Err(why) => {
                    for w in &*work {
                        self.reject_work(origin, w, why);
                    }
                    return Ok(());
                }
            }
        };
        cmd.begin(needed)?;
        let mut f = Flight {
            fence: self.next,
            cmd,
            bytes: needed,
            observations: vec![],
            _work: work.clone(),
            resources: vec![],
            patched: vec![],
            table_uploads: vec![],
            sources: vec![],
        };
        let mut scratch_batch = ScratchBatcher::default();
        let mut offset = 16;
        for w in work.iter() {
            let res = match &w.op {
                Op::Build(b) => self.build(
                    &mut f,
                    &mut scratch_batch,
                    w,
                    b,
                    offset,
                    origin == Origin::Empty,
                ),
                Op::Copy { dst, src, mode } => {
                    scratch_batch.flush(&f.cmd.list);
                    self.copy(&mut f, *dst, *src, *mode, offset, origin)
                }
                Op::Rejected { why, .. } => Err(*why),
            };
            if let Err(why) = res {
                self.reject_work(origin, w, why);
            }
            offset += report_size(w);
        }
        scratch_batch.flush(&f.cmd.list);
        let list = f.cmd.finish(needed)?;
        // No fallible allocation or publication outside the submit lock. Signal failure
        // is fatal, since without a fence no resource can safely be released.
        unsafe {
            self.queue.ExecuteCommandLists(&[Some(list)]);
            self.pending.push_back(f);
            self.queue
                .Signal(&self.fence, self.next)
                .map_err(|_| "DXR queue signal")?;
        }
        self.next += 1;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn result(
        &mut self,
        bytes: u64,
        top: bool,
        flags: D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAGS,
        shape: Option<Vec<geometry::UpdateGeometry>>,
        full_size: u64,
        update_scratch: u64,
        consecutive_refits: u32,
    ) -> Checked<Arc<Entry>> {
        let storage = self.mem.as_storage(bytes, top)?;
        let pins = if top {
            let slot: PinSlot = Arc::new(Mutex::new(Vec::new()));
            memory::keep_on_resource(&storage.buffer, slot.clone())?;
            Some(slot)
        } else {
            None
        };
        self.generation += 1;
        Ok(Arc::new(Entry {
            storage,
            top,
            serial: self.generation,
            write_serial: AtomicU64::new(self.generation),
            flags,
            shape,
            update_scratch,
            full_size,
            consecutive_refits,
            compact_size: AtomicU64::new(0),
            ready: AtomicBool::new(false),
            deps: Mutex::new(None),
            pins,
        }))
    }

    fn take_inst_buffer(&mut self, need: u64) -> Checked<ID3D12Resource> {
        let best = self
            .free_inst
            .iter()
            .enumerate()
            .filter_map(|(i, b)| {
                let w = unsafe { b.GetDesc().Width };
                (w >= need).then_some((i, w))
            })
            .min_by_key(|&(_, w)| w)
            .map(|(i, _)| i);
        if let Some(i) = best {
            return Ok(self.free_inst.swap_remove(i));
        }
        let want = need
            .checked_next_power_of_two()
            .unwrap_or(need)
            .max(MIN_INST_POOL_BYTES);
        match self.mem.buffer(
            want,
            D3D12_HEAP_TYPE_DEFAULT,
            D3D12_RESOURCE_STATE_COMMON,
            D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
        ) {
            Ok(buf) => Ok(buf),
            Err(_) if want > need => self.mem.buffer(
                need,
                D3D12_HEAP_TYPE_DEFAULT,
                D3D12_RESOURCE_STATE_COMMON,
                D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
            ),
            Err(why) => Err(why),
        }
    }

    fn ensure_scratch(&mut self, need: u64) -> Checked<(u64, ID3D12Resource)> {
        let aligned = need.max(AS_ALIGN).next_multiple_of(65536);
        if self.scratch.as_ref().is_none_or(|(n, _)| *n < aligned) {
            let cur = self.scratch.as_ref().map_or(0, |(n, _)| *n);
            let want = aligned
                .max(MIN_SCRATCH_BYTES)
                .max(cur.saturating_mul(2).min(64 << 20))
                .next_multiple_of(65536);
            let (cap, buf) = match self.mem.buffer(
                want,
                D3D12_HEAP_TYPE_DEFAULT,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
            ) {
                Ok(buf) => (want, buf),
                Err(_) if want > aligned => (
                    aligned,
                    self.mem.buffer(
                        aligned,
                        D3D12_HEAP_TYPE_DEFAULT,
                        D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                        D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                    )?,
                ),
                Err(why) => return Err(why),
            };
            self.scratch = Some((cap, buf));
        }
        let (cap, buf) = self.scratch.as_ref().unwrap();
        Ok((*cap, buf.clone()))
    }

    fn build(
        &mut self,
        f: &mut Flight,
        scratch_batch: &mut ScratchBatcher,
        w: &Work,
        b: &capture::Build,
        off: u64,
        empty: bool,
    ) -> Checked<()> {
        const UPDATE: i32 = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PERFORM_UPDATE.0;
        const ALLOW: i32 = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_ALLOW_UPDATE.0;
        let native = w.input.as_ref().map_err(|e| *e)?;
        let top = matches!(native, Native::Tlas(..));
        let raw = b.desc.inputs.flags as i32;
        let auto_refit = refit_dynamic_blas() && !top && b.rehydrate.is_none();
        let period = refit_period();
        // The low six bits are the console's standard DXR flags. Do not pass
        // platform-only optimization bits through to NVIDIA's driver.
        let mut flags = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAGS(
            (raw & 0x1f) | D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_ALLOW_COMPACTION.0,
        );
        if flags.0 & 12 == 12 {
            return Err("conflicting AS build preferences");
        }
        if top {
            flags.0 &= !ALLOW;
        } else if auto_refit {
            flags.0 |= ALLOW;
        }
        let shape = match native {
            Native::Blas(g) if flags.0 & ALLOW != 0 || raw & UPDATE != 0 => {
                geometry::update_shape(g)
            }
            _ => None,
        };
        let explicit_update = raw & UPDATE != 0;
        let candidate_va = if explicit_update {
            b.desc.source_va
        } else if auto_refit {
            b.desc.dest_va
        } else {
            0
        };
        let source = (candidate_va != 0 && (explicit_update || auto_refit))
            .then(|| self.entries.get(&candidate_va))
            .flatten()
            .filter(|e| {
                !top && !e.top
                    && e.flags.0 & ALLOW != 0
                    && (e.flags.0 & !ALLOW) == (flags.0 & !ALLOW)
                    && shape.is_some()
                    && e.shape == shape
                    && e.update_scratch > 0
                    && (!auto_refit || period == 0 || e.consecutive_refits < period)
            })
            .cloned();
        let mut inputs = D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS {
            Type: if top {
                D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL
            } else {
                D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL
            },
            Flags: flags,
            DescsLayout: D3D12_ELEMENTS_LAYOUT_ARRAY,
            ..Default::default()
        };
        let (num, inst_src) = match native {
            Native::Blas(g) => {
                inputs.NumDescs = g.len() as u32;
                inputs.Anonymous.pGeometryDescs = g.as_ptr();
                (0, 0)
            }
            Native::Tlas(n, gpu) => {
                inputs.NumDescs = *n;
                (*n, *gpu)
            }
            Native::Copy => return Err("missing AS build inputs"),
        };
        let mut pre = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO::default();
        unsafe {
            self.mem
                .dev
                .GetRaytracingAccelerationStructurePrebuildInfo(&inputs, &mut pre)
        };
        if pre.ResultDataMaxSizeInBytes == 0 {
            return Err("unsupported DXR prebuild inputs");
        }
        let (result_bytes, scratch_need) = source.as_ref().map_or(
            (pre.ResultDataMaxSizeInBytes, pre.ScratchDataSizeInBytes),
            |e| (e.full_size, e.update_scratch),
        );
        let (scratch_cap, scratch) = self.ensure_scratch(scratch_need)?;
        let reuse = if top && !empty && num > 0 && !strict_tlas() {
            self.entries
                .get(&b.desc.dest_va)
                .filter(|e| e.top && e.ready.load(Relaxed) && e.storage.bytes >= result_bytes)
                .cloned()
        } else {
            None
        };
        let in_place = reuse.is_some();
        let next_refits = source
            .as_ref()
            .map_or(0, |e| e.consecutive_refits.saturating_add(1));
        let entry = if let Some(existing) = reuse {
            existing
        } else if top && !empty && num > 0 && !strict_tlas() {
            let reserve = result_bytes
                .checked_next_power_of_two()
                .unwrap_or(result_bytes)
                .max(MIN_TLAS_RESERVE_BYTES);
            match self.result(reserve, top, flags, shape.clone(), result_bytes, 0, 0) {
                Ok(e) => e,
                Err(_) if reserve > result_bytes => {
                    self.result(result_bytes, top, flags, shape, result_bytes, 0, 0)?
                }
                Err(why) => return Err(why),
            }
        } else {
            self.result(
                result_bytes,
                top,
                source.as_ref().map_or(flags, |e| e.flags),
                shape,
                result_bytes,
                source
                    .as_ref()
                    .map_or(pre.UpdateScratchDataSizeInBytes, |e| e.update_scratch),
                next_refits,
            )?
        };
        let report = unsafe { f.cmd.report.GetGPUVirtualAddress() } + off;
        // Complete all fallible allocations before recording any GPU commands.
        let (inst, new_table) = if num != 0 {
            let inst = self.take_inst_buffer(u64::from(num) * 64)?;
            let new_table = if self.table.is_none() {
                match Table::build(
                    &mut self.mem,
                    &self.entries,
                    &mut self.table_words,
                    &mut self.free_table_bufs,
                    &mut self.free_table_uploads,
                ) {
                    Ok(pair) => Some(pair),
                    Err(why) => {
                        if self.free_inst.len() < MAX_FREE_INST
                            && unsafe { inst.GetDesc().Width } <= MAX_POOLED_INST_BYTES
                        {
                            self.free_inst.push(inst);
                        }
                        return Err(why);
                    }
                }
            } else {
                None
            };
            (Some(inst), new_table)
        } else {
            (None, None)
        };

        // Past this point every step is infallible.
        if top {
            // Flush any batched BLAS builds (and their postbuild report writes) before
            // patching instances and building a TLAS that may reference them.
            scratch_batch.flush(&f.cmd.list);
        }
        if let Some((snapshot, upload)) = new_table {
            // Each new hash version uses its own upload + GPU snapshot until fence completion.
            // Multiple TLAS builds in one batch can see different contents of the same guest VA.
            unsafe {
                f.cmd.list.ResourceBarrier(&[crate::host::transition(
                    &snapshot.buf,
                    D3D12_RESOURCE_STATE_COMMON,
                    D3D12_RESOURCE_STATE_COPY_DEST,
                )]);
                f.cmd
                    .list
                    .CopyBufferRegion(&snapshot.buf, 0, &upload, 0, commands::TABLE_BYTES);
                f.cmd.list.ResourceBarrier(&[crate::host::transition(
                    &snapshot.buf,
                    D3D12_RESOURCE_STATE_COPY_DEST,
                    D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                )]);
            }
            f.table_uploads.push(upload);
            self.table = Some(snapshot);
        }
        let mut table = None;
        if let Some(inst) = inst {
            let t = self.table.as_ref().unwrap().clone();
            self.patch
                .record(&f.cmd.list, num, inst_src, &t, &inst, report + 16);
            inputs.Anonymous.InstanceDescs = unsafe { inst.GetGPUVirtualAddress() };
            f.patched.push(inst);
            table = Some(t);
        }
        let kind = if empty {
            Kind::Empty
        } else if top {
            Kind::Top
        } else if source.is_some() {
            Kind::Refit
        } else if b.rehydrate.is_some() {
            Kind::Static
        } else {
            Kind::Dynamic
        };
        if let Some(src) = &source {
            inputs.Flags.0 = src.flags.0 | UPDATE;
        }
        let src_gpu = source.as_ref().map_or(0, |e| e.storage.gpu);
        let scratch_off = if top {
            0
        } else {
            let (need_barrier, off) = scratch_batch.alloc(
                scratch.as_raw() as usize,
                scratch_cap,
                scratch_need,
                entry.storage.gpu,
                src_gpu,
            );
            if need_barrier {
                barrier(&f.cmd.list);
            } else if scratch_batch.written_dests.len() > 1 {
                self.stats.blas_batched += 1;
            }
            off
        };
        let desc = D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC {
            Inputs: inputs,
            DestAccelerationStructureData: entry.storage.gpu,
            SourceAccelerationStructureData: src_gpu,
            ScratchAccelerationStructureData: unsafe { scratch.GetGPUVirtualAddress() }
                + scratch_off,
        };
        let post = [
            D3D12_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO_DESC {
                DestBuffer: report,
                InfoType: D3D12_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO_COMPACTED_SIZE,
            },
        ];
        // All fallible setup precedes GPU mutation/publication. BLAS destinations are
        // always fresh immutable slab slices; steady-state TLASes rebuild in-place on
        // DIRECT between UAV barriers so shader-visible descriptors remain stable.
        unsafe {
            f.cmd
                .list
                .BuildRaytracingAccelerationStructure(&desc, Some(&post))
        };
        if top {
            barrier(&f.cmd.list);
        }
        if !f.resources.iter().any(|r| r == &scratch) {
            f.resources.push(scratch);
        }
        if let Some(src) = source {
            f.sources.push(src);
        }
        let write_serial = if in_place {
            self.stats.tlas_reused += 1;
            self.generation += 1;
            entry.write_serial.store(self.generation, Relaxed);
            self.generation
        } else {
            self.replace(b.desc.dest_va, Some(entry.clone()));
            entry.serial
        };
        f.observations.push(Observation {
            key: b.desc.dest_va,
            entry,
            offset: off,
            table,
            kind,
            compacted_from: None,
            in_place,
            write_serial,
        });
        Ok(())
    }

    fn copy(
        &mut self,
        f: &mut Flight,
        dst: u64,
        src: u64,
        mode: u32,
        off: u64,
        origin: Origin,
    ) -> Checked<()> {
        if mode != 1 {
            return Err("console ignores copy mode");
        }
        if dst == 0 || src == 0 || !dst.is_multiple_of(256) || !src.is_multiple_of(256) {
            return Err("invalid compact address");
        }
        let s = self
            .entries
            .get(&src)
            .cloned()
            .ok_or("compact source missing")?;
        let n = s.compact_size.load(Relaxed);
        if n == 0 || n > s.storage.bytes || s.top && !s.ready.load(Relaxed) {
            return Err("compact source incomplete");
        }
        // Always allocates a fresh non-overlapping slice/buffer (including when dst == src,
        // because `s` remains alive through `f.sources` until `f.fence` completes).
        let e = self.result(
            n,
            s.top,
            s.flags,
            s.shape.clone(),
            s.full_size,
            s.update_scratch,
            s.consecutive_refits,
        )?;
        if s.top {
            // Snapshot the source TLAS's exact current BLAS pins into `e`'s own PinSlot
            // so future in-place rebuilds of `s` cannot release BLASes referenced by `e`.
            let copied_pins = s
                .pins
                .as_ref()
                .map(|p| p.lock().unwrap_or_else(|err| err.into_inner()).clone())
                .unwrap_or_default();
            if let Some(dst_pins) = &e.pins {
                *dst_pins.lock().unwrap_or_else(|err| err.into_inner()) = copied_pins;
            }
            let copied_deps = s.deps.lock().unwrap_or_else(|err| err.into_inner()).clone();
            *e.deps.lock().unwrap_or_else(|err| err.into_inner()) = copied_deps;
        }

        unsafe {
            f.cmd.list.CopyRaytracingAccelerationStructure(
                e.storage.gpu,
                s.storage.gpu,
                D3D12_RAYTRACING_ACCELERATION_STRUCTURE_COPY_MODE_COMPACT,
            );
        }
        barrier(&f.cmd.list);
        let post = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO_DESC {
            DestBuffer: unsafe { f.cmd.report.GetGPUVirtualAddress() } + off,
            InfoType: D3D12_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO_COMPACTED_SIZE,
        };
        unsafe {
            f.cmd
                .list
                .EmitRaytracingAccelerationStructurePostbuildInfo(&post, &[e.storage.gpu]);
        }
        let compacted_from = (origin == Origin::Maintenance).then_some(s.serial);
        f.sources.push(s);
        if compacted_from.is_none() {
            self.replace(dst, Some(e.clone()));
        }
        let write_serial = e.serial;
        f.observations.push(Observation {
            key: dst,
            entry: e,
            offset: off,
            table: None,
            kind: if origin == Origin::Maintenance {
                Kind::AutoCompact
            } else {
                Kind::Copy
            },
            compacted_from,
            in_place: false,
            write_serial,
        });
        Ok(())
    }
}

fn report_size(w: &Work) -> u64 {
    if matches!(w.input,Ok(Native::Tlas(n,_)) if n>0) {
        (16 + PATCH_REPORT).next_multiple_of(16)
    } else {
        16
    }
}
