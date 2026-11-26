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

struct Entry {
    storage: Storage,
    top: bool,
    serial: u64,
    flags: D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAGS,
    shape: Option<Vec<geometry::UpdateGeometry>>,
    update_scratch: u64,
    full_size: u64,
    compact_size: AtomicU64,
    ready: AtomicBool,
    deps: OnceLock<Vec<(u64, u64)>>,
}
struct Observation {
    key: u64,
    entry: Arc<Entry>,
    offset: u64,
    table: Option<Arc<Table>>,
    kind: Kind,
    compacted_from: Option<u64>,
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
        if let Some((folder, frame, after)) = crate::replay::record::dxr_boundary() {
            if let Err(err) = packet::save(&mut e, &work, folder, frame, after) {
                tracing::error!(target:"victory_gpu::host", %err, "DXR packet incomplete");
                let _ = std::fs::write(folder.join("dxr-failed.txt"), err.to_string());
            }
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
            dynamic_blas=s.dynamic, static_blas=s.static_blas,refits=s.refits,tlas=s.tlas,compacted=s.compacted+s.auto_compacted, auto_compacted=s.auto_compacted, auto_refused=s.auto_refused,
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

    fn collect(&mut self) -> Checked<()> {
        let done = unsafe { self.fence.GetCompletedValue() };
        if done == u64::MAX {
            return Err("device removed from DXR fence");
        }
        while self.pending.front().is_some_and(|p| p.fence <= done) {
            let f = self.pending.pop_front().unwrap();
            let bytes = commands::read_bytes(&f.cmd.read, f.bytes as usize)?;
            let u64_at = |at: usize| u64::from_le_bytes(bytes[at..at + 8].try_into().unwrap());
            self.stats.gpu_ticks += u64_at(8).saturating_sub(u64_at(0));
            for obs in f.observations {
                {
                    let size = u64_at(obs.offset as usize);
                    if size == 0 || size > obs.entry.storage.bytes {
                        if matches!(obs.kind, Kind::AutoCompact) {
                            self.stats.auto_refused += 1;
                        } else if !matches!(obs.kind, Kind::Empty) {
                            self.stats
                                .reject(1, "invalid postbuild compacted size", false);
                        }
                        if self
                            .entries
                            .get(&obs.key)
                            .is_some_and(|e| Arc::ptr_eq(e, &obs.entry))
                        {
                            self.replace(obs.key, None);
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
                }
                if let Some(serial) = obs.compacted_from {
                    // A guest TLAS may have started using the source while this
                    // optional copy was in flight. Keep that valid version in place.
                    if self
                        .entries
                        .get(&obs.key)
                        .is_none_or(|e| e.serial != serial)
                        || self.tlas_uses(obs.key, serial)
                    {
                        self.stats.auto_refused += 1;
                        continue;
                    }
                    self.replace(obs.key, Some(obs.entry.clone()));
                }
                if let Some(table) = obs.table {
                    let base = obs.offset as usize + 16;
                    let missing = u32::from_le_bytes(bytes[base..base + 4].try_into().unwrap());
                    self.stats.missing_blas += u64::from(missing);
                    if missing != 0 {
                        self.stats.reject(1, "TLAS references missing BLAS", false);
                        if self
                            .entries
                            .get(&obs.key)
                            .is_some_and(|e| Arc::ptr_eq(e, &obs.entry))
                        {
                            self.replace(obs.key, None);
                        }
                        continue;
                    }
                    let mut deps = Vec::new();
                    let mut pins = Vec::new();
                    for (slot, key, entry) in &table.slots {
                        if bytes[base + 8 + slot / 8] & (1 << (slot % 8)) != 0 {
                            deps.push((*key, entry.serial));
                            pins.push(entry.clone());
                        }
                    }
                    // A BLAS could already have been rebuilt by a later queued submission.
                    let current = deps
                        .iter()
                        .all(|(k, g)| self.entries.get(k).is_some_and(|e| e.serial == *g));
                    memory::keep_on_resource(&obs.entry.storage.buffer, pins)?;
                    obs.entry.deps.set(deps).ok();
                    obs.entry.ready.store(current, Relaxed);
                    if !current {
                        self.stats.stale += 1;
                        if self
                            .entries
                            .get(&obs.key)
                            .is_some_and(|e| Arc::ptr_eq(e, &obs.entry))
                        {
                            self.replace(obs.key, None);
                        }
                    }
                } else {
                    let valid = obs.entry.deps.get().is_none_or(|ds| {
                        ds.iter()
                            .all(|(k, g)| self.entries.get(k).is_some_and(|e| e.serial == *g))
                    });
                    obs.entry.ready.store(valid, Relaxed);
                    if !valid
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
            if self.pool.len() < 8 && f.cmd.capacity <= 2 << 20 {
                self.pool.push(f.cmd);
            }
        }
        Ok(())
    }

    fn replace(&mut self, key: u64, new: Option<Arc<Entry>>) {
        let old = self.entries.remove(&key);
        let changed_blas = old.as_ref().is_some_and(|e| !e.top);
        if changed_blas || new.as_ref().is_some_and(|e| !e.top) {
            self.table = None;
        }
        if changed_blas {
            self.entries.retain(|_, t| {
                let stale = t.top
                    && t.deps
                        .get()
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
            || self
                .entries
                .values()
                .any(|t| t.top && t.deps.get().is_some_and(|ds| ds.contains(&(key, serial))))
    }

    fn maintain(&mut self) -> Checked<()> {
        // Do not churn dependencies of a valid/pending TLAS to save a few KiB.
        if self
            .pending
            .iter()
            .any(|f| f.observations.iter().any(|o| o.entry.top))
        {
            return Ok(());
        }
        let mut ops = Vec::new();
        while ops.len() < 8
            && let Some((key, serial)) = self.compactions.pop_front()
        {
            if self.entries.get(&key).is_none_or(|e| e.serial != serial)
                || self.tlas_uses(key, serial)
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
            // No readable destination. Existing descriptors cannot certify which
            // address the console call may have replaced; keep only the empty TLAS.
            self.entries.retain(|&k, _| k == 0);
            self.table = None;
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
        // Real Copy is a separate ABI call/boundary. If its postbuild query is still
        // in flight, wait for that producer, never substitute CLONE or guess a size.
        if let [w] = &*work
            && let Op::Copy { src, mode: 1, .. } = w.op
            && let Some(e) = self.entries.get(&src)
            && e.compact_size.load(Relaxed) == 0
            && let Some(f) = self
                .pending
                .iter()
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
            sources: vec![],
        };
        let mut offset = 16;
        for w in work.iter() {
            let res = match &w.op {
                Op::Build(b) => self.build(&mut f, w, b, offset, origin == Origin::Empty),
                Op::Copy { dst, src, mode } => self.copy(&mut f, *dst, *src, *mode, offset, origin),
                Op::Rejected { why, .. } => Err(*why),
            };
            if let Err(why) = res {
                self.reject_work(origin, w, why);
            }
            offset += report_size(w);
        }
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

    fn result(
        &mut self,
        bytes: u64,
        top: bool,
        flags: D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAGS,
        shape: Option<Vec<geometry::UpdateGeometry>>,
        full_size: u64,
        update_scratch: u64,
    ) -> Checked<Arc<Entry>> {
        let storage = self.mem.as_storage(bytes, top)?;
        self.generation += 1;
        Ok(Arc::new(Entry {
            storage,
            top,
            serial: self.generation,
            flags,
            shape,
            update_scratch,
            full_size,
            compact_size: AtomicU64::new(0),
            ready: AtomicBool::new(false),
            deps: OnceLock::new(),
        }))
    }

    fn build(
        &mut self,
        f: &mut Flight,
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
        }
        let shape = match native {
            Native::Blas(g) if flags.0 & ALLOW != 0 || raw & UPDATE != 0 => {
                geometry::update_shape(g)
            }
            _ => None,
        };
        let source = self
            .entries
            .get(&b.desc.source_va)
            .filter(|e| {
                !top && !e.top
                    && raw & UPDATE != 0
                    && e.flags.0 & ALLOW != 0
                    && (e.flags.0 & !ALLOW) == (flags.0 & !ALLOW)
                    && shape.is_some()
                    && e.shape == shape
                    && e.update_scratch > 0
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
        let (result_bytes, scratch) = source.as_ref().map_or(
            (pre.ResultDataMaxSizeInBytes, pre.ScratchDataSizeInBytes),
            |e| (e.full_size, e.update_scratch),
        );
        if self
            .scratch
            .as_ref()
            .is_none_or(|(n, _)| *n < scratch.max(256))
        {
            let n = scratch.max(256).next_multiple_of(65536);
            let buf = self.mem.buffer(
                n,
                D3D12_HEAP_TYPE_DEFAULT,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
            )?;
            self.scratch = Some((n, buf));
        }
        let scratch = self.scratch.as_ref().unwrap().1.clone();
        let entry = self.result(
            result_bytes,
            top,
            source.as_ref().map_or(flags, |e| e.flags),
            shape,
            result_bytes,
            source
                .as_ref()
                .map_or(pre.UpdateScratchDataSizeInBytes, |e| e.update_scratch),
        )?;
        let report = unsafe { f.cmd.report.GetGPUVirtualAddress() } + off;
        let mut table = None;
        if num != 0 {
            let inst = self.mem.buffer(
                u64::from(num) * 64,
                D3D12_HEAP_TYPE_DEFAULT,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
            )?;
            if self.table.is_none() {
                let (snapshot, upload) = Table::new(&mut self.mem, &self.entries)?;
                // Each new hash version has its own immutable upload. Multiple TLAS
                // builds in one batch can see different contents of the same guest VA.
                unsafe {
                    f.cmd.list.CopyBufferRegion(
                        &snapshot.buf,
                        0,
                        &upload,
                        0,
                        commands::SLOTS as u64 * 16,
                    );
                    f.cmd.list.ResourceBarrier(&[crate::host::transition(
                        &snapshot.buf,
                        D3D12_RESOURCE_STATE_COPY_DEST,
                        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                    )]);
                }
                f.resources.push(upload);
                self.table = Some(snapshot);
            }
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
        let desc = D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC {
            Inputs: inputs,
            DestAccelerationStructureData: entry.storage.gpu,
            SourceAccelerationStructureData: source.as_ref().map_or(0, |e| e.storage.gpu),
            ScratchAccelerationStructureData: unsafe { scratch.GetGPUVirtualAddress() },
        };
        let post = [
            D3D12_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO_DESC {
                DestBuffer: report,
                InfoType: D3D12_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO_COMPACTED_SIZE,
            },
        ];
        // All fallible setup precedes GPU mutation/publication. Immutable destinations
        // preserve older TLAS snapshots, including when a title asks for in-place update.
        unsafe {
            f.cmd
                .list
                .BuildRaytracingAccelerationStructure(&desc, Some(&post))
        };
        barrier(&f.cmd.list);
        f.resources.push(scratch);
        if let Some(src) = source {
            f.sources.push(src);
        }
        self.replace(b.desc.dest_va, Some(entry.clone()));
        f.observations.push(Observation {
            key: b.desc.dest_va,
            entry,
            offset: off,
            table,
            kind,
            compacted_from: None,
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
        if dst == 0 || src == 0 || dst % 256 != 0 || src % 256 != 0 {
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
        let e = self.result(
            n,
            s.top,
            s.flags,
            s.shape.clone(),
            s.full_size,
            s.update_scratch,
        )?;
        if s.top {
            // A descriptor can hold the compacted resource after the source is replaced.
            // Keep the original immutable dependency owner with this TLAS.
            memory::keep_on_resource(&e.storage.buffer, s.clone())?;
            e.deps.set(s.deps.get().cloned().unwrap_or_default()).ok();
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
