//! First uploads of textures picked by index on a copy queue of their own
//! (`VICTORY_UPLOAD_AHEAD`): read, detiled and copied by threads of ours as
//! soon as the texture pass has made them, and the slots that wait for them
//! shown when the copy has run. Recorded into the pass's list, as every other
//! upload is, the copy ran with the frame that needs it, so a slot that shows
//! its texture only after its upload ([`crate::descriptor::host::show_after`])
//! was black or grey in that frame, and in the next.
//!
//! Only a texture that no list has used is taken: made in the last second,
//! in the common state, and never asked for an upload before
//! ([`crate::resource::HostTexture::ahead`]). It stays in the common state
//! (a copy queue's copies leave it there), from which the card's shaders read
//! it as any texture.

use std::collections::VecDeque;
use std::sync::atomic::{
    AtomicU8, AtomicU64,
    Ordering::{AcqRel, Acquire, Relaxed, Release},
};
use std::sync::{Arc, Condvar, Mutex, OnceLock};

use windows::Win32::Graphics::Direct3D12::*;
use windows::core::Interface;

use super::Staging;
use crate::object::This;
use crate::resource::HostTexture;

/// [`HostTexture::ahead`]: no upload was asked for yet; one is being staged
/// (`CLAIMING`); one is on its way here (`PENDING`); it has run (`DONE`); one
/// was recorded into a list (`LISTED`).
pub(crate) const NONE: u8 = 0;
const PENDING: u8 = 1;
const DONE: u8 = 2;
pub(crate) const LISTED: u8 = 3;
const CLAIMING: u8 = 4;

/// Bytes of title memory on their way at most: past it uploads go into the
/// lists, as before.
const MOST: u64 = 96 << 20;
/// Bytes and textures of one submission at most.
const BATCH: (u64, usize) = (16 << 20, 64);

/// Whether it is on (`VICTORY_UPLOAD_AHEAD`; off).
pub(crate) fn on() -> bool {
    #[cfg(feature = "diagnostics")]
    if let Some(on) = crate::ab::pick("VICTORY_UPLOAD_AHEAD") {
        return on;
    }
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_UPLOAD_AHEAD").is_some_and(|v| v != "0"))
}

enum Callbacks {
    Waiting(Vec<Box<dyn FnOnce(bool) + Send>>),
    Done,
    Failed,
}

/// An upload on its way.
pub(crate) struct Job {
    staging: Arc<Staging>,
    copy: Arc<HostTexture>,
    asked: std::time::Instant,
    /// What to do when it has run (`Done`: run with `true`; `Failed`: run with `false`).
    then: Mutex<Callbacks>,
}

impl Job {
    /// The staging buffer this ahead job will refresh and upload.
    pub(crate) fn staging(&self) -> &Arc<Staging> {
        &self.staging
    }

    /// Calls `f` when the copy has run: now if it has, never if it failed.
    pub(crate) fn then(&self, f: impl FnOnce() + Send + 'static) {
        self.on_complete(move |ok| {
            if ok {
                f();
            }
        });
    }

    /// Calls `f(true)` when the copy has landed on the GPU, or `f(false)` if
    /// the job failed before submission.
    pub(crate) fn on_complete(&self, f: impl FnOnce(bool) + Send + 'static) {
        let mut then = self.then.lock().unwrap_or_else(|e| e.into_inner());
        match &mut *then {
            Callbacks::Waiting(list) => list.push(Box::new(f)),
            Callbacks::Done => {
                drop(then);
                f(true);
            }
            Callbacks::Failed => {
                drop(then);
                f(false);
            }
        }
    }
}

/// An owned newly claimed ahead job that is guaranteed to be dispatched via
/// [`send`] (either explicitly via [`Self::send`] after attaching callbacks, or
/// automatically on drop) so it can never be stranded in `ON_THEIR_WAY`.
pub(crate) struct NewJob(Option<Arc<Job>>);

impl NewJob {
    pub(crate) fn job(&self) -> &Arc<Job> {
        self.0.as_ref().expect("NewJob still owned before send")
    }

    pub(crate) fn send(mut self) {
        if let Some(job) = self.0.take() {
            send(&job);
        }
    }
}

impl Drop for NewJob {
    fn drop(&mut self) {
        if let Some(job) = self.0.take() {
            send(&job);
        }
    }
}

/// What a texture asked for here gets.
pub(crate) enum Taken {
    /// A job of its own, dispatched via [`NewJob::send`] (or on drop) once
    /// callbacks are attached.
    New(NewJob),
    /// One already on its way, or that has run.
    Known(Arc<Job>),
}

impl Taken {
    pub(crate) fn job(&self) -> &Arc<Job> {
        match self {
            Self::New(new_job) => new_job.job(),
            Self::Known(job) => job,
        }
    }
}

static QUEUE: (Mutex<VecDeque<Arc<Job>>>, Condvar) = (Mutex::new(VecDeque::new()), Condvar::new());
/// Those on their way, by host copy ([`HostTexture::serial`]).
static ON_THEIR_WAY: Mutex<Vec<(u64, Arc<Job>)>> = Mutex::new(Vec::new());
static LANDED: Condvar = Condvar::new();
static BYTES: AtomicU64 = AtomicU64::new(0);
/// Since the start: uploads, bytes, microseconds from the asking until the
/// copy had run, the longest; askings turned down; waits of lists.
static STATS: [AtomicU64; 6] = [const { AtomicU64::new(0) }; 6];

/// Takes the first upload of `resource`'s levels from `first` on, `there` of
/// whose bytes are mapped (`u64::MAX`: not known). `None`: the list's.
pub(crate) fn take(resource: This, first: u32, there: u64) -> Option<Taken> {
    if on() { claim(resource, first, there) } else { None }
}

fn claim(resource: This, first: u32, there: u64) -> Option<Taken> {
    workers()?;
    let turned_down = || {
        STATS[4].fetch_add(1, Relaxed);
        None
    };
    let copy = crate::resource::host_copy(resource)?;
    match copy.ahead.load(Acquire) {
        CLAIMING => {
            let mut way = ON_THEIR_WAY.lock().unwrap_or_else(|e| e.into_inner());
            while copy.ahead.load(Acquire) == CLAIMING {
                way = LANDED.wait(way).unwrap_or_else(|e| e.into_inner());
            }
            if copy.ahead.load(Acquire) == PENDING {
                return way.iter().find(|(serial, _)| *serial == copy.serial).map(|(_, job)| Taken::Known(job.clone()));
            }
            return turned_down();
        }
        PENDING => {
            let way = ON_THEIR_WAY.lock().unwrap_or_else(|e| e.into_inner());
            return way.iter().find(|(serial, _)| *serial == copy.serial).map(|(_, job)| Taken::Known(job.clone()));
        }
        _ => {}
    }
    // (`VICTORY_RESIDENCY=1`: less on its way with 32 GiB of RAM or less,
    // none while commit is short.)
    let most = if crate::residency::limited_host_ram() { 32 << 20 } else { MOST };
    if !crate::residency::allow_staging_growth(false)
        || BYTES.load(Relaxed) >= most
        || copy.flags() & 0x7 != 0
        || copy.state.load(Relaxed) != 0
        || copy.grown.load(Relaxed)
        || copy.tiled.is_some()
        || copy.holds.lock().unwrap_or_else(|e| e.into_inner()).is_some()
        || !crate::pool::fresh(&copy.texture)
        || super::host_wrote(resource as usize)
        || crate::resource::alias_of(resource).is_some()
    {
        return turned_down();
    }
    if copy.ahead.compare_exchange(NONE, CLAIMING, AcqRel, Acquire).is_err() {
        return turned_down();
    }
    struct ClaimGuard<'a> {
        copy: &'a HostTexture,
        armed: bool,
    }
    impl Drop for ClaimGuard<'_> {
        fn drop(&mut self) {
            if self.armed {
                super::PICKING.set(None);
                let _way = ON_THEIR_WAY.lock().unwrap_or_else(|e| e.into_inner());
                self.copy.ahead.store(LISTED, Release);
                LANDED.notify_all();
            }
        }
    }
    let mut claim_guard = ClaimGuard { copy: &copy, armed: true };
    super::PICKING.set(Some(there));
    // Build a private staging buffer without inserting into global `STAGING`,
    // so no concurrent pass can observe `is_staged(resource) == true` and
    // misclassify this one-shot ahead upload as table-managed.
    let staging = super::staging_private(resource, &copy, first);
    super::PICKING.set(None);
    let Some(staging) = staging.filter(|s| {
        !s.on_demand && s.first == s.level0 && !s.uploaded.load(Acquire) && s.once.load(Relaxed)
    }) else {
        drop(claim_guard);
        return turned_down();
    };
    if super::no_wait() != 0 && super::fill_unlocked() {
        staging.expected.store(if there == u64::MAX { super::mapped_bytes(staging.memory, staging.read) } else { there }, Relaxed);
    }
    let job = Arc::new(Job {
        staging,
        copy: copy.clone(),
        asked: std::time::Instant::now(),
        then: Mutex::new(Callbacks::Waiting(Vec::new())),
    });
    BYTES.fetch_add(job.staging.read as u64, Relaxed);
    {
        let mut way = ON_THEIR_WAY.lock().unwrap_or_else(|e| e.into_inner());
        way.push((job.copy.serial, job.clone()));
        job.copy.ahead.store(PENDING, Release);
        claim_guard.armed = false;
        LANDED.notify_all();
    }
    Some(Taken::New(NewJob(Some(job))))
}

/// Sends a job of [`take`] on its way.
pub(crate) fn send(job: &Arc<Job>) {
    QUEUE.0.lock().unwrap_or_else(|e| e.into_inner()).push_back(job.clone());
    QUEUE.1.notify_one();
}

/// A list is about to record an upload into `copy`: one on its way here runs
/// first, and no other is taken after it.
pub(crate) fn listed(copy: &HostTexture) {
    match copy.ahead.compare_exchange(NONE, LISTED, AcqRel, Acquire) {
        Err(CLAIMING) | Err(PENDING) => {}
        _ => return,
    }
    STATS[5].fetch_add(1, Relaxed);
    let mut way = ON_THEIR_WAY.lock().unwrap_or_else(|e| e.into_inner());
    while copy.ahead.load(Acquire) == CLAIMING
        || way.iter().any(|(serial, _)| *serial == copy.serial)
    {
        way = LANDED.wait(way).unwrap_or_else(|e| e.into_inner());
    }
}

/// The copy queue and its threads, made on first use.
fn workers() -> Option<&'static ID3D12CommandQueue> {
    /// Free-threaded.
    struct Queue(ID3D12CommandQueue);
    // SAFETY: a command queue is free-threaded.
    unsafe impl Send for Queue {}
    // SAFETY: as above.
    unsafe impl Sync for Queue {}
    static WORKERS: OnceLock<Option<Queue>> = OnceLock::new();
    WORKERS
        .get_or_init(|| {
            let host = crate::host::host()?;
            // SAFETY: plain object creation.
            let queue: ID3D12CommandQueue = unsafe {
                host.device
                    .CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC {
                        Type: D3D12_COMMAND_LIST_TYPE_COPY,
                        ..Default::default()
                    })
                    .ok()?
            };
            for n in 0..2 {
                let worker = Worker::new(&host.device, queue.clone()).ok()?;
                std::thread::Builder::new().name(format!("uploads ahead {n}")).spawn(move || worker.work()).ok()?;
            }
            Some(Queue(queue))
        })
        .as_ref()
        .map(|q| &q.0)
}

struct Worker {
    device: ID3D12Device,
    queue: ID3D12CommandQueue,
    allocator: ID3D12CommandAllocator,
    list: ID3D12GraphicsCommandList,
    fence: ID3D12Fence,
}

// SAFETY: a worker's objects are its thread's own; the device and queue are free-threaded.
unsafe impl Send for Worker {}

impl Worker {
    fn make_list(
        device: &ID3D12Device,
    ) -> windows::core::Result<(ID3D12CommandAllocator, ID3D12GraphicsCommandList)> {
        // SAFETY: plain object creation.
        unsafe {
            let allocator: ID3D12CommandAllocator =
                device.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COPY)?;
            let list: ID3D12GraphicsCommandList =
                device.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_COPY, &allocator, None)?;
            list.Close()?;
            Ok((allocator, list))
        }
    }

    fn new(device: &ID3D12Device, queue: ID3D12CommandQueue) -> windows::core::Result<Worker> {
        let (allocator, list) = Self::make_list(device)?;
        // SAFETY: plain object creation.
        let fence = unsafe { device.CreateFence(0, D3D12_FENCE_FLAG_NONE)? };
        Ok(Worker {
            device: device.clone(),
            queue,
            allocator,
            list,
            fence,
        })
    }

    fn work(mut self) {
        let mut value = 0;
        loop {
            let mut jobs = Vec::new();
            {
                let mut queue = QUEUE.0.lock().unwrap_or_else(|e| e.into_inner());
                while queue.is_empty() {
                    queue = QUEUE.1.wait(queue).unwrap_or_else(|e| e.into_inner());
                }
                let mut bytes = 0;
                while bytes < BATCH.0 && jobs.len() < BATCH.1
                    && let Some(job) = queue.pop_front()
                {
                    bytes += job.staging.read as u64;
                    jobs.push(job);
                }
            }
            value += 1;
            match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| self.copy(&jobs, value))) {
                Ok(Ok(())) => finish(jobs, true),
                Ok(Err(e)) => {
                    tracing::warn!(error = %e, "uploads ahead did not run");
                    finish(jobs, false);
                    // A failed `Close` or `Reset` can leave the command list poisoned;
                    // replace the allocator/list pair before taking another batch.
                    match Self::make_list(&self.device) {
                        Ok((allocator, list)) => {
                            self.allocator = allocator;
                            self.list = list;
                        }
                        Err(recreate) => {
                            crate::host::fail(format_args!(
                                "uploads ahead copy list failed ({e}) and could not be recreated ({recreate})"
                            ));
                        }
                    }
                }
                Err(_) => {
                    finish(jobs, false);
                    crate::host::fail(format_args!("uploads ahead worker panicked"));
                }
            }
        }
    }

    fn copy(&self, jobs: &[Arc<Job>], value: u64) -> windows::core::Result<()> {
        // SAFETY: this worker's allocator and list, whose last submission has
        // run; the jobs hold what the list names until after the wait.
        let cmd: ID3D12CommandList = unsafe {
            self.allocator.Reset()?;
            self.list.Reset(&self.allocator, None)?;
            for job in jobs {
                job.staging.refresh();
                job.record(&self.list);
            }
            self.list.Close()?;
            self.list.cast()?
        };
        // SAFETY: closed command list and fence owned by this worker.
        unsafe {
            self.queue.ExecuteCommandLists(&[Some(cmd)]);
            if let Err(e) = self.queue.Signal(&self.fence, value) {
                crate::host::fail(format_args!(
                    "uploads ahead copy-queue Signal failed at fence {value} ({e})"
                ));
            }
            let early = self.fence.GetCompletedValue();
            if early == u64::MAX {
                crate::host::fail(format_args!(
                    "uploads ahead copy-queue device removed before wait (fence={value}, completed=u64::MAX)"
                ));
            }
            if early < value {
                // (Without an event: this waits synchronously.)
                if let Err(e) = self
                    .fence
                    .SetEventOnCompletion(value, windows::Win32::Foundation::HANDLE::default())
                {
                    crate::host::fail(format_args!(
                        "uploads ahead copy-queue SetEventOnCompletion failed at fence {value} ({e})"
                    ));
                }
                let after = self.fence.GetCompletedValue();
                if after == u64::MAX || after < value {
                    crate::host::fail(format_args!(
                        "uploads ahead copy-queue wait did not confirm fence {value} (completed={after:#x})"
                    ));
                }
            }
        }
        Ok(())
    }
}

impl Job {
    /// # Safety
    /// `list` is an open copy list, and the job lives until it has run.
    unsafe fn record(&self, list: &ID3D12GraphicsCommandList) {
        let (s, texture) = (&self.staging, &self.copy.texture);
        for (i, footprint) in s.footprints.iter().enumerate() {
            let to = D3D12_TEXTURE_COPY_LOCATION {
                // SAFETY: a borrowed pointer the location never releases.
                pResource: unsafe { std::mem::transmute_copy(texture) },
                Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
                Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: s.first - s.level0 + i as u32 },
            };
            let from = D3D12_TEXTURE_COPY_LOCATION {
                // SAFETY: as above.
                pResource: unsafe { std::mem::transmute_copy(&s.buffer) },
                Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
                Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                    PlacedFootprint: D3D12_PLACED_SUBRESOURCE_FOOTPRINT { Offset: s.base + footprint.Offset, ..*footprint },
                },
            };
            // SAFETY: the caller's.
            unsafe { list.CopyTextureRegion(&to, 0, 0, 0, &from, None) };
        }
    }
}

/// The jobs' copies have run (`ok`), or failed and must fall back to the list.
fn finish(jobs: Vec<Arc<Job>>, ok: bool) {
    for job in &jobs {
        if ok {
            job.staging.uploaded.store(true, Release);
            job.staging.dirty.store(false, Release);
            job.staging.note_held();
            job.copy.ahead.store(DONE, Release);
        } else {
            job.copy.ahead.store(LISTED, Release);
        }
        BYTES.fetch_sub(job.staging.read as u64, Relaxed);
    }
    {
        let mut way = ON_THEIR_WAY.lock().unwrap_or_else(|e| e.into_inner());
        way.retain(|(serial, _)| !jobs.iter().any(|job| job.copy.serial == *serial));
        LANDED.notify_all();
    }
    if !ok {
        for job in &jobs {
            let prev = std::mem::replace(
                &mut *job.then.lock().unwrap_or_else(|e| e.into_inner()),
                Callbacks::Failed,
            );
            if let Callbacks::Waiting(list) = prev {
                for f in list {
                    f(false);
                }
            }
        }
        return;
    }
    for job in &jobs {
        let us = job.asked.elapsed().as_micros() as u64;
        let n = STATS[0].fetch_add(1, Relaxed) + 1;
        let bytes = STATS[1].fetch_add(job.staging.read as u64, Relaxed) + job.staging.read as u64;
        let all = STATS[2].fetch_add(us, Relaxed) + us;
        let most = STATS[3].fetch_max(us, Relaxed).max(us);
        if n == 1 || n.is_multiple_of(5000) {
            tracing::info!(
                "textures uploaded ahead of the lists (VICTORY_UPLOAD_AHEAD=1), since the start: {n}, {} MB, {:.1} ms from the asking until they were there, the longest {:.0}; {} turned down, {} waited for by lists",
                bytes >> 20,
                all as f64 / n as f64 / 1e3,
                most as f64 / 1e3,
                STATS[4].load(Relaxed),
                STATS[5].load(Relaxed)
            );
        }
        let prev = std::mem::replace(&mut *job.then.lock().unwrap_or_else(|e| e.into_inner()), Callbacks::Done);
        if let Callbacks::Waiting(list) = prev {
            for f in list {
                f(true);
            }
        }
    }
    crate::descriptor::host::flip_now();
}

/// Whether `resource`'s host copy was filled here: no list still brings what
/// it shows.
pub(crate) fn there(resource: This) -> bool {
    crate::resource::made_copy(resource).is_some_and(|copy| copy.ahead.load(Acquire) == DONE)
}

/// The state of [`HostTexture::ahead`] of a new host copy.
pub(crate) fn fresh() -> AtomicU8 {
    AtomicU8::new(NONE)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The texels of a placed texture arrive by the copy queue, all levels of
    /// them, and what waits for them is called then; a second asking finds the
    /// first on its way, and a later one is the list's.
    #[test]
    fn a_placed_texture_is_filled_ahead_of_the_lists() {
        use windows_sys::Win32::System::Memory::{MEM_COMMIT, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc};
        let Some(host) = crate::host::host() else { return };
        // 256x256 BC7, nine levels.
        let desc = [3u32, 0, 0, 0, 256, 0, 256, 0x9_0001, 98, 1, 0, 0x105, 0, 0];
        let whole = crate::resource::placed_bytes(&desc).unwrap() as usize;
        // SAFETY: memory of our own that stays.
        let at = unsafe { VirtualAlloc(std::ptr::null(), whole, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE) }.cast::<u8>();
        // SAFETY: as above.
        unsafe { (0..whole).for_each(|i| at.add(i).write((i * 13 + i / 509) as u8)) };
        let texture = crate::resource::texture_at(at as u64, &desc).unwrap();
        crate::resource::want_level(texture, 0);
        let copy = crate::resource::host_copy(texture).unwrap();
        if !crate::pool::fresh(&copy.texture) {
            return;
        }
        assert!(!there(texture));
        let Some(Taken::New(new_job)) = claim(texture, 0, u64::MAX) else { panic!("not taken") };
        let job = new_job.job().clone();
        let (told, heard) = std::sync::mpsc::channel();
        let also = told.clone();
        let Some(Taken::Known(same)) = claim(texture, 0, u64::MAX) else { panic!("not known") };
        assert!(Arc::ptr_eq(&job, &same));
        job.then(move || told.send(1).unwrap());
        same.then(move || also.send(2).unwrap());
        new_job.send();
        let wait = std::time::Duration::from_secs(20);
        assert_eq!((heard.recv_timeout(wait), heard.recv_timeout(wait)), (Ok(1), Ok(2)));
        assert!(there(texture) && claim(texture, 0, u64::MAX).is_none());
        // Level 0 as the card has it, and as the layout says.
        let layout = crate::resource::texture_memory(texture)
            .and_then(|(_, d)| crate::resource::texture_layout(&d).map(|t| t.planes[0].layout.clone()))
            .unwrap();
        let (pitch, rows) = (1024usize, 64usize);
        let mut expected = vec![0u8; pitch * rows];
        // SAFETY: the memory above.
        let tiled = unsafe { std::slice::from_raw_parts(at, whole) };
        layout.detile(tiled, 0, 0, &mut expected, pitch, pitch * rows).unwrap();
        let back = host.read_back_buffer((pitch * rows) as u64).unwrap();
        let list = host.new_list().unwrap();
        list.close();
        list.reset();
        let footprint = D3D12_PLACED_SUBRESOURCE_FOOTPRINT {
            Offset: 0,
            Footprint: D3D12_SUBRESOURCE_FOOTPRINT {
                Format: windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT(98),
                Width: 256,
                Height: 256,
                Depth: 1,
                RowPitch: pitch as u32,
            },
        };
        let to = D3D12_TEXTURE_COPY_LOCATION {
            // SAFETY: a borrowed pointer the location never releases.
            pResource: unsafe { std::mem::transmute_copy(&back) },
            Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { PlacedFootprint: footprint },
        };
        let from = D3D12_TEXTURE_COPY_LOCATION {
            // SAFETY: as above.
            pResource: unsafe { std::mem::transmute_copy(&copy.texture) },
            Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: 0 },
        };
        // SAFETY: an open list, live resources; the mapping spans the buffer.
        unsafe {
            list.list.CopyTextureRegion(&to, 0, 0, 0, &from, None);
            list.close();
            host.execute(&[&list]);
            let mut data = std::ptr::null_mut();
            back.Map(0, None, Some(&mut data)).unwrap();
            let seen = std::slice::from_raw_parts(data.cast::<u8>(), pitch * rows);
            assert!(seen == expected.as_slice(), "level 0's blocks");
            back.Unmap(0, None);
        }
    }
}
