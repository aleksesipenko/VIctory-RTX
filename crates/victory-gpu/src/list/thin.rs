//! List calls that only write down their arguments (`VICTORY_THIN=1`); what
//! they mean is worked out by threads of ours, beside the title's.
//!
//! Why: the title makes 45,000 to 60,000 list calls a frame. Each was
//! translated inside the call, 3.5 us of it, 160 to 190 ms of processor time a
//! frame on the title's five recording threads; its own code between the calls
//! is a tenth of that. Nothing else of the title's frame could go on
//! meanwhile: recording took 55 of 65 ms, and behind it the card stood idle
//! for half the frame (timeline waits-3). On the console such a call writes a
//! few words into the list's memory and the card reads them when the list
//! runs. So it is here now: a call appends its arguments (and copies of what
//! they point at) to a stream of words; threads of ours run the stream, call
//! by call and in order, through the list methods there were before, on a list
//! of ours ([`Recording::back`]) that the title never sees. They begin while
//! the title still records (every [`CHUNK`] words are handed over), several
//! lists at a time. A submission goes into the line of submissions
//! (`crate::queue::behind`) as it was made, by the recordings' names; the
//! thread that serves the line goes on with it when they are translated.
//!
//! What may be looked at later than before, and why that is all right: all
//! that the card of the console looks at when the list runs (descriptor heaps,
//! constants, upload memory, indirect arguments): it must stay as it is until
//! the list has run, and no fence behind the list is set before. What D3D12
//! reads at the call is copied at the call: the arrays and structures passed,
//! and the render target and depth views, whose slots may be written again
//! right after. Objects named (pipelines, resources) must live until the list
//! has run; a recording the title resets or releases without submitting it is
//! not run any further.

use super::*;
use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering::*};
use std::sync::{Arc, Condvar, Mutex, OnceLock};

use crate::descriptor::RtvRecord;
use crate::draw::{IndexView, VertexView};
use windows::Win32::Foundation::RECT;
use windows::Win32::Graphics::Direct3D12::D3D12_VIEWPORT;

/// Words after which a stream is handed to the translating threads.
const CHUNK: usize = 1024;

/// Whether list calls are only written down (`VICTORY_THIN=1`; needs the line
/// of submissions, which it turns on).
pub(crate) fn on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        wanted() && crate::host::host().is_some() && crate::queue::line_on() && {
            start();
            true
        }
    })
}

/// Whether it is asked for.
pub(crate) fn wanted() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_THIN").is_some_and(|v| v == "1"))
}

/// What a title list keeps: the recording being written and the words of it
/// not handed over yet.
#[derive(Default)]
pub(super) struct Front {
    recording: Option<Arc<Recording>>,
    words: Vec<u64>,
    closed: bool,
    submitted: bool,
}

impl Drop for Front {
    fn drop(&mut self) {
        self.leave();
    }
}

/// Frees any unconsumed `UPLOAD` `Box<Upload>` pointers left in a stream that
/// was never passed through [`translate`].
fn drop_unconsumed_uploads(stream: &[u64]) {
    let mut i = 0;
    while i < stream.len() {
        let (op, len) = (stream[i] & 0xffff, (stream[i] >> 16) as usize);
        if op == UPLOAD && len >= 1 && i + 1 < stream.len() {
            // SAFETY: `allocate_upload` created this `Box<Upload>` once, and
            // this stream was never translated.
            unsafe { drop(Box::from_raw(stream[i + 1] as *mut Upload)) };
        }
        i += 1 + len;
    }
}

impl Front {
    /// Lets go of the recording there is: one never submitted is not run any
    /// further, and we wait for any in-flight worker step to exit before
    /// returning so the title cannot free referenced objects under it.
    fn leave(&mut self) {
        if let Some(recording) = self.recording.take()
            && !self.submitted
        {
            recording.abandon();
        }
        drop_unconsumed_uploads(&self.words);
        self.words.clear();
    }

    /// Begins a recording, with `pipeline` bound.
    pub(super) fn begin(&mut self, kind: u32, pipeline: This) {
        self.leave();
        self.recording = Some(Arc::new(Recording::new(kind)));
        self.closed = false;
        self.submitted = false;
        self.put(RESET, &[pipeline as u64]);
    }

    #[inline]
    fn put(&mut self, op: u64, args: &[u64]) {
        if self.closed || self.recording.is_none() {
            return;
        }
        self.words.push(op | (args.len() as u64) << 16);
        self.words.extend_from_slice(args);
        if self.words.len() >= CHUNK {
            self.hand_over();
        }
    }

    /// [`Self::put`] of some words and then the bytes of some things.
    fn put_with(&mut self, op: u64, args: &[u64], blobs: &[&[u8]]) {
        if self.closed || self.recording.is_none() {
            return;
        }
        let at = self.words.len();
        self.words.push(0);
        self.words.extend_from_slice(args);
        for blob in blobs {
            let from = self.words.len();
            self.words.resize(from + blob.len().div_ceil(8), 0);
            // SAFETY: the words just made hold the bytes; plain data.
            unsafe {
                std::ptr::copy_nonoverlapping(blob.as_ptr(), self.words.as_mut_ptr().add(from).cast(), blob.len())
            };
        }
        let len = (self.words.len() - at - 1) as u64;
        self.words[at] = op | len << 16;
        if self.words.len() >= CHUNK {
            self.hand_over();
        }
    }

    fn hand_over(&mut self) {
        let Some(recording) = &self.recording else { return };
        if self.words.is_empty() {
            return;
        }
        let fresh = held(&WORDS).pop().unwrap_or_else(|| Vec::with_capacity(CHUNK + 256));
        let words = std::mem::replace(&mut self.words, fresh);
        CHUNKS.fetch_add(1, Relaxed);
        let schedule = {
            let mut inner = held(&recording.inner);
            inner.chunks.push_back(words);
            !std::mem::replace(&mut inner.scheduled, true)
        };
        if schedule {
            let mut ready = held(&READY);
            if recording.urgent.load(Acquire) {
                ready.push_front(recording.clone());
            } else {
                ready.push_back(recording.clone());
            }
            drop(ready);
            WORK.notify_one();
        }
    }

    fn close(&mut self) {
        if self.closed {
            return;
        }
        self.put(CLOSE, &[]);
        self.closed = true;
        self.hand_over();
    }
}

/// Streams' memory to use again.
static WORDS: Mutex<Vec<Vec<u64>>> = Mutex::new(Vec::new());
/// Recordings with words to translate.
static READY: Mutex<VecDeque<Arc<Recording>>> = Mutex::new(VecDeque::new());
static WORK: Condvar = Condvar::new();
/// Lists of ours nobody uses, closed.
static BACKS: Mutex<Vec<usize>> = Mutex::new(Vec::new());

/// Drops pooled backing lists down to `keep` (via `list_release` so refcounted
/// `Object` and DXR fields clean up properly).
pub(crate) fn trim_backs(keep: usize) {
    let gone: Vec<usize> = {
        let mut backs = held(&BACKS);
        let n = backs.len().saturating_sub(keep);
        backs.drain(..n).collect()
    };
    for ptr in gone {
        super::list_release(ptr as This);
    }
}
/// Since the last line of the log: streams handed over; calls translated and
/// the microseconds that took (all threads), recordings translated, and
/// recordings let go of in the middle.
static CHUNKS: AtomicU64 = AtomicU64::new(0);
static STATS: [AtomicU64; 4] = [const { AtomicU64::new(0) }; 4];

fn held<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

/// One recording of a title list: what was written of it and not translated
/// yet, and the list of ours it goes to.
pub(crate) struct Recording {
    kind: u32,
    inner: Mutex<Inner>,
    /// Translated to its end.
    done: AtomicBool,
    moved: Condvar,
    abandoned: AtomicBool,
    /// Already submitted to a queue (`submitted`); prioritize ahead of speculative work.
    urgent: AtomicBool,
}

#[derive(Default)]
struct Inner {
    chunks: VecDeque<Vec<u64>>,
    dxr: VecDeque<crate::dxr::Captured>,
    /// Streams translated (commands may name places in them until the end).
    kept: Vec<Vec<u64>>,
    /// It is among [`READY`] or being translated.
    scheduled: bool,
    /// A worker thread is currently executing a step of [`Recording::run`]
    /// outside this lock (`pipeline::bound` or `translate`).
    running: bool,
    /// The list of ours (0: none yet).
    back: usize,
    /// That list was closed by the stream.
    closed: bool,
}

impl Recording {
    fn new(kind: u32) -> Self {
        Recording {
            kind,
            inner: Mutex::default(),
            done: AtomicBool::new(false),
            moved: Condvar::new(),
            abandoned: AtomicBool::new(false),
            urgent: AtomicBool::new(false),
        }
    }

    /// Marks this unsubmitted recording abandoned and waits until any
    /// in-flight worker step (`pipeline::bound` or `translate`) finishes, so
    /// the title cannot free objects referenced by this recording while a
    /// worker still dereferences them.
    fn abandon(&self) {
        let mut inner = held(&self.inner);
        self.abandoned.store(true, Release);
        while inner.running {
            inner = self.moved.wait(inner).unwrap_or_else(|e| e.into_inner());
        }
    }

    pub(crate) fn done(&self) -> bool {
        self.done.load(Acquire)
    }

    /// Waits until it is translated.
    pub(crate) fn wait(&self) {
        let mut inner = held(&self.inner);
        while !self.done() {
            inner = self.moved.wait(inner).unwrap_or_else(|e| e.into_inner());
        }
    }

    /// The list of ours it went to (null: it was never begun).
    pub(crate) fn back(&self) -> This {
        held(&self.inner).back as This
    }

    /// Translates what there is.
    fn run(&self) {
        self.run_with(crate::pipeline::bound, |back, chunk, left, dxr| {
            translate(back, chunk, left, dxr)
        });
    }

    fn run_with<P, T>(&self, mut on_pipeline: P, mut on_translate: T)
    where
        P: FnMut(This),
        T: FnMut(
            This,
            &[u64],
            &dyn Fn() -> bool,
            &mut dyn FnMut() -> Option<crate::dxr::Captured>,
        ) -> (u64, bool),
    {
        struct RunningGuard<'a>(&'a Recording);
        impl Drop for RunningGuard<'_> {
            fn drop(&mut self) {
                let mut inner = held(&self.0.inner);
                if inner.running {
                    inner.running = false;
                    self.0.moved.notify_all();
                }
            }
        }
        enum Step {
            Empty { chunk: Vec<u64>, pipeline: This },
            Translate { chunk: Vec<u64>, back: This },
        }
        loop {
            let step = {
                let mut inner = held(&self.inner);
                let Some(chunk) = inner.chunks.pop_front() else {
                    inner.scheduled = false;
                    return;
                };
                // Empty RESET+CLOSE recording: finish without allocating a backing
                // host command list (GPL-3.0 provenance: maren-halv/VIctory-DXR f1d9f52).
                if inner.back == 0
                    && !inner.closed
                    && inner.chunks.is_empty()
                    && inner.dxr.is_empty()
                    && chunk.len() == 3
                    && chunk[0] == (RESET | (1 << 16))
                    && chunk[2] == CLOSE
                {
                    if self.abandoned.load(Acquire) {
                        // Abandoned before we began: do not dereference `pipeline`.
                        inner.closed = true;
                        inner.scheduled = false;
                        self.done.store(true, Release);
                        self.moved.notify_all();
                        drop(inner);
                        give_back(vec![chunk]);
                        STATS[2].fetch_add(1, Relaxed);
                        crate::queue::line_moved();
                        return;
                    }
                    // Mark `running` under `inner` before unlocking so a concurrent
                    // `abandon()` waits until `on_pipeline` has returned.
                    inner.running = true;
                    Step::Empty { pipeline: chunk[1] as This, chunk }
                } else {
                    if inner.back == 0 {
                        inner.back = back_list(self.kind) as usize;
                    }
                    inner.running = true;
                    Step::Translate { chunk, back: inner.back as This }
                }
            };
            let _running = RunningGuard(self);
            match step {
                Step::Empty { chunk, pipeline } => {
                    // Record pipeline bind while `inner.running` blocks `abandon()`
                    // and before publishing `done`, so neither queue retirement nor
                    // abandonment can free `pipeline` under us.
                    on_pipeline(pipeline);
                    give_back(vec![chunk]);
                    let mut inner = held(&self.inner);
                    inner.closed = true;
                    inner.scheduled = false;
                    inner.running = false;
                    self.done.store(true, Release);
                    self.moved.notify_all();
                    drop(inner);
                    STATS[2].fetch_add(1, Relaxed);
                    crate::queue::line_moved();
                    return;
                }
                Step::Translate { chunk, back } => {
                    let began = std::time::Instant::now();
                    let left = || self.abandoned.load(Acquire);
                    let mut dxr = || held(&self.inner).dxr.pop_front();
                    let (calls, closed) = on_translate(back, &chunk, &left, &mut dxr);
                    STATS[0].fetch_add(calls, Relaxed);
                    STATS[1].fetch_add(began.elapsed().as_micros() as u64, Relaxed);
                    let mut inner = held(&self.inner);
                    inner.kept.push(chunk);
                    inner.running = false;
                    if closed {
                        inner.closed = true;
                        let kept = std::mem::take(&mut inner.kept);
                        self.done.store(true, Release);
                        self.moved.notify_all();
                        drop(inner);
                        give_back(kept);
                        STATS[2].fetch_add(1, Relaxed);
                        crate::queue::line_moved();
                    } else {
                        self.moved.notify_all();
                    }
                }
            }
        }
    }
}

impl Drop for Recording {
    fn drop(&mut self) {
        let inner = self.inner.get_mut().unwrap_or_else(|e| e.into_inner());
        for chunk in &inner.chunks {
            drop_unconsumed_uploads(chunk);
        }
        give_back(inner.chunks.drain(..).chain(std::mem::take(&mut inner.kept)).collect());
        if inner.back != 0 {
            if !inner.closed {
                // Left in the middle.
                STATS[3].fetch_add(1, Relaxed);
                super::close(inner.back as This);
            }
            let release_immediately = crate::dxr::requested()
                && (self.abandoned.load(Acquire) || super::cut_list(inner.back as This));
            // Earlier DXR pieces may already pin GPU inputs: release those
            // backing lists immediately instead of parking them in BACKS.
            // Otherwise cap pooled backing lists at 64 to avoid unbounded
            // memory growth (GPL-3.0 provenance: maren-halv/VIctory-DXR d5efb36).
            release_or_park_back(&BACKS, inner.back, release_immediately);
        }
    }
}

fn release_or_park_back(pool: &Mutex<Vec<usize>>, back: usize, release_immediately: bool) {
    if release_immediately {
        super::list_release(back as This);
        return;
    }
    let mut backs = held(pool);
    if backs.len() < 64 {
        backs.push(back);
    } else {
        drop(backs);
        super::list_release(back as This);
    }
}

fn give_back(streams: Vec<Vec<u64>>) {
    let mut words = held(&WORDS);
    for mut s in streams {
        if words.len() < 1024 && s.capacity() <= 16 * CHUNK {
            s.clear();
            words.push(s);
        }
    }
}

/// A list of ours, closed, with `refs == 1` so `list_release` drops `Box<List>`.
fn back_list(kind: u32) -> This {
    back_list_in(&BACKS, kind, super::boxed_closed)
}

fn back_list_in(
    pool: &Mutex<Vec<usize>>,
    kind: u32,
    make: impl FnOnce(u32) -> Box<List>,
) -> This {
    if let Some(back) = held(pool).pop() {
        // SAFETY: a list of ours nobody uses; its refcount is already 1.
        unsafe { (*(back as *mut List)).kind = kind };
        return back as This;
    }
    let list: This = Box::into_raw(make(kind)).cast_const().cast();
    // `Object::new` initializes `refs` to 0 (`hand_out` normally bumps it to 1);
    // give internal backing lists their single owning reference so
    // `super::list_release` -> `object::release_boxed::<List>` actually drops
    // the `Box<List>` instead of returning `0` on `0u32.checked_sub(1)`.
    object::add_ref(list);
    list
}

fn start() {
    let threads = std::env::var("VICTORY_THIN_THREADS").ok().and_then(|v| v.trim().parse().ok()).unwrap_or(8usize).clamp(1, 32);
    for i in 0..threads {
        let worker = move || {
            crate::cores::keep_many_on_fast_cores("victory-list");
            crate::cores::prioritize("victory-list", crate::cores::Role::Translator);
            crate::queue::no_drain_here();
            loop {
                let recording = {
                    let mut ready = held(&READY);
                    loop {
                        match ready.pop_front() {
                            Some(r) => break r,
                            None => ready = WORK.wait(ready).unwrap_or_else(|e| e.into_inner()),
                        }
                    }
                };
                if std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| recording.run())).is_err() {
                    crate::host::fail(format_args!("a thread that translates lists panicked"));
                }
            }
        };
        if let Err(e) = std::thread::Builder::new().name(format!("victory-list-{i}")).spawn(worker) {
            tracing::warn!(error = %e, "a thread less to translate lists");
        }
    }
    tracing::info!(threads, "list calls are written down and translated by threads of ours (VICTORY_THIN=1)");
}

/// The recordings of submitted lists, in order (one that was not begun is left
/// out).
pub(crate) fn submitted(lists: &[This]) -> Arc<[Arc<Recording>]> {
    let recordings: Arc<[Arc<Recording>]> = lists
        .iter()
        .filter(|l| !l.is_null())
        .filter_map(|&l| {
            // SAFETY: the title passes its lists, which nobody records into now.
            let front = unsafe { &mut (*l.cast_mut().cast::<List>()).thin };
            if let Some(recording) = &front.recording {
                recording.urgent.store(true, Release);
            }
            front.close();
            front.submitted = true;
            front.recording.clone()
        })
        .collect();
    // Promote any already-queued submitted recordings ahead of unsubmitted work.
    let pending: Vec<Arc<Recording>> = recordings.iter().filter(|r| !r.done()).cloned().collect();
    if !pending.is_empty() {
        let mut ready = held(&READY);
        let mut promoted = 0usize;
        for rec in pending.into_iter().rev() {
            if let Some(pos) = ready.iter().position(|q| Arc::ptr_eq(q, &rec))
                && pos > 0
                && let Some(item) = ready.remove(pos)
            {
                ready.push_front(item);
                promoted += 1;
            }
        }
        drop(ready);
        if promoted > 0 {
            WORK.notify_all();
        }
    }
    recordings
}

/// For the frame-rate line.
pub(crate) fn line(frames: f64) {
    if !wanted() || !on() {
        return;
    }
    let s: Vec<f64> = STATS.iter().map(|s| s.swap(0, Relaxed) as f64 / frames).collect();
    tracing::info!(
        "list calls translated by threads of ours per frame (VICTORY_THIN): {:.0} in {:.1} ms (all threads), {:.1} lists, {:.1} pieces; left in the middle {:.2}",
        s[0],
        s[1] / 1000.0,
        s[2],
        CHUNKS.swap(0, Relaxed) as f64 / frames,
        s[3]
    );
}

#[inline]
fn front<'a>(this: This) -> &'a mut Front {
    // SAFETY: the title passes one of our lists; a list is recorded by one
    // thread at a time.
    unsafe { &mut (*this.cast_mut().cast::<List>()).thin }
}

/// An argument as a word.
trait Word: Copy {
    fn word(self) -> u64;
    fn unword(w: u64) -> Self;
}
macro_rules! words {
    ($($t:ty),*) => { $(impl Word for $t {
        #[inline]
        fn word(self) -> u64 { self as u64 }
        #[inline]
        fn unword(w: u64) -> Self { w as $t }
    })* };
}
words!(u8, u32, i32, u64, This);
impl Word for f32 {
    #[inline]
    fn word(self) -> u64 {
        u64::from(self.to_bits())
    }
    #[inline]
    fn unword(w: u64) -> Self {
        f32::from_bits(w as u32)
    }
}

/// The methods whose arguments are all plain values: the one the title calls,
/// and the call made of it later.
macro_rules! plain {
    ($($op:ident $name:ident = $was:path, ($($a:ident: $t:ty),*);)*) => {
        #[allow(clippy::upper_case_acronyms, non_camel_case_types)]
        #[repr(u64)]
        enum Plain { $($op),* , END }
        $(extern "system" fn $name(this: This $(, $a: $t)*) {
            front(this).put(Plain::$op as u64, &[$(Word::word($a)),*]);
        })*
        /// Whether `op` was one of them; done on `list`.
        #[inline]
        fn run_plain(op: u64, list: This, w: &[u64]) -> bool {
            match op {
                $(n if n == Plain::$op as u64 => {
                    let mut words = w.iter();
                    $(let $a: $t = Word::unword(words.next().copied().unwrap_or(0));)*
                    let _ = words;
                    $was(list $(, $a)*);
                })*
                _ => return false,
            }
            true
        }
    };
}

plain! {
    TOPOLOGY topology = state::topology, (v: u32);
    STENCIL_REF stencil_ref = state::stencil_ref, (v: u32);
    DEPTH_BOUNDS depth_bounds = state::depth_bounds, (min: f32, max: f32);
    COMPUTE_TABLE compute_table = state::compute_table, (slot: u32, handle: u64);
    GRAPHICS_TABLE graphics_table = state::graphics_table, (slot: u32, handle: u64);
    COMPUTE_CONSTANT compute_constant = state::compute_constant, (slot: u32, value: u32, offset: u32);
    GRAPHICS_CONSTANT graphics_constant = state::graphics_constant, (slot: u32, value: u32, offset: u32);
    COMPUTE_VIEW compute_view = state::compute_view, (slot: u32, address: u64);
    GRAPHICS_VIEW graphics_view = state::graphics_view, (slot: u32, address: u64);
    SET_PIPELINE set_pipeline = super::set_pipeline, (pipeline: This);
    DRAW draw = state::draw, (vertices: u32, instances: u32, first: u32, first_instance: u32);
    DRAW_INDEXED draw_indexed = state::draw_indexed, (indices: u32, instances: u32, first: u32, base: i32, first_instance: u32);
    DISPATCH dispatch = state::dispatch, (x: u32, y: u32, z: u32);
    DISPATCH_X dispatch_x = super::dispatch_x, (x: u32, y: u32, z: u32, flags: u32);
    DISPATCH_MESH dispatch_mesh = state::dispatch_mesh, (x: u32, y: u32, z: u32);
    DISPATCH_MESH_X dispatch_mesh_x = state::dispatch_mesh_x, (x: u32, flags: u64, indirect: u64, payload: u64);
    COPY_BUFFER copy_buffer_region = super::copy_buffer_region, (dst: This, dst_offset: u64, src: This, src_offset: u64, len: u64);
    COPY_RESOURCE copy_resource = super::copy_resource, (dst: This, src: This);
    COPY_RESOURCE_X copy_resource_x = super::copy_resource_x, (dst: This, src: This, flags: u32);
    COPY_MEMORY copy_memory = super::copy_memory, (dst: u64, src: u64, len: u64, flags: u32);
    RESOLVE resolve = super::resolve, (dst: This, dst_sub: u32, src: This, src_sub: u32, format: u32);
    RESOLVE_QUERIES resolve_queries = super::resolve_queries, (heap: This, kind: u32, first: u32, count: u32, buffer: This, offset: u64);
    EXECUTE_INDIRECT execute_indirect = super::execute_indirect, (signature: This, max: u32, arguments: This, offset: u64, count: This, count_offset: u64);
    WRITE32 write32 = super::write32, (address: u64, value: u32, flush: u32, flags: u32);
    WRITE64 write64 = super::write64, (address: u64, value: u64, flush: u32, flags: u32);
    WAIT_VALUE wait_value = super::wait_value, (address: u64, comparison: u32, reference: u64);
    WAIT_VALUE64 wait_value64 = super::wait_value64, (address: u64, comparison: u32, reference: u64);
    SYNC sync = super::sync, (flags: u32, control: u32, address: u64, size: u64);
}

const fn after(n: u64) -> u64 {
    Plain::END as u64 + n
}
const RESET: u64 = after(0);
const CLOSE: u64 = after(1);
const UPLOAD: u64 = after(2);
const BARRIER: u64 = after(3);
const BARRIER_GROUPS: u64 = after(4);
const VIEWPORTS: u64 = after(5);
const SCISSORS: u64 = after(6);
const BLEND_FACTOR: u64 = after(7);
const HEAPS: u64 = after(8);
const COMPUTE_CONSTANTS: u64 = after(9);
const GRAPHICS_CONSTANTS: u64 = after(10);
const INDEX_BUFFER: u64 = after(11);
const VERTEX_BUFFERS: u64 = after(12);
const TARGETS: u64 = after(13);
const CLEAR_RTV: u64 = after(14);
const CLEAR_DSV: u64 = after(15);
const COPY_TEXTURE: u64 = after(16);
const UPSCALE: u64 = after(17);
const DXR: u64 = after(18);

/// The marker and its owned payload belong to the same recording. Reset or
/// abandonment drops unconsumed batches, even before the first chunk is sent.
pub(super) fn dxr(this: This, batch: crate::dxr::Captured) {
    let f = front(this);
    if f.closed || f.recording.is_none() {
        crate::dxr::unrecorded(batch.count(), "closed thin list");
        return;
    }
    held(&f.recording.as_ref().unwrap().inner)
        .dxr
        .push_back(batch);
    f.put(DXR, &[]);
}

/// Writes the title's upscale down with the list's calls (`super::upscale`);
/// false if the list has no recording being written.
pub(super) fn upscale(this: This, up: &super::Upscale) -> bool {
    let front = front(this);
    if front.closed || front.recording.is_none() {
        return false;
    }
    front.put(UPSCALE, &(*up).to_words());
    true
}

/// The bytes of `count` things at `items` (none if null).
fn bytes_of<'a, T>(items: *const T, count: usize) -> &'a [u8] {
    if items.is_null() || count == 0 {
        return &[];
    }
    // SAFETY: the title passes `count` items.
    unsafe { std::slice::from_raw_parts(items.cast(), count * size_of::<T>()) }
}

/// Where the things are in a stream (null for none).
fn at<T>(w: &[u64], from: usize, count: usize) -> *const T {
    if count == 0 || from >= w.len() { std::ptr::null() } else { w[from..].as_ptr().cast() }
}

/// [`at`] where an empty array is not null.
fn at_or<T>(w: &[u64], from: usize) -> *const T {
    // SAFETY: at most one past the end.
    unsafe { w.as_ptr().add(from.min(w.len())).cast() }
}

/// A view's record as it is now (zeros for no view), so that its slot may be
/// written again: the copy stands for the handle.
fn record(handle: u64) -> [u64; RECORD] {
    let mut words = [0u64; RECORD];
    if handle != 0 {
        // SAFETY: a descriptor slot the title passes (they are larger).
        unsafe {
            std::ptr::copy_nonoverlapping(handle as *const u8, words.as_mut_ptr().cast(), size_of::<RtvRecord>())
        };
    }
    words
}
const RECORD: usize = size_of::<RtvRecord>().div_ceil(8);

extern "system" fn list_close(this: This) -> HRESULT {
    front(this).close();
    S_OK
}
extern "system" fn list_reset(this: This, allocator: This, pipeline: This) -> HRESULT {
    if allocator.is_null() {
        return E_INVALIDARG;
    }
    // SAFETY: the title passes one of our lists.
    let kind = unsafe { (*this.cast::<List>()).kind };
    front(this).begin(kind, pipeline);
    S_OK
}
extern "system" fn list_reset_x(this: This, allocator: This, pipeline: This, _flags: u32) -> HRESULT {
    list_reset(this, allocator, pipeline)
}
extern "system" fn allocate_upload(this: This, size: u64, alignment: u64) -> *mut u8 {
    let align = usize::try_from(alignment).unwrap_or(0).max(16);
    let Some(len) = usize::try_from(size).ok().and_then(|s| s.checked_add(align)) else {
        return std::ptr::null_mut();
    };
    if this.is_null() || !align.is_power_of_two() {
        return std::ptr::null_mut();
    }
    let front = front(this);
    if front.closed || front.recording.is_none() {
        return std::ptr::null_mut();
    }
    let Some(mut block) = Upload::new(len) else {
        return std::ptr::null_mut();
    };
    let start = block.start();
    // SAFETY: the offset is below `align`, so `size` bytes fit after it.
    let place = unsafe { start.add(start.align_offset(align)) };
    front.put(UPLOAD, &[Box::into_raw(Box::new(block)) as u64]);
    place
}
extern "system" fn barrier(this: This, count: u32, barriers: *const Barrier) -> u32 {
    if !barriers.is_null() {
        front(this).put_with(BARRIER, &[u64::from(count)], &[bytes_of(barriers, count as usize)]);
    }
    0
}
extern "system" fn barrier_groups(this: This, count: u32, groups: *const u8) {
    if super::barrier_groups_need_uav(count, groups) {
        front(this).put(BARRIER_GROUPS, &[u64::from(count)]);
    }
}
extern "system" fn viewports(this: This, count: u32, items: *const D3D12_VIEWPORT) {
    let count = if items.is_null() { 0 } else { count };
    front(this).put_with(VIEWPORTS, &[u64::from(count)], &[bytes_of(items, count as usize)]);
}
extern "system" fn scissors(this: This, count: u32, items: *const RECT) {
    let count = if items.is_null() { 0 } else { count };
    front(this).put_with(SCISSORS, &[u64::from(count)], &[bytes_of(items, count as usize)]);
}
extern "system" fn blend_factor(this: This, factor: *const [f32; 4]) {
    let given = !factor.is_null();
    front(this).put_with(BLEND_FACTOR, &[u64::from(given)], &[bytes_of(factor, usize::from(given))]);
}
extern "system" fn heaps(this: This, count: u32, items: *const This) {
    let count = if items.is_null() { 0 } else { count };
    front(this).put_with(HEAPS, &[u64::from(count)], &[bytes_of(items, count as usize)]);
}
fn constants(this: This, op: u64, slot: u32, count: u32, values: *const u32, offset: u32) {
    let count = if values.is_null() { 0 } else { count };
    front(this).put_with(
        op,
        &[u64::from(slot) | u64::from(offset) << 32, u64::from(count)],
        &[bytes_of(values, count as usize)],
    );
}
extern "system" fn compute_constants(this: This, slot: u32, count: u32, values: *const u32, offset: u32) {
    constants(this, COMPUTE_CONSTANTS, slot, count, values, offset);
}
extern "system" fn graphics_constants(this: This, slot: u32, count: u32, values: *const u32, offset: u32) {
    constants(this, GRAPHICS_CONSTANTS, slot, count, values, offset);
}
extern "system" fn index_buffer(this: This, view: *const IndexView) {
    let given = !view.is_null();
    front(this).put_with(INDEX_BUFFER, &[u64::from(given)], &[bytes_of(view, usize::from(given))]);
}
extern "system" fn vertex_buffers(this: This, start: u32, count: u32, views: *const VertexView) {
    let given = !views.is_null();
    front(this).put_with(
        VERTEX_BUFFERS,
        &[u64::from(start) | u64::from(count) << 32, u64::from(given)],
        &[bytes_of(views, count as usize)],
    );
}
extern "system" fn targets(this: This, count: u32, handles: *const u64, one_range: i32, depth: *const u64) {
    let count = if handles.is_null() { 0 } else { (count as usize).min(8) };
    // SAFETY: the title passes a handle or null.
    let depth = unsafe { depth.as_ref() }.copied().unwrap_or(0);
    let mut records = [0u64; RECORD * 9];
    let mut present = 0u64;
    for i in 0..count {
        // SAFETY: the title passes `count` handles, or the first of a range
        // of views of 128 bytes.
        let handle = unsafe { if one_range != 0 { *handles + 128 * i as u64 } else { *handles.add(i) } };
        present |= u64::from(handle != 0) << i;
        records[i * RECORD..(i + 1) * RECORD].copy_from_slice(&record(handle));
    }
    records[count * RECORD..(count + 1) * RECORD].copy_from_slice(&record(depth));
    let records = &records[..(count + 1) * RECORD];
    front(this).put_with(
        TARGETS,
        &[count as u64, present, u64::from(depth != 0)],
        &[bytes_of(records.as_ptr(), records.len())],
    );
}
extern "system" fn clear_rtv(this: This, handle: u64, color: *const [f32; 4], count: u32, rects: *const Rect) {
    if handle == 0 || color.is_null() {
        return;
    }
    let count = if rects.is_null() { 0 } else { count };
    let view = record(handle);
    front(this).put_with(
        CLEAR_RTV,
        &[u64::from(count)],
        &[bytes_of(view.as_ptr(), RECORD), bytes_of(color, 1), bytes_of(rects, count as usize)],
    );
}
extern "system" fn clear_dsv(
    this: This,
    handle: u64,
    flags: u32,
    depth: f32,
    stencil: u8,
    count: u32,
    rects: *const Rect,
) {
    if handle == 0 {
        return;
    }
    let count = if rects.is_null() { 0 } else { count };
    let view = record(handle);
    front(this).put_with(
        CLEAR_DSV,
        &[u64::from(count), u64::from(flags) | u64::from(stencil) << 32, Word::word(depth)],
        &[bytes_of(view.as_ptr(), RECORD), bytes_of(rects, count as usize)],
    );
}
extern "system" fn copy_texture_region(
    this: This,
    dst: *const CopyLocation,
    x: u32,
    y: u32,
    z: u32,
    src: *const CopyLocation,
    source_box: *const [u32; 6],
) {
    if dst.is_null() || src.is_null() {
        return;
    }
    let boxed = !source_box.is_null();
    front(this).put_with(
        COPY_TEXTURE,
        &[u64::from(x) | u64::from(y) << 32, u64::from(z) | u64::from(boxed) << 32],
        &[bytes_of(dst, 1), bytes_of(src, 1), bytes_of(source_box, usize::from(boxed))],
    );
}
#[allow(clippy::too_many_arguments)] // the method's own arguments
extern "system" fn copy_texture_region_x(
    this: This,
    dst: *const CopyLocation,
    x: u32,
    y: u32,
    z: u32,
    src: *const CopyLocation,
    source_box: *const [u32; 6],
    _flags: u32,
) {
    copy_texture_region(this, dst, x, y, z, src, source_box);
}

/// Makes the calls of a stream on `list`: how many, and whether the last one
/// closed it. Once `left` says so only what must not be lost is done.
fn translate(
    list: This,
    stream: &[u64],
    left: impl Fn() -> bool,
    mut dxr: impl FnMut() -> Option<crate::dxr::Captured>,
) -> (u64, bool) {
    let (mut i, mut calls, mut closed) = (0, 0, false);
    let mut skip = left();
    while i < stream.len() {
        let (op, len) = (stream[i] & 0xffff, (stream[i] >> 16) as usize);
        let w = &stream[i + 1..i + 1 + len];
        i += 1 + len;
        calls += 1;
        if calls % 64 == 0 {
            skip = left();
        }
        if op == UPLOAD {
            // SAFETY: made by `allocate_upload`, named once.
            let block = *unsafe { Box::from_raw(w[0] as *mut Upload) };
            // SAFETY: a list of ours.
            let uploads = unsafe { &(*list.cast::<List>()).uploads };
            held(uploads).push(block);
            continue;
        }
        if op == DXR {
            let batch = dxr().expect("thin DXR marker without payload");
            if !skip {
                super::dxr_boundary(list, batch);
            }
            continue;
        }
        if skip && op != CLOSE && op != RESET {
            continue;
        }
        if run_plain(op, list, w) {
            continue;
        }
        let n = w.first().map_or(0, |&n| n as usize);
        match op {
            RESET => {
                // (Only whether there is an allocator is looked at.)
                let allocator: This = std::ptr::NonNull::<u64>::dangling().as_ptr().cast_const().cast();
                super::reset(list, allocator, if skip { std::ptr::null() } else { w[0] as This });
            }
            CLOSE => {
                super::close(list);
                closed = true;
            }
            BARRIER => {
                super::barrier(list, n as u32, at_or(w, 1));
            }
            BARRIER_GROUPS => super::barrier_groups(list, n as u32, std::ptr::null()),
            VIEWPORTS => state::viewports(list, n as u32, at(w, 1, n)),
            SCISSORS => state::scissors(list, n as u32, at(w, 1, n)),
            BLEND_FACTOR => state::blend_factor(list, at(w, 1, n)),
            HEAPS => state::heaps(list, n as u32, at(w, 1, n)),
            COMPUTE_CONSTANTS => {
                state::compute_constants(list, w[0] as u32, w[1] as u32, at(w, 2, w[1] as usize), (w[0] >> 32) as u32)
            }
            GRAPHICS_CONSTANTS => {
                state::graphics_constants(list, w[0] as u32, w[1] as u32, at(w, 2, w[1] as usize), (w[0] >> 32) as u32)
            }
            INDEX_BUFFER => state::index_buffer(list, at(w, 1, n)),
            VERTEX_BUFFERS => {
                let views = if w[1] != 0 { at_or(w, 2) } else { std::ptr::null() };
                state::vertex_buffers(list, w[0] as u32, (w[0] >> 32) as u32, views);
            }
            TARGETS => {
                let mut handles = [0u64; 8];
                let place = |k: usize| w[3 + k * RECORD..].as_ptr() as u64;
                for (k, h) in handles.iter_mut().enumerate().take(n) {
                    if w[1] >> k & 1 != 0 {
                        *h = place(k);
                    }
                }
                let depth = if w[2] != 0 { place(n) } else { 0 };
                state::targets(list, n as u32, handles.as_ptr(), 0, &depth);
            }
            CLEAR_RTV => {
                super::clear_rtv(
                    list,
                    w[1..].as_ptr() as u64,
                    at_or(w, 1 + RECORD),
                    n as u32,
                    at(w, 1 + RECORD + 2, n),
                );
            }
            CLEAR_DSV => {
                super::clear_dsv(
                    list,
                    w[3..].as_ptr() as u64,
                    w[1] as u32,
                    Word::unword(w[2]),
                    (w[1] >> 32) as u8,
                    n as u32,
                    at(w, 3 + RECORD, n),
                );
            }
            COPY_TEXTURE => {
                let step = size_of::<CopyLocation>().div_ceil(8);
                let boxed = (w[1] >> 32) as usize;
                super::copy_texture_region(
                    list,
                    at_or(w, 2),
                    w[0] as u32,
                    (w[0] >> 32) as u32,
                    w[1] as u32,
                    at_or(w, 2 + step),
                    at(w, 2 + 2 * step, boxed),
                );
            }
            UPSCALE => {
                if let Some(up) = super::Upscale::from_words(w)
                    && !super::upscale_now(list, &up)
                {
                    static TOLD: std::sync::Once = std::sync::Once::new();
                    TOLD.call_once(|| tracing::warn!("a written-down upscale could not be recorded: its output stays as it is"));
                }
            }
            _ => {}
        }
    }
    (calls, closed)
}

/// The table of a list whose calls are written down: that of the others with
/// the methods that record replaced (slots as in [`super::METHODS`]).
pub(super) const fn table(newer: bool) -> [*const (); 160] {
    let mut f = super::list_table(newer);
    let mut i = 0;
    while i < MINE.len() {
        let (slot, method) = MINE[i];
        f[if newer { newer_slot(slot) } else { slot }] = method;
        i += 1;
    }
    if newer {
        f[146] = barrier_groups as *const ();
        f[148] = allocate_upload as *const ();
    }
    f
}

const MINE: [(usize, *const ()); 50] = [
    (9, list_close as *const ()),
    (10, list_reset as *const ()),
    (12, draw as *const ()),
    (13, draw_indexed as *const ()),
    (14, dispatch as *const ()),
    (15, copy_buffer_region as *const ()),
    (16, copy_texture_region as *const ()),
    (17, copy_resource as *const ()),
    (19, resolve as *const ()),
    (20, topology as *const ()),
    (21, viewports as *const ()),
    (22, scissors as *const ()),
    (23, blend_factor as *const ()),
    (24, stencil_ref as *const ()),
    (25, set_pipeline as *const ()),
    (26, barrier as *const ()),
    (30, resolve_queries as *const ()),
    (32, heaps as *const ()),
    (35, compute_table as *const ()),
    (36, graphics_table as *const ()),
    (37, compute_constant as *const ()),
    (38, graphics_constant as *const ()),
    (39, compute_constants as *const ()),
    (40, graphics_constants as *const ()),
    (41, compute_view as *const ()),
    (42, graphics_view as *const ()),
    (43, compute_view as *const ()),
    (44, graphics_view as *const ()),
    (45, compute_view as *const ()),
    (46, graphics_view as *const ()),
    (47, index_buffer as *const ()),
    (48, vertex_buffers as *const ()),
    (50, targets as *const ()),
    (51, clear_dsv as *const ()),
    (52, clear_rtv as *const ()),
    (59, execute_indirect as *const ()),
    (60, copy_memory as *const ()),
    (68, depth_bounds as *const ()),
    (69, write32 as *const ()),
    (70, write64 as *const ()),
    (71, wait_value as *const ()),
    (72, wait_value64 as *const ()),
    (78, dispatch_x as *const ()),
    (83, copy_texture_region_x as *const ()),
    (84, copy_resource_x as *const ()),
    (98, dispatch_mesh_x as *const ()),
    (127, dispatch_mesh as *const ()),
    (129, sync as *const ()),
    (130, sync as *const ()),
    (137, list_reset_x as *const ()),
];

#[cfg(test)]
mod tests {
    use super::*;

    /// Every method of the tables that records is written down too: what is
    /// left in both tables alike records nothing.
    #[test]
    fn every_recording_method_is_written_down() {
        const NOTHING: [usize; 16] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 28, 29, 33, 34, 73, 99, 124];
        let mine: Vec<usize> = MINE.iter().map(|m| m.0).collect();
        for (slot, _) in METHODS.iter().chain(MORE_METHODS.iter()) {
            assert!(mine.contains(slot) != NOTHING.contains(slot), "slot {slot}");
        }
        for &slot in &mine {
            assert!(
                METHODS.iter().chain(MORE_METHODS.iter()).any(|m| m.0 == slot),
                "slot {slot} is written down and not there"
            );
        }
    }

    /// Arguments come out of a stream as they went in.
    #[test]
    fn a_stream_gives_back_what_was_written() {
        let mut f = Front::default();
        f.recording = Some(Arc::new(Recording::new(0)));
        f.put(Plain::DRAW_INDEXED as u64, &[3, 1, 0, Word::word(-7i32), 5]);
        let viewport = [1.0f32, 2.0, 3.0, 4.0, 0.0, 1.0];
        f.put_with(VIEWPORTS, &[1], &[bytes_of(viewport.as_ptr(), 6)]);
        f.put_with(COMPUTE_CONSTANTS, &[4 | 2 << 32, 3], &[bytes_of([7u32, 8, 9].as_ptr(), 3)]);
        let s = std::mem::take(&mut f.words);
        assert_eq!((s[0] & 0xffff, s[0] >> 16), (Plain::DRAW_INDEXED as u64, 5));
        assert_eq!(<i32 as Word>::unword(s[4]), -7);
        let w = &s[6..];
        assert_eq!((w[0] & 0xffff, w[0] >> 16, w[1]), (VIEWPORTS, 4, 1));
        // SAFETY: what was written just now.
        assert_eq!(unsafe { *at::<[f32; 6]>(&w[1..], 1, 1) }, viewport);
        let w = &w[5..];
        assert_eq!((w[0] & 0xffff, w[0] >> 16), (COMPUTE_CONSTANTS, 4));
        // SAFETY: as above.
        assert_eq!(unsafe { *at::<[u32; 3]>(&w[1..], 2, 3) }, [7, 8, 9]);
        assert!(at::<u32>(&w[1..], 2, 0).is_null());
    }

    /// A view's record is the one at the call, also if its slot is written
    /// again right after.
    #[test]
    fn a_target_is_the_one_at_the_call() {
        let mut slot = [0x5256_5476u64, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
        let kept = record(slot.as_ptr() as u64);
        slot[1] = 99;
        assert_eq!(slot[1], 99);
        assert_eq!(kept[1], 2);
        assert_eq!(record(0), [0; RECORD]);
    }

    /// An empty RESET+CLOSE recording completes via the fast-path without
    /// allocating a backing host command list.
    #[test]
    fn empty_reset_close_fast_path_skips_back_list() {
        let rec = Arc::new(Recording::new(0));
        {
            let mut inner = held(&rec.inner);
            inner.chunks.push_back(vec![RESET | (1 << 16), 0, CLOSE]);
            inner.scheduled = true;
        }
        rec.run();
        assert!(rec.done(), "empty recording finishes immediately");
        assert!(rec.back().is_null(), "no backing list allocated");
        let inner = held(&rec.inner);
        assert!(inner.closed && !inner.scheduled && inner.chunks.is_empty());
    }

    /// Empty RESET+CLOSE fast-path calls `pipeline::bound` strictly before
    /// publishing `done`, and skips binding if the recording was abandoned.
    #[test]
    fn empty_reset_close_binds_pipeline_before_publishing_done() {
        let sentinel = 0x1234_5678usize as This;
        let rec = Arc::new(Recording::new(0));
        {
            let mut inner = held(&rec.inner);
            inner.chunks.push_back(vec![RESET | (1 << 16), sentinel as u64, CLOSE]);
            inner.scheduled = true;
        }
        let mut saw_pipeline = None;
        rec.run_with(
            |p| {
                assert!(!rec.done(), "pipeline::bound must run before done is published");
                saw_pipeline = Some(p);
            },
            |_, _, _, _| unreachable!("empty recording must take fast-path"),
        );
        assert_eq!(saw_pipeline, Some(sentinel));
        assert!(rec.done());
        assert!(rec.back().is_null());

        let abandoned = Arc::new(Recording::new(0));
        abandoned.abandoned.store(true, Release);
        {
            let mut inner = held(&abandoned.inner);
            inner.chunks.push_back(vec![RESET | (1 << 16), sentinel as u64, CLOSE]);
            inner.scheduled = true;
        }
        let mut called = false;
        abandoned.run_with(
            |_| called = true,
            |_, _, _, _| unreachable!("abandoned empty recording must take fast-path"),
        );
        assert!(!called, "abandoned recording must not dereference pipeline");
        assert!(abandoned.done());
        assert!(abandoned.back().is_null());
    }

    /// A concurrent `abandon()` waits until an in-flight empty RESET+CLOSE
    /// fast-path `on_pipeline` call finishes, preventing use-after-abandon on
    /// `pipeline`.
    #[test]
    fn abandon_waits_for_in_flight_empty_fast_path_pipeline_bind() {
        let sentinel_addr = 0x5678_9abcusize;
        let rec = Arc::new(Recording::new(0));
        {
            let mut inner = held(&rec.inner);
            inner.chunks.push_back(vec![RESET | (1 << 16), sentinel_addr as u64, CLOSE]);
            inner.scheduled = true;
        }
        let entered = Arc::new(std::sync::Barrier::new(2));
        let (release_tx, release_rx) = std::sync::mpsc::channel::<()>();
        let pipeline_done = Arc::new(AtomicBool::new(false));
        let abandon_returned = Arc::new(AtomicBool::new(false));
        let saw_pipeline_done_after_abandon = Arc::new(AtomicBool::new(false));

        let worker = {
            let rec = rec.clone();
            let entered = entered.clone();
            let pipeline_done = pipeline_done.clone();
            std::thread::spawn(move || {
                rec.run_with(
                    |p| {
                        assert_eq!(p as usize, sentinel_addr);
                        assert!(!rec.done());
                        entered.wait();
                        release_rx.recv().unwrap();
                        pipeline_done.store(true, Release);
                    },
                    |_, _, _, _| unreachable!("empty recording must take fast-path"),
                );
            })
        };

        // Worker is now past the `abandoned == false` check and inside `on_pipeline`.
        entered.wait();

        let abandon_entered = Arc::new(std::sync::Barrier::new(2));
        let abandoner = {
            let rec = rec.clone();
            let abandon_entered = abandon_entered.clone();
            let pipeline_done = pipeline_done.clone();
            let abandon_returned = abandon_returned.clone();
            let saw_pipeline_done_after_abandon = saw_pipeline_done_after_abandon.clone();
            std::thread::spawn(move || {
                abandon_entered.wait();
                rec.abandon();
                saw_pipeline_done_after_abandon.store(pipeline_done.load(Acquire), Release);
                abandon_returned.store(true, Release);
            })
        };

        abandon_entered.wait();
        // Wait until `abandon()` has set `abandoned = true` and parked on `moved`.
        while !rec.abandoned.load(Acquire) {
            std::thread::yield_now();
        }
        // Acquire `inner` once to confirm `abandon()` has entered `moved.wait`
        // (or is blocked) while `running` is still true and `abandon()` has not returned.
        {
            let inner = held(&rec.inner);
            assert!(inner.running);
            assert!(!abandon_returned.load(Acquire), "abandon() must not return while on_pipeline runs");
        }

        // Release `on_pipeline` and confirm `abandon()` unblocks strictly afterwards.
        release_tx.send(()).unwrap();
        worker.join().unwrap();
        abandoner.join().unwrap();
        assert!(abandon_returned.load(Acquire));
        assert!(pipeline_done.load(Acquire));
        assert!(
            saw_pipeline_done_after_abandon.load(Acquire),
            "abandon() returned before on_pipeline finished"
        );
        assert!(rec.done());
        assert!(rec.back().is_null());
    }

    /// A concurrent `abandon()` also waits until an in-flight non-empty
    /// `translate` step finishes before returning.
    #[test]
    fn abandon_waits_for_in_flight_non_empty_translate() {
        let sentinel_back = 0xdead_beefusize;
        let rec = Arc::new(Recording::new(0));
        {
            let mut inner = held(&rec.inner);
            inner.back = sentinel_back;
            inner.chunks.push_back(vec![
                RESET | (1 << 16),
                0,
                (Plain::TOPOLOGY as u64) | (1 << 16),
                4,
                CLOSE,
            ]);
            inner.scheduled = true;
        }
        let entered = Arc::new(std::sync::Barrier::new(2));
        let (release_tx, release_rx) = std::sync::mpsc::channel::<()>();
        let translate_done = Arc::new(AtomicBool::new(false));
        let abandon_returned = Arc::new(AtomicBool::new(false));
        let saw_translate_done_after_abandon = Arc::new(AtomicBool::new(false));

        let worker = {
            let rec = rec.clone();
            let entered = entered.clone();
            let translate_done = translate_done.clone();
            std::thread::spawn(move || {
                rec.run_with(
                    |_| unreachable!("non-empty chunk must not hit empty fast-path"),
                    |back, chunk, left, _dxr| {
                        assert_eq!(back as usize, sentinel_back);
                        assert_eq!(chunk.len(), 5);
                        assert!(!left());
                        entered.wait();
                        release_rx.recv().unwrap();
                        assert!(left(), "abandon() flag must be visible inside translate");
                        translate_done.store(true, Release);
                        (3, true)
                    },
                );
            })
        };

        entered.wait();

        let abandon_entered = Arc::new(std::sync::Barrier::new(2));
        let abandoner = {
            let rec = rec.clone();
            let abandon_entered = abandon_entered.clone();
            let translate_done = translate_done.clone();
            let abandon_returned = abandon_returned.clone();
            let saw_translate_done_after_abandon = saw_translate_done_after_abandon.clone();
            std::thread::spawn(move || {
                abandon_entered.wait();
                rec.abandon();
                saw_translate_done_after_abandon.store(translate_done.load(Acquire), Release);
                abandon_returned.store(true, Release);
            })
        };

        abandon_entered.wait();
        while !rec.abandoned.load(Acquire) {
            std::thread::yield_now();
        }
        {
            let inner = held(&rec.inner);
            assert!(inner.running);
            assert!(
                !abandon_returned.load(Acquire),
                "abandon() must not return while translate runs"
            );
        }

        release_tx.send(()).unwrap();
        worker.join().unwrap();
        abandoner.join().unwrap();
        assert!(abandon_returned.load(Acquire));
        assert!(translate_done.load(Acquire));
        assert!(
            saw_translate_done_after_abandon.load(Acquire),
            "abandon() returned before non-empty translate finished"
        );
        assert!(rec.done());
        // Clear dummy sentinel before `Drop for Recording` runs.
        held(&rec.inner).back = 0;
    }

    fn dummy_wait() -> ValueWait {
        ValueWait {
            address: 0x2000,
            comparison: 3,
            reference: 1,
            wide: false,
        }
    }

    fn has_value_wait(addr: usize) -> bool {
        super::VALUE_WAITS
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .iter()
            .any(|&(ptr, _)| ptr == addr)
    }

    fn track_drop_via_value_wait(list: This) -> usize {
        let addr = list as usize;
        super::VALUE_WAITS
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .push((addr, dummy_wait()));
        addr
    }

    /// `back_list()` must initialize `refs == 1` so `super::list_release`
    /// actually drops `Box<List>` instead of leaking with `refs == 0`, and
    /// reusing a list through the pool must keep `refs == 1` without bumping it.
    #[test]
    fn back_list_starts_with_one_ref_and_drops_on_release() {
        let pool: Mutex<Vec<usize>> = Mutex::new(Vec::new());

        let list = back_list_in(&pool, 0, super::boxed_without_host);
        let addr = track_drop_via_value_wait(list);
        assert!(has_value_wait(addr));
        assert_eq!(super::list_release(list), 0);
        assert!(
            !has_value_wait(addr),
            "back_list must have refs == 1 so list_release drops Box<List> and runs Drop"
        );

        // Pool round-trip must preserve refs == 1 (no double add_ref).
        let pooled = back_list_in(&pool, 0, super::boxed_without_host);
        let pooled_addr = track_drop_via_value_wait(pooled);
        release_or_park_back(&pool, pooled_addr, false);
        assert!(
            has_value_wait(pooled_addr),
            "parked backing list must stay alive in pool"
        );
        let reused = back_list_in(&pool, 2, super::boxed_without_host);
        assert_eq!(reused as usize, pooled_addr);
        assert_eq!(super::list_type(reused), 2);
        assert_eq!(
            super::list_release(reused),
            0,
            "reused backing list must still have refs == 1"
        );
        assert!(
            !has_value_wait(pooled_addr),
            "releasing a reused backing list must drop Box<List>"
        );
    }

    /// Pooling up to 64 backing lists keeps them alive; evicting beyond 64,
    /// immediate DXR/abandoned releases, and draining the pool all drop cleanly
    /// with zero leaks and zero double-frees.
    #[test]
    fn backs_pool_reuse_and_eviction_beyond_64_drops_cleanly() {
        let pool: Mutex<Vec<usize>> = Mutex::new(Vec::new());
        let mut lists = Vec::with_capacity(70);
        let mut addrs = Vec::with_capacity(70);
        // Allocate all 70 up front so every pointer address is strictly distinct.
        for _ in 0..70 {
            let list = back_list_in(&pool, 0, super::boxed_without_host);
            addrs.push(track_drop_via_value_wait(list));
            lists.push(list);
        }

        // Park all 70: first 64 stay pooled, last 6 are evicted and dropped.
        for &list in &lists {
            release_or_park_back(&pool, list as usize, false);
        }
        assert_eq!(held(&pool).len(), 64);
        for &addr in &addrs[..64] {
            assert!(has_value_wait(addr), "pooled backing list {addr:#x} must remain alive");
        }
        for &addr in &addrs[64..] {
            assert!(!has_value_wait(addr), "evicted backing list {addr:#x} must be dropped");
        }

        // Pop 10, verify kind update, and release immediately (DXR/abandoned path).
        let mut popped = Vec::with_capacity(10);
        for k in 0..10u32 {
            let reused = back_list_in(&pool, k, super::boxed_without_host);
            assert_eq!(super::list_type(reused), k);
            assert!(has_value_wait(reused as usize));
            popped.push(reused as usize);
        }
        assert_eq!(held(&pool).len(), 54);
        for addr in popped {
            release_or_park_back(&pool, addr, true);
            assert!(
                !has_value_wait(addr),
                "immediate release of reused backing list {addr:#x} must drop Box<List>"
            );
        }

        // Drain the remaining 54 pooled lists cleanly.
        while let Some(addr) = held(&pool).pop() {
            assert_eq!(super::list_release(addr as This), 0);
            assert!(!has_value_wait(addr));
        }
        for &addr in &addrs {
            assert!(!has_value_wait(addr), "leaked backing list {addr:#x}");
        }
    }
}
