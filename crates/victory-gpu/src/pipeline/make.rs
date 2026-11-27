//! Host pipelines are made by workers, not by the title thread that first
//! records a draw with them. Translating a shader and the host driver's
//! compile take from milliseconds to seconds each (structurisation of the
//! biggest compute programs ~0.4 s, some driver compiles 3.4 s), and
//! the title's first world frame wants over 1,200 new pipelines: made
//! on its recording threads, that frame took more than 30 s and the
//! title's GPU watchdog ended the run (decision 0007, "Pipelines made off
//! the title's threads").
//!
//! A first use queues the pipeline and waits for it within a budget per
//! frame, so the cheap ones (menus, one-off passes such as a colour-grading
//! bake) are there at once as before; past the budget a draw or dispatch
//! whose pipeline is not ready is dropped, as an untranslated one is.
//!
//! The default budget is half a second per title frame. A pass the title
//! runs only once and whose pipeline takes longer is lost. Passes over the
//! screen (no vertex input, no depth) have workers and a longer budget of
//! their own ([`screen_budget`]): the title fills its lighting tables with
//! such draws once, at the world's first frame, and behind a world's worth of
//! pipelines they were dropped; that run had no lamps and no diffuse light to
//! its end (run textures-d2, the first after a new translation; Belobug
//! 18e57db8 found the same).

use std::collections::VecDeque;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::atomic::{AtomicBool, AtomicU8, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, OnceLock};
use std::time::{Duration, Instant};

use super::Shader;
use super::host::{GraphicsState, HostPipeline};

/// Per-pipeline frame wait state for `VICTORY_PIPELINE_WAIT_AGAIN_MS`
/// (GPL-3.0 provenance: Belobug/VIctory-Belobug-AMD#410).
#[derive(Default)]
struct WaitFrameState {
    /// First frame in which a draw waited for this pipeline (`None`: never).
    first_frame: Option<u64>,
    /// Frame `used_us` belongs to.
    active_frame: u64,
    /// Microseconds reserved/spent by draws on this pipeline in `active_frame`
    /// after `first_frame`.
    used_us: u64,
}

/// What a host pipeline is made from, and the host pipeline once made.
pub(crate) struct Pending {
    shaders: Vec<Arc<Shader>>,
    pub(crate) graphics: Option<Box<GraphicsState>>,
    queue_state: AtomicU8,
    /// Taken by a worker from [`Queue::slow`].
    was_slow: AtomicBool,
    made: OnceLock<Option<HostPipeline>>,
    /// What identifies it across runs (shader containers and state), and
    /// whether a draw or dispatch has requested it yet.
    key: String,
    used: AtomicBool,
    /// A pass over the screen ([`GraphicsState::screen_pass`]) or a confirmed
    /// opening-scene / bake producer that must never sit behind cold RT shaders.
    screen: bool,
    /// Confirmed first-scene / bake producer prepared early even on a cold cache.
    scene_input: bool,
    /// Why this pipeline is intentionally skipped (e.g. RT resolve without BVH).
    left_out: Option<&'static str>,
    /// How long draws have waited for this pipeline so far, in microseconds
    /// ([`wait_once`]).
    waited_us: AtomicU64,
    /// Per-`(pipeline, frame)` state for [`wait_again_budget`], stored inline
    /// on `Pending` (no external map or sentinel collision).
    wait_state: Mutex<WaitFrameState>,
}

fn named(shaders: &[Arc<Shader>], name: &[u8]) -> bool {
    shaders
        .iter()
        .any(|s| s.stage == "compute" && s.container.windows(name.len()).any(|w| w == name))
}

fn left_out(shaders: &[Arc<Shader>]) -> Option<&'static str> {
    static SKIP_RT_RESOLVE: OnceLock<bool> = OnceLock::new();
    let skip_rt = *SKIP_RT_RESOLVE.get_or_init(|| {
        let builder = std::env::var_os("VICTORY_BVH_BUILDER").is_some_and(|v| v == "1")
            || crate::dxr::requested();
        if builder {
            return false;
        }
        // Without a live BVH builder or native DXR sidecar, ~1,000 CS_RaytracingResolve
        // shaders shade only misses while clogging the cold compiler queue; skip unless
        // VICTORY_RT_RESOLVE=1 (codec-d/cool_fox 8cc31ab).
        std::env::var_os("VICTORY_RT_RESOLVE").is_none_or(|v| v == "0")
    });
    if skip_rt && named(shaders, b"CS_RaytracingResolve") {
        return Some("shades ray-tracing hits without BVH, left out");
    }
    static SKIP_ACTIVATE_RAYS: OnceLock<bool> = OnceLock::new();
    let skip_activate = *SKIP_ACTIVATE_RAYS.get_or_init(|| {
        let builder = std::env::var_os("VICTORY_BVH_BUILDER").is_some_and(|v| v == "1")
            || crate::dxr::requested();
        if builder {
            return std::env::var_os("VICTORY_ACTIVATE_RAYS").is_some_and(|v| v == "0");
        }
        std::env::var_os("VICTORY_ACTIVATE_RAYS").is_none_or(|v| v == "0")
    });
    if skip_activate && named(shaders, b"CS_ScreenProbes_ActivateRays") {
        return Some("ray activation shader without BVH, left out");
    }
    static SKIP_RAYMARCHED_LOCAL_LIGHTS: OnceLock<bool> = OnceLock::new();
    let skip_raymarched = *SKIP_RAYMARCHED_LOCAL_LIGHTS.get_or_init(|| {
        let builder = std::env::var_os("VICTORY_BVH_BUILDER").is_some_and(|v| v == "1")
            || crate::dxr::requested();
        if builder {
            return std::env::var_os("VICTORY_RAYMARCHED_LOCAL_LIGHTS").is_some_and(|v| v == "0");
        }
        // Without a live BVH builder or native DXR sidecar, skip raymarched
        // local-light compute pass unless VICTORY_RAYMARCHED_LOCAL_LIGHTS=1
        // (GPL-3.0 provenance: maren-halv/VIctory-DXR d5efb36).
        std::env::var_os("VICTORY_RAYMARCHED_LOCAL_LIGHTS").is_none_or(|v| v == "0")
    });
    if skip_raymarched && named(shaders, b"CS_GatherRaymarchedLocalLightsClassified") {
        return Some("raymarched local lights compute pass without BVH, left out");
    }
    None
}

/// Confirmed first-scene producers. Prepare these at creation even on a cold
/// launch so cutscene lighting, tiling, and bakes never miss their first frame.
fn scene_input(shaders: &[Arc<Shader>]) -> bool {
    shaders.iter().any(|s| {
        matches!(
            s.hash.as_str(),
            "970c4fbb3b017c0e"
                | "068c90badc86f1fd"
                | "e4a7e088c7780ce8"
                | "ebe4e53f031f5249"
                | "2cc4c197ee29e291"
                | "369cbe906ea72764"
        ) || crate::draw::BAKE_PIXEL_SHADERS.contains(&s.hash.as_str())
    })
}

// SAFETY: D3D12 objects are free-threaded; the graphics state (whose input
// elements point into its own names) is only read after creation.
unsafe impl Send for Pending {}
// SAFETY: as above.
unsafe impl Sync for Pending {}

impl Pending {
    pub(crate) fn new(shaders: Vec<Arc<Shader>>, graphics: Option<Box<GraphicsState>>) -> Pending {
        let mut key = blake3::Hasher::new();
        for s in &shaders {
            key.update(s.stage.as_bytes());
            key.update(s.hash.as_bytes());
        }
        key.update(graphics.as_ref().map(|g| g.describe()).unwrap_or_default().as_bytes());
        let scene_input = scene_input(&shaders);
        let screen = (graphics.as_ref().is_some_and(|g| g.screen_pass())
            && shaders.iter().all(|s| s.stage != "mesh"))
            || scene_input;
        let left_out = left_out(&shaders);
        Pending {
            screen,
            scene_input,
            left_out,
            waited_us: AtomicU64::new(0),
            wait_state: Mutex::new(WaitFrameState::default()),
            shaders,
            graphics,
            queue_state: AtomicU8::new(NOT_QUEUED),
            was_slow: AtomicBool::new(false),
            made: OnceLock::new(),
            key: key.finalize().to_hex()[..32].to_owned(),
            used: AtomicBool::new(false),
        }
    }

    pub(super) fn key(&self) -> &str {
        &self.key
    }

    fn make(&self) {
        self.make_with(|| {
            let _building = crate::timeline::span("pipeline: build a host pipeline");
            let began = Instant::now();
            let made = HostPipeline::new(&self.shaders, self.graphics.as_deref());
            super::library::note_time(&self.key, began.elapsed().as_millis() as u64);
            made
        });
    }

    // Provenance (GPL-3.0): fxceless #1353 — catch panics so the pipeline worker
    // never dies and waiters always wake.
    fn make_with(&self, build: impl FnOnce() -> Option<HostPipeline>) {
        self.made.get_or_init(|| {
            match catch_unwind(AssertUnwindSafe(build)) {
                Ok(pipeline) => pipeline,
                Err(_) => {
                    tracing::warn!(shaders = ?self.shaders, "host pipeline build panicked");
                    None
                }
            }
        });
        let (lock, done) = &DONE;
        let _held = lock.lock().unwrap_or_else(|e| e.into_inner());
        done.notify_all();
    }
}

/// Why a pipeline has no host pipeline (diagnostics): its shaders that do
/// not translate, else "refused" or "being made"; `None` if it has one.
#[cfg(feature = "diagnostics")]
pub(crate) fn missing(p: &Pending) -> Option<String> {
    if let Some(why) = p.left_out {
        return Some(why.into());
    }
    match p.made.get() {
        Some(Some(_)) => None,
        None => Some("being made".into()),
        Some(None) => {
            let failed: Vec<String> = p
                .shaders
                .iter()
                .filter(|s| matches!(s.translated.get(), Some(None)))
                .map(|s| format!("{} {}", s.stage, s.hash))
                .collect();
            Some(if failed.is_empty() {
                format!("refused {:?}", p.shaders)
            } else {
                failed.join(" + ")
            })
        }
    }
}

/// The hashes of the shaders `p` is made of.
pub(crate) fn hashes(p: &Pending) -> String {
    let hashes: Vec<&str> = p.shaders.iter().map(|s| s.hash.as_str()).collect();
    hashes.join("+")
}

/// Signalled whenever a worker has made a pipeline.
static DONE: (Mutex<()>, Condvar) = (Mutex::new(()), Condvar::new());

/// Waiting allowed per title frame, over all threads.
fn wait_budget() -> Duration {
    static BUDGET: OnceLock<Duration> = OnceLock::new();
    *BUDGET.get_or_init(|| {
        let millis = std::env::var("VICTORY_PIPELINE_WAIT_MS")
            .ok()
            .and_then(|value| value.trim().parse::<u64>().ok())
            .unwrap_or(500);
        tracing::info!(millis, "host pipeline wait budget per frame");
        Duration::from_millis(millis)
    })
}

/// As [`wait_budget`], for passes over the screen (`VICTORY_SCREEN_WAIT_MS`,
/// 10 s; 0: like the others). Below the title's 30 s watchdog.
fn screen_budget() -> Duration {
    static BUDGET: OnceLock<Duration> = OnceLock::new();
    *BUDGET.get_or_init(|| {
        let millis = std::env::var("VICTORY_SCREEN_WAIT_MS").ok().and_then(|v| v.trim().parse().ok()).unwrap_or(10_000);
        Duration::from_millis(millis)
    })
}

fn parse_wait_again_ms(raw: Option<&str>) -> Option<Duration> {
    let millis = raw?.trim().parse::<u64>().ok()?;
    Some(Duration::from_millis(millis))
}

/// What a pipeline still being made may take of a frame's budget after the
/// first frame that waited for it (`VICTORY_PIPELINE_WAIT_AGAIN_MS`; unset:
/// [`wait_budget`] every frame, as before). The budget is a frame's, so one
/// 3.4 s compile cost about seven frames of half a second each; with this a
/// draw whose pipeline is not made by then is dropped sooner (it pops in
/// later) instead of holding up each frame. Not for passes over the screen.
/// (GPL-3.0 provenance: Belobug/VIctory-Belobug-AMD#410.)
fn wait_again_budget() -> Option<Duration> {
    static BUDGET: OnceLock<Option<Duration>> = OnceLock::new();
    *BUDGET.get_or_init(|| {
        let raw = std::env::var("VICTORY_PIPELINE_WAIT_AGAIN_MS").ok();
        let budget = parse_wait_again_ms(raw.as_deref())?;
        tracing::info!(
            millis = budget.as_millis() as u64,
            "host pipeline wait budget after its first frame"
        );
        Some(budget)
    })
}

/// Reserves what a draw may wait for this pipeline in `frame`, returning
/// `(wait_budget, reserved_again_us)`.
///
/// - When `again` is `None` (switch unset or screen pass) or `available` is
///   zero, `wait_state` is untouched so a pipeline dropped on an already-spent
///   frame keeps its full first-frame wait on the next frame.
/// - In the first frame (`first_frame` is `None` or `Some(frame)`), all draws
///   get `available` (bounded by the frame/shared/`wait_once` budgets).
/// - In any subsequent frame (`first_frame != Some(frame)`), all draws and
///   concurrent waiters on this pipeline share one per-`(pipeline, frame)` cap
///   of `again`: `reserved_again_us` is charged up front before waiting and any
///   unused portion is refunded by [`refund_again_budget`].
fn reserve_again_budget(
    wait_state: &Mutex<WaitFrameState>,
    frame: u64,
    available: Duration,
    again: Option<Duration>,
) -> (Duration, u64) {
    let Some(again) = again.filter(|_| !available.is_zero()) else {
        return (available, 0);
    };
    let mut state = wait_state.lock().unwrap_or_else(|e| e.into_inner());
    match state.first_frame {
        None => {
            state.first_frame = Some(frame);
            state.active_frame = frame;
            state.used_us = 0;
            (available, 0)
        }
        Some(first) if first == frame => (available, 0),
        Some(_) => {
            if state.active_frame != frame {
                state.active_frame = frame;
                state.used_us = 0;
            }
            let again_us = again.as_micros().min(u128::from(u64::MAX)) as u64;
            let rem_us = again_us.saturating_sub(state.used_us);
            let avail_us = available.as_micros().min(u128::from(u64::MAX)) as u64;
            let budget_us = avail_us.min(rem_us);
            state.used_us = state.used_us.saturating_add(budget_us);
            (Duration::from_micros(budget_us), budget_us)
        }
    }
}

/// Refunds any unused portion of a subsequent-frame reservation back to
/// `wait_state` if `frame` is still active.
fn refund_again_budget(
    wait_state: &Mutex<WaitFrameState>,
    frame: u64,
    reserved_us: u64,
    waited: Duration,
) {
    if reserved_us == 0 {
        return;
    }
    let elapsed_us = waited.as_micros().min(u128::from(u64::MAX)) as u64;
    let unused_us = reserved_us.saturating_sub(elapsed_us);
    if unused_us == 0 {
        return;
    }
    let mut state = wait_state.lock().unwrap_or_else(|e| e.into_inner());
    if state.active_frame == frame {
        state.used_us = state.used_us.saturating_sub(unused_us);
    }
}

/// Whether passes over the screen go before the others.
fn screens_first() -> bool {
    !screen_budget().is_zero()
}

/// Pipelines queued and not made yet.
static WAITING: AtomicU64 = AtomicU64::new(0);

/// Pipelines queued and not made yet (for the frame line).
pub(crate) fn waiting() -> u64 {
    WAITING.load(Ordering::Relaxed)
}

/// The host pipeline if it is made (or is made within the frame's budget);
/// queues it on first use.
pub(crate) fn ready(p: &Arc<Pending>) -> Option<&HostPipeline> {
    if p.left_out.is_some() {
        return None;
    }
    // A still-pending pipeline was requested by the title too. Prewarm it
    // on the next launch, even if this run never got to use it.
    if !p.used.load(Ordering::Relaxed) && !p.used.swap(true, Ordering::Relaxed) {
        super::library::note_used(&p.key);
    }
    if p.made.get().is_none() {
        start(p, REQUESTED);
        wait(p, crate::frame::current());
    }
    p.made.get()?.as_ref()
}

/// Makes `p` on this thread if it is not made yet (waiting for a worker
/// that is making it); whether it has a host pipeline.
pub(crate) fn now(p: &Arc<Pending>) -> bool {
    if p.left_out.is_some() {
        return false;
    }
    p.make();
    p.made.get().is_some_and(|made| made.is_some())
}

/// At the title's creation of a pipeline: starts it at once if an earlier
/// run drew with it (during loading, so the world's first frames find it),
/// or if it is a confirmed opening-scene / bake producer.
pub(crate) fn early(p: &Arc<Pending>) {
    if p.left_out.is_some() {
        return;
    }
    if !p.shaders.is_empty() && (p.scene_input || super::library::wanted_early(&p.key)) {
        super::library::EARLY.fetch_add(1, Ordering::Relaxed);
        start(p, if p.scene_input { REQUESTED } else { EARLY });
    }
}

/// Whether the frame's budget is one for all threads together: a thread
/// that waits takes what is left of it and gives back what it did not
/// use. Counted only when a wait had ended, the budget was one per thread:
/// each of the title's five recording threads began its first wait of a
/// frame with nothing counted yet and waited the whole half second, 2.5 s
/// of waiting in every frame in which pipelines were being made (runs
/// 34-49: "2504 ms waiting for pipelines" in each such frame).
/// On by default; `VICTORY_PIPELINE_WAIT_SHARED=0` counts as before.
fn shared_budget() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_PIPELINE_WAIT_SHARED").is_none_or(|v| v != "0"))
}

/// Whether one pipeline is waited for at most one budget over the whole run,
/// rather than re-stalling every frame while a slow driver compile finishes
/// (`trustno1-coder/VIctory-NVIDIA` 77e13a9). On by default;
/// `VICTORY_PIPELINE_WAIT_ONCE=0` waits every frame as before.
fn wait_once() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_PIPELINE_WAIT_ONCE").is_none_or(|v| v != "0"))
}

/// Waits for `p` while frame `frame` has budget left.
fn wait(p: &Pending, frame: u64) {
    let screen = p.screen && screens_first();
    let budget = if screen { screen_budget() } else { wait_budget() };
    if budget.is_zero() {
        return;
    }
    let own = if wait_once() {
        let own = budget.saturating_sub(Duration::from_micros(p.waited_us.load(Ordering::Relaxed)));
        if own.is_zero() {
            return;
        }
        own
    } else {
        Duration::MAX
    };
    // The frame the budget belongs to, and what was waited of it (or is
    // taken by a thread waiting now); of the passes over the screen too.
    static SPENT: [Mutex<(u64, Duration)>; 2] = [const { Mutex::new((u64::MAX, Duration::ZERO)) }; 2];
    let spent_of = &SPENT[usize::from(screen)];
    let again = if screen { None } else { wait_again_budget() };
    let shared = shared_budget();
    let (left, reserved_again_us) = {
        let mut spent = spent_of.lock().unwrap_or_else(|e| e.into_inner());
        if spent.0 != frame {
            *spent = (frame, Duration::ZERO);
        }
        let available = budget.saturating_sub(spent.1).min(own);
        // Marked only when it may wait, so one dropped without waiting has
        // its full budget the next frame. Up to five recording threads may
        // wait for it at once. Only what is taken here is given back below.
        let (left, reserved_again_us) = reserve_again_budget(&p.wait_state, frame, available, again);
        if shared {
            spent.1 += left;
        }
        (left, reserved_again_us)
    };
    if left.is_zero() {
        return;
    }
    let waiting = crate::timeline::span("pipeline: first use waits for a host pipeline");
    let started = Instant::now();
    let (lock, done) = &DONE;
    let mut held = lock.lock().unwrap_or_else(|e| e.into_inner());
    while p.made.get().is_none() {
        let rest = left.saturating_sub(started.elapsed());
        if rest.is_zero() {
            break;
        }
        held = done.wait_timeout(held, rest).unwrap_or_else(|e| e.into_inner()).0;
    }
    drop(held);
    let waited = started.elapsed().min(left);
    waiting.end();
    if reserved_again_us != 0 {
        refund_again_budget(&p.wait_state, frame, reserved_again_us, waited);
    }
    if wait_once() && p.made.get().is_none() {
        let before = p.waited_us.fetch_add(waited.as_micros() as u64, Ordering::Relaxed);
        if Duration::from_micros(before) + waited >= budget {
            tracing::info!(
                shaders = ?p.shaders,
                ms = budget.as_millis() as u64,
                "a pipeline not made within its wait: its draws are dropped until it is (VICTORY_PIPELINE_WAIT_ONCE=0: waited for every frame)"
            );
        }
    }
    if screen && waited > Duration::from_millis(500) {
        tracing::info!(
            ms = waited.as_millis() as u64,
            made = p.made.get().is_some(),
            shaders = ?p.shaders,
            "a pass over the screen was waited for (VICTORY_SCREEN_WAIT_MS=0: dropped past half a second a frame)"
        );
    }
    {
        let mut spent = spent_of.lock().unwrap_or_else(|e| e.into_inner());
        // A frame that began meanwhile has its own budget.
        if spent.0 == frame {
            if shared {
                spent.1 = spent.1.saturating_sub(left - waited);
            } else {
                spent.1 += waited;
            }
        }
    }
    #[cfg(feature = "diagnostics")]
    crate::copystat::add(
        crate::copystat::C::PipelineWaitUs,
        started.elapsed().as_micros() as u64,
    );
}

const NOT_QUEUED: u8 = 0;
const EARLY: u8 = 1;
const REQUESTED: u8 = 2;
const MAKING: u8 = 3;

/// How many workers may make pipelines at once that are known to take long
/// and that no draw has asked for: a quarter of them. On a fresh cache (a
/// change of the translator or the driver) the early list begins with
/// programs that take tens of seconds each: all 16 workers were in them
/// through the whole of the title's loading, 97 pipelines made in 20 s, and
/// the first minute and a half of the world ran at 0.5 to 1.6 s a frame,
/// dropping up to 2,900 draws a frame (run shaders-1).
/// `VICTORY_PIPELINE_SLOW=0`: in the order the title creates them, as before.
fn slow_limit() -> usize {
    static LIMIT: OnceLock<usize> = OnceLock::new();
    *LIMIT.get_or_init(|| {
        if std::env::var_os("VICTORY_PIPELINE_SLOW").is_some_and(|v| v == "0") {
            usize::MAX
        } else {
            (worker_count(None, std::thread::available_parallelism().map_or(2, |n| n.get())) / 4).max(1)
        }
    })
}

struct Queue {
    /// Requested passes over the screen: before all others, and the only ones
    /// [`SCREEN_WORKERS`] take.
    screen: VecDeque<Arc<Pending>>,
    early: VecDeque<Arc<Pending>>,
    /// Those of the early ones known to take long, and how many of them are
    /// being made ([`slow_limit`]).
    slow: VecDeque<Arc<Pending>>,
    slow_now: usize,
    requested: VecDeque<Arc<Pending>>,
}

impl Queue {
    const fn new() -> Self {
        Self {
            screen: VecDeque::new(),
            early: VecDeque::new(),
            slow: VecDeque::new(),
            slow_now: 0,
            requested: VecDeque::new(),
        }
    }

    /// Queues an unrequested pipeline or promotes one queued at creation.
    /// Returns whether it is new to the queue, or `None` if nothing changed.
    fn push(&mut self, p: &Arc<Pending>, priority: u8) -> Option<bool> {
        self.push_as(p, priority, slow_limit() != usize::MAX && super::library::slow(&p.key))
    }

    fn push_as(&mut self, p: &Arc<Pending>, priority: u8, slow: bool) -> Option<bool> {
        let prev = p.queue_state.load(Ordering::Relaxed);
        if prev >= priority {
            return None;
        }
        p.queue_state.store(priority, Ordering::Relaxed);
        if priority == REQUESTED && p.screen && screens_first() {
            self.screen.push_back(p.clone());
        } else if priority == REQUESTED {
            self.requested.push_back(p.clone());
        } else if slow {
            self.slow.push_back(p.clone());
        } else {
            self.early.push_back(p.clone());
        }
        Some(prev == NOT_QUEUED)
    }

    fn pop(&mut self, screens_only: bool, slow_limit: usize) -> Option<Arc<Pending>> {
        // An early entry can have been promoted and claimed already.
        let unclaimed = |list: &mut VecDeque<Arc<Pending>>| {
            std::iter::from_fn(|| list.pop_front()).find(|p| p.queue_state.load(Ordering::Relaxed) == EARLY)
        };
        let mut p = self.screen.pop_front();
        if p.is_none() && !screens_only {
            p = self.requested.pop_front().or_else(|| unclaimed(&mut self.early));
            if p.is_none() && self.slow_now < slow_limit {
                p = unclaimed(&mut self.slow);
                if let Some(p) = &p {
                    p.was_slow.store(true, Ordering::Relaxed);
                    self.slow_now += 1;
                }
            }
        }
        let p = p?;
        p.queue_state.store(MAKING, Ordering::Relaxed);
        Some(p)
    }
}

static QUEUE: Mutex<Queue> = Mutex::new(Queue::new());
static AVAILABLE: Condvar = Condvar::new();

/// Start or promote work for a draw without waiting behind the entire early
/// list. Queue changes and worker claims happen under the same lock.
fn start(p: &Arc<Pending>, priority: u8) {
    if p.queue_state.load(Ordering::Relaxed) >= priority {
        return;
    }
    if !workers() {
        if p.queue_state.swap(MAKING, Ordering::Relaxed) == NOT_QUEUED {
            p.make();
        }
        return;
    }
    let mut queue = QUEUE.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(new) = queue.push(p, priority) {
        if new {
            PEAK.fetch_max(WAITING.fetch_add(1, Ordering::Relaxed) + 1, Ordering::Relaxed);
        }
        // (One woken could be a worker that takes passes over the screen
        // alone: it would leave the pipeline to nobody.)
        AVAILABLE.notify_all();
    }
}

/// Workers that make passes over the screen alone, besides the others.
const SCREEN_WORKERS: usize = 2;

fn next(screens_only: bool) -> Arc<Pending> {
    let mut queue = QUEUE.lock().unwrap_or_else(|e| e.into_inner());
    loop {
        if let Some(p) = queue.pop(screens_only, slow_limit()) {
            return p;
        }
        queue = AVAILABLE.wait(queue).unwrap_or_else(|e| e.into_inner());
    }
}

/// How many workers make pipelines: `asked` (`VICTORY_PIPELINE_WORKERS`, 1 to
/// 64), else two thirds of the logical processors, 2 to 16. They run below the
/// title's own threads, and a cold cache queues hundreds at once.
fn worker_count(asked: Option<usize>, processors: usize) -> usize {
    worker_count_for(asked, processors, false)
}

fn worker_count_for(asked: Option<usize>, processors: usize, limited_ram: bool) -> usize {
    match asked {
        Some(n) => n.clamp(1, 64),
        None if limited_ram => (processors * 3 / 4).clamp(2, 8),
        None => (processors * 2 / 3).clamp(2, 16),
    }
}

/// Starts the workers ([`worker_count`]) on first use.
fn workers() -> bool {
    static RUNNING: OnceLock<bool> = OnceLock::new();
    *RUNNING.get_or_init(|| {
        let mut running = false;
        let workers = worker_count_for(
            std::env::var("VICTORY_PIPELINE_WORKERS")
                .ok()
                .and_then(|v| v.trim().parse().ok()),
            std::thread::available_parallelism().map_or(2, |n| n.get()),
            crate::residency::limited_host_ram(),
        );
        tracing::info!(workers, "pipeline workers");
        let screens = if screens_first() { SCREEN_WORKERS } else { 0 };
        for i in 0..workers + screens {
            let screens_only = i >= workers;
            let spawned = std::thread::Builder::new()
                .name(format!("victory-pipelines-{i}"))
                .spawn(move || {
                    // Below the title's own threads (loading, recording),
                    // and the streaming one, which is below normal itself
                    // (`VICTORY_PIPELINE_LOWEST=0`: level with it, as before).
                    let lowest = !screens_only && std::env::var_os("VICTORY_PIPELINE_LOWEST").is_none_or(|v| v != "0");
                    // SAFETY: a plain call on our own thread.
                    unsafe {
                        use windows::Win32::System::Threading::*;
                        let priority = if lowest { THREAD_PRIORITY_LOWEST } else { THREAD_PRIORITY_BELOW_NORMAL };
                        let _ = SetThreadPriority(GetCurrentThread(), priority);
                    }
                    loop {
                        let p = next(screens_only);
                        p.make();
                        if p.was_slow.load(Ordering::Relaxed) {
                            let mut queue = QUEUE.lock().unwrap_or_else(|e| e.into_inner());
                            queue.slow_now -= 1;
                            AVAILABLE.notify_all();
                        }
                        if WAITING.fetch_sub(1, Ordering::Relaxed) == 1 {
                            drained();
                        } else {
                            super::library::save(false);
                        }
                    }
                });
            if let Err(e) = spawned {
                tracing::warn!(error = %e, "no pipeline worker");
            } else {
                running = true;
            }
        }
        running
    })
}

/// The most pipelines waiting at once since the workers last ran out.
static PEAK: AtomicU64 = AtomicU64::new(0);

/// The workers ran out of work: logged after a batch of 10 or more, and
/// the cache saved (at most every 20 s).
fn drained() {
    use super::library::{EARLY, LOADED, MADE};
    if PEAK.swap(0, Ordering::Relaxed) >= 10 {
        tracing::info!(
            made = MADE.load(Ordering::Relaxed),
            loaded = LOADED.load(Ordering::Relaxed),
            started_early = EARLY.load(Ordering::Relaxed),
            "host pipelines all made"
        );
    }
    super::library::save(false);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn worker_count_follows_the_processors_and_the_switch() {
        assert_eq!(worker_count(None, 1), 2);
        assert_eq!(worker_count(None, 24), 16);
        assert_eq!(worker_count(None, 64), 16);
        assert_eq!(worker_count(Some(0), 24), 1);
        assert_eq!(worker_count(Some(20), 24), 20);
    }

    /// A first use queues the pipeline and, within the budget, gets the
    /// worker's answer (here: no host pipeline for no shaders).
    #[test]
    fn first_use_is_made_by_a_worker() {
        let p = Arc::new(Pending::new(Vec::new(), None));
        assert!(ready(&p).is_none());
        assert_ne!(p.queue_state.load(Ordering::Relaxed), NOT_QUEUED);
        let started = Instant::now();
        while p.made.get().is_none() && started.elapsed() < Duration::from_secs(10) {
            std::thread::yield_now();
        }
        assert!(matches!(p.made.get(), Some(None)));
    }

    #[test]
    fn pending_pipeline_is_recorded_for_the_next_launch() {
        let mut p = Pending::new(Vec::new(), None);
        p.key = "pending-pipeline-test".to_owned();
        // Simulate a worker that has taken the job but cannot finish it yet.
        p.queue_state.store(MAKING, Ordering::Relaxed);
        let p = Arc::new(p);

        assert!(!super::super::library::wanted_early(&p.key));
        assert!(ready(&p).is_none());
        assert!(p.made.get().is_none());
        assert!(super::super::library::wanted_early(&p.key));
    }

    #[test]
    fn draws_promote_early_pipelines_without_claiming_them_twice() {
        let earlier = Arc::new(Pending::new(Vec::new(), None));
        let promoted = Arc::new(Pending::new(Vec::new(), None));
        let requested = Arc::new(Pending::new(Vec::new(), None));
        let mut queue = Queue::new();

        assert_eq!(queue.push(&earlier, EARLY), Some(true));
        assert_eq!(queue.push(&promoted, EARLY), Some(true));
        assert_eq!(queue.push(&requested, REQUESTED), Some(true));
        assert_eq!(queue.push(&promoted, REQUESTED), Some(false));
        assert_eq!(queue.push(&promoted, REQUESTED), None);

        assert!(Arc::ptr_eq(&queue.pop(false, 1).unwrap(), &requested));
        assert!(Arc::ptr_eq(&queue.pop(false, 1).unwrap(), &promoted));
        assert_eq!(queue.push(&promoted, REQUESTED), None);
        assert!(Arc::ptr_eq(&queue.pop(false, 1).unwrap(), &earlier));
        assert!(queue.pop(false, 1).is_none());
    }

    #[test]
    fn pipelines_known_to_take_long_leave_workers_for_the_others() {
        let new = || Arc::new(Pending::new(Vec::new(), None));
        let (long, longer, asked, quick) = (new(), new(), new(), new());
        let mut queue = Queue::new();
        for p in [&long, &longer, &asked] {
            assert_eq!(queue.push_as(p, EARLY, true), Some(true));
        }
        assert_eq!(queue.push_as(&quick, EARLY, false), Some(true));
        // The quick one first, then one long one at a time; one a draw asks
        // for waits for nobody.
        assert!(Arc::ptr_eq(&queue.pop(false, 1).unwrap(), &quick));
        assert!(Arc::ptr_eq(&queue.pop(false, 1).unwrap(), &long));
        assert!(queue.pop(false, 1).is_none());
        assert_eq!(queue.push_as(&asked, REQUESTED, true), Some(false));
        assert!(Arc::ptr_eq(&queue.pop(false, 1).unwrap(), &asked));
        assert!(!asked.was_slow.load(Ordering::Relaxed));
        assert!(queue.pop(false, 1).is_none());
        queue.slow_now -= 1;
        assert!(Arc::ptr_eq(&queue.pop(false, 1).unwrap(), &longer));
        assert!(queue.pop(false, 1).is_none());
    }

    #[test]
    fn wait_again_switch_parses_optional_milliseconds() {
        assert_eq!(parse_wait_again_ms(None), None);
        assert_eq!(parse_wait_again_ms(Some("")), None);
        assert_eq!(parse_wait_again_ms(Some("invalid")), None);
        assert_eq!(parse_wait_again_ms(Some("-5")), None);
        assert_eq!(parse_wait_again_ms(Some("0")), Some(Duration::ZERO));
        assert_eq!(parse_wait_again_ms(Some(" 8 ")), Some(Duration::from_millis(8)));
    }

    /// A pipeline still being made waits the full budget in the first frame
    /// that waits for it (including `frame == u64::MAX`), keeps that first
    /// wait if dropped on an exhausted frame, and caps all subsequent-frame
    /// draws and concurrent waiters combined to `VICTORY_PIPELINE_WAIT_AGAIN_MS`
    /// per `(pipeline, frame)`, refunding unused time on early wakeup.
    #[test]
    fn wait_again_budget_enforces_per_pipeline_frame_cap_and_u64_max() {
        let full = Duration::from_millis(500);
        let again = Some(Duration::from_millis(8));
        let p = Pending::new(Vec::new(), None);

        // Dropped without waiting on an exhausted frame: not stamped.
        assert_eq!(reserve_again_budget(&p.wait_state, 7, Duration::ZERO, again), (Duration::ZERO, 0));
        assert_eq!(p.wait_state.lock().unwrap().first_frame, None);

        // First frame at `u64::MAX` gets full budget with no sentinel collision.
        assert_eq!(reserve_again_budget(&p.wait_state, u64::MAX, Duration::from_millis(200), again), (Duration::from_millis(200), 0));
        assert_eq!(p.wait_state.lock().unwrap().first_frame, Some(u64::MAX));
        // Another draw in the same first frame (`u64::MAX`) still gets full remaining budget.
        assert_eq!(reserve_again_budget(&p.wait_state, u64::MAX, Duration::from_millis(120), again), (Duration::from_millis(120), 0));

        // Frame 0 after `u64::MAX` is a subsequent frame: 10 consecutive draws
        // on the same pending pipeline spend at most 8 ms total.
        let mut total_ms = 0u128;
        for _ in 0..10 {
            let (left, reserved) = reserve_again_budget(&p.wait_state, 0, full, again);
            total_ms += left.as_millis();
            // Simulate waiting the full granted slice (no refund).
            refund_again_budget(&p.wait_state, 0, reserved, left);
        }
        assert_eq!(total_ms, 8);

        // Next frame (frame 1) resets per-frame usage; an early wakeup refunds
        // unused microseconds so later draws in the same frame may use the rest,
        // and concurrent waiters cannot exceed the cap.
        let (left1, res1) = reserve_again_budget(&p.wait_state, 1, Duration::from_millis(5), again);
        assert_eq!((left1, res1), (Duration::from_millis(5), 5_000));
        // While draw 1 is still waiting, two concurrent waiters contend:
        // waiter 2 gets the remaining 3 ms, waiter 3 gets 0 ms.
        let (left2, res2) = reserve_again_budget(&p.wait_state, 1, full, again);
        assert_eq!((left2, res2), (Duration::from_millis(3), 3_000));
        let (left3, res3) = reserve_again_budget(&p.wait_state, 1, full, again);
        assert_eq!((left3, res3), (Duration::ZERO, 0));
        // Waiter 1 wakes early after 2 ms (refunds 3 ms); waiter 2 spends all 3 ms.
        refund_again_budget(&p.wait_state, 1, res1, Duration::from_millis(2));
        refund_again_budget(&p.wait_state, 1, res2, Duration::from_millis(3));
        // A later draw in frame 1 can claim the 3 ms refunded by waiter 1, then exhausted.
        let (left4, res4) = reserve_again_budget(&p.wait_state, 1, full, again);
        assert_eq!((left4, res4), (Duration::from_millis(3), 3_000));
        assert_eq!(reserve_again_budget(&p.wait_state, 1, full, again), (Duration::ZERO, 0));

        // Advancing to frame 2 resets the budget; a stale refund from frame 1
        // must not affect frame 2's usage.
        let (left_f2, res_f2) = reserve_again_budget(&p.wait_state, 2, full, again);
        assert_eq!((left_f2, res_f2), (Duration::from_millis(8), 8_000));
        refund_again_budget(&p.wait_state, 1, res4, Duration::ZERO);
        assert_eq!(p.wait_state.lock().unwrap().used_us, 8_000);
        assert_eq!(reserve_again_budget(&p.wait_state, 2, full, again), (Duration::ZERO, 0));

        // Zero again budget skips subsequent-frame waiting completely.
        assert_eq!(reserve_again_budget(&p.wait_state, 3, full, Some(Duration::ZERO)), (Duration::ZERO, 0));
        // Unset / screen pass (`again == None`) leaves available budget untouched.
        assert_eq!(reserve_again_budget(&p.wait_state, 4, Duration::from_millis(300), None), (Duration::from_millis(300), 0));

        // Concurrent threads in a subsequent frame collectively reserve at most `again`.
        let p = Arc::new(Pending::new(Vec::new(), None));
        assert_eq!(reserve_again_budget(&p.wait_state, 10, full, again), (full, 0));
        let reserved_total = AtomicU64::new(0);
        std::thread::scope(|s| {
            for _ in 0..8 {
                let p = Arc::clone(&p);
                let reserved_total = &reserved_total;
                s.spawn(move || {
                    for _ in 0..4 {
                        let (left, res) = reserve_again_budget(&p.wait_state, 11, Duration::from_millis(3), again);
                        assert_eq!(left, Duration::from_micros(res));
                        reserved_total.fetch_add(res, Ordering::Relaxed);
                    }
                });
            }
        });
        assert_eq!(reserved_total.load(Ordering::Relaxed), 8_000);
    }

    /// Passes over the screen go before all others, and workers of their own
    /// take nothing else.
    #[test]
    fn passes_over_the_screen_go_first() {
        let of = |screen| {
            let mut p = Pending::new(Vec::new(), None);
            p.screen = screen;
            Arc::new(p)
        };
        let (world, screen) = (of(false), of(true));
        let mut queue = Queue::new();
        assert_eq!(queue.push(&world, REQUESTED), Some(true));
        assert_eq!(queue.push(&screen, REQUESTED), Some(true));
        assert!(Arc::ptr_eq(&queue.pop(true, 1).unwrap(), &screen));
        assert!(queue.pop(true, 1).is_none());
        assert!(Arc::ptr_eq(&queue.pop(false, 1).unwrap(), &world));
    }
}
