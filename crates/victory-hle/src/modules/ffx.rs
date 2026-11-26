//! `amd_fidelityfx_loader_xbox.dll`: the upscale API (`ffxCreateContext`,
//! `ffxDestroyContext`, `ffxQuery`, `ffxDispatch`) when `VICTORY_UPSCALE` is
//! `dlss` or `stretch`.
//!
//! The console's shipping `amd_fidelityfx_upscaler_xbox.dll` hardcodes
//! `maxUpscaleSize == 1920x1080` and `maxRenderSize == 1280x720` (for Series S)
//! and returns a zero jitter phase count for any other size (crashing the title
//! with integer divide-by-zero). Taking `ffxCreateContext`, `ffxDestroyContext`,
//! `ffxQuery`, and `ffxDispatch` for upscale contexts lets DLSS / stretch run
//! cleanly at arbitrary output resolutions and render scales (including
//! `1920x1080 -> 2560x1440` and `1920x1080 -> 3840x2160`) while passing any
//! non-upscale effects through to the title's DLL.
//!
//! The structures are the public FidelityFX API's (`ffx_api.h`,
//! `ffx_upscale.h`): a header of a type and a next pointer, and resources of
//! a pointer, a 32-byte description and a state.

use std::ffi::{c_char, c_void};
use std::sync::Mutex;
use std::sync::atomic::{AtomicU32, AtomicUsize, Ordering};

/// The DLL whose upscale entry points are taken.
pub const LOADER: &str = "amd_fidelityfx_loader_xbox.dll";

static REAL_CREATE: AtomicUsize = AtomicUsize::new(0);
static REAL_DESTROY: AtomicUsize = AtomicUsize::new(0);
static REAL_QUERY: AtomicUsize = AtomicUsize::new(0);
static REAL_DISPATCH: AtomicUsize = AtomicUsize::new(0);

/// Active synthetic upscale contexts and their `CreateUpscale.flags`.
static OUR_CONTEXTS: Mutex<Vec<(usize, u32)>> = Mutex::new(Vec::new());

fn alloc_our_context(flags: u32) -> *mut c_void {
    let ptr = Box::into_raw(Box::new(0x5649_4354_444C_5353u64)).cast::<c_void>();
    OUR_CONTEXTS
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .push((ptr as usize, flags));
    ptr
}

fn free_our_context(ptr: *mut c_void) -> bool {
    let mut list = OUR_CONTEXTS.lock().unwrap_or_else(|e| e.into_inner());
    let Some(pos) = list.iter().position(|&(p, _)| p == ptr as usize) else {
        return false;
    };
    list.swap_remove(pos);
    // SAFETY: allocated by `Box::into_raw` in `alloc_our_context`.
    unsafe { drop(Box::from_raw(ptr.cast::<u64>())) };
    true
}

fn our_context_flags(ptr: *mut c_void) -> Option<u32> {
    OUR_CONTEXTS
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .iter()
        .find(|&&(p, _)| p == ptr as usize)
        .map(|&(_, flags)| flags)
}

fn is_our_context(ptr: *mut c_void) -> bool {
    our_context_flags(ptr).is_some()
}

/// Records one of the title DLL's real entry points for pass-through.
pub fn set_real(name: &str, address: usize) {
    match name {
        "ffxCreateContext" => REAL_CREATE.store(address, Ordering::Relaxed),
        "ffxDestroyContext" => REAL_DESTROY.store(address, Ordering::Relaxed),
        "ffxQuery" => REAL_QUERY.store(address, Ordering::Relaxed),
        "ffxDispatch" => REAL_DISPATCH.store(address, Ordering::Relaxed),
        _ => {}
    }
}

#[repr(C)]
struct Header {
    kind: u64,
    next: *const Header,
}

#[repr(C)]
struct CreateUpscale {
    header: Header,
    flags: u32,
    max_render: [u32; 2],
    max_output: [u32; 2],
    message: usize,
}

#[repr(C)]
struct QueryUpscaleRatio {
    header: Header,
    quality_mode: u32,
    out_ratio: *mut f32,
}

#[repr(C)]
struct QueryRenderResolution {
    header: Header,
    display_width: u32,
    display_height: u32,
    quality_mode: u32,
    out_render_width: *mut u32,
    out_render_height: *mut u32,
}

#[repr(C)]
struct QueryJitterPhaseCount {
    header: Header,
    render_width: u32,
    display_width: u32,
    out_phase_count: *mut i32,
}

#[repr(C)]
struct QueryJitterOffset {
    header: Header,
    index: i32,
    phase_count: i32,
    out_x: *mut f32,
    out_y: *mut f32,
}

#[repr(C)]
struct QueryMemoryUsage {
    header: Header,
    usage: *mut [u64; 2],
}

#[repr(C)]
struct QueryMemoryUsageV2 {
    header: Header,
    device: *mut c_void,
    max_render: [u32; 2],
    max_output: [u32; 2],
    flags: u32,
    usage: *mut [u64; 2],
}

#[repr(C)]
struct QueryProviderVersion {
    header: Header,
    id: u64,
    name: *const c_char,
}

const _: () = assert!(size_of::<CreateUpscale>() == 0x30);
const _: () = assert!(std::mem::offset_of!(QueryJitterPhaseCount, out_phase_count) == 0x18);
const _: () = assert!(std::mem::offset_of!(QueryJitterOffset, out_x) == 0x18);
const _: () = assert!(std::mem::offset_of!(QueryJitterOffset, out_y) == 0x20);

#[repr(C)]
struct Resource {
    resource: *const c_void,
    description: [u32; 8],
    state: u32,
}

#[repr(C)]
struct Upscale {
    header: Header,
    list: *const c_void,
    colour: Resource,
    depth: Resource,
    motion: Resource,
    exposure: Resource,
    reactive: Resource,
    composition: Resource,
    output: Resource,
    jitter: [f32; 2],
    motion_scale: [f32; 2],
    render: [u32; 2],
    upscale: [u32; 2],
    // The rest of `ffxDispatchDescUpscale` (`ffx_upscale.h`), read by the
    // probe only.
    enable_sharpening: u8,
    sharpness: f32,
    frame_time_delta: f32,
    pre_exposure: f32,
    reset: u8,
    camera_near: f32,
    camera_far: f32,
    camera_fov: f32,
    view_to_meters: f32,
    flags: u32,
}

/// `VICTORY_UPSCALE=dlss`: use NVIDIA DLSS when built with the `dlss` feature.
fn want_dlss() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_UPSCALE").is_some_and(|v| v == "dlss"))
}

/// `VICTORY_UPSCALE_PROBE=1`: logs what the title gives each upscale (the
/// inputs a temporal upscaler such as DLSS or FSR 2 needs), now and then.
fn probe() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_UPSCALE_PROBE").is_some_and(|v| v == "1"))
}

const _: () = assert!(size_of::<Resource>() == 48);
const _: () = assert!(std::mem::offset_of!(Upscale, output) == 0x138);

/// `FFX_API_RETURN_OK`.
const OK: u32 = 0;

/// `FFX_API_RETURN_ERROR`: what a call nothing could answer returns.
const ERROR: u32 = 1;

/// `FFX_API_CREATE_CONTEXT_DESC_TYPE_UPSCALE`.
const CREATE_UPSCALE: u64 = 0x0001_0000;
/// `FFX_API_DISPATCH_DESC_TYPE_UPSCALE`.
const UPSCALE: u64 = 0x0001_0001;
/// `FFX_API_QUERY_DESC_TYPE_UPSCALE_GETUPSCALERATIOFROMQUALITYMODE`.
const QUERY_UPSCALE_RATIO: u64 = 0x0001_0002;
/// `FFX_API_QUERY_DESC_TYPE_UPSCALE_GETRENDERRESOLUTIONFROMQUALITYMODE`.
const QUERY_RENDER_RESOLUTION: u64 = 0x0001_0003;
/// `FFX_API_QUERY_DESC_TYPE_UPSCALE_GETJITTERPHASECOUNT`.
const QUERY_JITTER_PHASE_COUNT: u64 = 0x0001_0004;
/// `FFX_API_QUERY_DESC_TYPE_UPSCALE_GETJITTEROFFSET`.
const QUERY_JITTER_OFFSET: u64 = 0x0001_0005;
/// `FFX_API_QUERY_DESC_TYPE_UPSCALE_GPU_MEMORY_USAGE`.
const QUERY_MEMORY_USAGE: u64 = 0x0001_0008;
/// `FFX_API_QUERY_DESC_TYPE_UPSCALE_GPU_MEMORY_USAGE_V2`.
const QUERY_MEMORY_USAGE_V2: u64 = 0x0001_0009;
/// `FFX_API_QUERY_DESC_TYPE_GET_PROVIDER_VERSION`.
const QUERY_PROVIDER_VERSION: u64 = 0x0000_0006;

type CreateFn = extern "system" fn(*mut *mut c_void, *const Header, *const c_void) -> u32;
type DestroyFn = extern "system" fn(*mut *mut c_void, *const c_void) -> u32;
type QueryFn = extern "system" fn(*mut *mut c_void, *mut Header) -> u32;
type Dispatch = extern "system" fn(*mut *mut c_void, *const Header) -> u32;

fn upscale_ratio_for_quality(mode: u32) -> f32 {
    match mode {
        0 => 1.0, // NativeAA
        1 => 1.5, // Quality
        2 => 1.7, // Balanced
        3 => 2.0, // Performance
        4 => 3.0, // UltraPerformance
        _ => 1.5,
    }
}

fn jitter_phase_count(render_width: u32, display_width: u32) -> i32 {
    let ratio = display_width.max(1) as f32 / render_width.max(1) as f32;
    (8.0_f32 * ratio * ratio).ceil().max(1.0) as i32
}

fn halton(mut index: i32, base: i32) -> f32 {
    let mut f = 1.0_f32;
    let mut r = 0.0_f32;
    while index > 0 {
        f /= base as f32;
        r += f * (index % base) as f32;
        index /= base;
    }
    r
}

fn jitter_offset(index: i32, phase_count: i32) -> (f32, f32) {
    let phase = phase_count.max(1);
    let idx = index.rem_euclid(phase) + 1;
    (halton(idx, 2) - 0.5, halton(idx, 3) - 0.5)
}

extern "system" fn create_context(
    context: *mut *mut c_void,
    desc: *const Header,
    mem_cb: *const c_void,
) -> u32 {
    // SAFETY: the title passes a creation descriptor chain.
    if let Some(header) = unsafe { desc.as_ref() }
        && header.kind == CREATE_UPSCALE
        && !context.is_null()
    {
        // SAFETY: an upscale creation descriptor, by its type.
        let c = unsafe { &*desc.cast::<CreateUpscale>() };
        // SAFETY: out pointer from caller.
        unsafe { *context = alloc_our_context(c.flags) };
        tracing::info!(
            max_render = ?c.max_render,
            max_output = ?c.max_output,
            flags = format_args!("{:#x}", c.flags),
            "upscale context created (host DLSS/stretch; bypassing console FSR4 size limits)"
        );
        return OK;
    }
    match REAL_CREATE.load(Ordering::Relaxed) {
        0 => ERROR,
        // SAFETY: title DLL's `ffxCreateContext`.
        real => unsafe { std::mem::transmute::<usize, CreateFn>(real)(context, desc, mem_cb) },
    }
}

extern "system" fn destroy_context(context: *mut *mut c_void, mem_cb: *const c_void) -> u32 {
    // SAFETY: caller's context pointer.
    if !context.is_null() && free_our_context(unsafe { *context }) {
        unsafe { *context = std::ptr::null_mut() };
        return OK;
    }
    match REAL_DESTROY.load(Ordering::Relaxed) {
        0 => ERROR,
        // SAFETY: title DLL's `ffxDestroyContext`.
        real => unsafe { std::mem::transmute::<usize, DestroyFn>(real)(context, mem_cb) },
    }
}

extern "system" fn query(context: *mut *mut c_void, desc: *mut Header) -> u32 {
    // SAFETY: caller's query descriptor.
    let Some(header) = (unsafe { desc.as_ref() }) else {
        return ERROR;
    };
    let ours = !context.is_null() && is_our_context(unsafe { *context });
    match header.kind {
        QUERY_JITTER_PHASE_COUNT => {
            // SAFETY: matched by kind.
            let q = unsafe { &*desc.cast::<QueryJitterPhaseCount>() };
            if !q.out_phase_count.is_null() {
                unsafe { *q.out_phase_count = jitter_phase_count(q.render_width, q.display_width) };
            }
            return OK;
        }
        QUERY_JITTER_OFFSET => {
            // SAFETY: matched by kind.
            let q = unsafe { &*desc.cast::<QueryJitterOffset>() };
            let (x, y) = jitter_offset(q.index, q.phase_count);
            if !q.out_x.is_null() {
                unsafe { *q.out_x = x };
            }
            if !q.out_y.is_null() {
                unsafe { *q.out_y = y };
            }
            return OK;
        }
        QUERY_UPSCALE_RATIO => {
            // SAFETY: matched by kind.
            let q = unsafe { &*desc.cast::<QueryUpscaleRatio>() };
            if !q.out_ratio.is_null() {
                unsafe { *q.out_ratio = upscale_ratio_for_quality(q.quality_mode) };
            }
            return OK;
        }
        QUERY_RENDER_RESOLUTION => {
            // SAFETY: matched by kind.
            let q = unsafe { &*desc.cast::<QueryRenderResolution>() };
            let ratio = upscale_ratio_for_quality(q.quality_mode);
            if !q.out_render_width.is_null() {
                unsafe { *q.out_render_width = ((q.display_width as f32) / ratio).round() as u32 };
            }
            if !q.out_render_height.is_null() {
                unsafe { *q.out_render_height = ((q.display_height as f32) / ratio).round() as u32 };
            }
            return OK;
        }
        QUERY_MEMORY_USAGE if ours => {
            // SAFETY: matched by kind.
            let q = unsafe { &*desc.cast::<QueryMemoryUsage>() };
            if !q.usage.is_null() {
                unsafe { *q.usage = [0, 0] };
            }
            return OK;
        }
        QUERY_MEMORY_USAGE_V2 => {
            // SAFETY: matched by kind.
            let q = unsafe { &*desc.cast::<QueryMemoryUsageV2>() };
            if !q.usage.is_null() {
                unsafe { *q.usage = [0, 0] };
            }
            return OK;
        }
        QUERY_PROVIDER_VERSION if ours => {
            // SAFETY: matched by kind.
            let q = unsafe { &mut *desc.cast::<QueryProviderVersion>() };
            q.id = 0x0004_0500;
            q.name = c"4.5.0-DLSS".as_ptr();
            return OK;
        }
        other if ours => {
            static TOLD: std::sync::Once = std::sync::Once::new();
            TOLD.call_once(|| {
                tracing::warn!(kind = format_args!("{other:#x}"), "unhandled FidelityFX query on host upscale context");
            });
            return OK;
        }
        _ => {}
    }
    match REAL_QUERY.load(Ordering::Relaxed) {
        0 => ERROR,
        // SAFETY: title DLL's `ffxQuery`.
        real => unsafe { std::mem::transmute::<usize, QueryFn>(real)(context, desc) },
    }
}

extern "system" fn dispatch(context: *mut *mut c_void, desc: *const Header) -> u32 {
    // SAFETY: the title passes a dispatch description, whose header says
    // what follows it.
    if let Some(header) = unsafe { desc.as_ref() } {
        static SEEN: AtomicU32 = AtomicU32::new(0);
        let first = SEEN.fetch_add(1, Ordering::Relaxed) < 4;
        if header.kind == UPSCALE {
            // SAFETY: an upscale description, by its type.
            let d = unsafe { &*desc.cast::<Upscale>() };
            if first {
                tracing::info!(
                    list = ?d.list,
                    colour = ?d.colour.resource,
                    colour_size = ?(d.colour.description[2], d.colour.description[3]),
                    output = ?d.output.resource,
                    output_size = ?(d.output.description[2], d.output.description[3]),
                    render = ?d.render,
                    upscale = ?d.upscale,
                    "upscale dispatch"
                );
            }
            #[cfg(windows)]
            if probe() {
                static N: AtomicU32 = AtomicU32::new(0);
                let n = N.fetch_add(1, Ordering::Relaxed);
                // the first 12 upscales, then one in 300
                if n < 12 || n.is_multiple_of(300) {
                    let inputs = victory_gpu::list::upscale_inputs(&[
                        d.colour.resource,
                        d.depth.resource,
                        d.motion.resource,
                        d.exposure.resource,
                        d.reactive.resource,
                        d.composition.resource,
                        d.output.resource,
                    ]);
                    tracing::info!(
                        n,
                        colour = %inputs[0],
                        depth = %inputs[1],
                        motion = %inputs[2],
                        exposure = %inputs[3],
                        reactive = %inputs[4],
                        composition = %inputs[5],
                        output = %inputs[6],
                        jitter = ?d.jitter,
                        motion_scale = ?d.motion_scale,
                        render = ?d.render,
                        upscale = ?d.upscale,
                        sharpening = d.enable_sharpening,
                        sharpness = d.sharpness,
                        frame_time_delta = d.frame_time_delta,
                        pre_exposure = d.pre_exposure,
                        reset = d.reset,
                        near = d.camera_near,
                        far = d.camera_far,
                        fov = d.camera_fov,
                        meters = d.view_to_meters,
                        flags = format_args!("{:#x}", d.flags),
                        "upscale probe"
                    );
                }
            }
            #[cfg(windows)]
            {
                let up = victory_gpu::list::Upscale {
                    colour: d.colour.resource as u64,
                    depth: d.depth.resource as u64,
                    motion: d.motion.resource as u64,
                    exposure: d.exposure.resource as u64,
                    output: d.output.resource as u64,
                    render: d.render,
                    jitter: d.jitter,
                    motion_scale: d.motion_scale,
                    reset: d.reset != 0,
                    pre_exposure: d.pre_exposure,
                    near: d.camera_near,
                    far: d.camera_far,
                    frame_ms: d.frame_time_delta,
                    create_flags: if !context.is_null() {
                        // SAFETY: caller's context pointer.
                        our_context_flags(unsafe { *context }).unwrap_or(0)
                    } else {
                        0
                    },
                    dlss: want_dlss(),
                };
                if victory_gpu::list::upscale(d.list, &up) {
                    return 0;
                }
            }
            if first {
                tracing::warn!("upscale dispatch not stretched: left to the title's upscaler");
            }
        } else if first {
            tracing::info!(
                kind = format_args!("{:#x}", header.kind),
                "another FidelityFX dispatch"
            );
        }
    }
    // Never pass our synthetic sentinel context pointer to the title's real FSR DLL.
    if !context.is_null() && is_our_context(unsafe { *context }) {
        return ERROR;
    }
    match REAL_DISPATCH.load(Ordering::Relaxed) {
        // Neither stretched nor passed on: the output was not written, and
        // the title is told so instead of presenting it as it is.
        0 => {
            static TOLD: std::sync::Once = std::sync::Once::new();
            TOLD.call_once(|| {
                tracing::error!(
                    "a FidelityFX dispatch was not answered: no stretch was recorded and the title's own ffxDispatch was not found"
                );
            });
            ERROR
        }
        // SAFETY: the title's own `ffxDispatch`, with its own arguments.
        real => unsafe { std::mem::transmute::<usize, Dispatch>(real)(context, desc) },
    }
}

/// Resolves one of the FidelityFX loader entry points we intercept when
/// `VICTORY_UPSCALE=dlss|stretch` is active.
pub fn resolve(name: &str) -> Option<usize> {
    match name {
        "ffxCreateContext" => Some(create_context as *const () as usize),
        "ffxDestroyContext" => Some(destroy_context as *const () as usize),
        "ffxQuery" => Some(query as *const () as usize),
        "ffxDispatch" => Some(dispatch as *const () as usize),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::mem::offset_of;

    /// The upscale description as the public API lays it out: what the
    /// stretch reads sits where the title wrote it.
    #[test]
    fn the_upscale_description_is_laid_out_as_the_public_api() {
        assert_eq!(size_of::<Header>(), 16);
        assert_eq!(offset_of!(Upscale, list), 0x10);
        assert_eq!(offset_of!(Upscale, colour), 0x18);
        assert_eq!(offset_of!(Upscale, depth), 0x48);
        assert_eq!(offset_of!(Upscale, motion), 0x78);
        assert_eq!(offset_of!(Upscale, output), 0x138);
        assert_eq!(offset_of!(Upscale, jitter), 0x168);
        assert_eq!(offset_of!(Upscale, motion_scale), 0x170);
        assert_eq!(offset_of!(Upscale, render), 0x178);
        assert_eq!(offset_of!(Upscale, upscale), 0x180);
        assert_eq!(offset_of!(Upscale, enable_sharpening), 0x188);
        assert_eq!(offset_of!(Upscale, sharpness), 0x18c);
        assert_eq!(offset_of!(Upscale, reset), 0x198);
        assert_eq!(offset_of!(Upscale, camera_near), 0x19c);
        assert_eq!(offset_of!(Upscale, flags), 0x1ac);
        assert_eq!(offset_of!(Resource, description), 8);
        assert_eq!(offset_of!(Resource, state), 40);
    }

    #[test]
    fn jitter_queries_never_return_zero_and_stay_in_half_pixel_bounds() {
        assert_eq!(jitter_phase_count(1920, 2560), 15);
        assert_eq!(jitter_phase_count(1280, 1920), 18);
        assert_eq!(jitter_phase_count(1280, 2560), 32);
        for i in 0..32 {
            let (x, y) = jitter_offset(i, 15);
            assert!((-0.5..=0.5).contains(&x), "x={x}");
            assert!((-0.5..=0.5).contains(&y), "y={y}");
        }
    }

    /// A dispatch that is not an upscale, with no DLL of the title's to
    /// pass it to, is an error and not a silent success.
    #[test]
    fn a_dispatch_nothing_answers_is_an_error() {
        let before = REAL_DISPATCH.swap(0, Ordering::Relaxed);
        let other = Header {
            kind: 0x0002_0001,
            next: std::ptr::null(),
        };
        assert_eq!(dispatch(std::ptr::null_mut(), &other), ERROR);
        assert_eq!(dispatch(std::ptr::null_mut(), std::ptr::null()), ERROR);
        REAL_DISPATCH.store(before, Ordering::Relaxed);
    }
}
