//! XboxBvhBuilder.dll sidecar hooks. Export names and x64 signatures adapted from
//! maren-halv/VIctory-DXR @ e1740b3 (GPL-3.0). The title loader maps and attaches
//! the original module once, with its normal import resolver and TLS.
use std::ffi::c_void;
use std::sync::OnceLock;
use victory_gpu::dxr::{self, GuestBatchedDesc2, GuestBuildAsDesc};

pub const DLL: &str = "XboxBvhBuilder.dll";
const SYM_CTOR: &str = "??0XboxBvhBuilder@@QEAA@PEAUID3D12Device11@@_N@Z";
const SYM_DTOR: &str = "??1XboxBvhBuilder@@QEAA@XZ";
const SYM_BUILD_BATCHED_2X: &str = "?BuildRaytracingAccelerationStructureBatched2X@XboxBvhBuilder@@QEAAXPEAUID3D12GraphicsCommandList6@@IPEBUD3D12XBOX_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_BATCHED_DESC2@@@Z";
const SYM_COPY: &str = "?CopyRaytracingAccelerationStructure@XboxBvhBuilder@@QEAAXPEAUID3D12GraphicsCommandList4@@_K1W4D3D12_RAYTRACING_ACCELERATION_STRUCTURE_COPY_MODE@@@Z";
const SYM_PREBUILD: &str = "?GetRaytracingAccelerationStructurePrebuildInfo@XboxBvhBuilder@@QEAAXPEBUD3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS@@PEBUD3D12XBOX_XDXR_STREAM_DESC@@PEAUD3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO@@@Z";
const SYM_REHYDRATE_BATCHED_2X: &str = "?RehydrateRaytracingAccelerationStructureBatched2X@XboxBvhBuilder@@QEAAJPEAUID3D12GraphicsCommandList8@@IPEBUD3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC@@PEB_K@Z";
const SYM_SET_64BIT_MORTON: &str = "?Set64BitMortonCodes@XboxBvhBuilder@@QEAAX_N@Z";

type FnBuildBatched2X =
    unsafe extern "system" fn(*mut c_void, *const c_void, u32, *const GuestBatchedDesc2);
type FnCopy = unsafe extern "system" fn(*mut c_void, *const c_void, u64, u64, u32);
type FnRehydrateBatched2X = unsafe extern "system" fn(
    *mut c_void,
    *const c_void,
    u32,
    *const GuestBuildAsDesc,
    *const u64,
) -> i32;
struct Originals {
    build: FnBuildBatched2X,
    copy: FnCopy,
    rehydrate: FnRehydrateBatched2X,
}

static REAL: OnceLock<Originals> = OnceLock::new();

/// For functions with no sidecar work the import stays bound directly to the DLL.
/// Validate all seven up front, before any title code can call a wrapper.
pub fn install(image: &victory_loader::LoadedImage) -> std::io::Result<()> {
    let sym = |name: &str| {
        image
            .export(name)
            .ok_or_else(|| std::io::Error::other(format!("missing {DLL}!{name}")))
    };
    for name in [SYM_CTOR, SYM_DTOR, SYM_PREBUILD, SYM_SET_64BIT_MORTON] {
        sym(name)?;
    }
    // SAFETY: exact decorated export names uniquely identify these x64 signatures.
    let originals = unsafe {
        Originals {
            build: std::mem::transmute::<usize, FnBuildBatched2X>(sym(SYM_BUILD_BATCHED_2X)?),
            copy: std::mem::transmute::<usize, FnCopy>(sym(SYM_COPY)?),
            rehydrate: std::mem::transmute::<usize, FnRehydrateBatched2X>(sym(
                SYM_REHYDRATE_BATCHED_2X,
            )?),
        }
    };
    REAL.set(originals)
        .map_err(|_| std::io::Error::other("duplicate native RT builder installation"))?;
    tracing::info!("native RT: verified seven title builder imports");
    Ok(())
}

pub use dxr::requested;

pub fn resolve(name: &str) -> Option<usize> {
    REAL.get()?;
    Some(match name {
        SYM_BUILD_BATCHED_2X => build as *const () as usize,
        SYM_COPY => copy as *const () as usize,
        SYM_REHYDRATE_BATCHED_2X => rehydrate as *const () as usize,
        _ => return None,
    })
}

unsafe extern "system" fn build(
    this: *mut c_void,
    list: *const c_void,
    count: u32,
    descs: *const GuestBatchedDesc2,
) {
    let batch = dxr::capture_build(count, descs);
    // SAFETY: arguments as received from the title, forwarded without modification.
    unsafe { (REAL.get().unwrap().build)(this, list, count, descs) };
    unsafe { batch.enqueue(list) };
}

unsafe extern "system" fn copy(
    this: *mut c_void,
    list: *const c_void,
    dst: u64,
    src: u64,
    mode: u32,
) {
    // SAFETY: arguments as received. Only COMPACT has an effect in the console DLL.
    unsafe { (REAL.get().unwrap().copy)(this, list, dst, src, mode) };
    unsafe { dxr::Captured::copy(dst, src, mode).enqueue(list) };
}

unsafe extern "system" fn rehydrate(
    this: *mut c_void,
    list: *const c_void,
    count: u32,
    descs: *const GuestBuildAsDesc,
    extra: *const u64,
) -> i32 {
    let mut batch = dxr::capture_rehydrate(count, descs, extra);
    // SAFETY: the real DLL validates the whole batch; its HRESULT belongs to the caller.
    let hr = unsafe { (REAL.get().unwrap().rehydrate)(this, list, count, descs, extra) };
    if hr < 0 {
        tracing::warn!(target: "victory_gpu::host", count, hresult = format_args!("{:#010x}", hr as u32), "native RT rehydrate rejected by title DLL");
    }
    batch.rehydrated(hr);
    unsafe { batch.enqueue(list) };
    hr
}
