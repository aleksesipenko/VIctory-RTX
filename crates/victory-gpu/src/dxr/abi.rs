//! x64 console ABI, adapted from maren-halv/VIctory-DXR @ e1740b3 (GPL-3.0).
use std::mem::offset_of;
/// Guest `D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS` (`0x18` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct GuestBuildAsInputs {
    pub kind: u32,
    pub flags: u32,
    pub num_descs: u32,
    pub descs_layout: u32,
    pub ptr_or_va: u64,
}
const _: () = assert!(size_of::<GuestBuildAsInputs>() == 0x18);

/// Guest `D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC` (`0x30` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct GuestBuildAsDesc {
    pub dest_va: u64,
    pub inputs: GuestBuildAsInputs,
    pub source_va: u64,
    pub scratch_va: u64,
}
const _: () = assert!(size_of::<GuestBuildAsDesc>() == 0x30);
const _: () = assert!(offset_of!(GuestBuildAsDesc, inputs) == 0x08);
const _: () = assert!(offset_of!(GuestBuildAsDesc, source_va) == 0x20);
const _: () = assert!(offset_of!(GuestBuildAsDesc, scratch_va) == 0x28);

/// Guest `D3D12_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO_DESC` (`0x10` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct GuestPostbuildInfoDesc {
    pub dest_buffer: u64,
    pub info_type: u32,
    pub _pad: u32,
}
const _: () = assert!(size_of::<GuestPostbuildInfoDesc>() == 0x10);

/// Guest `D3D12XBOX_XDXR_STREAM_DESC` (`0x10` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct GuestStreamDesc {
    pub size_in_bytes: u64,
    pub data: u64,
}
const _: () = assert!(size_of::<GuestStreamDesc>() == 0x10);

/// Guest `D3D12XBOX_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_BATCHED_DESC2` (`0x50` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct GuestBatchedDesc2 {
    pub desc: GuestBuildAsDesc,
    pub num_postbuild_info_descs: u32,
    pub _pad: u32,
    pub postbuild_info_descs: u64,
    pub stream_desc: GuestStreamDesc,
}
const _: () = assert!(size_of::<GuestBatchedDesc2>() == 0x50);
const _: () = assert!(offset_of!(GuestBatchedDesc2, num_postbuild_info_descs) == 0x30);
const _: () = assert!(offset_of!(GuestBatchedDesc2, postbuild_info_descs) == 0x38);
const _: () = assert!(offset_of!(GuestBatchedDesc2, stream_desc) == 0x40);

/// Guest `D3D12_RAYTRACING_GEOMETRY_TRIANGLES_DESC` (`0x30` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct GuestGeometryTrianglesDesc {
    pub transform3x4: u64,
    pub index_format: u32,
    pub vertex_format: u32,
    pub index_count: u32,
    pub vertex_count: u32,
    pub index_buffer: u64,
    pub vertex_buffer_va: u64,
    pub vertex_buffer_stride: u64,
}
const _: () = assert!(size_of::<GuestGeometryTrianglesDesc>() == 0x30);

/// Guest `D3D12_RAYTRACING_GEOMETRY_AABBS_DESC` (`0x18` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct GuestGeometryAabbsDesc {
    pub aabb_count: u64,
    pub aabbs_va: u64,
    pub aabbs_stride: u64,
}
const _: () = assert!(size_of::<GuestGeometryAabbsDesc>() == 0x18);

#[repr(C)]
#[derive(Clone, Copy)]
pub union GuestGeometryUnion {
    pub triangles: GuestGeometryTrianglesDesc,
    pub aabbs: GuestGeometryAabbsDesc,
}

/// Guest `D3D12_RAYTRACING_GEOMETRY_DESC` (`0x38` bytes).
#[repr(C)]
#[derive(Clone, Copy)]
pub struct GuestGeometryDesc {
    pub kind: u32,
    pub flags: u32,
    pub data: GuestGeometryUnion,
}
const _: () = assert!(size_of::<GuestGeometryDesc>() == 0x38);
