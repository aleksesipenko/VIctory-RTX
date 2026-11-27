use super::capture::{Build, Checked, Geometry, Inputs, Op};
use crate::host::HostList;
use windows::Win32::Graphics::{Direct3D12::*, Dxgi::Common::*};

/// Bind and refresh exactly the same host copy, at record time. A later lookup
/// by guest VA could resolve to a different buffer after the title reuses memory.
pub(crate) struct Work {
    pub(super) op: Op,
    pub(super) input: Checked<Native>,
    #[cfg_attr(not(any(test, feature = "diagnostics")), allow(dead_code))]
    // pins through the last-use fence
    pub(super) buffers: Vec<Binding>,
}

#[cfg_attr(not(any(test, feature = "diagnostics")), allow(dead_code))] // ranges for the diagnostic GPU packet
pub(super) struct Binding {
    pub guest: u64,
    pub bytes: u64,
    pub offset: u64,
    pub buffer: ID3D12Resource,
}

pub(super) enum Native {
    Blas(Vec<D3D12_RAYTRACING_GEOMETRY_DESC>),
    Tlas(u32, u64),
    Copy,
}

impl Work {
    pub(super) fn prepare(op: Op, list: &HostList) -> Self {
        Self::bind(op, |va, len| {
            let s = crate::shadow::dxr_input(va, len)
                .ok_or("GPU input unavailable or outside peak budget")?;
            if s.left(va) < len
                || s.offset(va)
                    .checked_add(len)
                    .is_none_or(|end| end > s.width())
            {
                return Err("GPU input outside host buffer");
            }
            s.refresh(list, va, len);
            Ok((s.buffer.clone(), s.offset(va)))
        })
    }

    pub(super) fn bind(
        op: Op,
        mut resolve: impl FnMut(u64, u64) -> Checked<(ID3D12Resource, u64)>,
    ) -> Self {
        let mut buffers = Vec::new();
        let mut addr = |va: u64, len: u64, align: u64| -> Checked<u64> {
            if va == 0 || !va.is_multiple_of(align) || len == 0 || va.checked_add(len).is_none() {
                return Err("invalid GPU input range");
            }
            let (buf, off) = resolve(va, len)?;
            // SAFETY: live buffer supplied for the full range.
            let gpu = unsafe { buf.GetGPUVirtualAddress() }
                .checked_add(off)
                .ok_or("GPU address overflow")?;
            if !gpu.is_multiple_of(align) {
                return Err("unaligned host GPU input");
            }
            buffers.push(Binding {
                guest: va,
                bytes: len,
                offset: off,
                buffer: buf,
            });
            Ok(gpu)
        };
        let input = match &op {
            Op::Rejected { why, .. } => Err(*why),
            Op::Copy { .. } => Ok(Native::Copy),
            Op::Build(b) => translate(b, &mut addr),
        };
        Self { op, input, buffers }
    }
}

pub(super) fn translate(
    b: &Build,
    addr: &mut impl FnMut(u64, u64, u64) -> Checked<u64>,
) -> Checked<Native> {
    if let Some(r) = &b.rehydrate {
        if r.hr < 0 {
            return Err("rehydrate HRESULT");
        }
        if r.header.as_ref().map_err(|e| *e)?[0] & 0x0100_0000 == 0 {
            return Err("serialized flag is clear");
        }
    }
    if b.desc.dest_va == 0 || !b.desc.dest_va.is_multiple_of(256) {
        return Err("invalid AS destination");
    }
    if !b.stream.is_empty() {
        return Err("unsupported XDXR geometry stream");
    }
    match b.inputs.as_ref().map_err(|e| *e)? {
        Inputs::Tlas { count, address } => {
            // Phase 2 patches arrays of 64-byte instance records, dispatching in X.
            let gpu = if *count == 0 {
                0
            } else {
                addr(*address, u64::from(*count) * 64, 16)?
            };
            Ok(Native::Tlas(*count, gpu))
        }
        Inputs::Blas(specs) => {
            let mut geoms = Vec::with_capacity(specs.len());
            let mut primitives = 0u64;
            for g in specs {
                let (desc, count) = geometry(g, addr)?;
                primitives = primitives
                    .checked_add(count)
                    .ok_or("primitive count overflow")?;
                if primitives > (1 << 29) {
                    return Err("DXR primitive limit");
                }
                geoms.push(desc);
            }
            Ok(Native::Blas(geoms))
        }
    }
}

fn span(count: u64, stride: u64, element: u64) -> Checked<u64> {
    count
        .checked_sub(1)
        .and_then(|n| n.checked_mul(stride))
        .and_then(|n| n.checked_add(element))
        .ok_or("invalid geometry extent")
}

fn geometry(
    g: &Geometry,
    addr: &mut impl FnMut(u64, u64, u64) -> Checked<u64>,
) -> Checked<(D3D12_RAYTRACING_GEOMETRY_DESC, u64)> {
    let (ty, flags, data, count) = match *g {
        Geometry::Triangles(flags, t) => {
            let vf = DXGI_FORMAT(crate::resource::pc_format(t.vertex_format) as i32);
            let (bytes, align) = match vf {
                DXGI_FORMAT_R32G32B32_FLOAT => (12, 4),
                DXGI_FORMAT_R32G32_FLOAT => (8, 4),
                DXGI_FORMAT_R16G16B16A16_FLOAT
                | DXGI_FORMAT_R16G16B16A16_SNORM
                | DXGI_FORMAT_R16G16B16A16_UNORM => (8, 2),
                DXGI_FORMAT_R16G16_FLOAT | DXGI_FORMAT_R16G16_SNORM | DXGI_FORMAT_R16G16_UNORM => {
                    (4, 2)
                }
                DXGI_FORMAT_R8G8B8A8_SNORM | DXGI_FORMAT_R8G8B8A8_UNORM => (4, 1),
                DXGI_FORMAT_R8G8_SNORM | DXGI_FORMAT_R8G8_UNORM => (2, 1),
                DXGI_FORMAT_R10G10B10A2_UNORM => (4, 4),
                _ => return Err("unsupported DXR vertex format"),
            };
            if t.vertex_buffer_stride % align != 0 {
                return Err("invalid vertex stride");
            }
            let vb = addr(
                t.vertex_buffer_va,
                span(u64::from(t.vertex_count), t.vertex_buffer_stride, bytes)?,
                align,
            )?;
            let raw_idx_fmt = DXGI_FORMAT(crate::resource::pc_format(t.index_format) as i32);
            // Xbox Series X `XboxBvhBuilder.dll` accepts typeless 16/32-bit index formats
            // (`DXGI_FORMAT_R16_TYPELESS` = 53, `DXGI_FORMAT_R32_TYPELESS` = 39), whereas
            // PC D3D12 DXR 1.1 requires canonical `R16_UINT` (57) / `R32_UINT` (42) and
            // `DXGI_FORMAT_UNKNOWN` with a null `IndexBuffer` for non-indexed triangles.
            let (fmt, ib, index_count, vertices) =
                match (raw_idx_fmt, t.index_buffer, t.index_count) {
                    (DXGI_FORMAT_UNKNOWN, _, 0) => (DXGI_FORMAT_UNKNOWN, 0, 0, t.vertex_count),
                    (
                        DXGI_FORMAT_R16_UINT
                        | DXGI_FORMAT_R16_TYPELESS
                        | DXGI_FORMAT_R32_UINT
                        | DXGI_FORMAT_R32_TYPELESS,
                        0,
                        0,
                    ) => (DXGI_FORMAT_UNKNOWN, 0, 0, t.vertex_count),
                    (DXGI_FORMAT_R16_UINT | DXGI_FORMAT_R16_TYPELESS, va, n) if n > 0 => {
                        (DXGI_FORMAT_R16_UINT, addr(va, u64::from(n) * 2, 2)?, n, n)
                    }
                    (DXGI_FORMAT_R32_UINT | DXGI_FORMAT_R32_TYPELESS, va, n) if n > 0 => {
                        (DXGI_FORMAT_R32_UINT, addr(va, u64::from(n) * 4, 4)?, n, n)
                    }
                    _ => return Err("invalid index format or buffer"),
                };
            if vertices % 3 != 0 {
                return Err("incomplete triangle");
            }
            let transform = if t.transform3x4 == 0 {
                0
            } else {
                addr(t.transform3x4, 48, 16)?
            };
            let triangles = D3D12_RAYTRACING_GEOMETRY_TRIANGLES_DESC {
                Transform3x4: transform,
                IndexFormat: fmt,
                VertexFormat: vf,
                IndexCount: index_count,
                VertexCount: t.vertex_count,
                IndexBuffer: ib,
                VertexBuffer: D3D12_GPU_VIRTUAL_ADDRESS_AND_STRIDE {
                    StartAddress: vb,
                    StrideInBytes: t.vertex_buffer_stride,
                },
            };
            (
                D3D12_RAYTRACING_GEOMETRY_TYPE_TRIANGLES,
                flags,
                D3D12_RAYTRACING_GEOMETRY_DESC_0 {
                    Triangles: triangles,
                },
                u64::from(vertices / 3),
            )
        }
        Geometry::Boxes(flags, a) => {
            if a.aabbs_stride % 8 != 0 {
                return Err("invalid AABB stride");
            }
            let gpu = addr(a.aabbs_va, span(a.aabb_count, a.aabbs_stride, 24)?, 8)?;
            (
                D3D12_RAYTRACING_GEOMETRY_TYPE_PROCEDURAL_PRIMITIVE_AABBS,
                flags,
                D3D12_RAYTRACING_GEOMETRY_DESC_0 {
                    AABBs: D3D12_RAYTRACING_GEOMETRY_AABBS_DESC {
                        AABBCount: a.aabb_count,
                        AABBs: D3D12_GPU_VIRTUAL_ADDRESS_AND_STRIDE {
                            StartAddress: gpu,
                            StrideInBytes: a.aabbs_stride,
                        },
                    },
                },
                a.aabb_count,
            )
        }
    };
    if flags & !3 != 0 {
        return Err("unsupported geometry flags");
    }
    Ok((
        D3D12_RAYTRACING_GEOMETRY_DESC {
            Type: ty,
            Flags: D3D12_RAYTRACING_GEOMETRY_FLAGS(flags as i32),
            Anonymous: data,
        },
        count,
    ))
}

#[derive(Clone, PartialEq, Eq)]
pub(super) struct UpdateGeometry {
    flags: D3D12_RAYTRACING_GEOMETRY_FLAGS,
    format: DXGI_FORMAT,
    vertices: u32,
    stride: u64,
}

/// DXR forbids changing index contents or active/inactive primitives on an
/// update. A GPU write can change a float vertex, transform or AABB to NaN.
/// Unindexed normalized vertices with no transform cannot become inactive.
pub(super) fn update_shape(
    geoms: &[D3D12_RAYTRACING_GEOMETRY_DESC],
) -> Option<Vec<UpdateGeometry>> {
    geoms
        .iter()
        .map(|g| {
            if g.Type != D3D12_RAYTRACING_GEOMETRY_TYPE_TRIANGLES {
                return None;
            }
            // SAFETY: the discriminant selects the triangle union member.
            let t = unsafe { g.Anonymous.Triangles };
            if t.IndexBuffer != 0
                || t.Transform3x4 != 0
                || !matches!(
                    t.VertexFormat,
                    DXGI_FORMAT_R16G16B16A16_SNORM
                        | DXGI_FORMAT_R16G16B16A16_UNORM
                        | DXGI_FORMAT_R16G16_SNORM
                        | DXGI_FORMAT_R16G16_UNORM
                        | DXGI_FORMAT_R8G8B8A8_SNORM
                        | DXGI_FORMAT_R8G8B8A8_UNORM
                        | DXGI_FORMAT_R8G8_SNORM
                        | DXGI_FORMAT_R8G8_UNORM
                        | DXGI_FORMAT_R10G10B10A2_UNORM
                )
            {
                return None;
            }
            Some(UpdateGeometry {
                flags: g.Flags,
                format: t.VertexFormat,
                vertices: t.VertexCount,
                stride: t.VertexBuffer.StrideInBytes,
            })
        })
        .collect()
}
