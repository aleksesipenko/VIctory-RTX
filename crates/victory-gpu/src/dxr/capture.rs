//! Copy CPU descriptors at the DLL call. Addresses of GPU data are resolved separately.
//! A bad element rejects its entire AS, preserving geometry indices and batch accounting.
use super::abi::*;
use crate::object::This;
use std::ffi::c_void;
use std::mem::MaybeUninit;
use windows_sys::Win32::System::{
    Diagnostics::Debug::ReadProcessMemory, Threading::GetCurrentProcess,
};

pub(super) type Checked<T> = Result<T, &'static str>;

#[derive(Debug)]
pub(super) enum Geometry {
    Triangles(u32, GuestGeometryTrianglesDesc),
    Boxes(u32, GuestGeometryAabbsDesc),
}

#[derive(Debug)]
pub(super) enum Inputs {
    Blas(Vec<Geometry>),
    Tlas { count: u32, address: u64 },
}

#[derive(Debug)]
pub(super) struct Build {
    pub desc: GuestBuildAsDesc,
    pub inputs: Checked<Inputs>,
    #[cfg_attr(not(any(test, feature = "diagnostics")), allow(dead_code))] // L1 packet payload
    pub postbuild: Vec<GuestPostbuildInfoDesc>,
    pub stream: Vec<u8>,
    pub rehydrate: Option<Rehydrate>,
}

#[derive(Debug)]
pub(super) struct Rehydrate {
    #[cfg_attr(not(any(test, feature = "diagnostics")), allow(dead_code))]
    // original header selection in L1
    pub extra: Option<u64>,
    pub header: Checked<[u32; 16]>,
    pub hr: i32,
}

#[derive(Debug)]
pub(super) enum Op {
    Build(Build),
    Copy { dst: u64, src: u64, mode: u32 },
    Rejected { count: u32, why: &'static str },
}

#[derive(Debug, Default)]
pub struct Captured(pub(super) Vec<Op>);

impl Captured {
    pub fn copy(dst: u64, src: u64, mode: u32) -> Self {
        Self(vec![Op::Copy { dst, src, mode }])
    }

    pub fn rehydrated(&mut self, hr: i32) {
        for op in &mut self.0 {
            if let Op::Build(b) = op
                && let Some(r) = &mut b.rehydrate
            {
                r.hr = hr;
            }
        }
    }

    /// # Safety
    /// A non-null pointer must name a live victory command list being recorded.
    pub unsafe fn enqueue(self, list: *const c_void) {
        if self.0.is_empty() {
            return;
        }
        if list.is_null() {
            super::unrecorded(self.count(), "null command list");
        } else {
            crate::list::queue_dxr(list as This, self);
        }
    }

    pub(crate) fn count(&self) -> u64 {
        self.0.iter().map(Op::item_count).sum()
    }
}

impl Op {
    pub(super) fn item_count(&self) -> u64 {
        match self {
            Op::Rejected { count, .. } => u64::from(*count),
            _ => 1,
        }
    }
}

// Only called on the plain integer/union ABI types above. ReadProcessMemory reports
// short/unmapped reads, including a dangling nested pointer, without touching guest pages.
fn read<T: Copy>(address: u64) -> Checked<T> {
    let mut out = MaybeUninit::<T>::uninit();
    let mut read = 0;
    // SAFETY: our output spans size_of::<T>(); all bit patterns of the ABI types are valid.
    unsafe {
        if address == 0
            || ReadProcessMemory(
                GetCurrentProcess(),
                address as _,
                out.as_mut_ptr().cast(),
                size_of::<T>(),
                &mut read,
            ) == 0
            || read != size_of::<T>()
        {
            Err("unreadable CPU descriptor")
        } else {
            Ok(out.assume_init())
        }
    }
}

fn element<T: Copy>(base: u64, i: u32) -> Checked<T> {
    read(
        base.checked_add(u64::from(i) * size_of::<T>() as u64)
            .ok_or("CPU array overflow")?,
    )
}

fn inputs(d: GuestBuildAsInputs) -> Checked<Inputs> {
    match d.kind {
        0 if d.descs_layout == 0 && d.num_descs <= (1 << 24) => Ok(Inputs::Tlas {
            count: d.num_descs,
            address: d.ptr_or_va,
        }),
        1 if d.num_descs > 0 && d.num_descs <= 65_536 => {
            let mut geoms = Vec::new();
            for i in 0..d.num_descs {
                let g: GuestGeometryDesc = match d.descs_layout {
                    0 => element(d.ptr_or_va, i)?,
                    1 => read(element::<u64>(d.ptr_or_va, i)?)?,
                    _ => return Err("unsupported geometry layout"),
                };
                // SAFETY: the discriminant selects the plain ABI payload.
                geoms.push(unsafe {
                    match g.kind {
                        0 => Geometry::Triangles(g.flags, g.data.triangles),
                        1 => Geometry::Boxes(g.flags, g.data.aabbs),
                        _ => return Err("unsupported geometry type"),
                    }
                });
            }
            Ok(Inputs::Blas(geoms))
        }
        _ => Err("unsupported AS type, count or layout"),
    }
}

fn batch(count: u32, mut item: impl FnMut(u32) -> Checked<Build>) -> Captured {
    if count > 65_536 {
        return Captured(vec![Op::Rejected {
            count,
            why: "batch descriptor limit",
        }]);
    }
    Captured(
        (0..count)
            .map(|i| match item(i) {
                Ok(b) => Op::Build(b),
                Err(why) => Op::Rejected { count: 1, why },
            })
            .collect(),
    )
}

fn extras(e: GuestBatchedDesc2) -> Checked<(Vec<GuestPostbuildInfoDesc>, Vec<u8>)> {
    if e.num_postbuild_info_descs > 4096 {
        return Err("postbuild descriptor limit");
    }
    let postbuild = (0..e.num_postbuild_info_descs)
        .map(|i| element(e.postbuild_info_descs, i))
        .collect::<Checked<Vec<_>>>()?;
    let len = e.stream_desc.size_in_bytes;
    if len > 1 << 20 {
        return Err("XDXR stream descriptor limit");
    }
    let mut stream = vec![0; len as usize];
    if len != 0 {
        let mut got = 0;
        // SAFETY: sized output; the source is checked by Windows.
        if unsafe {
            ReadProcessMemory(
                GetCurrentProcess(),
                e.stream_desc.data as _,
                stream.as_mut_ptr().cast(),
                stream.len(),
                &mut got,
            )
        } == 0
            || got != stream.len()
        {
            return Err("unreadable XDXR stream");
        }
    }
    Ok((postbuild, stream))
}

pub fn capture_build(count: u32, descs: *const GuestBatchedDesc2) -> Captured {
    batch(count, |i| {
        let e: GuestBatchedDesc2 = element(descs as u64, i)?;
        let (inputs, postbuild, stream) = match extras(e) {
            Ok((p, s)) => (inputs(e.desc.inputs), p, s),
            Err(e) => (Err(e), vec![], vec![]),
        };
        Ok(Build {
            desc: e.desc,
            inputs,
            postbuild,
            stream,
            rehydrate: None,
        })
    })
}

pub fn capture_rehydrate(
    count: u32,
    descs: *const GuestBuildAsDesc,
    extra: *const u64,
) -> Captured {
    batch(count, |i| {
        let d: GuestBuildAsDesc = element(descs as u64, i)?;
        let selected = (!extra.is_null())
            .then(|| element(extra as u64, i))
            .transpose();
        // XboxBvhBuilder.dll +0x14780: a present extra array takes precedence even
        // if its element is zero. +0x147a1 tests the serialized header, not Inputs.Flags.
        let header = selected.and_then(|item| read(item.unwrap_or(d.source_va)));
        let extra = selected.unwrap_or(None);
        Ok(Build {
            desc: d,
            inputs: inputs(d.inputs),
            postbuild: vec![],
            stream: vec![],
            rehydrate: Some(Rehydrate {
                extra,
                header,
                hr: 0,
            }),
        })
    })
}
