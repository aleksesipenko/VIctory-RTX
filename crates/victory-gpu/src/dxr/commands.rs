use super::{Entry, capture::Checked, memory::Memory};
use crate::host::{compile, transition};
use std::{collections::HashMap, sync::Arc};
use windows::Win32::Graphics::Direct3D12::*;
use windows::core::{Interface, s};

pub(super) const SLOTS: usize = 65536;
pub(super) const PATCH_REPORT: u64 = 8 + SLOTS as u64 / 8;
pub(super) type Slot = (usize, u64, Arc<Entry>);

pub(super) fn barrier(list: &ID3D12GraphicsCommandList4) {
    // SAFETY: default UAV union is a null pResource (all UAV/AS writes).
    unsafe {
        list.ResourceBarrier(&[D3D12_RESOURCE_BARRIER {
            Type: D3D12_RESOURCE_BARRIER_TYPE_UAV,
            ..Default::default()
        }])
    };
}

pub(super) fn upload(mem: &mut Memory, data: &[u8]) -> Checked<ID3D12Resource> {
    let r = mem.buffer(
        data.len() as u64,
        D3D12_HEAP_TYPE_UPLOAD,
        D3D12_RESOURCE_STATE_GENERIC_READ,
        D3D12_RESOURCE_FLAG_NONE,
    )?;
    let mut ptr = std::ptr::null_mut();
    // SAFETY: exact-size mapped upload. It is immutable from Unmap until retirement.
    unsafe {
        r.Map(0, Some(&D3D12_RANGE::default()), Some(&mut ptr))
            .map_err(|_| "map upload")?;
        std::ptr::copy_nonoverlapping(data.as_ptr(), ptr.cast(), data.len());
        r.Unmap(0, None);
    }
    Ok(r)
}

pub(super) fn read_bytes(r: &ID3D12Resource, size: usize) -> Checked<Vec<u8>> {
    let mut ptr = std::ptr::null_mut();
    // SAFETY: caller has verified completion of the producing fence.
    unsafe {
        r.Map(
            0,
            Some(&D3D12_RANGE {
                Begin: 0,
                End: size,
            }),
            Some(&mut ptr),
        )
        .map_err(|_| "map DXR report")?;
        let bytes = std::slice::from_raw_parts(ptr.cast(), size).to_vec();
        r.Unmap(0, Some(&D3D12_RANGE::default()));
        Ok(bytes)
    }
}

pub(super) struct Table {
    pub buf: ID3D12Resource,
    pub slots: Vec<Slot>,
}
impl Table {
    pub fn new(
        mem: &mut Memory,
        by_title: &HashMap<u64, Arc<Entry>>,
    ) -> Checked<(Arc<Self>, ID3D12Resource)> {
        let mut words = vec![[0u32; 4]; SLOTS];
        let mut slots = Vec::new();
        for (&guest, entry) in by_title.iter().filter(|(_, e)| !e.top) {
            let key = guest >> 8;
            let lo = key as u32;
            let hi = (key >> 32) as u32;
            let mut h = lo ^ hi.wrapping_mul(0x9e3779b9);
            h ^= h >> 16;
            h = h.wrapping_mul(0x85ebca6b);
            h ^= h >> 13;
            h = h.wrapping_mul(0xc2b2ae35);
            h ^= h >> 16;
            let i = (0..128)
                .map(|n| (h as usize + n) & (SLOTS - 1))
                .find(|&i| words[i] == [0; 4])
                .ok_or("BLAS hash probe limit")?;
            let gpu = entry.storage.gpu;
            words[i] = [lo, hi, gpu as u32, (gpu >> 32) as u32];
            slots.push((i, guest, entry.clone()));
        }
        // SAFETY: no padding in arrays of four u32s. The shader reads little-endian words.
        let bytes = unsafe { std::slice::from_raw_parts(words.as_ptr().cast(), words.len() * 16) };
        let staging = upload(mem, bytes)?;
        let buf = mem.buffer(
            bytes.len() as u64,
            D3D12_HEAP_TYPE_DEFAULT,
            D3D12_RESOURCE_STATE_COMMON,
            D3D12_RESOURCE_FLAG_NONE,
        )?;
        Ok((Arc::new(Table { buf, slots }), staging))
    }
}

pub(super) struct Patcher {
    root: ID3D12RootSignature,
    pso: ID3D12PipelineState,
}
impl Patcher {
    pub fn new(dev: &ID3D12Device5) -> Checked<Self> {
        let code = compile(include_str!("patch.hlsl"), s!("main"), s!("cs_5_1"))
            .map_err(|_| "compile instance patcher")?;
        // SAFETY: embedded root signature, bytecode lives through PSO creation.
        unsafe {
            let root: ID3D12RootSignature = dev
                .CreateRootSignature(0, &code)
                .map_err(|_| "patch root signature")?;
            let pso = dev
                .CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
                    pRootSignature: std::mem::transmute_copy(&root),
                    CS: D3D12_SHADER_BYTECODE {
                        pShaderBytecode: code.as_ptr().cast(),
                        BytecodeLength: code.len(),
                    },
                    ..Default::default()
                })
                .map_err(|_| "patch compute PSO")?;
            Ok(Self { root, pso })
        }
    }

    pub fn record(
        &self,
        list: &ID3D12GraphicsCommandList4,
        count: u32,
        src: u64,
        table: &Table,
        dst: &ID3D12Resource,
        stats: u64,
    ) {
        // SAFETY: root views are sized by the checked allocations; parent flight pins all four.
        unsafe {
            list.SetComputeRootSignature(&self.root);
            list.SetPipelineState(&self.pso);
            list.SetComputeRootShaderResourceView(1, src);
            list.SetComputeRootShaderResourceView(2, table.buf.GetGPUVirtualAddress());
            list.SetComputeRootUnorderedAccessView(3, dst.GetGPUVirtualAddress());
            list.SetComputeRootUnorderedAccessView(4, stats);
            for first in (0..count).step_by(65535 * 64) {
                let args = [count, first];
                list.SetComputeRoot32BitConstants(0, 2, args.as_ptr().cast(), 0);
                list.Dispatch((count - first).div_ceil(64).min(65535), 1, 1);
            }
            list.ResourceBarrier(&[transition(
                dst,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
            )]);
        }
    }
}

/// The readback buffer carries only postbuild sizes, patch audit data and GPU timestamps.
/// The host never round-trips a vertex or instance to build a TLAS.
pub(super) struct Commands {
    pub alloc: ID3D12CommandAllocator,
    pub list: ID3D12GraphicsCommandList4,
    pub report: ID3D12Resource,
    pub read: ID3D12Resource,
    zero: ID3D12Resource,
    timestamps: ID3D12QueryHeap,
    pub capacity: u64,
}

impl Commands {
    pub fn new(mem: &mut Memory, capacity: u64) -> Checked<Self> {
        let report = mem.buffer(
            capacity,
            D3D12_HEAP_TYPE_DEFAULT,
            D3D12_RESOURCE_STATE_COMMON,
            D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
        )?;
        let read = mem.buffer(
            capacity,
            D3D12_HEAP_TYPE_READBACK,
            D3D12_RESOURCE_STATE_COPY_DEST,
            D3D12_RESOURCE_FLAG_NONE,
        )?;
        let zero = upload(mem, &vec![0; capacity as usize])?;
        // SAFETY: direct command list with matching allocator, closed for reset below.
        unsafe {
            let alloc = mem
                .dev
                .CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)
                .map_err(|_| "DXR allocator")?;
            let list: ID3D12GraphicsCommandList4 = mem
                .dev
                .CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc, None)
                .map_err(|_| "DXR command list")?;
            list.Close().map_err(|_| "close new DXR list")?;
            let mut timestamps = None;
            mem.dev
                .CreateQueryHeap(
                    &D3D12_QUERY_HEAP_DESC {
                        Type: D3D12_QUERY_HEAP_TYPE_TIMESTAMP,
                        Count: 2,
                        NodeMask: 0,
                    },
                    &mut timestamps,
                )
                .map_err(|_| "DXR timestamp heap")?;
            Ok(Self {
                alloc,
                list,
                report,
                read,
                zero,
                timestamps: timestamps.ok_or("null query heap")?,
                capacity,
            })
        }
    }

    pub fn begin(&self, size: u64) -> Checked<()> {
        // SAFETY: the allocator and reusable report are fence-retired. Every report
        // list explicitly leaves the buffer COMMON, including across Execute boundaries.
        unsafe {
            self.alloc
                .Reset()
                .and_then(|()| self.list.Reset(&self.alloc, None))
                .map_err(|_| "DXR list reset")?;
            self.list
                .EndQuery(&self.timestamps, D3D12_QUERY_TYPE_TIMESTAMP, 0);
            self.list.ResourceBarrier(&[transition(
                &self.report,
                D3D12_RESOURCE_STATE_COMMON,
                D3D12_RESOURCE_STATE_COPY_DEST,
            )]);
            self.list
                .CopyBufferRegion(&self.report, 0, &self.zero, 0, size);
            self.list.ResourceBarrier(&[transition(
                &self.report,
                D3D12_RESOURCE_STATE_COPY_DEST,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            )]);
            barrier(&self.list);
        }
        Ok(())
    }
    pub fn finish(&self, size: u64) -> Checked<ID3D12CommandList> {
        unsafe {
            self.list.ResourceBarrier(&[transition(
                &self.report,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                D3D12_RESOURCE_STATE_COPY_SOURCE,
            )]);
            self.list
                .CopyBufferRegion(&self.read, 0, &self.report, 0, size);
            self.list.ResourceBarrier(&[transition(
                &self.report,
                D3D12_RESOURCE_STATE_COPY_SOURCE,
                D3D12_RESOURCE_STATE_COMMON,
            )]);
            self.list
                .EndQuery(&self.timestamps, D3D12_QUERY_TYPE_TIMESTAMP, 1);
            self.list.ResolveQueryData(
                &self.timestamps,
                D3D12_QUERY_TYPE_TIMESTAMP,
                0,
                2,
                &self.read,
                0,
            );
            self.list.Close().map_err(|_| "close DXR command list")?;
            self.list.cast().map_err(|_| "DXR list interface")
        }
    }
}
