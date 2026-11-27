//! Diagnostic companion to the frame recorder. Every file ends in the BLAKE3
//! of every preceding byte. No AS is read with CopyBufferRegion: pre-frame AS
//! generations use the driver's SERIALIZE operation and its relocation table.
use super::*;
use std::{
    fs::{self, File, OpenOptions},
    io::{self, BufWriter, Write},
    path::Path,
};

fn checked<T>(r: Checked<T>) -> io::Result<T> {
    r.map_err(io::Error::other)
}
struct Output {
    w: BufWriter<File>,
    hash: blake3::Hasher,
}
impl Output {
    fn data(&mut self, b: &[u8]) -> io::Result<()> {
        self.w.write_all(b)?;
        self.hash.update(b);
        Ok(())
    }
    fn word(&mut self, n: u64) -> io::Result<()> {
        self.data(&n.to_le_bytes())
    }
    fn blob(&mut self, b: &[u8]) -> io::Result<()> {
        self.word(b.len() as u64)?;
        self.data(b)
    }
}

fn publish(path: &Path, write: impl FnOnce(&mut Output) -> io::Result<()>) -> io::Result<()> {
    let part = path.with_extension("partial");
    let file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&part)?;
    let mut out = Output {
        w: BufWriter::new(file),
        hash: blake3::Hasher::new(),
    };
    write(&mut out)?;
    out.w.write_all(out.hash.finalize().as_bytes())?;
    out.w.flush()?;
    drop(out);
    fs::rename(part, path)
}

fn metadata(out: &mut Output, op: &Op) -> io::Result<()> {
    match op {
        Op::Rejected { count, why } => {
            out.word(0)?;
            out.word(u64::from(*count))?;
            out.blob(why.as_bytes())
        }
        Op::Copy { dst, src, mode } => {
            out.word(1)?;
            out.word(*dst)?;
            out.word(*src)?;
            out.word(u64::from(*mode))
        }
        Op::Build(b) => {
            out.word(if b.rehydrate.is_some() { 3 } else { 2 })?;
            let d = &b.desc;
            for n in [
                d.dest_va,
                d.source_va,
                d.scratch_va,
                u64::from(d.inputs.kind),
                u64::from(d.inputs.flags),
                u64::from(d.inputs.num_descs),
                u64::from(d.inputs.descs_layout),
                d.inputs.ptr_or_va,
            ] {
                out.word(n)?;
            }
            if let Some(r) = &b.rehydrate {
                out.word(u64::from(r.extra.is_some()))?;
                out.word(r.extra.unwrap_or(0))?;
                out.word(r.hr as u32 as u64)?;
                out.word(u64::from(r.header.is_ok()))?;
                if let Ok(words) = r.header {
                    for w in words {
                        out.data(&w.to_le_bytes())?;
                    }
                }
            }
            out.word(b.postbuild.len() as u64)?;
            for p in &b.postbuild {
                out.word(p.dest_buffer)?;
                out.word(u64::from(p.info_type))?;
            }
            out.blob(&b.stream)?;
            match &b.inputs {
                Err(why) => {
                    out.word(0)?;
                    out.blob(why.as_bytes())
                }
                Ok(capture::Inputs::Tlas { count, address }) => {
                    out.word(1)?;
                    out.word(u64::from(*count))?;
                    out.word(*address)
                }
                Ok(capture::Inputs::Blas(geoms)) => {
                    out.word(2)?;
                    out.word(geoms.len() as u64)?;
                    for g in geoms {
                        match g {
                            capture::Geometry::Triangles(flags, t) => {
                                for n in [
                                    0,
                                    u64::from(*flags),
                                    t.transform3x4,
                                    u64::from(t.index_format),
                                    u64::from(t.vertex_format),
                                    u64::from(t.index_count),
                                    u64::from(t.vertex_count),
                                    t.index_buffer,
                                    t.vertex_buffer_va,
                                    t.vertex_buffer_stride,
                                ] {
                                    out.word(n)?;
                                }
                            }
                            capture::Geometry::Boxes(flags, a) => {
                                for n in [
                                    1,
                                    u64::from(*flags),
                                    a.aabb_count,
                                    a.aabbs_va,
                                    a.aabbs_stride,
                                ] {
                                    out.word(n)?;
                                }
                            }
                        }
                    }
                    Ok(())
                }
            }
        }
    }
}

struct Transfer {
    alloc: ID3D12CommandAllocator,
    list: ID3D12GraphicsCommandList4,
}
impl Transfer {
    fn new(e: &Engine) -> Checked<Self> {
        unsafe {
            let alloc = e
                .mem
                .dev
                .CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)
                .map_err(|_| "capture allocator")?;
            let list: ID3D12GraphicsCommandList4 = e
                .mem
                .dev
                .CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc, None)
                .map_err(|_| "capture list")?;
            list.Close().map_err(|_| "close capture list")?;
            Ok(Self { alloc, list })
        }
    }
    fn run(&self, e: &mut Engine, record: impl FnOnce(&ID3D12GraphicsCommandList4)) -> Checked<()> {
        unsafe {
            self.alloc
                .Reset()
                .and_then(|()| self.list.Reset(&self.alloc, None))
                .map_err(|_| "reset capture list")?;
            record(&self.list);
            self.list.Close().map_err(|_| "close capture")?;
            let cmd = self.list.cast().map_err(|_| "capture interface")?;
            e.queue.ExecuteCommandLists(&[Some(cmd)]);
            let n = e.next;
            e.next += 1;
            // A lost signal cannot be recovered by dropping buffers still in use.
            if e.queue.Signal(&e.fence, n).is_err() || e.wait(n).is_err() {
                crate::host::fail(format_args!("DXR capture fence failed"));
            }
            Ok(())
        }
    }
}

pub(super) fn save(
    e: &mut Engine,
    work: &[Work],
    folder: &Path,
    frame: u32,
    after: usize,
) -> io::Result<()> {
    victory_core::ensure_outside_repo(folder).map_err(io::Error::other)?;
    let folder = folder.join("native-rt");
    fs::create_dir_all(&folder)?;
    let transfer = checked(Transfer::new(e))?;
    let seed = folder.join("initial.vrt");
    if !seed.exists() {
        if let Some(f) = e.pending.back() {
            checked(e.wait(f.fence))?;
        }
        publish(&seed, |out| seeds(e, &transfer, out))?;
    }
    let seq = e.next;
    let name = if work.is_empty() {
        format!("frame-{frame}-opening-{after}.vrt")
    } else {
        format!("frame-{frame}-after-{after}-batch-{seq}.vrt")
    };
    publish(&folder.join(name), |out| {
        out.data(b"VRTBATCH")?;
        out.word(1)?;
        out.word(u64::from(frame))?;
        out.word(after as u64)?;
        out.word(seq)?;
        #[cfg(feature = "diagnostics")]
        {
            let pages = crate::memory::dxr_page_buffers();
            out.word(pages.len() as u64)?;
            for gpu in pages {
                let (id, offset) = crate::replay::record::dxr_buffer(gpu)
                    .ok_or_else(|| io::Error::other("uncaptured BVH page buffer"))?;
                out.word(u64::from(id))?;
                out.word(offset)?;
            }
        }
        #[cfg(not(feature = "diagnostics"))]
        out.word(0)?;
        out.word(work.len() as u64)?;
        let max = work
            .iter()
            .flat_map(|w| &w.buffers)
            .map(|b| b.bytes)
            .max()
            .unwrap_or(0)
            .min(8 << 20);
        let read = if max != 0 {
            Some(checked(e.mem.buffer(
                max,
                D3D12_HEAP_TYPE_READBACK,
                D3D12_RESOURCE_STATE_COPY_DEST,
                D3D12_RESOURCE_FLAG_NONE,
            ))?)
        } else {
            None
        };
        for w in work {
            metadata(out, &w.op)?;
            out.blob(w.input.as_ref().err().copied().unwrap_or("").as_bytes())?;
            out.word(w.buffers.len() as u64)?;
            for b in &w.buffers {
                out.word(b.guest)?;
                out.word(b.bytes)?;
                let read = read.as_ref().unwrap();
                let mut offset = 0;
                while offset < b.bytes {
                    let n = max.min(b.bytes - offset);
                    checked(transfer.run(e, |l| unsafe {
                        l.CopyBufferRegion(read, 0, &b.buffer, b.offset + offset, n)
                    }))?;
                    out.data(&checked(commands::read_bytes(read, n as usize))?)?;
                    offset += n;
                }
            }
        }
        Ok(())
    })
}

fn seeds(e: &mut Engine, t: &Transfer, out: &mut Output) -> io::Result<()> {
    let mut seen = std::collections::HashSet::new();
    let mut live: Vec<(u64, Arc<Entry>)> = Vec::new();
    for (&key, entry) in &e.entries {
        if entry.ready.load(Relaxed) && seen.insert(entry.storage.gpu) {
            live.push((key, entry.clone()));
        }
    }
    // Include older immutable BLAS generations retained only by a live TLAS's PinSlot
    // so serialized TLAS relocation tables always resolve every referenced BLAS GPU VA.
    for entry in e.entries.values() {
        if entry.top
            && entry.ready.load(Relaxed)
            && let Some(slot) = &entry.pins
        {
            for (blas_key, blas) in slot.lock().unwrap_or_else(|err| err.into_inner()).iter() {
                if blas.ready.load(Relaxed) && seen.insert(blas.storage.gpu) {
                    live.push((*blas_key, blas.clone()));
                }
            }
        }
    }
    live.sort_by_key(|(key, entry)| (entry.top, *key, entry.serial));
    let va: Vec<_> = live.iter().map(|(_, v)| v.storage.gpu).collect();
    out.data(b"VRTSEEDS")?;
    out.word(1)?;
    out.word(live.len() as u64)?;
    if va.is_empty() {
        return Ok(());
    }
    let n = (va.len() * 16) as u64;
    let info = checked(e.mem.buffer(
        n,
        D3D12_HEAP_TYPE_DEFAULT,
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
    ))?;
    let rb = checked(e.mem.buffer(
        n,
        D3D12_HEAP_TYPE_READBACK,
        D3D12_RESOURCE_STATE_COPY_DEST,
        D3D12_RESOURCE_FLAG_NONE,
    ))?;
    checked(t.run(e, |l| unsafe {
        l.EmitRaytracingAccelerationStructurePostbuildInfo(
            &D3D12_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO_DESC {
                DestBuffer: info.GetGPUVirtualAddress(),
                InfoType: D3D12_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO_SERIALIZATION,
            },
            &va,
        );
        l.ResourceBarrier(&[crate::host::transition(
            &info,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_COPY_SOURCE,
        )]);
        l.CopyBufferRegion(&rb, 0, &info, 0, n);
    }))?;
    let info = checked(commands::read_bytes(&rb, n as usize))?;
    let sizes: Vec<_> = info
        .chunks_exact(16)
        .map(|s| u64::from_le_bytes(s[..8].try_into().unwrap()))
        .collect();
    let max = *sizes.iter().max().unwrap();
    let staging = checked(e.mem.buffer(
        max,
        D3D12_HEAP_TYPE_DEFAULT,
        D3D12_RESOURCE_STATE_COMMON,
        D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
    ))?;
    let read = checked(e.mem.buffer(
        max,
        D3D12_HEAP_TYPE_READBACK,
        D3D12_RESOURCE_STATE_COPY_DEST,
        D3D12_RESOURCE_FLAG_NONE,
    ))?;
    for ((key, entry), size) in live.into_iter().zip(sizes) {
        if size < size_of::<D3D12_SERIALIZED_RAYTRACING_ACCELERATION_STRUCTURE_HEADER>() as u64 {
            return Err(io::Error::other("invalid AS serialization size"));
        }
        checked(t.run(e, |l| unsafe {
            l.ResourceBarrier(&[crate::host::transition(
                &staging,
                D3D12_RESOURCE_STATE_COMMON,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            )]);
            l.CopyRaytracingAccelerationStructure(
                staging.GetGPUVirtualAddress(),
                entry.storage.gpu,
                D3D12_RAYTRACING_ACCELERATION_STRUCTURE_COPY_MODE_SERIALIZE,
            );
            l.ResourceBarrier(&[crate::host::transition(
                &staging,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                D3D12_RESOURCE_STATE_COPY_SOURCE,
            )]);
            l.CopyBufferRegion(&read, 0, &staging, 0, size);
            l.ResourceBarrier(&[crate::host::transition(
                &staging,
                D3D12_RESOURCE_STATE_COPY_SOURCE,
                D3D12_RESOURCE_STATE_COMMON,
            )]);
        }))?;
        // Rebase the serialized TLAS's BLAS pointer table on load with this
        // stored-address -> guest key -> new-address map. DriverMatchingIdentifier
        // in the D3D12 serialization header must match the replay adapter.
        out.word(key)?;
        out.word(entry.serial)?;
        out.word(u64::from(entry.top))?;
        out.word(entry.storage.gpu)?;
        out.blob(&checked(commands::read_bytes(&read, size as usize))?)?;
    }
    Ok(())
}
