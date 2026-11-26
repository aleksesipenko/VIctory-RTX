use super::*;
use abi::{
    GuestBuildAsInputs, GuestGeometryDesc, GuestGeometryTrianglesDesc, GuestGeometryUnion,
    GuestPostbuildInfoDesc, GuestStreamDesc,
};
use windows::Win32::Graphics::Dxgi::{
    Common::*, CreateDXGIFactory2, DXGI_CREATE_FACTORY_FLAGS, IDXGIFactory4,
};

fn guest(dst: u64, kind: u32, count: u32, data: u64) -> GuestBatchedDesc2 {
    GuestBatchedDesc2 {
        desc: GuestBuildAsDesc {
            dest_va: dst,
            inputs: GuestBuildAsInputs {
                kind,
                flags: 0,
                num_descs: count,
                descs_layout: 0,
                ptr_or_va: data,
            },
            source_va: 0,
            scratch_va: 0,
        },
        num_postbuild_info_descs: 0,
        _pad: 0,
        postbuild_info_descs: 0,
        stream_desc: GuestStreamDesc {
            size_in_bytes: 0,
            data: 0,
        },
    }
}

fn triangle() -> GuestGeometryDesc {
    GuestGeometryDesc {
        kind: 0,
        flags: 1,
        data: GuestGeometryUnion {
            triangles: GuestGeometryTrianglesDesc {
                transform3x4: 0,
                index_format: DXGI_FORMAT_UNKNOWN.0 as u32,
                vertex_format: DXGI_FORMAT_R32G32B32_FLOAT.0 as u32,
                index_count: 0,
                vertex_count: 3,
                index_buffer: 0,
                vertex_buffer_va: 0x10_0000,
                vertex_buffer_stride: 12,
            },
        },
    }
}

#[test]
fn descriptor_batch_survives_reuse_and_rejects_each_bad_element() {
    let mut geom = triangle();
    let mut pointers = [&raw const geom as u64];
    let mut pb = GuestPostbuildInfoDesc {
        dest_buffer: 0xa000,
        info_type: 2,
        _pad: 0,
    };
    let mut e = guest(0x100, 1, 1, pointers.as_ptr() as u64);
    e.desc.inputs.descs_layout = 1;
    e.num_postbuild_info_descs = 1;
    e.postbuild_info_descs = &raw const pb as u64;
    let mut batch = [e, e];
    batch[1].desc.inputs.ptr_or_va = 0;
    let c = capture_build(2, batch.as_ptr());
    pointers[0] = 0;
    geom.flags = 0;
    pb.info_type = 99;
    std::hint::black_box((pointers, geom, pb));
    assert_eq!(c.count(), 2);
    let Op::Build(b) = &c.0[0] else { panic!() };
    let Ok(capture::Inputs::Blas(g)) = &b.inputs else {
        panic!()
    };
    assert!(matches!(g[..], [capture::Geometry::Triangles(1, _)]));
    assert_eq!(b.postbuild[0].info_type, 2);
    assert!(matches!(&c.0[1],Op::Build(b) if b.inputs.is_err()));
}

#[test]
fn rehydrate_uses_header_address_even_when_extra_is_zero() {
    let src = [0u32; 16];
    let mut extra = [0u32; 16];
    extra[0] = 0x0100_0006;
    let mut d = guest(0x100, 0, 0, 0).desc;
    d.source_va = src.as_ptr() as u64;
    for (p, want) in [
        (None, Some(0)),
        (Some(extra.as_ptr() as u64), Some(extra[0])),
        (Some(0), None),
    ] {
        let mut b = capture_rehydrate(
            1,
            &d,
            p.as_ref().map_or(std::ptr::null(), std::ptr::from_ref),
        );
        b.rehydrated(0x80070057u32 as i32);
        let Op::Build(op) = &b.0[0] else { panic!() };
        let r = op.rehydrate.as_ref().unwrap();
        assert_eq!(op.desc.source_va, d.source_va);
        assert_eq!(r.extra, p);
        assert_eq!(r.header.as_ref().ok().map(|h| h[0]), want);
        assert_eq!(r.hr, 0x80070057u32 as i32);
    }
}

#[test]
fn phase_two_budget_uses_the_smaller_ceiling() {
    let gib = 1 << 30;
    assert_eq!(memory::limit(12800 << 20, 16 * gib), 12800 << 20);
    assert_eq!(memory::limit(12800 << 20, 8 * gib), 7 * gib);
    assert_eq!(
        memory::limit(12800 << 20, 12 * gib),
        12 * gib - (12 * gib) / 10
    );
    assert_eq!(memory::limit(12800 << 20, gib / 2), 0);
}

fn send(side: &Sidecar, captured: Captured, input: &ID3D12Resource) {
    let work = captured
        .0
        .into_iter()
        .map(|o| Work::bind(o, |_, _| Ok((input.clone(), 0))))
        .collect();
    side.execute(work);
}
fn finish(side: &Sidecar) {
    let mut e = side.lock();
    let n = e.next - 1;
    e.wait(n).unwrap();
}
fn instance(blas: u64) -> [u8; 64] {
    let mut a = [0; 64];
    for i in [0, 5, 10] {
        a[4 * i..4 * i + 4].copy_from_slice(&1.0f32.to_le_bytes());
    }
    a[48..52].copy_from_slice(&0xd5123456u32.to_le_bytes());
    a[52..56].copy_from_slice(&0x01765432u32.to_le_bytes());
    a[56..].copy_from_slice(&blas.to_le_bytes());
    a
}

#[test]
fn dxr_ada_debug_layer() {
    // D3D12's debug switch is process-wide and must precede device creation,
    // including under an unfiltered parallel cargo test run.
    const NAME: &str = "dxr::tests::dxr_ada_debug_layer";
    if std::env::var_os("VICTORY_TEST_ISOLATED_DXR").is_none() {
        let status = std::process::Command::new(std::env::current_exe().unwrap())
            .args(["--exact", NAME, "--nocapture"])
            .env("VICTORY_TEST_ISOLATED_DXR", "1")
            .status()
            .unwrap();
        assert!(status.success(), "isolated Ada gate failed");
        return;
    }
    victory_shader::emit::host::enable_debug_layer().expect("Windows Graphics Tools required");
    let name = "NVIDIA GeForce RTX 4070 Ti SUPER";
    let h = victory_shader::emit::host::Host::on_adapter(name).unwrap();
    assert_eq!(h.adapter_name().as_deref(), Some(name));
    let _info = h
        .device
        .cast::<ID3D12InfoQueue>()
        .expect("debug layer required");
    let (queue, adapter) = unsafe {
        let factory: IDXGIFactory4 = CreateDXGIFactory2(DXGI_CREATE_FACTORY_FLAGS(0)).unwrap();
        (
            h.device
                .CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC {
                    Type: D3D12_COMMAND_LIST_TYPE_DIRECT,
                    ..Default::default()
                })
                .unwrap(),
            factory
                .EnumAdapterByLuid(h.device.GetAdapterLuid())
                .unwrap(),
        )
    };
    let side = Sidecar::new(&h.device, &queue, Some(&adapter)).unwrap();
    let geom = triangle();
    let mut blas = guest(0x1000, 1, 1, &raw const geom as u64);
    blas.desc.inputs.flags = 1;
    let vb: Vec<_> = [0f32, 0., 0., 1., 0., 0., 0., 1., 0.]
        .into_iter()
        .flat_map(f32::to_le_bytes)
        .collect();
    let vertex = commands::upload(&mut side.lock().mem, &vb).unwrap();
    send(&side, capture_build(1, &blas), &vertex);
    send(&side, Captured::copy(0x2000, 0x1000, 1), &vertex); // query still pending is a valid case
    let mut bytes = Vec::from(instance(0x2000));
    bytes.extend_from_slice(&instance(0)); // legal inactive instance, including a nonzero mask
    let input = commands::upload(&mut side.lock().mem, &bytes).unwrap();
    let tlas = guest(0x3000, 0, 2, 0x30_0000);
    send(&side, capture_build(1, &tlas), &input);
    // Read back the production patcher's *output* independently; byte 56..64
    // of the live instance is the only field allowed to change.
    let (out, expected_va) = {
        let e = side.lock();
        let f = e.pending.back().unwrap();
        (f.patched[0].clone(), e.entries[&0x2000].storage.gpu)
    };
    let patched = read_patched(&side, &out, bytes.len());
    let mut expected = bytes.clone();
    expected[56..64].copy_from_slice(&expected_va.to_le_bytes());
    assert_eq!(patched, expected);
    finish(&side);
    let top = side.srv(0x3000).expect("complete TLAS");
    let empty = side.srv(0).expect("built empty TLAS");
    let heap: ID3D12DescriptorHeap = unsafe {
        h.device
            .CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                NumDescriptors: 2,
                Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
                NodeMask: 0,
            })
            .unwrap()
    };
    let at = unsafe { heap.GetCPUDescriptorHandleForHeapStart() };
    crate::descriptor::host::tlas_srv(&h.device, &top, at);
    let step = unsafe {
        h.device
            .GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV)
    } as usize;
    crate::descriptor::host::tlas_srv(
        &h.device,
        &empty,
        D3D12_CPU_DESCRIPTOR_HANDLE { ptr: at.ptr + step },
    );
    {
        let e = side.lock();
        let s = &e.stats;
        assert_eq!(
            (
                s.input,
                s.built,
                s.failed,
                s.skipped,
                s.missing_blas,
                s.compacted,
                s.tlas
            ),
            (3, 3, 0, 0, 0, 1, 1)
        );
        eprintln!(
            "adapter={name} DXR>=1.1 dynamic_blas={} compacted={} tlas={} missing_blas=0 compacted_size={}B",
            s.dynamic,
            s.compacted,
            s.tlas,
            e.entries[&0x1000].compact_size.load(Relaxed)
        );
    }
    // Freeze actual GPU versions at this point; pre-existing BLAS/TLAS use DXR
    // serialization, including an address map for TLAS relocation.
    let dir = tempfile::tempdir().unwrap();
    let snapshot: Vec<_> = capture_build(1, &tlas)
        .0
        .into_iter()
        .map(|o| Work::bind(o, |_, _| Ok((input.clone(), 0))))
        .collect();
    packet::save(&mut side.lock(), &snapshot, dir.path(), 7, 2).unwrap();
    let mut files = 0;
    for file in std::fs::read_dir(dir.path().join("native-rt")).unwrap() {
        let data = std::fs::read(file.unwrap().path()).unwrap();
        let (body, digest) = data.split_at(data.len() - 32);
        assert_eq!(blake3::hash(body).as_bytes(), digest);
        if body.starts_with(b"VRTBATCH") {
            assert!(body.ends_with(&bytes));
        } else {
            assert_eq!(&body[..8], b"VRTSEEDS");
            assert_eq!(u64::from_le_bytes(body[16..24].try_into().unwrap()), 4);
        }
        files += 1;
    }
    assert_eq!(files, 2);
    assert_trace(&h, &side, &top, 0);
    // Rehydrate two real geometries in order. The first triangle is outside the
    // ray. Hardware must report geometry index 1 for the second triangle.
    let mut second = geom;
    unsafe {
        second.data.triangles.vertex_buffer_va += 0x10000;
    }
    let geoms = [geom, second];
    let header = [0x0100_0000u32; 16];
    let mut static_desc = guest(0x8000, 1, 2, geoms.as_ptr() as u64).desc;
    static_desc.source_va = header.as_ptr() as u64;
    let shifted: Vec<u8> = [4f32, 0., 0., 5., 0., 0., 4., 1., 0.]
        .into_iter()
        .flat_map(f32::to_le_bytes)
        .chain(vb.iter().copied())
        .collect();
    let static_vb = commands::upload(&mut side.lock().mem, &shifted).unwrap();
    let ops = capture_rehydrate(1, &static_desc, std::ptr::null())
        .0
        .into_iter()
        .map(|op| {
            Work::bind(op, |va, _| {
                Ok((static_vb.clone(), if va == 0x10_0000 { 0 } else { 36 }))
            })
        })
        .collect();
    side.execute(ops);
    let si = commands::upload(&mut side.lock().mem, &instance(0x8000)).unwrap();
    send(
        &side,
        capture_build(1, &guest(0x9000, 0, 1, 0x90_0000)),
        &si,
    );
    finish(&side);
    assert_trace(&h, &side, &side.srv(0x9000).unwrap(), 1);
    assert_eq!(side.lock().stats.static_blas, 1);
    eprintln!(
        "Ada inline RayQuery dynamic GeometryIndex=0, rehydrated GeometryIndex=1, mask and instance attributes verified"
    );
    // A legal inactive float triangle (NaN x) cannot replace an active one
    // by refit. A full build must invalidate the TLAS without harming its old SRV.
    blas.desc.dest_va = 0x2000;
    blas.desc.source_va = 0x2000;
    blas.desc.inputs.flags = 0x21;
    let mut inactive = vb.clone();
    inactive[..4].copy_from_slice(&f32::NAN.to_le_bytes());
    let changed = commands::upload(&mut side.lock().mem, &inactive).unwrap();
    send(&side, capture_build(1, &blas), &changed);
    finish(&side);
    assert!(side.srv(0x3000).is_none());
    assert_eq!(side.lock().stats.refits, 0);
    assert_trace(&h, &side, &top, 0); // this older SRV still owns the previous BLAS
    assert!(side.srv(0x9000).is_some());

    let mut norm_geom = triangle();
    norm_geom.data.triangles.vertex_format = DXGI_FORMAT_R16G16B16A16_SNORM.0 as u32;
    norm_geom.data.triangles.vertex_buffer_stride = 8;
    let mut norm = guest(0xb000, 1, 1, &raw const norm_geom as u64);
    norm.desc.inputs.flags = 1;
    let vertices: Vec<_> = [0i16, 0, 0, 0, 32767, 0, 0, 0, 0, 32767, 0, 0]
        .into_iter()
        .flat_map(i16::to_le_bytes)
        .collect();
    let fixed = commands::upload(&mut side.lock().mem, &vertices).unwrap();
    send(&side, capture_build(1, &norm), &fixed);
    norm.desc.source_va = norm.desc.dest_va;
    norm.desc.inputs.flags = 0x21;
    send(&side, capture_build(1, &norm), &fixed);
    finish(&side);
    assert_eq!(side.lock().stats.refits, 1);
    eprintln!(
        "Ada refit: SNORM updated; float active-to-inactive full rebuild, old TLAS still traces"
    );

    let missing = commands::upload(&mut side.lock().mem, &instance(0x5000)).unwrap();
    let one = guest(0x4000, 0, 1, 0x40_0000);
    send(&side, capture_build(1, &one), &missing);
    finish(&side);
    assert!(side.srv(0x4000).is_none());
    send(&side, Captured::copy(0x3000, 0x2000, 0), &vertex);
    finish(&side);
    let e = side.lock();
    assert_eq!(
        (e.stats.missing_blas, e.stats.failed, e.stats.skipped),
        (1, 1, 1)
    );
    assert_eq!(
        e.stats.input,
        e.stats.built + e.stats.skipped + e.stats.failed
    );
    drop(e);
    drop(top);
    drop(empty);
    drop(heap);
    drop(side);
    unsafe { h.device.GetDeviceRemovedReason() }.unwrap();
    h.check_debug_errors().unwrap();
    eprintln!("D3D12 InfoQueue Error/Corruption=0");
}

fn assert_trace(
    h: &victory_shader::emit::host::Host,
    side: &Sidecar,
    top: &ID3D12Resource,
    geometry: u32,
) {
    use victory_shader::emit::rt_replay::{HIT_WORDS, QueryMode, ray_query_shader};
    let shader = ray_query_shader(1, QueryMode::Closest).unwrap();
    let (root, pso) = h.compute_pipeline(&shader).unwrap();
    let mut bindings: Vec<_> = shader
        .root_srvs
        .iter()
        .map(|&s| (s, false))
        .chain(shader.root_uavs.iter().map(|&s| (s, true)))
        .collect();
    bindings.sort_unstable();
    let mut words = vec![2u32, 0, 0, 0];
    for mask in [1, 2] {
        words.extend([0.25f32, 0.25, -1., 0., 0., 0., 1., 10.].map(f32::to_bits));
        words.extend([mask, 0, 0, 0]);
    }
    let rays: Vec<_> = words.into_iter().flat_map(u32::to_le_bytes).collect();
    let mut e = side.lock();
    let rays = commands::upload(&mut e.mem, &rays).unwrap();
    let bytes = (2 * HIT_WORDS * 4) as u64;
    let out = e
        .mem
        .buffer(
            bytes,
            D3D12_HEAP_TYPE_DEFAULT,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
        )
        .unwrap();
    let read = e
        .mem
        .buffer(
            bytes,
            D3D12_HEAP_TYPE_READBACK,
            D3D12_RESOURCE_STATE_COPY_DEST,
            D3D12_RESOURCE_FLAG_NONE,
        )
        .unwrap();
    unsafe {
        let alloc: ID3D12CommandAllocator = h
            .device
            .CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)
            .unwrap();
        let list: ID3D12GraphicsCommandList4 = h
            .device
            .CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc, None)
            .unwrap();
        let heaps = [
            D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
            D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER,
        ]
        .map(|ty| {
            Some(
                h.device
                    .CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                        Type: ty,
                        NumDescriptors: 1,
                        Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
                        NodeMask: 0,
                    })
                    .unwrap(),
            )
        });
        list.SetDescriptorHeaps(&heaps);
        list.SetPipelineState(&pso);
        list.SetComputeRootSignature(&root);
        list.SetComputeRoot32BitConstant(0, 0, 0);
        for (i, &binding) in bindings.iter().enumerate() {
            let i = i as u32 + 1;
            match binding {
                (0, true) => list.SetComputeRootUnorderedAccessView(i, out.GetGPUVirtualAddress()),
                (0, false) => list.SetComputeRootShaderResourceView(i, rays.GetGPUVirtualAddress()),
                (victory_shader::emit::SLOT_TLAS_0, false) => {
                    list.SetComputeRootShaderResourceView(i, top.GetGPUVirtualAddress())
                }
                _ => panic!("unexpected oracle binding"),
            }
        }
        list.Dispatch(1, 1, 1);
        list.ResourceBarrier(&[crate::host::transition(
            &out,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_COPY_SOURCE,
        )]);
        list.CopyBufferRegion(&read, 0, &out, 0, bytes);
        list.Close().unwrap();
        e.queue.ExecuteCommandLists(&[Some(list.cast().unwrap())]);
        let n = e.next;
        e.next += 1;
        e.queue.Signal(&e.fence, n).unwrap();
        e.wait(n).unwrap();
    }
    let data = commands::read_bytes(&read, bytes as usize).unwrap();
    let hits: Vec<_> = data
        .chunks_exact(4)
        .map(|w| u32::from_le_bytes(w.try_into().unwrap()))
        .collect();
    assert_eq!(
        &hits[..9],
        &[
            1,
            1f32.to_bits(),
            0.25f32.to_bits(),
            0.25f32.to_bits(),
            0x123456,
            0,
            0x765432,
            geometry,
            0
        ]
    );
    assert!(
        hits[HIT_WORDS..].iter().all(|&w| w == 0),
        "excluded instance mask must miss"
    );
}

fn read_patched(side: &Sidecar, src: &ID3D12Resource, len: usize) -> Vec<u8> {
    let mut e = side.lock();
    let dst = e
        .mem
        .buffer(
            len as u64,
            D3D12_HEAP_TYPE_READBACK,
            D3D12_RESOURCE_STATE_COPY_DEST,
            D3D12_RESOURCE_FLAG_NONE,
        )
        .unwrap();
    // Separate Execute on the same queue: the buffer has decayed to COMMON.
    unsafe {
        let alloc: ID3D12CommandAllocator = e
            .mem
            .dev
            .CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)
            .unwrap();
        let list: ID3D12GraphicsCommandList = e
            .mem
            .dev
            .CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc, None)
            .unwrap();
        list.CopyBufferRegion(&dst, 0, src, 0, len as u64);
        list.Close().unwrap();
        e.queue.ExecuteCommandLists(&[Some(list.cast().unwrap())]);
        let n = e.next;
        e.next += 1;
        e.queue.Signal(&e.fence, n).unwrap();
        e.wait(n).unwrap();
    }
    commands::read_bytes(&dst, len).unwrap()
}
