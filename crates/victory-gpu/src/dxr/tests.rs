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
    let require = std::env::var_os("VICTORY_REQUIRE_DXR_TESTS").is_some_and(|v| v == "1");
    if let Err(e) = victory_shader::emit::host::enable_debug_layer() {
        if require {
            panic!("Windows Graphics Tools required: {e}");
        }
        eprintln!("skipping {NAME}: Windows Graphics Tools debug layer unavailable ({e})");
        return;
    }
    let h = match std::env::var("VICTORY_D3D12_ADAPTER") {
        Ok(name) if !name.trim().is_empty() => victory_shader::emit::host::Host::on_adapter(name.trim()),
        _ => victory_shader::emit::host::Host::new(false),
    };
    let h = match h {
        Ok(h) => h,
        Err(e) => {
            if require {
                panic!("DXR 1.1 hardware adapter required: {e}");
            }
            eprintln!("skipping {NAME}: no SM 6.6 hardware adapter ({e})");
            return;
        }
    };
    let mut o5 = D3D12_FEATURE_DATA_D3D12_OPTIONS5::default();
    let tier_ok = unsafe {
        h.device
            .CheckFeatureSupport(
                D3D12_FEATURE_D3D12_OPTIONS5,
                (&raw mut o5).cast(),
                size_of_val(&o5) as u32,
            )
            .is_ok()
    } && o5.RaytracingTier.0 >= D3D12_RAYTRACING_TIER_1_1.0;
    if !tier_ok {
        if require {
            panic!("adapter lacks DXR Tier 1.1");
        }
        eprintln!("skipping {NAME}: hardware adapter lacks DXR Tier 1.1");
        return;
    }
    let name = h.adapter_name().unwrap_or_else(|| "unknown".to_owned());
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
    // by refit. In strict mode (`VICTORY_DXR_STRICT_TLAS=1`), a full build
    // invalidates TLAS 0x3000 without harming its held `top` SRV; in live mode
    // (`!strict_tlas()`), the full rebuild writes to a fresh immutable slab slice
    // so TLAS 0x3000 stays live and traces its pinned previous BLAS.
    let old_2000_gpu = side.lock().entries[&0x2000].storage.gpu;
    blas.desc.dest_va = 0x2000;
    blas.desc.source_va = 0x2000;
    blas.desc.inputs.flags = 0x21;
    let mut inactive = vb.clone();
    inactive[..4].copy_from_slice(&f32::NAN.to_le_bytes());
    let changed = commands::upload(&mut side.lock().mem, &inactive).unwrap();
    send(&side, capture_build(1, &blas), &changed);
    finish(&side);
    if strict_tlas() {
        assert!(side.srv(0x3000).is_none());
    } else {
        assert_eq!(side.srv(0x3000), Some(top.clone()));
        // Verify P2 #3: diagnostic seed capture includes the older pinned BLAS 0x2000
        // generation still referenced by live TLAS 0x3000 so every serialized TLAS
        // relocation pointer resolves in `VRTSEEDS`.
        let new_2000_gpu = side.lock().entries[&0x2000].storage.gpu;
        assert_ne!(new_2000_gpu, old_2000_gpu);
        let seed_dir = tempfile::tempdir().unwrap();
        packet::save(&mut side.lock(), &[], seed_dir.path(), 8, 0).unwrap();
        let raw = std::fs::read(seed_dir.path().join("native-rt").join("initial.vrt")).unwrap();
        let (body, digest) = raw.split_at(raw.len() - 32);
        assert_eq!(blake3::hash(body).as_bytes(), digest);
        assert_eq!(&body[..8], b"VRTSEEDS");
        let mut cursor = 16usize;
        let read_u64 = |pos: &mut usize| -> u64 {
            let v = u64::from_le_bytes(body[*pos..*pos + 8].try_into().unwrap());
            *pos += 8;
            v
        };
        let count = read_u64(&mut cursor) as usize;
        let mut blas_vas = std::collections::HashSet::new();
        let mut blas_2000_count = 0usize;
        let mut tlas_3000_blob = None;
        for _ in 0..count {
            let key = read_u64(&mut cursor);
            let _serial = read_u64(&mut cursor);
            let is_top = read_u64(&mut cursor) != 0;
            let gpu_va = read_u64(&mut cursor);
            let blob_len = read_u64(&mut cursor) as usize;
            let blob = &body[cursor..cursor + blob_len];
            cursor += blob_len;
            if is_top {
                if key == 0x3000 {
                    tlas_3000_blob = Some(blob);
                }
            } else {
                blas_vas.insert(gpu_va);
                if key == 0x2000 {
                    blas_2000_count += 1;
                }
            }
        }
        assert_eq!(cursor, body.len());
        assert_eq!(
            blas_2000_count, 2,
            "VRTSEEDS must serialize both the older pinned and newer BLAS 0x2000 generations"
        );
        assert!(blas_vas.contains(&old_2000_gpu));
        assert!(blas_vas.contains(&new_2000_gpu));
        let tlas_blob = tlas_3000_blob.expect("live TLAS 0x3000 serialized in VRTSEEDS");
        let hdr_size = size_of::<D3D12_SERIALIZED_RAYTRACING_ACCELERATION_STRUCTURE_HEADER>();
        assert!(tlas_blob.len() >= hdr_size);
        let num_ptrs = u64::from_le_bytes(tlas_blob[48..56].try_into().unwrap()) as usize;
        assert!(num_ptrs >= 1, "TLAS 0x3000 must serialize BLAS relocation pointers");
        let mut found_old = false;
        for i in 0..num_ptrs {
            let off = hdr_size + i * 8;
            let ptr = u64::from_le_bytes(tlas_blob[off..off + 8].try_into().unwrap());
            if ptr != 0 {
                assert!(
                    blas_vas.contains(&ptr),
                    "TLAS relocation pointer {ptr:#x} missing from VRTSEEDS BLAS set"
                );
                if ptr == old_2000_gpu {
                    found_old = true;
                }
            }
        }
        assert!(
            found_old,
            "TLAS 0x3000 relocation table must resolve old pinned BLAS 0x2000 VA"
        );
    }
    assert_eq!(side.lock().stats.refits, 0);
    assert_trace(&h, &side, &top, 0); // this SRV still pins the previous BLAS
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
    {
        let e = side.lock();
        assert_eq!(
            (e.stats.missing_blas, e.stats.failed, e.stats.skipped),
            (1, 1, 1)
        );
        assert_eq!(
            e.stats.input,
            e.stats.built + e.stats.skipped + e.stats.failed
        );
    }

    // Track A hardware verification on Ada under D3D12 Debug Layer:
    // 1. Canonicalization of Xbox typeless index formats (`R16_TYPELESS` = 53,
    //    `R32_TYPELESS` = 39) + `ScratchBatcher` multi-BLAS batching in one submit.
    // 2. Self-`CopyAs` (`dst == src`, mode=1) allocates a fresh non-overlapping slice.
    // 3. Steady-state in-place TLAS reuse (in live mode) preserves the exact same
    //    `ID3D12Resource`, replaces old BLAS pins after fence completion, preserves
    //    later queued rebuilds when an earlier rebuild fails, and retains candidate
    //    BLAS pins on failed latest in-place builds until resource destruction.
    let mut ib_vb = vb.clone(); // offset 0..36: 3 float3 vertices
    ib_vb.resize(64, 0);
    for idx in [0u16, 1, 2] {
        ib_vb.extend_from_slice(&idx.to_le_bytes()); // offset 64..70: R16 indices
    }
    ib_vb.resize(80, 0);
    for idx in [0u32, 1, 2] {
        ib_vb.extend_from_slice(&idx.to_le_bytes()); // offset 80..92: R32 indices
    }
    let idx_buf = commands::upload(&mut side.lock().mem, &ib_vb).unwrap();
    let mut g_r16 = triangle();
    g_r16.data.triangles.index_format = DXGI_FORMAT_R16_TYPELESS.0 as u32;
    g_r16.data.triangles.index_count = 3;
    g_r16.data.triangles.index_buffer = 0x20_0000;
    let mut g_r32 = triangle();
    g_r32.data.triangles.index_format = DXGI_FORMAT_R32_TYPELESS.0 as u32;
    g_r32.data.triangles.index_count = 3;
    g_r32.data.triangles.index_buffer = 0x20_0100;
    let typeless_batch = [
        guest(0xc000, 1, 1, &raw const g_r16 as u64),
        guest(0xd000, 1, 1, &raw const g_r32 as u64),
    ];
    let ops: Arc<[Work]> = capture_build(2, typeless_batch.as_ptr())
        .0
        .into_iter()
        .map(|op| {
            Work::bind(op, |va, _| {
                let off = match va {
                    0x10_0000 => 0,
                    0x20_0000 => 64,
                    0x20_0100 => 80,
                    _ => unreachable!(),
                };
                Ok((idx_buf.clone(), off))
            })
        })
        .collect();
    side.execute(ops);
    finish(&side);
    assert!(
        side.lock().stats.blas_batched >= 1,
        "ScratchBatcher must batch independent BLASes in one submit"
    );

    // Build TLAS 0xe000 referencing R16_TYPELESS BLAS 0xc000.
    let inst_c = commands::upload(&mut side.lock().mem, &instance(0xc000)).unwrap();
    send(
        &side,
        capture_build(1, &guest(0xe000, 0, 1, 0xe0_0000)),
        &inst_c,
    );
    finish(&side);
    let stable_tlas = side.srv(0xe000).expect("R16_TYPELESS TLAS");
    assert_trace(&h, &side, &stable_tlas, 0);

    // Self-CopyAs (dst == src == 0xd000) must allocate a distinct non-overlapping slice.
    let (old_d_gpu, old_d_bytes) = {
        let e = side.lock();
        let entry = &e.entries[&0xd000];
        (entry.storage.gpu, entry.storage.bytes)
    };
    send(&side, Captured::copy(0xd000, 0xd000, 1), &vertex);
    finish(&side);
    {
        let e = side.lock();
        let entry = &e.entries[&0xd000];
        let new_gpu = entry.storage.gpu;
        let new_bytes = entry.storage.bytes;
        assert_ne!(new_gpu, old_d_gpu, "self-CopyAs must not alias source VA");
        assert!(
            new_gpu + new_bytes <= old_d_gpu || old_d_gpu + old_d_bytes <= new_gpu,
            "self-CopyAs must allocate a non-overlapping slice"
        );
    }

    // Rebuild TLAS 0xe000 referencing self-compacted R32_TYPELESS BLAS 0xd000.
    let c_entry = side.lock().entries[&0xc000].clone();
    let d_entry = side.lock().entries[&0xd000].clone();
    let inst_d = commands::upload(&mut side.lock().mem, &instance(0xd000)).unwrap();
    send(
        &side,
        capture_build(1, &guest(0xe000, 0, 1, 0xe0_0000)),
        &inst_d,
    );
    finish(&side);
    if strict_tlas() {
        let strict_rebuilt = side.srv(0xe000).expect("strict TLAS rebuild");
        assert_ne!(
            strict_rebuilt, stable_tlas,
            "strict mode must allocate a fresh TLAS resource"
        );
        assert_eq!(side.lock().stats.tlas_reused, 0);
        assert_trace(&h, &side, &strict_rebuilt, 0);
        drop(strict_rebuilt);
        drop(c_entry);
        drop(d_entry);
        drop(stable_tlas);
    } else {
        assert_eq!(
            side.srv(0xe000),
            Some(stable_tlas.clone()),
            "steady-state TLAS rebuild must reuse the same ID3D12Resource in-place"
        );
        assert_trace(&h, &side, &stable_tlas, 0);
        {
            let e = side.lock();
            assert!(e.stats.tlas_reused >= 1);
            let pins = e.entries[&0xe000]
                .pins
                .as_ref()
                .unwrap()
                .lock()
                .unwrap();
            assert!(
                !pins.iter().any(|(_, p)| Arc::ptr_eq(p, &c_entry)),
                "old BLAS 0xc000 pin must be released after in-place TLAS fence completes"
            );
            assert!(
                pins.iter().any(|(_, p)| Arc::ptr_eq(p, &d_entry)),
                "new BLAS 0xd000 pin must be held by the reused TLAS PinSlot"
            );
        }

        // P1 #2 regression 1 (same batch): failed in-place TLAS f1 (all missing)
        // followed by valid in-place TLAS f2 (0xc000) in ONE batch must keep 0xe000
        // present, ready, and traceable.
        let both_inst_bytes: Vec<u8> = instance(0x5000)
            .into_iter()
            .chain(instance(0xc000))
            .collect();
        let both_inst = commands::upload(&mut side.lock().mem, &both_inst_bytes).unwrap();
        let pair_batch = [
            guest(0xe000, 0, 1, 0xe1_0000),
            guest(0xe000, 0, 1, 0xe1_0040),
        ];
        let pair_ops: Arc<[Work]> = capture_build(2, pair_batch.as_ptr())
            .0
            .into_iter()
            .map(|op| {
                Work::bind(op, |va, _| {
                    let off = if va == 0xe1_0000 { 0 } else { 64 };
                    Ok((both_inst.clone(), off))
                })
            })
            .collect();
        side.execute(pair_ops);
        finish(&side);
        assert_eq!(
            side.srv(0xe000),
            Some(stable_tlas.clone()),
            "earlier failed in-place rebuild in same batch must not evict later valid rebuild"
        );
        assert_trace(&h, &side, &stable_tlas, 0);
        {
            let e = side.lock();
            let pins = e.entries[&0xe000]
                .pins
                .as_ref()
                .unwrap()
                .lock()
                .unwrap();
            assert!(pins.iter().any(|(_, p)| Arc::ptr_eq(p, &c_entry)));
        }

        // P1 #2 regression 2 (separate queued flights, deterministic): stall DIRECT
        // behind an unsignaled fence so f1 (all missing) and f2 (valid 0xd000) are
        // guaranteed to both be pending simultaneously before collection.
        let gate: ID3D12Fence = unsafe {
            h.device
                .CreateFence(0, D3D12_FENCE_FLAG_NONE)
                .expect("gate fence")
        };
        unsafe { queue.Wait(&gate, 1).expect("queue wait on gate") };
        send(
            &side,
            capture_build(1, &guest(0xe000, 0, 1, 0xe2_0000)),
            &missing,
        );
        send(
            &side,
            capture_build(1, &guest(0xe000, 0, 1, 0xe2_0040)),
            &inst_d,
        );
        assert!(
            side.lock().pending.len() >= 2,
            "both f1 and f2 must be pending simultaneously before collection"
        );
        unsafe { gate.Signal(1).expect("signal gate fence") };
        finish(&side);
        assert_eq!(
            side.srv(0xe000),
            Some(stable_tlas.clone()),
            "earlier failed in-place flight must not evict later queued valid flight"
        );
        assert_trace(&h, &side, &stable_tlas, 0);
        {
            let e = side.lock();
            let pins = e.entries[&0xe000]
                .pins
                .as_ref()
                .unwrap()
                .lock()
                .unwrap();
            assert!(pins.iter().any(|(_, p)| Arc::ptr_eq(p, &d_entry)));
            assert!(
                !pins.iter().any(|(_, p)| Arc::ptr_eq(p, &c_entry)),
                "c_entry must not be in PinSlot prior to the failed-latest retention test"
            );
        }

        // P1 #1 regression: verify a brand-new candidate BLAS (`c_entry`, NOT in
        // PinSlot before the failed build) is merged into PinSlot by `retain_table_pins`
        // even when a later BLAS rebuild replaces `0xc000` and clears `self.table`
        // before `f_bad` is collected, stays alive while the external `stable_tlas`
        // COM resource is held, and releases once `stable_tlas` drops after
        // `finish_slot_readers()`.
        let c_weak = Arc::downgrade(&c_entry);
        drop(c_entry);
        drop(d_entry);
        let gate2: ID3D12Fence = unsafe {
            h.device
                .CreateFence(0, D3D12_FENCE_FLAG_NONE)
                .expect("gate2 fence")
        };
        unsafe { queue.Wait(&gate2, 1).expect("queue wait on gate2") };
        send(
            &side,
            capture_build(1, &guest(0xe000, 0, 1, 0xe3_0000)),
            &missing,
        );
        // Replace 0xc000 while f_bad is still pending on GPU (clears `self.table` too).
        let replace_c: Arc<[Work]> = capture_build(1, &guest(0xc000, 1, 1, &raw const g_r16 as u64))
            .0
            .into_iter()
            .map(|op| {
                Work::bind(op, |va, _| {
                    let off = if va == 0x10_0000 { 0 } else { 64 };
                    Ok((idx_buf.clone(), off))
                })
            })
            .collect();
        side.execute(replace_c);
        assert!(
            side.lock().pending.len() >= 2,
            "failed TLAS and replacement BLAS must both be pending before collection"
        );
        unsafe { gate2.Signal(1).expect("signal gate2 fence") };
        finish(&side);
        assert!(side.srv(0xe000).is_none());
        assert!(
            c_weak.strong_count() > 0,
            "brand-new candidate BLAS must be retained by PinSlot while external TLAS handle is held"
        );
        side.finish_slot_readers();
        drop(stable_tlas);
        assert_eq!(
            c_weak.strong_count(),
            0,
            "retained candidate BLAS must release once external TLAS handle drops"
        );
    }
    {
        let e = side.lock();
        assert_eq!(
            e.stats.input,
            e.stats.built + e.stats.skipped + e.stats.failed
        );
        assert!(!e.free_inst.is_empty(), "inst buffer must be returned to pool");
        assert!(
            !e.free_table_uploads.is_empty(),
            "table upload buffer must be returned to pool"
        );
    }
    drop(top);
    drop(empty);
    drop(heap);
    drop(side);
    unsafe { h.device.GetDeviceRemovedReason() }.unwrap();
    h.check_debug_errors().unwrap();
    eprintln!("D3D12 InfoQueue Error/Corruption=0");
}

#[test]
fn index_format_canonicalizes_typeless_and_validates_non_indexed() {
    let mut mock_addr = |va: u64, len: u64, align: u64| -> Checked<u64> {
        if va == 0 || !va.is_multiple_of(align) || len == 0 || va.checked_add(len).is_none() {
            return Err("invalid GPU input range");
        }
        Ok(0x1_0000_0000 + va)
    };
    let mut build_for = |index_format: DXGI_FORMAT,
                         index_buffer: u64,
                         index_count: u32,
                         rehydrate: bool|
     -> Checked<D3D12_RAYTRACING_GEOMETRY_TRIANGLES_DESC> {
        let mut t = unsafe { triangle().data.triangles };
        t.index_format = index_format.0 as u32;
        t.index_buffer = index_buffer;
        t.index_count = index_count;
        let b = capture::Build {
            desc: GuestBuildAsDesc {
                dest_va: 0x1000,
                inputs: GuestBuildAsInputs {
                    kind: 1,
                    flags: 0,
                    num_descs: 1,
                    descs_layout: 0,
                    ptr_or_va: 0,
                },
                source_va: 0,
                scratch_va: 0,
            },
            inputs: Ok(capture::Inputs::Blas(vec![capture::Geometry::Triangles(
                1, t,
            )])),
            postbuild: vec![],
            stream: vec![],
            rehydrate: rehydrate.then_some(capture::Rehydrate {
                extra: None,
                header: Ok([0x0100_0000; 16]),
                hr: 0,
            }),
        };
        match geometry::translate(&b, &mut mock_addr)? {
            Native::Blas(geoms) => Ok(unsafe { geoms[0].Anonymous.Triangles }),
            _ => panic!("expected BLAS"),
        }
    };

    // R16_TYPELESS (53) -> canonical R16_UINT (57), both dynamic and rehydrated static.
    for rehydrate in [false, true] {
        let r16 = build_for(DXGI_FORMAT_R16_TYPELESS, 0x20_0002, 3, rehydrate).unwrap();
        assert_eq!(r16.IndexFormat, DXGI_FORMAT_R16_UINT);
        assert_eq!(r16.IndexCount, 3);
        assert_eq!(r16.IndexBuffer, 0x1_0020_0002);
    }

    // R32_TYPELESS (39) -> canonical R32_UINT (42).
    let r32 = build_for(DXGI_FORMAT_R32_TYPELESS, 0x20_0004, 3, false).unwrap();
    assert_eq!(r32.IndexFormat, DXGI_FORMAT_R32_UINT);
    assert_eq!(r32.IndexCount, 3);
    assert_eq!(r32.IndexBuffer, 0x1_0020_0004);

    // Non-indexed cases normalize to DXGI_FORMAT_UNKNOWN with IndexBuffer = 0.
    let stale_unknown = build_for(DXGI_FORMAT_UNKNOWN, 0xdead_beef, 0, false).unwrap();
    assert_eq!(stale_unknown.IndexFormat, DXGI_FORMAT_UNKNOWN);
    assert_eq!(stale_unknown.IndexBuffer, 0);
    assert_eq!(stale_unknown.IndexCount, 0);

    let zero_typeless = build_for(DXGI_FORMAT_R16_TYPELESS, 0, 0, false).unwrap();
    assert_eq!(zero_typeless.IndexFormat, DXGI_FORMAT_UNKNOWN);
    assert_eq!(zero_typeless.IndexBuffer, 0);
    assert_eq!(zero_typeless.IndexCount, 0);

    // Malformed cases remain rejected.
    assert_eq!(
        build_for(DXGI_FORMAT_R16_TYPELESS, 0x20_0000, 0, false).unwrap_err(),
        "invalid index format or buffer"
    );
    assert_eq!(
        build_for(DXGI_FORMAT_R16_TYPELESS, 0, 3, false).unwrap_err(),
        "invalid GPU input range"
    );
    assert_eq!(
        build_for(DXGI_FORMAT_R8_UINT, 0x20_0000, 3, false).unwrap_err(),
        "invalid index format or buffer"
    );
}

#[test]
fn scratch_batcher_bumps_aligns_and_flushes_on_hazards() {
    let mut b = ScratchBatcher::default();
    assert!(!b.has_pending());

    // Consecutive independent BLAS builds bump-allocate on 256B boundaries without barriers.
    assert_eq!(b.alloc(1, 2048, 100, 0x1000, 0), (false, 0));
    assert_eq!(b.alloc(1, 2048, 300, 0x2000, 0), (false, 256));
    assert_eq!(b.alloc(1, 2048, 512, 0x3000, 0), (false, 768));
    assert!(b.has_pending());

    // Reading an unbarriered dest as a refit source forces a UAV barrier and resets cursor.
    assert_eq!(b.alloc(1, 2048, 256, 0x4000, 0x2000), (true, 0));

    // Overwriting an unbarriered dest forces a UAV barrier and resets cursor.
    assert_eq!(b.alloc(1, 2048, 256, 0x4000, 0), (true, 0));

    // Exhausting scratch capacity forces a UAV barrier and resets cursor.
    assert_eq!(b.alloc(1, 2048, 1536, 0x5000, 0), (false, 256));
    assert_eq!(b.alloc(1, 2048, 512, 0x6000, 0), (true, 0));

    // Growing to a new scratch buffer resets cursor to 0 without dropping hazard tracking.
    assert_eq!(b.alloc(2, 4096, 512, 0x7000, 0), (false, 0));
    assert_eq!(b.alloc(2, 4096, 256, 0x8000, 0x6000), (true, 0));

    b.reset();
    assert!(!b.has_pending());
}

#[test]
fn cull_mask_and_refit_period_parsing() {
    use commands::{CULL_MASK_TAG_MODE, parse_cull_mask_cfg};
    assert_eq!(parse_cull_mask_cfg(None), 0);
    assert_eq!(parse_cull_mask_cfg(Some("")), 0);
    assert_eq!(parse_cull_mask_cfg(Some("0")), 0);
    assert_eq!(parse_cull_mask_cfg(Some("off")), 0);
    assert_eq!(parse_cull_mask_cfg(Some("false")), 0);
    assert_eq!(parse_cull_mask_cfg(Some("1")), 0x80);
    assert_eq!(parse_cull_mask_cfg(Some("on")), 0x80);
    assert_eq!(parse_cull_mask_cfg(Some("0x80")), 0x80);
    assert_eq!(parse_cull_mask_cfg(Some("64")), 0x40);
    assert_eq!(parse_cull_mask_cfg(Some("tag:0x80")), CULL_MASK_TAG_MODE | 0x80);
    assert_eq!(parse_cull_mask_cfg(Some("TAG:1")), CULL_MASK_TAG_MODE | 0x80);
    assert_eq!(parse_cull_mask_cfg(Some("tag:0")), 0);
    assert_eq!(parse_cull_mask_cfg(Some("256")), 0);
    assert_eq!(parse_cull_mask_cfg(Some("bogus")), 0);

    assert_eq!(parse_refit_period(None), DEFAULT_REFIT_PERIOD);
    assert_eq!(parse_refit_period(Some("0")), 0);
    assert_eq!(parse_refit_period(Some("128")), 128);
    assert_eq!(parse_refit_period(Some("nope")), DEFAULT_REFIT_PERIOD);
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
