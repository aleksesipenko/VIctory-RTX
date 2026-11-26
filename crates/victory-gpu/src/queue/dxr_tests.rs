use super::*;
use crate::dxr::{Captured, GuestBatchedDesc2, capture_build};
use windows::Win32::Graphics::Direct3D12::{
    D3D12_FEATURE_D3D12_OPTIONS5, D3D12_FEATURE_DATA_D3D12_OPTIONS5, D3D12_RAYTRACING_TIER_1_1,
    ID3D12GraphicsCommandList,
};
use windows::core::{IUnknown, Interface};

#[test]
fn dxr_thin_submission_order_and_abandonment() {
    const NAME: &str = "queue::dxr_tests::dxr_thin_submission_order_and_abandonment";
    if std::env::var_os("VICTORY_TEST_DXR_THIN").is_none() {
        let mut command = std::process::Command::new(std::env::current_exe().unwrap());
        #[cfg(feature = "diagnostics")]
        let dir = tempfile::tempdir().unwrap();
        #[cfg(feature = "diagnostics")]
        command
            .env("VICTORY_RECORD", dir.path())
            .env("VICTORY_RECORD_AT", "f1")
            .env("VICTORY_RECORD_ALWAYS", "1");
        let ok = command
            .args(["--exact", NAME, "--nocapture"])
            .env("VICTORY_TEST_DXR_THIN", "1")
            .env("VICTORY_NATIVE_RT", "1")
            .env("VICTORY_THIN", "1")
            .env("VICTORY_RELAY", "2")
            .env_remove("VICTORY_D3D12_DEBUG")
            .status()
            .unwrap();
        assert!(ok.success());
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
    // Probe DXR 1.1 inside this isolated child before `crate::host::host()`
    // (which intentionally fails closed when `VICTORY_NATIVE_RT=1` lacks DXR 1.1).
    let probe = match std::env::var("VICTORY_D3D12_ADAPTER") {
        Ok(name) if !name.trim().is_empty() => {
            victory_shader::emit::host::Host::on_adapter(name.trim())
        }
        _ => victory_shader::emit::host::Host::new(false),
    };
    let tier_ok = probe.as_ref().is_ok_and(|h| {
        let mut o5 = D3D12_FEATURE_DATA_D3D12_OPTIONS5::default();
        // SAFETY: plain feature support query on a live device.
        let ok = unsafe {
            h.device
                .CheckFeatureSupport(
                    D3D12_FEATURE_D3D12_OPTIONS5,
                    (&raw mut o5).cast(),
                    size_of_val(&o5) as u32,
                )
                .is_ok()
        };
        ok && o5.RaytracingTier.0 >= D3D12_RAYTRACING_TIER_1_1.0
    });
    drop(probe);
    if !tier_ok {
        if require {
            panic!("DXR 1.1 hardware adapter required");
        }
        eprintln!("skipping {NAME}: no DXR Tier 1.1 hardware adapter");
        return;
    }
    let Some(host) = crate::host::host() else {
        if require {
            panic!("DXR 1.1 hardware adapter required");
        }
        eprintln!("skipping {NAME}: no D3D12 host device");
        return;
    };
    let Some(side) = host.dxr.as_ref() else {
        if require {
            panic!("DXR 1.1 sidecar required");
        }
        eprintln!("skipping {NAME}: host adapter lacks DXR Tier 1.1");
        return;
    };
    let debug = victory_shader::emit::host::Host {
        device: host.device.clone(),
    };
    assert!(crate::list::thin::on());
    let make = |kind| {
        let mut list = std::ptr::null();
        let allocator = std::ptr::NonNull::<u64>::dangling()
            .as_ptr()
            .cast::<Object>()
            .cast_const();
        let iid = std::ptr::from_ref(&ID3D12GraphicsCommandList::IID).cast();
        assert_eq!(
            crate::list::create_list(
                std::ptr::null(),
                0,
                kind,
                allocator,
                std::ptr::null(),
                iid,
                &mut list
            ),
            S_OK
        );
        unsafe { IUnknown::from_raw(list.cast_mut().cast()) }
    };
    let record_top = |list: &IUnknown, key| {
        // All-zero inputs are a legal zero-instance TLAS. No external vertex
        // fixture or game data is needed for the guest submission contract.
        let mut d: GuestBatchedDesc2 = unsafe { std::mem::zeroed() };
        d.desc.dest_va = key;
        unsafe { capture_build(1, &d).enqueue(list.as_raw()) };
        d.desc.dest_va = 0;
        std::hint::black_box(d);
    };
    // Public title API, including the optional placement pair following the heap description.
    let mut raw_heap = std::ptr::null();
    let desc = [0u32, 4, 1, 0, 0, 0, 0, 0];
    let iid = windows::Win32::Graphics::Direct3D12::ID3D12DescriptorHeap::IID;
    assert_eq!(
        crate::descriptor::create_heap(
            std::ptr::null(),
            desc.as_ptr().cast(),
            (&raw const iid).cast(),
            &mut raw_heap
        ),
        S_OK
    );
    let owned_heap = unsafe { IUnknown::from_raw(raw_heap.cast_mut().cast()) };
    let heap = crate::descriptor::heap(owned_heap.as_raw().cast());
    let mut words = [0u32; 10];
    words[1] =
        windows::Win32::Graphics::Direct3D12::D3D12_SRV_DIMENSION_RAYTRACING_ACCELERATION_STRUCTURE
            .0 as u32;
    words[4] = 0x1000;
    crate::descriptor::create_srv(std::ptr::null(), std::ptr::null(), &words, heap.start);
    let visible = heap.host().unwrap();
    assert_eq!(visible.kept_in(0), side.srv(0));

    let consumer = make(0);
    unsafe { Captured::copy(0x2000, 0x1000, 1).enqueue(consumer.as_raw()) };
    let discarded = make(0);
    record_top(&discarded, 0x4000);
    drop(discarded);
    let producer = make(2);
    record_top(&producer, 0x1000);
    assert!(side.srv(0x1000).is_none() && side.srv(0x2000).is_none());
    let mut queue = std::ptr::null();
    let desc = QueueDesc {
        kind: 0,
        priority: 0,
        flags: 0,
        nodes: 1,
    };
    assert_eq!(
        create(std::ptr::null(), &desc, &QUEUE_IIDS[0], &mut queue),
        S_OK
    );
    let queue = unsafe { IUnknown::from_raw(queue.cast_mut().cast()) };
    let lists: [This; 2] = [producer.as_raw().cast(), consumer.as_raw().cast()];
    execute(queue.as_raw().cast(), 2, lists.as_ptr());
    before_sync();
    crate::host::settle_for_fence();
    side.collect();
    assert!(side.srv(0x1000).is_some(), "submitted compute producer");
    assert!(
        side.srv(0x2000).is_some(),
        "copy recorded earlier, submitted later"
    );
    assert!(side.srv(0x4000).is_none(), "discarded before submission");
    assert_eq!(
        visible.kept_in(0),
        side.srv(0x1000),
        "slot written before build must advance at submit"
    );
    let first = side.srv(0x1000).unwrap();
    // The same closed list is executable again, on a later submission or twice
    // within one call; preserve the recorded boundaries until the guest resets it.
    let repeats: [This; 2] = [producer.as_raw().cast(); 2];
    execute(queue.as_raw().cast(), 2, repeats.as_ptr());
    before_sync();
    crate::host::settle_for_fence();
    side.collect();
    assert_ne!(side.srv(0x1000), Some(first));
    assert_eq!(visible.kept_in(0), side.srv(0x1000));
    #[cfg(feature = "diagnostics")]
    {
        use crate::replay::record;
        use windows::Win32::Graphics::{Direct3D12::*, Dxgi::Common::*};
        let mut target = None;
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
            Width: 8,
            Height: 8,
            DepthOrArraySize: 1,
            MipLevels: 1,
            Format: DXGI_FORMAT_R8G8B8A8_UNORM,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            ..Default::default()
        };
        let props = D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_DEFAULT,
            CreationNodeMask: 1,
            VisibleNodeMask: 1,
            ..Default::default()
        };
        unsafe {
            host.device.CreateCommittedResource(
                &props,
                D3D12_HEAP_FLAG_NONE,
                &desc,
                D3D12_RESOURCE_STATE_COMMON,
                None,
                &mut target,
            )
        }
        .unwrap();
        let target: ID3D12Resource = target.unwrap();
        for frame in 0..4 {
            let list = make(0);
            record_top(&list, 0xa000 + frame * 256);
            let command: This = list.as_raw().cast();
            execute(queue.as_raw().cast(), 1, &command);
            before_sync();
            crate::host::settle_for_fence();
            record::present(&target, D3D12_RESOURCE_STATE_COMMON);
        }
        let folder = record::finished().expect("diagnostic frame should complete");
        assert!(!folder.join("dxr-failed.txt").exists());
        assert!(folder.join("frame.bin").metadata().unwrap().len() > 0);
        let mut opening = false;
        let mut batch = false;
        let mut seeds = false;
        for f in std::fs::read_dir(folder.join("native-rt")).unwrap() {
            let path = f.unwrap().path();
            let name = path.file_name().unwrap().to_str().unwrap();
            let raw = std::fs::read(&path).unwrap();
            let (body, hash) = raw.split_at(raw.len() - 32);
            assert_eq!(blake3::hash(body).as_bytes(), hash);
            opening |= name.contains("-opening-") && body.starts_with(b"VRTBATCH");
            batch |= name.contains("-batch-") && body.starts_with(b"VRTBATCH");
            seeds |= name == "initial.vrt" && body.starts_with(b"VRTSEEDS");
        }
        assert!(opening && batch && seeds);
        eprintln!(
            "Synthetic diagnostic frame: frame.bin + DXR seeds, opening and ordered batch, BLAKE3 verified"
        );
    }
    debug.check_debug_errors().unwrap();
    eprintln!(
        "THIN=1 reversed recording order, guest ExecuteCommandLists, discarded list: InfoQueue Error/Corruption=0"
    );
}
