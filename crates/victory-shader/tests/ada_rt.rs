//! Acceptance on the named physical adapter. A separate test executable lets the
//! debug layer start before the first device is created in this process.
//!
//! `cargo test --locked -p victory-shader --test ada_rt -- --nocapture`
//! runs the L0 gate. Run the large packet with the debug layer using
//! `cargo test --locked -p victory-shader --release --test ada_rt l1_debug_packet -- --ignored --exact --nocapture`.
//! Substitute `l1_timestamps` for the release timing run. Both read the
//! binary packet named by `VICTORY_RT_PACKET` when set, and otherwise
//! construct, checksum and decode the same deterministic workload.
#![cfg(windows)]

mod rt {
    pub mod bench;
    pub mod gpu;
    pub mod oracle;
    pub mod scenes;
}

use rt::{
    gpu::{Gpu, Replay},
    oracle, scenes,
};
use victory_shader::{
    emit::{
        host::{Host, enable_debug_layer},
        rt_replay::QueryMode,
    },
    rt_packet::Packet,
};
use windows::Win32::Graphics::Direct3D12::*;
use windows::core::Interface;

fn ada(debug: bool) -> Host {
    if debug {
        enable_debug_layer().expect("install Windows Graphics Tools");
    }
    let name = "NVIDIA GeForce RTX 4070 Ti SUPER";
    let host = Host::on_adapter(name).expect("physical Ada adapter required");
    assert_eq!(host.adapter_name().as_deref(), Some(name));
    assert_eq!(
        host.device.cast::<ID3D12InfoQueue>().is_ok(),
        debug,
        "run release timing in its own process with the debug layer disabled"
    );
    let mut opts = D3D12_FEATURE_DATA_D3D12_OPTIONS5::default();
    // SAFETY: feature and destination structure agree.
    unsafe {
        host.device
            .CheckFeatureSupport(
                D3D12_FEATURE_D3D12_OPTIONS5,
                (&raw mut opts).cast(),
                size_of_val(&opts) as u32,
            )
            .unwrap();
    }
    assert!(opts.RaytracingTier.0 >= D3D12_RAYTRACING_TIER_1_1.0);
    eprintln!("adapter={name} DXR>=1.1 debug={debug}");
    host
}

fn clean(host: &Host) {
    host.check_debug_errors().unwrap();
    eprintln!("D3D12 errors/corruption=0");
}

#[test]
fn ada_phase1() {
    use victory_shader::emit::{
        SLOT_FAULTS, SLOT_MEMORY, SLOT_STRIDES, SLOT_TABLE_STRIDES, SLOT_TLAS_0, SLOT_TLAS_1,
        SLOT_TLAS_2,
    };
    assert_eq!(
        [
            SLOT_STRIDES,
            SLOT_MEMORY,
            SLOT_FAULTS,
            SLOT_TABLE_STRIDES,
            SLOT_TLAS_0,
            SLOT_TLAS_1,
            SLOT_TLAS_2
        ],
        [255, 254, 253, 252, 251, 250, 249]
    );
    for n in [0, 4] {
        assert!(victory_shader::emit::rt_replay::ray_query_shader(n, QueryMode::Closest).is_err());
    }
    let mut gpu = Gpu::new(ada(true));
    for layers in 1..=3 {
        let p = scenes::small(layers);
        let bytes = p.encode().unwrap();
        let packet = Packet::decode(&bytes).unwrap();
        assert_eq!(p, packet);
        let nodes = victory_shader::rt_packet::bvh4::pack(&packet).unwrap();
        assert!(
            nodes
                .chunks_exact(16)
                .any(|w| w[12..16] == [1, 0, 1, 0x0402]),
            "pair leaf geometry/prim0/prim1/I-J swizzles at +48/+52/+56/+60"
        );
        let expected = |n: usize| &packet.rays[n].1;
        assert_eq!(
            (expected(0)[0].t, expected(0)[0].u, expected(0)[0].v),
            (3., 0.25, 0.625)
        );
        assert_eq!(
            (
                expected(1)[0].geometry,
                expected(1)[0].primitive,
                expected(1)[0].u,
                expected(1)[0].v
            ),
            (1, 1, 0.375, 0.5)
        );
        assert_eq!(expected(2).len(), 2, "shared edge must hit one of the pair");
        assert!(expected(3).is_empty());
        assert_eq!(
            (expected(5)[0].id, expected(5)[0].front, expected(5)[0].t),
            (0x123456, true, 5.5)
        );
        assert!(expected(6).is_empty());
        assert_eq!(
            expected(8).first().map(|h| h.id),
            (layers >= 2).then_some(0x41)
        );
        assert_eq!(
            expected(9).first().map(|h| h.id),
            (layers >= 3).then_some(0x42)
        );
        assert_eq!(expected(11)[0].t, 8.);
        assert_eq!(
            expected(12).first().map(|h| h.t),
            (layers >= 2).then_some(1.)
        );
        assert!(expected(15).is_empty(), "zero-masked instance");
        assert_eq!(
            (expected(16)[0].id, expected(16)[0].front),
            (0x61, false),
            "mirror alone preserves object winding"
        );
        assert_eq!(
            (expected(17)[0].id, expected(17)[0].front, expected(17)[0].t),
            (0x62, true, 2.03125)
        );
        assert!(expected(18).is_empty(), "disjoint D5/20 masks");
        let all = oracle::Oracle::new(&packet).hits(&packet.rays[0].0, false);
        assert_eq!(
            all.iter().map(|h| h.t).collect::<Vec<_>>(),
            [8., 3.],
            "first input obstacle is not closest"
        );

        rt::bench::checked(&mut gpu, &packet);
    }
    let mut empty_near = scenes::small(2);
    empty_near.tlases[0].clear();
    oracle::freeze(&mut empty_near);
    rt::bench::checked(&mut gpu, &empty_near);

    let p = scenes::crossed_obstacles();
    rt::bench::checked(&mut gpu, &p);
    let first = Replay::new(&mut gpu, &p, QueryMode::FirstCandidate);
    first.build(&gpu, false);
    first.build(&gpu, true);
    first.trace(&gpu, true);
    let [_, dxr] = first.results(&mut gpu);
    oracle::candidates(&p, &dxr);
    let different = dxr
        .chunks_exact(10)
        .zip(&p.rays)
        .filter(|(candidate, (_, hits))| {
            let t = hits[0].t;
            f64::from(f32::from_bits(candidate[1])) > t + oracle::t_tolerance(t)
        })
        .count();
    eprintln!(
        "dual-obstacle rays: {different}/{} first candidates farther than closest",
        p.rays.len()
    );
    assert!(
        different > 0,
        "fixture did not distinguish first-candidate and closest traversal on Ada"
    );
    clean(&gpu.host);
}

#[test]
#[ignore = "full L1 packet under the D3D12 debug layer"]
fn l1_debug_packet() {
    let p = rt::bench::packet();
    let mut gpu = Gpu::new(ada(true));
    rt::bench::checked(&mut gpu, &p);
    clean(&gpu.host);
}

#[test]
#[ignore = "dedicated release timestamp benchmark on RTX 4070 Ti SUPER"]
fn l1_timestamps() {
    if cfg!(debug_assertions) {
        panic!("measure with cargo test --release");
    }
    let p = rt::bench::packet();
    let mut gpu = Gpu::new(ada(false));
    rt::bench::measure(&mut gpu, &p);
}
