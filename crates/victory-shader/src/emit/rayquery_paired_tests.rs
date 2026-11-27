//! Regression for #854. Inject a recognized loop at the matcher/lowerer
//! boundary so this test also runs without game assets. The surrounding program
//! is guest ISA, including the EXEC comparison, descriptor loads and all stores.
use super::*;
use crate::emit::{
    host::{Host, enable_debug_layer},
    rayquery_tests::{Bindings, InstanceSpec, TriangleGeom, run_scene},
    validator::Validator,
    wave64_tests::native_with,
};
use crate::{io, ir, resource, structure};
use lift::{RayOutputBinding, RayTemplate};
use windows::Win32::Graphics::Direct3D12::D3D12_RAYTRACING_GEOMETRY_FLAG_OPAQUE;

fn kernel(wide: bool) -> (Vec<u8>, usize, usize) {
    let mov = |dst: u32, src: u32| 0x7e00_0200 | dst << 17 | src;
    let cvt = |dst: u32, src: u32| 0x7e00_0c00 | dst << 17 | (256 + src);
    let op = |op: u32, dst: u32, a: u32, b: u32| op << 25 | dst << 17 | b << 9 | a;
    let mut code = vec![
        // s_load_dwordx4 s[12:15], s[4:5], 32 (the second heap slot).
        0x3d << 26 | 2 << 18 | 12 << 6 | 2,
        125 << 25 | 32,
        cvt(1, 0),
        op(3, 1, 255, 1),
        0.25f32.to_bits(),   // origin.x = lane + .25
        op(0x16, 2, 133, 0), // half = lane >> 5
        op(0x1b, 3, 135, 0), // lane & 7
        mov(4, 129),
        op(0x1a, 4, 258, 4), // inclusion mask = 1 << half
        cvt(5, 2),
        op(3, 5, 242, 5), // direction.z = float(half) + 1
        cvt(6, 2),
        op(8, 6, 240, 6), // tMin = float(half) * .5
        op(0x1b, 7, 131, 0),
        cvt(7, 7),
        op(3, 7, 255, 7),
        1.25f32.to_bits(), // keep hits off the tMax boundary
        op(0x1b, 8, 132, 0),
        op(0x1a, 8, 134, 8),                          // flags = (lane & 4) << 6
        0x7c00_0000 | 0xc5 << 17 | 3 << 9 | 258,      // v_cmp_ne_u32 vcc, v2, v3
        0xbe90_246a,                                  // s_and_saveexec_b64 s[16:17], vcc
        if wide { 0xbf88_000a } else { 0xbf82_0000 }, // EXECZ skips the trace for the whole wave
    ];
    if wide {
        // The branch can enter both virtual halves because one lane is live.
        // The guest intersection itself is guarded by its incoming EXEC.
        code.extend([
            0xf19c_9f07,
            0x0003_0801,
            0x0504_0302,
            0x0908_0706,
            0x000c_0b0a,
        ]);
    }
    for reg in 1..=4 {
        code.push(mov(reg, 128));
    } // trace live-outs
    code.push(0xbf82_0000); // trace -> readback
    code.push(0xbefe_0410); // restore EXEC so every launched lane reads its result
    for reg in 5..=7 {
        code.push(mov(reg, 128));
    } // post-loop metadata loads
    code.extend([mov(8, 106), mov(9, 107)]); // materialize both dwords of VCC
    code.push(op(0x1a, 15, 134, 0)); // lane's 64-byte row
    for reg in 0..=9 {
        code.extend([0xe070_1000 | (4 * reg), 0x8002_0000 | reg << 8 | 15]);
    }

    let native = native_with(&code, true, 64, 0);
    let p = ir::build(&native.code, true, native.vgprs).unwrap();
    assert_eq!(p.blocks.len(), 3);
    let st = if wide {
        structure::structure_wide(&p, &HashSet::from([0]))
    } else {
        structure::structure(&p)
    }
    .unwrap();
    let io = io::raise(&p, &native).unwrap();
    let res = resource::track(&p, &native.layout, native.stage, Some(&st)).unwrap();
    let parts = Parts {
        native: &native,
        p: &p,
        st: &st,
        io: &io,
        res: &res,
        inputs: vec![],
        outputs: vec![],
        prims: vec![],
        mesh_stride: 0,
    };
    let written = |block: usize, reg| {
        p.blocks[block]
            .insts
            .iter()
            .rev()
            .find_map(|i| i.write(reg))
            .unwrap()
    };
    let pre = |r| RayInput::Val(written(0, Reg::V(r)));
    let tmax = written(0, Reg::V(7));
    let tlas = p.blocks[0]
        .insts
        .iter()
        .position(|i| i.write(Reg::S(12)).is_some())
        .unwrap();
    let meta = [
        RayMetaKind::InstanceId,
        RayMetaKind::GeometryIndexShl8,
        RayMetaKind::PrimitiveIndex,
    ];
    let outputs = [
        RayOutput::CommittedStatusTriNode,
        RayOutput::CommittedRayT,
        RayOutput::CommittedBary(0),
        RayOutput::CommittedBary(1),
    ];
    let region = RayLoop {
        rq_group: 0,
        entry_block: 1,
        exit_pred: 1,
        exit_block: 2,
        skipped_blocks: vec![],
        template: RayTemplate::Bvh64,
        kind: RayLoopKind::OpaqueOnly,
        tlas_load: (0, tlas),
        active_ray_mask: None, // Bvh64 has no extra checkerboard mask
        origin: [
            pre(1),
            RayInput::Const(0.25f32.to_bits()),
            RayInput::Const(0),
        ],
        dir: [RayInput::Const(0), RayInput::Const(0), pre(5)],
        t_min: Some(pre(6)),
        t_max: pre(7),
        instance_mask: pre(4),
        ray_flags: pre(8),
        outputs: outputs
            .into_iter()
            .enumerate()
            .map(|(i, output)| RayOutputBinding {
                value: written(1, Reg::V(1 + i as u8)),
                output,
                fallback: (output == RayOutput::CommittedRayT).then_some(tmax),
            })
            .collect(),
        completed_timeouts: vec![],
        dead_exit_masks: vec![],
        shared_entry: None,
        committed_meta_loads: meta
            .into_iter()
            .enumerate()
            .map(|(i, kind)| ((2, i + 1), kind))
            .collect(),
    };
    // The matcher is exercised separately on the real corpus; this fixture
    // also passes through the production admission rule for a matched region.
    let plan = with_paired_lanes(|| LiftPlan::admit(&parts, vec![region], true));
    assert_eq!(plan.entries.len(), 1);
    let bad = HashMap::new();
    set_thread_max_wave(Some(32));
    let result = with_paired_lanes(|| Lower::new(&parts, &bad, &plan).run(true));
    set_thread_max_wave(None);
    let lowered = match result {
        Ok(shader) => shader,
        Err(Stop::Fail(e)) => panic!("paired lowering: {e}"),
        Err(Stop::Retry(v, w)) => panic!("paired lowering: {v:?}: {w}"),
    };
    assert_eq!(lowered.wave_size, Some(32));
    assert!(lowered.root_srvs.is_empty() && lowered.root_uavs.is_empty());
    let n = lowered.user_data.iter().map(|d| d.index + 1).max().unwrap();
    let count_slot = lowered
        .user_data
        .iter()
        .find(|d| d.root == RootType::Other(crate::emit::USER_DISPATCH_THREADS_X))
        .unwrap();
    assert_eq!(count_slot.dword, 64);
    (lowered.dxil, n as usize, count_slot.index as usize)
}

#[test]
fn ada_paired_lift_all_64_lanes() {
    // D3D12 requires enabling the debug layer before the first device.
    // The rest of the unit suite creates devices on parallel test threads.
    const CHILD: &str = "VICTORY_TEST_PAIRED_RAYQUERY_CHILD";
    if std::env::var_os(CHILD).is_none() {
        let output = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "emit::lower::rayquery_lower::paired_tests::ada_paired_lift_all_64_lanes",
                "--nocapture",
            ])
            .env(CHILD, "1")
            .output()
            .unwrap();
        let log = String::from_utf8_lossy(&output.stderr);
        assert!(
            output.status.success()
                && (log.contains("paired parity=64/64") || log.contains("skipped: no DXR 1.1")),
            "{log}"
        );
        eprintln!("{log}");
        return;
    }
    let require = std::env::var_os("VICTORY_REQUIRE_DXR_TESTS").is_some_and(|v| v == "1");
    if let Err(e) = enable_debug_layer() {
        if require {
            panic!("Windows Graphics Tools required: {e}");
        }
        eprintln!("skipped: no DXR 1.1 debug layer ({e})");
        return;
    }
    let host = match std::env::var("VICTORY_D3D12_ADAPTER") {
        Ok(name) if !name.trim().is_empty() => Host::on_adapter(name.trim()),
        _ => Host::new(false),
    };
    let host = match host {
        Ok(h) => h,
        Err(e) => {
            if require {
                panic!("DXR 1.1 hardware adapter required: {e}");
            }
            eprintln!("skipped: no DXR 1.1 hardware adapter ({e})");
            return;
        }
    };
    let mut o5 = windows::Win32::Graphics::Direct3D12::D3D12_FEATURE_DATA_D3D12_OPTIONS5::default();
    let tier_ok = unsafe {
        host.device
            .CheckFeatureSupport(
                windows::Win32::Graphics::Direct3D12::D3D12_FEATURE_D3D12_OPTIONS5,
                (&raw mut o5).cast(),
                size_of_val(&o5) as u32,
            )
            .is_ok()
    } && o5.RaytracingTier.0 >= windows::Win32::Graphics::Direct3D12::D3D12_RAYTRACING_TIER_1_1.0;
    if !tier_ok || host.translation_caps() != Some((32, false)) {
        if require {
            panic!(
                "adapter lacks DXR Tier 1.1 or Wave32 caps: tier_ok={tier_ok} caps={:?}",
                host.translation_caps()
            );
        }
        eprintln!(
            "skipped: no DXR 1.1 Wave32 hardware adapter (tier_ok={tier_ok}, caps={:?})",
            host.translation_caps()
        );
        return;
    }
    let verts = [0, 1].map(|g| {
        (0..16)
            .flat_map(|prim| {
                let x = (prim * 2 + g) as f32;
                let z = (1 + (prim & 3)) as f32;
                [[x, 0., z], [x + 1., 0., z], [x, 1., z]]
            })
            .collect::<Vec<_>>()
    });
    let geoms = verts.each_ref().map(|vertices| TriangleGeom {
        vertices,
        flags: D3D12_RAYTRACING_GEOMETRY_FLAG_OPAQUE,
    });
    let instances = [0, 1].map(|h| InstanceSpec {
        transform: [
            1.,
            0.,
            0.,
            h as f32 * 31.,
            0.,
            1.,
            0.,
            0.,
            0.,
            0.,
            1.,
            h as f32,
        ],
        instance_id: [17, 93][h],
        instance_mask: 1 << h,
        contrib: 0,
        flags: 0,
    });
    for wide in [false, true] {
        let (mut dxil, n, count_slot) = kernel(wide);
        Validator::load(Validator::default_path().as_deref())
            .unwrap()
            .validate(&mut dxil)
            .unwrap();
        for count in [64usize, 45] {
            let mut constants = vec![0; n];
            constants[count_slot] = count as u32;
            let words = run_scene(
                &host,
                &dxil,
                Bindings::Guest(constants),
                &geoms,
                &instances,
                64 * 16,
            )
            .unwrap();
            let active = |lane: usize| lane < count && lane & 7 != lane >> 5;
            let masks = [0, 1]
                .map(|half| (0..32).fold(0u32, |m, n| m | (u32::from(active(32 * half + n)) << n)));
            let mut hits = [0; 2];
            for lane in 0..64 {
                if lane >= count {
                    assert_eq!(
                        &words[lane * 16..lane * 16 + 10],
                        &[0; 10],
                        "unlaunched lane {lane} wrote a row"
                    );
                    continue;
                }
                let half = lane >> 5;
                let x = lane - half * 31;
                let t = ((1 + ((x / 2) & 3)) + half) as f32 / (1 + half) as f32;
                let tmax = 1.25 + (lane & 3) as f32;
                let hit = active(lane) && lane & 4 == 0 && x < 32 && t <= tmax;
                hits[half] += u32::from(hit);
                let uv = if hit { 0.25f32.to_bits() } else { 0 };
                let mut want = [
                    lane as u32,
                    u32::from(hit),
                    if hit { t } else { tmax }.to_bits(),
                    uv,
                    uv,
                    if hit { [17, 93][half] } else { 0 },
                    if hit { (x as u32 & 1) << 8 } else { 0 },
                    if hit { x as u32 / 2 } else { 0 },
                    masks[0],
                    masks[1],
                ];
                if wide && !active(lane) {
                    // The guest never wrote these VGPRs in this lane.
                    want[1..5].copy_from_slice(&[
                        (lane as f32 + 0.25).to_bits(),
                        half as u32,
                        (lane & 7) as u32,
                        1 << half,
                    ]);
                }
                assert_eq!(
                    &words[lane * 16..lane * 16 + 10],
                    &want,
                    "console lane {lane}"
                );
            }
            host.check_debug_errors().unwrap();
            eprintln!(
                "paired parity={count}/{count} hits={hits:?} VCC={masks:x?} wide={wide} D3D12 errors/corruption=0"
            );
        }
    }
}
