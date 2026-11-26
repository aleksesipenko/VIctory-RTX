//! Offline acceptance. Feed concatenated console DXBC containers on stdin or set
//! VICTORY_RT_CORPUS to such a file. Only native compute programs with an actual
//! decoded BVH intersection instruction enter the denominator.
#![cfg(windows)]

use std::{
    collections::HashSet,
    io::{self, Read},
};
use victory_shader::{
    Container, FourCC, HwStage,
    emit::{
        self,
        host::{Host, enable_debug_layer},
        validator::Validator,
    },
    gfx10,
    native::StageIo,
    native_programs,
};

#[test]
#[ignore = "requires a user-supplied trace shader corpus and the named Ada adapter"]
fn ada_trace_corpus() {
    enable_debug_layer().unwrap();
    let host = Host::on_adapter("NVIDIA GeForce RTX 4070 Ti SUPER").unwrap();
    assert_eq!(
        host.adapter_name().as_deref(),
        Some("NVIDIA GeForce RTX 4070 Ti SUPER")
    );
    assert_eq!(host.translation_caps(), Some((32, false)));
    emit::set_host_max_wave(32);
    let mut validator = Validator::load(Validator::default_path().as_deref()).unwrap();
    let mut input: Box<dyn Read> = match std::env::var_os("VICTORY_RT_CORPUS") {
        Some(path) => Box::new(std::fs::File::open(path).unwrap()),
        None => Box::new(io::stdin().lock()),
    };
    let mut seen = HashSet::new();
    let (mut traces, mut lifted, mut paired, mut validated, mut pipelines) = (0, 0, 0, 0, 0);
    let mut failed = Vec::new();
    loop {
        let mut header = [0; 32];
        if input.read(&mut header[..1]).unwrap() == 0 {
            break;
        }
        input.read_exact(&mut header[1..]).unwrap();
        assert_eq!(&header[..4], b"DXBC");
        let size = u32::from_le_bytes(header[24..28].try_into().unwrap()) as usize;
        assert!((32..=64 << 20).contains(&size));
        let mut bytes = vec![0; size];
        bytes[..32].copy_from_slice(&header);
        input.read_exact(&mut bytes[32..]).unwrap();
        let c = Container::parse(&bytes).unwrap();
        let Some(part) = c.part(FourCC::DXIL).or(c.part(FourCC::ACCC)) else {
            continue;
        };
        let digest = blake3::hash(&bytes);
        if !seen.insert(digest) {
            continue;
        }
        let Some(native) = native_programs(part.data)
            .unwrap()
            .into_iter()
            .find(|n| n.stage == HwStage::Cs)
        else {
            continue;
        };
        if !gfx10::decode(&native.code).unwrap().iter().any(|i| {
            matches!(
                i.name(),
                Some("image_bvh_intersect_ray" | "image_bvh64_intersect_ray")
            )
        }) {
            continue;
        }
        let StageIo::Cs(cs) = native.io else {
            unreachable!()
        };
        traces += 1;
        let id = digest.to_hex()[..16].to_owned();
        eprintln!("translating {id}");
        let result = (|| -> Result<(), String> {
            let shader = emit::translate(&bytes).map_err(|e| e.to_string())?;
            let hash = blake3::hash(format!("{shader:?}").as_bytes()).to_hex();
            let c = Container::parse(&shader.dxil).unwrap();
            let sfi = c.part(FourCC::SFI0).unwrap().data;
            let rt = u64::from_le_bytes(sfi[..8].try_into().unwrap()) & 0x10_0000 != 0;
            let psv = c.part(FourCC::PSV0).unwrap().data;
            let threads: u32 = psv[40..52]
                .chunks_exact(4)
                .map(|w| u32::from_le_bytes(w.try_into().unwrap()))
                .product();
            let guest: u32 = cs.threads.iter().product();
            let pair = native.wave64 && threads * 2 == guest;
            assert!(threads == guest || pair, "invalid dispatch shape");
            lifted += usize::from(rt);
            paired += usize::from(pair);
            eprintln!(
                "{id} wave64={} threads={guest}/{threads} paired={pair} lifted={rt} shader={hash}",
                native.wave64
            );
            // The validator signs in place. Give the driver the original bytes
            // that a normal translation would dispatch.
            validator.validate(&mut shader.dxil.clone())?;
            validated += 1;
            host.compute_pipeline(&shader)?;
            pipelines += 1;
            Ok(())
        })();
        if let Err(e) = result {
            eprintln!("{id} FAILED {e}");
            failed.push((id, e));
        }
    }
    assert!(failed.is_empty(), "{failed:?}");
    host.check_debug_errors().unwrap();
    eprintln!(
        "corpus={traces} lifted={lifted} paired={paired} dxil={validated} pso={pipelines} D3D12 errors/corruption=0"
    );
    assert!(traces > 0);
    if let Ok(expected) = std::env::var("VICTORY_RT_EXPECT_TRACES") {
        assert_eq!(
            traces,
            expected.parse::<usize>().unwrap(),
            "incomplete corpus"
        );
    }
    if let Ok(expected) = std::env::var("VICTORY_RT_EXPECT_LIFTED") {
        assert_eq!(
            lifted,
            expected.parse::<usize>().unwrap(),
            "unexpected RayQuery count"
        );
    } else if std::env::var_os("VICTORY_NATIVE_RT").is_some_and(|v| v == "1") {
        assert!(
            lifted > 0,
            "native RT selected but no trace shader contains RayQuery"
        );
    } else {
        assert_eq!(lifted, 0);
    }
}
