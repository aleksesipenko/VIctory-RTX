//! Frozen workload, balanced randomized four-dispatch blocks. GPU barriers,
//! CPU preparation, binding, transfers and readback are outside each interval.
use super::{
    gpu::{self, Gpu, Replay},
    oracle, scenes,
};
use victory_shader::{emit::rt_replay::QueryMode, rt_packet::Packet};
use windows::Win32::Graphics::Direct3D12::*;

const WARM: usize = 20;
const PAIRS: usize = 100;
const INTERVALS: usize = 6;
const SEED: u64 = 0xacce_4070_11ab_ba01;
struct Random(u64);
impl Random {
    fn index(&mut self, n: usize) -> usize {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        ((u128::from(self.0) * n as u128) >> 64) as usize
    }
}
fn orders(n: usize, rng: &mut Random) -> Vec<bool> {
    let mut orders: Vec<_> = (0..n).map(|i| i % 2 == 0).collect();
    for i in (1..n).rev() {
        let j = rng.index(i + 1);
        orders.swap(i, j);
    }
    orders
}
fn quantile(sorted: &[f64], p: f64) -> f64 {
    let x = (sorted.len() - 1) as f64 * p;
    let lo = x.floor() as usize;
    let hi = x.ceil() as usize;
    sorted[lo] + (sorted[hi] - sorted[lo]) * x.fract()
}
fn summary(mut v: Vec<f64>) -> (f64, f64) {
    v.sort_by(f64::total_cmp);
    (quantile(&v, 0.5), quantile(&v, 0.95))
}
fn paired_ratio(sw: &[f64], hw: &[f64]) -> (f64, f64, f64) {
    // Log ratios give each block equal multiplicative weight. Resample
    // complete paired blocks, never the individual A/B timestamps.
    let logs: Vec<_> = sw.iter().zip(hw).map(|(a, b)| (a / b).ln()).collect();
    let n = logs.len();
    let ratio = (logs.iter().sum::<f64>() / n as f64).exp();
    let mut rng = Random(SEED ^ 0x95);
    let mut bootstrap = Vec::with_capacity(10_000);
    for _ in 0..10_000 {
        let sum = (0..n).map(|_| logs[rng.index(n)]).sum::<f64>();
        bootstrap.push((sum / n as f64).exp());
    }
    bootstrap.sort_by(f64::total_cmp);
    (
        ratio,
        quantile(&bootstrap, 0.025),
        quantile(&bootstrap, 0.975),
    )
}

pub fn packet() -> Packet {
    let bytes = match std::env::var_os("VICTORY_RT_PACKET") {
        Some(path) => std::fs::read(path).expect("read VICTORY_RT_PACKET"),
        None => scenes::workload().encode().unwrap(),
    };
    let p = Packet::decode(&bytes).unwrap();
    let triangles: usize = p.blases.iter().flatten().map(|g| g.triangles.len()).sum();
    let hits = p.rays.iter().filter(|(_, e)| !e.is_empty()).count();
    eprintln!(
        "packet={} BLAS={} triangles={} TLAS={} instances={} rays={} hits={} misses={}",
        blake3::hash(&bytes),
        p.blases.len(),
        triangles,
        p.tlases.len(),
        p.tlases.iter().map(Vec::len).sum::<usize>(),
        p.rays.len(),
        hits,
        p.rays.len() - hits
    );
    p
}

pub fn checked(g: &mut Gpu, p: &Packet) -> Replay {
    let replay = Replay::new(g, p, QueryMode::Closest);
    replay.build(g, false);
    replay.build(g, true);
    replay.trace(g, false);
    replay.trace(g, true);
    let [sw, hw] = replay.results(g);
    oracle::parity(p, &sw, &hw);
    replay
}

pub fn measure(g: &mut Gpu, p: &Packet) {
    let replay = checked(g, p);
    let mut rng = Random(SEED);
    let order: Vec<_> = orders(WARM, &mut rng)
        .into_iter()
        .chain(orders(PAIRS, &mut rng))
        .collect();
    let count = (order.len() * INTERVALS * 2) as u32;
    let heap = g.timestamps(count);
    let rb = g.readback(u64::from(count) * 8);
    // SAFETY: all timestamp pairs are written on the same direct queue and
    // resolved once, in bounds, before the fence and the first CPU read.
    let frequency = unsafe { g.queue.GetTimestampFrequency().unwrap() };
    assert!(frequency > 0);
    let mut i = 0;
    let mut stamp = || {
        unsafe {
            g.list.EndQuery(&heap, D3D12_QUERY_TYPE_TIMESTAMP, i);
        }
        i += 1;
    };
    // Record warmup and measured blocks contiguously. The GPU stays fed,
    // including across block boundaries, without changing machine clocks.
    for &hw_first in &order {
        for top in [false, true] {
            stamp();
            replay.record_builds(g, top);
            stamp();
            g.barrier();
        }
        for hw in [hw_first, !hw_first, !hw_first, hw_first] {
            replay.bind(g, hw);
            stamp();
            replay.dispatch(g);
            stamp();
            g.barrier();
        }
    }
    assert_eq!(i, count);
    unsafe {
        g.list
            .ResolveQueryData(&heap, D3D12_QUERY_TYPE_TIMESTAMP, 0, count, &rb, 0);
    }
    g.submit();
    let values = gpu::words(&rb, count as usize * 2);
    let ticks: Vec<_> = values
        .chunks_exact(2)
        .map(|a| u64::from(a[0]) | u64::from(a[1]) << 32)
        .collect();
    let micros: Vec<_> = ticks
        .chunks_exact(2)
        .map(|q| {
            assert!(q[1] > q[0], "invalid timestamp interval {q:?}");
            (q[1] - q[0]) as f64 * 1e6 / frequency as f64
        })
        .collect();
    let (mut blas, mut tlas, mut sw, mut hw) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for (&hw_first, times) in order.iter().zip(micros.chunks_exact(INTERVALS)).skip(WARM) {
        blas.push(times[0]);
        tlas.push(times[1]);
        let outer = (times[2] + times[5]) * 0.5;
        let inner = (times[3] + times[4]) * 0.5;
        sw.push(if hw_first { inner } else { outer });
        hw.push(if hw_first { outer } else { inner });
    }
    assert_eq!(sw.len(), PAIRS);
    let [a, b] = replay.results(g);
    oracle::parity(p, &a, &b);
    println!(
        "warmup={WARM} measured={PAIRS} ABBA=50 BAAB=50 seed={SEED:#x} frequency_hz={frequency}"
    );
    println!("stage                 p50_us       p95_us");
    for (label, v) in [
        ("BLAS build", &blas),
        ("TLAS build", &tlas),
        ("SW dispatch", &sw),
        ("Ada RayQuery", &hw),
    ] {
        let (p50, p95) = summary(v.clone());
        println!("{label:18} {p50:11.4}  {p95:11.4}");
    }
    let (ratio, lo, hi) = paired_ratio(&sw, &hw);
    println!(
        "paired SW/DXR geometric_mean={ratio:.4}x CI_low={lo:.4} CI_high={hi:.4} (95%, 10000 block bootstraps)"
    );
}
