//! Independent f64 MГ¶llerвЂ“Trumbore in world space. No guest node decoding,
//! traversal order, inverse transform or DXIL arithmetic is shared with SW.
use victory_shader::rt_packet::{Hit, Packet, Ray};

type Vec3 = [f64; 3];
fn sub(a: Vec3, b: Vec3) -> Vec3 {
    std::array::from_fn(|k| a[k] - b[k])
}
fn cross(a: Vec3, b: Vec3) -> Vec3 {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}
fn dot(a: Vec3, b: Vec3) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

struct Primitive {
    a: Vec3,
    e1: Vec3,
    e2: Vec3,
    mask: u8,
    flip: bool,
    attributes: Hit,
}
pub struct Oracle(Vec<Vec<Primitive>>);
impl Oracle {
    pub fn new(packet: &Packet) -> Self {
        Self(
            packet
                .tlases
                .iter()
                .map(|instances| {
                    let mut primitives = Vec::new();
                    for (idx, i) in instances.iter().enumerate() {
                        let rows = i.transform.map(|r| r.map(f64::from));
                        let [a, b, c] = rows.map(|[x, y, z, _]| [x, y, z]);
                        // DXR defines front facing in object space. Cancel the
                        // orientation reversal of a mirrored object-to-world transform.
                        let flip = (dot(a, cross(b, c)) < 0.) ^ i.front_ccw;
                        for (g, geom) in packet.blases[i.blas as usize].iter().enumerate() {
                            for (p, tri) in geom.triangles.iter().enumerate() {
                                let [a, b, c] = tri.map(|v| {
                                    rows.map(|[x, y, z, w]| dot([x, y, z], v.map(f64::from)) + w)
                                });
                                let attributes = Hit {
                                    t: 0.,
                                    u: 0.,
                                    v: 0.,
                                    id: i.id,
                                    instance: idx as u32,
                                    contribution: i.contribution,
                                    geometry: g as u32,
                                    primitive: p as u32,
                                    front: false,
                                };
                                primitives.push(Primitive {
                                    a,
                                    e1: sub(b, a),
                                    e2: sub(c, a),
                                    mask: i.mask,
                                    flip,
                                    attributes,
                                });
                            }
                        }
                    }
                    primitives
                })
                .collect(),
        )
    }

    pub fn hits(&self, r: &Ray, closest_only: bool) -> Vec<Hit> {
        let origin = r.origin.map(f64::from);
        let dir = r.direction.map(f64::from);
        for tlas in &self.0 {
            let mut hits = Vec::new();
            for p in tlas {
                if p.mask & r.mask == 0 {
                    continue;
                }
                let h = cross(dir, p.e2);
                let det = dot(p.e1, h);
                if det == 0. {
                    continue;
                }
                let rel = sub(origin, p.a);
                let u = dot(rel, h) / det;
                if !(0. ..=1.).contains(&u) {
                    continue;
                }
                let q = cross(rel, p.e1);
                let v = dot(dir, q) / det;
                if v < 0. || u + v > 1. {
                    continue;
                }
                let t = dot(p.e2, q) / det;
                if t > f64::from(r.t_min) && t < f64::from(r.t_max) {
                    hits.push(Hit {
                        t,
                        u,
                        v,
                        front: (det > 0.) ^ p.flip,
                        ..p.attributes
                    });
                }
            }
            if !hits.is_empty() {
                if closest_only {
                    let min = hits.iter().map(|h| h.t).fold(f64::INFINITY, f64::min);
                    // Classify coincident surfaces/edges at double precision,
                    // independently of the much wider GPU comparison tolerance.
                    hits.retain(|h| (h.t - min).abs() <= 8. * f64::EPSILON * min.abs().max(1.));
                }
                return hits;
            }
        }
        Vec::new()
    }
}

pub fn freeze(p: &mut Packet) {
    let oracle = Oracle::new(p);
    for (r, hits) in &mut p.rays {
        *hits = oracle.hits(r, true);
    }
}

pub fn t_tolerance(t: f64) -> f64 {
    let rounded = t as f32;
    let ulp = if rounded >= f32::MAX {
        f64::from(f32::MAX) - f64::from(f32::MAX.next_down())
    } else {
        f64::from(rounded.next_up()) - f64::from(rounded)
    };
    (8. * ulp).max(1e-5 * t.abs().max(1.))
}

fn close(a: &Hit, b: &Hit, t_ref: f64) -> bool {
    (a.t - b.t).abs() <= t_tolerance(t_ref)
        && (a.u - b.u).abs() <= 2e-5
        && (a.v - b.v).abs() <= 2e-5
        && (
            a.id,
            a.instance,
            a.contribution,
            a.geometry,
            a.primitive,
            a.front,
        ) == (
            b.id,
            b.instance,
            b.contribution,
            b.geometry,
            b.primitive,
            b.front,
        )
}

fn unpack(w: &[u32]) -> Option<Hit> {
    if w[0] == 0 {
        assert_eq!(w, [0; 10], "miss must initialize the entire record");
        return None;
    }
    assert_eq!(w[0], 1);
    assert!(w[9] <= 1);
    let f = |i: usize| f64::from(f32::from_bits(w[i]));
    Some(Hit {
        t: f(1),
        u: f(2),
        v: f(3),
        id: w[4],
        instance: w[5],
        contribution: w[6],
        geometry: w[7],
        primitive: w[8],
        front: w[9] != 0,
    })
}

pub fn parity(packet: &Packet, sw: &[u32], dxr: &[u32]) {
    assert_eq!(sw.len(), packet.rays.len() * 10);
    assert_eq!(sw.len(), dxr.len());
    let oracle = Oracle::new(packet);
    for (n, ((ray, saved), (a, b))) in packet
        .rays
        .iter()
        .zip(sw.chunks_exact(10).zip(dxr.chunks_exact(10)))
        .enumerate()
    {
        let expected = oracle.hits(ray, true);
        assert!(
            saved.len() == expected.len()
                && expected
                    .iter()
                    .all(|h| saved.iter().any(|e| close(e, h, h.t))),
            "packet reference differs from f64 oracle, ray {n}: saved {saved:?}, oracle {expected:?}"
        );
        let (a, b) = (unpack(a), unpack(b));
        for (path, result) in [("SW", a), ("DXR", b)] {
            assert!(
                match result {
                    Some(h) =>
                        expected.iter().any(|e| close(&h, e, e.t))
                            && saved.iter().any(|e| close(&h, e, e.t)),
                    None => expected.is_empty(),
                },
                "ray {n} {path}: {result:?}, expected {expected:?}"
            );
        }
        if expected.len() == 1 {
            assert!(
                close(&a.unwrap(), &b.unwrap(), expected[0].t),
                "SW/DXR ray {n}: {a:?} vs {b:?}"
            );
        }
    }
}

pub fn candidates(p: &Packet, words: &[u32]) {
    let o = Oracle::new(p);
    assert_eq!(words.len(), p.rays.len() * 10);
    for ((r, _), w) in p.rays.iter().zip(words.chunks_exact(10)) {
        let acceptable = o.hits(r, false);
        let actual = unpack(w);
        assert!(
            match actual {
                Some(h) => acceptable.iter().any(|e| close(&h, e, e.t)),
                None => acceptable.is_empty(),
            },
            "candidate {actual:?}, allowed {acceptable:?}"
        );
    }
}
