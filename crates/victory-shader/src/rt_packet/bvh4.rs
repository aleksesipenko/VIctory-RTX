//! Pack v1 geometry as a software BVH4 for the replay kernel. Offsets are
//! relative to this single raw buffer. No per-ray CPU culling or hit table.
use super::{Packet, Point, Transform, Triangle, invalid};
use crate::bvh::NONE;
use std::io::Result;

#[derive(Clone, Copy)]
struct Node {
    ptr: u32,
    bounds: [f32; 6],
}

fn bounds(points: impl IntoIterator<Item = Point>) -> [f32; 6] {
    let mut b = [
        f32::INFINITY,
        f32::INFINITY,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::NEG_INFINITY,
        f32::NEG_INFINITY,
    ];
    for p in points {
        for a in 0..3 {
            b[a] = b[a].min(p[a]);
            b[3 + a] = b[3 + a].max(p[a]);
        }
    }
    b
}

fn put(buf: &mut Vec<u32>, kind: u32, words: &[u32]) -> Result<u32> {
    assert!(buf.len().is_multiple_of(16) && words.len().is_multiple_of(16));
    let byte = u32::try_from(buf.len() * 4)
        .map_err(|_| invalid("BVH4 exceeds raw buffer address space"))?;
    byte.checked_add(size_of_val(words) as u32)
        .ok_or_else(|| invalid("BVH4 node crosses address limit"))?;
    buf.extend_from_slice(words);
    Ok((byte / 8) | kind)
}

fn tree(buf: &mut Vec<u32>, leaves: &mut [Node]) -> Result<Node> {
    if leaves.len() == 1 {
        return Ok(leaves[0]);
    }
    let bbox = bounds(leaves.iter().flat_map(|n| {
        [
            [n.bounds[0], n.bounds[1], n.bounds[2]],
            [n.bounds[3], n.bounds[4], n.bounds[5]],
        ]
    }));
    let axis = (0..3)
        .max_by(|&a, &b| (bbox[3 + a] - bbox[a]).total_cmp(&(bbox[3 + b] - bbox[b])))
        .unwrap();
    let center = |n: &Node| n.bounds[axis] * 0.5 + n.bounds[3 + axis] * 0.5;
    leaves.sort_by(|a, b| center(a).total_cmp(&center(b)));
    let mut words = [0; 32];
    words[..4].fill(NONE);
    let chunk = leaves.len().div_ceil(4);
    for (i, group) in leaves.chunks_mut(chunk).enumerate() {
        let child = tree(buf, group)?;
        words[i] = child.ptr;
        words[4 + 6 * i..10 + 6 * i].copy_from_slice(&child.bounds.map(f32::to_bits));
    }
    Ok(Node {
        ptr: put(buf, 5, &words)?,
        bounds: bbox,
    })
}

/// Preserve original vertex order through the I/J swizzle at +60. For a
/// pair, shared vertices occupy slots 1 and 2 in the 64-byte leaf.
fn pair(a: Triangle, b: Triangle) -> Option<[Point; 4]> {
    let lone = |x: Triangle, y: Triangle| x.into_iter().find(|p| !y.contains(p));
    let [p, q] = [lone(a, b)?, lone(b, a)?];
    let mut shared = a.into_iter().filter(|p| b.contains(p));
    let (s, t) = (shared.next()?, shared.next()?);
    (s != t && shared.next().is_none()).then_some([p, s, t, q])
}

fn leaf(
    buf: &mut Vec<u32>,
    nodes: &mut Vec<Node>,
    g: u32,
    prim: u32,
    a: Triangle,
    second: Option<(Triangle, [Point; 4])>,
    opaque: bool,
) -> Result<()> {
    let verts = second.map_or([a[0], a[1], a[2], a[0]], |(_, v)| v);
    let mut words = [0; 16];
    for (i, p) in verts.iter().enumerate() {
        words[i * 3..i * 3 + 3].copy_from_slice(&p.map(f32::to_bits));
    }
    words[12] = g | ((opaque as u32) << 24);
    words[13] = prim;
    words[14] = if second.is_some() { prim + 1 } else { NONE };
    let id = |tri: Triangle, physical: Triangle| {
        let index = |p: Point| physical.iter().position(|&v| v == p).unwrap() as u32;
        index(tri[1]) | index(tri[2]) << 2 | ((opaque as u32) << 7)
    };
    // A single (including degenerate) triangle is always stored in source order.
    words[15] = 9 | ((opaque as u32) << 7);
    if let Some((b, _)) = second {
        words[15] =
            id(a, [verts[0], verts[1], verts[2]]) | id(b, [verts[1], verts[3], verts[2]]) << 8;
    }
    let ptr = put(buf, 0, &words)?;
    nodes.push(Node {
        ptr,
        bounds: bounds(a),
    });
    if let Some((b, _)) = second {
        nodes.push(Node {
            ptr: ptr | 1,
            bounds: bounds(b),
        });
    }
    Ok(())
}

fn inverse(m: Transform) -> Transform {
    let m = m.map(|r| r.map(f64::from));
    let cofactor = |r: usize, c: usize| {
        m[(r + 1) % 3][(c + 1) % 3] * m[(r + 2) % 3][(c + 2) % 3]
            - m[(r + 1) % 3][(c + 2) % 3] * m[(r + 2) % 3][(c + 1) % 3]
    };
    let det = (0..3).map(|c| m[0][c] * cofactor(0, c)).sum::<f64>();
    std::array::from_fn(|r| {
        let a: [f64; 3] = std::array::from_fn(|c| cofactor(c, r) / det);
        [
            a[0],
            a[1],
            a[2],
            -(a[0] * m[0][3] + a[1] * m[1][3] + a[2] * m[2][3]),
        ]
        .map(|f| f as f32)
    })
}

/// Header dwords 0..3 are ordered TLAS roots. Instance leaves hold the
/// inverse 3x4, ID/mask, contribution/flags, BLAS root and TLAS-local index.
pub fn pack(packet: &Packet) -> Result<Vec<u32>> {
    packet.validate()?;
    let mut buf = vec![NONE; 16];
    let roots: Vec<_> = packet
        .blases
        .iter()
        .map(|blas| {
            let mut nodes = Vec::new();
            for (g, geom) in blas.iter().enumerate() {
                let mut i = 0;
                while i < geom.triangles.len() {
                    let a = geom.triangles[i];
                    let second = geom
                        .triangles
                        .get(i + 1)
                        .and_then(|&b| pair(a, b).map(|verts| (b, verts)));
                    leaf(
                        &mut buf,
                        &mut nodes,
                        g as u32,
                        i as u32,
                        a,
                        second,
                        geom.opaque,
                    )?;
                    i += 1 + usize::from(second.is_some());
                }
            }
            tree(&mut buf, &mut nodes)
        })
        .collect::<Result<_>>()?;
    for (n, tlas) in packet.tlases.iter().enumerate() {
        let mut nodes = Vec::new();
        for (idx, i) in tlas.iter().enumerate() {
            let blas = roots[i.blas as usize];
            let mut words = [0; 16];
            for (d, f) in words
                .iter_mut()
                .zip(inverse(i.transform).into_iter().flatten())
            {
                if !f.is_finite() {
                    return Err(invalid("instance inverse exceeds f32"));
                }
                *d = f.to_bits();
            }
            words[12] = i.id | u32::from(i.mask) << 24;
            words[13] = i.contribution | ((i.front_ccw as u32) << 25);
            words[14] = blas.ptr;
            words[15] = idx as u32;
            let corners = (0..8).map(|bits| {
                let [x, y, z]: Point =
                    std::array::from_fn(|a| blas.bounds[a + 3 * ((bits >> a) & 1)]);
                i.transform.map(|r| {
                    let [a, b, c, d] = r.map(f64::from);
                    (a * f64::from(x) + b * f64::from(y) + c * f64::from(z) + d) as f32
                })
            });
            let mut b = bounds(corners);
            if !b.iter().all(|v| v.is_finite()) {
                return Err(invalid("instance bounds exceed f32"));
            }
            for a in 0..3 {
                b[a] = b[a].next_down();
                b[3 + a] = b[3 + a].next_up();
            }
            nodes.push(Node {
                ptr: put(&mut buf, 6, &words)?,
                bounds: b,
            });
        }
        if !nodes.is_empty() {
            buf[n] = tree(&mut buf, &mut nodes)?.ptr;
        }
    }
    Ok(buf)
}
