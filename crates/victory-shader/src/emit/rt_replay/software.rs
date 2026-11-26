//! GPU traversal of the packed TLAS and BLAS. Each lane owns two bounded
//! stacks; the inner walk returns to the outer instance walk. The replay
//! builder partitions into four equal-sized subtrees, so even a u32-sized
//! buffer needs fewer than 64 pending children per stack.
use super::*;
use crate::bvh::NONE;
use crate::emit::lower::intersect::{NodeAt, NodeCode, Ray as NodeRay};

type Words = [V; HIT_WORDS];
const DEPTH: u32 = 64;

#[derive(Clone, Copy)]
struct Scene {
    nodes: V,
    stack: V,
    base: V,
}

fn stack_at(s: &mut Shader, b: usize, stack: V, base: V, n: V) -> V {
    let at = s.m.bin(b, Bin::Add, base, n);
    s.m.element(b, stack, at)
}

fn walk(
    s: &mut Shader,
    start: usize,
    scene: Scene,
    root: V,
    ray: Ray,
    initial: Words,
    leaf: impl FnOnce(&mut Shader, usize, V, V, Words) -> (usize, Words),
) -> (usize, Words) {
    let zero = s.m.i32(0);
    let one = s.m.i32(1);
    let none = s.m.i32(NONE);
    let nonempty = s.m.cmp(start, pred::NE, root, none);
    let count = s.m.cast(start, Cast::ZExt, nonempty, s.t.i32);
    let first = stack_at(s, start, scene.stack, scene.base, zero);
    s.m.store(start, first, root);
    let one_f = s.m.f32(1f32.to_bits());
    let inverse = ray.direction.map(|d| s.m.bin(start, Bin::SDiv, one_f, d));
    let head = s.m.new_block();
    let pop = s.m.new_block();
    let box_node = s.m.new_block();
    let leaf_node = s.m.new_block();
    let exit = s.m.new_block();
    s.m.terminate(start, Term::Br(head));
    let sp = s.m.phi(head, s.t.i32);
    let best: Words = std::array::from_fn(|_| s.m.phi(head, s.t.i32));
    let more = s.m.cmp(head, pred::NE, sp, zero);
    s.m.terminate(head, Term::CondBr(more, pop, exit));
    let top = s.m.bin(pop, Bin::Sub, sp, one);
    let ptr = stack_at(s, pop, scene.stack, scene.base, top);
    let ptr = s.m.load(pop, ptr);
    let k7 = s.m.i32(7);
    let kind = s.m.bin(pop, Bin::And, ptr, k7);
    let clear = s.m.i32(!7);
    let off = s.m.bin(pop, Bin::And, ptr, clear);
    let shift = s.m.i32(3);
    let byte = s.m.bin(pop, Bin::Shl, off, shift);
    let k5 = s.m.i32(5);
    let is_box = s.m.cmp(pop, pred::EQ, kind, k5);
    s.m.terminate(pop, Term::CondBr(is_box, box_node, leaf_node));
    let children = integers::<4>(s, box_node, scene.nodes, byte);
    let mut coords = Vec::with_capacity(24);
    for i in 0..6 {
        let p = add_const(s, box_node, byte, 16 + i * 16);
        coords.extend(load::<4>(s, box_node, scene.nodes, p));
    }
    let extent = s.m.cast(box_node, Cast::Bitcast, best[1], s.t.f32);
    let r = NodeRay {
        extent,
        origin: ray.origin,
        direction: ray.direction,
        inverse,
    };
    let ids = NodeCode { s, cur: box_node }.box_hits(&coords, &children, &r, None, true);
    let mut next = top;
    for id in ids.into_iter().rev() {
        let dst = stack_at(s, box_node, scene.stack, scene.base, next);
        s.m.store(box_node, dst, id);
        let active = s.m.cmp(box_node, pred::NE, id, none);
        let push = s.m.cast(box_node, Cast::ZExt, active, s.t.i32);
        next = s.m.bin(box_node, Bin::Add, next, push);
    }
    s.m.terminate(box_node, Term::Br(head));
    let (leaf_end, hit) = leaf(s, leaf_node, byte, kind, best);
    s.m.terminate(leaf_end, Term::Br(head));
    s.m.set_phi(sp, vec![(count, start), (next, box_node), (top, leaf_end)]);
    for i in 0..HIT_WORDS {
        s.m.set_phi(
            best[i],
            vec![(initial[i], start), (best[i], box_node), (hit[i], leaf_end)],
        );
    }
    (exit, best)
}

fn transform(s: &mut Shader, b: usize, m: [[V; 4]; 3], v: [V; 3], point: bool) -> [V; 3] {
    m.map(|r| {
        let x = s.m.bin(b, Bin::Mul, r[0], v[0]);
        let y = s.m.bin(b, Bin::Mul, r[1], v[1]);
        let z = s.m.bin(b, Bin::Mul, r[2], v[2]);
        let xy = s.m.bin(b, Bin::Add, x, y);
        let xyz = s.m.bin(b, Bin::Add, xy, z);
        if point {
            s.m.bin(b, Bin::Add, xyz, r[3])
        } else {
            xyz
        }
    })
}

fn instance(
    s: &mut Shader,
    b: usize,
    scene: Scene,
    ray: Ray,
    byte: V,
    best: Words,
) -> (usize, Words) {
    let p = add_const(s, b, byte, 48);
    let [id_mask, contrib_flags, root, idx] = integers(s, b, scene.nodes, p);
    let k24 = s.m.i32(24);
    let mask = s.m.bin(b, Bin::LShr, id_mask, k24);
    let common = s.m.bin(b, Bin::And, mask, ray.mask);
    let zero = s.m.i32(0);
    let visible = s.m.cmp(b, pred::NE, common, zero);
    let enter = s.m.new_block();
    let join = s.m.new_block();
    s.m.terminate(b, Term::CondBr(visible, enter, join));
    let matrix = std::array::from_fn(|n| {
        let addr = add_const(s, enter, byte, n as u32 * 16);
        load::<4>(s, enter, scene.nodes, addr)
    });
    let local = Ray {
        origin: transform(s, enter, matrix, ray.origin, true),
        direction: transform(s, enter, matrix, ray.direction, false),
        ..ray
    };
    let mask24 = s.m.i32(0xff_ffff);
    let id = s.m.bin(enter, Bin::And, id_mask, mask24);
    let contribution = s.m.bin(enter, Bin::And, contrib_flags, mask24);
    let flag = s.m.i32(1 << 25);
    let flag = s.m.bin(enter, Bin::And, contrib_flags, flag);
    let ccw = s.m.cmp(enter, pred::NE, flag, zero);
    let one = s.m.f32(1f32.to_bits());
    let inverse = local.direction.map(|d| s.m.bin(enter, Bin::SDiv, one, d));
    let (end, hit) = walk(
        s,
        enter,
        scene,
        root,
        local,
        best,
        |s, b, byte, kind, best| {
            let r = NodeRay {
                origin: local.origin,
                direction: local.direction,
                inverse,
                extent: local.t_max,
            };
            let at = NodeAt::Buffer {
                handle: scene.nodes,
                byte,
            };
            let raw = NodeCode { s, cur: b }.triangle(at, kind, &r);
            let [num, det, un, vn] = raw.map(|i| s.m.cast(b, Cast::Bitcast, i, s.t.f32));
            let t = s.m.bin(b, Bin::SDiv, num, det);
            let old = s.m.cast(b, Cast::Bitcast, best[1], s.t.f32);
            let inside = s.m.cmp(b, pred::OGT, t, local.t_min);
            let closer = s.m.cmp(b, pred::OLT, t, old);
            let take = s.m.bin(b, Bin::And, inside, closer);
            let p = add_const(s, b, byte, 48);
            let [geom, p0, p1] = integers(s, b, scene.nodes, p);
            let geometry = s.m.bin(b, Bin::And, geom, mask24);
            let first = s.m.cmp(b, pred::EQ, kind, zero);
            let primitive = s.m.select(b, first, p0, p1);
            let fzero = s.m.f32(0);
            let front = s.m.cmp(b, pred::OGT, det, fzero);
            let front = s.m.bin(b, Bin::Xor, front, ccw);
            let uv = [un, vn].map(|n| s.m.bin(b, Bin::SDiv, n, det));
            let current = hit_words(
                s,
                b,
                TriangleHit {
                    t,
                    uv,
                    instance: id,
                    index: idx,
                    contribution,
                    geometry,
                    primitive,
                    front,
                },
            );
            (
                b,
                std::array::from_fn(|i| s.m.select(b, take, current[i], best[i])),
            )
        },
    );
    s.m.terminate(end, Term::Br(join));
    let merged = std::array::from_fn(|i| {
        let v = s.m.phi(join, s.t.i32);
        s.m.set_phi(v, vec![(best[i], b), (hit[i], end)]);
        v
    });
    (join, merged)
}

/// Software traversal on the same rays and ordered TLAS inputs as RayQuery.
/// t1 is the BVH4 buffer built by `rt_packet::bvh4::pack`.
pub fn software_shader(tlas_count: usize) -> Result<HostShader, ShaderError> {
    tlas_slots(tlas_count)?;
    let mut s = Shader::new(false);
    let io = inputs(&mut s);
    let nodes = s.root_srv_handle(1);
    s.lds_dwords = GROUP_SIZE * DEPTH * 2;
    let stack = s.lds();
    let zero = s.m.i32(0);
    let lane = s.op(
        io.start,
        "flattenedThreadIdInGroup.i32",
        Attr::ReadNone,
        s.t.i32,
        96,
        &[],
    );
    let depth = s.m.i32(DEPTH);
    let top_base = s.m.bin(io.start, Bin::Mul, lane, depth);
    let bottom_base = add_const(&mut s, io.start, top_base, GROUP_SIZE * DEPTH);
    let top = Scene {
        nodes,
        stack,
        base: top_base,
    };
    let bottom = Scene {
        base: bottom_base,
        ..top
    };
    let mut initial = [zero; HIT_WORDS];
    initial[1] = s.m.cast(io.start, Cast::Bitcast, io.ray.t_max, s.t.i32);
    let mut start = io.start;
    for i in 0..tlas_count {
        let at = s.m.i32(4 * i as u32);
        let [root] = integers(&mut s, start, nodes, at);
        let (end, hit) = walk(
            &mut s,
            start,
            top,
            root,
            io.ray,
            initial,
            |s, b, byte, _, best| instance(s, b, bottom, io.ray, byte, best),
        );
        let found = s.m.cmp(end, pred::NE, hit[0], zero);
        let store = s.m.new_block();
        start = s.m.new_block();
        s.m.terminate(end, Term::CondBr(found, store, start));
        write(&mut s, store, &io, hit);
        s.m.terminate(store, Term::Ret);
    }
    write(&mut s, start, &io, [zero; HIT_WORDS]);
    s.m.terminate(start, Term::Ret);
    Ok(finish(s))
}
