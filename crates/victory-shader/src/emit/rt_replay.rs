//! Standalone compute kernels for the narrow RT replay. Inputs have no
//! process GPU addresses. t0 is a ray buffer (16-byte header with count,
//! then 48 bytes/ray: origin/tMin, direction/tMax, mask and padding).
//! u0 receives ten dwords/ray: status, t, u, v, ID, instance index,
//! contribution, geometry index, primitive index, front face. All are in space 1.
mod software;
pub use software::software_shader;

use super::HostShader;
use super::bitcode::{Attr, Bin, Cast, Term, V, pred};
use super::ray_query::{Intersection, Ray, RayQuery, TriangleHit};
use super::shader::{Shader, TLAS_SLOTS};
use crate::ShaderError;
use crate::native::HwStage;

pub const GROUP_SIZE: u32 = 32;
pub const HIT_WORDS: usize = 10;

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum QueryMode {
    Closest,
    /// Diagnostic: expose the first triangle as a non-opaque candidate,
    /// write its candidate attributes, commit it and terminate traversal.
    FirstCandidate,
}

fn add_const(s: &mut Shader, b: usize, base: V, value: u32) -> V {
    let k = s.m.i32(value);
    s.m.bin(b, Bin::Add, base, k)
}

fn load<const N: usize>(s: &mut Shader, b: usize, handle: V, byte: V) -> [V; N] {
    assert!((1..=4).contains(&N));
    let undef = s.m.undef(s.t.i32);
    let mask = s.m.i8((1 << N) - 1);
    let align = s.m.i32(4);
    let value = s.op(
        b,
        "rawBufferLoad.f32",
        Attr::ReadOnly,
        s.t.res_ret,
        139,
        &[handle, byte, undef, mask, align],
    );
    std::array::from_fn(|i| s.m.extract(b, value, i as u32))
}

fn integers<const N: usize>(s: &mut Shader, b: usize, h: V, byte: V) -> [V; N] {
    load(s, b, h, byte).map(|v| s.m.cast(b, Cast::Bitcast, v, s.t.i32))
}

struct Io {
    ray: Ray,
    out: V,
    dest: V,
    start: usize,
}

fn inputs(s: &mut Shader) -> Io {
    s.threads = Some([GROUP_SIZE, 1, 1]);
    let b = s.entry;
    let zero = s.m.i32(0);
    let rays = s.root_srv_handle(0);
    let out = s.root_uav_handle(0);
    let tid = s.op(b, "threadId.i32", Attr::ReadNone, s.t.i32, 93, &[zero]);
    let [count] = integers(s, b, rays, zero);
    let active = s.m.cmp(b, pred::IULT, tid, count);
    let start = s.m.new_block();
    let end = s.m.new_block();
    s.m.terminate(b, Term::CondBr(active, start, end));
    s.m.terminate(end, Term::Ret);
    let b = start;
    let stride = s.m.i32(48);
    let base = s.m.bin(b, Bin::Mul, tid, stride);
    let p = add_const(s, b, base, 16);
    let [ox, oy, oz, t_min] = load(s, b, rays, p);
    let p = add_const(s, b, base, 32);
    let [dx, dy, dz, t_max] = load(s, b, rays, p);
    let p = add_const(s, b, base, 48);
    let [mask] = integers(s, b, rays, p);
    let stride = s.m.i32(4 * HIT_WORDS as u32);
    let dest = s.m.bin(b, Bin::Mul, tid, stride);
    Io {
        ray: Ray {
            origin: [ox, oy, oz],
            t_min,
            direction: [dx, dy, dz],
            t_max,
            mask,
        },
        out,
        dest,
        start,
    }
}

fn write(s: &mut Shader, b: usize, io: &Io, words: [V; HIT_WORDS]) {
    let undef = s.m.undef(s.t.i32);
    let align = s.m.i32(4);
    for (i, chunk) in words.chunks(4).enumerate() {
        let p = add_const(s, b, io.dest, i as u32 * 16);
        let mask = s.m.i8((1 << chunk.len()) - 1);
        let mut v = [undef; 4];
        v[..chunk.len()].copy_from_slice(chunk);
        s.op(
            b,
            "rawBufferStore.i32",
            Attr::NoUnwind,
            s.t.void,
            140,
            &[io.out, p, undef, v[0], v[1], v[2], v[3], mask, align],
        );
    }
}

fn hit_words(s: &mut Shader, b: usize, h: TriangleHit) -> [V; HIT_WORDS] {
    let [t, u, v] = [h.t, h.uv[0], h.uv[1]].map(|f| s.m.cast(b, Cast::Bitcast, f, s.t.i32));
    let front = s.m.cast(b, Cast::ZExt, h.front, s.t.i32);
    [
        s.m.i32(1),
        t,
        u,
        v,
        h.instance,
        h.index,
        h.contribution,
        h.geometry,
        h.primitive,
        front,
    ]
}

fn finish(s: Shader) -> HostShader {
    HostShader {
        stage: HwStage::Cs,
        root_srvs: s.root_srvs.iter().map(|x| x.0).collect(),
        root_uavs: s.root_uavs.iter().map(|x| x.0).collect(),
        dxil: s.finish(),
        user_data: vec![],
        samplers: vec![],
        static_samplers: vec![],
        root_cbvs: vec![],
        root_indexed_views: vec![],
    }
}

fn tlas_slots(n: usize) -> Result<&'static [u8], ShaderError> {
    if (1..=TLAS_SLOTS.len()).contains(&n) {
        Ok(&TLAS_SLOTS[..n])
    } else {
        Err(ShaderError::Malformed(
            "RT replay needs one to three TLAS inputs",
        ))
    }
}

/// Closest within each TLAS. Only a miss falls through to the next TLAS.
pub fn ray_query_shader(tlas_count: usize, mode: QueryMode) -> Result<HostShader, ShaderError> {
    let slots = tlas_slots(tlas_count)?;
    let mut s = Shader::new(false);
    let io = inputs(&mut s);
    let zero = s.m.i32(0);
    // Geometry in a v1 packet is triangles only. Never accept the first
    // hit in the closest path. Force non-opaque only for the diagnostic.
    let flags = if mode == QueryMode::FirstCandidate {
        2
    } else {
        0x200
    };
    let rq = RayQuery::allocate(&mut s, flags);
    let mut start = io.start;
    for &slot in slots {
        let tlas = s.root_srv_handle(slot);
        rq.trace(&mut s, start, tlas, zero, io.ray);
        let step = s.m.new_block();
        let candidate = s.m.new_block();
        let done = s.m.new_block();
        s.m.terminate(start, Term::Br(step));
        let more = rq.proceed(&mut s, step);
        s.m.terminate(step, Term::CondBr(more, candidate, done));
        if mode == QueryMode::FirstCandidate {
            let kind = rq.candidate_type(&mut s, candidate);
            let triangle = s.m.cmp(candidate, pred::EQ, kind, zero);
            let inspect = s.m.new_block();
            s.m.terminate(candidate, Term::CondBr(triangle, inspect, step));
            let candidate = inspect;
            let h = rq.triangle(&mut s, candidate, Intersection::Candidate);
            let words = hit_words(&mut s, candidate, h);
            rq.commit_triangle(&mut s, candidate);
            rq.abort(&mut s, candidate);
            write(&mut s, candidate, &io, words);
            s.m.terminate(candidate, Term::Ret);
        } else {
            rq.commit_triangle(&mut s, candidate);
            s.m.terminate(candidate, Term::Br(step));
        }
        let status = rq.status(&mut s, done);
        let one = s.m.i32(1);
        let hit = s.m.cmp(done, pred::EQ, status, one);
        let found = s.m.new_block();
        start = s.m.new_block();
        s.m.terminate(done, Term::CondBr(hit, found, start));
        // Other committed accessors are undefined on a miss.
        let h = rq.triangle(&mut s, found, Intersection::Committed);
        let words = hit_words(&mut s, found, h);
        write(&mut s, found, &io, words);
        s.m.terminate(found, Term::Ret);
    }
    write(&mut s, start, &io, [zero; HIT_WORDS]);
    s.m.terminate(start, Term::Ret);
    Ok(finish(s))
}
