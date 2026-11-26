//! SM 6.6 inline ray queries. State access is valid only after Proceed has
//! exposed a candidate or CommittedStatus has identified a triangle.
use super::bitcode::{Attr, V};
use super::shader::{Shader, flags};

#[derive(Clone, Copy)]
pub(super) struct Ray {
    pub origin: [V; 3],
    pub t_min: V,
    pub direction: [V; 3],
    pub t_max: V,
    pub mask: V,
}

#[derive(Clone, Copy)]
pub(super) struct RayQuery(V);

pub(super) enum Intersection {
    Candidate,
    Committed,
}

pub(super) struct TriangleHit {
    pub t: V,
    pub uv: [V; 2],
    pub instance: V,
    pub index: V,
    pub contribution: V,
    pub geometry: V,
    pub primitive: V,
    pub front: V,
}

impl RayQuery {
    pub fn allocate(s: &mut Shader, flags: u32) -> Self {
        s.flags |= flags::RAYTRACING_1_1;
        let flags = s.m.i32(flags);
        Self(s.op(
            s.entry,
            "allocateRayQuery",
            Attr::NoUnwind,
            s.t.i32,
            178,
            &[flags],
        ))
    }

    pub fn trace(self, s: &mut Shader, b: usize, tlas: V, flags: V, ray: Ray) {
        let [ox, oy, oz] = ray.origin;
        let [dx, dy, dz] = ray.direction;
        s.op(
            b,
            "rayQuery_TraceRayInline",
            Attr::NoUnwind,
            s.t.void,
            179,
            &[
                self.0, tlas, flags, ray.mask, ox, oy, oz, ray.t_min, dx, dy, dz, ray.t_max,
            ],
        );
    }

    pub fn proceed(self, s: &mut Shader, b: usize) -> V {
        s.op(
            b,
            "rayQuery_Proceed.i1",
            Attr::NoUnwind,
            s.t.i1,
            180,
            &[self.0],
        )
    }

    pub fn abort(self, s: &mut Shader, b: usize) {
        s.op(
            b,
            "rayQuery_Abort",
            Attr::NoUnwind,
            s.t.void,
            181,
            &[self.0],
        );
    }

    pub fn commit_triangle(self, s: &mut Shader, b: usize) {
        s.op(
            b,
            "rayQuery_CommitNonOpaqueTriangleHit",
            Attr::NoUnwind,
            s.t.void,
            182,
            &[self.0],
        );
    }

    pub fn candidate_type(self, s: &mut Shader, b: usize) -> V {
        self.integer(s, b, 185)
    }

    pub fn status(self, s: &mut Shader, b: usize) -> V {
        self.integer(s, b, 184)
    }

    fn integer(self, s: &mut Shader, b: usize, opcode: u32) -> V {
        s.op(
            b,
            "rayQuery_StateScalar.i32",
            Attr::ReadOnly,
            s.t.i32,
            opcode,
            &[self.0],
        )
    }

    pub fn triangle(self, s: &mut Shader, b: usize, state: Intersection) -> TriangleHit {
        let (t, uv, id, index, contribution, geometry, primitive, front) = match state {
            Intersection::Candidate => (199, 193, 202, 201, 214, 203, 204, 191),
            Intersection::Committed => (200, 194, 208, 207, 215, 209, 210, 192),
        };
        TriangleHit {
            t: s.op(
                b,
                "rayQuery_StateScalar.f32",
                Attr::ReadOnly,
                s.t.f32,
                t,
                &[self.0],
            ),
            uv: std::array::from_fn(|n| {
                let c = s.m.i8(n as u8);
                s.op(
                    b,
                    "rayQuery_StateVector.f32",
                    Attr::ReadOnly,
                    s.t.f32,
                    uv,
                    &[self.0, c],
                )
            }),
            instance: self.integer(s, b, id),
            index: self.integer(s, b, index),
            contribution: self.integer(s, b, contribution),
            geometry: self.integer(s, b, geometry),
            primitive: self.integer(s, b, primitive),
            front: s.op(
                b,
                "rayQuery_StateScalar.i1",
                Attr::ReadOnly,
                s.t.i1,
                front,
                &[self.0],
            ),
        }
    }
}
