//! SM 6.6 inline ray queries. State access is valid only after Proceed has
//! exposed a candidate or CommittedStatus has identified a triangle.
//! Adapted from maren-halv/VIctory-DXR @ cd87189 (GPL-3.0).

/// DXIL SM 6.6 `RayQuery` state query opcodes (`dx.op.rayQuery_State*`).
pub(super) mod rq_op {
    pub const COMMITTED_STATUS: u32 = 184;
    pub const CANDIDATE_TYPE: u32 = 185;
    pub const CANDIDATE_OBJECT_TO_WORLD_3X4: u32 = 186;
    pub const COMMITTED_OBJECT_TO_WORLD_3X4: u32 = 188;
    pub const COMMITTED_WORLD_TO_OBJECT_3X4: u32 = 189;
    pub const CANDIDATE_TRIANGLE_FRONT_FACE: u32 = 191;
    pub const COMMITTED_TRIANGLE_FRONT_FACE: u32 = 192;
    pub const CANDIDATE_TRIANGLE_BARYCENTRICS: u32 = 193;
    pub const COMMITTED_TRIANGLE_BARYCENTRICS: u32 = 194;
    pub const CANDIDATE_TRIANGLE_RAY_T: u32 = 199;
    pub const COMMITTED_RAY_T: u32 = 200;
    pub const CANDIDATE_INSTANCE_INDEX: u32 = 201;
    pub const CANDIDATE_INSTANCE_ID: u32 = 202;
    pub const CANDIDATE_GEOMETRY_INDEX: u32 = 203;
    pub const CANDIDATE_PRIMITIVE_INDEX: u32 = 204;
    pub const COMMITTED_INSTANCE_INDEX: u32 = 207;
    pub const COMMITTED_INSTANCE_ID: u32 = 208;
    pub const COMMITTED_GEOMETRY_INDEX: u32 = 209;
    pub const COMMITTED_PRIMITIVE_INDEX: u32 = 210;
    pub const CANDIDATE_INSTANCE_CONTRIBUTION: u32 = 214;
    pub const COMMITTED_INSTANCE_CONTRIBUTION: u32 = 215;
}

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

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct RayQuery(V);

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
        self.integer(s, b, rq_op::CANDIDATE_TYPE)
    }

    pub fn status(self, s: &mut Shader, b: usize) -> V {
        self.integer(s, b, rq_op::COMMITTED_STATUS)
    }

    pub(super) fn integer(self, s: &mut Shader, b: usize, opcode: u32) -> V {
        s.op(
            b,
            "rayQuery_StateScalar.i32",
            Attr::ReadOnly,
            s.t.i32,
            opcode,
            &[self.0],
        )
    }

    pub(super) fn scalar_bit(self, s: &mut Shader, b: usize, op: u32) -> V {
        s.op(
            b,
            "rayQuery_StateScalar.i1",
            Attr::ReadOnly,
            s.t.i1,
            op,
            &[self.0],
        )
    }

    pub(super) fn scalar_float(self, s: &mut Shader, b: usize, op: u32) -> V {
        s.op(
            b,
            "rayQuery_StateScalar.f32",
            Attr::ReadOnly,
            s.t.f32,
            op,
            &[self.0],
        )
    }

    pub(super) fn vector(self, s: &mut Shader, b: usize, op: u32, component: u8) -> V {
        let c = s.m.i8(component);
        s.op(
            b,
            "rayQuery_StateVector.f32",
            Attr::ReadOnly,
            s.t.f32,
            op,
            &[self.0, c],
        )
    }

    pub(super) fn matrix(self, s: &mut Shader, b: usize, op: u32, row: u32, col: u8) -> V {
        let r = s.m.i32(row);
        let c = s.m.i8(col);
        s.op(
            b,
            "rayQuery_StateMatrix.f32",
            Attr::ReadOnly,
            s.t.f32,
            op,
            &[self.0, r, c],
        )
    }

    pub fn triangle(self, s: &mut Shader, b: usize, state: Intersection) -> TriangleHit {
        use rq_op::*;
        let (t, uv, id, index, contribution, geometry, primitive, front) = match state {
            Intersection::Candidate => (
                CANDIDATE_TRIANGLE_RAY_T,
                CANDIDATE_TRIANGLE_BARYCENTRICS,
                CANDIDATE_INSTANCE_ID,
                CANDIDATE_INSTANCE_INDEX,
                CANDIDATE_INSTANCE_CONTRIBUTION,
                CANDIDATE_GEOMETRY_INDEX,
                CANDIDATE_PRIMITIVE_INDEX,
                CANDIDATE_TRIANGLE_FRONT_FACE,
            ),
            Intersection::Committed => (
                COMMITTED_RAY_T,
                COMMITTED_TRIANGLE_BARYCENTRICS,
                COMMITTED_INSTANCE_ID,
                COMMITTED_INSTANCE_INDEX,
                COMMITTED_INSTANCE_CONTRIBUTION,
                COMMITTED_GEOMETRY_INDEX,
                COMMITTED_PRIMITIVE_INDEX,
                COMMITTED_TRIANGLE_FRONT_FACE,
            ),
        };
        TriangleHit {
            t: self.scalar_float(s, b, t),
            uv: std::array::from_fn(|n| self.vector(s, b, uv, n as u8)),
            instance: self.integer(s, b, id),
            index: self.integer(s, b, index),
            contribution: self.integer(s, b, contribution),
            geometry: self.integer(s, b, geometry),
            primitive: self.integer(s, b, primitive),
            front: self.scalar_bit(s, b, front),
        }
    }
}
