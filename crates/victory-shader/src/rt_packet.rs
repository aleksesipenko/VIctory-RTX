//! Narrow triangle replay v1. The envelope is eight magic bytes, u32 version,
//! u64 payload size, 32 BLAKE3 bytes over the payload, then the payload.
//! All numbers are little-endian. Lists begin with a u32 element count.
//! No native structs, pointers, alignment padding or host GPU addresses.
//!
//! Payload order: BLAS list (geometry list: opacity u32, triangle list of
//! 9 f32), ordered TLAS list (instance list: BLAS ordinal, ID, mask,
//! contribution, front-CCW u32, row-major object-to-world 3x4 f32),
//! ray list (origin, tMin, direction, tMax, mask, acceptable closest hits).
//! A hit stores f64 t/u/v then six u32: ID, TLAS-local instance index,
//! contribution, geometry, primitive, front. Zero hits = miss; multiple
//! hits identify a tie, with each candidate's own attributes. Only a miss
//! falls through to the next TLAS. Replay reconstructs all GPU addresses.
pub mod bvh4;

use std::io::{Error, ErrorKind, Result};

pub type Point = [f32; 3];
pub type Triangle = [Point; 3];
pub type Transform = [[f32; 4]; 3];
pub type Blas = Vec<Geometry>;
pub type Tlas = Vec<Instance>;
const MAGIC: &[u8; 8] = b"VRTNARR\0";
pub const VERSION: u32 = 1;

#[derive(Clone, Debug, PartialEq)]
pub struct Geometry {
    pub opaque: bool,
    pub triangles: Vec<Triangle>,
}
#[derive(Clone, Debug, PartialEq)]
pub struct Instance {
    pub blas: u32,
    pub id: u32,
    pub mask: u8,
    pub contribution: u32,
    pub front_ccw: bool,
    pub transform: Transform,
}
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Ray {
    pub origin: Point,
    pub t_min: f32,
    pub direction: Point,
    pub t_max: f32,
    pub mask: u8,
}
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Hit {
    pub t: f64,
    pub u: f64,
    pub v: f64,
    pub id: u32,
    pub instance: u32,
    pub contribution: u32,
    pub geometry: u32,
    pub primitive: u32,
    pub front: bool,
}
#[derive(Clone, Debug, PartialEq)]
pub struct Packet {
    pub blases: Vec<Blas>,
    pub tlases: Vec<Tlas>,
    pub rays: Vec<(Ray, Vec<Hit>)>,
}

fn invalid(msg: &'static str) -> Error {
    Error::new(ErrorKind::InvalidData, msg)
}

struct Reader<'a>(&'a [u8]);
impl Reader<'_> {
    fn take<const N: usize>(&mut self) -> Result<[u8; N]> {
        let (a, b) = self
            .0
            .split_at_checked(N)
            .ok_or_else(|| invalid("truncated RT packet"))?;
        self.0 = b;
        Ok(a.try_into().unwrap())
    }
    fn word(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.take()?))
    }
    fn float(&mut self) -> Result<f32> {
        Ok(f32::from_bits(self.word()?))
    }
    fn point(&mut self) -> Result<Point> {
        Ok([self.float()?, self.float()?, self.float()?])
    }
    fn row(&mut self) -> Result<[f32; 4]> {
        let [x, y, z] = self.point()?;
        Ok([x, y, z, self.float()?])
    }
    fn mask(&mut self) -> Result<u8> {
        self.word()?
            .try_into()
            .map_err(|_| invalid("instance/ray mask"))
    }
    fn flag(&mut self) -> Result<bool> {
        match self.word()? {
            0 => Ok(false),
            1 => Ok(true),
            _ => Err(invalid("invalid boolean")),
        }
    }
    fn list<T>(
        &mut self,
        min_bytes: usize,
        mut item: impl FnMut(&mut Self) -> Result<T>,
    ) -> Result<Vec<T>> {
        let n = self.word()? as usize;
        if n > self.0.len() / min_bytes {
            return Err(invalid("list extends past packet"));
        }
        (0..n).map(|_| item(self)).collect()
    }
}
struct Writer(Vec<u8>);
impl Writer {
    fn word(&mut self, x: u32) {
        self.0.extend(x.to_le_bytes());
    }
    fn float(&mut self, x: f32) {
        self.word(x.to_bits());
    }
    fn count(&mut self, n: usize) -> Result<()> {
        self.word(n.try_into().map_err(|_| invalid("list exceeds u32"))?);
        Ok(())
    }
}

impl Packet {
    pub fn validate(&self) -> Result<()> {
        if self.blases.is_empty() || !(1..=3).contains(&self.tlases.len()) || self.rays.is_empty() {
            return Err(invalid("need BLAS, 1..=3 TLAS and rays"));
        }
        for b in &self.blases {
            if b.is_empty() || b.len() > 0x100_0000 {
                return Err(invalid("geometry count"));
            }
            for g in b {
                if g.triangles.is_empty()
                    || !g
                        .triangles
                        .iter()
                        .flatten()
                        .flatten()
                        .all(|v| v.is_finite())
                {
                    return Err(invalid("empty or non-finite geometry"));
                }
            }
        }
        for i in self.tlases.iter().flatten() {
            let [a, b, c] = i.transform.map(|row| row.map(f64::from));
            let det = a[0] * (b[1] * c[2] - b[2] * c[1]) - a[1] * (b[0] * c[2] - b[2] * c[0])
                + a[2] * (b[0] * c[1] - b[1] * c[0]);
            if i.blas as usize >= self.blases.len()
                || i.id > 0xff_ffff
                || i.contribution > 0xff_ffff
                || !i.transform.iter().flatten().all(|v| v.is_finite())
                || det == 0.0
            {
                return Err(invalid(
                    "invalid instance reference, bit field or transform",
                ));
            }
        }
        for (r, hits) in &self.rays {
            if !r.origin.iter().chain(&r.direction).all(|v| v.is_finite())
                || r.direction == [0.; 3]
                || !(0. <= r.t_min && r.t_min <= r.t_max)
            {
                return Err(invalid("invalid ray or extent"));
            }
            if hits
                .iter()
                .any(|h| !h.t.is_finite() || !h.u.is_finite() || !h.v.is_finite())
            {
                return Err(invalid("non-finite expected hit"));
            }
        }
        Ok(())
    }

    pub fn encode(&self) -> Result<Vec<u8>> {
        self.validate()?;
        let mut w = Writer(Vec::new());
        w.count(self.blases.len())?;
        for b in &self.blases {
            w.count(b.len())?;
            for g in b {
                w.word(g.opaque as u32);
                w.count(g.triangles.len())?;
                for &v in g.triangles.iter().flatten().flatten() {
                    w.float(v);
                }
            }
        }
        w.count(self.tlases.len())?;
        for t in &self.tlases {
            w.count(t.len())?;
            for i in t {
                for v in [
                    i.blas,
                    i.id,
                    i.mask.into(),
                    i.contribution,
                    i.front_ccw as u32,
                ] {
                    w.word(v);
                }
                for v in i.transform.into_iter().flatten() {
                    w.float(v);
                }
            }
        }
        w.count(self.rays.len())?;
        for (r, hits) in &self.rays {
            for f in r
                .origin
                .into_iter()
                .chain([r.t_min])
                .chain(r.direction)
                .chain([r.t_max])
            {
                w.float(f);
            }
            w.word(r.mask.into());
            w.count(hits.len())?;
            for h in hits {
                for v in [h.t, h.u, h.v] {
                    w.0.extend(v.to_le_bytes());
                }
                for v in [
                    h.id,
                    h.instance,
                    h.contribution,
                    h.geometry,
                    h.primitive,
                    h.front as u32,
                ] {
                    w.word(v);
                }
            }
        }
        let mut out = Vec::with_capacity(52 + w.0.len());
        out.extend(MAGIC);
        out.extend(VERSION.to_le_bytes());
        out.extend((w.0.len() as u64).to_le_bytes());
        out.extend(blake3::hash(&w.0).as_bytes());
        out.extend(w.0);
        Ok(out)
    }

    pub fn decode(bytes: &[u8]) -> Result<Self> {
        let mut r = Reader(bytes);
        if &r.take::<8>()? != MAGIC || r.word()? != VERSION {
            return Err(invalid("RT packet magic/version"));
        }
        let len = u64::from_le_bytes(r.take()?);
        let hash = r.take::<32>()?;
        if len != r.0.len() as u64 || hash != *blake3::hash(r.0).as_bytes() {
            return Err(invalid("RT packet length/checksum"));
        }
        let blases = r.list(4, |r| {
            r.list(8, |r| {
                Ok(Geometry {
                    opaque: r.flag()?,
                    triangles: r.list(36, |r| Ok([r.point()?, r.point()?, r.point()?]))?,
                })
            })
        })?;
        let tlases = r.list(4, |r| {
            r.list(68, |r| {
                Ok(Instance {
                    blas: r.word()?,
                    id: r.word()?,
                    mask: r.mask()?,
                    contribution: r.word()?,
                    front_ccw: r.flag()?,
                    transform: [r.row()?, r.row()?, r.row()?],
                })
            })
        })?;
        let rays = r.list(40, |r| {
            let ray = Ray {
                origin: r.point()?,
                t_min: r.float()?,
                direction: r.point()?,
                t_max: r.float()?,
                mask: r.mask()?,
            };
            let hits = r.list(48, |r| {
                Ok(Hit {
                    t: f64::from_le_bytes(r.take()?),
                    u: f64::from_le_bytes(r.take()?),
                    v: f64::from_le_bytes(r.take()?),
                    id: r.word()?,
                    instance: r.word()?,
                    contribution: r.word()?,
                    geometry: r.word()?,
                    primitive: r.word()?,
                    front: r.flag()?,
                })
            })?;
            Ok((ray, hits))
        })?;
        if !r.0.is_empty() {
            return Err(invalid("trailing RT packet payload"));
        }
        let p = Self {
            blases,
            tlases,
            rays,
        };
        p.validate()?;
        Ok(p)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> Packet {
        Packet {
            blases: vec![vec![Geometry {
                opaque: true,
                triangles: vec![[[0., 0., 3.], [1., 0., 3.], [0., 1., 3.]]],
            }]],
            tlases: vec![vec![Instance {
                blas: 0,
                id: 0xabc,
                mask: 0xd5,
                contribution: 7,
                front_ccw: false,
                transform: [[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.]],
            }]],
            rays: vec![(
                Ray {
                    origin: [0.125, 0.25, 0.],
                    t_min: 0.,
                    direction: [0., 0., 1.],
                    t_max: f32::INFINITY,
                    mask: 0xc0,
                },
                vec![Hit {
                    t: 3.,
                    u: 0.125,
                    v: 0.25,
                    id: 0xabc,
                    instance: 0,
                    contribution: 7,
                    geometry: 0,
                    primitive: 0,
                    front: false,
                }],
            )],
        }
    }

    #[test]
    fn v1_wire_layout_survives_relocation() {
        let p = fixture();
        let bytes = p.encode().unwrap();
        assert_eq!(bytes.len(), 272);
        assert_eq!(&bytes[..8], b"VRTNARR\0");
        assert_eq!(u64::from_le_bytes(bytes[12..20].try_into().unwrap()), 220);
        // Fixed wire offsets, independent of the decoder.
        let word = |at| u32::from_le_bytes(bytes[at..at + 4].try_into().unwrap());
        assert_eq!(
            [word(8), word(52), word(60), word(76), word(120)],
            [1, 1, 1, 3f32.to_bits(), 0xd5]
        );
        assert_eq!(Packet::decode(&bytes).unwrap(), p);
    }

    #[test]
    fn corrupted_truncated_and_unknown_packets_are_refused() {
        let bytes = fixture().encode().unwrap();
        for n in [0, 7, 12, 51, bytes.len() - 1] {
            assert!(Packet::decode(&bytes[..n]).is_err());
        }
        for at in [0, 8, 12, 20, bytes.len() - 1] {
            let mut corrupt = bytes.clone();
            corrupt[at] ^= 1;
            assert!(Packet::decode(&corrupt).is_err());
        }
        let mut extra = bytes.clone();
        extra.push(0);
        assert!(Packet::decode(&extra).is_err());
        // A valid checksum does not make an impossible count a valid allocation.
        let mut huge = bytes;
        huge[52..56].copy_from_slice(&u32::MAX.to_le_bytes());
        let hash = blake3::hash(&huge[52..]);
        huge[20..52].copy_from_slice(hash.as_bytes());
        assert!(Packet::decode(&huge).is_err());
    }

    #[test]
    fn invalid_blas_reference_and_singular_transform_fail_before_replay() {
        let mut p = fixture();
        p.tlases[0][0].blas = 1;
        assert!(p.encode().is_err());
        p.tlases[0][0].blas = 0;
        p.tlases[0][0].transform[0] = [0.; 4];
        assert!(p.encode().is_err());
    }
}
