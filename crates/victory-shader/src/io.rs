//! I/O raising (decision 0003): what a program's input registers, its
//! interpolation and its exports mean.
//!
//! Pixel shaders get the VGPRs `SPI_PS_INPUT_ADDR` enables, in a fixed
//! order (barycentrics, then position, front face...). `v_interp_p1` and
//! `v_interp_p2` (and their f16 forms) evaluate an attribute from a pair of
//! barycentrics; `v_interp_mov` reads one vertex's value. After the 32-bit
//! export patch ([`NativeProgram::patched_code`]) exports carry plain
//! 32-bit values: `mrt<n>` is render target n, `mrtz` depth, stencil and
//! sample mask, `null` nothing. A lane whose EXEC bit is clear at the
//! export that ends the program (`done`) is discarded: that export's guard
//! after structurisation.
//!
//! Vertex shaders that run as the hardware VS get the vertex id in v0, the
//! primitive id in v2 and the instance id in v3; the fetch shader loads
//! channel k of each attribute into its VGPR + k. `exp pos0` is the
//! position, `exp param<n>` the output with parameter n.
//!
//! Attributes and outputs are named by their row: a register of the
//! container's signature part.
//!
//! Compute programs get the thread id in the group in v0-v2 and the group
//! id in the SGPRs after the user data, as `COMPUTE_PGM_RSRC2` enables them.
//!
//! Mesh programs run as one NGG stage. The program computes its thread
//! index from the wave index (s3 bits 24-27) and the lane (`v_mbcnt`), and
//! its group id from v5 of the first lane (`v_readfirstlane`: the group
//! times the group size). Lane n exports vertex n (`exp pos`, `exp param`)
//! and primitive n (`exp prim`: 9-bit vertex indices at bits 0, 10, 20,
//! each with an edge flag above it, bit 31 culls); `param` entries with
//! bit 7 are per-primitive outputs. One lane sends the output counts
//! (`s_sendmsg` GS_ALLOC_REQ, m0 = vertices | primitives << 12).

use std::collections::HashMap;

use crate::ShaderError;
use crate::gfx10::{Fields, Operand};
use crate::ir::{EXEC, Exit, Inst, Program, Reg, Value, operand_reads};
use crate::native::{HwStage, NativeProgram, PsInput, StageIo, Usage};
use crate::resource::Site;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Input {
    VertexId,
    PrimitiveId,
    InstanceId,
    /// A vertex attribute the fetch shader loads.
    Attribute {
        row: u8,
        channel: u8,
    },
    /// `index` 0 is i, 1 is j (2 the pull model's w).
    Barycentric {
        kind: Barycentric,
        index: u8,
    },
    /// Pixel position channel (0-3: x, y, z, w).
    Position(u8),
    FrontFace,
    /// Sample id and other bits.
    Ancillary,
    SampleCoverage,
    FixedPosition,
    LineStipple,
    /// Compute: thread-group id component (an SGPR).
    GroupId(u8),
    /// Compute: thread id within the group, component (a VGPR).
    ThreadIdInGroup(u8),
    /// Compute: the thread-group info SGPR (wave index and count).
    GroupInfo,
    /// Mesh: v5, the group id times the group size (read as uniform).
    MeshGroup,
    /// Mesh: s3, the wave's index in the group in bits 24-27.
    MergedWaveInfo,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Barycentric {
    PerspectiveSample,
    PerspectiveCenter,
    PerspectiveCentroid,
    PerspectivePullModel,
    LinearSample,
    LinearCenter,
    LinearCentroid,
}

/// The pixel input VGPRs in hardware order: one entry per
/// `SPI_PS_INPUT_ADDR` bit.
const PS_VGPRS: [(u8, Option<Barycentric>); 16] = [
    (2, Some(Barycentric::PerspectiveSample)),
    (2, Some(Barycentric::PerspectiveCenter)),
    (2, Some(Barycentric::PerspectiveCentroid)),
    (3, Some(Barycentric::PerspectivePullModel)),
    (2, Some(Barycentric::LinearSample)),
    (2, Some(Barycentric::LinearCenter)),
    (2, Some(Barycentric::LinearCentroid)),
    (1, None), // line stipple
    (1, None), // position x
    (1, None), // y
    (1, None), // z
    (1, None), // w
    (1, None), // front face
    (1, None), // ancillary
    (1, None), // sample coverage
    (1, None), // fixed-point position
];

fn ps_system(bit: usize) -> Input {
    match bit {
        7 => Input::LineStipple,
        8..=11 => Input::Position(bit as u8 - 8),
        12 => Input::FrontFace,
        13 => Input::Ancillary,
        14 => Input::SampleCoverage,
        _ => Input::FixedPosition,
    }
}

/// An attribute channel of the input signature.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Attribute {
    pub usage: Usage,
    pub row: u8,
    pub channel: u8,
    /// The upper f16 half of the channel (packed fp16 inputs).
    pub high: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Interp {
    /// This site's result (`v_interp_p2*`) is the attribute interpolated
    /// with `barycentric`; `first` are the `v_interp_p1*` halves it
    /// completes (one per path where the compiler shares a first half).
    Interpolated {
        attribute: Attribute,
        barycentric: Barycentric,
        first: Vec<Site>,
    },
    /// One vertex's value (`v_interp_mov_f32`), as encoded: 0 for P10,
    /// 1 for P20, 2 for P0 (the provoking vertex).
    Vertex { attribute: Attribute, vertex: u8 },
    /// A half of an interpolation whose i or j is computed rather than
    /// loaded: each half evaluates as the hardware does, from the
    /// attribute's vertex values (`v_interp_p1*`: P0 + i P10, with P10 =
    /// P1 - P0; `v_interp_p2*`: its old dst + j P20, with P20 = P2 - P0).
    Weighted { attribute: Attribute },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Target {
    Color(u8),
    /// Depth, stencil, sample mask (channels as `DB_SHADER_CONTROL` enables).
    Depth,
    Null,
    Position(u8),
    Param {
        usage: Usage,
        row: u8,
    },
    /// `param0` of a vertex shader without outputs: the hardware wants one
    /// parameter export regardless.
    Unread,
    /// The primitive export of a merged or mesh stage.
    Primitive,
    /// A mesh program's per-primitive output (a row of `PSG1`).
    PrimitiveParam {
        usage: Usage,
        row: u8,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Export {
    pub site: Site,
    pub target: Target,
    /// Per channel, as the instruction enables them.
    pub values: [Option<Value>; 4],
    /// Two f16 values per dword.
    pub compressed: bool,
    /// The program's last export (for pixels: kills the lanes not in EXEC).
    pub done: bool,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Io {
    /// Input values the hardware loads, with their meaning; other VGPRs
    /// read before written hold undefined values.
    pub inputs: Vec<(Value, Input)>,
    pub interps: Vec<(Site, Interp)>,
    pub exports: Vec<Export>,
    /// Compute: how many SGPRs the hardware initializes
    /// ([`ComputeIo::initialized_sgprs`](crate::native::ComputeIo::initialized_sgprs)).
    pub initialized_sgprs: Option<u8>,
}

/// Raises the I/O of `p`, built from `native`'s patched code. Fails for
/// stages whose stage data is not read and on constructs not covered yet.
pub fn raise(p: &Program, native: &NativeProgram) -> Result<Io, ShaderError> {
    let unsupported = |offset, what| ShaderError::Unsupported { offset, what };
    let (loaded, ps_inputs, params) = match (&native.io, native.stage) {
        (
            StageIo::Ps {
                inputs, input_addr, ..
            },
            HwStage::Ps,
        ) => {
            // A pixel shader exports no parameters; `[]` never reaches `Unread`.
            let loaded = ps_vgprs(*input_addr)
                .into_iter()
                .map(|(r, i)| (Reg::V(r), i))
                .collect();
            (loaded, &inputs[..], &[][..])
        }
        (StageIo::Vs { inputs, outputs }, HwStage::Vs) => {
            let mut loaded = vec![
                (Reg::V(0), Input::VertexId),
                (Reg::V(2), Input::PrimitiveId),
                (Reg::V(3), Input::InstanceId),
            ];
            for i in inputs {
                for channel in (0..4).filter(|c| i.mask >> c & 1 != 0) {
                    let input = Input::Attribute {
                        row: i.row,
                        channel,
                    };
                    loaded.push((Reg::V(i.vgpr + channel), input));
                }
            }
            (loaded, &[][..], &outputs[..])
        }
        // Compute: thread ids in v0-v2, group ids and the group info in the
        // SGPRs after the user data.
        (StageIo::Cs(cs), HwStage::Cs) => {
            let (ids, info) = cs.system_sgprs();
            let mut loaded: Vec<(Reg, Input)> = (0..cs.thread_id_vgprs())
                .map(|c| (Reg::V(c), Input::ThreadIdInGroup(c)))
                .collect();
            for (c, s) in ids.into_iter().enumerate() {
                if let Some(s) = s {
                    loaded.push((Reg::S(s), Input::GroupId(c as u8)));
                }
            }
            loaded.extend(info.map(|s| (Reg::S(s), Input::GroupInfo)));
            (loaded, &[][..], &[][..])
        }
        (StageIo::Ms(ms), HwStage::Ms) => {
            if let Some(offset) = mesh_group_per_lane_read(p) {
                return Err(unsupported(offset, "the mesh group input read per lane"));
            }
            let loaded = vec![
                (Reg::V(5), Input::MeshGroup),
                (Reg::S(3), Input::MergedWaveInfo),
            ];
            (loaded, &[][..], &ms.outputs[..])
        }
        _ => return Err(unsupported(0, "I/O of this stage")),
    };
    let inputs: Vec<(Value, Input)> = p
        .inputs
        .iter()
        .filter_map(|&(reg, v)| loaded.iter().find(|l| l.0 == reg).map(|l| (v, l.1)))
        .collect();
    let input_of = |v: Value| inputs.iter().find(|i| i.0 == v).map(|i| i.1);
    let attribute = |inst: &Inst, attr: &Operand| match *attr {
        Operand::Attr { index, chan, high } => ps_inputs
            .iter()
            .find(|i: &&PsInput| i.attr == index)
            .map(|i| Attribute {
                usage: i.usage,
                row: i.row,
                channel: chan,
                high,
            })
            .ok_or(unsupported(
                inst.machine.offset,
                "an interpolated attribute the stage data does not list",
            )),
        _ => Err(unsupported(inst.machine.offset, "an interpolation operand")),
    };
    // A VGPR value's barycentric kind and index, if it is one as loaded.
    let barycentric = |v: Option<Value>| match v.and_then(input_of) {
        Some(Input::Barycentric { kind, index }) => Some((kind, index)),
        _ => None,
    };

    let mut io = Io {
        inputs: inputs.clone(),
        initialized_sgprs: match &native.io {
            StageIo::Cs(cs) => Some(cs.initialized_sgprs()),
            _ => None,
        },
        ..Io::default()
    };
    let defs: std::collections::HashMap<Value, Site> = p
        .blocks
        .iter()
        .enumerate()
        .flat_map(|(b, block)| {
            block
                .insts
                .iter()
                .enumerate()
                .flat_map(move |(i, inst)| inst.writes.iter().map(move |w| (w.1, (b, i))))
        })
        .collect();
    let phis: std::collections::HashMap<Value, (usize, usize)> = p
        .blocks
        .iter()
        .enumerate()
        .flat_map(|(b, block)| {
            block
                .phis
                .iter()
                .enumerate()
                .map(move |(i, phi)| (phi.value, (b, i)))
        })
        .collect();
    let halves = Halves {
        p,
        defs: &defs,
        phis: &phis,
    };
    let mut first_halves = Vec::new();
    for (b, block) in p.blocks.iter().enumerate() {
        for (i, inst) in block.insts.iter().enumerate() {
            let site = (b, i);
            let at = inst.machine.offset;
            let name = inst.name();
            match &inst.fields {
                Fields::Vintrp { dst, src, attr } => {
                    let attribute = attribute(inst, attr)?;
                    match (name, *src) {
                        ("v_interp_mov_f32", Operand::Const(vertex)) => {
                            io.interps
                                .push((site, Interp::Vertex { attribute, vertex }));
                        }
                        ("v_interp_p1_f32", _) => first_halves.push((site, attribute)),
                        ("v_interp_p2_f32", Operand::Vector { reg, .. }) => {
                            let Operand::Vector { reg: d, .. } = *dst else {
                                return Err(unsupported(at, "an interpolation operand"));
                            };
                            // The first half's result is this one's old dst.
                            let interp = halves
                                .pair(
                                    attr,
                                    inst.read(Reg::V(d)),
                                    inst.read(Reg::V(reg)),
                                    &barycentric,
                                )
                                .map_err(|what| unsupported(at, what))?;
                            push_pair(&mut io, site, interp, attribute);
                        }
                        _ => return Err(unsupported(at, "an interpolation form")),
                    }
                }
                Fields::Valu(v) if name.starts_with("v_interp_") => match name {
                    "v_interp_p1ll_f16" => {
                        first_halves.push((site, attribute(inst, &v.src[0])?));
                    }
                    "v_interp_p2_f16" => {
                        let attribute = attribute(inst, &v.src[0])?;
                        let read = |k: usize| match v.src.get(k) {
                            Some(&Operand::Vector { reg, .. }) => inst.read(Reg::V(reg)),
                            _ => None,
                        };
                        let interp = halves
                            .pair(&v.src[0], read(2), read(1), &barycentric)
                            .map_err(|what| unsupported(at, what))?;
                        push_pair(&mut io, site, interp, attribute);
                    }
                    _ => return Err(unsupported(at, "an interpolation form")),
                },
                Fields::Exp {
                    target,
                    src,
                    compr,
                    done,
                    ..
                } => {
                    let target = match *target {
                        t @ 0..=7 => Target::Color(t),
                        8 => Target::Depth,
                        9 => Target::Null,
                        t @ 12..=16 => Target::Position(t - 12),
                        20 => Target::Primitive,
                        32 if params.is_empty() => Target::Unread,
                        t @ 32..=63 => {
                            // Bit 7: a mesh program's per-primitive output.
                            let o = params.iter().find(|o| o.param & 0x7F == t - 32).ok_or(
                                unsupported(at, "a parameter export the stage data does not list"),
                            )?;
                            let (usage, row) = (o.usage, o.row);
                            if o.param & 0x80 != 0 {
                                Target::PrimitiveParam { usage, row }
                            } else {
                                Target::Param { usage, row }
                            }
                        }
                        _ => return Err(unsupported(at, "an export target")),
                    };
                    let values = src.map(|s| match s {
                        Some(Operand::Vector { reg, .. }) => inst.read(Reg::V(reg)),
                        _ => None,
                    });
                    io.exports.push(Export {
                        site,
                        target,
                        values,
                        compressed: *compr,
                        done: *done,
                    });
                }
                _ => {}
            }
        }
    }
    if first_halves.is_empty() {
        return Ok(io);
    }
    // A loaded i/j pair normally becomes one host interpolation: its first
    // half need not be evaluated. A first half also read as data (or used
    // without a second half) must keep P0 + i * P10. Follow phi uses too,
    // without forcing ordinary pairs to read per-vertex attributes.
    let folded: std::collections::HashSet<_> = io
        .interps
        .iter()
        .filter_map(|(s, i)| matches!(i, Interp::Interpolated { .. }).then_some(*s))
        .collect();
    let mut used: std::collections::HashSet<_> = p
        .blocks
        .iter()
        .enumerate()
        .flat_map(|(b, blk)| {
            blk.insts
                .iter()
                .enumerate()
                .map(move |(i, inst)| ((b, i), inst))
        })
        .filter(|(s, _)| !folded.contains(s))
        .flat_map(|(_, inst)| inst.reads.iter().map(|r| r.1))
        .collect();
    loop {
        let before = used.len();
        for phi in p.blocks.iter().flat_map(|b| &b.phis) {
            if used.contains(&phi.value) {
                used.extend(&phi.args);
            }
        }
        if used.len() == before {
            break;
        }
    }
    for (site, attribute) in first_halves {
        if !io.interps.iter().any(|(s, _)| *s == site)
            && p.blocks[site.0].insts[site.1]
                .writes
                .iter()
                .any(|w| used.contains(&w.1))
        {
            io.interps.push((site, Interp::Weighted { attribute }));
        }
    }
    Ok(io)
}

/// The pixel input VGPRs that `addr` enables, with their meaning.
fn ps_vgprs(addr: u32) -> Vec<(u8, Input)> {
    let mut out = Vec::new();
    let mut next = 0u8;
    for (bit, &(count, kind)) in PS_VGPRS.iter().enumerate() {
        if addr >> bit & 1 == 0 {
            continue;
        }
        for index in 0..count {
            let input = match kind {
                Some(kind) => Input::Barycentric { kind, index },
                None => ps_system(bit),
            };
            out.push((next + index, input));
        }
        next += count;
    }
    out
}

/// An interpolation before its attribute is known: `barycentric` none
/// when i or j is computed.
struct Pair {
    barycentric: Option<Barycentric>,
    first: Vec<Site>,
}

impl Pair {
    fn with(self, attribute: Attribute) -> Interp {
        match self.barycentric {
            Some(barycentric) => Interp::Interpolated {
                attribute,
                barycentric,
                first: self.first,
            },
            None => Interp::Weighted { attribute },
        }
    }
}

/// Records interpolation `pair` at `site`, and its first halves too when
/// they evaluate on their own (computed barycentrics).
fn push_pair(io: &mut Io, site: Site, pair: Pair, attribute: Attribute) {
    if pair.barycentric.is_none() {
        for &f in &pair.first {
            if !io.interps.iter().any(|(s, _)| *s == f) {
                io.interps.push((f, Interp::Weighted { attribute }));
            }
        }
    }
    io.interps.push((site, pair.with(attribute)));
}

/// Finds the first halves of interpolations.
struct Halves<'a> {
    p: &'a Program,
    defs: &'a std::collections::HashMap<Value, Site>,
    phis: &'a std::collections::HashMap<Value, (usize, usize)>,
}

impl Halves<'_> {
    /// The interpolation whose second half reads the first half's result
    /// `acc`, the barycentric `j`, and attribute `attr`: every first half
    /// must read the same attribute channel. With j and every first half's
    /// i loaded as one kind's, the pair interpolates with that kind;
    /// otherwise each half evaluates on its own (`Interp::Weighted`).
    fn pair(
        &self,
        attr: &Operand,
        acc: Option<Value>,
        j: Option<Value>,
        barycentric: &impl Fn(Option<Value>) -> Option<(Barycentric, u8)>,
    ) -> Result<Pair, &'static str> {
        let mut first = Vec::new();
        self.sites(
            acc.ok_or("an interpolation without its first half")?,
            &mut Vec::new(),
            &mut first,
        )?;
        let mut kind = match barycentric(j) {
            Some((kind, 1)) => Some(kind),
            _ => None,
        };
        for &(b, i) in &first {
            let inst = &self.p.blocks[b].insts[i];
            let (a, i) = match &inst.fields {
                Fields::Vintrp {
                    src: Operand::Vector { reg, .. },
                    attr,
                    ..
                } if inst.name() == "v_interp_p1_f32" => (*attr, inst.read(Reg::V(*reg))),
                Fields::Valu(v) if inst.name() == "v_interp_p1ll_f16" => match v.src.get(1) {
                    Some(&Operand::Vector { reg, .. }) => (v.src[0], inst.read(Reg::V(reg))),
                    _ => return Err("an interpolation without its first half"),
                },
                _ => return Err("an interpolation without its first half"),
            };
            if a != *attr {
                return Err("interpolation halves of different attributes");
            }
            if kind.is_some_and(|k| barycentric(i) != Some((k, 0))) {
                kind = None;
            }
        }
        Ok(Pair {
            barycentric: kind,
            first,
        })
    }

    /// The instructions `v` comes from, through phis (`seen`: phis on the
    /// way; a loop back to one adds nothing).
    fn sites(
        &self,
        v: Value,
        seen: &mut Vec<Value>,
        out: &mut Vec<Site>,
    ) -> Result<(), &'static str> {
        if let Some(&site) = self.defs.get(&v) {
            if !out.contains(&site) {
                out.push(site);
            }
            return Ok(());
        }
        let &(b, i) = self
            .phis
            .get(&v)
            .ok_or("an interpolation without its first half")?;
        if !seen.contains(&v) {
            seen.push(v);
            for &a in &self.p.blocks[b].phis[i].args {
                self.sites(a, seen, out)?;
            }
        }
        Ok(())
    }
}

type ExecMask = (Value, Option<Value>);

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum CmpArg {
    Ssa(Value),
    Const(u8),
    Literal(u32),
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum Factor {
    Mask(ExecMask),
    Cmp(&'static str, CmpArg, CmpArg),
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum GroupKeep {
    /// Original mesh group input (`v5`) may still live in any active lane.
    AnyLane,
    /// Original mesh group input was overwritten in the union of all masks in
    /// `covered`, but lanes outside that union still hold the original input.
    Outside(Vec<ExecMask>),
}

struct ExecTracker<'a> {
    p: &'a Program,
    def_inst: HashMap<Value, (usize, usize)>,
    def_phi: HashMap<Value, (usize, usize)>,
    factors_cache: std::cell::RefCell<HashMap<ExecMask, Vec<Factor>>>,
}

impl<'a> ExecTracker<'a> {
    fn new(p: &'a Program) -> Self {
        let mut def_inst = HashMap::new();
        let mut def_phi = HashMap::new();
        for (b, blk) in p.blocks.iter().enumerate() {
            for (pi, phi) in blk.phis.iter().enumerate() {
                def_phi.insert(phi.value, (b, pi));
            }
            for (ii, inst) in blk.insts.iter().enumerate() {
                for &(_, v) in &inst.writes {
                    def_inst.insert(v, (b, ii));
                }
            }
        }
        Self {
            p,
            def_inst,
            def_phi,
            factors_cache: std::cell::RefCell::new(HashMap::new()),
        }
    }

    fn exec_of(&self, inst: &Inst) -> Option<ExecMask> {
        let lo = inst.read(Reg::S(EXEC))?;
        let hi = if self.p.wave64 {
            Some(inst.read(Reg::S(EXEC + 1))?)
        } else {
            None
        };
        Some((lo, hi))
    }

    fn read_mask_op(&self, inst: &Inst, op: &Operand) -> Option<ExecMask> {
        let Operand::Scalar { code, dwords } = *op else {
            return None;
        };
        if code == crate::gfx10::NULL || (self.p.wave64 && dwords < 2) {
            return None;
        }
        let lo = inst.read(Reg::S(code))?;
        let hi = if self.p.wave64 {
            Some(inst.read(Reg::S(code + 1))?)
        } else {
            None
        };
        Some((lo, hi))
    }

    /// If edge `q -> b` is the EXEC-zero branch of `s_cbranch_execz` / `_execnz`,
    /// returns the `EXEC` mask proven zero along that edge.
    fn no_lanes_exec(&self, q: usize, b: usize) -> Option<ExecMask> {
        let last = self.p.blocks[q].insts.last()?;
        let is_zero_edge = match (last.name(), self.p.blocks[q].exit) {
            ("s_cbranch_execz", Exit::Branch { taken, not_taken }) => {
                taken == b && not_taken != b
            }
            ("s_cbranch_execnz", Exit::Branch { taken, not_taken }) => {
                not_taken == b && taken != b
            }
            _ => false,
        };
        if is_zero_edge {
            self.exec_of(last)
        } else {
            None
        }
    }

    fn mask_def_inst(&self, m: ExecMask) -> Option<&'a Inst> {
        let &(b, ii) = self.def_inst.get(&m.0)?;
        if let Some(hi) = m.1
            && self.def_inst.get(&hi) != Some(&(b, ii))
        {
            return None;
        }
        Some(&self.p.blocks[b].insts[ii])
    }

    fn mask_def_phi(&self, m: ExecMask) -> Option<(usize, usize, Option<usize>)> {
        let &(b, pi_lo) = self.def_phi.get(&m.0)?;
        let pi_hi = match m.1 {
            Some(hi) => {
                let &(b_hi, pi) = self.def_phi.get(&hi)?;
                if b_hi != b {
                    return None;
                }
                Some(pi)
            }
            None => None,
        };
        Some((b, pi_lo, pi_hi))
    }

    fn phi_edge_masks(&self, b: usize, pi_lo: usize, pi_hi: Option<usize>) -> Vec<(usize, ExecMask)> {
        let lo_phi = &self.p.blocks[b].phis[pi_lo];
        let hi_phi = pi_hi.map(|i| &self.p.blocks[b].phis[i]);
        self.p.blocks[b]
            .preds
            .iter()
            .enumerate()
            .map(|(k, &pred)| (pred, (lo_phi.args[k], hi_phi.map(|h| h.args[k]))))
            .collect()
    }

    fn is_new_exec_of(&self, inst: &Inst, m: ExecMask) -> bool {
        inst.write(Reg::S(EXEC)) == Some(m.0)
            && (!self.p.wave64 || inst.write(Reg::S(EXEC + 1)) == m.1)
    }

    fn is_zero_op(op: &Operand) -> bool {
        matches!(
            *op,
            Operand::Const(128) | Operand::Literal(0) | Operand::Scalar { code: crate::gfx10::NULL, .. }
        )
    }

    /// Whether `m` is provably empty (`0`, no lanes enabled).
    fn is_zero_mask(&self, m: ExecMask, depth: u8) -> bool {
        let m = self.unwrap_simple(m, depth);
        let Some(inst) = self.mask_def_inst(m) else {
            return false;
        };
        let Fields::Sop { src, .. } = &inst.fields else {
            return false;
        };
        match inst.name() {
            "s_mov_b32" | "s_mov_b64" => src.first().is_some_and(Self::is_zero_op),
            "s_and_saveexec_b32" | "s_and_saveexec_b64" if self.is_new_exec_of(inst, m) => {
                src.first().is_some_and(Self::is_zero_op)
            }
            "s_and_b32" | "s_and_b64" => src.iter().take(2).any(Self::is_zero_op),
            "s_xor_b32" | "s_xor_b64" if src.len() >= 2 => {
                match (
                    self.read_mask_op(inst, &src[0]),
                    self.read_mask_op(inst, &src[1]),
                ) {
                    (Some(a), Some(b)) => {
                        depth > 0 && self.unwrap_simple(a, depth - 1) == self.unwrap_simple(b, depth - 1)
                    }
                    _ => false,
                }
            }
            _ => false,
        }
    }

    /// Syntactically proves `m ⊆ target` without consulting `visiting`, so a
    /// narrowing backedge whose node happens to be an ancestor on `compute_factors`'s
    /// recursion stack (e.g. waterfall `s_and_saveexec` -> `s_xor_b64` -> `exec_phi`)
    /// is still recognized as `⊆ phi_mask`.
    fn narrows_mask(&self, m: ExecMask, target: ExecMask, depth: u8) -> bool {
        let u = self.unwrap_simple(m, 16);
        if u == target || self.is_zero_mask(u, 4) {
            return true;
        }
        if depth == 0 {
            return false;
        }
        let Some(inst) = self.mask_def_inst(u) else {
            return false;
        };
        let name = inst.name();
        if name.starts_with("v_cmpx_") && self.is_new_exec_of(inst, u) {
            return self
                .exec_of(inst)
                .is_some_and(|old| self.narrows_mask(old, target, depth - 1));
        }
        let Fields::Sop { src, .. } = &inst.fields else {
            return false;
        };
        match name {
            "s_and_saveexec_b32" | "s_and_saveexec_b64" if self.is_new_exec_of(inst, u) => {
                self.exec_of(inst)
                    .is_some_and(|old| self.narrows_mask(old, target, depth - 1))
                    || src
                        .first()
                        .and_then(|op| self.read_mask_op(inst, op))
                        .is_some_and(|mask| self.narrows_mask(mask, target, depth - 1))
            }
            "s_and_b32" | "s_and_b64" if src.len() >= 2 => src[..2].iter().any(|op| {
                self.read_mask_op(inst, op)
                    .is_some_and(|mask| self.narrows_mask(mask, target, depth - 1))
            }),
            "s_andn2_saveexec_b32" | "s_andn2_saveexec_b64" if self.is_new_exec_of(inst, u) => src
                .first()
                .and_then(|op| self.read_mask_op(inst, op))
                .is_some_and(|mask| self.narrows_mask(mask, target, depth - 1)),
            "s_andn2_wrexec_b32" | "s_andn2_wrexec_b64" | "s_andn2_b32" | "s_andn2_b64" => src
                .first()
                .and_then(|op| self.read_mask_op(inst, op))
                .is_some_and(|mask| self.narrows_mask(mask, target, depth - 1)),
            "s_andn1_saveexec_b32" | "s_andn1_saveexec_b64" if self.is_new_exec_of(inst, u) => {
                self.exec_of(inst)
                    .is_some_and(|old| self.narrows_mask(old, target, depth - 1))
            }
            "s_andn1_wrexec_b32" | "s_andn1_wrexec_b64" => self
                .exec_of(inst)
                .is_some_and(|old| self.narrows_mask(old, target, depth - 1)),
            "s_or_b32" | "s_or_b64" | "s_xor_b32" | "s_xor_b64" if src.len() >= 2 => {
                let side = |op: &Operand| {
                    Self::is_zero_op(op)
                        || self
                            .read_mask_op(inst, op)
                            .is_some_and(|mask| self.narrows_mask(mask, target, depth - 1))
                };
                side(&src[0]) && side(&src[1])
            }
            _ => false,
        }
    }

    /// Unfolds a phi arm through subset-preserving union/xor/phi operations into
    /// its leaf masks, omitting empty (`0`) and inductive self/SCC phi references.
    fn collect_phi_leaves(
        &self,
        phi_mask: ExecMask,
        m: ExecMask,
        depth: u8,
        seen: &mut Vec<ExecMask>,
        leaves: &mut Vec<ExecMask>,
    ) {
        let u = self.unwrap_simple(m, 16);
        if u == phi_mask
            || self.is_zero_mask(u, 8)
            || self.narrows_mask(u, phi_mask, 8)
            || seen.contains(&u)
        {
            return;
        }
        seen.push(u);
        if depth > 0
            && let Some(inst) = self.mask_def_inst(u)
            && let Fields::Sop { src, .. } = &inst.fields
        {
            let name = inst.name();
            if matches!(name, "s_or_b32" | "s_or_b64" | "s_xor_b32" | "s_xor_b64")
                && src.len() >= 2
            {
                let op_or_zero = |op: &Operand| -> Option<Option<ExecMask>> {
                    if Self::is_zero_op(op) {
                        Some(None)
                    } else {
                        self.read_mask_op(inst, op).map(Some)
                    }
                };
                if let (Some(a), Some(b)) = (op_or_zero(&src[0]), op_or_zero(&src[1])) {
                    if let Some(am) = a {
                        self.collect_phi_leaves(phi_mask, am, depth - 1, seen, leaves);
                    }
                    if let Some(bm) = b {
                        self.collect_phi_leaves(phi_mask, bm, depth - 1, seen, leaves);
                    }
                    return;
                }
            }
            if matches!(
                name,
                "s_or_saveexec_b32"
                    | "s_or_saveexec_b64"
                    | "s_xor_saveexec_b32"
                    | "s_xor_saveexec_b64"
            ) && self.is_new_exec_of(inst, u)
                && let (Some(old), Some(op)) = (self.exec_of(inst), src.first())
            {
                if Self::is_zero_op(op) {
                    self.collect_phi_leaves(phi_mask, old, depth - 1, seen, leaves);
                    return;
                } else if let Some(mask) = self.read_mask_op(inst, op) {
                    self.collect_phi_leaves(phi_mask, old, depth - 1, seen, leaves);
                    self.collect_phi_leaves(phi_mask, mask, depth - 1, seen, leaves);
                    return;
                }
            }
        }
        if depth > 0
            && let Some((b, pi_lo, pi_hi)) = self.mask_def_phi(u)
        {
            let is_exec_phi = self.p.blocks[b].phis[pi_lo].reg == Reg::S(EXEC);
            for (pred, arm) in self.phi_edge_masks(b, pi_lo, pi_hi) {
                if is_exec_phi && self.no_lanes_exec(pred, b).is_some() {
                    continue;
                }
                self.collect_phi_leaves(phi_mask, arm, depth - 1, seen, leaves);
            }
            return;
        }
        if !leaves.contains(&u) {
            leaves.push(u);
        }
    }

    /// Whether `m` is provably all lanes enabled (`-1`, e.g. `s_mov_b64 exec, -1`
    /// or `s_or_saveexec_b64 _, -1`, or a phi whose arms are all `-1`).
    fn is_all_lanes(&self, m: ExecMask, depth: u8) -> bool {
        let m = self.unwrap_simple(m, depth);
        if let Some(inst) = self.mask_def_inst(m) {
            let Fields::Sop { src, .. } = &inst.fields else {
                return false;
            };
            return match inst.name() {
                "s_mov_b32" | "s_mov_b64" => src.first() == Some(&Operand::Const(193)),
                "s_or_saveexec_b32" | "s_or_saveexec_b64" if self.is_new_exec_of(inst, m) => {
                    src.first() == Some(&Operand::Const(193))
                        || (depth > 0
                            && (self
                                .exec_of(inst)
                                .is_some_and(|old| self.is_all_lanes(old, depth - 1))
                                || src
                                    .first()
                                    .and_then(|op| self.read_mask_op(inst, op))
                                    .is_some_and(|mask| self.is_all_lanes(mask, depth - 1))))
                }
                "s_nand_saveexec_b32" | "s_nand_saveexec_b64" if self.is_new_exec_of(inst, m) => {
                    src.first() == Some(&Operand::Const(128))
                        || (depth > 0
                            && src
                                .first()
                                .and_then(|op| self.read_mask_op(inst, op))
                                .is_some_and(|mask| self.is_zero_mask(mask, depth - 1)))
                }
                "s_or_b32" | "s_or_b64" => {
                    src.contains(&Operand::Const(193))
                        || (depth > 0
                            && src.iter().take(2).any(|op| {
                                self.read_mask_op(inst, op)
                                    .is_some_and(|sm| self.is_all_lanes(sm, depth - 1))
                            }))
                }
                _ => false,
            };
        }
        if depth > 0
            && let Some((b, pi_lo, pi_hi)) = self.mask_def_phi(m)
        {
            let arms = self.phi_edge_masks(b, pi_lo, pi_hi);
            let mut has_non_self = false;
            let all = arms.into_iter().all(|(_, arm)| {
                if arm == m {
                    true
                } else {
                    has_non_self = true;
                    self.is_all_lanes(arm, depth - 1)
                }
            });
            return has_non_self && all;
        }
        false
    }

    fn unwrap_scalar_val(&self, mut v: Value, depth: u8) -> Value {
        for _ in 0..depth {
            let Some(&(b, ii)) = self.def_inst.get(&v) else {
                break;
            };
            let inst = &self.p.blocks[b].insts[ii];
            let next = match (inst.name(), &inst.fields) {
                ("s_mov_b32", Fields::Sop { src, .. }) => match src.first() {
                    Some(Operand::Scalar { code, .. }) if *code != crate::gfx10::NULL => {
                        inst.read(Reg::S(*code))
                    }
                    _ => None,
                },
                ("v_mov_b32", Fields::Valu(f))
                    if f.sdwa.is_none()
                        && f.dpp.is_none()
                        && f.abs == 0
                        && f.neg == 0
                        && !f.clamp
                        && f.omod == 0
                        && self.exec_of(inst).is_some_and(|e| self.is_all_lanes(e, 8)) =>
                {
                    match f.src.first() {
                        Some(Operand::Vector { reg, .. }) => inst.read(Reg::V(*reg)),
                        Some(Operand::Scalar { code, .. }) if *code != crate::gfx10::NULL => {
                            inst.read(Reg::S(*code))
                        }
                        _ => None,
                    }
                }
                _ => None,
            };
            let Some(n) = next else { break };
            v = n;
        }
        v
    }

    fn cmp_factor(&self, inst: &Inst) -> Option<Factor> {
        let rest = inst
            .name()
            .strip_prefix("v_cmpx_")
            .or_else(|| inst.name().strip_prefix("v_cmp_"))?;
        let Fields::Valu(f) = &inst.fields else {
            return None;
        };
        if f.src.len() != 2
            || f.abs != 0
            || f.neg != 0
            || f.opsel != 0
            || f.sdwa.is_some()
            || f.dpp.is_some()
        {
            return None;
        }
        let arg = |op: &Operand| -> Option<CmpArg> {
            match *op {
                Operand::Vector { reg, dwords } if dwords <= 1 => {
                    Some(CmpArg::Ssa(self.unwrap_scalar_val(inst.read(Reg::V(reg))?, 4)))
                }
                Operand::Scalar { code, dwords } if dwords <= 1 => {
                    Some(CmpArg::Ssa(self.unwrap_scalar_val(inst.read(Reg::S(code))?, 4)))
                }
                Operand::Const(k) if k < 251 => Some(CmpArg::Const(k)),
                Operand::Literal(x) => Some(CmpArg::Literal(x)),
                _ => None,
            }
        };
        let (a, b) = (arg(&f.src[0])?, arg(&f.src[1])?);
        let (op, a, b) = match rest {
            "gt_u32" => ("lt_u32", b, a),
            "ge_u32" => ("le_u32", b, a),
            "gt_i32" => ("lt_i32", b, a),
            "ge_i32" => ("le_i32", b, a),
            "gt_f32" => ("lt_f32", b, a),
            "ge_f32" => ("le_f32", b, a),
            "eq_u32" | "lg_u32" | "eq_i32" | "lg_i32" | "eq_f32" | "lg_f32" | "o_f32" | "u_f32" => {
                if a <= b {
                    (rest, a, b)
                } else {
                    (rest, b, a)
                }
            }
            _ => (rest, a, b),
        };
        Some(Factor::Cmp(op, a, b))
    }

    /// Unwraps syntactic copies (`s_mov`, saved old `EXEC` of `saveexec`,
    /// `s_and_saveexec` under all-lanes `EXEC`, and identical-arm phis) without
    /// invoking factor analysis, keeping `unwrap_simple -> collect_factors ->
    /// unwrap_copy` strictly layered and linear.
    fn unwrap_simple(&self, m: ExecMask, depth: u8) -> ExecMask {
        if depth == 0 {
            return m;
        }
        if let Some(inst) = self.mask_def_inst(m) {
            let name = inst.name();
            if matches!(name, "s_mov_b32" | "s_mov_b64")
                && let Fields::Sop { src, .. } = &inst.fields
                && let Some(op) = src.first()
                && let Some(next) = self.read_mask_op(inst, op)
            {
                return self.unwrap_simple(next, depth - 1);
            }
            if name.contains("saveexec")
                && !self.is_new_exec_of(inst, m)
                && let Some(old) = self.exec_of(inst)
            {
                return self.unwrap_simple(old, depth - 1);
            }
            if matches!(name, "s_and_saveexec_b32" | "s_and_saveexec_b64")
                && self.is_new_exec_of(inst, m)
                && let Fields::Sop { src, .. } = &inst.fields
                && let (Some(old), Some(mask)) = (
                    self.exec_of(inst),
                    src.first().and_then(|op| self.read_mask_op(inst, op)),
                )
                && self.is_all_lanes(old, depth - 1)
            {
                return self.unwrap_simple(mask, depth - 1);
            }
        }
        if let Some((b, pi_lo, pi_hi)) = self.mask_def_phi(m) {
            let mut target: Option<ExecMask> = None;
            let mut same = true;
            for (_, arm) in self.phi_edge_masks(b, pi_lo, pi_hi) {
                let u = self.unwrap_simple(arm, depth - 1);
                match target {
                    None => target = Some(u),
                    Some(t) if t == u => {}
                    Some(_) => {
                        same = false;
                        break;
                    }
                }
            }
            if same && let Some(t) = target {
                return t;
            }
        }
        m
    }

    /// Unwraps `m` to an equal mask, including `s_or_b32/64(a, b)` where one
    /// operand is proven a subset of the other (`a ⊆ b ⇒ a | b == b`).
    fn unwrap_copy(&self, m: ExecMask, depth: u8) -> ExecMask {
        let m = self.unwrap_simple(m, depth);
        if depth == 0 {
            return m;
        }
        if let Some(inst) = self.mask_def_inst(m)
            && matches!(inst.name(), "s_or_b32" | "s_or_b64")
            && let Fields::Sop { src, .. } = &inst.fields
            && src.len() >= 2
            && let (Some(a), Some(b)) = (
                self.read_mask_op(inst, &src[0]),
                self.read_mask_op(inst, &src[1]),
            )
        {
            if self.implies_inner(a, b, depth - 1) {
                return self.unwrap_copy(b, depth - 1);
            }
            if self.implies_inner(b, a, depth - 1) {
                return self.unwrap_copy(a, depth - 1);
            }
        }
        m
    }

    fn push_factor(out: &mut Vec<Factor>, f: Factor) {
        if !out.contains(&f) {
            out.push(f);
        }
    }

    fn factors_of(&self, m: ExecMask) -> Vec<Factor> {
        if let Some(cached) = self.factors_cache.borrow().get(&m) {
            return cached.clone();
        }
        let mut visiting = Vec::new();
        let mut hit_cycle = false;
        let out = self.compute_factors(m, 96, &mut visiting, &mut hit_cycle);
        if !hit_cycle {
            self.factors_cache.borrow_mut().insert(m, out.clone());
        }
        out
    }

    fn compute_factors(
        &self,
        m: ExecMask,
        depth: u8,
        visiting: &mut Vec<ExecMask>,
        hit_cycle: &mut bool,
    ) -> Vec<Factor> {
        if let Some(cached) = self.factors_cache.borrow().get(&m) {
            return cached.clone();
        }
        if visiting.contains(&m) {
            *hit_cycle = true;
            return vec![Factor::Mask(m)];
        }
        if depth == 0 {
            *hit_cycle = true;
            return vec![Factor::Mask(m)];
        }
        visiting.push(m);
        let mut local_cycle = false;
        let mut out = vec![Factor::Mask(m)];
        let u = self.unwrap_simple(m, depth.min(16));
        if u != m {
            for f in self.compute_factors(u, depth - 1, visiting, &mut local_cycle) {
                Self::push_factor(&mut out, f);
            }
        }
        if let Some(inst) = self.mask_def_inst(m) {
            let name = inst.name();
            if (name.starts_with("v_cmpx_") && self.is_new_exec_of(inst, m))
                || name.starts_with("v_cmp_")
            {
                if let Some(old) = self.exec_of(inst) {
                    for f in self.compute_factors(old, depth - 1, visiting, &mut local_cycle) {
                        Self::push_factor(&mut out, f);
                    }
                }
                if let Some(cf) = self.cmp_factor(inst) {
                    Self::push_factor(&mut out, cf);
                }
            } else if let Fields::Sop { src, .. } = &inst.fields {
                if matches!(name, "s_and_saveexec_b32" | "s_and_saveexec_b64")
                    && self.is_new_exec_of(inst, m)
                {
                    if let Some(old) = self.exec_of(inst) {
                        for f in self.compute_factors(old, depth - 1, visiting, &mut local_cycle) {
                            Self::push_factor(&mut out, f);
                        }
                    }
                    if let Some(op) = src.first()
                        && let Some(mask) = self.read_mask_op(inst, op)
                    {
                        for f in self.compute_factors(mask, depth - 1, visiting, &mut local_cycle) {
                            Self::push_factor(&mut out, f);
                        }
                    }
                } else if matches!(name, "s_and_b32" | "s_and_b64") && src.len() >= 2 {
                    for op in &src[..2] {
                        if let Some(mask) = self.read_mask_op(inst, op) {
                            for f in self.compute_factors(mask, depth - 1, visiting, &mut local_cycle)
                            {
                                Self::push_factor(&mut out, f);
                            }
                        }
                    }
                } else if matches!(name, "s_andn2_saveexec_b32" | "s_andn2_saveexec_b64")
                    && self.is_new_exec_of(inst, m)
                {
                    // Per RDNA2 ISA: D = old_exec; EXEC = src0 & ~old_exec, so new_exec ⊆ src0.
                    if let Some(op) = src.first()
                        && let Some(mask) = self.read_mask_op(inst, op)
                    {
                        for f in self.compute_factors(mask, depth - 1, visiting, &mut local_cycle) {
                            Self::push_factor(&mut out, f);
                        }
                    }
                } else if matches!(name, "s_andn2_wrexec_b32" | "s_andn2_wrexec_b64") {
                    // Per RDNA2 ISA: D = EXEC = src0 & ~old_exec (both dst and EXEC ⊆ src0).
                    if let Some(op) = src.first()
                        && let Some(mask) = self.read_mask_op(inst, op)
                    {
                        for f in self.compute_factors(mask, depth - 1, visiting, &mut local_cycle) {
                            Self::push_factor(&mut out, f);
                        }
                    }
                } else if (matches!(name, "s_andn1_saveexec_b32" | "s_andn1_saveexec_b64")
                    && self.is_new_exec_of(inst, m))
                    || matches!(name, "s_andn1_wrexec_b32" | "s_andn1_wrexec_b64")
                {
                    // Per RDNA2 ISA: EXEC = ~src0 & old_exec (and for wrexec, D = EXEC), so ⊆ old_exec.
                    if let Some(old) = self.exec_of(inst) {
                        for f in self.compute_factors(old, depth - 1, visiting, &mut local_cycle) {
                            Self::push_factor(&mut out, f);
                        }
                    }
                } else if matches!(name, "s_andn2_b32" | "s_andn2_b64")
                    && let Some(op) = src.first()
                    && let Some(mask) = self.read_mask_op(inst, op)
                {
                    for f in self.compute_factors(mask, depth - 1, visiting, &mut local_cycle) {
                        Self::push_factor(&mut out, f);
                    }
                } else if matches!(name, "s_or_b32" | "s_or_b64" | "s_xor_b32" | "s_xor_b64")
                    && src.len() >= 2
                    && let (Some(a), Some(b)) = (
                        self.read_mask_op(inst, &src[0]),
                        self.read_mask_op(inst, &src[1]),
                    )
                {
                    // Boolean algebra: if `a ⊆ X` and `b ⊆ X`, then both `a | b ⊆ X`
                    // and `a ^ b ⊆ X`.
                    let fa = self.compute_factors(a, depth - 1, visiting, &mut local_cycle);
                    let fb = self.compute_factors(b, depth - 1, visiting, &mut local_cycle);
                    for f in fa {
                        if fb.contains(&f) {
                            Self::push_factor(&mut out, f);
                        }
                    }
                } else if matches!(
                    name,
                    "s_or_saveexec_b32"
                        | "s_or_saveexec_b64"
                        | "s_xor_saveexec_b32"
                        | "s_xor_saveexec_b64"
                ) && self.is_new_exec_of(inst, m)
                    && let (Some(old), Some(mask)) = (
                        self.exec_of(inst),
                        src.first().and_then(|op| self.read_mask_op(inst, op)),
                    )
                {
                    let fa = self.compute_factors(old, depth - 1, visiting, &mut local_cycle);
                    let fb = self.compute_factors(mask, depth - 1, visiting, &mut local_cycle);
                    for f in fa {
                        if fb.contains(&f) {
                            Self::push_factor(&mut out, f);
                        }
                    }
                }
            }
        }
        if let Some((b, pi_lo, pi_hi)) = self.mask_def_phi(m) {
            let is_exec_phi = self.p.blocks[b].phis[pi_lo].reg == Reg::S(EXEC);
            let mut seen_leaves = vec![m];
            let mut leaves = Vec::new();
            for (pred, arm) in self.phi_edge_masks(b, pi_lo, pi_hi) {
                if is_exec_phi && self.no_lanes_exec(pred, b).is_some() {
                    continue;
                }
                self.collect_phi_leaves(m, arm, 16, &mut seen_leaves, &mut leaves);
            }
            let mut common: Option<Vec<Factor>> = None;
            for leaf in leaves {
                // A leaf returning to an active phi through strictly narrowing
                // operations (v_cmpx / s_and / s_andn2 / subset or/xor)
                // satisfies leaf ⊆ phi by induction, so omit it from the intersection.
                let f = self.compute_factors(leaf, depth - 1, visiting, &mut local_cycle);
                if f.contains(&Factor::Mask(m)) {
                    continue;
                }
                common = Some(match common {
                    None => f,
                    Some(prev) => prev.into_iter().filter(|x| f.contains(x)).collect(),
                });
            }
            if let Some(shared) = common {
                for f in shared {
                    Self::push_factor(&mut out, f);
                }
            }
        }
        visiting.pop();
        if local_cycle {
            *hit_cycle = true;
        } else {
            self.factors_cache.borrow_mut().insert(m, out.clone());
        }
        out
    }

    fn required_factors(&self, w: ExecMask, depth: u8, out: &mut Vec<Factor>) {
        let w = self.unwrap_copy(w, depth.min(16));
        if self.is_all_lanes(w, depth.min(16)) {
            return;
        }
        if depth > 0
            && let Some(inst) = self.mask_def_inst(w)
        {
            let name = inst.name();
            if ((name.starts_with("v_cmpx_") && self.is_new_exec_of(inst, w))
                || name.starts_with("v_cmp_"))
                && let (Some(old), Some(cf)) = (self.exec_of(inst), self.cmp_factor(inst))
            {
                // Per GCN/RDNA2 ISA, both `v_cmpx_*` and `v_cmp_*` write
                // `EXEC & P` (inactive lanes produce 0 bits), so proving
                // `r ⊆ dst` requires both `r ⊆ EXEC` and `r ⇒ P`.
                self.required_factors(old, depth - 1, out);
                Self::push_factor(out, cf);
                return;
            }
            if let Fields::Sop { src, .. } = &inst.fields {
                if matches!(name, "s_and_saveexec_b32" | "s_and_saveexec_b64")
                    && self.is_new_exec_of(inst, w)
                    && let (Some(old), Some(mask)) = (
                        self.exec_of(inst),
                        src.first().and_then(|op| self.read_mask_op(inst, op)),
                    )
                {
                    self.required_factors(old, depth - 1, out);
                    self.required_factors(mask, depth - 1, out);
                    return;
                }
                if matches!(name, "s_and_b32" | "s_and_b64")
                    && src.len() >= 2
                    && let (Some(a), Some(b)) = (
                        self.read_mask_op(inst, &src[0]),
                        self.read_mask_op(inst, &src[1]),
                    )
                {
                    self.required_factors(a, depth - 1, out);
                    self.required_factors(b, depth - 1, out);
                    return;
                }
            }
        }
        Self::push_factor(out, Factor::Mask(w));
    }

    /// Returns `true` if the comparison factor `rf` proves `req` (`rf => req`).
    ///
    /// Handles exact equality and the unconditional 32-bit unsigned bit-shift
    /// theorem: for every `x, y, s: u32` (with `k = s & 31`, even when `y << k`
    /// wraps mod 2^32), `x < (y << k) ==> (x >> k) < y`.
    fn cmp_factor_implies(&self, rf: &Factor, req: &Factor, rf_all: &[Factor]) -> bool {
        if rf == req {
            return true;
        }
        let (
            Factor::Cmp("lt_u32", CmpArg::Ssa(x), CmpArg::Ssa(y_shl)),
            Factor::Cmp("lt_u32", CmpArg::Ssa(x_shr), CmpArg::Ssa(y)),
        ) = (rf, req)
        else {
            return false;
        };
        let Some(&(b_shl, i_shl)) = self.def_inst.get(y_shl) else {
            return false;
        };
        let Some(&(b_shr, i_shr)) = self.def_inst.get(x_shr) else {
            return false;
        };
        let shl_inst = &self.p.blocks[b_shl].insts[i_shl];
        let shr_inst = &self.p.blocks[b_shr].insts[i_shr];
        // Match `y_shl = s_lshl_b32(y, k)` and `x_shr = v_lshrrev_b32(k, x)` with
        // identical shift amount `k`, under an EXEC mask covering `r`.
        let arg_of = |inst: &Inst, op: &Operand| -> Option<CmpArg> {
            match *op {
                Operand::Vector { reg, dwords } if dwords <= 1 => {
                    Some(CmpArg::Ssa(self.unwrap_scalar_val(inst.read(Reg::V(reg))?, 4)))
                }
                Operand::Scalar { code, dwords } if code != crate::gfx10::NULL && dwords <= 1 => {
                    Some(CmpArg::Ssa(self.unwrap_scalar_val(inst.read(Reg::S(code))?, 4)))
                }
                Operand::Const(k) if k < 251 => Some(CmpArg::Const(k)),
                Operand::Literal(v) => Some(CmpArg::Literal(v)),
                _ => None,
            }
        };
        let shl_parts = match (shl_inst.name(), &shl_inst.fields) {
            ("s_lshl_b32", Fields::Sop { src, .. }) if src.len() >= 2 => {
                match (arg_of(shl_inst, &src[0]), arg_of(shl_inst, &src[1])) {
                    (Some(base), Some(shift)) => Some((base, shift)),
                    _ => None,
                }
            }
            _ => None,
        };
        let shr_parts = match (shr_inst.name(), &shr_inst.fields) {
            ("v_lshrrev_b32", Fields::Valu(f))
                if f.src.len() >= 2
                    && f.sdwa.is_none()
                    && f.dpp.is_none()
                    && self
                        .exec_of(shr_inst)
                        .is_some_and(|e| self.is_all_lanes(e, 8) || rf_all.contains(&Factor::Mask(e))) =>
            {
                match (arg_of(shr_inst, &f.src[0]), arg_of(shr_inst, &f.src[1])) {
                    (Some(shift), Some(base)) => Some((base, shift)),
                    _ => None,
                }
            }
            _ => None,
        };
        matches!(
            (shl_parts, shr_parts),
            (Some((y2, k1)), Some((x2, k2)))
                if y2 == CmpArg::Ssa(*y) && x2 == CmpArg::Ssa(*x) && k1 == k2
        )
    }

    fn implies_inner(&self, r: ExecMask, w: ExecMask, depth: u8) -> bool {
        if r == w {
            return true;
        }
        if depth == 0 {
            return false;
        }
        if self.is_all_lanes(w, depth.min(16) - 1) {
            return true;
        }
        let rf = self.factors_of(r);
        if rf.contains(&Factor::Mask(w)) {
            return true;
        }
        let mut wf = Vec::new();
        self.required_factors(w, depth, &mut wf);
        wf.iter()
            .all(|req| rf.iter().any(|f| self.cmp_factor_implies(f, req, &rf)))
    }

    /// Soundly proves `r ⊆ w` (every active lane of `r` was active in `w`).
    fn implies(&self, r: ExecMask, w: ExecMask) -> bool {
        self.implies_inner(r, w, 24)
    }

    /// If `mask` is the complement of some `part` inside `parent` (`mask =
    /// part ^ parent` with `part ⊆ parent`, or `mask = parent & !cond` with
    /// `part = parent & cond`), returns `Some(parent)` when `part` is covered
    /// by `masks`.
    fn complement_parent(&self, mask: ExecMask, masks: &[ExecMask]) -> Option<ExecMask> {
        let m = self.unwrap_copy(mask, 16);
        let inst = self.mask_def_inst(m)?;
        let Fields::Sop { src, .. } = &inst.fields else {
            return None;
        };
        let name = inst.name();
        // `s_xor_b32/64 exec, part, parent` (standard GFX10 if/else flip):
        // when `part ⊆ parent`, `part ∪ (part ^ parent) == parent`.
        if matches!(name, "s_xor_b32" | "s_xor_b64") && src.len() >= 2 {
            let a = self.read_mask_op(inst, &src[0])?;
            let b = self.read_mask_op(inst, &src[1])?;
            for (part, parent) in [(a, b), (b, a)] {
                let parent = self.unwrap_copy(parent, 16);
                if self.implies(part, parent) && masks.iter().any(|&c| self.implies(part, c)) {
                    return Some(parent);
                }
            }
        }
        // `s_andn2_saveexec_b64`: `src0 & ~old_exec` (only for new EXEC);
        // `s_andn2_wrexec_b64`: `src0 & ~old_exec` (for both dst and new EXEC);
        // `s_andn2_b32/64`: `src0 & ~src1`.
        let andn2 = if matches!(name, "s_andn2_saveexec_b32" | "s_andn2_saveexec_b64")
            && self.is_new_exec_of(inst, m)
            || matches!(name, "s_andn2_wrexec_b32" | "s_andn2_wrexec_b64")
        {
            Some((
                src.first().and_then(|op| self.read_mask_op(inst, op))?,
                self.exec_of(inst)?,
            ))
        } else if matches!(name, "s_andn2_b32" | "s_andn2_b64") && src.len() >= 2 {
            Some((
                self.read_mask_op(inst, &src[0])?,
                self.read_mask_op(inst, &src[1])?,
            ))
        } else {
            None
        };
        if let Some((parent, cond)) = andn2 {
            let parent = self.unwrap_copy(parent, 16);
            let cond = self.unwrap_copy(cond, 16);
            let pf = self.factors_of(parent);
            let cf = self.factors_of(cond);
            for &c in masks {
                let mut wf = Vec::new();
                self.required_factors(c, 16, &mut wf);
                // If `c` is `parent & cond` (every factor implied by `parent` or `cond`),
                // then `c ∪ (parent & !cond) == parent`.
                let covered_by_parent_and_cond =
                    !wf.is_empty() && wf.iter().all(|f| pf.contains(f) || cf.contains(f));
                if covered_by_parent_and_cond {
                    return Some(parent);
                }
            }
        }
        None
    }

    fn add_covered(&self, masks: &mut Vec<ExecMask>, m: ExecMask) {
        let mut queue = vec![self.unwrap_copy(m, 8)];
        while let Some(cur) = queue.pop() {
            if masks.iter().any(|&existing| self.implies(cur, existing)) {
                continue;
            }
            if let Some(parent) = self.complement_parent(cur, masks) {
                queue.push(parent);
                continue;
            }
            // Also check if an existing mask is the complement of `cur`.
            let promoted: Vec<ExecMask> = masks
                .iter()
                .filter_map(|&existing| self.complement_parent(existing, &[cur]))
                .collect();
            masks.retain(|&existing| !self.implies(existing, cur));
            masks.push(cur);
            for p in promoted {
                queue.push(p);
            }
        }
        masks.sort_by_key(|&(lo, hi)| (lo.0, hi.map(|v| v.0)));
        masks.dedup();
    }

    fn join_keep(&self, a: Option<GroupKeep>, b: Option<GroupKeep>) -> Option<GroupKeep> {
        match (a, b) {
            (None, x) | (x, None) => x,
            (Some(GroupKeep::AnyLane), _) | (_, Some(GroupKeep::AnyLane)) => {
                Some(GroupKeep::AnyLane)
            }
            (Some(GroupKeep::Outside(ma)), Some(GroupKeep::Outside(mb))) => {
                // Sound intersection of `(⋃ ma) ∩ (⋃ mb)`: keep every mask from
                // either side that is covered by at least one mask on the other.
                let mut both = Vec::new();
                for &x in &ma {
                    if mb.iter().any(|&y| self.implies(x, y)) {
                        self.add_covered(&mut both, x);
                    }
                }
                for &y in &mb {
                    if ma.iter().any(|&x| self.implies(y, x)) {
                        self.add_covered(&mut both, y);
                    }
                }
                if both.is_empty() {
                    Some(GroupKeep::AnyLane)
                } else {
                    Some(GroupKeep::Outside(both))
                }
            }
        }
    }
}

/// Checks that the mesh group input (`v5`) is never consumed per lane,
/// including after partial-`EXEC` overwrites of `v5` whose inactive lanes still
/// retain the original group input. Returns the offending instruction offset if
/// a per-lane read is not proven covered by a prior full overwrite.
fn mesh_group_per_lane_read(p: &Program) -> Option<usize> {
    let exec = ExecTracker::new(p);

    let mut v5_group: HashMap<Value, GroupKeep> = HashMap::new();
    for &(r, v) in &p.inputs {
        if r == Reg::V(5) {
            v5_group.insert(v, GroupKeep::AnyLane);
        }
    }
    if v5_group.is_empty() {
        return None;
    }

    // Only clear `v5_group` when all hardware lanes (`-1`) are proven overwritten;
    // keeping `Outside([entry_exec])` preserves reads under `re ⊆ entry_exec` while
    // still catching subsequent widening such as `s_or_saveexec_b64 _, -1`.
    let covers_all = |masks: &[ExecMask]| masks.iter().any(|&c| exec.is_all_lanes(c, 8));

    // Returns `true` when `combined` strictly weakens the covered set compared to `prev`,
    // preventing non-terminating oscillation between syntactically distinct equivalent masks.
    let keep_weakened = |prev: Option<&GroupKeep>, combined: &GroupKeep| -> bool {
        match (prev, combined) {
            (None, _) => true,
            (Some(GroupKeep::AnyLane), GroupKeep::AnyLane) => false,
            (Some(GroupKeep::Outside(_)), GroupKeep::AnyLane) => true,
            (Some(GroupKeep::AnyLane), GroupKeep::Outside(_)) => false,
            (Some(GroupKeep::Outside(pm)), GroupKeep::Outside(cm)) => {
                !pm.iter().all(|&p| cm.iter().any(|&c| exec.implies(p, c)))
            }
        }
    };

    // Fixed-point provenance propagation across v5 phis and v5-writing instructions.
    let mut changed = true;
    while changed {
        changed = false;
        for (b, blk) in p.blocks.iter().enumerate() {
            for phi in &blk.phis {
                if phi.reg != Reg::V(5) {
                    continue;
                }
                let mut merged: Option<GroupKeep> = None;
                for (&arg, &pred) in phi.args.iter().zip(&blk.preds) {
                    let arm = match v5_group.get(&arg).cloned() {
                        None => None,
                        Some(keep) => match exec.no_lanes_exec(pred, b) {
                            // Along an EXEC-zero branch, every lane is outside `zero_exec`,
                            // so carrying `v5` over this edge only retains `MeshGroup` in
                            // lanes outside `zero_exec`.
                            Some(zero_exec) => {
                                let mut m = match keep {
                                    GroupKeep::AnyLane => Vec::new(),
                                    GroupKeep::Outside(prev) => prev,
                                };
                                exec.add_covered(&mut m, zero_exec);
                                if covers_all(&m) {
                                    None
                                } else {
                                    Some(GroupKeep::Outside(m))
                                }
                            }
                            None => Some(keep),
                        },
                    };
                    merged = exec.join_keep(merged, arm);
                }
                if let Some(next) = merged {
                    let prev = v5_group.get(&phi.value).cloned();
                    let combined = exec.join_keep(prev.clone(), Some(next)).unwrap();
                    if keep_weakened(prev.as_ref(), &combined) {
                        v5_group.insert(phi.value, combined);
                        changed = true;
                    }
                }
            }
            for inst in &blk.insts {
                let Some(out_v) = inst.write(Reg::V(5)) else {
                    continue;
                };
                let Some(old_keep) = inst.read(Reg::V(5)).and_then(|v| v5_group.get(&v).cloned())
                else {
                    continue;
                };
                let full_overwrite = !operand_reads(p, inst).contains(&Reg::V(5));
                let next = if !full_overwrite {
                    Some(old_keep)
                } else {
                    match exec.exec_of(inst) {
                        Some(we) => {
                            let mut covered = match old_keep {
                                GroupKeep::AnyLane => Vec::new(),
                                GroupKeep::Outside(prev) => prev,
                            };
                            exec.add_covered(&mut covered, we);
                            if covers_all(&covered) {
                                None
                            } else {
                                Some(GroupKeep::Outside(covered))
                            }
                        }
                        None => Some(old_keep),
                    }
                };
                if let Some(next) = next {
                    let prev = v5_group.get(&out_v).cloned();
                    let combined = exec.join_keep(prev.clone(), Some(next)).unwrap();
                    if keep_weakened(prev.as_ref(), &combined) {
                        v5_group.insert(out_v, combined);
                        changed = true;
                    }
                }
            }
        }
    }

    for blk in &p.blocks {
        for inst in &blk.insts {
            let Some(keep) = inst.read(Reg::V(5)).and_then(|v| v5_group.get(&v)) else {
                continue;
            };
            if inst.name() == "v_readfirstlane_b32" {
                continue;
            }
            let uses_old_v5 =
                inst.write(Reg::V(5)).is_none() || operand_reads(p, inst).contains(&Reg::V(5));
            if !uses_old_v5 {
                continue;
            }
            // Cross-lane instructions that can read from an inactive lane outside
            // `exec_of(inst)` are never covered by a partial-EXEC overwrite of v5.
            // (`ds_swizzle_b32`, `ds_bpermute_b32`, and `fi:0` DPP/permlane ballot
            // source-lane activity in `emit/lower.rs` and yield 0 or preserve dst
            // for inactive source lanes, so they never read `MeshGroup` from an
            // uncovered lane.)
            let cross_lane_unchecked = match &inst.fields {
                Fields::Valu(f) => {
                    inst.name() == "v_readlane_b32"
                        || (matches!(inst.name(), "v_permlane16_b32" | "v_permlanex16_b32")
                            && f.opsel & 1 != 0)
                        || (f.dpp.is_some_and(|d| {
                            matches!(
                                d,
                                crate::gfx10::Dpp::Row { fi: true, .. }
                                    | crate::gfx10::Dpp::Lanes { fi: true, .. }
                            )
                        }) && matches!(
                            f.src.first(),
                            Some(Operand::Vector { reg: 5, .. })
                        ))
                }
                Fields::Ds { .. } => inst.name() == "ds_permute_b32",
                _ => false,
            };
            let covered = match keep {
                GroupKeep::AnyLane => false,
                GroupKeep::Outside(_) if cross_lane_unchecked => false,
                GroupKeep::Outside(masks) => exec
                    .exec_of(inst)
                    .is_some_and(|re| masks.iter().any(|&we| exec.implies(re, we))),
            };
            if !covered {
                return Some(inst.machine.offset);
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::native::ExportPatch;

    const S_ENDPGM: u32 = 0xBF81_0000;

    fn program(words: &[u32]) -> Program {
        let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        crate::ir::build(&code, false, 256).unwrap()
    }

    fn pixel(inputs: Vec<PsInput>, input_addr: u32) -> NativeProgram {
        NativeProgram {
            stage: HwStage::Ps,
            wave64: false,
            vgprs: 256,
            layout: vec![],
            used: vec![],
            io: StageIo::Ps {
                inputs,
                input_addr,
                db_shader_control: 0,
                patches: Vec::<ExportPatch>::new(),
            },
            code: vec![],
        }
    }

    const TEXCOORD: PsInput = PsInput {
        usage: Usage::Generic,
        row: 1,
        mask: 3,
        attr: 0,
        flat: false,
        fp16: false,
        pack2: false,
        row2: 1,
    };

    #[test]
    fn interpolation_pairs_and_color_exports() {
        // Blit's pixel shader: TEXCOORD.xy from the perspective-center i/j
        // in v0/v1, then exported unconverted to MRT0.
        let p = program(&[
            0xC808_0000, // v_interp_p1_f32 v2, v0, attr0.x
            0xC809_0001, // v_interp_p2_f32 v2, v1, attr0.x
            0xF800_180F,
            0x0302_0102, // exp mrt0, v2, v1, v2, v3 done vm
            S_ENDPGM,
        ]);
        let io = raise(&p, &pixel(vec![TEXCOORD], 2)).unwrap();
        let input = |reg| p.inputs.iter().find(|i| i.0 == Reg::V(reg)).unwrap().1;
        let center = |index| Input::Barycentric {
            kind: Barycentric::PerspectiveCenter,
            index,
        };
        assert!(io.inputs.contains(&(input(0), center(0))));
        assert!(io.inputs.contains(&(input(1), center(1))));
        // v2 and v3 are read before written but hold nothing.
        assert!(!io.inputs.iter().any(|i| i.0 == input(3)));
        assert_eq!(
            io.interps,
            [(
                (0, 1),
                Interp::Interpolated {
                    attribute: Attribute {
                        usage: Usage::Generic,
                        row: 1,
                        channel: 0,
                        high: false
                    },
                    barycentric: Barycentric::PerspectiveCenter,
                    first: vec![(0, 0)],
                }
            )]
        );
        let [e] = &io.exports[..] else {
            panic!("{:?}", io.exports)
        };
        assert_eq!(
            (e.target, e.done, e.compressed),
            (Target::Color(0), true, false)
        );
        assert_eq!(e.values[0], p.blocks[0].insts[1].write(Reg::V(2)));
    }

    #[test]
    fn unpaired_or_unknown_interpolation_fails() {
        // p2 alone: its old dst is no p1's result.
        let lone = program(&[0xC809_0001, S_ENDPGM]); // v_interp_p2_f32 v2, v1, attr0.x
        assert!(raise(&lone, &pixel(vec![TEXCOORD], 2)).is_err());
        // attr1 is not in the stage data.
        let other = program(&[0xC808_0400, S_ENDPGM]); // v_interp_p1_f32 v2, v0, attr1.x
        assert!(raise(&other, &pixel(vec![TEXCOORD], 2)).is_err());
    }

    #[test]
    fn computed_barycentrics_evaluate_each_half() {
        // i scaled before the first half: both halves stand on their own.
        let p = program(&[
            0x1006_00F0, // v_mul_f32 v3, 0.5, v0
            0xC808_0003, // v_interp_p1_f32 v2, v3, attr0.x
            0xC809_0001, // v_interp_p2_f32 v2, v1, attr0.x
            S_ENDPGM,
        ]);
        let io = raise(&p, &pixel(vec![TEXCOORD], 2)).unwrap();
        let attribute = Attribute {
            usage: Usage::Generic,
            row: 1,
            channel: 0,
            high: false,
        };
        let mut interps = io.interps.clone();
        interps.sort_by_key(|i| i.0);
        assert_eq!(
            interps,
            [
                ((0, 1), Interp::Weighted { attribute }),
                ((0, 2), Interp::Weighted { attribute }),
            ]
        );
    }

    #[test]
    fn first_halves_read_through_a_phi_keep_their_partial_values() {
        let p = program(&[
            0xBF06_8000, // s_cmp_eq_u32 s0, 0
            0xBF85_0002, // s_cbranch_scc1 -> second first half
            0xC808_0000, // v_interp_p1_f32 v2, v0, attr0.x
            0xBF82_0001, // s_branch -> export
            0xC808_0000, // v_interp_p1_f32 v2, v0, attr0.x
            0xF800_180F,
            0x0202_0202, // exp mrt0, v2, v2, v2, v2 done vm
            S_ENDPGM,
        ]);
        let io = raise(&p, &pixel(vec![TEXCOORD], 2)).unwrap();
        assert_eq!(io.interps.len(), 2);
        assert!(
            io.interps
                .iter()
                .all(|(_, i)| matches!(i, Interp::Weighted { .. }))
        );
    }

    #[test]
    fn a_standalone_f16_first_half_is_weighted() {
        let p = program(&[
            0xD742_0002,
            0x0002_0000, // v_interp_p1ll_f16 v2, v0, attr0.x
            0xF800_180F,
            0x0202_0202,
            S_ENDPGM,
        ]);
        let io = raise(
            &p,
            &pixel(
                vec![PsInput {
                    fp16: true,
                    ..TEXCOORD
                }],
                2,
            ),
        )
        .unwrap();
        assert!(matches!(&io.interps[..], [(_, Interp::Weighted { .. })]));
    }

    #[test]
    fn mesh_inputs_and_primitive_export() {
        let mesh = |words: &[u32]| {
            let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
            let p = crate::ir::build(&code, true, 256).unwrap();
            let native = NativeProgram {
                stage: HwStage::Ms,
                wave64: true,
                vgprs: 256,
                layout: vec![],
                used: vec![],
                io: StageIo::Ms(crate::native::MeshIo {
                    outputs: vec![],
                    clip_cntl: 0,
                    threads: 128,
                    max_vertices: 128,
                    max_primitives: 128,
                    topology: 2,
                    lds_bytes: 0,
                }),
                code,
            };
            raise(&p, &native)
        };
        let io = mesh(&[
            0x7E04_0505, // v_readfirstlane_b32 s2, v5
            0xF800_0941,
            0x0000_0009, // exp prim, v9, off, off, off done
            S_ENDPGM,
        ])
        .unwrap();
        assert!(io.inputs.iter().any(|i| i.1 == Input::MeshGroup));
        assert_eq!(io.exports[0].target, Target::Primitive);
        // The group input is uniform only as the first lane reads it.
        assert!(mesh(&[0x7E02_0305, S_ENDPGM]).is_err()); // v_mov_b32 v1, v5
        // Also as an operand of a write to v5, or after a join.
        let per_lane: [&[u32]; 5] = [
            &[0x4A0A_0305, S_ENDPGM],              // v_add_nc_u32 v5, v5, v1
            &[0x7E0A_0D05, S_ENDPGM],              // v_cvt_f32_u32 v5, v5
            &[0xDC30_8000, 0x0500_0005, S_ENDPGM], // global_load_dword v5, v5, s[0:1]
            &[0x7E0A_CB01, S_ENDPGM],              // v_swap_b32 v5, v1
            &[
                0xBF06_8001, // s_cmp_eq_u32 s1, 0
                0xBF85_0001, // s_cbranch_scc1 -> 0xc
                0x7E0A_0280, // v_mov_b32 v5, 0
                0x7E02_0305, // 0xc: v_mov_b32 v1, v5
                S_ENDPGM,
            ],
        ];
        let accepted: Vec<_> = per_lane.iter().filter(|w| mesh(w).is_ok()).collect();
        assert!(accepted.is_empty(), "accepted: {accepted:08X?}");
        // Overwriting v5 reads the group input only for lanes outside EXEC:
        // v_mov_b32 v5, 0; v_mov_b32 v1, v5.
        assert!(mesh(&[0x7E0A_0280, 0x7E02_0305, S_ENDPGM]).is_ok());
        // Skipping that write with s_cbranch_execz is the same case: under the
        // same mask, the read sees only the 0.
        let reuse = [
            0x7E04_0505, // v_readfirstlane_b32 s2, v5
            0x7D88_0084, // v_cmp_gt_u32 vcc, 4, v0
            0xBE86_246A, // s_and_saveexec_b64 s[6:7], vcc
            0xBF88_0001, // s_cbranch_execz -> 0x14
            0x7E0A_0280, // v_mov_b32 v5, 0
            0x88FE_067E, // 0x14: s_or_b64 exec, exec, s[6:7]
            0xBE86_246A, // s_and_saveexec_b64 s[6:7], vcc
            0xBF88_0001, // s_cbranch_execz -> 0x24
            0x7E02_0305, // v_mov_b32 v1, v5
            0x88FE_067E, // 0x24: s_or_b64 exec, exec, s[6:7]
            S_ENDPGM,
        ];
        assert_eq!(mesh(&reuse).map(|_| ()), Ok(()));
        // Overwriting v5 with full-32-bit SDWA (DWORD / UNUSED_PAD / UNUSED_SEXT),
        // `fi:0` full-mask `bound_ctrl:1` DPP / DPP8, or full-32-bit "16" mnemonics
        // must NOT be rejected as per-lane v5 reads.
        for ok_overwrite in [
            &[0x7E04_0505, 0x7E0A_0CF9, 0x0005_0602, S_ENDPGM][..], // v_cvt_f32_u32_sdwa dst_sel:DWORD
            &[0x7E04_0505, 0x7E0A_14F9, 0x0006_0602, S_ENDPGM][..], // v_cvt_f16_f32_sdwa dst_sel:DWORD
            &[0x7E04_0505, 0x7E0A_14F9, 0x0006_0C02, S_ENDPGM][..], // v_cvt_f16_f32_sdwa WORD_0 UNUSED_SEXT
            &[0x7E04_0505, 0x7E0A_02FA, 0xFF08_B102, S_ENDPGM][..], // v_mov_b32_dpp quad_perm fi:0 bound_ctrl:1
            &[0x7E04_0505, 0x7E0A_02E9, 0x0000_0002, S_ENDPGM][..], // v_mov_b32_dpp8 fi:0
            &[0x7E04_0505, 0x7E0A_1700, S_ENDPGM][..],              // v_cvt_f32_f16 v5, v0
            &[0x7E04_0505, 0x7E0A_C500, S_ENDPGM][..],              // v_sat_pk_u8_i16 v5, v0
            &[0x7E04_0505, 0x5E0A_0300, S_ENDPGM][..],              // v_cvt_pkrtz_f16_f32 v5, v0, v1
        ] {
            assert!(mesh(ok_overwrite).is_ok(), "rejected: {ok_overwrite:08X?}");
        }
        // True partial overwrites of v5 (SDWA UNUSED_PRESERVE, masked/unbounded/fi:0-bc:0
        // or conservative fi:1 DPP, half-width 16-bit, permlane16 fi=0/bc=0) still read
        // old v5 per lane and must be rejected.
        for bad_partial in [
            &[0x7E04_0505, 0x7E0A_0CF9, 0x0005_1402, S_ENDPGM][..], // v_cvt_f32_u32_sdwa dst_sel:WORD_0 UNUSED_PRESERVE
            &[0x7E04_0505, 0x7E0A_02FA, 0xFF04_B102, S_ENDPGM][..], // v_mov_b32_dpp quad_perm fi:1 stays conservative
            &[0x7E04_0505, 0x7E0A_02EA, 0x0000_0002, S_ENDPGM][..], // v_mov_b32_dpp8fi fi:1 stays conservative
            &[0x7E04_0505, 0x7E0A_02FA, 0x3F08_B102, S_ENDPGM][..], // wave64: row_mask:0x3 masks rows 2..3
            &[0x7E04_0505, 0x7E0A_02FA, 0xEF08_B102, S_ENDPGM][..], // v_mov_b32_dpp masked row_mask:0xe
            &[0x7E04_0505, 0x7E0A_02FA, 0xFF05_0102, S_ENDPGM][..], // v_mov_b32_dpp row_shl:1 bound_ctrl:0
            &[0x7E04_0505, 0x7E0A_02FA, 0xFF00_B102, S_ENDPGM][..], // v_mov_b32_dpp quad_perm fi:0 bound_ctrl:0
            &[0x7E04_0505, 0x640A_0300, S_ENDPGM][..],              // v_add_f16 v5, v0, v1
            &[0x7E04_0505, 0xD777_0005, 0x0004_0100, S_ENDPGM][..], // v_permlane16_b32 fi=0 bc=0
        ] {
            assert!(mesh(bad_partial).is_err(), "accepted: {bad_partial:08X?}");
        }
        // Wave32 mesh with row_mask=0x3 enables both existing rows and must pass.
        let mesh32 = |words: &[u32]| {
            let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
            let p = crate::ir::build(&code, false, 256).unwrap();
            let native = NativeProgram {
                stage: HwStage::Ms,
                wave64: false,
                vgprs: 256,
                layout: vec![],
                used: vec![],
                io: StageIo::Ms(crate::native::MeshIo {
                    outputs: vec![],
                    clip_cntl: 0,
                    threads: 128,
                    max_vertices: 128,
                    max_primitives: 128,
                    topology: 2,
                    lds_bytes: 0,
                }),
                code,
            };
            raise(&p, &native)
        };
        assert!(mesh32(&[0x7E04_0505, 0x7E0A_02FA, 0x3F08_B102, S_ENDPGM]).is_ok());

        // Partial-EXEC overwrite followed by EXEC restore and per-lane read of v5:
        // inactive lanes during the write retained the original mesh group input,
        // so reading v5 after restoring EXEC must be rejected!
        let partial_exec_leak = [
            0x7E04_0505, // v_readfirstlane_b32 s2, v5
            0x7D88_0084, // v_cmp_gt_u32 vcc, 4, v0
            0xBE86_246A, // s_and_saveexec_b64 s[6:7], vcc
            0x7E0A_1700, // v_cvt_f32_f16 v5, v0 (only writes active lanes!)
            0x88FE_067E, // s_or_b64 exec, exec, s[6:7] (restores inactive lanes!)
            0x7E02_0305, // v_mov_b32 v1, v5 (reads retained MeshGroup in lanes outside vcc!)
            S_ENDPGM,
        ];
        assert!(mesh(&partial_exec_leak).is_err());

        // Reading v5 under a DIFFERENT mask after restore must also be rejected.
        let partial_exec_different_mask = [
            0x7E04_0505, // v_readfirstlane_b32 s2, v5
            0x7D88_0084, // v_cmp_gt_u32 vcc, 4, v0
            0xBE86_246A, // s_and_saveexec_b64 s[6:7], vcc
            0x7E0A_1700, // v_cvt_f32_f16 v5, v0
            0x88FE_067E, // s_or_b64 exec, exec, s[6:7]
            0x7D88_0088, // v_cmp_gt_u32 vcc, 8, v0 (wider mask!)
            0xBE86_246A, // s_and_saveexec_b64 s[6:7], vcc
            0x7E02_0305, // v_mov_b32 v1, v5
            S_ENDPGM,
        ];
        assert!(mesh(&partial_exec_different_mask).is_err());

        // Reading v5 inside the SAME narrowed EXEC block (before restore) is safe.
        let partial_exec_same_block = [
            0x7E04_0505, // v_readfirstlane_b32 s2, v5
            0x7D88_0084, // v_cmp_gt_u32 vcc, 4, v0
            0xBE86_246A, // s_and_saveexec_b64 s[6:7], vcc
            0x7E0A_1700, // v_cvt_f32_f16 v5, v0
            0x7E02_0305, // v_mov_b32 v1, v5 (same EXEC mask as write)
            0x88FE_067E, // s_or_b64 exec, exec, s[6:7]
            S_ENDPGM,
        ];
        assert!(mesh(&partial_exec_same_block).is_ok());

        // Overwriting v5 in BOTH arms of an if/else (`s_and_saveexec` + `s_xor_b64`)
        // covers every active lane of the restored EXEC and must pass.
        let if_else_both_arms = [
            0x7E04_0505, // v_readfirstlane_b32 s2, v5
            0x7D88_0084, // v_cmp_gt_u32 vcc, 4, v0
            0xBE86_246A, // s_and_saveexec_b64 s[6:7], vcc
            0x7E0A_1700, // v_cvt_f32_f16 v5, v0 (then arm)
            0x89FE_067E, // s_xor_b64 exec, exec, s[6:7] (else arm)
            0x7E0A_0280, // v_mov_b32 v5, 0 (else arm)
            0x88FE_067E, // s_or_b64 exec, exec, s[6:7] (restore)
            0x7E02_0305, // v_mov_b32 v1, v5
            S_ENDPGM,
        ];
        assert!(mesh(&if_else_both_arms).is_ok());

        // `s_andn2_saveexec_b64 s[6:7], s[8:9]` sets EXEC = s[8:9] & ~old_exec
        // (complement of old_exec inside s[8:9]), so reading v5 there after only
        // writing under old_exec MUST be rejected!
        let andn2_saveexec_else_leak = [
            0x7E04_0505, // v_readfirstlane_b32 s2, v5
            0xBE88_047E, // s_mov_b64 s[8:9], exec (parent)
            0x7D88_0084, // v_cmp_gt_u32 vcc, 4, v0
            0xBE86_246A, // s_and_saveexec_b64 s[6:7], vcc
            0x7E0A_1700, // v_cvt_f32_f16 v5, v0 (only writes vcc lanes)
            0xBE86_2708, // s_andn2_saveexec_b64 s[6:7], s[8:9] (now ~vcc lanes active!)
            0x7E02_0305, // v_mov_b32 v1, v5 (reads retained MeshGroup!)
            S_ENDPGM,
        ];
        assert!(mesh(&andn2_saveexec_else_leak).is_err());

        // Widening EXEC past entry EXEC via `s_or_saveexec_b64 _, -1` activates
        // previously inactive lanes that still hold MeshGroup and must be rejected.
        let widen_past_entry_leak = [
            0x7E04_0505, // v_readfirstlane_b32 s2, v5
            0x7E0A_0280, // v_mov_b32 v5, 0 (writes only entry EXEC lanes)
            0xBE86_25C1, // s_or_saveexec_b64 s[6:7], -1 (enables all 64 lanes!)
            0x7E02_0305, // v_mov_b32 v1, v5 (inactive entry lanes retained MeshGroup!)
            S_ENDPGM,
        ];
        assert!(mesh(&widen_past_entry_leak).is_err());

        // Cross-lane unchecked reads (`v_readlane_b32`, DPP `fi:1`) can read an
        // inactive lane outside a partial-EXEC overwrite of v5 and must be rejected.
        let cross_lane_readlane_leak = [
            0x7E04_0505, // v_readfirstlane_b32 s2, v5
            0x7D88_0084, // v_cmp_gt_u32 vcc, 4, v0
            0xBE86_246A, // s_and_saveexec_b64 s[6:7], vcc
            0x7E0A_1700, // v_cvt_f32_f16 v5, v0 (only writes vcc lanes)
            0xD760_0004,
            0x0001_0105, // v_readlane_b32 s4, v5, 0 (ignores EXEC!)
            S_ENDPGM,
        ];
        assert!(mesh(&cross_lane_readlane_leak).is_err());
        let cross_lane_dpp_fi1_leak = [
            0x7E04_0505, // v_readfirstlane_b32 s2, v5
            0x7D88_0084, // v_cmp_gt_u32 vcc, 4, v0
            0xBE86_246A, // s_and_saveexec_b64 s[6:7], vcc
            0x7E0A_1700, // v_cvt_f32_f16 v5, v0 (only writes vcc lanes)
            0x7E02_02FA,
            0xFF04_B105, // v_mov_b32_dpp v1, v5 quad_perm:[1,0,3,2] fi:1 (fetches inactive lanes!)
            S_ENDPGM,
        ];
        assert!(mesh(&cross_lane_dpp_fi1_leak).is_err());
        // `ds_swizzle_b32` under the same EXEC mask is safe because `emit/lower.rs`
        // ballots source-lane activity and returns 0 for inactive source lanes;
        // reading through `ds_swizzle_b32` after restoring EXEC still fails.
        let cross_lane_swizzle_same_exec = [
            0x7E04_0505, // v_readfirstlane_b32 s2, v5
            0x7D84_0080, // v_cmp_eq_u32 vcc, 0, v0 (lane 0 only)
            0xBE86_246A, // s_and_saveexec_b64 s[6:7], vcc
            0x7E0A_1700, // v_cvt_f32_f16 v5, v0 (only writes lane 0)
            0xD8D4_80B1,
            0x0100_0005, // ds_swizzle_b32 v1, v5 offset:0x80B1 (lane 0 reads lane 1 -> 0 in lower)
            S_ENDPGM,
        ];
        assert_eq!(mesh(&cross_lane_swizzle_same_exec).map(|_| ()), Ok(()));
        let cross_lane_swizzle_after_restore = [
            0x7E04_0505, // v_readfirstlane_b32 s2, v5
            0x7D84_0080, // v_cmp_eq_u32 vcc, 0, v0 (lane 0 only)
            0xBE86_246A, // s_and_saveexec_b64 s[6:7], vcc
            0x7E0A_1700, // v_cvt_f32_f16 v5, v0 (only writes lane 0)
            0x88FE_067E, // s_or_b64 exec, exec, s[6:7] (restores inactive lanes!)
            0xD8D4_80B1,
            0x0100_0005, // ds_swizzle_b32 v1, v5 offset:0x80B1 (reads MeshGroup from lane 1!)
            S_ENDPGM,
        ];
        assert!(mesh(&cross_lane_swizzle_after_restore).is_err());

        // Unconditional u32 shift theorem: `v0 < (s4 << s5) ==> (v0 >> s5) < s4`.
        let shift_cmp_safe = [
            0x7E04_0505, // v_readfirstlane_b32 s2, v5
            0xBEFE_04C1, // s_mov_b64 exec, -1
            0x2C02_0005, // v_lshrrev_b32 v1, s5, v0 (v1 = v0 >> s5)
            0x7DA8_0204, // v_cmpx_gt_u32 exec, s4, v1 (exec = (v0 >> s5) < s4)
            0x7E0A_1700, // v_cvt_f32_f16 v5, v0 (writes v5 under (v0 >> s5) < s4)
            0xBEFE_04C1, // s_mov_b64 exec, -1
            0x8F08_0504, // s_lshl_b32 s8, s4, s5 (s8 = s4 << s5)
            0x7D88_0008, // v_cmp_gt_u32 vcc, s8, v0 (vcc = v0 < (s4 << s5))
            0xBE86_246A, // s_and_saveexec_b64 s[6:7], vcc
            0x7E02_0305, // v_mov_b32 v1, v5 (every active lane satisfies (v0 >> s5) < s4!)
            S_ENDPGM,
        ];
        assert_eq!(mesh(&shift_cmp_safe).map(|_| ()), Ok(()));

        // `v_cmp_*` produces `EXEC & P` (inactive lanes are 0 bits), so repeating
        // the same comparison under a wider EXEC must NOT be proven covered by
        // the narrower write mask.
        let vcmp_different_exec_leak = [
            0x7E04_0505, // v_readfirstlane_b32 s2, v5
            0xBE88_047E, // s_mov_b64 s[8:9], exec (save initial EXEC)
            0xBEFE_0481, // s_mov_b64 exec, 1 (lane 0 only!)
            0x7D88_0084, // v_cmp_gt_u32 vcc, 4, v0 (W = (v0 < 4) & {lane 0})
            0xBEFE_046A, // s_mov_b64 exec, vcc (EXEC = W)
            0x7E0A_0280, // v_mov_b32 v5, 0 (writes only lane 0!)
            0xBEFE_0408, // s_mov_b64 exec, s[8:9] (restore initial EXEC)
            0x7D88_0084, // v_cmp_gt_u32 vcc, 4, v0 (R = (v0 < 4) & initial EXEC)
            0xBEFE_046A, // s_mov_b64 exec, vcc (EXEC = R, enables lanes 1..3!)
            0x7E02_0305, // v_mov_b32 v1, v5 (lanes 1..3 retained MeshGroup!)
            S_ENDPGM,
        ];
        assert!(mesh(&vcmp_different_exec_leak).is_err());
    }

    #[test]
    fn pixel_vgprs_follow_the_input_address() {
        // Linear center (bit 5), then position x and y (bits 8, 9).
        let v = ps_vgprs(1 << 5 | 1 << 8 | 1 << 9);
        assert_eq!(
            v,
            [
                (
                    0,
                    Input::Barycentric {
                        kind: Barycentric::LinearCenter,
                        index: 0
                    }
                ),
                (
                    1,
                    Input::Barycentric {
                        kind: Barycentric::LinearCenter,
                        index: 1
                    }
                ),
                (2, Input::Position(0)),
                (3, Input::Position(1)),
            ]
        );
    }
}
