//! A bounded first-iteration proof. Keep the guest prefix instructions; prune
//! an edge only when no incoming lane/root/limit case can take it.
use super::*;
use crate::ir::RayPrefix;
use std::collections::BTreeMap;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Fact {
    Unknown,
    Int(u32),
    // The incoming EXEC mask, as a mask (never an integer value).
    Incoming,
    Outside,    // bitwise complement of the incoming EXEC mask
    Root(bool), // signed root > 0
}
use Fact::{Incoming, Int, Root, Unknown};

struct First<'a, 'p> {
    a: &'a Analyzer<'p>,
    nodes: &'a HashSet<usize>,
    root: Value,
    incoming: Value,
    lane: u32,
    positive: bool,
    live: bool,
    limit: bool,
    words: HashMap<Value, Fact>,
}

impl First<'_, '_> {
    fn word(&self, v: Value, depth: u32) -> Fact {
        if depth > 16 {
            return Unknown;
        }
        if let Some(&x) = self.words.get(&v) {
            return x;
        }
        if v == self.incoming {
            return if self.live { Incoming } else { Int(0) };
        }
        if v == self.root {
            return Root(self.positive);
        }
        let Some(((b, i), inst)) = self.a.def_inst(v) else {
            return Unknown;
        };
        if self.nodes.contains(&b)
            || self.a.st.guards[b][i].is_some()
            || !address::plain(inst)
            || inst.writes.first().map(|r| r.1) != Some(v)
        {
            return Unknown;
        }
        let operand = |op: &Operand| self.operand(inst, op, depth + 1);
        match inst.name() {
            "s_mov_b32" | "v_mov_b32" => first_src_operand(inst).map_or(Unknown, operand),
            "s_movk_i32" => {
                if let Fields::Sop { imm: Some(k), .. } = inst.fields {
                    Int(k as i16 as i32 as u32)
                } else {
                    Unknown
                }
            }
            "v_mbcnt_lo_u32_b32"
                if two_src_operands(inst)
                    .is_some_and(|(m, k)| operand(m) == Int(u32::MAX) && operand(k) == Int(0)) =>
            {
                Int(self.lane)
            }
            "v_lshlrev_b32" => {
                if let Some((k, src)) = two_src_operands(inst)
                    && let (Int(k), Int(x)) = (operand(k), operand(src))
                {
                    Int(x.wrapping_shl(k))
                } else {
                    Unknown
                }
            }
            _ => Unknown,
        }
    }
    fn operand(&self, i: &Inst, op: &Operand, depth: u32) -> Fact {
        match Word::operand(i, op) {
            Some(Word::Imm(k)) => Int(k),
            Some(Word::Ssa(v)) => self.word(v, depth),
            None => Unknown,
        }
    }
    fn mask(&self, f: Fact) -> Option<bool> {
        match f {
            Int(0) => Some(false),
            Incoming => Some(true),
            Int(u32::MAX) => Some(true),
            _ => None,
        }
    }
    fn lane_mask(&self, on: bool) -> Fact {
        if on && self.live { Incoming } else { Int(0) }
    }
    fn run_inst(&mut self, i: &Inst, dispatch: bool, pop: bool) -> Option<()> {
        let exec = i.read(Reg::S(EXEC)).map(|v| self.word(v, 0));
        let ops = match &i.fields {
            Fields::Valu(f) => f.src.as_slice(),
            Fields::Sop { src, .. } => src.as_slice(),
            _ => &[],
        };
        let arg = |n: usize| ops.get(n).map_or(Unknown, |o| self.operand(i, o, 0));
        let (a, b, c) = (arg(0), arg(1), arg(2));
        let scc = i.read(Reg::Scc).map(|v| self.word(v, 0));
        let mut condition = None;
        let val = if !address::plain(i) {
            Unknown
        } else {
            match (i.name(), a, b, c) {
                ("s_mov_b32" | "v_mov_b32", a, _, _) => a,
                ("s_add_i32" | "s_add_u32" | "v_add_nc_u32", Int(a), Int(b), _) => {
                    Int(a.wrapping_add(b))
                }
                ("v_and_b32" | "s_and_b32", Int(a), Int(b), _) => Int(a & b),
                ("v_lshrrev_b32", Int(n), Int(x), _) => Int(x.wrapping_shr(n)),
                ("v_bfe_u32", Int(x), Int(k), Int(n)) => {
                    Int((x >> (k & 31)) & ((1u32 << (n & 31)) - 1))
                }
                ("s_xor_b32", Int(a), Int(b), _) => Int(a ^ b),
                ("s_xor_b32", Incoming, Int(u32::MAX), _) => Fact::Outside,
                ("s_and_saveexec_b32", Fact::Outside, _, _) if exec == Some(Incoming) => Int(0),
                ("s_andn2_b32", a, b, _) if a != Unknown && a == b => Int(0),
                ("s_andn2_b32", a, Int(0), _) => a,
                ("s_and_b32", a, b, _) if a != Unknown && a == b => a,
                ("s_and_b32", Int(0), _, _) | ("s_and_b32", _, Int(0), _) => Int(0),
                ("s_cselect_b32", a, b, _) => match scc {
                    Some(Int(1)) => a,
                    Some(Int(0)) => b,
                    _ if a == b => a,
                    _ => Unknown,
                },
                ("v_cndmask_b32", a, b, m) => match self.mask(m) {
                    Some(true) => b,
                    Some(false) => a,
                    None if a == b => a,
                    _ => Unknown,
                },
                ("s_cmpk_lt_u32", Int(a), _, _)
                    if let Fields::Sop { imm: Some(k), .. } = i.fields =>
                {
                    condition = Some(a < u32::from(k));
                    Unknown
                }
                ("s_cmpk_gt_u32", Int(a), _, _)
                    if let Fields::Sop { imm: Some(k), .. } = i.fields =>
                {
                    condition = Some(a > u32::from(k));
                    Unknown
                }
                ("s_cmp_le_u32", Int(a), Int(b), _) => {
                    condition = Some(a <= b);
                    Unknown
                }
                (name, _, _, _) if dispatch && name.starts_with("s_cmp") => {
                    condition = Some(self.limit);
                    Unknown
                }
                ("v_cmpx_gt_i32", Int(1), Root(p), _) => self.lane_mask(!p),
                ("v_cmpx_gt_i32", Int(a), Int(b), _) => self.lane_mask((a as i32) > (b as i32)),
                ("v_cmpx_lt_i32", Int(a), Int(b), _) => self.lane_mask((a as i32) < (b as i32)),
                ("v_cmp_gt_u32" | "v_cmpx_gt_u32", Int(a), Int(b), _) => self.lane_mask(a > b),
                ("v_cmp_eq_u32" | "v_cmpx_eq_u32", Int(a), Int(b), _) => self.lane_mask(a == b),
                ("v_cmp_ne_u32" | "v_cmpx_ne_u32", Int(a), Int(b), _) => self.lane_mask(a != b),
                ("s_and_saveexec_b32", Int(u32::MAX), _, _) => exec.unwrap_or(Unknown),
                ("s_and_saveexec_b32", a, _, _) => a,
                _ if pop => Int(u32::MAX),
                _ => Unknown,
            }
        };
        for (idx, &(reg, w)) in i.writes.iter().enumerate() {
            let mut x = if reg == Reg::Scc {
                let truth = condition.or_else(|| match i.name() {
                    "s_andn2_b32" | "s_and_b32" => self.mask(val),
                    "s_and_saveexec_b32" => Some(self.mask(val)? && self.mask(exec?)?),
                    _ => None,
                });
                truth.map_or(Unknown, |c| Int(u32::from(c)))
            } else if i.name() == "s_and_saveexec_b32" && reg != Reg::S(EXEC) {
                exec.unwrap_or(Unknown)
            } else if idx == 0 || i.name() == "s_and_saveexec_b32" && reg == Reg::S(EXEC) {
                val
            } else {
                Unknown
            };
            // Lane transfers ignore EXEC. No result of an unmodelled cross-lane
            // operation may become a known zero when the mask is empty.
            if (i.name().starts_with("v_read") || i.name() == "v_writelane_b32")
                && !matches!(reg, Reg::V(_))
            {
                x = Unknown;
            } else if matches!(i.fields, Fields::Valu(_) | Fields::Ds { .. })
                || i.name() == "s_and_saveexec_b32" && reg == Reg::S(EXEC)
            {
                let enabled = exec.and_then(|v| self.mask(v));
                let old = if matches!(reg, Reg::V(_)) {
                    i.read(reg).map_or(Unknown, |v| self.word(v, 0))
                } else {
                    Int(0)
                };
                x = match enabled {
                    Some(true) => x,
                    Some(false) => old,
                    None if x == old => x,
                    None => Unknown,
                };
            }
            self.words.insert(w, x);
        }
        Some(())
    }
}

fn lds(i: &Inst) -> bool {
    matches!(
        i.fields,
        Fields::Ds { .. } | Fields::Flat { lds: true, .. } | Fields::Buffer { lds: true, .. }
    )
}
fn ds_operands(i: &Inst) -> Option<(Value, Value, u16)> {
    let Fields::Ds {
        gds: false,
        addr: Some(addr),
        ref data,
        offset0,
        offset1,
        ..
    } = i.fields
    else {
        return None;
    };
    Some((
        operand_val(i, &addr)?,
        operand_val(i, data.first()?)?,
        u16::from_le_bytes([offset0, offset1]),
    ))
}

pub(super) fn prefix(
    a: &Analyzer<'_>,
    entry: (usize, usize),
    d: usize,
    bvh: usize,
    root: Value,
    nodes: &HashSet<usize>,
    threads: u32,
) -> Option<RayPrefix> {
    let (pre, h) = entry;
    // A different physical wave could run past its push before this pop.
    if a.p.wave64 || threads == 0 || threads > 32 {
        return None;
    }
    let incoming = a.p.blocks[h]
        .insts
        .iter()
        .find_map(|i| i.read(Reg::S(EXEC)))?;
    let incoming = match a.defs[incoming.0 as usize]? {
        Def::Phi(b, k) if b == h => {
            a.p.blocks[h].phis[k].args[a.p.blocks[h].preds.iter().position(|&x| x == pre)?]
        }
        _ => incoming,
    };
    let init = a.p.blocks[pre].insts.iter().rev().find(|i| lds(i))?;
    if init.name() != "ds_write_b32" || init.read(Reg::S(EXEC)) != Some(incoming) {
        return None;
    }
    let (stack, data, offset) = ds_operands(init)?;
    let mut edges = BTreeMap::<usize, Vec<usize>>::new();
    let mut sentinel_pop = None;
    // Each lane gets a different sentinel slot. Enumerate the low stack-index
    // bits rather than assuming mbcnt is a group-wide thread number.
    let mut slots = HashSet::new();
    let mut uniform = HashMap::new();
    if a.p.blocks[d]
        .insts
        .iter()
        .filter(|i| i.name().starts_with("s_cmp"))
        .count()
        > 1
    {
        return None;
    }
    for lane in 0..32 {
        for positive in [false, true] {
            for limit in [false, true] {
                for live in [false, true] {
                    let mut f = First {
                        a,
                        nodes,
                        root,
                        incoming,
                        lane,
                        positive,
                        live,
                        limit,
                        words: HashMap::new(),
                    };
                    let Int(base) = f.word(stack, 0) else {
                        return None;
                    };
                    if f.word(data, 0) != Int(u32::MAX) || base % 4 != 0 {
                        return None;
                    }
                    let addr = base.checked_add(u32::from(offset))?;
                    if !positive && !limit && !live && !slots.insert(addr) {
                        return None;
                    }
                    let (mut pred, mut b) = (pre, h);
                    let mut path = Vec::new();
                    let mut popped = false;
                    let mut decisions = Vec::new();
                    while nodes.contains(&b) && b != bvh {
                        if path.contains(&b) || path.len() >= 16 {
                            return None;
                        }
                        path.push(b);
                        let bb = &a.p.blocks[b];
                        let edge = bb.preds.iter().position(|&x| x == pred)?;
                        for ph in &bb.phis {
                            let x = f.word(ph.args[edge], 0);
                            f.words.insert(ph.value, x);
                        }
                        for (j, i) in bb.insts.iter().enumerate() {
                            let pop = lds(i);
                            if pop {
                                if popped || i.name() != "ds_wrxchg_rtn_b32" {
                                    return None;
                                }
                                let (s, v, k) = ds_operands(i)?;
                                if !matches!(f.word(s,0),Int(x) if x.checked_add(u32::from(k))==Some(addr))
                                    || f.word(v, 0) != Int(u32::MAX)
                                {
                                    return None;
                                }
                                if sentinel_pop
                                    .replace((b, j))
                                    .is_some_and(|prev| prev != (b, j))
                                {
                                    return None;
                                }
                                popped = true;
                            }
                            f.run_inst(i, b == d, pop)?;
                            if let Some(v) = i.write(Reg::S(EXEC)) {
                                let x = f.word(v, 0);
                                if !matches!(x, Incoming | Int(0)) {
                                    return None;
                                }
                                decisions.push((b, j, x == Incoming));
                            }
                        }
                        let next = match bb.exit {
                            Exit::Jump(x) => x,
                            Exit::Branch { taken, not_taken } => {
                                let last = bb.insts.last()?;
                                let yes = match last.name() {
                                    "s_cbranch_execz" => {
                                        !f.mask(f.word(last.read(Reg::S(EXEC))?, 0))?
                                    }
                                    "s_cbranch_scc1" => f.word(last.read(Reg::Scc)?, 0) == Int(1),
                                    _ => return None,
                                };
                                if last.name() == "s_cbranch_scc1"
                                    && !matches!(f.word(last.read(Reg::Scc)?, 0), Int(0 | 1))
                                {
                                    return None;
                                }
                                if yes { taken } else { not_taken }
                            }
                            _ => return None,
                        };
                        let nexts = edges.entry(b).or_default();
                        if !nexts.contains(&next) {
                            nexts.push(next)
                        }
                        decisions.push((b, next, false));
                        pred = b;
                        b = next;
                    }
                    // A scalar branch observes the entire wave. A set of independent
                    // per-lane paths would not prove what happens in a mixed wave.
                    let key = (positive, limit, live);
                    if let Some(prev) = uniform.get(&key) {
                        if prev != &decisions {
                            return None;
                        }
                    } else {
                        uniform.insert(key, decisions);
                    }
                    if b == bvh && (!positive || !live || path != [h, d]) {
                        return None;
                    }
                }
            }
        }
    }
    Some(RayPrefix {
        preheader: pre,
        header: h,
        edges,
        sentinel_pop,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_root_and_zero_limit_retain_their_guest_exits() {
        let words = [
            0xbe8c_037e, // s12 = incoming EXEC
            0xd765_0002,
            0x0001_00c1,                               // mbcnt(-1,0)
            (0x1a << 25) | (2 << 17) | (2 << 9) | 130, // v2 = lane * 4
            0x7e02_02c1,
            0xd834_0000,
            0x0000_0102, // per-lane sentinel
            0xbf82_0000, // preheader -> header
            0x7c00_0000 | (0x94 << 17) | 129,
            0xbf88_0002, // root <= 0
            0xbe88_0380,
            0xbf82_0008, // null root returns guest status 0
            0xbefe_030c,
            0xbf0b_8001, // restore EXEC, compare limit <= 0
            0x857e_0c80,
            0xbf88_0002, // s_cselect_b32 exec,0,s12; execz
            0xbe88_0382,
            0xbf82_fff6, // reached intersection, loop back
            0xbe88_0381,
            0xbf82_0000, // skipped traversal returns timeout 1
            0xbf81_0000u32,
        ];
        let bytes = words
            .into_iter()
            .flat_map(u32::to_le_bytes)
            .collect::<Vec<_>>();
        let p = crate::ir::build(&bytes, false, 4).unwrap();
        let st = crate::structure::structure(&p).unwrap();
        let res = Resources::default();
        let scalar = HashMap::new();
        let a = Analyzer::new(&p, &st, &res, &scalar);
        let root = p.blocks[1].insts[0].read(Reg::V(0)).unwrap();
        let plan = prefix(&a, (0, 1), 3, 4, root, &HashSet::from([1, 3, 4]), 32).unwrap();
        let q = crate::ir::peel_ray_dispatch(&p, &[plan]).unwrap();
        let Exit::Jump(h) = q.blocks[0].exit else {
            panic!("missing peeled header")
        };
        let Exit::Branch {
            taken: d,
            not_taken: null,
        } = q.blocks[h].exit
        else {
            panic!("lost root branch")
        };
        let Exit::Branch {
            taken: timeout,
            not_taken: trace,
        } = q.blocks[d].exit
        else {
            panic!("lost limit branch")
        };
        assert_eq!((null, timeout, trace), (2, 5, 4));
        assert_eq!(
            q.blocks[null].insts[0].machine.words,
            p.blocks[2].insts[0].machine.words
        );
        assert_eq!(
            q.blocks[timeout].insts[0].machine.words,
            p.blocks[5].insts[0].machine.words
        );
    }
}
