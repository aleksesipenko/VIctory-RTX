//! Identify the direct first-iteration path from the guest TLAS root. Keep
//! both guest EXEC branches, including the iteration-limit check, in the IR.
use super::*;

pub(crate) fn peel(p: &Program, st: &Structured, res: &Resources, threads: u32) -> Option<Program> {
    if p.wave64
        || !res
            .bvh
            .iter()
            .any(|&((b, i), _)| p.blocks[b].insts[i].name() == "image_bvh64_intersect_ray")
    {
        return None;
    }
    let scalar = res.scalar.iter().copied().collect();
    let a = Analyzer::new(p, st, res, &scalar);
    let regions = a
        .candidates()
        .into_iter()
        .filter_map(|(h, nodes, bvhs)| {
            let [site] = bvhs[..] else { return None };
            let (pre, dispatch, root) = a.root_dispatch(h, &nodes, site)?;
            super::initial::prefix(&a, (pre, h), dispatch, site.0, root, &nodes, threads)
        })
        .collect::<Vec<_>>();
    if std::env::var_os("VICTORY_WAVE_WHY").is_some() {
        eprintln!("Root dispatch versions: {}", regions.len())
    }
    crate::ir::peel_ray_dispatch(p, &regions)
}

impl Analyzer<'_> {
    fn root_dispatch(
        &self,
        h: usize,
        nodes: &HashSet<usize>,
        site: Site,
    ) -> Option<(usize, usize, Value)> {
        let head = &self.p.blocks[h];
        let b = site.0;
        let Exit::Branch { taken: d, .. } = head.exit else {
            return None;
        };
        let dispatch = &self.p.blocks[d];
        if !nodes.contains(&d)
            || h == d
            || d == b
            || h == b
            || head.insts.last()?.name() != "s_cbranch_execz"
            || dispatch.insts.last()?.name() != "s_cbranch_execz"
            || !matches!(dispatch.exit,Exit::Branch{not_taken,..} if not_taken==b)
        {
            return None;
        }
        let t = self.find_tlas_load(RayTemplate::Bvh64, site, nodes, h)?;
        let t = &self.p.blocks[t.0].insts[t.1];
        let tlas = (t.writes.first()?.1, t.writes.get(1)?.1);
        let incoming = head
            .preds
            .iter()
            .copied()
            .filter(|x| !nodes.contains(x))
            .collect::<Vec<_>>();
        let [pre] = incoming[..] else { return None };
        let path = [(h, pre), (d, h), (b, d)];
        let cmp = head.insts.iter().rev().nth(1)?;
        let (one, node) = two_src_operands(cmp)?;
        if cmp.name() != "v_cmpx_gt_i32" || !address::plain(cmp) || const_operand(one) != Some(1) {
            return None;
        }
        let root = self.path_word(Word::operand(cmp, node)?, &path, 0)?;
        let Word::Ssa(r) = root else { return None };
        let ((rb, ri), load) = self.def_inst(r)?;
        if !self.dominates(rb, h)
            || nodes.contains(&rb)
            || self.st.guards[rb][ri].is_some()
            || load.name() != "global_load_dword"
        {
            return None;
        }
        // A scalar descriptor and a fixed byte offset identify one immutable
        // TLAS root for this wave. Even a dominating VGPR load keeps stale
        // data in lanes it did not execute. Use the preheader's actual EXEC,
        // never the mask from a traversal backedge.
        if !self.mask_covers(load, cmp, &path) {
            return None;
        }
        let (lo, hi) = flat_address(load)?;
        let (_, offset) = flat_or_smem_load_info(load, lo)?;
        if self
            .extract_base_low_offset(lo, tlas.0, 0)?
            .checked_add(offset)
            != Some(48)
            || !self.tlas_high_byte(Word::Ssa(hi), tlas.1, 0)
        {
            return None;
        }

        let image = &self.p.blocks[b].insts[site.1];
        if image.name() != "image_bvh64_intersect_ray" {
            return None;
        }
        let args = image_addr_values(image)?;
        let low = self.unguarded_def(args[0])?;
        let high = self.unguarded_def(args[1])?;
        if !self.is_wide_sum(low, high, args[0], args[1]) {
            return None;
        }
        let (x, y) = two_src_operands(low)?;
        let (x, y) = (
            self.path_word(Word::operand(low, x)?, &path, 0)?,
            self.path_word(Word::operand(low, y)?, &path, 0)?,
        );
        let Word::Ssa(shift) = (if x == root {
            y
        } else if y == root {
            x
        } else {
            return None;
        }) else {
            return None;
        };
        let sh = self.unguarded_def(shift)?;
        if sh.name() != "v_lshlrev_b64" || !address::plain(sh) || sh.writes[0].1 != shift {
            return None;
        }
        if ![low, high, sh]
            .iter()
            .all(|i| self.mask_covers(i, image, &path))
        {
            return None;
        }
        let (k, pair) = two_src_operands(sh)?;
        if const_operand(k) != Some(5)
            || self.path_word(Word::Ssa(upper_operand(sh, pair)?), &path, 0)? != Word::Imm(0)
        {
            return None;
        }
        let Word::Ssa(base) = self.path_word(Word::operand(sh, pair)?, &path, 0)? else {
            return None;
        };
        let shr = self.unguarded_def(base)?;
        let (pair, k) = two_src_operands(shr)?;
        if shr.name() != "s_lshr_b64"
            || shr.writes[0].1 != base
            || const_operand(k) != Some(8)
            || self.extract_base_low_offset(operand_val(shr, pair)?, tlas.0, 0)? != 0
            || !self.tlas_high_byte(Word::Ssa(upper_operand(shr, pair)?), tlas.1, 0)
        {
            return None;
        }
        let (x, y) = two_src_operands(high)?;
        let upper = Word::Ssa(sh.writes.get(1)?.1);
        let (x, y) = (
            self.path_word(Word::operand(high, x)?, &path, 0)?,
            self.path_word(Word::operand(high, y)?, &path, 0)?,
        );
        ((x == upper && y == Word::Imm(0)) || (y == upper && x == Word::Imm(0)))
            .then_some((pre, d, r))
    }

    pub(super) fn mask_covers(
        &self,
        writer: &Inst,
        reader: &Inst,
        path: &[(usize, usize)],
    ) -> bool {
        let mask = |i: &Inst| {
            let edge = |r: Reg| {
                let mut v = i.read(r)?;
                for _ in 0..path.len() {
                    let Some(Def::Phi(b, k)) = self.defs[v.0 as usize] else {
                        break;
                    };
                    let Some(&(_, pred)) = path.iter().find(|e| e.0 == b) else {
                        break;
                    };
                    let n = self.p.blocks[b].preds.iter().position(|&p| p == pred)?;
                    v = self.p.blocks[b].phis[k].args[n];
                }
                Some(v)
            };
            let lo = edge(Reg::S(EXEC))?;
            let hi = if self.p.wave64 {
                Some(edge(Reg::S(EXEC + 1))?)
            } else {
                None
            };
            Some((lo, hi))
        };
        let (Some(w), Some(r)) = (mask(writer), mask(reader)) else {
            return false;
        };
        w == r
            || matches!((self.st.mask_bits(w.0,w.1),self.st.mask_bits(r.0,r.1)),
           (Some(w),Some(r)) if !self.bools_meet(&[(r,true),(w,false)]))
    }

    /// PHIs are evaluated on these particular edges, never on a back edge.
    /// An unknown select is a copy only if its two data operands agree.
    fn path_word(&self, w: Word, path: &[(usize, usize)], depth: usize) -> Option<Word> {
        if depth > 16 {
            return None;
        }
        let w = self.copy_word(w);
        let Word::Ssa(v) = w else { return Some(w) };
        match self.defs[v.0 as usize]? {
            Def::Phi(b, i) if let Some(&(_, pred)) = path.iter().find(|e| e.0 == b) => {
                let idx = self.p.blocks[b].preds.iter().position(|p| *p == pred)?;
                self.path_word(
                    Word::Ssa(self.p.blocks[b].phis[i].args[idx]),
                    path,
                    depth + 1,
                )
            }
            Def::Inst(b, i)
                if self.p.blocks[b].insts[i].name() == "v_cndmask_b32"
                    && self.st.guards[b][i].is_none() =>
            {
                let inst = &self.p.blocks[b].insts[i];
                if !address::plain(inst) {
                    return None;
                }
                let (a, b) = two_src_operands(inst)?;
                let a = self.path_word(Word::operand(inst, a)?, path, depth + 1)?;
                (a == self.path_word(Word::operand(inst, b)?, path, depth + 1)?).then_some(a)
            }
            _ => Some(w),
        }
    }
}

pub(super) fn upper_operand(inst: &Inst, op: &Operand) -> Option<Value> {
    match *op {
        Operand::Vector { reg, dwords: 2 } => inst.read(Reg::V(reg.checked_add(1)?)),
        Operand::Scalar { code, dwords: 2 } => inst.read(Reg::S(code.checked_add(1)?)),
        _ => None,
    }
}

fn flat_address(inst: &Inst) -> Option<(Value, Value)> {
    let Fields::Flat {
        addr: Some(ref op @ Operand::Vector { dwords: 2, .. }),
        saddr: Some(Operand::Scalar { code: NULL, .. }),
        ..
    } = inst.fields
    else {
        return None;
    };
    Some((operand_val(inst, op)?, upper_operand(inst, op)?))
}
