//! Identify the direct first-iteration path from the guest TLAS root. Keep
//! both guest EXEC branches, including the iteration-limit check, in the IR.
use super::*;

pub(crate) fn peel(p: &Program, st: &Structured, res: &Resources, threads: u32) -> Option<Program> {
    if p.wave64 {
        return None;
    }
    if !res
        .bvh
        .iter()
        .any(|&((b, i), _)| p.blocks[b].insts[i].name() == "image_bvh64_intersect_ray")
    {
        if threads == 0 || threads > 32 || res.bvh.len() < 2 {
            return None;
        }
        let scalar = res.scalar.iter().copied().collect();
        let mut selected = None;
        super::isolated_ray_loops(p, st, res, &scalar, |loops| {
            let [r] = loops.as_slice() else {
                return false;
            };
            let Some((guard, pre)) = r.shared_entry else {
                return false;
            };
            let mut nodes: HashSet<_> = r.skipped_blocks.iter().copied().collect();
            nodes.insert(r.entry_block);
            if let RayLoopKind::AnyHit(ah) = &r.kind {
                nodes.extend(&ah.body_blocks);
            }
            selected = crate::ir::version_shared_ray(p, r.entry_block, pre, &nodes, guard);
            selected.is_some()
        });
        return selected;
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
    /// The guard can only specialize this exact first-intersection predicate:
    /// EXEC & (node_type != 6) & bit(work, lane_id). Reading the loop-carried
    /// PHIs here is valid for the body; the new scalar checks run on its sole
    /// external edge, before any header instruction or backedge.
    pub(super) fn shared_entry(
        &self,
        h: usize,
        nodes: &HashSet<usize>,
        bvh: Site,
    ) -> Option<(crate::ir::SharedRayEntry, usize, Value)> {
        if self.p.wave64 {
            return None;
        }
        let bb = &self.p.blocks[h];
        let mut preds = bb.preds.iter().copied().filter(|p| !nodes.contains(p));
        let pre = preds.next()?;
        let arm = &self.p.blocks[bvh.0];
        if preds.next().is_some()
            || arm.preds != [h]
            || !matches!(bb.exit,Exit::Branch{not_taken,..} if not_taken==bvh.0)
            || bb.insts.last()?.name() != "s_cbranch_execz"
        {
            return None;
        }
        // Guard blocks overwrite SCC; the original header must kill it before reading it.
        let first = bb
            .insts
            .iter()
            .find(|i| i.read(Reg::Scc).is_some() || i.write(Reg::Scc).is_some())?;
        if first.read(Reg::Scc).is_some() || first.write(Reg::Scc).is_none() {
            return None;
        }
        let save = bb.insts.iter().rev().nth(1)?;
        let entry = save.read(Reg::S(EXEC))?;
        if save.name() != "s_and_saveexec_b32"
            || save.write(Reg::S(EXEC)) != bb.insts.last()?.read(Reg::S(EXEC))
            || arm.insts[bvh.1].read(Reg::S(EXEC)) != save.write(Reg::S(EXEC))
            || bb.insts[..bb.insts.len() - 2]
                .iter()
                .any(|i| i.write(Reg::S(EXEC)).is_some())
        {
            return None;
        }
        let local = |v: Value| {
            let ((b, _), i) = self.def_inst(v)?;
            (b == h
                && address::plain(i)
                && i.writes.first()?.1 == v
                && (!matches!(i.fields, Fields::Valu(_)) || i.read(Reg::S(EXEC)) == Some(entry)))
            .then_some(i)
        };
        let mask = local(operand_val(save, first_src_operand(save)?)?)?;
        if mask.name() != "s_and_b32" {
            return None;
        }
        let (m, n) = two_src_operands(mask)?;
        let (m, n) = (local(operand_val(mask, m)?)?, local(operand_val(mask, n)?)?);
        let (sel, cmp) = if m.name() == "s_cselect_b32" {
            (m, n)
        } else {
            (n, m)
        };
        let (curr, z) = two_src_operands(sel)?;
        if sel.name() != "s_cselect_b32"
            || operand_val(sel, curr) != Some(entry)
            || const_operand(z) != Some(0)
            || cmp.name() != "v_cmp_ne_u32"
        {
            return None;
        }
        let test = self.def_inst(sel.read(Reg::Scc)?)?.1;
        let (kind, six) = two_src_operands(test)?;
        if test.name() != "s_cmp_lg_u32" || const_operand(six) != Some(6) {
            return None;
        }
        let kind = local(operand_val(test, kind)?)?;
        if kind.name() != "s_and_b32" {
            return None;
        }
        let node = other_val_from_const_op(kind, 7)?;
        let bits = local(other_val_from_const_op(cmp, 0)?)?;
        if bits.name() != "v_and_b32" {
            return None;
        }
        let (l, r) = two_src_operands(bits)?;
        let work = [l, r].into_iter().find_map(|op| {
            matches!(op, Operand::Scalar { .. })
                .then(|| operand_val(bits, op))
                .flatten()
        })?;
        let bit = [l, r].into_iter().find_map(|op| {
            matches!(op, Operand::Vector { .. })
                .then(|| operand_val(bits, op))
                .flatten()
        })?;
        let shift = local(bit)?;
        let (lane, one) = two_src_operands(shift)?;
        if shift.name() != "v_lshlrev_b32" || const_operand(one) != Some(1) {
            return None;
        }
        let ((b, _), lane) = self.def_inst(operand_val(shift, lane)?)?;
        let (ones, zero) = two_src_operands(lane)?;
        if nodes.contains(&b)
            || !self.dominates(b, h)
            || lane.name() != "v_mbcnt_lo_u32_b32"
            || !address::plain(lane)
            || const_operand(ones) != Some(u32::MAX)
            || const_operand(zero) != Some(0)
        {
            return None;
        }
        let initial = match self.defs[entry.0 as usize] {
            Some(Def::Phi(b, j)) if b == h => {
                bb.phis[j].args[bb.preds.iter().position(|&p| p == pre)?]
            }
            _ => entry,
        };
        let lane_is_current = self.mask_covers(lane, shift, &[(h, pre)])
            || self.mask_source(initial, lane.read(Reg::S(EXEC))?, true, 0);
        self.shared_root(h, pre, nodes, bvh, node, work)?;
        let (Some(Reg::S(root)), Some(Reg::S(work))) = (self.value_reg(node), self.value_reg(work))
        else {
            return None;
        };
        if [root, work]
            .into_iter()
            .any(|r| r >= 106 || bb.insts.iter().any(|i| i.write(Reg::S(r)).is_some()))
        {
            return None;
        }
        Some((
            crate::ir::SharedRayEntry {
                root,
                work,
                require_full_work: !lane_is_current,
            },
            pre,
            bit,
        ))
    }

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
