//! Check that a first-accepted triangle actually retires its lane from
//! both the per-lane BLAS and shared TLAS work lists.
use super::*;

impl Analyzer<'_> {
    pub(super) fn shadow_commit(
        &self,
        nodes: &HashSet<usize>,
        tri: &TriangleHitInfo,
        h: usize,
        (primary, leaf): (Site, Site),
        t: Value,
        extent: Value,
    ) -> Option<()> {
        let HitMode::AnyHit(ah) = &tri.mode else {
            return None;
        };
        let op = ah.opacity_block;
        let tail = ah.tail_block;
        let image = &self.p.blocks[leaf.0].insts[leaf.1];
        let outer = &self.p.blocks[primary.0].insts[primary.1];
        if image_addr_values(image)?.get(1) != Some(&extent)
            || image_addr_values(outer)?
                .get(1)
                .and_then(|&v| self.value_reg(v))
                != Some(tri.committed_t_reg)
        {
            return None;
        }
        let (cb, writer) = self.only_writer(nodes, tri.committed_t_reg)?;
        let commit = &self.p.blocks[cb];
        let opacity = &self.p.blocks[op];
        let save = opacity.insts.iter().rev().nth(1)?;
        if commit.preds.len() != 2
            || writer.name() != "v_mov_b32"
            || !address::plain(writer)
            || !self.mask_source(
                writer.read(Reg::S(EXEC))?,
                save.read(Reg::S(EXEC))?,
                false,
                0,
            )
        {
            return None;
        }
        let phi = commit
            .phis
            .iter()
            .find(|ph| Some(ph.value) == single_reg_src(writer))?;
        let o = commit.preds.iter().position(|&p| p == op)?;
        let a = commit.preds.iter().position(|&p| p == tail)?;
        if phi.args[o] != t {
            return None;
        }
        let accepted = |i: &Inst| {
            let Fields::Valu(f) = &i.fields else {
                return None;
            };
            let [no, yes, m] = f.src.as_slice() else {
                return None;
            };
            if i.name() != "v_cndmask_b32"
                || !address::plain(i)
                || operand_val(i, m) != Some(ah.accept_mask)
                || !self.mask_source(i.read(Reg::S(EXEC))?, save.write(Reg::S(EXEC))?, false, 0)
            {
                return None;
            }
            let (yes, no) = if ah.accept_invert {
                (no, yes)
            } else {
                (yes, no)
            };
            Some((Word::operand(i, yes)?, Word::operand(i, no)?))
        };
        let ((b, _), tsel) = self.def_inst(phi.args[a])?;
        if b != tail || accepted(tsel) != Some((Word::Ssa(t), Word::Ssa(extent))) {
            return None;
        }
        let status = self.p.blocks[tail]
            .insts
            .iter()
            .find(|i| accepted(i) == Some((Word::Imm(0), Word::Imm(3))))?;
        let status_reg = status.writes.first()?.0;
        let opaque = opacity
            .insts
            .iter()
            .rev()
            .find(|i| i.write(status_reg).is_some())?;
        if opaque.name() != "v_mov_b32"
            || !address::plain(opaque)
            || const_operand(first_src_operand(opaque)?) != Some(0)
            || opaque.read(Reg::S(EXEC)) != save.read(Reg::S(EXEC))
        {
            return None;
        }
        // The zero produced on the accept edge must survive every later
        // tail instruction, then be the actual SSA input to the latch.
        let accepted_status = status.write(status_reg)?;
        let opaque_status = opaque.write(status_reg)?;
        if self.p.blocks[tail]
            .insts
            .iter()
            .rev()
            .find_map(|i| i.write(status_reg))
            != Some(accepted_status)
        {
            return None;
        }
        let joined = commit
            .phis
            .iter()
            .find(|ph| {
                ph.reg == status_reg && ph.args[o] == opaque_status && ph.args[a] == accepted_status
            })?
            .value;
        for &b in &ah.body_blocks {
            for i in &self.p.blocks[b].insts {
                if i.write(status_reg).is_some()
                    && (i.name() == "v_writelane_b32"
                        || !self.mask_source(
                            i.read(Reg::S(EXEC))?,
                            save.write(Reg::S(EXEC))?,
                            true,
                            0,
                        ))
                {
                    return None;
                }
            }
        }
        let (latch, leave, sel) = nodes.iter().find_map(|&b| {
            let block = &self.p.blocks[b];
            let Exit::Branch { taken, not_taken } = block.exit else {
                return None;
            };
            if self.p.blocks[taken].exit != Exit::Jump(leaf.0) || self.latch_limit(b).is_none() {
                return None;
            }
            let select = block.insts.iter().rev().nth(6)?;
            (select.name() == "v_cndmask_b32"
                && two_src_operands(select).is_some_and(|(z, st)| {
                    const_operand(z) == Some(0)
                        && operand_val(select, st).and_then(|v| self.value_reg(v))
                            == Some(status_reg)
                }))
            .then_some((b, not_taken, select))
        })?;
        // No redefinition of accepted status before the latch can turn the
        // accepted zero back into a continuing state. The saved latch EXEC
        // covers every lane of the intersection.
        let mut seen = HashSet::new();
        let mut todo = vec![cb];
        while let Some(b) = todo.pop() {
            if !seen.insert(b) {
                continue;
            }
            if !nodes.contains(&b)
                || self.p.blocks[b]
                    .insts
                    .iter()
                    .any(|i| i.write(status_reg).is_some())
            {
                return None;
            }
            if b != latch {
                todo.extend(self.p.blocks[b].exit.successors());
            }
        }
        let (_, state) = two_src_operands(sel)?;
        let mut values = vec![operand_val(sel, state)?];
        let mut checked = HashSet::new();
        while let Some(v) = values.pop() {
            if v == joined || !checked.insert(v) {
                continue;
            }
            let Def::Phi(b, k) = self.defs[v.0 as usize]? else {
                return None;
            };
            if b == cb || !seen.contains(&b) {
                return None;
            }
            let mut from_commit = self.p.blocks[b]
                .preds
                .iter()
                .zip(&self.p.blocks[b].phis[k].args)
                .filter(|(p, _)| seen.contains(p))
                .peekable();
            from_commit.peek()?;
            values.extend(from_commit.map(|(_, v)| *v));
        }
        let lane_exec = image.read(Reg::S(EXEC))?;
        if !self.mask_source(sel.read(Reg::S(EXEC))?, lane_exec, false, 0) {
            return None;
        }
        // Until the inner loop exits, later iterations cannot overwrite
        // a retired lane's status: every such writer is predicated on the
        // current iteration's entry EXEC.
        let status_out = sel.writes.first()?.0;
        let mut seen = HashSet::new();
        let mut todo = vec![leaf.0];
        while let Some(b) = todo.pop() {
            if b == leave || !seen.insert(b) {
                continue;
            }
            if !nodes.contains(&b) {
                return None;
            }
            for i in &self.p.blocks[b].insts {
                if i.write(status_out).is_some()
                    && (i.name() == "v_writelane_b32"
                        || !self.mask_source(i.read(Reg::S(EXEC))?, lane_exec, true, 0))
                {
                    return None;
                }
            }
            todo.extend(self.p.blocks[b].exit.successors());
        }
        self.shared_retirement(nodes, h, primary, leaf, leave, sel)
    }

    fn only_writer(&self, nodes: &HashSet<usize>, r: Reg) -> Option<(usize, &Inst)> {
        let mut writers = nodes
            .iter()
            .flat_map(|&b| self.p.blocks[b].insts.iter().map(move |i| (b, i)))
            .filter(|(_, i)| i.write(r).is_some());
        let first = writers.next()?;
        writers.next().is_none().then_some(first)
    }

    fn shared_retirement(
        &self,
        nodes: &HashSet<usize>,
        h: usize,
        primary: Site,
        leaf: Site,
        leave: usize,
        select: &Inst,
    ) -> Option<()> {
        let (guard, pre, _) = self.shared_entry(h, nodes, primary)?;
        if guard.require_full_work {
            return None;
        }
        let work = self.p.blocks[h]
            .phis
            .iter()
            .find(|p| p.reg == Reg::S(guard.work))?;
        let initial = self.p.blocks[h].preds.iter().position(|&b| b == pre)?;
        let end = &self.p.blocks[leave];
        let status = select.writes.first()?.1;
        let phi = end.phis.iter().find(|p| {
            p.args.contains(&status)
                && p.args.iter().all(|&v| {
                    v == status
                        || self.def_inst(v).is_some_and(|(_, i)| {
                            i.name() == "v_mov_b32"
                                && address::plain(i)
                                && first_src_operand(i).and_then(const_operand) == Some(3)
                        })
                })
        })?;
        let test = end.insts.iter().find(|i| {
            i.name() == "v_cmp_eq_u32"
                && address::plain(i)
                && other_val_from_const_op(i, 0) == Some(phi.value)
        })?;
        let restored = test.read(Reg::S(EXEC))?;
        let inner = &self.p.blocks[leaf.0];
        let current = inner
            .insts
            .iter()
            .find(|i| i.name() == "image_bvh_intersect_ray")?
            .read(Reg::S(EXEC))?;
        if !self.mask_source(current, restored, true, 0) {
            // A loop-carried EXEC can only shrink relative to the external
            // edge. Its exact induction is checked by watchdog_backedge.
            let ph = inner.phis.iter().find(|p| p.value == current)?;
            let ext = inner
                .preds
                .iter()
                .position(|&b| !self.dominates(leaf.0, b))?;
            if !self.mask_source(ph.args[ext], restored, true, 0) {
                return None;
            }
        }
        let wave = self.p.inputs.iter().find(|(r, _)| *r == Reg::S(EXEC))?.1;
        let tested = test.writes.first()?.1;
        let (global, updated) = end.insts.iter().find_map(|op| {
            if op.name() != "s_and_b32" {
                return None;
            }
            let (x, y) = two_src_operands(op)?;
            for (g, n) in [(x, y), (y, x)] {
                let gv = operand_val(op, g)?;
                let Some(gp) = self.p.blocks[h].phis.iter().find(|p| p.value == gv) else {
                    continue;
                };
                let not = self.def_inst(operand_val(op, n)?)?.1;
                if not.name() != "s_not_b32" {
                    continue;
                }
                let diff = single_reg_src(not)?;
                if !self.work_contains_only(diff, tested, 0) {
                    continue;
                }
                let mut todo = vec![diff];
                while let Some(v) = todo.pop() {
                    if v == tested
                        || self.mask_source(v, wave, false, 0)
                        || self.mask_source(v, restored, false, 0)
                    {
                        continue;
                    }
                    let and = self.def_inst(v)?.1;
                    if and.name() != "s_and_b32" {
                        return None;
                    }
                    let (a, b) = two_src_operands(and)?;
                    todo.extend([operand_val(and, a)?, operand_val(and, b)?]);
                }
                if self.only_writer(nodes, gp.reg)?.1.writes.first()?.1 != op.writes.first()?.1 {
                    continue;
                }
                return Some((gp, op.writes.first()?.1));
            }
            None
        })?;
        if !self.work_contains_only(work.args[initial], global.args[initial], 0) {
            return None;
        }
        // Bind the paired node and work on every outer header edge. A normal
        // child mask is a ballot of the four results of this TLAS intersection.
        // A popped mask is explicitly intersected with current global work.
        let (_, image) =
            self.def_inst(image_data_values(&self.p.blocks[primary.0].insts[primary.1])?[0])?;
        let children = image_data_values(image)?;
        let mut todo: Vec<_> = work
            .args
            .iter()
            .enumerate()
            .filter(|(k, _)| *k != initial)
            .map(|(_, v)| *v)
            .collect();
        let mut seen = HashSet::new();
        while let Some(v) = todo.pop() {
            if !seen.insert(v) {
                continue;
            }
            match self.defs[v.0 as usize]? {
                Def::Phi(b, j) if nodes.contains(&b) && b != h => {
                    todo.extend(&self.p.blocks[b].phis[j].args)
                }
                Def::Inst(b, j) if nodes.contains(&b) => {
                    let i = &self.p.blocks[b].insts[j];
                    match i.name() {
                        "s_mov_b32" if first_src_operand(i).and_then(const_operand) == Some(0) => {}
                        "v_readfirstlane_b32" => {
                            let src = single_reg_src(i)?;
                            let (_, and) = self.def_inst(src)?;
                            if and.name() != "v_and_b32"
                                || !address::plain(and)
                                || !self.mask_covers(and, i, &[])
                            {
                                return None;
                            }
                            let (x, y) = two_src_operands(and)?;
                            if ![x, y].iter().any(|op| {
                                operand_val(and, op)
                                    .is_some_and(|v| self.global_value(v, global.value, updated, 0))
                            }) {
                                return None;
                            }
                        }
                        "v_readlane_b32" => self.child_table(i, &children, primary, h)?,
                        _ => return None,
                    }
                }
                _ => return None,
            }
        }
        Some(())
    }

    fn global_value(&self, v: Value, prior: Value, updated: Value, d: u8) -> bool {
        if v == prior || v == updated {
            return true;
        }
        d < 20
            && matches!(self.defs[v.0 as usize],Some(Def::Phi(b,k)) if self.p.blocks[b].phis[k].args.iter().all(|&v|self.global_value(v,prior,updated,d+1)))
    }

    fn child_table(&self, read: &Inst, children: &[Value], primary: Site, h: usize) -> Option<()> {
        let (table, idx) = two_src_operands(read)?;
        let i = self.def_inst(operand_val(read, idx)?)?.1;
        if !matches!(i.name(), "s_add_i32" | "s_add_u32") {
            return None;
        }
        let idx = other_val_from_const_op(i, 1)?;
        let mask = self.def_inst(idx)?.1;
        if mask.name() != "s_and_b32" {
            return None;
        }
        let score = other_val_from_const_op(mask, 7)?;
        if !self.even_score(score, mask, 0) {
            return None;
        }
        let mut slots = 0xaau32;
        let mut v = operand_val(read, table)?;
        let reg = self.value_reg(v)?;
        for _ in 0..64 {
            let (_, write) = self.def_inst(v)?;
            if write.name() != "v_writelane_b32" {
                return None;
            }
            let (bits, slot) = two_src_operands(write)?;
            let slot = const_operand(slot)?;
            if slot >= 32 {
                return None;
            }
            if slots & (1 << slot) != 0 {
                if !self.child_bits(operand_val(write, bits)?, write, children, primary, h, 0) {
                    return None;
                }
                slots &= !(1 << slot);
                if slots == 0 {
                    return Some(());
                }
            }
            v = write.read(reg)?;
        }
        None
    }

    fn even_score(&self, v: Value, reader: &Inst, d: u8) -> bool {
        if d > 15 {
            return false;
        }
        let Some((_, i)) = self.def_inst(v) else {
            return false;
        };
        if !address::plain(i)
            || (matches!(i.fields, Fields::Valu(_))
                && i.name() != "v_readfirstlane_b32"
                && !self.mask_covers(i, reader, &[]))
        {
            return false;
        }
        let val = |op: &Operand| {
            const_operand(op).is_some_and(|x| x & 1 == 0)
                || operand_val(i, op).is_some_and(|x| self.even_score(x, i, d + 1))
        };
        match (&i.fields, i.name()) {
            (_, "v_readfirstlane_b32" | "v_mov_b32" | "s_mov_b32") => {
                first_src_operand(i).is_some_and(val)
            }
            (Fields::Valu(f), "v_max_u32" | "v_max3_u32") => f.src.iter().all(val),
            (Fields::Valu(f), "v_lshl_or_b32") => {
                matches!(f.src.as_slice(),[_,shift,tag] if const_operand(shift)==Some(3) && const_operand(tag).is_some_and(|k|k&1==0))
            }
            (_, "s_lshl_b32") => {
                two_src_operands(i).is_some_and(|(_, k)| const_operand(k) == Some(3))
            }
            _ => false,
        }
    }

    fn child_bits(
        &self,
        v: Value,
        reader: &Inst,
        children: &[Value],
        primary: Site,
        h: usize,
        d: u8,
    ) -> bool {
        if d > 15 {
            return false;
        }
        let Some((_, i)) = self.def_inst(v) else {
            return false;
        };
        if !address::plain(i) || i.writes.first().is_none_or(|w| w.1 != v) {
            return false;
        }
        let src = |op: &Operand| {
            operand_val(i, op).is_some_and(|x| self.child_bits(x, i, children, primary, h, d + 1))
        };
        match (&i.fields, i.name()) {
            (_, "v_readfirstlane_b32" | "s_mov_b32") => first_src_operand(i).is_some_and(src),
            (_, "v_mov_b32") => {
                self.mask_covers(i, reader, &[]) && first_src_operand(i).is_some_and(src)
            }
            (Fields::Valu(f), "v_or3_b32" | "v_or_b32") => {
                self.mask_covers(i, reader, &[]) && f.src.iter().all(src)
            }
            (Fields::Sop { src: ops, .. }, "s_or_b32") => ops.iter().all(src),
            (Fields::Sop { src: ops, .. }, "s_and_b32") => ops.iter().any(src),
            (_, "v_cmp_eq_u32") => {
                let Some(initial) = self.p.blocks[h]
                    .insts
                    .iter()
                    .rev()
                    .nth(1)
                    .and_then(|save| save.read(Reg::S(EXEC)))
                else {
                    return false;
                };
                if i.read(Reg::S(EXEC))
                    .is_none_or(|m| !self.mask_source(m, initial, true, 0))
                {
                    return false;
                }
                let Some((a, b)) = two_src_operands(i) else {
                    return false;
                };
                let nonnegative = |op| {
                    operand_val(i, op)
                        .and_then(|v| self.def_inst(v))
                        .is_some_and(|(_, m)| m.name() == "s_max_i32" && has_const_operand(m, 0))
                };
                let child = |op| {
                    let v = operand_val(i, op)?;
                    let Def::Phi(b, j) = self.defs[v.0 as usize]? else {
                        return None;
                    };
                    let ph = &self.p.blocks[b].phis[j];
                    let fill = |v, depth| self.minus_one_in_header(v, h, depth);
                    (self.p.blocks[b].preds.len() == 2
                        && self.p.blocks[b].preds.iter().zip(&ph.args).all(|(&p, &x)| {
                            p == h && fill(x, 0) || p == primary.0 && children.contains(&x)
                        }))
                    .then_some(())
                };
                nonnegative(a) && child(b).is_some() || nonnegative(b) && child(a).is_some()
            }
            _ => false,
        }
    }
    fn minus_one_in_header(&self, v: Value, h: usize, d: u8) -> bool {
        if d > 6 {
            return false;
        }
        let Some(((b, _), i)) = self.def_inst(v) else {
            return false;
        };
        b == h
            && i.name() == "v_mov_b32"
            && address::plain(i)
            && first_src_operand(i).is_some_and(|op| {
                const_operand(op) == Some(u32::MAX)
                    || operand_val(i, op).is_some_and(|x| self.minus_one_in_header(x, h, d + 1))
            })
    }
}
