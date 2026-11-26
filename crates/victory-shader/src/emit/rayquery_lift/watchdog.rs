//! Only hide a completed traversal's software step count if every observable
//! use is the very unsigned timeout test that terminates the guest traversal.
use super::*;

impl Analyzer<'_> {
    pub(super) fn completed_watchdog(
        &self,
        root: Value,
        nodes: &HashSet<usize>,
        removed: &HashSet<usize>,
        header: usize,
    ) -> Option<Vec<Site>> {
        let Some(reg @ Reg::V(_)) = self.value_reg(root) else {
            return None;
        };
        let mut todo = vec![root];
        let mut family = HashSet::new();
        let (mut init, mut increment) = (false, false);
        while let Some(v) = todo.pop() {
            if !family.insert(v) {
                continue;
            }
            let def = self.defs[v.0 as usize]?;
            match def {
                Def::Phi(b, k) if nodes.contains(&b) => {
                    let ph = &self.p.blocks[b].phis[k];
                    if ph.reg != reg {
                        return None;
                    }
                    for (&pred, &arg) in self.p.blocks[b].preds.iter().zip(&ph.args) {
                        if b == header && !nodes.contains(&pred) {
                            if self.eval_const(arg, 0) != Some(0) {
                                return None;
                            }
                            init = true;
                        } else {
                            todo.push(arg)
                        }
                    }
                }
                Def::Inst(b, j) if nodes.contains(&b) => {
                    let i = &self.p.blocks[b].insts[j];
                    if !address::plain(i)
                        || i.write(reg) != Some(v)
                        || i.writes.first().map(|w| w.1) != Some(v)
                    {
                        return None;
                    }
                    let src = match i.name() {
                        "v_add_nc_u32" | "v_add_co_u32" => {
                            increment = true;
                            other_val_from_const_op(i, 1)
                        }
                        "v_mov_b32" => single_reg_src(i),
                        _ => return None,
                    };
                    let src = src?;
                    todo.push(src);
                    if self.st.guards[b][j].is_some() {
                        let old = i.read(reg)?;
                        todo.push(old)
                    }
                }
                _ if self.eval_const(v, 0) == Some(0) => {}
                _ => return None,
            }
        }
        if !init || !increment {
            return None;
        }
        let mut limits = HashSet::new();
        for &b in nodes {
            let bb = &self.p.blocks[b];
            if bb.insts.last().is_none_or(|i| i.name() != "s_cbranch_scc1") {
                continue;
            }
            // The watchdog mask joins the empty-node mask. The loop then
            // removes these lanes from its work mask and branches on SCC.
            // This sequence must terminate the block: SCC selects its real backedge.
            {
                let tail = &bb.insts[bb.insts.len().saturating_sub(7)..];
                let [cmp, empty, flag, join, save, left, branch] = tail else {
                    continue;
                };
                if ![cmp, empty, flag].into_iter().all(address::plain)
                    || empty.name() != "v_cmp_gt_i32"
                    || !has_const_operand(empty, 1)
                    || flag.name() != "v_cndmask_b32"
                    || join.name() != "s_or_b32"
                    || save.name() != "s_and_saveexec_b32"
                    || left.name() != "s_andn2_b32"
                    || branch.name() != "s_cbranch_scc1"
                {
                    continue;
                }
                let Some(&(_, mask)) = cmp.writes.first() else {
                    continue;
                };
                let Some(&(_, empty_mask)) = empty.writes.first() else {
                    continue;
                };
                let Some(&(_, union)) = join.writes.first() else {
                    continue;
                };
                let Fields::Valu(f) = &flag.fields else {
                    continue;
                };
                if !matches!(f.src.as_slice(),[z,ones,c] if const_operand(z)==Some(0) && const_operand(ones)==Some(u32::MAX) && operand_val(flag,c)==Some(mask))
                {
                    continue;
                }
                if !join.reads.iter().any(|&(_, v)| v == mask)
                    || !join.reads.iter().any(|&(_, v)| v == empty_mask)
                    || !first_src_operand(save).is_some_and(|a| operand_val(save, a) == Some(union))
                    || save.write(Reg::S(EXEC)) != left.read(Reg::S(EXEC))
                    || left.write(Reg::Scc) != branch.read(Reg::Scc)
                    || !self.watchdog_backedge(b, empty, save, left)
                {
                    continue;
                }
                if let Some((count, limit, strict)) = self.watchdog_compare(cmp)
                    && family.contains(&count)
                {
                    limits.insert((limit, strict));
                }
            }
        }
        if limits.is_empty() {
            return None;
        }

        let mut todo = vec![root];
        let mut seen = HashSet::new();
        let mut compared = false;
        let mut dynamic = Vec::new();
        while let Some(v) = todo.pop() {
            if !seen.insert(v) {
                continue;
            }
            if seen.len() > 4096 {
                return None;
            }
            for &(b, j) in &self.uses[v.0 as usize] {
                if removed.contains(&b) {
                    continue;
                }
                let Some(j) = j else {
                    todo.extend(
                        self.p.blocks[b]
                            .phis
                            .iter()
                            .filter(|p| p.args.contains(&v))
                            .map(|p| p.value),
                    );
                    continue;
                };
                let i = &self.p.blocks[b].insts[j];
                let reads = operand_reads(self.p, i);
                if !i.reads.iter().any(|&(r, x)| v == x && reads.contains(&r)) {
                    if self.st.guards[b][j].is_some() {
                        todo.extend(
                            i.reads
                                .iter()
                                .filter(|x| x.1 == v)
                                .filter_map(|&(r, _)| i.write(r)),
                        );
                    }
                    continue;
                }
                if i.name() == "v_mov_b32" && address::plain(i) && single_reg_src(i) == Some(v) {
                    todo.push(i.writes[0].1);
                    continue;
                }
                let (count, limit, strict) = self.watchdog_compare(i)?;
                // The represented completed count is zero. For <= this is
                // false only if the limit is known nonzero.
                if count != v || !limits.contains(&(limit, strict)) {
                    return None;
                }
                if !strict && !matches!(limit,Word::Imm(k) if k!=0) {
                    // A zero limit makes limit <= 0 true. Only suppress that
                    // comparator on the lanes of this single, non-reentrant
                    // query and prove its operand is still this count.
                    self.post_path.as_ref()?;
                    if !self.native_count(v, &family, nodes, 0) {
                        return None;
                    }
                    dynamic.push((b, j));
                }
                compared = true;
            }
        }
        compared.then_some(dynamic)
    }

    fn watchdog_backedge(&self, latch: usize, empty: &Inst, save: &Inst, left: &Inst) -> bool {
        if self.p.wave64 {
            return false;
        }
        let Exit::Branch {
            taken: resume,
            not_taken: leave,
        } = self.p.blocks[latch].exit
        else {
            return false;
        };
        let resume_block = &self.p.blocks[resume];
        let Exit::Jump(h) = resume_block.exit else {
            return false;
        };
        let header = &self.p.blocks[h];
        let [enable] = resume_block.insts.as_slice() else {
            return false;
        };
        if resume_block.preds != [latch]
            || !self.dominates(h, latch)
            || header.preds.len() != 2
            || enable.name() != "s_and_b32"
        {
            return false;
        }
        let Some(back) = header.preds.iter().position(|&p| p == resume) else {
            return false;
        };
        if self.dominates(h, header.preds[1 - back]) {
            return false;
        }
        // remaining = previous_work & !finished_exec. The next iteration
        // must read that remaining work, including as the next loop PHI.
        let Some((before, finished)) = two_src_operands(left) else {
            return false;
        };
        let (Some(previous), Some(finished)) =
            (operand_val(left, before), operand_val(left, finished))
        else {
            return false;
        };
        let Some(&(reg, remaining)) = left.writes.first() else {
            return false;
        };
        if reg == Reg::S(EXEC)
            || self.value_reg(previous) != Some(reg)
            || Some(finished) != save.write(Reg::S(EXEC))
            || !header
                .phis
                .iter()
                .any(|p| p.value == previous && p.args[back] == remaining)
        {
            return false;
        }
        let Some(&(saved_reg, saved)) = save.writes.first() else {
            return false;
        };
        if saved_reg == Reg::S(EXEC)
            || !enable.reads.iter().any(|&(_, v)| v == saved)
            || !enable.reads.iter().any(|&(_, v)| v == remaining)
        {
            return false;
        }
        let Some(next_exec) = enable.write(Reg::S(EXEC)) else {
            return false;
        };
        let Some(image) = header
            .insts
            .iter()
            .find(|i| i.name() == "image_bvh_intersect_ray")
        else {
            return false;
        };
        let Some(node) = image_addr_values(image).and_then(|v| v.first().copied()) else {
            return false;
        };
        let Some((one, next_node)) = two_src_operands(empty) else {
            return false;
        };
        let Some(first_work) = header
            .phis
            .iter()
            .find(|p| p.value == previous)
            .map(|p| p.args[1 - back])
        else {
            return false;
        };
        let Some(exec_phi) = header
            .phis
            .iter()
            .find(|p| Some(p.value) == image.read(Reg::S(EXEC)))
        else {
            return false;
        };
        let Some(latch_exec) = save.read(Reg::S(EXEC)) else {
            return false;
        };
        if const_operand(one) != Some(1)
            || !header
                .phis
                .iter()
                .any(|p| p.value == node && Some(p.args[back]) == operand_val(empty, next_node))
            || exec_phi.args[back] != next_exec
            || !self.same_work_mask(first_work, exec_phi.args[1 - back])
            || !self.same_work_mask(latch_exec, exec_phi.value)
        {
            return false;
        }
        // The fallthrough must leave the natural loop. If it can get back
        // to the latch without re-entering the header, it is another iteration.
        let mut seen = HashSet::from([h]);
        let mut work = vec![resume];
        while let Some(b) = work.pop() {
            if b == leave {
                return false;
            }
            if seen.insert(b) {
                work.extend(&self.p.blocks[b].preds)
            }
        }
        leave != h
            && seen.iter().all(|&b| {
                b == latch
                    || self.p.blocks[b]
                        .exit
                        .successors()
                        .iter()
                        .all(|s| seen.contains(s))
            })
    }

    // Saving EXEC copies its old value into the explicit destination.
    // The header may have distinct PHIs for work and EXEC; compare their
    // *preheader* inputs, not the values carried on the backedge.
    fn same_work_mask(&self, a: Value, b: Value) -> bool {
        let source = |mut v: Value| {
            for _ in 0..32 {
                match self.copy_word(Word::Ssa(v)) {
                    Word::Imm(k) => return Word::Imm(k),
                    Word::Ssa(x) => v = x,
                }
                let Some((_, i)) = self.def_inst(v) else {
                    break;
                };
                if !i.name().ends_with("saveexec_b32")
                    || !matches!(i.writes.first(),Some((Reg::S(s),w)) if *s!=EXEC && *w==v)
                {
                    break;
                }
                let Some(old) = i.read(Reg::S(EXEC)) else {
                    break;
                };
                v = old;
            }
            Word::Ssa(v)
        };
        source(a) == source(b)
            || matches!((self.st.mask_bits(a,None),self.st.mask_bits(b,None)),(Some(a),Some(b)) if a==b)
    }

    fn native_count(
        &self,
        v: Value,
        family: &HashSet<Value>,
        nodes: &HashSet<usize>,
        depth: usize,
    ) -> bool {
        // Only values produced inside the native traversal carry its
        // provenance. A preheader zero can also reach this PHI on a guest path.
        if family.contains(&v) && self.def_block(v).is_some_and(|b| nodes.contains(&b)) {
            return true;
        }
        if depth >= 32 {
            return false;
        }
        match self.defs[v.0 as usize] {
            Some(Def::Phi(b, k)) => {
                let mut args = self.p.blocks[b]
                    .preds
                    .iter()
                    .zip(&self.p.blocks[b].phis[k].args)
                    .filter(|(p, _)| self.native_edge(b, **p))
                    .peekable();
                args.peek().is_some()
                    && args.all(|(_, a)| self.native_count(*a, family, nodes, depth + 1))
            }
            Some(Def::Inst(b, k)) => {
                let i = &self.p.blocks[b].insts[k];
                i.name() == "v_mov_b32"
                    && address::plain(i)
                    && self.st.guards[b][k].is_none()
                    && single_reg_src(i)
                        .is_some_and(|a| self.native_count(a, family, nodes, depth + 1))
            }
            _ => false,
        }
    }

    /// The peeled entry has two copies of the same dispatch: one outside
    /// the hardware loop and one on its backedge. Verify that the scalar
    /// step test both sets this flag and clears the dispatch state before
    /// the next intersection. Arbitrary flags derived from a counter do
    /// not have this property.
    pub(super) fn completed_flag(&self, v: Value, nodes: &HashSet<usize>, header: usize) -> bool {
        let h = &self.p.blocks[header];
        if self.p.wave64
            || !h
                .insts
                .iter()
                .any(|i| i.name() == "image_bvh64_intersect_ray")
        {
            return false;
        }
        let mut incoming = h.preds.iter().filter(|p| !nodes.contains(p));
        let Some(&pre) = incoming.next() else {
            return false;
        };
        if incoming.next().is_some() {
            return false;
        }
        let Some(&d) = h
            .preds
            .iter()
            .find(|p| nodes.contains(p) && self.p.blocks[**p].offset == self.p.blocks[pre].offset)
        else {
            return false;
        };
        let Some(reg @ Reg::V(_)) = self.value_reg(v) else {
            return false;
        };
        let bb = &self.p.blocks[d];
        let external = &self.p.blocks[pre];
        if external.insts.len() != bb.insts.len()
            || external
                .insts
                .iter()
                .zip(&bb.insts)
                .any(|(a, b)| a.machine.words != b.machine.words)
            || !matches!(external.exit,Exit::Branch{not_taken,..} if not_taken==header)
        {
            return false;
        }
        let Some((j, flag)) = bb.insts.iter().enumerate().find(|(_, i)| {
            i.write(reg).is_some()
                && select_operands(i).is_some_and(|(a, b, _)| {
                    const_operand(a) == Some(0) && const_operand(b) == Some(u32::MAX)
                })
        }) else {
            return false;
        };
        let mut seen = HashSet::new();
        let mut todo = vec![v];
        let mut flag_seen = false;
        while let Some(x) = todo.pop() {
            if !seen.insert(x) {
                continue;
            }
            if seen.len() > 512 {
                return false;
            }
            match self.defs[x.0 as usize] {
                Some(Def::Inst(b, i))
                    if (b == d || b == pre)
                        && self.p.blocks[b].insts[i].machine.offset == flag.machine.offset
                        && self.p.blocks[b].insts[i].write(reg) == Some(x) =>
                {
                    flag_seen = true;
                }
                Some(Def::Phi(b, i)) if nodes.contains(&b) => {
                    todo.extend(&self.p.blocks[b].phis[i].args)
                }
                Some(Def::Inst(b, i))
                    if nodes.contains(&b)
                        && self.p.blocks[b].insts[i].name() == "v_mov_b32"
                        && self.st.guards[b][i].is_none()
                        && address::plain(&self.p.blocks[b].insts[i]) =>
                {
                    let src = single_reg_src(&self.p.blocks[b].insts[i]);
                    if let Some(src) = src {
                        todo.push(src)
                    } else if self.eval_const(x, 0) != Some(0) {
                        return false;
                    }
                }
                _ if self.eval_const(x, 0) == Some(0) => {}
                _ => return false,
            }
        }
        if !flag_seen {
            return false;
        }
        if !matches!(bb.exit,Exit::Branch{not_taken,..} if not_taken==header)
            || bb
                .insts
                .last()
                .is_none_or(|i| i.name() != "s_cbranch_execz")
        {
            return false;
        }
        let Some(gate) = bb.insts.iter().rev().nth(1) else {
            return false;
        };
        let Some((target, state)) = two_src_operands(gate) else {
            return false;
        };
        let Some(state) = operand_val(gate, state) else {
            return false;
        };
        if gate.name() != "v_cmpx_eq_u32"
            || const_operand(target) != Some(3)
            || !address::plain(gate)
        {
            return false;
        }
        let Some(exec) = gate.read(Reg::S(EXEC)) else {
            return false;
        };
        let Some(((sb, _), normalize)) = self.def_inst(state) else {
            return false;
        };
        let Some((status, three, node_mask)) = select_operands(normalize) else {
            return false;
        };
        let (Some(status), Some(node_mask)) = (
            operand_val(normalize, status),
            operand_val(normalize, node_mask),
        ) else {
            return false;
        };
        let Some(((cb, _), node_cmp)) = self.def_inst(node_mask) else {
            return false;
        };
        if sb != d
            || cb != d
            || const_operand(three) != Some(3)
            || node_cmp.name() != "v_cmp_eq_u32"
            || other_val_from_const_op(node_cmp, 4) != Some(status)
        {
            return false;
        }
        let Some(((b, _), clear)) = self.def_inst(status) else {
            return false;
        };
        let Some((_, zero, mask)) = select_operands(clear) else {
            return false;
        };
        if b != d || const_operand(zero) != Some(0) {
            return false;
        }
        let Some(((b, _), disabled)) = operand_val(clear, mask).and_then(|v| self.def_inst(v))
        else {
            return false;
        };
        let Some((on, off)) = two_src_operands(disabled) else {
            return false;
        };
        if b != d || disabled.name() != "s_cselect_b32" {
            return false;
        }
        let expires_when =
            if operand_val(disabled, on) == Some(exec) && const_operand(off) == Some(0) {
                true
            } else if operand_val(disabled, off) == Some(exec) && const_operand(on) == Some(0) {
                false
            } else {
                return false;
            };
        let Some(scc) = disabled.read(Reg::Scc) else {
            return false;
        };
        let Some(((b, _), cmp)) = self.def_inst(scc) else {
            return false;
        };
        let Some((count, limit_ok)) = (match (&cmp.fields, cmp.name(), expires_when) {
            (
                Fields::Sop {
                    src, imm: Some(k), ..
                },
                "s_cmpk_gt_u32",
                true,
            )
            | (
                Fields::Sop {
                    src, imm: Some(k), ..
                },
                "s_cmpk_lt_u32",
                false,
            ) => Some((src.first(), *k >= 65534)),
            (Fields::Sop { src, .. }, "s_cmp_le_u32", true) => Some((
                src.get(1),
                src.first().is_some_and(|x| {
                    Word::operand(cmp, x).is_some_and(|w| match w {
                        Word::Imm(_) => true,
                        Word::Ssa(v) => self.def_block(v).is_some_and(|b| !nodes.contains(&b)),
                    })
                }),
            )),
            (Fields::Sop { src, .. }, "s_cmp_eq_u32", true)
                if src.get(1).and_then(const_operand) == Some(u32::MAX) =>
            {
                Some((src.first(), true))
            }
            _ => None,
        }) else {
            return false;
        };
        let Some(((ib, _), inc)) = count
            .and_then(|x| operand_val(cmp, x))
            .and_then(|v| self.def_inst(v))
        else {
            return false;
        };
        if b != d
            || ib != d
            || !limit_ok
            || !matches!(inc.name(), "s_add_i32" | "s_add_u32")
            || !other_val_from_const_op(inc, 1)
                .is_some_and(|v| self.value_reg(v) == inc.writes.first().map(|w| w.0))
        {
            return false;
        }
        let Some(&(step_reg, _)) = inc.writes.first() else {
            return false;
        };
        if nodes.iter().copied().any(|b| {
            self.p.blocks[b]
                .insts
                .iter()
                .any(|i| i.machine.offset != inc.machine.offset && i.write(step_reg).is_some())
        }) {
            return false;
        }
        let Some((z, ones, cond)) = select_operands(flag) else {
            return false;
        };
        if const_operand(z) != Some(0) || const_operand(ones) != Some(u32::MAX) {
            return false;
        }
        let Some(((b, _), eq)) = operand_val(flag, cond).and_then(|v| self.def_inst(v)) else {
            return false;
        };
        if b != d || eq.name() != "v_cmp_eq_u32" {
            return false;
        }
        let Some(((b, _), and)) = other_val_from_const_op(eq, 1).and_then(|v| self.def_inst(v))
        else {
            return false;
        };
        if b != d || and.name() != "v_and_b32" {
            return false;
        }
        let Some(((b, _), bit)) = other_val_from_const_op(and, 1).and_then(|v| self.def_inst(v))
        else {
            return false;
        };
        let Some((a, c)) = two_src_operands(bit) else {
            return false;
        };
        b == d
            && bit.name() == "s_cselect_b32"
            && bit.read(Reg::Scc) == Some(scc)
            && (const_operand(a), const_operand(c))
                == if expires_when {
                    (Some(1), Some(0))
                } else {
                    (Some(0), Some(1))
                }
            && [gate, normalize, node_cmp, clear, flag, eq, and]
                .iter()
                .all(|i| i.read(Reg::S(EXEC)) == Some(exec) && address::plain(i))
            && self.st.guards[d][j].is_none()
    }

    fn watchdog_compare(&self, i: &Inst) -> Option<(Value, Word, bool)> {
        if !address::plain(i) {
            return None;
        }
        let (limit, count) = two_src_operands(i)?;
        let strict = match i.name() {
            "v_cmp_lt_u32" => true,
            "v_cmp_le_u32" => false,
            _ => return None,
        };
        Some((
            operand_val(i, count)?,
            self.copy_word(Word::operand(i, limit)?),
            strict,
        ))
    }
}

fn select_operands(i: &Inst) -> Option<(&Operand, &Operand, &Operand)> {
    let Fields::Valu(f) = &i.fields else {
        return None;
    };
    let [a, b, c] = f.src.as_slice() else {
        return None;
    };
    (i.name() == "v_cndmask_b32" && address::plain(i)).then_some((a, b, c))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_a_watchdog_that_removes_work_from_the_backedge_can_hide_a_count() {
        let sop2 = |op: u32, d: u32, a: u32, b: u32| 0x8000_0000 | op << 23 | d << 16 | b << 8 | a;
        let vop2 = |op: u32, d: u32, a: u32, b: u32| op << 25 | d << 17 | b << 9 | a;
        for (initial, left, expected) in [(126u32, 8, true), (126, 126, false), (128, 8, false)] {
            let mut words = vec![
                0x7e08_0280,
                0xbe88_0300 | initial,
                0x7e04_0206,
                0xbf82_0001,           // count=0, work=EXEC, node=s6
                sop2(0x0e, 126, 9, 8), // resume: EXEC=saved & remaining
            ];
            let image = words.len();
            words.extend([
                0xf198_9f07,
                0x0003_0802,
                0x0504_0302,
                0x0908_0706,
                0x000c_0b0a,
            ]);
            words.extend([
                0x7e04_0308,                                 // next node = first intersection result
                vop2(0x25, 4, 129, 4),                       // ++count
                0x7c00_0000 | (0xc1 << 17) | (4 << 9) | 129, // timeout = 1 < count
                0xd484_000b,
                129 | (258 << 9), // s11 = 1 > next_node
                0xd501_0003,
                128 | (193 << 9) | (106 << 18), // flag = timeout ? -1 : 0
                sop2(0x10, 10, 106, 11),
                0xbe89_3c0a,              // union, save EXEC
                sop2(0x14, 8, left, 126), // work & !finished, or the incorrect EXEC & !EXEC
            ]);
            let delta = 4i32 - words.len() as i32 - 1;
            words.push(0xbf85_0000 | u32::from(delta as i16 as u16));
            words.extend([0x7c00_0000 | (0xc1 << 17) | (4 << 9) | 129, 0xbf81_0000]);
            let bytes = words
                .into_iter()
                .flat_map(u32::to_le_bytes)
                .collect::<Vec<_>>();
            let p = crate::ir::build(&bytes, false, 64).unwrap();
            let st = crate::structure::structure(&p).unwrap();
            let res = Resources::default();
            let scalar = HashMap::new();
            let a = Analyzer::new(&p, &st, &res, &scalar);
            let h = p.blocks.iter().position(|b| b.offset == image * 4).unwrap();
            assert_eq!(p.blocks[h].insts[0].name(), "image_bvh_intersect_ray");
            let count = p.blocks[h]
                .insts
                .iter()
                .find(|i| i.name() == "v_add_nc_u32")
                .unwrap()
                .write(Reg::V(4))
                .unwrap();
            let nodes = HashSet::from([1, h]);
            assert_eq!(
                a.completed_watchdog(count, &nodes, &nodes, h).is_some(),
                expected
            );
        }
    }
}
