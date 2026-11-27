//! The dynamic XDXR shadow template packs the ray and instance flags together.
//! Recover the original DXR flags only when both the opacity and face tests use them.
use super::*;

impl Analyzer<'_> {
    pub(super) fn alu_args<const N: usize>(
        &self,
        v: Value,
        name: &str,
        reader: &Inst,
    ) -> Option<[Word; N]> {
        let ((b, j), i) = self.def_inst(v)?;
        if i.name() != name
            || i.writes.first()?.1 != v
            || !address::plain(i)
            || (self.st.guards[b][j].is_some() && !self.mask_covers(i, reader, &[]))
        {
            return None;
        }
        let src = match &i.fields {
            Fields::Valu(f) => f.src.as_slice(),
            Fields::Sop { src, .. } => src.as_slice(),
            _ => return None,
        };
        if src.len() != N {
            return None;
        }
        let mut words = [Word::Imm(0); N];
        for (word, arg) in words.iter_mut().zip(src) {
            *word = Word::operand(i, arg)?;
        }
        Some(words)
    }

    fn shadow_flag_values(&self, v: Value, lanes: Bool, depth: usize) -> Option<Vec<u32>> {
        if depth > 12 {
            return None;
        }
        let mut values = Vec::new();
        if let Some(Def::Phi(b, k)) = self.defs[v.0 as usize] {
            for &arg in &self.p.blocks[b].phis[k].args {
                values.extend(self.shadow_flag_values(arg, lanes, depth + 1)?);
            }
        } else {
            let ((b, j), i) = self.def_inst(v)?;
            if !address::plain(i) || i.writes.first()?.1 != v {
                return None;
            }
            if let Some(g) = self.st.guards[b][j]
                && self.bools_meet(&[(g, false), (lanes, true)])
            {
                values.extend(self.shadow_flag_values(i.read(i.writes[0].0)?, lanes, depth + 1)?);
            }
            let alternatives = |arg: &Operand| -> Option<Vec<u32>> {
                match Word::operand(i, arg)? {
                    Word::Imm(k) => Some(vec![k]),
                    Word::Ssa(a) => self.shadow_flag_values(a, lanes, depth + 1),
                }
            };
            let (a, b) = two_src_operands(i)?;
            match i.name() {
                "v_cndmask_b32" => {
                    values.extend(alternatives(a)?);
                    values.extend(alternatives(b)?);
                }
                "v_or_b32" => {
                    let b = alternatives(b)?;
                    for x in alternatives(a)? {
                        for y in &b {
                            values.push(x | y);
                        }
                    }
                }
                _ => return None,
            }
        }
        values.sort_unstable();
        values.dedup();
        (values.len() <= 8 && !values.is_empty()).then_some(values)
    }

    // XDXR may reload an immutable instance header for the triangle load,
    // flags and hardware descriptor. Both address dwords and the component
    // must agree; neither a later SGPR reuse nor an adjacent field suffices.
    pub(super) fn same_header_word(&self, a: Value, b: Value) -> bool {
        let (a, b) = (self.copy_word(Word::Ssa(a)), self.copy_word(Word::Ssa(b)));
        if a == b {
            return true;
        }
        let key = |w: Word| {
            let Word::Ssa(v) = w else {
                return None;
            };
            let (_, i) = self.def_inst(v)?;
            if i.name() != "s_load_dwordx4" {
                return None;
            }
            let Fields::Smem { base, .. } = &i.fields else {
                return None;
            };
            let lo = operand_val(i, base)?;
            (flat_or_smem_load_info(i, lo) == Some((4, 0))).then_some((
                lo,
                entry::upper_operand(i, base)?,
                i.writes.iter().position(|&(_, w)| w == v)?,
            ))
        };
        key(a).is_some_and(|a| key(b) == Some(a))
    }

    // Each node address is the *same* 64-bit sum used by its intersection:
    // (descriptor_word0 << 8) + ((node >> 3) << 6). Both shifts zero-extend
    // their inputs, so bitwise OR with 48 selects the metadata dword.
    fn shadow_node_sum(
        &self,
        low: Value,
        high: Value,
        base: Value,
        node: Value,
        exec: Option<Value>,
    ) -> Option<()> {
        let scalar = exec.is_none();
        let (_, add) = self.def_inst(low)?;
        let (_, carry) = self.def_inst(high)?;
        if !self.is_wide_sum(add, carry, low, high)
            || exec.is_some_and(|x| {
                add.read(Reg::S(EXEC)) != Some(x) || carry.read(Reg::S(EXEC)) != Some(x)
            })
        {
            return None;
        }
        let (a, b) = two_src_operands(add)?;
        let (ah, bh) = two_src_operands(carry)?;
        let los = [operand_val(add, a)?, operand_val(add, b)?];
        let his = [operand_val(carry, ah)?, operand_val(carry, bh)?];
        let mut base_ok = false;
        let mut node_ok = false;
        for lo in los {
            let (_, shift) = self.def_inst(lo)?;
            if !address::plain(shift)
                || shift.writes.first()?.1 != lo
                || !his.contains(&shift.writes.get(1)?.1)
            {
                return None;
            }
            let (a, b) = two_src_operands(shift)?;
            let (pair, k) = match shift.name() {
                "s_lshl_b64" => (a, const_operand(b)?),
                "v_lshlrev_b64" if !scalar && shift.read(Reg::S(EXEC)) == exec => {
                    (b, const_operand(a)?)
                }
                _ => return None,
            };
            let x = operand_val(shift, pair)?;
            let hi = entry::upper_operand(shift, pair)?;
            if self.copy_word(Word::Ssa(hi)) != Word::Imm(0) {
                // A local guarded clear also supplies zero to this vector shift.
                if self.alu_args::<1>(hi, "v_mov_b32", shift)? != [Word::Imm(0)] {
                    return None;
                }
            }
            match k {
                8 if shift.name() == "s_lshl_b64" && self.same_header_word(x, base) => {
                    base_ok = true
                }
                6 => {
                    let args = if scalar {
                        self.alu_args(x, "s_lshr_b32", shift)?
                    } else {
                        self.alu_args(x, "v_lshrrev_b32", shift)?
                    };
                    let expected = if scalar {
                        [Word::Ssa(node), Word::Imm(3)]
                    } else {
                        [Word::Imm(3), Word::Ssa(node)]
                    };
                    if args != expected {
                        return None;
                    }
                    node_ok = true;
                }
                _ => return None,
            }
        }
        (base_ok && node_ok).then_some(())
    }

    // The box arm can overwrite the intersection result VGPRs. At a triangle
    // consumer those lanes took the complementary arm on this very iteration.
    pub(super) fn mask_source(&self, v: Value, cap: Value, subset: bool, _depth: u8) -> bool {
        let mut cap = cap;
        for _ in 0..32 {
            if v == cap {
                return true;
            }
            let Some((_, i)) = self.def_inst(cap) else {
                break;
            };
            let old = match i.name() {
                "s_mov_b32" => single_reg_src(i),
                "s_and_saveexec_b32"
                    if i.writes
                        .first()
                        .is_some_and(|&(r, w)| r != Reg::S(EXEC) && w == cap) =>
                {
                    i.read(Reg::S(EXEC))
                }
                _ => None,
            };
            let Some(x) = old else {
                break;
            };
            cap = x;
        }
        self.mask_implies(v, cap, subset, &mut HashSet::new(), &mut 0)
    }

    fn mask_implies(
        &self,
        v: Value,
        cap: Value,
        subset: bool,
        path: &mut HashSet<Value>,
        n: &mut usize,
    ) -> bool {
        *n += 1;
        if *n > 8192 || path.len() > 96 {
            return false;
        }
        if v == cap || !path.insert(v) {
            return true;
        }
        let mut test = |v| self.mask_implies(v, cap, subset, path, n);
        let result = match self.defs[v.0 as usize] {
            Some(Def::Phi(b, k)) => self.p.blocks[b]
                .preds
                .iter()
                .zip(&self.p.blocks[b].phis[k].args)
                .all(|(&p, &x)| {
                    // Comparing with this iteration's cap cannot use a copy of
                    // that cap from the previous iteration of the same loop.
                    (!self.dominates(b, p)
                        || self.def_block(cap).is_none_or(|c| !self.dominates(b, c)))
                        && test(x)
                }),
            Some(Def::Inst(b, j)) => {
                let i = &self.p.blocks[b].insts[j];
                let first = i.writes.first().map(|w| w.1) == Some(v);
                let operand = |op: &Operand| {
                    subset && const_operand(op) == Some(0)
                        || operand_val(i, op).is_some_and(&mut test)
                };
                match (&i.fields, i.name()) {
                    (_, "s_mov_b32") if first => first_src_operand(i).is_some_and(operand),
                    (_, "s_and_saveexec_b32")
                        if first || subset && i.write(Reg::S(EXEC)) == Some(v) =>
                    {
                        i.read(Reg::S(EXEC)).is_some_and(test)
                    }
                    (Fields::Sop { src, .. }, "s_and_b32") if first && subset => {
                        src.iter().any(operand)
                    }
                    (Fields::Sop { src, .. }, "s_andn2_b32") if first && subset => {
                        src.first().is_some_and(operand)
                    }
                    (Fields::Sop { src, .. }, "s_or_b32" | "s_cselect_b32") if first && subset => {
                        src.iter().all(operand)
                    }
                    (Fields::Valu(_), op)
                        if subset
                            && op.starts_with("v_cmp")
                            && (first || i.write(Reg::S(EXEC)) == Some(v)) =>
                    {
                        i.read(Reg::S(EXEC)).is_some_and(test)
                    }
                    _ => false,
                }
            }
            _ => false,
        };
        path.remove(&v);
        result
    }

    fn triangle_word(&self, v: Value, result: Value, box_exec: Value, depth: u8) -> bool {
        if v == result {
            return true;
        }
        if depth > 20 {
            return false;
        }
        match self.defs[v.0 as usize] {
            Some(Def::Phi(b, j)) => self.p.blocks[b].phis[j]
                .args
                .iter()
                .all(|&x| self.triangle_word(x, result, box_exec, depth + 1)),
            Some(Def::Inst(b, j)) => {
                let i = &self.p.blocks[b].insts[j];
                matches!(i.fields, Fields::Valu(_))
                    && i.name() != "v_writelane_b32"
                    && i.read(Reg::S(EXEC))
                        .is_some_and(|g| self.mask_source(g, box_exec, true, 0))
                    && self
                        .value_reg(v)
                        .and_then(|r| i.read(r))
                        .is_some_and(|old| self.triangle_word(old, result, box_exec, depth + 1))
            }
            _ => false,
        }
    }

    fn box_partition(&self, image: &Inst, load: &Inst, node: Value, proc: usize) -> Option<Value> {
        let same_cmp = |i: &Inst| {
            let (four, kind) = two_src_operands(i)?;
            (address::plain(i)
                && const_operand(four) == Some(4)
                && self.alu_args(operand_val(i, kind)?, "v_and_b32", i)?
                    == [Word::Imm(6), Word::Ssa(node)]
                && i.read(Reg::S(EXEC)) == image.read(Reg::S(EXEC)))
            .then_some(())
        };
        let leaf_mask = load.read(Reg::S(EXEC))?;
        let leaf = self.def_inst(leaf_mask)?.1;
        if leaf.name() != "v_cmpx_ne_u32" || leaf.write(Reg::S(EXEC)) != Some(leaf_mask) {
            return None;
        }
        same_cmp(leaf)?;
        let [prev] = self.p.blocks[proc].preds.as_slice() else {
            return None;
        };
        let bb = &self.p.blocks[*prev];
        let subtract = bb.insts.iter().rev().nth(1)?;
        if subtract.name() != "s_andn2_b32"
            || bb.insts.last()?.name() != "s_cbranch_execz"
            || !matches!(bb.exit,Exit::Branch{not_taken,..} if not_taken==proc)
            || subtract.write(Reg::S(EXEC)) != bb.insts.last()?.read(Reg::S(EXEC))
        {
            return None;
        }
        let skip = self.p.blocks[proc].insts.iter().rev().nth(1)?;
        if skip.read(Reg::S(EXEC)) != subtract.write(Reg::S(EXEC)) {
            return None;
        }
        let (before, boxes) = two_src_operands(subtract)?;
        let before = operand_val(subtract, before)?;
        let gate = self.def_inst(before)?.1;
        if gate.name() != "s_and_saveexec_b32"
            || gate.writes.first()?.1 != before
            || !self.mask_source(
                gate.read(Reg::S(EXEC))?,
                image.read(Reg::S(EXEC))?,
                false,
                0,
            )
        {
            return None;
        }
        let test = self
            .def_inst(operand_val(gate, first_src_operand(gate)?)?)?
            .1;
        let box_exec = gate.write(Reg::S(EXEC))?;
        same_cmp(test)?;
        (test.name() == "v_cmp_eq_u32"
            && self.mask_source(operand_val(subtract, boxes)?, box_exec, false, 0))
        .then_some(box_exec)
    }

    fn shadow_geometry(
        &self,
        primary: Site,
        leaf: Site,
        tri_word: Value,
        instance: &Inst,
        proc: usize,
        before: usize,
    ) -> Option<Value> {
        use Word::{Imm, Ssa};
        let image = |site: Site| &self.p.blocks[site.0].insts[site.1];
        let node_desc = |i: &Inst| {
            let node = *image_addr_values(i)?.first()?;
            let Fields::Image {
                rsrc: Operand::Scalar { code, .. },
                ..
            } = i.fields
            else {
                return None;
            };
            Some((node, i.read(Reg::S(code))?))
        };
        let (node, blas) = node_desc(image(leaf))?;
        let (root, tlas) = node_desc(image(primary))?;
        let [Ssa(root)] = self.alu_args(root, "v_mov_b32", image(primary))? else {
            return None;
        };
        let (_, load) = self.def_inst(tri_word)?;
        let Fields::Flat {
            addr: Some(ref pair @ Operand::Vector { dwords: 2, .. }),
            saddr: Some(Operand::Scalar { code: NULL, .. }),
            offset: 0,
            ..
        } = load.fields
        else {
            return None;
        };
        if load.name() != "global_load_dword" || load.writes.first()?.1 != tri_word {
            return None;
        }
        let addr = operand_val(load, pair)?;
        let [Imm(48), Ssa(low)] = self.alu_args(addr, "v_or_b32", load)? else {
            return None;
        };
        self.shadow_node_sum(
            low,
            entry::upper_operand(load, pair)?,
            blas,
            node,
            load.read(Reg::S(EXEC)),
        )?;
        if !self.same_header_word(blas, instance.writes.get(3)?.1) {
            return None;
        }
        let bb = &self.p.blocks[proc];
        let skip = bb.insts.iter().rev().nth(1)?;
        if !matches!(bb.exit,Exit::Branch{not_taken,..} if not_taken==before)
            || bb.insts.last()?.name() != "s_cbranch_execz"
            || skip.name() != "v_cmpx_ne_u32"
            || !address::plain(skip)
            || skip.write(Reg::S(EXEC)) != bb.insts.last()?.read(Reg::S(EXEC))
        {
            return None;
        }
        let (seven, kind) = two_src_operands(skip)?;
        if const_operand(seven) != Some(7)
            || self.alu_args(operand_val(skip, kind)?, "v_and_b32", skip)? != [Imm(7), Ssa(node)]
        {
            return None;
        }
        // The containing TLAS instance supplies both the instance flags (word
        // one) and this BLAS descriptor (word three), from the +48 header.
        let Fields::Smem { base: ref pair, .. } = instance.fields else {
            return None;
        };
        let low = operand_val(instance, pair)?;
        if flat_or_smem_load_info(instance, low) != Some((4, 0)) {
            return None;
        }
        let [Ssa(selected)] = self.alu_args(low, "v_readfirstlane_b32", instance)? else {
            return None;
        };
        let select = self.def_inst(selected)?.1;
        let [Ssa(raw), Ssa(tag), Ssa(mask)] =
            self.alu_args(selected, "v_cndmask_b32", self.def_inst(low)?.1)?
        else {
            return None;
        };
        let [Ssa(tag)] = self.alu_args(tag, "v_mov_b32", select)? else {
            return None;
        };
        if self.alu_args(tag, "s_or_b32", select)? != [Ssa(raw), Imm(48)] {
            return None;
        }
        let gate = self.def_inst(mask)?.1;
        let [Ssa(exec), Imm(0)] = self.alu_args(mask, "s_cselect_b32", select)? else {
            return None;
        };
        let scc = gate.read(Reg::Scc)?;
        let cmp = self.def_inst(scc)?.1;
        let (kind, six) = two_src_operands(cmp)?;
        if cmp.name() != "s_cmp_eq_u32"
            || const_operand(six) != Some(6)
            || self.alu_args(operand_val(cmp, kind)?, "s_and_b32", cmp)? != [Ssa(root), Imm(7)]
            || select.read(Reg::S(EXEC)) != Some(exec)
            || self.def_inst(low)?.1.read(Reg::S(EXEC)) != Some(exec)
        {
            return None;
        }
        let Ssa(hi) = self.copy_word(Ssa(entry::upper_operand(instance, pair)?)) else {
            return None;
        };
        self.shadow_node_sum(raw, hi, tlas, root, None)?;
        // Reaching this BLAS requires the very mask selecting the +48
        // instance header above, with an empty mask taking the bypass.
        let enters = self.p.blocks.iter().enumerate().any(|(b, bb)| {
            let Exit::Branch { taken, not_taken } = bb.exit else {
                return false;
            };
            if !self.dominates(not_taken, leaf.0)
                || self.p.blocks[not_taken].preds != [b]
                || !self.dominates(b, not_taken)
                || self.dominates(not_taken, taken)
                || bb.insts.last().is_none_or(|i| i.name() != "s_cbranch_vccz")
            {
                return false;
            }
            let Some(vcc) = bb.insts.last().and_then(|i| i.read(Reg::S(crate::ir::VCC))) else {
                return false;
            };
            self.alu_args(vcc, "s_and_b32", bb.insts.last().unwrap())
                .is_some_and(|args| {
                    args == [Ssa(exec), Ssa(mask)] || args == [Ssa(mask), Ssa(exec)]
                })
        });
        enters.then_some(())?;
        self.box_partition(image(leaf), load, node, proc)
    }

    pub(super) fn shadow_flags(
        &self,
        nodes: &HashSet<usize>,
        tri: &TriangleHitInfo,
        header: usize,
        primary: Site,
        tri_bvh: Site,
    ) -> Option<Value> {
        use Word::{Imm, Ssa};
        self.shared_entry(header, nodes, primary)?;
        if !tri.accept_first_hit || !matches!(tri.mode, HitMode::AnyHit(_)) {
            return None;
        }
        let hit = tri.hit_block;
        let opacity = tri.opacity_block?;
        let (_, opaque) = self.hoisted_opacity_gate(hit, opacity)?;
        let before = self.p.blocks[hit].preds[0];
        let bb = &self.p.blocks[before];
        let gate = bb.insts.iter().rev().nth(1)?;
        if gate.name() != "s_and_saveexec_b32"
            || !matches!(bb.exit,Exit::Branch{not_taken,..} if not_taken==hit)
            || bb.insts.last()?.name() != "s_cbranch_execz"
        {
            return None;
        }
        let compile = |v: Value| self.def_inst(v).map(|(_, i)| i);
        let mask = operand_val(gate, first_src_operand(gate)?)?;
        let [Imm(0), Ssa(culled)] = self.alu_args(mask, "v_cmp_eq_u32", gate)? else {
            return None;
        };
        let [Ssa(choice), Ssa(raw)] = self.alu_args(culled, "v_and_b32", compile(mask)?)? else {
            return None;
        };
        let choices = self.alu_args(choice, "v_cndmask_b32", compile(culled)?)?;
        if choices != [Imm(128), Imm(64), Ssa(opaque)] {
            return None;
        }
        let [Imm(1), Ssa(op)] = self.alu_args(opaque, "v_cmp_eq_u32", compile(choice)?)? else {
            return None;
        };
        let [Ssa(tri_word), Imm(1), Ssa(inst_op)] =
            self.alu_args(op, "v_and_or_b32", compile(opaque)?)?
        else {
            return None;
        };
        let [Ssa(word), Imm(4), Imm(2)] = self.alu_args(inst_op, "v_bfe_u32", compile(op)?)? else {
            return None;
        };
        // The upper instance bits leave the six low opacity/culling bits intact.
        let [Imm(0xffff_ff80), _, Ssa(low)] =
            self.alu_args(word, "v_bfi_b32", compile(inst_op)?)?
        else {
            return None;
        };
        let [Imm(60), Ssa(sh), _] = self.alu_args(low, "v_bfi_b32", compile(word)?)? else {
            return None;
        };
        let [Imm(2), Ssa(packed)] = self.alu_args(sh, "v_lshlrev_b32", compile(low)?)? else {
            return None;
        };
        let [Ssa(face), Imm(3), Ssa(mode)] = self.alu_args(packed, "v_and_or_b32", compile(sh)?)?
        else {
            return None;
        };
        let [Ssa(instance), Ssa(over), Ssa(condition)] =
            self.alu_args(mode, "v_cndmask_b32", compile(packed)?)?
        else {
            return None;
        };
        if self.alu_args(condition, "v_cmp_ne_u32", compile(mode)?)? != [Imm(0), Ssa(over)] {
            return None;
        }
        let [Imm(12), Ssa(raw_sh)] = self.alu_args(over, "v_and_b32", compile(mode)?)? else {
            return None;
        };
        if self.alu_args(raw_sh, "v_lshlrev_b32", compile(over)?)? != [Imm(2), Ssa(raw)] {
            return None;
        }
        let [Ssa(byte), Imm(12)] = self.alu_args(instance, "s_and_b32", compile(mode)?)? else {
            return None;
        };
        let [Ssa(inst_word), Imm(24)] = self.alu_args(byte, "s_lshr_b32", compile(instance)?)?
        else {
            return None;
        };
        let ((ib, _), load) = self.def_inst(inst_word)?;
        if !nodes.contains(&ib)
            || load.name() != "s_load_dwordx4"
            || load.writes.get(1)?.1 != inst_word
        {
            return None;
        }
        let proc = *bb.preds.first()?;
        for b in [proc, before, hit, opacity] {
            let ins = &self.p.blocks[b].insts;
            if ins[..ins.len().checked_sub(2)?]
                .iter()
                .any(|i| i.write(Reg::S(EXEC)).is_some())
            {
                return None;
            }
        }
        let skip = self.p.blocks[proc].insts.iter().rev().nth(1)?;
        let hit_gate = self.p.blocks[hit].insts.iter().rev().nth(1)?;
        let side_gate = self.p.blocks[opacity].insts.iter().rev().nth(1)?;
        if gate.read(Reg::S(EXEC)) != skip.write(Reg::S(EXEC))
            || hit_gate.read(Reg::S(EXEC)) != gate.write(Reg::S(EXEC))
            || side_gate.read(Reg::S(EXEC)) != hit_gate.write(Reg::S(EXEC))
        {
            return None;
        }
        let box_exec = self.shadow_geometry(primary, tri_bvh, tri_word, load, proc, before)?;
        // Instance cull-disable and front-counterclockwise bits normalize the
        // determinant sign before the hardware front/back culling test.
        let [Ssa(norm), Ssa(sign)] = self.alu_args(face, "v_mul_i32_i24", compile(packed)?)? else {
            return None;
        };
        let [Ssa(signed), Ssa(enable)] = self.alu_args(norm, "v_mul_i32_i24", compile(face)?)?
        else {
            return None;
        };
        if self.alu_args(enable, "s_and_b32", compile(norm)?)? != [Ssa(signed), Imm(1)] {
            return None;
        }
        let [Ssa(flip), Imm(0x20000)] = self.alu_args(signed, "s_bfe_i32", compile(norm)?)? else {
            return None;
        };
        if self.alu_args(flip, "s_xor_b32", compile(signed)?)? != [Ssa(byte), Imm(1)] {
            return None;
        }
        let [Imm(u32::MAX), Ssa(bits)] = self.alu_args(sign, "v_max_i32", compile(face)?)? else {
            return None;
        };
        let [Ssa(full), Imm(4), Imm(2)] = self.alu_args(bits, "v_bfe_i32", compile(sign)?)? else {
            return None;
        };
        if self.alu_args(full, "v_or_b32", compile(bits)?)? != [Imm(0x204), Ssa(raw)]
            || nodes.contains(&self.def_block(full)?)
        {
            return None;
        }
        let ((fb, _), flag_writer) = self.def_inst(full)?;
        let first = self.p.blocks[header]
            .insts
            .iter()
            .find(|i| i.read(Reg::S(EXEC)).is_some())?;
        if !self.dominates(fb, header)
            || self.p.blocks[header]
                .preds
                .iter()
                .filter(|&&p| !nodes.contains(&p))
                .any(|&p| !self.mask_covers(flag_writer, first, &[(header, p)]))
        {
            return None;
        }
        if !self
            .shadow_flag_values(raw, self.st.reach[self.def_block(full)?], 0)?
            .iter()
            .all(|k| matches!(k, 0x10 | 0x20 | 0x90 | 0xa0))
        {
            return None;
        }
        // The face comparison must actually participate in the distance gate,
        // using the determinant of this triangle and the same packed flags.
        let hb = &self.p.blocks[hit];
        let reciprocal = hb.insts.iter().find(|i| i.name() == "v_rcp_f32")?;
        let det = single_reg_src(reciprocal)?;
        let tested = hb.insts.iter().rev().nth(1)?;
        if tested.name() != "s_and_saveexec_b32"
            || !matches!(hb.exit,Exit::Branch{not_taken,..} if not_taken==opacity)
            || hb.insts.last()?.name() != "s_cbranch_execz"
        {
            return None;
        }
        let mut pending = vec![operand_val(tested, first_src_operand(tested)?)?];
        let mut orientation = false;
        let mut lower_bound = None;
        let mut upper_bound = None;
        let mut extent = None;
        let tri_words = image_data_values(&self.p.blocks[tri_bvh.0].insts[tri_bvh.1])?;
        let numerator = *tri_words.first()?;
        if !address::plain(reciprocal) || !self.triangle_word(det, *tri_words.get(1)?, box_exec, 0)
        {
            return None;
        }
        while let Some(m) = pending.pop() {
            let ((b, _), i) = self.def_inst(m)?;
            if b != hit || !address::plain(i) || i.writes.first()?.1 != m {
                return None;
            }
            match i.name() {
                "s_and_b32" => {
                    let (a, b) = two_src_operands(i)?;
                    pending.extend([operand_val(i, a)?, operand_val(i, b)?]);
                }
                "v_cmp_lt_i32" => {
                    let [Imm(u32::MAX), Ssa(product)] = self.alu_args(m, "v_cmp_lt_i32", tested)?
                    else {
                        return None;
                    };
                    let prod = compile(product)?;
                    let Fields::Valu(p) = &prod.fields else {
                        return None;
                    };
                    let sdwa = p.sdwa.as_ref()?;
                    // Sign-extend the high halfword of the float determinant.
                    // A nonnegative determinant with zero high halfword has an
                    // overflowing reciprocal and cannot pass the finite hit range.
                    if prod.name() != "v_mul_i32_i24"
                        || p.neg != 0
                        || p.abs != 0
                        || p.clamp
                        || p.omod != 0
                        || p.opsel != 0
                        || p.dpp.is_some()
                        || sdwa.dst_sel != 6
                        || sdwa.src_sel != [5, 6]
                        || sdwa.sext != 1
                        || prod.writes.first()?.1 != product
                        || !self.mask_covers(prod, i, &[])
                    {
                        return None;
                    }
                    let (d, f) = two_src_operands(prod)?;
                    let (d, f) = (operand_val(prod, d)?, operand_val(prod, f)?);
                    if d != det
                        || self.alu_args(f, "v_bfe_i32", compile(product)?)?
                            != [Ssa(word), Imm(2), Imm(2)]
                    {
                        return None;
                    }
                    orientation = true;
                }
                "v_cmp_ge_f32" | "v_cmp_lt_f32" => {
                    let (t, bound) = two_src_operands(i)?;
                    let t = operand_val(i, t)?;
                    let ((b, _), mul) = self.def_inst(t)?;
                    if b != hit
                        || !matches!(mul.name(), "v_mul_f32" | "v_mul_legacy_f32")
                        || !address::plain(mul)
                    {
                        return None;
                    }
                    let (a, c) = two_src_operands(mul)?;
                    let (a, c) = (operand_val(mul, a)?, operand_val(mul, c)?);
                    let reciprocal = reciprocal.writes.first()?.1;
                    if !((a == reciprocal && self.triangle_word(c, numerator, box_exec, 0))
                        || (c == reciprocal && self.triangle_word(a, numerator, box_exec, 0)))
                    {
                        return None;
                    }
                    if i.name() == "v_cmp_lt_f32" {
                        if extent.replace(operand_val(i, bound)?).is_some()
                            || self.value_reg(operand_val(i, bound)?) != Some(tri.committed_t_reg)
                            || upper_bound.replace(t).is_some()
                        {
                            return None;
                        }
                    } else {
                        let lo = match Word::operand(i, bound)? {
                            Imm(k) => RayInput::Const(k),
                            Ssa(x) => self.pre_loop_input(x, nodes, header, 0)?,
                        };
                        if lo != tri.t_min.unwrap_or(RayInput::Const(0))
                            || lower_bound.replace(t).is_some()
                        {
                            return None;
                        }
                    }
                }
                _ => return None,
            }
        }
        self.shadow_commit(
            nodes,
            tri,
            header,
            (primary, tri_bvh),
            upper_bound?,
            extent?,
        )?;
        (bb.preds.len() == 1 && orientation && lower_bound.is_some() && lower_bound == upper_bound)
            .then_some(full)
    }
}
