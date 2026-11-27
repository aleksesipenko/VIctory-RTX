//! Entry and hit-state proofs for the shared XDXR TLAS/BLAS traversal.
use super::*;

impl Analyzer<'_> {
    fn incoming_word(&self, v: Value, h: usize, pre: usize) -> Option<Value> {
        match self.defs[v.0 as usize] {
            Some(Def::Phi(b, k)) if b == h => Some(
                self.p.blocks[b].phis[k].args
                    [self.p.blocks[b].preds.iter().position(|&p| p == pre)?],
            ),
            _ => Some(v),
        }
    }

    pub(super) fn work_contains_only(&self, v: Value, mask: Value, depth: u8) -> bool {
        if v == mask {
            return true;
        }
        if depth > 12 {
            return false;
        }
        let value = |w: Value| self.copy_word(Word::Ssa(w));
        if value(v) == value(mask) {
            return true;
        }
        let Some((_, i)) = self.def_inst(v) else {
            return false;
        };
        if i.name() == "s_mov_b32" {
            return single_reg_src(i).is_some_and(|x| self.work_contains_only(x, mask, depth + 1));
        }
        if i.name() != "s_and_b32" || i.writes.first().is_none_or(|w| w.1 != v) {
            return false;
        }
        let args = |i: &Inst| {
            let (a, b) = two_src_operands(i)?;
            Some([value(operand_val(i, a)?), value(operand_val(i, b)?)])
        };
        if let Some((_, m)) = self.def_inst(mask)
            && m.name() == "s_and_b32"
            && m.writes.first().is_some_and(|w| w.1 == mask)
            && let (Some([a, b]), Some([c, d])) = (args(i), args(m))
            && ((a, b) == (c, d) || (a, b) == (d, c))
        {
            return true;
        }
        args(i).is_some_and(|xs| {
            xs.iter()
                .any(|x| matches!(*x,Word::Ssa(a) if self.work_contains_only(a,mask,depth+1)))
        })
    }

    pub(super) fn shared_root(
        &self,
        h: usize,
        pre: usize,
        nodes: &HashSet<usize>,
        site: Site,
        node: Value,
        work: Value,
    ) -> Option<()> {
        use Word::{Imm, Ssa};
        let image = &self.p.blocks[site.0].insts[site.1];
        let addr = *image_addr_values(image)?.first()?;
        if !self.mask_covers(self.def_inst(addr)?.1, image, &[])
            || self.alu_args(addr, "v_mov_b32", image)? != [Ssa(node)]
        {
            return None;
        }
        let Fields::Image {
            rsrc: Operand::Scalar { code, .. },
            ..
        } = image.fields
        else {
            return None;
        };
        let Ssa(base) = self.copy_word(Ssa(image.read(Reg::S(code))?)) else {
            return None;
        };
        let base = self.copy_word(Ssa(self.incoming_word(base, h, pre)?));
        let Ssa(root) = self.copy_word(Ssa(self.incoming_word(node, h, pre)?)) else {
            return None;
        };
        let ((rb, _), load) = self.def_inst(root)?;
        if nodes.contains(&rb) || !self.dominates(rb, h) || load.name() != "s_load_dword" {
            return None;
        }
        let Fields::Smem { base: pair, .. } = &load.fields else {
            return None;
        };
        let lo = operand_val(load, pair)?;
        let hi = entry::upper_operand(load, pair)?;
        let (raw, offset) = flat_or_smem_load_info(load, lo)?;
        if raw != 1 {
            return None;
        }
        let lo = match offset {
            48 => lo,
            0 => {
                let [Ssa(x), Imm(48)] = self.alu_args(lo, "s_or_b32", load)? else {
                    return None;
                };
                x
            }
            _ => return None,
        };
        let Ssa(lo) = self.copy_word(Ssa(lo)) else {
            return None;
        };
        let (_, sh) = self.def_inst(lo)?;
        let (pair, k) = two_src_operands(sh)?;
        if sh.name() != "s_lshl_b64"
            || sh.writes.first()?.1 != lo
            || self.copy_word(Ssa(hi)) != Ssa(sh.writes.get(1)?.1)
            || const_operand(k) != Some(8)
            || self.copy_word(Ssa(operand_val(sh, pair)?)) != base
            || self.copy_word(Ssa(entry::upper_operand(sh, pair)?)) != Imm(0)
        {
            return None;
        }
        let source = self.find_tlas_load(RayTemplate::Bvh32SharedTlas, site, nodes, h)?;
        let desc = &self.p.blocks[source.0].insts[source.1];
        let Ssa(mut packed) = base else {
            return None;
        };
        let (_, source) = self.def_inst(packed)?;
        if source.name() == "v_readlane_b32" {
            let (data, lane) = two_src_operands(source)?;
            let ff = self.def_inst(operand_val(source, lane)?)?.1;
            let mask = single_reg_src(ff)?;
            let copy = self.def_inst(operand_val(source, data)?)?.1;
            if ff.name() != "s_ff1_i32_b32"
                || copy.name() != "v_mov_b32"
                || !address::plain(copy)
                || !self.work_contains_only(mask, copy.read(Reg::S(EXEC))?, 0)
                || !self.work_contains_only(self.incoming_word(work, h, pre)?, mask, 0)
            {
                return None;
            }
            packed = single_reg_src(copy)?;
        }
        self.packed_tlas(
            packed,
            desc.writes.first()?.1,
            Some(desc.writes.get(1)?.1),
            0,
        )
        .then_some(())
    }

    /// The second BLAS enable term is the validity of *this instance's root*.
    /// Bits 31:30 are flags. The empty-root sentinel has every pointer bit set.
    fn instance_root_gate(
        &self,
        mask: Value,
        gate: &Inst,
        w2: Value,
        leaf: Site,
        pre: usize,
    ) -> Option<()> {
        use Word::{Imm, Ssa};
        let [Ssa(exec), Imm(0)] = self.alu_args(mask, "s_cselect_b32", gate)? else {
            return None;
        };
        if gate.read(Reg::S(EXEC)) != Some(exec) {
            return None;
        }
        let sel = self.def_inst(mask)?.1;
        let pos = self.def_inst(sel.read(Reg::Scc)?)?.1;
        let (root, zero) = two_src_operands(pos)?;
        if pos.name() != "s_cmp_gt_i32" || const_operand(zero) != Some(0) {
            return None;
        }
        let root = operand_val(pos, root)?;
        let [Ssa(clean), Imm(u32::MAX)] = self.alu_args(root, "s_cselect_b32", pos)? else {
            return None;
        };
        let valid = self.def_inst(self.def_inst(root)?.1.read(Reg::Scc)?)?.1;
        let (a, b) = two_src_operands(valid)?;
        if valid.name() != "s_cmp_lg_u32"
            || ![(a, b), (b, a)].iter().any(|&(x, y)| {
                operand_val(valid, x) == Some(clean)
                    && (const_operand(y) == Some(0x3fff_ffff)
                        || operand_val(valid, y).and_then(|v| self.eval_const(v, 0))
                            == Some(0x3fff_ffff))
            })
        {
            return None;
        }
        let and = self.def_inst(clean)?.1;
        let (x, y) = two_src_operands(and)?;
        if and.name() != "s_and_b32"
            || ![(x, y), (y, x)].iter().any(|&(src, k)| {
                operand_val(and, src).is_some_and(|v| self.same_header_word(v, w2))
                    && (const_operand(k) == Some(0x3fff_ffff)
                        || operand_val(and, k).and_then(|v| self.eval_const(v, 0))
                            == Some(0x3fff_ffff))
            })
        {
            return None;
        }
        let bb = &self.p.blocks[leaf.0];
        let image = &bb.insts[leaf.1];
        let node = *image_addr_values(image)?.first()?;
        if self.p.blocks[pre]
            .insts
            .iter()
            .chain(&bb.insts[..leaf.1])
            .any(|i| i.write(Reg::S(EXEC)).is_some())
        {
            return None;
        }
        let initial = self.incoming_word(node, leaf.0, pre)?;
        let ((b, _), write) = self.def_inst(initial)?;
        (b == pre
            && write.read(Reg::S(EXEC)) == gate.write(Reg::S(EXEC))
            && self.alu_args(initial, "v_mov_b32", write)? == [Ssa(root)])
        .then_some(())
    }

    pub(super) fn shared_filter(
        &self,
        h: usize,
        leaf: Site,
        nodes: &HashSet<usize>,
        guard: crate::ir::SharedRayEntry,
        outer_bits: Value,
    ) -> Option<RayInput> {
        let image = |s: Site| &self.p.blocks[s.0].insts[s.1];
        let Fields::Image {
            rsrc: Operand::Scalar { code, .. },
            ..
        } = image(leaf).fields
        else {
            return None;
        };
        let Word::Ssa(blas) = self.copy_word(Word::Ssa(image(leaf).read(Reg::S(code))?)) else {
            return None;
        };
        let (_, header) = self.def_inst(blas)?;
        if header.name() != "s_load_dwordx4" || header.writes.get(3)?.1 != blas {
            return None;
        }
        let w0 = header.writes.first()?.1;
        let find_phi = |reg| {
            self.p.blocks[h]
                .phis
                .iter()
                .find(|p| p.reg == reg)
                .map(|p| p.value)
        };
        let work = find_phi(Reg::S(guard.work))?;
        let wave = self.p.inputs.iter().find(|(r, _)| *r == Reg::S(EXEC))?.1;
        for &b in nodes {
            let bb = &self.p.blocks[b];
            let Some(branch) = bb.insts.last() else {
                continue;
            };
            let Exit::Branch { not_taken, .. } = bb.exit else {
                continue;
            };
            if self.p.blocks[not_taken].preds != [b] || !self.dominates(not_taken, leaf.0) {
                continue;
            }
            let Some((_, test)) = branch.read(Reg::Scc).and_then(|s| self.def_inst(s)) else {
                continue;
            };
            if !matches!(
                (branch.name(), test.name()),
                ("s_cbranch_scc0", "s_cmp_lg_u32") | ("s_cbranch_scc1", "s_cmp_eq_u32")
            ) {
                continue;
            }
            let Some(filtered) = other_val_from_const_op(test, 0) else {
                continue;
            };
            if !self.work_contains_only(filtered, work, 0) {
                continue;
            }
            let mut todo = vec![filtered];
            let mut filter = None;
            let mut exec_terms = Vec::new();
            let mut saw_work = false;
            let mut ok = true;
            while let Some(v) = todo.pop() {
                if v == work {
                    saw_work = true;
                    continue;
                }
                if self.mask_source(v, wave, false, 0) {
                    continue;
                }
                if self.value_reg(v) == Some(Reg::S(EXEC)) {
                    exec_terms.push(v);
                    continue;
                }
                let Some((_, op)) = self.def_inst(v) else {
                    ok = false;
                    break;
                };
                if op.name() == "s_and_b32" {
                    let (x, y) = two_src_operands(op)?;
                    todo.extend([operand_val(op, x)?, operand_val(op, y)?]);
                    continue;
                }
                if op.name() != "s_cselect_b32" {
                    ok = false;
                    break;
                }
                let (enabled, z) = two_src_operands(op)?;
                let exec = operand_val(op, enabled)?;
                if self.value_reg(exec) != Some(Reg::S(EXEC))
                    || const_operand(z) != Some(0)
                    || filter.is_some()
                {
                    ok = false;
                    break;
                }
                let cmp = self.def_inst(op.read(Reg::Scc)?)?.1;
                let (src, limit) = two_src_operands(cmp)?;
                let x = operand_val(cmp, src)?;
                let (w, mask) = match (cmp.name(), const_operand(limit)) {
                    ("s_cmp_gt_u32", Some(k)) => (x, decode_gt_threshold_mask(k)?),
                    ("s_cmp_lt_u32", _) => (
                        operand_val(cmp, limit)?,
                        decode_gt_threshold_mask(self.eval_const(x, 0)?)?,
                    ),
                    ("s_cmp_eq_u32", Some(1)) => {
                        let and = self.def_inst(x)?.1;
                        if and.name() != "s_and_b32" {
                            return None;
                        }
                        let shift = self.def_inst(other_val_from_const_op(and, 1)?)?.1;
                        let (word, amount) = two_src_operands(shift)?;
                        let bit = const_operand(amount)?;
                        if shift.name() != "s_lshr_b32" || !(24..32).contains(&bit) {
                            return None;
                        }
                        (operand_val(shift, word)?, 1 << (bit - 24))
                    }
                    ("s_cmp_lg_u32", Some(0)) => {
                        let and = self.def_inst(x)?.1;
                        let (l, r) = two_src_operands(and)?;
                        let k = const_operand(r)?;
                        if and.name() != "s_and_b32" || k & 0x00ff_ffff != 0 {
                            return None;
                        }
                        (operand_val(and, l)?, k >> 24)
                    }
                    _ => {
                        ok = false;
                        break;
                    }
                };
                if !self.same_header_word(w, w0) {
                    ok = false;
                    break;
                }
                filter = Some((mask, exec));
            }
            if !ok || !saw_work {
                continue;
            }
            let Some((mask, exec)) = filter else {
                continue;
            };
            if exec_terms
                .iter()
                .any(|&v| !self.mask_source(v, exec, false, 0))
            {
                continue;
            }
            // The inclusion test must reach the per-lane EXEC gate of this
            // BLAS, including its polarity and the same shift/work bit.
            let found = nodes.iter().any(|&e| {
                let bb = &self.p.blocks[e];
                let Exit::Branch { not_taken, .. } = bb.exit else {
                    return false;
                };
                if !self.p.blocks[leaf.0].preds.contains(&not_taken)
                    || self.p.blocks[not_taken].preds != [e]
                    || !self.dominates(e, leaf.0)
                    || !self.dominates(b, e)
                {
                    return false;
                }
                let validate = || {
                    let gate = bb.insts.iter().rev().nth(1)?;
                    if gate.name() != "s_and_saveexec_b32"
                        || gate.read(Reg::S(EXEC)) != Some(exec)
                        || bb.insts.last()?.name() != "s_cbranch_execz"
                    {
                        return None;
                    }
                    let m = operand_val(gate, first_src_operand(gate)?)?;
                    let [Word::Ssa(a), Word::Ssa(c)] = self.alu_args(m, "s_and_b32", gate)? else {
                        return None;
                    };
                    [(a, c), (c, a)].into_iter().find(|&(v, root_gate)| {
                        let Some([Word::Imm(0), Word::Ssa(x)]) =
                            self.alu_args(v, "v_cmp_ne_u32", gate)
                        else {
                            return false;
                        };
                        self.alu_args(x, "v_and_b32", gate).is_some_and(|terms| {
                            terms == [Word::Ssa(filtered), Word::Ssa(outer_bits)]
                                || terms == [Word::Ssa(outer_bits), Word::Ssa(filtered)]
                        }) && self
                            .instance_root_gate(
                                root_gate,
                                gate,
                                header.writes[2].1,
                                leaf,
                                not_taken,
                            )
                            .is_some()
                    })
                };
                validate().is_some()
            });
            if found {
                return Some(RayInput::Const(mask));
            }
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "requires the FarOnly 881646f629ed5b13 console shader from the trace corpus"]
    fn real_shared_blas_requires_its_root_and_its_lane_bit() {
        let bytes = std::fs::read(std::env::var_os("VICTORY_RT_SHADOW_FIXTURE").unwrap()).unwrap();
        assert!(
            blake3::hash(&bytes)
                .to_hex()
                .starts_with("881646f629ed5b13")
        );
        let container = crate::Container::parse(&bytes).unwrap();
        let part = container
            .part(crate::FourCC::DXIL)
            .or(container.part(crate::FourCC::ACCC))
            .unwrap();
        let native = crate::native_programs(part.data)
            .unwrap()
            .into_iter()
            .find(|n| n.stage == crate::HwStage::Cs)
            .unwrap();
        for change in [
            None,
            Some((0xbc68, 0xbf02_8000u32, 0xbf04_8000u32)),
            Some((0xbc38, 0x365e_5e10, 0x365e_0010)),
        ] {
            let mut code = native.patched_code();
            if let Some((at, old, new)) = change {
                // Same inclusion ballot, but either reverse the BLAS root
                // test or use another VGPR instead of the verified lane bit.
                assert_eq!(code[at..at + 4], old.to_le_bytes());
                code[at..at + 4].copy_from_slice(&new.to_le_bytes());
            }
            let p = crate::ir::build(&code, native.wave64, native.vgprs).unwrap();
            let st = crate::structure::structure(&p).unwrap();
            let res = crate::resource::track(&p, &native.layout, native.stage, Some(&st)).unwrap();
            let scalar = res.scalar.iter().copied().collect();
            let a = Analyzer::new(&p, &st, &res, &scalar);
            let (h, nodes, sites) = a
                .candidates()
                .into_iter()
                .find(|(b, _, _)| p.blocks[*b].offset == 0xb560)
                .unwrap();
            let (_, primary, leaf) = a.classify_template(&nodes, &sites).unwrap();
            let (guard, _, bit) = a.shared_entry(h, &nodes, primary).unwrap();
            assert_eq!(
                a.shared_filter(h, leaf, &nodes, guard, bit).is_some(),
                change.is_none()
            );
        }
    }
}
