//! Pair the two words of XDXR's 40-bit title-memory addresses. Low-word
//! provenance is tracked by the metadata analysis. A high word from another
//! address (including a carry from a different low sum) is never interchangeable.
use super::*;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(super) enum Word {
    Imm(u32),
    Ssa(Value),
}

impl Word {
    pub(super) fn operand(inst: &Inst, op: &Operand) -> Option<Self> {
        const_operand(op)
            .map(Self::Imm)
            .or_else(|| operand_val(inst, op).map(Self::Ssa))
    }
}

pub(super) fn plain(inst: &Inst) -> bool {
    !matches!(&inst.fields, Fields::Valu(f) if f.neg != 0 || f.abs != 0 || f.opsel != 0 ||
        f.clamp || f.omod != 0 || f.sdwa.is_some() || f.dpp.is_some())
}

impl Analyzer<'_> {
    pub(super) fn unguarded_def(&self, v: Value) -> Option<&Inst> {
        let ((b, i), inst) = self.def_inst(v)?;
        (self.st.guards[b][i].is_none() && plain(inst)).then_some(inst)
    }

    pub(super) fn copy_word(&self, mut w: Word) -> Word {
        for _ in 0..12 {
            let Word::Ssa(v) = w else { break };
            let Some(inst) = self.unguarded_def(v) else {
                break;
            };
            if inst.writes.first().map(|x| x.1) != Some(v) {
                break;
            }
            match inst.name() {
                "s_mov_b32" | "v_mov_b32" => {
                    if let Some(src) =
                        first_src_operand(inst).and_then(|op| Word::operand(inst, op))
                    {
                        w = src
                    } else {
                        break;
                    }
                }
                "s_movk_i32" => {
                    if let Fields::Sop { imm: Some(k), .. } = inst.fields {
                        return Word::Imm(k as i16 as i32 as u32);
                    } else {
                        break;
                    }
                }
                _ => break,
            }
        }
        w
    }

    // The memory directory indexes only the low eight bits of the high word.
    // Both the common 0xff and the preceding descriptor 0xffff mask preserve them.
    pub(super) fn tlas_high_byte(&self, high: Word, tlas_hi: Value, depth: usize) -> bool {
        if depth >= 6 {
            return false;
        }
        let hi = self.copy_word(high);
        if hi == Word::Ssa(tlas_hi) {
            return true;
        }
        let Word::Ssa(v) = hi else { return false };
        let Some(inst) = self.unguarded_def(v) else {
            return false;
        };
        if !matches!(inst.name(), "s_and_b32" | "v_and_b32") || inst.writes[0].1 != v {
            return false;
        }
        let Some((a, b)) = two_src_operands(inst) else {
            return false;
        };
        [(a,b),(b,a)].iter().any(|&(mask,src)| {
            matches!(Word::operand(inst,mask).map(|w|self.copy_word(w)),Some(Word::Imm(k)) if k & 255 == 255) &&
            Word::operand(inst,src).is_some_and(|w| self.tlas_high_byte(w,tlas_hi,depth+1))
        })
    }

    pub(super) fn is_wide_sum(&self, low: &Inst, high: &Inst, lo: Value, hi: Value) -> bool {
        if !plain(low)
            || !plain(high)
            || low.writes.first().map(|w| w.1) != Some(lo)
            || high.writes.first().map(|w| w.1) != Some(hi)
        {
            return false;
        }
        match (low.name(), high.name(), &low.fields, &high.fields) {
            ("s_add_u32", "s_addc_u32", _, _) => low
                .write(Reg::Scc)
                .is_some_and(|c| high.read(Reg::Scc) == Some(c)),
            ("v_add_co_u32", "v_add_co_ci_u32", Fields::Valu(l), Fields::Valu(r)) => {
                matches!((l.carry,r.src.get(2)),(Some(Operand::Scalar{code:out,..}),Some(Operand::Scalar{code:input,..})) if (0..1+u8::from(self.p.wave64)).all(|k|low.write(Reg::S(out+k)).is_some_and(|c|high.read(Reg::S(input+k))==Some(c))))
            }
            _ => false,
        }
    }

    /// XDXR's leaf has primitive indices at bytes 52 and 56. Slot one
    /// selects the second dword; every other slot selects the first.
    pub(super) fn primitive_slot_select(&self, v: Value, tags: &HashMap<Value, BvhTag>) -> bool {
        let Some(((block, _), inst)) = self.def_inst(v) else {
            return false;
        };
        if !plain(inst) || inst.writes.first().is_none_or(|w| w.1 != v) {
            return false;
        }
        let Fields::Valu(f) = &inst.fields else {
            return false;
        };
        let [a, b, m] = f.src.as_slice() else {
            return false;
        };
        if inst.name() != "v_cndmask_b32" || const_operand(a) != Some(52) {
            return false;
        }
        let (Some(offset), Some(mask)) = (operand_val(inst, b), operand_val(inst, m)) else {
            return false;
        };
        let reaching = |v| {
            let ((b, j), op) = self.def_inst(v)?;
            (plain(op)
                && op.writes.first()?.1 == v
                && (self.st.guards[b][j].is_none()
                    || b == block && self.mask_covers(op, inst, &[])))
            .then_some(op)
        };
        let (Some(add), Some(cmp)) = (reaching(offset), reaching(mask)) else {
            return false;
        };
        let (Fields::Valu(f), Fields::Valu(c)) = (&add.fields, &cmp.fields) else {
            return false;
        };
        let ([slot, shift, base], [two, index]) = (f.src.as_slice(), c.src.as_slice()) else {
            return false;
        };
        let Some(slot) = operand_val(add, slot) else {
            return false;
        };
        let mask_pair = !self.p.wave64
            || match *m {
                Operand::Scalar { code, dwords: 2 } => code.checked_add(1).is_some_and(|hi| {
                    cmp.write(Reg::S(hi))
                        .is_some_and(|value| inst.read(Reg::S(hi)) == Some(value))
                }),
                _ => false,
            };
        mask_pair
            && add.name() == "v_lshl_add_u32"
            && cmp.name() == "v_cmp_gt_u32"
            && const_operand(shift) == Some(2)
            && const_operand(base) == Some(52)
            && const_operand(two) == Some(2)
            && operand_val(cmp, index) == Some(slot)
            && tags.get(&slot) == Some(&BvhTag::TriSlot)
    }

    /// The low word of (TLAS >> 8) names all 40 address bits except the
    /// eight alignment zeros. When proving both words, the source high byte
    /// must come from this same descriptor on each feasible PHI edge.
    pub(super) fn packed_tlas(
        &self,
        v: Value,
        tlas: Value,
        high: Option<Value>,
        depth: usize,
    ) -> bool {
        if depth >= 10 {
            return false;
        }
        let Word::Ssa(v) = self.copy_word(Word::Ssa(v)) else {
            return false;
        };
        if let Some(Def::Phi(b, i)) = self.defs[v.0 as usize] {
            let mut feasible = self.p.blocks[b]
                .preds
                .iter()
                .zip(&self.p.blocks[b].phis[i].args)
                .filter(|(p, _)| self.native_edge(b, **p))
                .peekable();
            return feasible.peek().is_some()
                && feasible.all(|(_, a)| self.packed_tlas(*a, tlas, high, depth + 1));
        }
        let Some(i) = self.unguarded_def(v) else {
            return false;
        };
        if i.writes[0].1 != v {
            return false;
        }
        let Some((a, b)) = two_src_operands(i) else {
            return false;
        };
        let (p, k) = match i.name() {
            "s_lshr_b64" => (a, b),
            "v_lshrrev_b64" => (b, a),
            _ => return false,
        };
        const_operand(k) == Some(8)
            && operand_val(i, p).is_some_and(|lo| self.descriptor_low(lo, tlas, 0))
            && high.is_none_or(|hi| {
                super::entry::upper_operand(i, p)
                    .is_some_and(|h| self.tlas_high_byte(Word::Ssa(h), hi, 0))
            })
    }

    fn descriptor_low(&self, v: Value, tlas: Value, depth: usize) -> bool {
        if depth >= 6 {
            return false;
        }
        let Word::Ssa(v) = self.copy_word(Word::Ssa(v)) else {
            return false;
        };
        if v == tlas {
            return true;
        }
        let Some(i) = self.unguarded_def(v) else {
            return false;
        };
        i.writes[0].1 == v
            && matches!(i.name(), "s_and_b32" | "v_and_b32")
            && other_val_from_const_op(i, 0xffff_ff00)
                .is_some_and(|x| self.descriptor_low(x, tlas, depth + 1))
    }

    pub(super) fn unpacked_tlas(&self, v: Value, tlas: Value, high: Option<Value>) -> bool {
        let Some(sh) = self.unguarded_def(v) else {
            return false;
        };
        let Some((a, b)) = two_src_operands(sh) else {
            return false;
        };
        let (p, k) = match sh.name() {
            "s_lshl_b64" => (a, b),
            "v_lshlrev_b64" => (b, a),
            _ => return false,
        };
        sh.writes[0].1 == v
            && const_operand(k) == Some(8)
            && operand_val(sh, p).is_some_and(|x| self.packed_tlas(x, tlas, high, 0))
    }

    fn high_source(&self, word: Word, depth: usize) -> Word {
        let word = self.copy_word(word);
        if depth < 6
            && let Word::Ssa(v) = word
            && let Some(i) = self.unguarded_def(v)
            && matches!(i.name(), "s_and_b32" | "v_and_b32")
            && i.writes[0].1 == v
            && let Some(k) = other_const_arg_any(i)
            && k & 255 == 255
            && let Some(src) = other_val_from_const_op(i, k)
        {
            return self.high_source(Word::Ssa(src), depth + 1);
        }
        word
    }

    pub(super) fn address_pair(
        &self,
        lo: Word,
        hi: Word,
        tlas: (Value, Value),
        tags: &HashMap<Value, BvhTag>,
        depth: usize,
    ) -> bool {
        if depth > 8 {
            return false;
        }
        let lo = self.copy_word(lo);
        let hi = self.high_source(hi, 0);
        let Word::Ssa(v) = lo else {
            return hi == Word::Imm(0);
        };
        if self.unpacked_tlas(v, tlas.0, Some(tlas.1))
            && self
                .def_inst(v)
                .is_some_and(|(_, i)| i.writes.get(1).is_some_and(|w| Word::Ssa(w.1) == hi))
        {
            return true;
        }
        if self.extract_base_low_offset(v, tlas.0, 0).is_some()
            && self.tlas_high_byte(hi, tlas.1, 0)
        {
            return true;
        }
        if hi == Word::Imm(0) && tags.get(&v) == Some(&BvhTag::TriPrimOff) {
            return true; // scalar intra-node offset; both select arms are verified separately
        }
        if let Some(Def::Phi(b, pi)) = self.defs[v.0 as usize] {
            let his = match hi {
                Word::Ssa(h)
                    if let Some(Def::Phi(hb, i)) = self.defs[h.0 as usize]
                        && hb == b =>
                {
                    Some(&self.p.blocks[b].phis[i].args)
                }
                _ => None,
            };
            let mut feasible = self.p.blocks[b].phis[pi]
                .args
                .iter()
                .enumerate()
                .filter(|(i, _)| self.native_edge(b, self.p.blocks[b].preds[*i]))
                .peekable();
            return feasible.peek().is_some()
                && feasible.all(|(i, &l)| {
                    self.address_pair(
                        Word::Ssa(l),
                        his.map_or(hi, |a| Word::Ssa(a[i])),
                        tlas,
                        tags,
                        depth + 1,
                    )
                });
        }
        let Some(low) = self.unguarded_def(v) else {
            return false;
        };
        if low.writes.first().map(|r| r.1) != Some(v) {
            return false;
        }
        if matches!(low.name(), "s_or_b32" | "v_or_b32")
            && let Some(k) = other_const_arg_any(low)
            && k < 128
        {
            // OR of a low-word field never produces a carry into its high word.
            return other_val_from_const_op(low, k)
                .is_some_and(|x| self.address_pair(Word::Ssa(x), hi, tlas, tags, depth + 1));
        }
        if matches!(low.name(), "v_lshlrev_b64" | "s_lshl_b64") {
            let Some((a, b)) = two_src_operands(low) else {
                return false;
            };
            let (pair, amount) = if low.name() == "s_lshl_b64" {
                (a, b)
            } else {
                (b, a)
            };
            let upper = match *pair {
                Operand::Scalar { code, dwords: 2 } => low.read(Reg::S(code + 1)),
                Operand::Vector { reg, dwords: 2 } => low.read(Reg::V(reg + 1)),
                _ => None,
            };
            return low.writes.get(1).is_some_and(|r| Word::Ssa(r.1) == hi)
                && match const_operand(amount) {
                    Some(8) => tags.get(&v) == Some(&BvhTag::BlasBaseLo),
                    Some(6) => {
                        matches!(
                            tags.get(&v),
                            Some(BvhTag::InstByteOff(0) | BvhTag::TriByteOff(0))
                        ) && upper.is_some_and(|u| self.copy_word(Word::Ssa(u)) == Word::Imm(0))
                    }
                    _ => false,
                };
        }
        let Word::Ssa(h) = hi else { return false };
        let Some(high) = self.unguarded_def(h) else {
            return false;
        };
        if high.writes.first().map(|r| r.1) != Some(h) {
            return false;
        }
        if !self.is_wide_sum(low, high, v, h) {
            return false;
        }
        let operands = |i: &Inst| {
            let (a, b) = two_src_operands(i)?;
            Some((Word::operand(i, a)?, Word::operand(i, b)?))
        };
        let (Some((a, b)), Some((c, d))) = (operands(low), operands(high)) else {
            return false;
        };
        [(c, d), (d, c)].into_iter().any(|(x, y)| {
            self.address_pair(a, x, tlas, tags, depth + 1)
                && self.address_pair(b, y, tlas, tags, depth + 1)
        })
    }
}
