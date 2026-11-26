//! Adapted from maren-halv/VIctory-DXR @ cd87189 (GPL-3.0).
//! Connect recognized traversal regions to the per-lane IR lowering.
use super::super::ray_query::RayQuery;
use super::super::rayquery_lift::{
    self as lift, CandidateInput, PendingAnyHit, RayInput, RayLoop, RayLoopKind, RayMetaKind,
    RayMetaLoad, RayOutput, ResolvedRayInputs,
};
use super::*;

#[derive(Default)]
pub(super) struct LiftPlan {
    loops: Vec<RayLoop>,
    pub(super) entries: HashMap<usize, usize>,
    pub(super) skipped: HashSet<usize>,
    pub(super) meta: HashMap<Site, RayMetaLoad>,
    pub(super) skip_insts: HashSet<Site>,
    pub(super) anyhit_tails: HashMap<usize, usize>,
    pub(super) dead_lds: bool,
    pub(super) counted_pair: bool,
}

impl LiftPlan {
    pub(super) fn software(counted_pair: bool) -> Self {
        Self {
            counted_pair,
            ..Self::default()
        }
    }

    pub(super) fn recognize(x: &Parts, counted: bool) -> Self {
        let scalar = x.res.scalar.iter().copied().collect();
        let loops = lift::find_ray_loops(x.p, x.st, x.res, &scalar);
        if std::env::var_os("VICTORY_WAVE_WHY").is_some() {
            let mut kinds = [0; 3];
            for rl in &loops {
                kinds[match rl.kind {
                    RayLoopKind::OpaqueOnly => 0,
                    RayLoopKind::FrontBackTrack { .. } => 1,
                    RayLoopKind::AnyHit(_) => 2,
                }] += 1;
            }
            eprintln!(
                "RayQuery matched opaque/front-back/AnyHit={kinds:?} paired={}",
                PAIRED.get()
            );
        }
        Self::admit(x, loops, counted)
    }

    fn admit(x: &Parts, loops: Vec<RayLoop>, counted: bool) -> Self {
        // #854. A retained guest AnyHit CFG cannot be entered independently by
        // both virtual lanes. A wave-wide outer branch can also bring an inactive
        // lane to a trace. Keep the complete paired software program in these
        // cases, before substituting metadata, LDS writes or loop tails.
        let widened_outer = x.st.wide.iter().any(|b| {
            !loops
                .iter()
                .any(|rl| rl.entry_block == *b || rl.skipped_blocks.contains(b))
        });
        let paired_unsafe = PAIRED.get() && loops.iter().any(|rl| !paired_safe(x.p, rl));
        if widened_outer || paired_unsafe {
            if !loops.is_empty() && std::env::var_os("VICTORY_WAVE_WHY").is_some() {
                eprintln!(
                    "Wave64: preserving software traversal and its guest EXEC/AnyHit control flow"
                );
            }
            return Self::software(PAIRED.get() && counted);
        }
        // A shared guest metadata path after a software traversal must keep
        // reading the guest BVH. A previous hardware query can still be valid.
        if loops.iter().any(|rl| !rl.committed_meta_loads.is_empty())
            && x.res.bvh.iter().any(|&((b, _), _)| {
                !loops
                    .iter()
                    .any(|rl| rl.entry_block == b || rl.skipped_blocks.contains(&b))
            })
        {
            if std::env::var_os("VICTORY_WAVE_WHY").is_some() {
                eprintln!("RayQuery declined: guest and hardware traversal may share metadata");
            }
            return Self::software(PAIRED.get() && counted);
        }
        let single_wave = matches!(x.native.io, StageIo::Cs(cs)
            if cs.threads.iter().product::<u32>() <= if PAIRED.get() { 64 } else { 32 });
        let mut plan = Self::new(x.p, loops, single_wave);
        plan.counted_pair = PAIRED.get() && counted;
        plan
    }

    fn new(p: &Program, loops: Vec<RayLoop>, single_wave: bool) -> Self {
        let mut plan = Self::default();
        for (idx, rl) in loops.iter().enumerate() {
            plan.entries.insert(rl.entry_block, idx);
            plan.skipped.extend(&rl.skipped_blocks);
            for &(site, kind) in &rl.committed_meta_loads {
                plan.meta.insert(
                    site,
                    RayMetaLoad {
                        loop_idx: rl.rq_group,
                        candidate: false,
                        kind,
                    },
                );
            }
            if let RayLoopKind::AnyHit(ah) = &rl.kind {
                plan.anyhit_tails.insert(ah.tail_block, idx);
                plan.skip_insts
                    .extend(ah.skipped_tail_insts.iter().map(|i| (ah.tail_block, *i)));
                for &(site, kind) in &ah.cand_meta_loads {
                    plan.meta.insert(
                        site,
                        RayMetaLoad {
                            loop_idx: rl.rq_group,
                            candidate: true,
                            kind,
                        },
                    );
                }
            }
        }
        match dead_bvh_lds_writes(p, &plan.entries, &plan.skipped, single_wave) {
            Ok(Some(dead)) => {
                plan.dead_lds = true;
                plan.skip_insts.extend(dead);
            }
            Ok(None) => {}
            Err(op) => {
                if std::env::var_os("VICTORY_WAVE_WHY").is_some() {
                    eprintln!("RayQuery declined: observable LDS operation {op}");
                }
                return Self::default();
            }
        }
        plan.loops = loops;
        plan
    }

    pub(super) fn removed(&self, b: usize) -> bool {
        self.entries.contains_key(&b) || self.skipped.contains(&b)
    }
}

fn paired_safe(p: &Program, region: &RayLoop) -> bool {
    if matches!(region.kind, RayLoopKind::AnyHit(_)) {
        return false;
    }
    // A narrowing (or widening) of EXEC *inside* the removed CFG may gate
    // the first BVH instruction independently of entry EXEC. Preserve this
    // paired software program until that inner mask has an exact lowering.
    let mut blocks = std::iter::once(&region.entry_block).chain(&region.skipped_blocks);
    !blocks.any(|&b| {
        p.blocks[b].insts.iter().any(|inst| {
            inst.write(Reg::S(EXEC)).is_some()
                || inst.write(Reg::S(EXEC + 1)).is_some()
                || (region.template == lift::RayTemplate::Bvh32SharedTlas
                    && matches!(inst.name(), "s_and_b32" | "s_and_b64")
                    && inst
                        .reads
                        .iter()
                        .any(|(reg, _)| matches!(reg, Reg::S(n) if *n == EXEC || *n == EXEC + 1)))
        })
    })
}

/// EXEC on the edges entering the guest loop, before its own back-edge masks.
/// A lane may be disabled even when the preheader takes an unconditional jump.
fn entry_exec(p: &Program, st: &Structured, rl: &RayLoop) -> Option<Bool> {
    let internal = |b: usize| {
        b == rl.entry_block
            || rl.skipped_blocks.contains(&b)
            || matches!(&rl.kind, RayLoopKind::AnyHit(ah) if ah.body_blocks.contains(&b))
    };
    let before = |reg: Reg| {
        let initial = || p.inputs.iter().find(|(r, _)| *r == reg).map(|&(_, v)| v);
        let mut value = if rl.entry_block == 0 { initial() } else { None };
        let mut work: Vec<_> = p.blocks[rl.entry_block]
            .preds
            .iter()
            .copied()
            .filter(|&b| !internal(b))
            .collect();
        let mut seen = HashSet::new();
        while let Some(b) = work.pop() {
            if !seen.insert(b) {
                continue;
            }
            let bb = &p.blocks[b];
            let def = bb
                .insts
                .iter()
                .rev()
                .find_map(|i| i.write(reg))
                .or_else(|| bb.phis.iter().find(|ph| ph.reg == reg).map(|ph| ph.value));
            if let Some(v) = def.or_else(|| (b == 0).then(initial).flatten()) {
                if value.replace(v).is_some_and(|other| other != v) {
                    return None;
                }
            } else {
                work.extend(&bb.preds);
            }
        }
        value
    };
    let lo = before(Reg::S(EXEC))?;
    let hi = if p.wave64 {
        Some(before(Reg::S(EXEC + 1))?)
    } else {
        None
    };
    st.mask_bits(lo, hi)
}

/// A query and its optional opaque pre-pass. Never shared between lo and hi.
#[derive(Clone, Copy)]
struct Queries {
    rq: RayQuery,
    opaque: Option<RayQuery>,
    flags: u32,
    opaque_flags: u32,
}

#[derive(Default)]
pub(super) struct RayState {
    groups: Vec<Vec<Queries>>,
    pending: HashMap<usize, PendingAnyHit>,
    pub(super) candidates: HashMap<Value, (usize, RayQuery, CandidateInput)>,
    pub(super) cache: lift::CandidateCache,
    // At a merge, the last trace may have been inactive or bypassed. On
    // those paths even CommittedStatus on the uninitialized handle is invalid.
    valid: HashMap<(usize, RayQuery), V>,
    valid_phis: Vec<(usize, RayQuery, V)>,
}

fn is_lds_memory_inst(i: &Inst) -> bool {
    (matches!(i.fields, Fields::Ds { .. })
        && !(i.name().starts_with("ds_swizzle") || i.name().contains("permute")))
        || matches!(
            i.fields,
            Fields::Buffer { lds: true, .. } | Fields::Flat { lds: true, .. }
        )
}

/// An external reader may observe a removed traversal store from another lane or
/// wave. Reject the whole candidate in that case. Read-only traversal only makes
/// outside stores dead if no remaining instruction can observe them.
fn dead_bvh_lds_writes(
    p: &Program,
    ray_entries: &HashMap<usize, usize>,
    ray_skipped: &HashSet<usize>,
    single_wave: bool,
) -> Result<Option<HashSet<Site>>, &'static str> {
    if ray_entries.is_empty() {
        return Ok(None);
    }
    let mut used = false;
    let mut writes = HashSet::new();
    let mut readers = HashMap::new();
    let mut dead_writes = HashSet::new();
    for (b, blk) in p.blocks.iter().enumerate() {
        let inside = ray_entries.contains_key(&b) || ray_skipped.contains(&b);
        for (i, inst) in blk
            .insts
            .iter()
            .enumerate()
            .filter(|(_, inst)| is_lds_memory_inst(inst))
        {
            if inside {
                // Global data share and ordered/counter DS operations carry state
                // beyond one group's local traversal stack.
                if matches!(inst.fields, Fields::Ds { gds: true, .. })
                    || !(inst.name().starts_with("ds_read")
                        || inst.name().starts_with("ds_write")
                        || inst.name() == "ds_wrxchg_rtn_b32")
                {
                    return Err(inst.name());
                }
                used = true;
                if !inst.name().starts_with("ds_read") {
                    writes.insert(b);
                }
            } else if inst.name().starts_with("ds_write")
                && inst.writes.is_empty()
                && !matches!(inst.fields, Fields::Ds { gds: true, .. })
            {
                dead_writes.insert((b, i));
            } else {
                readers.insert(b, inst.name());
            }
        }
    }
    if !writes.is_empty() && !readers.is_empty() {
        // With one physical wave, a reader necessarily completed before *all*
        // removed stores if it dominates them and cannot be reached from any
        // store. Multiple waves may progress independently.
        if !single_wave {
            return Err("shared LDS across waves");
        }
        let order = crate::ir::reverse_postorder(&p.blocks);
        let mut index = vec![usize::MAX; p.blocks.len()];
        for (i, &b) in order.iter().enumerate() {
            index[b] = i;
        }
        let dom = crate::ir::idoms(&p.blocks, &order, &index);
        let mut later = HashSet::new();
        let mut work: Vec<_> = writes.iter().copied().collect();
        while let Some(b) = work.pop() {
            if later.insert(b) {
                work.extend(p.blocks[b].exit.successors());
            }
        }
        for (&reader, &op) in &readers {
            if later.contains(&reader)
                || writes.iter().any(|&b| {
                    let mut at = b;
                    while at != reader && at != 0 {
                        at = dom[at];
                    }
                    at != reader
                })
            {
                return Err(op);
            }
        }
    }
    Ok((used && readers.is_empty()).then_some(dead_writes))
}

impl Lower<'_> {
    pub(super) fn allocate_queries(&mut self) {
        let loops = &self.lift.loops;
        let Some(last) = loops.iter().map(|r| r.rq_group).max() else {
            return;
        };
        for group in 0..=last {
            let members = || loops.iter().filter(|r| r.rq_group == group);
            let base = members()
                .map(|r| lift::effective_const_ray_flags(r).unwrap_or(0))
                .reduce(|a, b| a & b)
                .expect("dense query groups");
            let two =
                super::super::native_rt_two_pass() && members().all(lift::is_two_pass_eligible);
            let flags = if two {
                base | lift::ray_flag::CULL_OPAQUE
            } else {
                base
            };
            let opaque_flags = base | lift::ray_flag::CULL_NON_OPAQUE;
            let lanes = if self.twin.is_some() { 2 } else { 1 };
            self.rays.groups.push(
                (0..lanes)
                    .map(|_| Queries {
                        rq: RayQuery::allocate(&mut self.s, flags),
                        opaque: two.then(|| RayQuery::allocate(&mut self.s, opaque_flags)),
                        flags,
                        opaque_flags,
                    })
                    .collect(),
            );
        }
    }

    pub(super) fn lift_ray_loop(&mut self, idx: usize) -> Result<(), Stop> {
        let rl = &self.lift.loops[idx];
        let exit = self.block[rl.exit_block].expect("lifted loop exit");
        if self.twin.is_some() {
            // A complete native traversal per virtual lane. The first trace
            // flows into the second even if it misses or its EXEC bit is clear.
            let hi = self.s.m.new_block();
            self.lift_lane(idx, hi)?;
            self.cur = hi;
            self.swap_lanes();
            let result = self.lift_lane(idx, exit);
            self.swap_lanes();
            self.tail[rl.exit_pred] = Some(result?);
        } else {
            self.tail[rl.exit_pred] = Some(self.lift_lane(idx, exit)?);
        }
        Ok(())
    }

    fn lift_lane(&mut self, idx: usize, exit: usize) -> Result<usize, Stop> {
        let rl = &self.lift.loops[idx];
        let offset = self.x.p.blocks[rl.entry_block].offset;
        let q = self.rays.groups[rl.rq_group][usize::from(self.upper)];
        let resolve = |this: &mut Self, input: RayInput| match input {
            RayInput::Const(k) => Ok(this.s.m.i32(k)),
            RayInput::Val(v) | RayInput::ShiftedMask(v) => {
                this.i32_of(v).map_err(|w| unsupported(offset, w))
            }
        };
        let tlas_src = Source::Loaded {
            load: rl.tlas_load,
            first: 0,
        };
        let tlas = self
            .handle(rl.tlas_load, &tlas_src, false, (16, 0))
            .map_err(|w| unsupported(offset, w))?;
        let incoming = entry_exec(self.x.p, self.x.st, rl)
            .ok_or_else(|| unsupported(offset, "unresolved RayQuery entry EXEC"))?;
        let sparse = rl
            .active_ray_mask
            .map(|value| {
                self.x
                    .st
                    .masks
                    .get(&value)
                    .copied()
                    .ok_or_else(|| unsupported(offset, "unresolved active RayQuery lane mask"))
            })
            .transpose()?;
        let mut active_ray = self.partial_active;
        for mask in [Some(incoming), sparse].into_iter().flatten() {
            if mask != Bool::TRUE {
                let lane = self
                    .boolean(mask, self.cur)
                    .map_err(|w| unsupported(offset, w))?;
                active_ray = Some(match active_ray {
                    Some(previous) => self.s.m.bin(self.cur, Bin::And, previous, lane),
                    None => lane,
                });
            }
        }
        let origin = [
            resolve(self, rl.origin[0])?,
            resolve(self, rl.origin[1])?,
            resolve(self, rl.origin[2])?,
        ];
        let t_min = rl.t_min.map(|i| resolve(self, i)).transpose()?;
        let dir = [
            resolve(self, rl.dir[0])?,
            resolve(self, rl.dir[1])?,
            resolve(self, rl.dir[2])?,
        ];
        let t_max = resolve(self, rl.t_max)?;
        let mut dynamic = |input| match input {
            RayInput::Const(_) => Ok(None),
            other => resolve(self, other).map(Some),
        };
        let instance_mask = dynamic(rl.instance_mask)?;
        let ray_flags = dynamic(rl.ray_flags)?;
        let init_min_front_t = match rl.kind {
            RayLoopKind::FrontBackTrack { init_min_front_t } => {
                Some(resolve(self, init_min_front_t)?)
            }
            _ => None,
        };
        let mut anyhit_inits = Vec::new();
        let mut anyhit_entry_dxil = None;
        let mut fallbacks = HashMap::new();
        if let RayLoopKind::AnyHit(ah) = &rl.kind {
            for sv in &ah.state_vars {
                anyhit_inits.push(resolve(self, sv.init)?);
            }
            anyhit_entry_dxil = Some(self.block[ah.entry_block].expect("anyhit entry"));
            for &(_, ci) in &ah.inputs {
                if let CandidateInput::CopyOf(pre) = ci {
                    fallbacks.insert(pre, self.i32_of(pre).map_err(|w| unsupported(offset, w))?);
                }
            }
        }
        for out in &rl.outputs {
            let copy = match out.output {
                RayOutput::CopyOf(pre) => Some(pre),
                _ => None,
            };
            for v in out.fallback.into_iter().chain(copy) {
                if let std::collections::hash_map::Entry::Vacant(entry) = fallbacks.entry(v) {
                    entry.insert(self.i32_of(v).map_err(|w| unsupported(offset, w))?);
                }
            }
        }
        let inputs = ResolvedRayInputs {
            rq: q.rq,
            rq_opaque: q.opaque,
            tlas,
            active_ray,
            origin,
            t_min,
            dir,
            t_max,
            instance_mask,
            ray_flags,
            group_const_flags: q.flags,
            group_opaque_const_flags: q.opaque_flags,
            init_min_front_t,
            anyhit_inits,
            anyhit_entry_dxil,
            fallbacks,
        };

        let emitted = lift::emit_ray_loop(&mut self.s, self.cur, exit, rl, &inputs);
        for (val, dxil_v) in emitted.outs {
            self.vals.insert(val, Val::I32(dxil_v));
        }
        if let RayLoopKind::AnyHit(ah) = &rl.kind
            && let Some(pending) = emitted.pending_anyhit
        {
            let rq = q.rq;
            for &(val_id, ci) in &ah.inputs {
                if matches!(
                    ci,
                    CandidateInput::RayT | CandidateInput::Bary(_) | CandidateInput::FrontFace
                ) {
                    self.rays
                        .candidates
                        .insert(val_id, (pending.anyhit_bridge_blk, rq, ci));
                }
            }
            for &(atom_v, const_b) in &ah.const_lane_atoms {
                let bv = self.s.m.bool(const_b);
                self.lane_bits.insert(atom_v, bv);
            }
            self.tail[ah.opacity_block] = Some(pending.anyhit_bridge_blk);
            self.rays.pending.insert(idx, pending);
        }
        let ran = active_ray.unwrap_or_else(|| self.s.m.bool(true));
        self.rays.valid.insert((emitted.tail, q.rq), ran);
        Ok(emitted.tail)
    }

    pub(super) fn finish_anyhit(&mut self, idx: usize) -> Result<(), Stop> {
        let RayLoopKind::AnyHit(ah) = &self.lift.loops[idx].kind else {
            unreachable!()
        };
        let offset = self.x.p.blocks[ah.tail_block].offset;
        let pending = self.rays.pending.remove(&idx).expect("pending AnyHit");
        let mut accept = match self.x.st.masks.get(&ah.accept_mask).copied() {
            Some(bm) => self
                .boolean(bm, self.cur)
                .map_err(|w| unsupported(offset, w))?,
            None => self
                .lane_bits
                .get(&ah.accept_mask)
                .copied()
                .ok_or_else(|| unsupported(offset, "any-hit accept mask"))?,
        };
        if ah.accept_invert {
            let t = self.s.m.bool(true);
            accept = self.s.m.bin(self.cur, Bin::Xor, accept, t);
        }
        let mut next_vals = Vec::with_capacity(ah.state_vars.len());
        for sv in &ah.state_vars {
            next_vals.push(self.i32_of(sv.next).map_err(|w| unsupported(offset, w))?);
        }
        let tail_blk = self.cur;
        self.s.m.terminate(
            tail_blk,
            Term::CondBr(accept, pending.do_commit_blk, pending.latch_blk),
        );
        for k in 0..ah.state_vars.len() {
            let back_val = if let Some(cb) = pending.cand_check_blk {
                self.s.m.set_phi(
                    pending.latch_phis[k],
                    vec![
                        (pending.state_phis[k], cb),
                        (next_vals[k], tail_blk),
                        (next_vals[k], pending.do_commit_blk),
                    ],
                );
                pending.latch_phis[k]
            } else {
                next_vals[k]
            };
            self.s.m.set_phi(
                pending.state_phis[k],
                vec![
                    (pending.init_vals[k], pending.start_blk),
                    (back_val, pending.latch_blk),
                ],
            );
        }
        Ok(())
    }
    fn query_valid(&mut self, block: usize, rq: RayQuery) -> V {
        if block == 0 {
            return self.s.m.bool(false);
        }
        *self.rays.valid.entry((block, rq)).or_insert_with(|| {
            let phi = self.s.m.phi(block, self.s.t.i1);
            self.rays.valid_phis.push((block, rq, phi));
            phi
        })
    }

    pub(super) fn finish_query_validity(&mut self) {
        if self.rays.valid_phis.is_empty() {
            return;
        }
        let preds = self.s.m.predecessors();
        while let Some((b, rq, phi)) = self.rays.valid_phis.pop() {
            let incoming = preds[b]
                .iter()
                .map(|&p| (self.query_valid(p, rq), p))
                .collect();
            self.s.m.set_phi(phi, incoming);
        }
    }

    pub(super) fn ray_metadata(
        &mut self,
        b: usize,
        i: usize,
        meta: RayMetaLoad,
    ) -> Result<(), Stop> {
        let inst = &self.x.p.blocks[b].insts[i];
        let guard = self
            .memory_guard(b, i)
            .map_err(|w| unsupported(inst.machine.offset, w))?;
        let q = self.rays.groups[meta.loop_idx][usize::from(self.upper)];
        let n = inst.writes.len();
        // Scalar DXR state queries replace exactly one guest dword. Matrix
        // loads may span rows but cannot cross the 3x4 transform into the ID.
        let width_ok = match meta.kind {
            RayMetaKind::BlasBaseZero => true,
            RayMetaKind::ObjectToWorld(offset) => {
                offset.is_multiple_of(4) && (offset as usize / 4).saturating_add(n) <= 12
            }
            _ => n == 1,
        };
        if !width_ok {
            return Err(unsupported(
                inst.machine.offset,
                "ambiguous RayQuery metadata width",
            ));
        }
        let zero = self.s.m.i32(0);
        let vals = if meta.kind == RayMetaKind::BlasBaseZero {
            vec![zero; n]
        } else {
            let done = self.s.m.new_block();
            let mut edges = Vec::new();
            let can_read = if meta.candidate {
                guard
            } else {
                let valid = self.query_valid(self.cur, q.rq);
                Some(if let Some(g) = guard {
                    self.s.m.bin(self.cur, Bin::And, valid, g)
                } else {
                    valid
                })
            };
            if let Some(g) = can_read {
                let active = self.s.m.new_block();
                self.s.m.terminate(self.cur, Term::CondBr(g, active, done));
                edges.push((vec![zero; n], self.cur));
                self.cur = active;
            }
            if meta.candidate {
                let words =
                    lift::emit_ray_meta_load(&mut self.s, self.cur, q.rq, true, meta.kind, n);
                edges.push((words, self.cur));
            } else {
                for rq in std::iter::once(q.rq).chain(q.opaque) {
                    let hit = self.s.m.new_block();
                    let next = self.s.m.new_block();
                    let status = rq.status(&mut self.s, self.cur);
                    let one = self.s.m.i32(1);
                    let tri = self.s.m.cmp(self.cur, pred::EQ, status, one);
                    self.s.m.terminate(self.cur, Term::CondBr(tri, hit, next));
                    let words = lift::emit_ray_meta_load(&mut self.s, hit, rq, false, meta.kind, n);
                    self.s.m.terminate(hit, Term::Br(done));
                    edges.push((words, hit));
                    self.cur = next;
                }
                edges.push((vec![zero; n], self.cur));
            }
            self.s.m.terminate(self.cur, Term::Br(done));
            self.cur = done;
            (0..n)
                .map(|k| {
                    let phi = self.s.m.phi(done, self.s.t.i32);
                    self.s
                        .m
                        .set_phi(phi, edges.iter().map(|(v, at)| (v[k], *at)).collect());
                    phi
                })
                .collect()
        };
        for (&(reg, _), x) in inst.writes.iter().zip(vals) {
            self.write(inst, reg, x, guard);
        }
        Ok(())
    }
}

#[cfg(all(test, windows))]
#[path = "rayquery_paired_tests.rs"]
mod paired_tests;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn live_lds_read_preserves_pretrace_stack_writes() {
        // Two IR blocks: preheader stack sentinel and a traversal stack pop.
        let mut words = vec![
            0xd834_0000u32,
            0x100,
            0xbf82_0000,
            0xd8d8_0000,
            0x0200_0000,
            0xbf81_0000,
        ];
        let make = |words: &[u32]| {
            let code = words
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<_>>();
            crate::ir::build(&code, false, 8).unwrap()
        };
        let entries = HashMap::from([(1, 0)]);
        let skipped = HashSet::new();
        assert_eq!(
            dead_bvh_lds_writes(&make(&words), &entries, &skipped, false),
            Ok(Some(HashSet::from([(0, 0)])))
        );
        words.splice(2..2, [0xd8d8_0000, 0x0200_0000]); // add a read outside the lifted region
        assert_eq!(
            dead_bvh_lds_writes(&make(&words), &entries, &skipped, false),
            Ok(None)
        );
        // The same externally visible read also observes a traversal store:
        // software must execute the region, even if the value is read by another lane.
        words.splice(5..5, [0xd834_0000, 0x100]);
        assert_eq!(
            dead_bvh_lds_writes(&make(&words), &entries, &skipped, false),
            Err("shared LDS across waves")
        );
        assert_eq!(
            dead_bvh_lds_writes(&make(&words), &entries, &skipped, true),
            Ok(None)
        );
        // A read after the store remains observable even in one physical wave.
        words.splice(
            words.len() - 1..words.len() - 1,
            [0xbf82_0000, 0xd8d8_0000, 0x0200_0000],
        );
        assert_eq!(
            dead_bvh_lds_writes(&make(&words), &entries, &skipped, true),
            Err("ds_read_b32")
        );
    }
}
