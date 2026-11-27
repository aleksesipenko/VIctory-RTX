//! What goes up to copies in the card's memory goes up as it was when its
//! submission was prepared (`VICTORY_SEND_OWN`, on by default; `0`: as
//! before).
//!
//! The sends were copies out of the copies' CPU sides, which the card makes
//! when it comes to the submission. By then later submissions are prepared
//! (two frames of them with `VICTORY_RELAY`), and their refreshes have
//! written the same CPU sides: the earlier submission's shaders then read a
//! later one's bytes. For what the title's CPU writes that is as on the
//! console (it does not write what the card still reads). But the title's
//! own buffer copies are commands in its lists, and those are done by us
//! when a submission is prepared: "a value to a buffer, a draw that reads
//! it" in one submission and the same with another value in the next is
//! ordinary, and both draws read the second value.
//!
//! So each submission's bytes are taken out of the CPU sides into memory of
//! the submission's own when they are taken from the dirty lists, and the
//! card copies them from there. Runs of a few words, which a refresh that
//! sends only what changed makes many of, go as pairs of offset and value,
//! stored by one dispatch per copy; the others as a copy each.

use super::*;
use crate::shadow::Shadow;

pub(super) fn own() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_SEND_OWN").is_none_or(|v| v != "0"))
}

/// Runs up to this many bytes go as pairs.
const SHORT: usize = 256;

/// Whether a run of `len` bytes at offset `at` of a copy goes as pairs.
fn short(at: usize, len: usize) -> bool {
    len <= SHORT && at.is_multiple_of(4) && len.is_multiple_of(4)
}

/// Whether a run of `len` bytes at offset `at` of a copy in the card's memory
/// is no copy command of its own, but pairs of one dispatch.
pub(crate) fn goes_as_words(at: usize, len: usize) -> bool {
    own() && short(at, len)
}

/// Whether the title's bytes that go in under flights' writes
/// (`Shadow::merge_under_writes`) go in by whole words
/// (`VICTORY_UNDER_WORDS=0`: by the bytes that changed, as before).
///
/// By bytes nearly every run (a float's three low bytes) was a copy command
/// of its own, and a submission with more than 4,096 of them stood until the
/// flights had landed: 4 to 14 ms of a steady frame (session play-1125-0203:
/// "more changed runs than a submission sends up"), for some 20 KB. Whole
/// words are pairs of one dispatch, and are not counted.
///
/// A word is taken to be the title's or the shaders', not shared: where the
/// title changed a byte of a word of which a flight's shader writes another
/// byte, that byte is as it was before the flight. The landing rests on the
/// same ([`crate::shadow::exact_back`]).
pub(crate) fn words_under() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| own() && std::env::var_os("VICTORY_UNDER_WORDS").is_none_or(|v| v != "0"))
}

const SCATTER: &str = r#"
#define RS "RootConstants(num32BitConstants=2, b0), SRV(t0), UAV(u0)"
cbuffer Part : register(b0) { uint at; uint count; };
ByteAddressBuffer pairs : register(t0);
RWByteAddressBuffer card : register(u0);
[RootSignature(RS)]
[numthreads(64, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    if (id.x >= count) return;
    uint2 p = pairs.Load2(at + id.x * 8u);
    card.Store(p.x, p.y);
}
"#;

struct Pass {
    root: ID3D12RootSignature,
    pso: ID3D12PipelineState,
}

fn pass(host: &Host) -> Option<&'static Pass> {
    static PASS: OnceLock<Option<Pass>> = OnceLock::new();
    PASS.get_or_init(|| {
        let code = compile(SCATTER, windows::core::s!("main"), windows::core::s!("cs_5_1")).ok()?;
        // SAFETY: the code lives through the calls; the root signature is
        // inside it.
        unsafe {
            let root: ID3D12RootSignature = host.device.CreateRootSignature(0, &code).ok()?;
            let pso: ID3D12PipelineState = host
                .device
                .CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
                    pRootSignature: std::mem::transmute_copy(&root),
                    CS: D3D12_SHADER_BYTECODE {
                        pShaderBytecode: code.as_ptr().cast(),
                        BytecodeLength: code.len(),
                    },
                    ..Default::default()
                })
                .ok()?;
            name(&pso, || "victory: changed words sent to the card".into());
            Some(Pass { root, pso })
        }
    })
    .as_ref()
}

/// Memory a submission's sends come from, mapped.
pub(super) struct Memory {
    buffer: ID3D12Resource,
    data: usize,
    size: u64,
}

// SAFETY: a D3D12 object (free-threaded) and its mapping, with one owner.
unsafe impl Send for Memory {}

impl Memory {
    pub(super) fn resource(&self) -> &ID3D12Resource {
        &self.buffer
    }
}

/// Pool of spare submission upload buffers, bounded by total retained bytes.
struct SparePool<T> {
    items: Vec<T>,
}

impl<T> SparePool<T> {
    const fn new() -> Self {
        Self { items: Vec::new() }
    }

    fn take(&mut self, min_size: u64, size_of: impl Fn(&T) -> u64) -> Option<T> {
        let fit = self
            .items
            .iter()
            .enumerate()
            .filter(|(_, m)| size_of(m) >= min_size)
            .min_by_key(|(_, m)| size_of(m))
            .map(|(i, _)| i);
        fit.map(|i| self.items.swap_remove(i))
    }

    /// Returns `item` to the pool only when `reusable` is true; evicted items
    /// over `cap` (or an unconfirmed detached `item`) are returned to drop
    /// outside the pool lock.
    fn finish(&mut self, item: T, reusable: bool, cap: u64, size_of: impl Fn(&T) -> u64) -> Vec<T> {
        if !reusable {
            return vec![item];
        }
        self.items.push(item);
        self.items.sort_by_key(|m| std::cmp::Reverse(size_of(m)));
        let mut evicted = Vec::new();
        while self.items.iter().map(&size_of).sum::<u64>() > cap {
            evicted.push(self.items.remove(0));
        }
        evicted
    }
}

/// What submissions that have run gave back.
static SPARE: Mutex<SparePool<Memory>> = Mutex::new(SparePool::new());

/// At most this much is kept (of the card's memory, with [`on_card`]).
fn kept() -> u64 {
    if on_card() {
        256 << 20
    } else if crate::residency::limited_host_ram() {
        192 << 20
    } else {
        512 << 20
    }
}

/// Whether the memory lies in the card's memory, written by the CPU through
/// the bus (`VICTORY_SEND_BAR=1`): the copies on the card's drawing queue, which
/// nothing overlaps, then cross no bus. In a captured frame they took 5.4 ms of
/// the card's 30 (342 copies), and take 0.4 this way; but filling it takes the
/// submitting thread 8-9 ms a frame instead of 3 (73-93 MB, run card-3 against
/// card-1), so it is a loss while the card waits for that thread: off by
/// default.
fn on_card() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_SEND_BAR").is_some_and(|v| v == "1"))
}

fn memory(host: &Host, size: u64) -> Option<Memory> {
    {
        let mut spare = SPARE.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(m) = spare.take(size, |m| m.size) {
            return Some(m);
        }
    }
    // By doubles, so that the next of about this size finds it: what
    // submissions send differs from frame to frame.
    let size = size.max(8 << 20).next_power_of_two();
    let mapped = if on_card() { host.mapped_card_buffer(size, false) } else { None };
    let (buffer, data) = match mapped {
        Some(made) => made,
        None => host.upload_buffer(size)?,
    };
    name(&buffer, || "bytes a submission sends up".into());
    Some(Memory {
        buffer,
        data: data as usize,
        size,
    })
}

/// The submission has run. Returns `memory` to [`SPARE`] only when `reusable`
/// is true; when `reusable` is false (an unconfirmed copy-queue submission may
/// still be reading `memory`), detaches it from [`SPARE`] so no subsequent CPU
/// write can overwrite it (`upq` holds the COM resource alive until the
/// copy-queue fence confirms or leaks it if `Signal` failed).
pub(super) fn finish_memory(memory: Memory, reusable: bool) {
    if !reusable {
        drop(memory);
        return;
    }
    let evicted = {
        let mut spare = SPARE.lock().unwrap_or_else(|e| e.into_inner());
        spare.finish(memory, true, kept(), |m| m.size)
    };
    drop(evicted);
}

/// Drops all idle submission upload buffers outside the pool lock (under host RAM pressure).
pub(super) fn trim() {
    let dropped = std::mem::take(&mut SPARE.lock().unwrap_or_else(|e| e.into_inner()).items);
    drop(dropped);
}

/// Since the last frame-rate line: bytes copied, runs copied, words stored,
/// microseconds of taking them out of the CPU sides.
pub(super) static STATS: [std::sync::atomic::AtomicU64; 4] = [const { std::sync::atomic::AtomicU64::new(0) }; 4];

/// One copy's sends: ranges copied (offset in the copy, in the memory,
/// bytes), and its pairs (where in the memory, how many).
struct Copy {
    runs: Vec<(usize, u64, usize)>,
    pairs: (u64, u32),
}

/// Where a submission's bytes are taken to.
pub(super) struct Own {
    memory: Memory,
    copies: Vec<Copy>,
    /// Bytes of memory used.
    used: u64,
}

/// Runs whose pairs are one piece of work.
const RUNS_AT_A_TIME: usize = 4096;

/// Takes what `dirty` sends up out of the CPU sides.
pub(super) fn take(host: &Host, dirty: &[(std::sync::Arc<Shadow>, Vec<(usize, usize)>)]) -> Option<Own> {
    let began = std::time::Instant::now();
    pass(host)?;
    // Runs of whole words of the copy: only those are pairs. (A byte beside
    // them in a word may be the card's.)
    let short = |&(a, b): &(usize, usize)| short(a, b - a);
    let mut at = 0u64;
    let mut copies = Vec::with_capacity(dirty.len());
    for (_, ranges) in dirty {
        let mut runs = Vec::new();
        for &(a, b) in ranges.iter().filter(|r| !short(r)) {
            runs.push((a, at, b - a));
            at += ((b - a) as u64).next_multiple_of(16);
        }
        let words: usize = ranges.iter().filter(|r| short(r)).map(|r| (r.1 - r.0) / 4).sum();
        copies.push(Copy { runs, pairs: (at, words as u32) });
        at += (words as u64 * 8).next_multiple_of(16);
    }
    if at == 0 {
        return None;
    }
    let memory = memory(host, at)?;
    let mut moves: Vec<(usize, usize, usize)> = Vec::new();
    let mut work: Vec<(usize, &[(usize, usize)], usize)> = Vec::new();
    for ((shadow, ranges), copy) in dirty.iter().zip(&copies) {
        let from = shadow.card_data() as usize;
        moves.extend(copy.runs.iter().map(|&(a, to, n)| (from + a, memory.data + to as usize, n)));
        // (80,000 runs of some words a frame: in pieces of work.)
        let mut to = memory.data + copy.pairs.0 as usize;
        for ranges in ranges.chunks(RUNS_AT_A_TIME) {
            let words: usize = ranges.iter().filter(|r| short(r)).map(|r| (r.1 - r.0) / 4).sum();
            if words > 0 {
                work.push((from, ranges, to));
                to += words * 8;
            }
        }
    }
    let one = |i: usize| {
        let (from, ranges, to) = work[i];
        let mut to = to as *mut [u32; 2];
        for &(a, b) in ranges.iter().filter(|r| short(r)) {
            for offset in (a..b).step_by(4) {
                // SAFETY: inside the copy's mapping (its CPU side, which
                // the thread that asked, and waits, alone writes in these
                // ranges) and inside the memory, which has room for every
                // pair, a piece of work's behind the last one's.
                unsafe {
                    let word = ((from + offset) as *const u32).read_unaligned();
                    to.write([offset as u32, word]);
                    to = to.add(1);
                }
            }
        }
    };
    crate::land::each_under(work.len(), &one);
    STATS[0].fetch_add(moves.iter().map(|m| m.2 as u64).sum(), Ordering::Relaxed);
    STATS[1].fetch_add(moves.len() as u64, Ordering::Relaxed);
    STATS[2].fetch_add(copies.iter().map(|c| u64::from(c.pairs.1)).sum(), Ordering::Relaxed);
    crate::shadow::copy_all(moves);
    STATS[3].fetch_add(began.elapsed().as_micros() as u64, Ordering::Relaxed);
    Some(Own { memory, copies, used: at })
}

impl Own {
    /// Records the sends into `list`; the copies that `looked` says are in
    /// `COPY_DEST`, the others in the common state. Those with a "before" are
    /// left readable by compute shaders.
    ///
    /// # Safety
    /// An open list of ours, whose compute state is set anew behind this.
    pub(super) unsafe fn record(
        &self,
        host: &Host,
        list: &ID3D12GraphicsCommandList,
        dirty: &[(std::sync::Arc<Shadow>, Vec<(usize, usize)>)],
        looked: impl Fn(&Shadow) -> bool,
        from: Option<&ID3D12Resource>,
    ) {
        // (With `crate::host::upq`: a copy of the memory in the card's.)
        let source = from.unwrap_or(&self.memory.buffer);
        let read = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
        let write = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
        let dest = D3D12_RESOURCE_STATE_COPY_DEST;
        let (mut first, mut last) = (Vec::new(), Vec::new());
        for ((shadow, _), copy) in dirty.iter().zip(&self.copies) {
            for &(a, from, n) in &copy.runs {
                // SAFETY: the caller's; live buffers, ranges inside them.
                unsafe { list.CopyBufferRegion(&shadow.buffer, a as u64, source, from, n as u64) };
            }
            let mut state = (looked(shadow) || !copy.runs.is_empty()).then_some(dest);
            if copy.pairs.1 > 0 {
                if let Some(was) = state {
                    first.push(transition(&shadow.buffer, was, write));
                }
                state = Some(write);
            }
            let hashed = shadow.card_sides().is_some_and(|(_, before, _)| before.is_some());
            if let Some(was) = state
                && hashed
                && (!copy.runs.is_empty() || copy.pairs.1 > 0)
            {
                last.push(transition(&shadow.buffer, was, read));
            }
        }
        // SAFETY: the caller's; live objects.
        unsafe {
            if !first.is_empty() {
                list.ResourceBarrier(&first);
            }
            if let Some(pass) = pass(host)
                && self.copies.iter().any(|c| c.pairs.1 > 0)
            {
                list.SetComputeRootSignature(&pass.root);
                list.SetPipelineState(&pass.pso);
                list.SetComputeRootShaderResourceView(1, source.GetGPUVirtualAddress());
                for ((shadow, _), copy) in dirty.iter().zip(&self.copies).filter(|(_, c)| c.pairs.1 > 0) {
                    list.SetComputeRootUnorderedAccessView(2, shadow.buffer.GetGPUVirtualAddress());
                    const MOST: u32 = 64 * D3D12_CS_DISPATCH_MAX_THREAD_GROUPS_PER_DIMENSION;
                    let (at, words) = copy.pairs;
                    let mut done = 0;
                    while done < words {
                        let n = (words - done).min(MOST);
                        let constants = [at as u32 + done * 8, n];
                        list.SetComputeRoot32BitConstants(0, 2, constants.as_ptr().cast(), 0);
                        list.Dispatch(n.div_ceil(64), 1, 1);
                        done += n;
                    }
                }
            }
            if !last.is_empty() {
                list.ResourceBarrier(&last);
            }
        }
    }

    /// The memory and the bytes of it used.
    pub(super) fn source(&self) -> (&ID3D12Resource, u64) {
        (&self.memory.buffer, self.used)
    }

    /// The memory, to be kept until the submission has run.
    pub(super) fn memory(self) -> Memory {
        self.memory
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    #[test]
    fn unconfirmed_copy_detaches_memory_from_spare_pool_without_reuse() {
        struct FakeMem {
            res: Arc<()>,
            size: u64,
        }

        let mut pool: SparePool<FakeMem> = SparePool::new();
        let cap = 64 << 20;

        // Confirmed submission returns memory to the spare pool for reuse.
        let ok_res = Arc::new(());
        let dropped = pool.finish(FakeMem { res: Arc::clone(&ok_res), size: 8 << 20 }, true, cap, |m| m.size);
        assert!(dropped.is_empty());
        let reused = pool.take(4 << 20, |m| m.size).expect("confirmed memory is reused");
        assert!(Arc::ptr_eq(&reused.res, &ok_res));

        // Unconfirmed copy-queue submission detaches memory from SPARE so future
        // CPU writes allocate fresh memory instead of overwriting it, while an
        // external copy-queue pin keeps the underlying resource alive until
        // copy fence confirmation (or permanently if unsignaled).
        let unconfirmed_res = Arc::new(());
        let copy_queue_pin = Arc::clone(&unconfirmed_res);
        let detached = pool.finish(
            FakeMem { res: Arc::clone(&unconfirmed_res), size: 8 << 20 },
            false,
            cap,
            |m| m.size,
        );
        assert_eq!(detached.len(), 1);
        assert!(pool.take(1, |m| m.size).is_none(), "unconfirmed memory must never re-enter SPARE");
        drop(detached);
        // Underlying resource remains alive while copy-queue pin is held.
        assert_eq!(Arc::strong_count(&unconfirmed_res), 2);
        drop(copy_queue_pin);
        // Released cleanly once copy-queue fence confirms and drops its pin.
        assert_eq!(Arc::strong_count(&unconfirmed_res), 1);
    }
}
