//! SSE4a's `extrq` and `insertq` with immediate bit-field operands, JIT-replaced
//! with SSE2 stubs on Intel CPUs (GPL-3.0 provenance: LAKSHYA-cmd-tech/VIctory #87 /
//! upstream #3042).
//!
//! Intel CPUs lack AMD SSE4a, so each `extrq`/`insertq` traps `#UD` (~1 us kernel
//! round-trip). On the first fault at a site, if the 5-byte `jmp rel32` fits
//! within 16 aligned bytes, we atomically patch it via `cmpxchg16b` to a nearby
//! SSE2 stub (`crate::reach`), eliminating subsequent `#UD` faults. Register-form
//! operands and sites crossing a 16-byte boundary remain emulated.
//! `VICTORY_PATCH_SSE4A=0` disables patching.

use std::sync::{Mutex, OnceLock};

use crate::reach::{STUBS_LEN, jump, near};

fn enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_PATCH_SSE4A").is_none_or(|v| v != "0"))
}

/// `extrq xmm, len, idx` (`66 [rex] 0f 78 /0 ib ib`, `dst` the r/m register) or
/// `insertq dst, src, len, idx` (`f2 [rex] 0f 78 /r ib ib`).
#[derive(Clone, Copy, Debug)]
struct Field {
    insert: bool,
    dst: u8,
    src: u8,
    /// Bits in the field, 1 to 64, and where it starts.
    len: u32,
    idx: u32,
    /// Bytes in the instruction.
    size: usize,
}

fn decode(code: &[u8]) -> Option<Field> {
    let prefix = *code.first()?;
    let (rex, at) = match *code.get(1)? {
        r @ 0x40..=0x4f => (r, 2),
        _ => (0, 1),
    };
    if !matches!(prefix, 0x66 | 0xf2) || code.get(at..at + 2)? != [0x0f, 0x78] {
        return None;
    }
    let &[modrm, len, idx] = code.get(at + 2..at + 5)? else {
        return None;
    };
    if modrm >> 6 != 3 {
        return None;
    }
    let reg = (modrm >> 3) & 7 | (rex & 4) << 1;
    let rm = modrm & 7 | (rex & 1) << 3;
    let insert = prefix == 0xf2;
    if !insert && reg != 0 {
        return None;
    }
    let len = match u32::from(len & 63) {
        0 => 64,
        n => n,
    };
    let (dst, src) = if insert { (reg, rm) } else { (rm, rm) };
    Some(Field {
        insert,
        dst,
        src,
        len,
        idx: u32::from(idx & 63),
        size: at + 5,
    })
}

/// The instruction `f` at `at` as SSE2 code placed at `place`, entered 16 bytes
/// in (the first 16 hold the mask `insertq` keeps for its `pand`), then a jump
/// back past it. Only the destination changes, its high quadword zeroed as Zen 2
/// and the emulator do; `insertq` borrows a third XMM register saved on the stack.
fn stub(f: Field, at: usize, place: usize) -> Option<Vec<u8>> {
    // `prefix [rex] 0f code` with registers `reg` and `rm`.
    fn op(prefix: u8, code: u8, reg: u8, rm: u8) -> Vec<u8> {
        let rex = (reg & 8) >> 1 | (rm & 8) >> 3;
        let mut v = vec![prefix];
        if rex != 0 {
            v.push(0x40 | rex);
        }
        v.extend([0x0f, code, 0xc0 | (reg & 7) << 3 | rm & 7]);
        v
    }
    // psrlq (`down`) or psllq of `x` by `by`, nothing for 0.
    fn shift(down: bool, x: u8, by: u32) -> Vec<u8> {
        if by == 0 {
            return Vec::new();
        }
        let mut v = op(0x66, 0x73, if down { 2 } else { 6 }, x);
        v.push(by as u8);
        v
    }
    let mut code = vec![0; 16];
    let x = f.dst;
    if !f.insert {
        code.extend(shift(true, x, f.idx));
        code.extend(shift(false, x, 64 - f.len));
        code.extend(shift(true, x, 64 - f.len));
        code.extend(op(0xf3, 0x7e, x, x)); // movq x, x (zeroes high qword)
    } else {
        let ones = if f.len == 64 { u64::MAX } else { (1 << f.len) - 1 };
        code[..8].copy_from_slice(&(!(ones << f.idx)).to_le_bytes());
        let t = (0..3).find(|&t| t != f.dst && t != f.src)?;
        code.extend([0x48, 0x8d, 0x64, 0x24, 0xf0]); // lea rsp, [rsp - 16]
        code.extend([0xf3, 0x0f, 0x7f, t << 3 | 4, 0x24]); // movdqu [rsp], t
        code.extend(op(0xf3, 0x7e, t, f.src)); // movq t, src
        code.extend(shift(false, t, 64 - f.len));
        code.extend(shift(true, t, 64 - f.len));
        code.extend(shift(false, t, f.idx));
        // pand dst, [rip + disp] (16-byte aligned mask at `place`, high 64 bits zero)
        code.extend(if x >= 8 { &[0x66, 0x44][..] } else { &[0x66] });
        let disp = -i32::try_from(code.len() + 7).ok()?;
        code.extend([0x0f, 0xdb, (x & 7) << 3 | 5]);
        code.extend(disp.to_le_bytes());
        code.extend(op(0x66, 0xeb, x, t)); // por dst, t
        code.extend([0xf3, 0x0f, 0x6f, t << 3 | 4, 0x24]); // movdqu t, [rsp]
        code.extend([0x48, 0x8d, 0x64, 0x24, 0x10]); // lea rsp, [rsp + 16]
    }
    let end = place + code.len();
    code.extend(jump(end, at + f.size)?);
    Some(code)
}

/// Reads up to `N` bytes starting at `at` (crossing adjacent readable regions
/// when needed) while holding [`crate::console_insn::lock_patching`] so Rust
/// never forms a non-atomic `&[u8]` over live self-patched code and no
/// concurrent `VirtualProtect`/`cmpxchg16b` races with the read.
pub(crate) fn read_code<const N: usize>(at: usize) -> Option<([u8; N], usize)> {
    let _patching = crate::console_insn::lock_patching();
    read_code_locked::<N>(at)
}

fn read_code_locked<const N: usize>(at: usize) -> Option<([u8; N], usize)> {
    let mut buf = [0u8; N];
    let got = read_code_into_locked(at, &mut buf);
    (got > 0).then_some((buf, got))
}

/// Reads up to `dst.len()` bytes starting at `at` via `ReadProcessMemory` while
/// holding [`crate::console_insn::lock_patching`], retrying across adjacent
/// readable memory regions until `dst` is full or an unreadable boundary is
/// reached. Returns the exact byte count read (`0` on failure).
pub(crate) fn read_code_into(at: usize, dst: &mut [u8]) -> usize {
    let _patching = crate::console_insn::lock_patching();
    read_code_into_locked(at, dst)
}

/// Reentrancy-safe variant of [`read_code_into`] for crash-time reporting and
/// `VICTORY_CRASH_SAVE` dumps: acquires [`crate::console_insn::PATCHING`]
/// unless already held on the crashing thread, then copies via
/// `ReadProcessMemory` without forming any Rust `&[u8]` over live memory.
pub(crate) fn read_code_into_reentrant(at: usize, dst: &mut [u8]) -> usize {
    let _patching = crate::console_insn::lock_patching_reentrant();
    read_code_into_locked(at, dst)
}

/// Caller must hold [`crate::console_insn::lock_patching`].
fn read_code_into_locked(at: usize, dst: &mut [u8]) -> usize {
    use windows_sys::Win32::System::Diagnostics::Debug::ReadProcessMemory;
    use windows_sys::Win32::System::Threading::GetCurrentProcess;

    if dst.is_empty() {
        return 0;
    }
    // SAFETY: pseudo-handle for current process.
    let process = unsafe { GetCurrentProcess() };
    let mut total = 0usize;
    while total < dst.len() {
        let Some(addr) = at.checked_add(total) else {
            break;
        };
        let remaining = dst.len() - total;
        let mut got = 0usize;
        // SAFETY: `dst[total..]` is a valid writable stack slice; the kernel
        // copies bytes without forming a Rust reference over live code memory.
        let ok = unsafe {
            ReadProcessMemory(
                process,
                addr as *const _,
                dst[total..].as_mut_ptr().cast(),
                remaining,
                &mut got,
            )
        };
        if got > 0 {
            total += got;
            if ok != 0 || total < dst.len() {
                continue;
            }
            break;
        }
        // If `ReadProcessMemory` returned 0 bytes on a cross-page span hitting
        // an unmapped page, capture any remaining readable bytes in the
        // current 4 KiB page before stopping.
        let page_left = 0x1000 - (addr & 0xfff);
        if page_left < remaining {
            let mut page_got = 0usize;
            let page_ok = unsafe {
                ReadProcessMemory(
                    process,
                    addr as *const _,
                    dst[total..].as_mut_ptr().cast(),
                    page_left,
                    &mut page_got,
                )
            };
            if page_got > 0 {
                total += page_got;
                if page_ok != 0 {
                    continue;
                }
            }
        }
        break;
    }
    total
}

/// Shared state protected by [`STUBS`] (strict global lock order:
/// `console_insn::PATCHING -> STUBS`).
struct Stubs {
    /// Active stub block `(base, used)`.
    active: Option<(usize, usize)>,
    /// All allocated stub arenas `(base, len)` kept live for emitted stubs.
    arenas: Vec<(usize, usize)>,
    /// Patched sites `(site_at, stub_entry)` installed by [`replace`].
    patched: Vec<(usize, usize)>,
}

impl Stubs {
    const fn new() -> Self {
        Self {
            active: None,
            arenas: Vec::new(),
            patched: Vec::new(),
        }
    }

    fn in_arena(&self, addr: usize) -> bool {
        self.arenas.iter().any(|&(base, len)| addr >= base && addr < base + len)
    }

    fn note_patched(&mut self, at: usize, target: usize) {
        if let Some(slot) = self.patched.iter_mut().find(|(a, _)| *a == at) {
            slot.1 = target;
        } else {
            self.patched.push((at, target));
        }
    }
}

static STUBS: Mutex<Stubs> = Mutex::new(Stubs::new());

/// Whether `at` is currently patched by [`replace`] with a live `jmp rel32`
/// whose destination matches its recorded SSE2 stub inside our stub arena.
///
/// Acquires `PATCHING -> STUBS` so any sibling thread faulting mid-patch waits
/// for `cmpxchg16b`, `VirtualProtect` restoration, and `FlushInstructionCache`
/// to finish, then validates the live 5-byte jump via `ReadProcessMemory`
/// (evicting stale entries if a freed code page was later reused).
pub(crate) fn was_patched(at: usize) -> bool {
    let _patching = crate::console_insn::lock_patching();
    let mut stubs = STUBS.lock().unwrap_or_else(|e| e.into_inner());
    let Some(pos) = stubs.patched.iter().position(|&(a, _)| a == at) else {
        return false;
    };
    let expected = stubs.patched[pos].1;
    let mut buf = [0u8; 5];
    if read_code_into_locked(at, &mut buf) != 5 || buf[0] != 0xe9 {
        stubs.patched.swap_remove(pos);
        return false;
    }
    let rel = i32::from_le_bytes([buf[1], buf[2], buf[3], buf[4]]);
    let dest = at.wrapping_add(5).wrapping_add_signed(rel as isize);
    if dest == expected && stubs.in_arena(dest) {
        true
    } else {
        stubs.patched.swap_remove(pos);
        false
    }
}

#[cfg(test)]
fn forget_patched(at: usize) {
    let _patching = crate::console_insn::lock_patching();
    let mut stubs = STUBS.lock().unwrap_or_else(|e| e.into_inner());
    stubs.patched.retain(|&(a, _)| a != at);
}

/// After an `#UD` at `at` was emulated: replaces it with a jump to its SSE2
/// [`stub`] if it is an immediate-form `extrq`/`insertq` and the 5-byte jump
/// fits within its aligned 16 bytes.
///
/// # Safety
/// `at` is in mapped executable code at an instruction that just faulted.
pub(crate) unsafe fn replace(at: usize) {
    if !enabled() || (at & 15) + 5 > 16 || !std::is_x86_feature_detected!("cmpxchg16b") {
        return;
    }
    let Some((buf, n)) = read_code::<7>(at) else {
        return;
    };
    if decode(&buf[..n]).is_none() {
        return;
    }
    // Strict lock order: `PATCHING` (shared with `console_insn`) -> `STUBS`.
    let _patching = crate::console_insn::lock_patching();
    let mut stubs = STUBS.lock().unwrap_or_else(|e| e.into_inner());
    // Re-read under `PATCHING` via `ReadProcessMemory`: another thread may have
    // already patched `at`, and no Rust `&[u8]` ever aliases live code bytes.
    let Some((buf, n)) = read_code_locked::<7>(at) else {
        return;
    };
    let Some(f) = decode(&buf[..n]) else {
        return;
    };
    let fits = |&(place, used): &(usize, usize)| {
        let body = stub(f, at, place + used)?;
        let jmp = jump(at, place + used + 16)?;
        (used + body.len().next_multiple_of(16) <= STUBS_LEN).then_some((body, jmp))
    };
    // A full block, or one out of +/-2 GiB reach, stays live for its stubs and
    // a new block is allocated near `at`.
    if stubs.active.as_ref().and_then(fits).is_none() {
        if let Some(place) = near(at) {
            stubs.arenas.push((place, STUBS_LEN));
            stubs.active = Some((place, 0));
        } else {
            stubs.active = None;
        }
    }
    let Some(((body, jmp), (place, used))) = stubs.active.as_mut().and_then(|s| Some((fits(&*s)?, s)))
    else {
        return;
    };
    let at_stub = *place + *used;
    let stub_entry = at_stub + 16;
    // SAFETY: fresh RWX stub memory not yet reachable from any thread.
    unsafe {
        std::ptr::copy_nonoverlapping(body.as_ptr(), at_stub as *mut u8, body.len());
        windows_sys::Win32::System::Diagnostics::Debug::FlushInstructionCache(
            windows_sys::Win32::System::Threading::GetCurrentProcess(),
            at_stub as *const _,
            body.len(),
        );
    }
    *used += body.len().next_multiple_of(16);
    let mut was = [0; 5];
    was.copy_from_slice(&buf[..5]);
    // SAFETY: `at` is mapped code, `STUBS` is held across both `patch` and
    // `note_patched`, and the stub preserves all registers/flags identically.
    if unsafe { patch(at, was, jmp) } {
        stubs.note_patched(at, stub_entry);
    }
}

/// Replaces the 5 bytes at `at` (which must equal `was`) with `jmp` in one
/// atomic 16-byte `cmpxchg16b` store. Returns `true` if `jmp` is now installed.
///
/// # Safety
/// `at` is in mapped code, the 5 bytes lie inside one aligned 16-byte word,
/// and the caller holds [`STUBS`].
unsafe fn patch(at: usize, was: [u8; 5], jmp: [u8; 5]) -> bool {
    use std::arch::x86_64::cmpxchg16b;
    use std::sync::atomic::Ordering::SeqCst;
    use windows_sys::Win32::System::Diagnostics::Debug::FlushInstructionCache;
    use windows_sys::Win32::System::Memory::{PAGE_EXECUTE_READWRITE, VirtualProtect};
    use windows_sys::Win32::System::Threading::GetCurrentProcess;

    let word = (at & !15) as *mut u128;
    let base = word as *const core::ffi::c_void;
    let here = (at & 15)..(at & 15) + 5;
    let mut old = 0;
    // SAFETY: the aligned 16 bytes lie in one page of mapped code; `cmpxchg16b`
    // was verified by the caller.
    unsafe {
        if VirtualProtect(base, 16, PAGE_EXECUTE_READWRITE, &mut old) == 0 {
            return false;
        }
        let mut now = cmpxchg16b(word, 0, 0, SeqCst, SeqCst);
        let done = loop {
            let mut bytes = now.to_le_bytes();
            if bytes[here.clone()] == jmp {
                break true;
            }
            if bytes[here.clone()] != was {
                break false;
            }
            bytes[here.clone()].copy_from_slice(&jmp);
            let there = cmpxchg16b(word, now, u128::from_le_bytes(bytes), SeqCst, SeqCst);
            if there == now {
                break true;
            }
            now = there;
        };
        VirtualProtect(base, 16, old, &mut old);
        FlushInstructionCache(GetCurrentProcess(), base, 16);
        done
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_EXECUTE_READWRITE, VirtualAlloc, VirtualFree,
    };

    fn mask(len: u32) -> u64 {
        if len == 64 { u64::MAX } else { (1 << len) - 1 }
    }

    #[test]
    fn an_extrq_or_insertq_becomes_sse2() {
        crate::crash::report_crashes();
        let (d_in, s_in) = (0x0123_4567_89ab_cdefu64, 0xfedc_ba98_7654_3210u64);
        let cases: [(&[u8], usize, bool); 6] = [
            (&[0xf2, 0x0f, 0x78, 0xc1, 0x10, 0x10], 1, true),
            (&[0x66, 0x0f, 0x78, 0xc2, 0x08, 0x14], 7, true),
            (&[0xf2, 0x45, 0x0f, 0x78, 0xc8, 0x0c, 0x28], 11, true),
            (&[0x66, 0x41, 0x0f, 0x78, 0xc1, 0x00, 0x04], 3, true),
            (&[0xf2, 0x0f, 0x78, 0xc9, 0x05, 0x03], 0, true),
            (&[0xf2, 0x0f, 0x78, 0xc1, 0x10, 0x10], 12, false),
        ];
        let replaced = !std::is_x86_feature_detected!("sse4a");
        for (insn, offset, fits) in cases {
            let f = decode(insn).unwrap();
            let (dst, src) = (f.dst, f.src);
            let d_in = if src == dst { s_in } else { d_in };
            let m = mask(f.len);
            let expected = if f.insert {
                d_in & !(m << f.idx) | (s_in & m) << f.idx
            } else {
                (d_in >> f.idx) & m
            };
            let (r, low) = (|x: u8| (x & 8) >> 1, |x: u8| (x & 7) << 3);
            let movq = |x: u8, gpr: u8| [0x66, 0x48 | r(x), 0x0f, 0x6e, 0xc0 | low(x) | gpr];
            let both = |x: u8| [0x66, 0x40 | r(x) | r(x) >> 2, 0x0f, 0x6c, 0xc0 | ((x & 7) * 9)];
            let store = |x: u8, at: u8| [0xf3, 0x41 | r(x), 0x0f, 0x7f, 0x40 | low(x), at];
            let mut code = vec![0x48, 0x8d, 0x64, 0x24, 0xd8];
            code.extend([0xf3, 0x44, 0x0f, 0x7f, 0x04, 0x24]);
            code.extend([0xf3, 0x44, 0x0f, 0x7f, 0x4c, 0x24, 0x10]);
            code.extend([0x66, 0x0f, 0x76, 0xc0, 0x66, 0x0f, 0x76, 0xc9, 0x66, 0x0f, 0x76, 0xd2]);
            code.extend(movq(dst, 1));
            code.extend(both(dst));
            if src != dst {
                code.extend(movq(src, 2));
                code.extend(both(src));
            }
            while code.len() % 16 != offset {
                code.push(0x90);
            }
            let here = code.len();
            code.extend(insn);
            code.extend(store(dst, 0));
            code.extend(store(src, 16));
            for x in 0..3 {
                code.extend(store(x, 32 + 16 * x));
            }
            code.extend([0xf3, 0x44, 0x0f, 0x6f, 0x04, 0x24]);
            code.extend([0xf3, 0x44, 0x0f, 0x6f, 0x4c, 0x24, 0x10]);
            code.extend([0x48, 0x8d, 0x64, 0x24, 0x28, 0xc3]);
            // SAFETY: fresh RWX memory written and called as a function.
            unsafe {
                let page = VirtualAlloc(
                    std::ptr::null(),
                    4096,
                    MEM_COMMIT | MEM_RESERVE,
                    PAGE_EXECUTE_READWRITE,
                );
                assert!(!page.is_null());
                let page = page as usize;
                std::ptr::copy_nonoverlapping(code.as_ptr(), page as *mut u8, code.len());
                let run: extern "system" fn(u64, u64, *mut [u64; 10]) = std::mem::transmute(page);
                for pass in 0..2 {
                    let mut out = [0u64; 10];
                    run(d_in, s_in, &mut out);
                    assert_eq!(out[..2], [expected, 0], "{f:?}, pass {pass}");
                    if src != dst {
                        assert_eq!(out[2..4], [s_in, s_in], "source kept: {f:?}, pass {pass}");
                    }
                    for x in 0..3u8 {
                        if x != dst && x != src {
                            let at = 4 + 2 * x as usize;
                            assert_eq!(out[at..at + 2], [u64::MAX; 2], "xmm{x} kept: {f:?}");
                        }
                    }
                }
                let now = std::slice::from_raw_parts((page + here) as *const u8, insn.len());
                if replaced && fits && enabled() {
                    assert_eq!(now[0], 0xe9, "a jump: {f:?}");
                    assert!(was_patched(page + here));
                } else {
                    assert_eq!(now, insn, "left alone: {f:?}");
                }
                forget_patched(page + here);
                VirtualFree(page as *mut _, 0, MEM_RELEASE);
            }
        }
        assert!(decode(&[0xf2, 0x0f, 0x79, 0xc1]).is_none(), "the field in a register");
        assert!(decode(&[0x66, 0x0f, 0x78, 0xca, 0x08, 0x14]).is_none(), "extrq with reg not 0");
    }

    #[test]
    fn was_patched_verifies_live_jump_target_and_evicts_reused_sites() {
        // SAFETY: dedicated RWX test page freed at the end of the test.
        unsafe {
            let page = VirtualAlloc(
                std::ptr::null(),
                4096,
                MEM_COMMIT | MEM_RESERVE,
                PAGE_EXECUTE_READWRITE,
            ) as usize;
            assert_ne!(page, 0);
            let insn = [0x66u8, 0x0f, 0x78, 0xc2, 0x08, 0x14];
            let site_a = page;
            let site_b = page + 32;
            std::ptr::copy_nonoverlapping(insn.as_ptr(), site_a as *mut u8, insn.len());
            std::ptr::copy_nonoverlapping(insn.as_ptr(), site_b as *mut u8, insn.len());
            replace(site_a);
            replace(site_b);
            if enabled() && std::is_x86_feature_detected!("cmpxchg16b") {
                assert!(was_patched(site_a));
                assert!(was_patched(site_b));
                let mut valid_jmp_a = [0u8; 5];
                let mut valid_jmp_b = [0u8; 5];
                assert_eq!(read_code_into(site_a, &mut valid_jmp_a), 5);
                assert_eq!(read_code_into(site_b, &mut valid_jmp_b), 5);
                // Site A: simulate reuse with a `jmp rel32` pointing outside our
                // stub arena; must reject & evict this live entry.
                let bad_jmp = jump(site_a, site_a + 64).unwrap();
                std::ptr::copy_nonoverlapping(bad_jmp.as_ptr(), site_a as *mut u8, bad_jmp.len());
                assert!(!was_patched(site_a));
                // Restore original valid jump and confirm the entry is truly absent.
                std::ptr::copy_nonoverlapping(valid_jmp_a.as_ptr(), site_a as *mut u8, 5);
                assert!(!was_patched(site_a));
                // Site B (still a live registered entry): simulate reuse with
                // non-jump bytes (e.g. `ud2`); must also reject & evict.
                std::ptr::copy_nonoverlapping([0x0fu8, 0x0b, 0x90, 0x90, 0x90].as_ptr(), site_b as *mut u8, 5);
                assert!(!was_patched(site_b));
                // Restore original valid jump and confirm site B is truly absent.
                std::ptr::copy_nonoverlapping(valid_jmp_b.as_ptr(), site_b as *mut u8, 5);
                assert!(!was_patched(site_b));
            }
            forget_patched(site_a);
            forget_patched(site_b);
            VirtualFree(page as *mut _, 0, MEM_RELEASE);
        }
    }

    #[test]
    fn read_code_crosses_adjacent_readable_regions_and_stops_at_unreadable() {
        use windows_sys::Win32::System::Memory::{PAGE_NOACCESS, PAGE_READONLY, PAGE_READWRITE, VirtualProtect};
        // SAFETY: 3 contiguous committed pages with distinct protections.
        unsafe {
            let base = VirtualAlloc(
                std::ptr::null(),
                3 * 0x1000,
                MEM_COMMIT | MEM_RESERVE,
                PAGE_READWRITE,
            ) as usize;
            assert_ne!(base, 0);
            // Write a 6-byte `extrq` straddling page 0 -> page 1.
            let insn = [0x66u8, 0x0f, 0x78, 0xc2, 0x08, 0x14];
            let cross = base + 0x1000 - 3;
            std::ptr::copy_nonoverlapping(insn.as_ptr(), cross as *mut u8, insn.len());
            let mut old = 0;
            // Give page 1 a different protection so VirtualQuery splits regions,
            // and make page 2 unreadable to verify stopping at a boundary.
            assert_ne!(VirtualProtect((base + 0x1000) as *const _, 0x1000, PAGE_READONLY, &mut old), 0);
            assert_ne!(VirtualProtect((base + 0x2000) as *const _, 0x1000, PAGE_NOACCESS, &mut old), 0);
            let mut buf = [0u8; 16];
            let got = read_code_into(cross, &mut buf);
            assert_eq!(got, 16);
            assert_eq!(&buf[..6], &insn);
            // Near end of page 1 before unreadable page 2: stops cleanly at 4 bytes.
            let got_edge = read_code_into(base + 0x2000 - 4, &mut buf);
            assert_eq!(got_edge, 4);
            VirtualFree(base as *mut _, 0, MEM_RELEASE);
        }
    }
}
