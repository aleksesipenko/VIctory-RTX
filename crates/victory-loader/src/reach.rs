//! Code of ours within reach of the title's: the two things a patch that
//! turns a title instruction into a `jmp rel32` to a stub needs, written from
//! the instruction encoding alone (renderer port #2's `descriptor_sites`,
//! `write_sites`).
//!
//! A `jmp rel32` is `E9` and a signed 32-bit displacement from the end of
//! the instruction, so its target must lie within ±2 GiB of it; the stubs are
//! put in memory allocated that close to the title's code.

/// The bytes of one region of stubs.
pub(crate) const STUBS_LEN: usize = 64 << 10;

/// Windows hands out memory on 64 KiB lines.
const GRANULE: usize = 64 << 10;

/// How far from the code a region of stubs may start, leaving room for the
/// region itself within the reach of a `jmp rel32`.
const REACH: usize = 0x7000_0000;

/// `jmp rel32` at `at` to `to`, if `to` is within its reach.
pub(crate) fn jump(at: usize, to: usize) -> Option<[u8; 5]> {
    let end = i128::try_from(at).ok()?.checked_add(5)?;
    let rel = i32::try_from(i128::try_from(to).ok()? - end).ok()?;
    let [a, b, c, d] = rel.to_le_bytes();
    Some([0xe9, a, b, c, d])
}

/// [`STUBS_LEN`] bytes of executable memory within reach of a `jmp rel32`
/// at `at`: the free region nearest above `at`, else the nearest below.
pub(crate) fn near(at: usize) -> Option<usize> {
    let from = (at & !(GRANULE - 1)).checked_add(GRANULE)?;
    let up = room_from(from, at.saturating_add(REACH), true);
    up.or_else(|| room_from(at.saturating_sub(REACH).max(GRANULE), at & !(GRANULE - 1), false))
}

/// Walks the address space over `[from, to)` (upward or downward) for a
/// free stretch of [`STUBS_LEN`] bytes on a 64 KiB line, and takes it.
fn room_from(from: usize, to: usize, upward: bool) -> Option<usize> {
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEM_FREE, MEM_RESERVE, MEMORY_BASIC_INFORMATION, PAGE_EXECUTE_READWRITE, VirtualAlloc, VirtualQuery,
    };
    let mut at = if upward { from } else { to.checked_sub(STUBS_LEN)? & !(GRANULE - 1) };
    while from <= at && at + STUBS_LEN <= to {
        // SAFETY: plain data filled in by the query.
        let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
        // SAFETY: a valid out pointer of the right size.
        if unsafe { VirtualQuery(at as *const _, &mut info, size_of_val(&info)) } == 0 {
            return None;
        }
        let (base, end) = (info.BaseAddress as usize, info.BaseAddress as usize + info.RegionSize);
        if info.State == MEM_FREE && at + STUBS_LEN <= end {
            // SAFETY: reserves and commits free memory at that address, or fails.
            let got = unsafe { VirtualAlloc(at as *const _, STUBS_LEN, MEM_RESERVE | MEM_COMMIT, PAGE_EXECUTE_READWRITE) } as usize;
            if got != 0 {
                return Some(got);
            }
        }
        at = if upward {
            end.next_multiple_of(GRANULE)
        } else {
            base.checked_sub(STUBS_LEN)? & !(GRANULE - 1)
        };
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The displacement counts from the end of the five bytes, both ways,
    /// and nothing past ±2 GiB is reached.
    #[test]
    fn a_jump_counts_from_its_end_and_stops_at_its_reach() {
        assert_eq!(jump(0x1000, 0x1005), Some([0xe9, 0, 0, 0, 0]));
        assert_eq!(jump(0x1000, 0x1000), Some([0xe9, 0xfb, 0xff, 0xff, 0xff]));
        assert_eq!(jump(0x1000, 0x2005), Some([0xe9, 0x00, 0x10, 0x00, 0x00]));
        let far = 0x1_0000_0000usize;
        assert_eq!(jump(far, far + 5 + i32::MAX as usize), Some([0xe9, 0xff, 0xff, 0xff, 0x7f]));
        assert_eq!(jump(far, far + 5 + i32::MAX as usize + 1), None);
        assert_eq!(jump(far, far + 5 - (1usize << 31)), Some([0xe9, 0, 0, 0, 0x80]));
        assert_eq!(jump(far, far + 4 - (1usize << 31)), None);
    }

    /// Room taken near some code of this process is within a jump of it,
    /// on a 64 KiB line, and usable.
    #[test]
    fn room_near_code_is_within_a_jump_of_it() {
        let code = a_jump_counts_from_its_end_and_stops_at_its_reach as *const () as usize;
        let room = near(code).expect("room near our own code");
        assert_eq!(room % GRANULE, 0);
        assert!(jump(code, room).is_some() && jump(room + STUBS_LEN - 5, code).is_some());
        // SAFETY: the region was just committed read-write-execute for us.
        unsafe {
            (room as *mut u8).write(0xc3);
            assert_eq!((room as *const u8).read(), 0xc3);
            windows_sys::Win32::System::Memory::VirtualFree(room as *mut _, 0, windows_sys::Win32::System::Memory::MEM_RELEASE);
        }
    }
}
