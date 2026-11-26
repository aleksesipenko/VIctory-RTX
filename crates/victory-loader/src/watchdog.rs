//! Opt-in watchdog disarm hook for known title builds (`VICTORY_NO_WATCHDOG=1`).
//!
//! Ported from `ZigZagBoi1/ScarlettPort` / `miaohaixiang1-netizen/cool-fox-12gb`
//! (GPL-3.0-or-later) using [`crate::resolution::Site`] and
//! [`crate::resolution::write`]: only per-build offsets and exact expected byte
//! sequences live here, compared byte-for-byte before patching.
//!
//! The title's watchdog thread intentionally faults the process when its main
//! thread has not advanced for ~30 one-second ticks (e.g. during cold shader /
//! pipeline compilation). Two sites perform the fault in build 1.0.2.1:
//! - `+0x2215c27`: the watchdog loop calls the configured handler pointer and
//!   falls back to `mov dword ptr [0], 0; int3`; jumping `+0x18` returns to the
//!   loop's normal continuation.
//! - `+0x23cf683`: the handler's deliberate null-pointer write
//!   (`mov dword ptr [0], 0x4d8bd421`); replaced with an 11-byte NOP.

use crate::frame_rate::Patched;
use crate::resolution::Site;

/// Environment variable that enables the watchdog disarm hook (`=1`).
pub const NO_WATCHDOG_ENV: &str = "VICTORY_NO_WATCHDOG";

/// First 16 hex digits of the BLAKE3 hash of supported build 1.0.2.1.
const BUILD_1_0_2_1: &str = "b99f876b74d4a77f";

const SKIP_CALL_AND_FALLBACK: Site = Site {
    at: 0x2215c27,
    what: "watchdog: skip handler call and fallback fault",
    // mov rax, [rip + disp32] (the handler pointer)
    bytes: &[0x48, 0x8B, 0x05, 0x1A, 0x07, 0xFA, 0x04],
};

// jmp +0x18 (normal loop continuation), then 5 NOPs
const SKIP_CALL_REPLACEMENT: &[u8] = &[0xEB, 0x18, 0x90, 0x90, 0x90, 0x90, 0x90];

const HANDLER_FAULT: Site = Site {
    at: 0x23cf683,
    what: "watchdog: handler deliberate fault",
    // mov dword ptr [0], 0x4d8bd421
    bytes: &[0xC7, 0x04, 0x25, 0x00, 0x00, 0x00, 0x00, 0x21, 0xD4, 0x8B, 0x4D],
};

// 11-byte canonical NOP
const HANDLER_FAULT_REPLACEMENT: &[u8] = &[0x66, 0x66, 0x2E, 0x0F, 0x1F, 0x84, 0x00, 0x00, 0x00, 0x00, 0x00];

/// Whether `VICTORY_NO_WATCHDOG=1` is set in the environment.
pub fn asked() -> bool {
    std::env::var_os(NO_WATCHDOG_ENV).is_some_and(|v| v == "1")
}

/// Applies the watchdog disarm patches when `VICTORY_NO_WATCHDOG=1` is set and
/// the build hash and bytes match.
///
/// # Safety
/// The image of `size` bytes at `base` stays mapped and no thread runs its
/// code yet.
#[cfg(windows)]
pub unsafe fn apply_asked(base: usize, size: usize, hash_hex: &str) -> usize {
    if !asked() {
        return 0;
    }
    if !hash_hex.starts_with(BUILD_1_0_2_1) {
        tracing::warn!(hash = hash_hex, "watchdog: {NO_WATCHDOG_ENV}=1 ignored for unknown build");
        return 0;
    }
    let mut patched = 0;
    for (site, code) in [
        (&SKIP_CALL_AND_FALLBACK, SKIP_CALL_REPLACEMENT),
        (&HANDLER_FAULT, HANDLER_FAULT_REPLACEMENT),
    ] {
        // SAFETY: forwarded from caller's contract.
        let done = unsafe { crate::resolution::write(base, size, site, code) };
        let offset = format!("{:#x}", site.at);
        match done {
            Patched::Done => {
                patched += 1;
                tracing::info!(site = %offset, what = site.what, "watchdog: patched");
            }
            Patched::Differs => tracing::warn!(
                site = %offset,
                what = site.what,
                "watchdog: not patched, bytes differ (another build?)"
            ),
            Patched::Failed(why) => tracing::warn!(
                site = %offset,
                what = site.what,
                why,
                "watchdog: not patched"
            ),
        }
        println!("watchdog: {} at title+{offset}: {done:?}", site.what);
    }
    patched
}
