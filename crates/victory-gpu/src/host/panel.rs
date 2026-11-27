//! Diagnostics: the run's numbers on the picture (`VICTORY_PANEL=1`; off by
//! default; F3 shows and hides it while our window has the focus): white
//! text on a black box in the upper left corner of what is shown, so that a
//! screenshot carries them, with what it costs itself on the submitting
//! thread. Made again twice a second from the counters of
//! the frame-rate line, memory and textures; a frame itself only copies the
//! box out of a buffer in the card's memory. Copied onto the swap chain's
//! buffer, behind the title's picture: saved frames and the frame recorder
//! see the title's own.

use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;
use windows_sys::Win32::UI::Input::KeyboardAndMouse::{GetAsyncKeyState, VK_F3};
use windows_sys::Win32::UI::WindowsAndMessaging::{GetForegroundWindow, GetWindowThreadProcessId};

use super::{Host, HostList, transition};

/// How often the numbers change.
const EVERY: Duration = Duration::from_millis(500);

/// A character's dots, and with the space beside and below it.
const GLYPH: (u32, u32) = (5, 7);
const CELL: (u32, u32) = (6, 9);
/// Characters in a line, lines, and the box's rim in dots.
const COLUMNS: u32 = 40;
const LINES: u32 = 11;
const RIM: u32 = 3;

fn asked() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| crate::switch::on("VICTORY_PANEL"))
}

/// Whether it is shown now.
fn shown() -> &'static AtomicBool {
    static SHOWN: OnceLock<AtomicBool> = OnceLock::new();
    SHOWN.get_or_init(|| AtomicBool::new(asked()))
}

/// F3 was pressed since the last call, with one of our windows in front.
fn pressed() -> bool {
    static DOWN: AtomicBool = AtomicBool::new(false);
    // SAFETY: a plain query.
    let down = unsafe { GetAsyncKeyState(i32::from(VK_F3)) } < 0;
    if DOWN.swap(down, Ordering::Relaxed) || !down {
        return false;
    }
    let mut process = 0;
    // SAFETY: plain queries; room for the answer.
    unsafe { GetWindowThreadProcessId(GetForegroundWindow(), &mut process) };
    process == std::process::id()
}

/// Seven rows of five dots, the leftmost the highest bit.
#[rustfmt::skip]
const fn glyph(c: u8) -> [u8; 7] {
    match c {
        b'0' => [0b01110, 0b10001, 0b10011, 0b10101, 0b11001, 0b10001, 0b01110],
        b'1' => [0b00100, 0b01100, 0b00100, 0b00100, 0b00100, 0b00100, 0b01110],
        b'2' => [0b01110, 0b10001, 0b00001, 0b00110, 0b01000, 0b10000, 0b11111],
        b'3' => [0b11110, 0b00001, 0b00001, 0b01110, 0b00001, 0b00001, 0b11110],
        b'4' => [0b00010, 0b00110, 0b01010, 0b10010, 0b11111, 0b00010, 0b00010],
        b'5' => [0b11111, 0b10000, 0b11110, 0b00001, 0b00001, 0b10001, 0b01110],
        b'6' => [0b00110, 0b01000, 0b10000, 0b11110, 0b10001, 0b10001, 0b01110],
        b'7' => [0b11111, 0b00001, 0b00010, 0b00010, 0b00100, 0b00100, 0b00100],
        b'8' => [0b01110, 0b10001, 0b10001, 0b01110, 0b10001, 0b10001, 0b01110],
        b'9' => [0b01110, 0b10001, 0b10001, 0b01111, 0b00001, 0b00010, 0b01100],
        b'A' => [0b00100, 0b01010, 0b10001, 0b10001, 0b11111, 0b10001, 0b10001],
        b'B' => [0b11110, 0b10001, 0b10001, 0b11110, 0b10001, 0b10001, 0b11110],
        b'C' => [0b01111, 0b10000, 0b10000, 0b10000, 0b10000, 0b10000, 0b01111],
        b'D' => [0b11110, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b11110],
        b'E' => [0b11111, 0b10000, 0b10000, 0b11110, 0b10000, 0b10000, 0b11111],
        b'F' => [0b11111, 0b10000, 0b10000, 0b11110, 0b10000, 0b10000, 0b10000],
        b'G' => [0b01111, 0b10000, 0b10000, 0b10011, 0b10001, 0b10001, 0b01111],
        b'H' => [0b10001, 0b10001, 0b10001, 0b11111, 0b10001, 0b10001, 0b10001],
        b'I' => [0b01110, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b01110],
        b'J' => [0b00111, 0b00010, 0b00010, 0b00010, 0b00010, 0b10010, 0b01100],
        b'K' => [0b10001, 0b10010, 0b10100, 0b11000, 0b10100, 0b10010, 0b10001],
        b'L' => [0b10000, 0b10000, 0b10000, 0b10000, 0b10000, 0b10000, 0b11111],
        b'M' => [0b10001, 0b11011, 0b10101, 0b10101, 0b10001, 0b10001, 0b10001],
        b'N' => [0b10001, 0b11001, 0b11001, 0b10101, 0b10011, 0b10011, 0b10001],
        b'O' => [0b01110, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01110],
        b'P' => [0b11110, 0b10001, 0b10001, 0b11110, 0b10000, 0b10000, 0b10000],
        b'Q' => [0b01110, 0b10001, 0b10001, 0b10001, 0b10101, 0b10010, 0b01101],
        b'R' => [0b11110, 0b10001, 0b10001, 0b11110, 0b10100, 0b10010, 0b10001],
        b'S' => [0b01111, 0b10000, 0b10000, 0b01110, 0b00001, 0b00001, 0b11110],
        b'T' => [0b11111, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100],
        b'U' => [0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01110],
        b'V' => [0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01010, 0b00100],
        b'W' => [0b10001, 0b10001, 0b10001, 0b10101, 0b10101, 0b11011, 0b10001],
        b'X' => [0b10001, 0b10001, 0b01010, 0b00100, 0b01010, 0b10001, 0b10001],
        b'Y' => [0b10001, 0b10001, 0b01010, 0b00100, 0b00100, 0b00100, 0b00100],
        b'Z' => [0b11111, 0b00001, 0b00010, 0b00100, 0b01000, 0b10000, 0b11111],
        b'.' => [0, 0, 0, 0, 0, 0b01100, 0b01100],
        b'/' => [0b00001, 0b00010, 0b00010, 0b00100, 0b01000, 0b01000, 0b10000],
        b'-' => [0, 0, 0, 0b11111, 0, 0, 0],
        b'|' => [0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100],
        b'%' => [0b11001, 0b11010, 0b00010, 0b00100, 0b01000, 0b01011, 0b10011],
        b' ' => [0; 7],
        _ => [0b11111, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b11111],
    }
}

/// The box's size in pixels, a dot being `scale` of them each way.
const fn size(scale: u32) -> (u32, u32) {
    ((COLUMNS * CELL.0 + 2 * RIM) * scale, (LINES * CELL.1 + 2 * RIM) * scale)
}

/// Pixels to a dot for a picture that high: a 720-line screenshot of it
/// shows one and a third.
fn scale(height: u32) -> u32 {
    (height / 540).max(1)
}

/// Whether a dot of the picture of `lines` is white.
fn lit(lines: &[String], x: u32, y: u32) -> bool {
    let (Some(x), Some(y)) = (x.checked_sub(RIM), y.checked_sub(RIM)) else {
        return false;
    };
    if x % CELL.0 >= GLYPH.0 || y % CELL.1 >= GLYPH.1 || x / CELL.0 >= COLUMNS {
        return false;
    }
    let Some(&c) = lines.get((y / CELL.1) as usize).and_then(|l| l.as_bytes().get((x / CELL.0) as usize)) else {
        return false;
    };
    (glyph(c.to_ascii_uppercase())[(y % CELL.1) as usize] >> (GLYPH.0 - 1 - x % CELL.0)) & 1 != 0
}

/// The bytes of a black and a white pixel of a swap chain's format.
fn colours(format: DXGI_FORMAT) -> Option<[&'static [u8]; 2]> {
    Some(match format {
        DXGI_FORMAT_R8G8B8A8_UNORM | DXGI_FORMAT_B8G8R8A8_UNORM => [&[0, 0, 0, 0xFF], &[0xFF; 4]],
        DXGI_FORMAT_R10G10B10A2_UNORM => [&[0, 0, 0, 0xC0], &[0xFF; 4]],
        // Halves of 0 and 1.
        DXGI_FORMAT_R16G16B16A16_FLOAT => [
            &[0, 0, 0, 0, 0, 0, 0, 0x3C],
            &[0, 0x3C, 0, 0x3C, 0, 0x3C, 0, 0x3C],
        ],
        _ => return None,
    })
}

/// The box as rows of `pitch` bytes.
fn painted(lines: &[String], scale: u32, colours: [&[u8]; 2], pitch: usize) -> Vec<u8> {
    let (width, height) = size(scale);
    let bytes = colours[0].len();
    let mut out = vec![0; pitch * height as usize];
    for y in 0..height / scale {
        let top = pitch * (y * scale) as usize;
        for x in 0..width / scale {
            let colour = colours[usize::from(lit(lines, x, y))];
            for i in 0..scale {
                let at = top + (x * scale + i) as usize * bytes;
                out[at..at + bytes].copy_from_slice(colour);
            }
        }
        for i in 1..scale as usize {
            out.copy_within(top..top + pitch, top + i * pitch);
        }
    }
    out
}

/// The counters read for it. They grow, and go back to 0 with the frame-rate
/// line (right behind a look of ours).
#[derive(Clone, Copy, Default)]
struct Counters([u64; COUNTERS]);

const COUNTERS: usize = 11;

impl Counters {
    fn now() -> Counters {
        use super::{CARD_CLOCK_US, COPY_BACK, PREPARING};
        use crate::copystat::{C, peek};
        let get = |a: &std::sync::atomic::AtomicU64| a.load(Ordering::Relaxed);
        Counters([
            super::worked(),
            get(&CARD_CLOCK_US[0]),
            get(&CARD_CLOCK_US[1]),
            get(&CARD_CLOCK_US[2]),
            get(&PREPARING[3]),
            get(&COPY_BACK[0]),
            get(&COPY_BACK[1]),
            get(&crate::shadow::CARD_STATS[0]),
            crate::draw::dropped_now(),
            peek(C::RecordNs),
            crate::draw::drawn(),
        ])
    }

    /// What came since `before`, per frame.
    fn since(self, before: Counters, frames: f64) -> [f64; COUNTERS] {
        let mut out = [0.0; COUNTERS];
        for (i, out) in out.iter_mut().enumerate() {
            let (was, is) = (before.0[i], self.0[i]);
            *out = if is >= was { is - was } else { is } as f64 / frames;
        }
        out
    }
}

fn gib(bytes: u64) -> f64 {
    bytes as f64 / (1u64 << 30) as f64
}

fn display_gib(bytes: Option<u64>) -> String {
    bytes.map_or_else(|| "N/A".to_owned(), |n| format!("{:.1}", gib(n)))
}

/// Format local process usage, OS budget, adapter dedicated capacity, and
/// non-local/shared process usage distinctly (`N/A` on query failure vs `0.0`
/// on a measured zero).
/// Provenance (GPL-3.0): Belobug/VIctory-Belobug-AMD#348.
fn video_memory_lines(
    local: Option<(u64, u64)>,
    dedicated: Option<u64>,
    shared: Option<u64>,
) -> [String; 2] {
    [
        format!(
            "LOCAL {} | BUDGET {} GIB",
            display_gib(local.map(|s| s.0)),
            display_gib(local.map(|s| s.1)),
        ),
        format!(
            "DED CAP {} | SHARED {} GIB",
            display_gib(dedicated),
            display_gib(shared),
        ),
    ]
}

/// Adapter dedicated VRAM capacity from DXGI for this sample.
/// Provenance (GPL-3.0): Belobug/VIctory-Belobug-AMD#348.
fn gpu_dedicated_memory(host: &Host) -> Option<u64> {
    let adapter = host.adapter.as_ref()?;
    // SAFETY: plain DXGI descriptor query.
    let desc = unsafe { adapter.GetDesc1() }.ok()?;
    Some(desc.DedicatedVideoMemory as u64)
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct SystemProcessorPerformanceInformation {
    idle_time: i64,
    kernel_time: i64,
    user_time: i64,
    dpc_time: i64,
    interrupt_time: i64,
    interrupt_count: u32,
    _reserved: u32,
}

const _: () = assert!(size_of::<SystemProcessorPerformanceInformation>() == 48);

type NtQuerySystemInformationFn = unsafe extern "system" fn(
    u32,
    *mut std::ffi::c_void,
    u32,
    *mut u32,
) -> i32;

fn nt_query_system_information() -> Option<NtQuerySystemInformationFn> {
    use windows_sys::Win32::System::LibraryLoader::{GetModuleHandleA, GetProcAddress};
    static FN: OnceLock<Option<NtQuerySystemInformationFn>> = OnceLock::new();
    *FN.get_or_init(|| {
        // SAFETY: static NUL-terminated ASCII module and export names.
        unsafe {
            let ntdll = GetModuleHandleA(c"ntdll.dll".as_ptr().cast());
            if ntdll.is_null() {
                return None;
            }
            let proc = GetProcAddress(ntdll, c"NtQuerySystemInformation".as_ptr().cast())?;
            Some(std::mem::transmute::<
                unsafe extern "system" fn() -> isize,
                NtQuerySystemInformationFn,
            >(proc))
        }
    })
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct CpuTimes {
    idle: i64,
    kernel: i64,
    user: i64,
}

const SYSTEM_PROCESSOR_PERFORMANCE_INFORMATION: u32 = 8;
const STATUS_INFO_LENGTH_MISMATCH: i32 = 0xC000_0004_u32 as i32;
const MAX_CPUS: usize = 1024;

/// Loops and grows the performance-information buffer on
/// `STATUS_INFO_LENGTH_MISMATCH` up to `MAX_CPUS`, handling multi-group systems.
fn query_cpu_times_with<F>(initial_cap: usize, mut query: F) -> Vec<CpuTimes>
where
    F: FnMut(&mut [SystemProcessorPerformanceInformation], &mut u32) -> i32,
{
    let entry_size = size_of::<SystemProcessorPerformanceInformation>();
    let mut cap = initial_cap.clamp(1, MAX_CPUS);
    for _ in 0..8 {
        let mut buf = vec![SystemProcessorPerformanceInformation::default(); cap];
        let mut returned = 0u32;
        let status = query(&mut buf, &mut returned);
        if status >= 0 {
            let count = (returned as usize / entry_size).min(buf.len());
            return buf[..count]
                .iter()
                .map(|p| CpuTimes {
                    idle: p.idle_time,
                    kernel: p.kernel_time,
                    user: p.user_time,
                })
                .collect();
        }
        if status == STATUS_INFO_LENGTH_MISMATCH {
            let needed = if returned > 0 {
                (returned as usize).div_ceil(entry_size)
            } else {
                cap.saturating_mul(2)
            };
            let next = needed.clamp(1, MAX_CPUS);
            if next <= cap {
                break;
            }
            cap = next;
        } else {
            break;
        }
    }
    Vec::new()
}

/// Per-core `(idle, kernel, user)` counters from Windows across all processor groups.
fn read_cpu_times() -> Vec<CpuTimes> {
    use windows_sys::Win32::System::Threading::{ALL_PROCESSOR_GROUPS, GetActiveProcessorCount};
    let Some(query) = nt_query_system_information() else {
        return Vec::new();
    };
    // SAFETY: `ALL_PROCESSOR_GROUPS` (0xFFFF) queries active logical processors across all groups.
    let active = unsafe { GetActiveProcessorCount(ALL_PROCESSOR_GROUPS) } as usize;
    query_cpu_times_with(active, |buf, returned| {
        let bytes = size_of_val(buf) as u32;
        // SAFETY: `buf` holds `bytes` writable bytes; `returned` is a valid out pointer.
        unsafe {
            query(
                SYSTEM_PROCESSOR_PERFORMANCE_INFORMATION,
                buf.as_mut_ptr().cast(),
                bytes,
                returned,
            )
        }
    })
}

/// How busy a core was between two reads of its `(idle, kernel, user)`
/// counters, in percent. On Windows `kernel` includes `idle`, so total elapsed
/// time is `kernel + user`. If any counter steps backward between two reads
/// (seen on a Ryzen 7 7700) or total elapsed time is non-positive, treat the
/// interval as invalid and read `0.0%` (idle) rather than panicking or
/// producing a false `100%` spike.
/// Provenance (GPL-3.0): Belobug/VIctory-Belobug-AMD#325.
fn busy_pct(prev: CpuTimes, cur: CpuTimes) -> f32 {
    if cur.idle < prev.idle || cur.kernel < prev.kernel || cur.user < prev.user {
        return 0.0;
    }
    let di = cur.idle.saturating_sub(prev.idle);
    let dk = cur.kernel.saturating_sub(prev.kernel);
    let du = cur.user.saturating_sub(prev.user);
    let span = dk.saturating_add(du);
    if span <= 0 {
        return 0.0;
    }
    let di = di.min(span) as f32;
    ((1.0 - di / span as f32) * 100.0).clamp(0.0, 100.0)
}

#[derive(Default)]
struct CpuMeter {
    prev: Vec<CpuTimes>,
    total_pct: Option<f32>,
    peak_core_pct: Option<f32>,
}

impl CpuMeter {
    fn reset(&mut self) {
        self.prev.clear();
        self.total_pct = None;
        self.peak_core_pct = None;
    }

    fn sample(&mut self) {
        self.update(read_cpu_times());
    }

    fn update(&mut self, now: Vec<CpuTimes>) {
        if now.is_empty() {
            self.reset();
            return;
        }
        if self.prev.len() == now.len() {
            let mut sum = 0.0f32;
            let mut peak = 0.0f32;
            for (&prev, &cur) in self.prev.iter().zip(&now) {
                let busy = busy_pct(prev, cur);
                sum += busy;
                peak = peak.max(busy);
            }
            self.total_pct = Some(sum / now.len() as f32);
            self.peak_core_pct = Some(peak);
        } else {
            self.total_pct = None;
            self.peak_core_pct = None;
        }
        self.prev = now;
    }
}

fn cpu_line(total_pct: Option<f32>, peak_core_pct: Option<f32>) -> String {
    match (total_pct, peak_core_pct) {
        (Some(total), Some(peak)) => format!("CPU {total:.0}%  PEAK CORE {peak:.0}%"),
        _ => "CPU N/A".to_owned(),
    }
}

/// The text. `c`: [`Counters`] per frame, `ms` a frame's time, `worst` the
/// longest frame's, `cost` what this takes of a frame, in microseconds.
fn text(
    c: &[f64; COUNTERS],
    ms: f64,
    worst: f64,
    cost: f64,
    host: &Host,
    cpu: &CpuMeter,
) -> Vec<String> {
    let (working, private) = crate::copystat::process_memory();
    let [local, shared] = host.sample_video_memory();
    let dedicated = gpu_dedicated_memory(host);
    let [vram_local, vram_capacity] = video_memory_lines(
        local.map(|s| (s.CurrentUsage, s.Budget)),
        dedicated,
        shared.map(|s| s.CurrentUsage),
    );
    let idle = (c[1] / 1e3).min(ms);
    vec![
        format!("FPS {:.1}  FRAME {ms:.1} MS  LONGEST {worst:.0}", 1e3 / ms.max(0.001)),
        format!("DRAWS {:.0}  DROPPED {:.0}", c[10], c[8]),
        format!("SUBMIT BUSY {:.1}  WAITS FOR GPU {:.1}", c[0] / 1e3, c[3] / 1e3),
        format!("GPU BUSY {:.1}  IDLE BEHIND US {:.1}", ms - idle, c[2] / 1e3),
        format!("RECORDING {:.0} MS  COPY BACK {:.1}", c[9] / 1e6, c[5] / 1e3),
        format!("MB UP {:.0}  TEXTURES {:.0}  BACK {:.0}", c[7] / 1e6, c[4] / 1e6, c[6] / 1e6),
        format!("COMMIT {:.1} GIB  IN MEMORY {:.1}", gib(private), gib(working)),
        vram_local,
        vram_capacity,
        cpu_line(cpu.total_pct, cpu.peak_core_pct),
        format!("THIS PANEL {:.0} US A FRAME", cost),
    ]
}

/// The buffers of a box that size and format (GPL-3.0 provenance:
/// Belobug/VIctory-Belobug-AMD #359 — stages through `list.scratch_at` and
/// pins `card` via `list.keep` instead of a 4-slot ring + `mem::forget`).
struct Buffers {
    format: DXGI_FORMAT,
    scale: u32,
    pitch: u32,
    card: ID3D12Resource,
    /// `card` has already been copied into once and now sits in `COPY_SOURCE`.
    /// Provenance (GPL-3.0): Belobug 5d6bd84.
    uploaded: bool,
}

// SAFETY: free-threaded resources, only mutated under the presenter's lock.
unsafe impl Send for Buffers {}

impl Buffers {
    fn new(host: &Host, format: DXGI_FORMAT, scale: u32) -> Option<Buffers> {
        let (width, height) = size(scale);
        let pitch = (width * colours(format)?[0].len() as u32).next_multiple_of(256);
        let bytes = u64::from(pitch) * u64::from(height);
        let heap = D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_DEFAULT,
            ..Default::default()
        };
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: bytes,
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            ..Default::default()
        };
        let mut card: Option<ID3D12Resource> = None;
        // SAFETY: plain creation of a buffer.
        unsafe {
            host.device
                .CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_COMMON, None, &mut card)
                .ok()?;
        }
        Some(Buffers { format, scale, pitch, card: card?, uploaded: false })
    }

    fn write(&mut self, list: &HostList, lines: &[String]) {
        let Some(colours) = colours(self.format) else { return };
        let bytes = u64::from(self.pitch) * u64::from(size(self.scale).1);
        let Some((stage, offset, mapped)) = list.scratch_at(bytes) else { return };
        let picture = painted(lines, self.scale, colours, self.pitch as usize);
        // SAFETY: a fresh mapped range of `bytes` held by the present's list
        // until its submission lands and the list resets.
        unsafe {
            std::ptr::copy_nonoverlapping(picture.as_ptr(), mapped, picture.len());
            if self.uploaded {
                list.list.ResourceBarrier(&[transition(
                    &self.card,
                    D3D12_RESOURCE_STATE_COPY_SOURCE,
                    D3D12_RESOURCE_STATE_COPY_DEST,
                )]);
            }
            list.list.CopyBufferRegion(&self.card, 0, &stage, offset, bytes);
            list.list.ResourceBarrier(&[transition(
                &self.card,
                D3D12_RESOURCE_STATE_COPY_DEST,
                D3D12_RESOURCE_STATE_COPY_SOURCE,
            )]);
        }
        self.uploaded = true;
    }
}

/// The panel of a presenter.
pub(super) struct Panel {
    since: Instant,
    frames: u32,
    last: Instant,
    worst: f64,
    /// What this took since the last text.
    cost: Duration,
    before: Counters,
    cpu: CpuMeter,
    lines: Vec<String>,
    changed: bool,
    buffers: Option<Buffers>,
    failed: bool,
}

impl Panel {
    pub(super) fn new() -> Panel {
        let now = Instant::now();
        Panel {
            since: now,
            frames: 0,
            last: now,
            worst: 0.0,
            cost: Duration::ZERO,
            before: Counters::default(),
            cpu: CpuMeter::default(),
            lines: Vec::new(),
            changed: false,
            buffers: None,
            failed: false,
        }
    }

    /// A frame was presented; `line`: the counters go back to 0 right behind
    /// this call. (Not shown it asks for the key's state and no more.)
    pub(super) fn frame(&mut self, line: bool) {
        if pressed() {
            shown().fetch_xor(true, Ordering::Relaxed);
        }
        if !shown().load(Ordering::Relaxed) {
            self.frames = 0;
            return;
        }
        let now = Instant::now();
        if self.frames == 0 && now - self.last > 2 * EVERY {
            // (Shown from here on.)
            (self.since, self.last, self.worst, self.before) = (now, now, 0.0, Counters::now());
            self.cpu.reset();
            self.cpu.sample();
            self.frames = 0;
        } else if self.cpu.prev.is_empty() {
            self.cpu.sample();
        }
        self.frames += 1;
        self.worst = self.worst.max((now - self.last).as_secs_f64() * 1e3);
        self.last = now;
        let elapsed = now - self.since;
        if elapsed < EVERY && !(line && elapsed * 2 >= EVERY) {
            return;
        }
        let Some(host) = super::host() else { return };
        let frames = f64::from(self.frames);
        let counters = Counters::now();
        let c = counters.since(self.before, frames);
        self.before = counters;
        self.cpu.sample();
        let cost = self.cost.as_secs_f64() * 1e6 / frames;
        self.lines = text(&c, elapsed.as_secs_f64() * 1e3 / frames, self.worst, cost, host, &self.cpu);
        self.changed = true;
        (self.since, self.frames, self.worst) = (now, 0, 0.0);
        self.cost = now.elapsed();
    }

    /// Records the copy of the box onto `back`, a buffer of the swap chain of
    /// that size and format in the state to be copied into.
    pub(super) fn draw(
        &mut self,
        host: &Host,
        list: &HostList,
        back: &ID3D12Resource,
        (width, height): (u32, u32),
        format: DXGI_FORMAT,
    ) {
        if !shown().load(Ordering::Relaxed) || self.lines.is_empty() || self.failed {
            return;
        }
        let began = Instant::now();
        // Not the title's: the frame recorder leaves it out.
        let _ours = crate::replay::record::unrecorded();
        let scale = scale(height);
        if self.buffers.as_ref().is_none_or(|b| (b.format, b.scale) != (format, scale)) {
            self.buffers = Buffers::new(host, format, scale);
            self.failed = self.buffers.is_none();
            self.changed = true;
        }
        let Some(buffers) = &mut self.buffers else { return };
        if std::mem::take(&mut self.changed) {
            buffers.write(list, &self.lines);
        }
        if !buffers.uploaded {
            return;
        }
        // Keep `card` alive until this submission lands even if format/scale changes next frame.
        list.keep(&buffers.card);
        let (w, h) = size(scale);
        let list = &list.list;
        // SAFETY: an open list and live resources; the box lies within both.
        unsafe {
            let from = D3D12_TEXTURE_COPY_LOCATION {
                pResource: std::mem::transmute_copy(&buffers.card),
                Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
                Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                    PlacedFootprint: D3D12_PLACED_SUBRESOURCE_FOOTPRINT {
                        Offset: 0,
                        Footprint: D3D12_SUBRESOURCE_FOOTPRINT {
                            Format: format,
                            Width: w,
                            Height: h,
                            Depth: 1,
                            RowPitch: buffers.pitch,
                        },
                    },
                },
            };
            let to = D3D12_TEXTURE_COPY_LOCATION {
                pResource: std::mem::transmute_copy(back),
                Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
                Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: 0 },
            };
            let part = D3D12_BOX {
                left: 0,
                top: 0,
                front: 0,
                right: w.min(width),
                bottom: h.min(height),
                back: 1,
            };
            list.CopyTextureRegion(&to, 0, 0, 0, &from, Some(&part));
        }
        self.cost += began.elapsed();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every character the lines use has its own picture, and no two look
    /// alike.
    #[test]
    fn characters_differ() {
        let used: Vec<u8> = (b'0'..=b'9').chain(b'A'..=b'Z').chain(*b"./-|% ").collect();
        for (i, a) in used.iter().enumerate() {
            assert_ne!(glyph(*a), glyph(b'#'), "{} has a picture", *a as char);
            assert!(glyph(*a).iter().all(|row| row >> GLYPH.0 == 0));
            for b in &used[i + 1..] {
                assert_ne!(glyph(*a), glyph(*b), "{} {}", *a as char, *b as char);
            }
        }
    }

    #[test]
    fn video_memory_keeps_budget_capacity_and_usage_distinct() {
        let lines = video_memory_lines(Some((17 << 30, 16 << 30)), Some(24 << 30), Some(1 << 29));
        assert_eq!(lines, ["LOCAL 17.0 | BUDGET 16.0 GIB", "DED CAP 24.0 | SHARED 0.5 GIB"]);
        let large = video_memory_lines(Some((512 << 30, 256 << 30)), Some(768 << 30), Some(128 << 30));
        for line in large {
            assert!(line.len() <= COLUMNS as usize, "{line}");
            assert!(line.bytes().all(|b| glyph(b.to_ascii_uppercase()) != glyph(b'#')), "{line}");
        }
    }

    #[test]
    fn missing_video_memory_samples_stay_unknown() {
        assert_eq!(
            video_memory_lines(None, Some(0), Some(0)),
            ["LOCAL N/A | BUDGET N/A GIB", "DED CAP 0.0 | SHARED 0.0 GIB"],
        );
        assert_eq!(
            video_memory_lines(Some((0, 1 << 30)), None, None),
            ["LOCAL 0.0 | BUDGET 1.0 GIB", "DED CAP N/A | SHARED N/A GIB"],
        );
    }

    #[test]
    fn a_box_is_painted() {
        let lines = vec!["I".to_owned(), " i".to_owned()];
        for scale in [1, 2, 4] {
            let (width, height) = size(scale);
            let pitch = (width * 4).next_multiple_of(256) as usize;
            let colours = colours(DXGI_FORMAT_R10G10B10A2_UNORM).unwrap();
            let picture = painted(&lines, scale, colours, pitch);
            assert_eq!(picture.len(), pitch * height as usize);
            let at = |x: u32, y: u32| &picture[y as usize * pitch + x as usize * 4..][..4];
            // The rim is black, the first letter's top row lit in its middle.
            assert_eq!(at(0, 0), colours[0]);
            assert_eq!(at((RIM + 2) * scale, RIM * scale), colours[1]);
            assert_eq!(at((RIM + 2) * scale + scale - 1, RIM * scale + scale - 1), colours[1]);
            assert_eq!(at(RIM * scale, RIM * scale), colours[0]);
            // Small letters are shown as capitals, a line and a cell on.
            assert_eq!(at((RIM + CELL.0 + 2) * scale, (RIM + CELL.1) * scale), colours[1]);
            // Nothing past a line's end.
            assert_eq!(at((RIM + CELL.0 * 2 + 2) * scale, (RIM + CELL.1) * scale), colours[0]);
        }
    }

    #[test]
    fn counters_that_went_back_to_nothing_count_from_there() {
        let (mut a, mut b) = (Counters::default(), Counters::default());
        (a.0[0], b.0[0]) = (10, 30);
        (a.0[1], b.0[1]) = (50, 20);
        let c = b.since(a, 10.0);
        assert_eq!((c[0], c[1]), (2.0, 2.0));
    }
}

#[cfg(test)]
mod busy_tests {
    use super::{
        COLUMNS, CpuMeter, CpuTimes, MAX_CPUS, STATUS_INFO_LENGTH_MISMATCH,
        SystemProcessorPerformanceInformation, busy_pct, cpu_line, glyph, query_cpu_times_with,
    };

    const fn ct(idle: i64, kernel: i64, user: i64) -> CpuTimes {
        CpuTimes { idle, kernel, user }
    }

    #[test]
    fn busy_is_the_share_of_time_not_idle() {
        assert_eq!(busy_pct(ct(0, 0, 0), ct(25, 60, 40)), 75.0);
        assert_eq!(busy_pct(ct(0, 0, 0), ct(100, 100, 0)), 0.0);
        assert_eq!(busy_pct(ct(0, 0, 0), ct(0, 50, 50)), 100.0);
    }

    #[test]
    fn counters_that_step_back_do_not_panic() {
        // Kernel counter steps back.
        assert_eq!(busy_pct(ct(10, 156_250, 0), ct(10, 0, 0)), 0.0);
        // Both idle and kernel step back.
        assert_eq!(busy_pct(ct(500, 1_000, 0), ct(400, 900, 0)), 0.0);
        // Idle steps back while kernel advances: invalid interval reads as 0% idle, not 100%.
        assert_eq!(busy_pct(ct(500, 1_000, 0), ct(400, 1_100, 0)), 0.0);
        // User counter steps back even when kernel + user advances.
        assert_eq!(busy_pct(ct(10, 500, 500), ct(10, 700, 400)), 0.0);
        // Zero total delta reads as 0%.
        assert_eq!(busy_pct(ct(10, 100, 50), ct(10, 100, 50)), 0.0);
        // Idle delta larger than total delta clamps cleanly to 0%.
        assert_eq!(busy_pct(ct(0, 100, 0), ct(200, 150, 0)), 0.0);
        // Extreme i64 values must saturate without overflow panics.
        assert_eq!(
            busy_pct(ct(i64::MAX, i64::MIN, 0), ct(i64::MIN, i64::MAX, 0)),
            0.0
        );
        let wide = busy_pct(ct(i64::MIN, i64::MIN, 0), ct(i64::MAX, i64::MAX, 0));
        assert!((0.0..=100.0).contains(&wide));
    }

    #[test]
    fn cpu_meter_handles_core_step_back_and_formats_line() {
        let mut meter = CpuMeter::default();
        assert_eq!(cpu_line(meter.total_pct, meter.peak_core_pct), "CPU N/A");
        meter.update(vec![ct(10, 156_250, 0), ct(0, 100, 0)]);
        assert_eq!(meter.total_pct, None);
        meter.update(vec![ct(10, 0, 0), ct(20, 160, 40)]);
        assert_eq!(meter.total_pct, Some(40.0));
        assert_eq!(meter.peak_core_pct, Some(80.0));
        let line = cpu_line(meter.total_pct, meter.peak_core_pct);
        assert_eq!(line, "CPU 40%  PEAK CORE 80%");
        assert!(line.len() <= COLUMNS as usize);
        assert!(line.bytes().all(|b| glyph(b.to_ascii_uppercase()) != glyph(b'#')));
    }

    #[test]
    fn cpu_query_grows_across_processor_groups_and_caps_cleanly() {
        let entry_size = size_of::<SystemProcessorPerformanceInformation>();
        let total_cpus = 128usize; // Spans two 64-core processor groups.
        let mut attempts = 0usize;
        let samples = query_cpu_times_with(64, |buf, returned| {
            attempts += 1;
            let needed = (total_cpus * entry_size) as u32;
            *returned = needed;
            if buf.len() < total_cpus {
                return STATUS_INFO_LENGTH_MISMATCH;
            }
            for (i, slot) in buf[..total_cpus].iter_mut().enumerate() {
                slot.idle_time = i as i64;
                slot.kernel_time = (i * 2) as i64;
                slot.user_time = (i * 3) as i64;
            }
            0
        });
        assert_eq!(attempts, 2);
        assert_eq!(samples.len(), total_cpus);
        assert_eq!(samples[127], ct(127, 254, 381));

        // Requests larger than MAX_CPUS terminate cleanly without looping forever.
        let mut cap_attempts = 0usize;
        let overflow = query_cpu_times_with(64, |_buf, returned| {
            cap_attempts += 1;
            *returned = ((MAX_CPUS + 64) * entry_size) as u32;
            STATUS_INFO_LENGTH_MISMATCH
        });
        assert!(overflow.is_empty());
        assert!(cap_attempts <= 3);
    }
}
