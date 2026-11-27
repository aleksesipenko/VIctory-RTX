//! Process-wide state the implementations need.

use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

static MAIN_IMAGE_BASE: AtomicUsize = AtomicUsize::new(0);

/// Records where the title's main image is mapped. The title expects
/// `GetModuleHandle(NULL)` and friends to return this, not our own exe.
pub fn set_main_image(base: usize) {
    MAIN_IMAGE_BASE.store(base, Ordering::Release);
    crate::image_watch::start(base);
}

pub fn main_image() -> Option<usize> {
    match MAIN_IMAGE_BASE.load(Ordering::Acquire) {
        0 => None,
        base => Some(base),
    }
}

/// Original file of each title image we mapped, by base address.
static IMAGE_PATHS: Mutex<Vec<(usize, PathBuf)>> = Mutex::new(Vec::new());

/// Records the file a title image was mapped from: the title asks for its
/// own path (`GetModuleFileName`) to find its data, and must not get our
/// cache copy.
pub fn set_image_path(base: usize, path: &Path) {
    let path = std::path::absolute(path).unwrap_or_else(|_| path.to_owned());
    let mut paths = IMAGE_PATHS.lock().unwrap_or_else(|e| e.into_inner());
    paths.retain(|(b, _)| *b != base);
    paths.push((base, path));
}

/// Forgets a title image that was unmapped again.
pub fn forget_image(base: usize) {
    let mut paths = IMAGE_PATHS.lock().unwrap_or_else(|e| e.into_inner());
    paths.retain(|(b, _)| *b != base);
}

pub fn image_path(base: usize) -> Option<PathBuf> {
    let paths = IMAGE_PATHS.lock().unwrap_or_else(|e| e.into_inner());
    paths
        .iter()
        .find(|(b, _)| *b == base)
        .map(|(_, p)| p.clone())
}

/// The title image whose original file is called `file` (any case).
pub fn image_named(file: &str) -> Option<usize> {
    let paths = IMAGE_PATHS.lock().unwrap_or_else(|e| e.into_inner());
    paths
        .iter()
        .find(|(_, p)| p.file_name().is_some_and(|n| n.eq_ignore_ascii_case(file)))
        .map(|(b, _)| *b)
}

static TITLE_ARGS: std::sync::OnceLock<Vec<String>> = std::sync::OnceLock::new();

/// The arguments the title's `main` gets (those after `--`). Set once, before
/// the title starts; later calls are ignored.
pub fn set_title_args(args: Vec<String>) {
    let _ = TITLE_ARGS.set(args);
}

pub fn title_args() -> &'static [String] {
    TITLE_ARGS.get().map_or(&[], Vec::as_slice)
}

/// Maps a title DLL the title loads itself (`LoadLibrary`) and returns its
/// base; `Ok(None)` if `file` is not one of the title's DLLs.
pub type TitleDllLoader = Box<dyn Fn(&str) -> Result<Option<usize>, String> + Send + Sync>;

static TITLE_DLL_LOADER: std::sync::OnceLock<TitleDllLoader> = std::sync::OnceLock::new();

/// Set once, by whoever mapped the title (it knows the folders and the
/// resolver); later calls are ignored.
pub fn set_title_dll_loader(loader: TitleDllLoader) {
    let _ = TITLE_DLL_LOADER.set(loader);
}

pub fn load_title_dll(file: &str) -> Result<Option<usize>, String> {
    TITLE_DLL_LOADER.get().map_or(Ok(None), |load| load(file))
}

/// What the title's manifest (`MicrosoftGame.config` beside its executable)
/// says about it, and where its persistent local storage lives.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Title {
    pub id: u32,
    /// The store's product id (empty if the manifest has none).
    pub store_id: String,
    /// Persistent local storage: our folder for it and the title's quota.
    /// Game saves live beside it ([`Title::saves`]).
    pub storage: PathBuf,
    pub storage_bytes: u64,
}

impl Title {
    /// Id, store id and storage quota from the manifest (zero or empty
    /// where it has none); the storage folder is left for the caller.
    pub fn from_manifest(text: &str) -> Title {
        let elements = elements(text);
        let get = |path: &str| {
            elements
                .iter()
                .find(|(p, _)| p == path)
                .map(|(_, text)| text.as_str())
        };
        let id = get("Game/TitleId").and_then(|v| u32::from_str_radix(v, 16).ok());
        let storage_mb = get("Game/PersistentLocalStorage/SizeMB").and_then(|v| v.parse().ok());
        Title {
            id: id.unwrap_or(0),
            store_id: get("Game/StoreId").unwrap_or_default().to_owned(),
            storage: PathBuf::new(),
            storage_bytes: storage_mb.unwrap_or(0) << 20,
        }
    }

    /// Where game saves live: next to persistent local storage, under the
    /// data directory, never the cache.
    pub fn saves(&self) -> PathBuf {
        self.storage.with_file_name("saves")
    }
}

/// The trimmed text of one element: comments inside it are skipped, not
/// treated as the end of its value. A manifest generator can write
/// `<TitleId><!-- generated -->639AA8DD</TitleId>`, which must still give
/// the title's id rather than an empty field.
fn field_text(rest: &str) -> String {
    let mut out = String::new();
    let mut rest = rest;
    loop {
        let Some(open) = rest.find('<') else {
            out.push_str(rest);
            break;
        };
        out.push_str(&rest[..open]);
        if rest[open..].starts_with("<!--") {
            let Some(end) = rest[open..].find("-->") else {
                break;
            };
            rest = &rest[open + end + 3..];
        } else {
            // A child element, a closing tag or anything else ends the text.
            break;
        }
    }
    out.trim().to_owned()
}

/// Each element of an XML text as its path from the root (`Game/TitleId`)
/// and its trimmed text up to the first child, the predefined entities
/// decoded; comments, CDATA sections, declarations and attributes are
/// skipped.
fn elements(text: &str) -> Vec<(String, String)> {
    let mut out = Vec::new();
    let mut path: Vec<&str> = Vec::new();
    let mut rest = text;
    while let Some(open) = rest.find('<') {
        rest = &rest[open..];
        let end = if rest.starts_with("<!--") {
            rest.find("-->").map(|e| e + 3)
        } else if rest.starts_with("<![CDATA[") {
            rest.find("]]>").map(|e| e + 3)
        } else {
            tag_end(rest)
        };
        let Some(end) = end else { break };
        let tag = &rest[1..end - 1];
        rest = &rest[end..];
        if tag.starts_with(['!', '?']) {
            continue;
        }
        if let Some(name) = tag.strip_prefix('/') {
            if let Some(at) = path.iter().rposition(|n| *n == name.trim()) {
                path.truncate(at);
            }
            continue;
        }
        let closed = tag.ends_with('/');
        let name = tag
            .trim_end_matches('/')
            .split_whitespace()
            .next()
            .unwrap_or("");
        path.push(name);
        let body = field_text(rest);
        let body = [("&lt;", "<"), ("&gt;", ">"), ("&quot;", "\""), ("&apos;", "'"), ("&amp;", "&")]
            .iter()
            .fold(body, |b, (entity, c)| b.replace(entity, c));
        out.push((path.join("/"), body));
        if closed {
            path.pop();
        }
    }
    out
}

/// Where the tag that starts `text` ends, past its `>`; a `>` in a quoted
/// attribute value is part of the value (XML allows it there unescaped).
fn tag_end(text: &str) -> Option<usize> {
    let mut quote = None;
    for (at, c) in text.char_indices() {
        match (quote, c) {
            (None, '"' | '\'') => quote = Some(c),
            (Some(open), _) if c == open => quote = None,
            (None, '>') => return Some(at + 1),
            _ => {}
        }
    }
    None
}

static TITLE: Mutex<Option<Title>> = Mutex::new(None);

pub fn set_title(title: Title) {
    *TITLE.lock().unwrap_or_else(|e| e.into_inner()) = Some(title);
}

/// The running title; the default (id 0, no storage) until set.
pub fn title() -> Title {
    TITLE
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .clone()
        .unwrap_or_default()
}

/// Gives the process the console's millisecond timers. Titles pace
/// themselves with short waits (the title's audio mixer sleeps
/// about 4 ms per 5.3 ms block) and cannot tune the timer on the console;
/// a desktop process waits in 15.6 ms ticks unless it asks for finer ones,
/// and Windows 11 drops that request while the window is hidden unless the
/// process opts out. Process-wide, for the whole run.
#[cfg(windows)]
pub fn use_console_timers() {
    use windows_sys::Win32::Media::timeBeginPeriod;
    use windows_sys::Win32::System::Threading::{
        GetCurrentProcess, PROCESS_POWER_THROTTLING_CURRENT_VERSION, PROCESS_POWER_THROTTLING_EXECUTION_SPEED,
        PROCESS_POWER_THROTTLING_IGNORE_TIMER_RESOLUTION, PROCESS_POWER_THROTTLING_STATE,
        ProcessPowerThrottling, SetProcessInformation,
    };
    // Also: the system does not run this process at reduced speed (its
    // "efficiency mode": lower clock, the slow cores of a hybrid processor)
    // when its window is not the one in front, as while its owner works in
    // another window or streams. The console has eight cores alike and no
    // such mode. `VICTORY_EFFICIENCY_MODE=1` leaves the system its choice.
    let execution = if std::env::var_os("VICTORY_EFFICIENCY_MODE").is_some_and(|v| v == "1") {
        0
    } else {
        PROCESS_POWER_THROTTLING_EXECUTION_SPEED
    };
    let keep = PROCESS_POWER_THROTTLING_STATE {
        Version: PROCESS_POWER_THROTTLING_CURRENT_VERSION,
        ControlMask: PROCESS_POWER_THROTTLING_IGNORE_TIMER_RESOLUTION | execution,
        StateMask: 0,
    };
    // SAFETY: plain process settings; `keep` outlives the call.
    unsafe {
        SetProcessInformation(
            GetCurrentProcess(),
            ProcessPowerThrottling,
            (&raw const keep).cast(),
            size_of_val(&keep) as u32,
        );
        timeBeginPeriod(1);
    }
    ahead_of_other_programs();
    keep_resident();
}

/// The title's streaming thread runs below normal priority (on the console
/// nobody else wants its core) and does all of the title's loading: a
/// program that keeps the processors busy at normal priority (a build, a
/// virus scan) starved it, loading took 40 to 60 s instead of 28 to 33, and
/// the opening ran without the models that were to be there (runs cpu-1,
/// play-1125-0203). The process runs above normal, as long as it runs.
/// `VICTORY_PROCESS_PRIORITY=normal`: as it was started.
#[cfg(windows)]
fn ahead_of_other_programs() {
    use windows_sys::Win32::System::Threading::{ABOVE_NORMAL_PRIORITY_CLASS, GetCurrentProcess, SetPriorityClass};
    if std::env::var_os("VICTORY_PROCESS_PRIORITY").is_some_and(|v| v == "normal" || v == "0") {
        return;
    }
    // SAFETY: a plain process setting.
    let set = unsafe { SetPriorityClass(GetCurrentProcess(), ABOVE_NORMAL_PRIORITY_CLASS) } != 0;
    tracing::info!(set, "the process runs above normal priority (VICTORY_PROCESS_PRIORITY=normal: not)");
}

#[cfg(windows)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct WorkingSetPlan {
    target: u64,
    floor: u64,
    hard: bool,
    soft_fallback: u64,
}

#[cfg(windows)]
fn working_set_plan(
    total_phys: u64,
    hard_env: Option<&str>,
    min_mb_env: Option<&str>,
) -> Option<WorkingSetPlan> {
    const MIB: u64 = 1 << 20;
    const GIB: u64 = 1 << 30;
    const MIN_EXPLICIT: u64 = 64 * MIB;

    if total_phys < 2 * GIB {
        return None;
    }
    let hard = match hard_env.map(str::trim) {
        Some("0") => false,
        Some("1") => true,
        _ => total_phys >= 24 * GIB,
    };
    let explicit_mb = min_mb_env.and_then(|v| v.trim().parse::<u64>().ok());
    match explicit_mb {
        Some(0) => None,
        Some(mb) => {
            let requested = mb.saturating_mul(MIB).max(MIN_EXPLICIT).min(total_phys);
            Some(WorkingSetPlan {
                target: requested,
                floor: MIN_EXPLICIT.min(requested),
                hard,
                soft_fallback: (total_phys / 2).min(requested),
            })
        }
        None => {
            let target = if hard {
                total_phys.saturating_sub(10 * GIB).max(total_phys / 2)
            } else {
                total_phys / 2
            };
            Some(WorkingSetPlan {
                target,
                floor: GIB,
                hard,
                soft_fallback: (total_phys / 2).min(target),
            })
        }
    }
}

/// Keeps the process working set resident against Windows standby-list
/// trimming (`VICTORY_HARD_WORKING_SET`, default on with >= 24 GiB RAM,
/// `=0` for a soft minimum; `VICTORY_MIN_WORKING_SET_MB=0` disables).
/// Provenance (GPL-3.0): kitty-core/Kitty6 a64da00.
#[cfg(windows)]
fn keep_resident() {
    use windows_sys::Win32::System::Memory::{
        QUOTA_LIMITS_HARDWS_MAX_DISABLE, QUOTA_LIMITS_HARDWS_MIN_DISABLE,
        QUOTA_LIMITS_HARDWS_MIN_ENABLE, SetProcessWorkingSetSizeEx,
    };
    use windows_sys::Win32::System::ProcessStatus::{
        K32GetPerformanceInfo, PERFORMANCE_INFORMATION,
    };
    use windows_sys::Win32::System::Threading::GetCurrentProcess;

    const GIB: u64 = 1 << 30;
    let mut perf = PERFORMANCE_INFORMATION {
        cb: size_of::<PERFORMANCE_INFORMATION>() as u32,
        ..Default::default()
    };
    // SAFETY: valid sized PERFORMANCE_INFORMATION struct.
    if unsafe { K32GetPerformanceInfo(&mut perf, perf.cb) } == 0 {
        return;
    }
    let total_phys = (perf.PhysicalTotal as u64).saturating_mul(perf.PageSize as u64);
    let hard_env = std::env::var("VICTORY_HARD_WORKING_SET").ok();
    let min_mb_env = std::env::var("VICTORY_MIN_WORKING_SET_MB").ok();
    let Some(plan) = working_set_plan(total_phys, hard_env.as_deref(), min_mb_env.as_deref())
    else {
        return;
    };
    let max = total_phys;
    let try_set = |mut want: u64, floor: u64, is_hard: bool| -> Option<u64> {
        let flags = if is_hard {
            QUOTA_LIMITS_HARDWS_MIN_ENABLE
        } else {
            QUOTA_LIMITS_HARDWS_MIN_DISABLE
        } | QUOTA_LIMITS_HARDWS_MAX_DISABLE;
        while want >= floor {
            // SAFETY: plain process working-set quota setting.
            let ok = unsafe {
                SetProcessWorkingSetSizeEx(GetCurrentProcess(), want as usize, max as usize, flags)
            } != 0;
            if ok {
                return Some(want);
            }
            if want == floor {
                break;
            }
            let next = if want > 2 * GIB {
                want.saturating_sub(2 * GIB)
            } else {
                (want / 4) * 3
            };
            want = if next < floor { floor } else { next };
        }
        None
    };
    if let Some(min_bytes) = try_set(plan.target, plan.floor, plan.hard) {
        tracing::info!(
            min_mb = min_bytes >> 20,
            hard = plan.hard,
            "minimum working set configured (VICTORY_HARD_WORKING_SET=0: soft; VICTORY_MIN_WORKING_SET_MB=0: off)"
        );
        return;
    }
    if plan.hard && let Some(min_bytes) = try_set(plan.soft_fallback, plan.floor, false) {
        tracing::warn!(
            min_mb = min_bytes >> 20,
            "hard minimum working set refused; fell back to soft minimum"
        );
        return;
    }
    tracing::warn!(
        floor_mb = plan.floor >> 20,
        "Windows refused minimum working set down to floor"
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_quoted_angle_bracket_does_not_end_its_tag() {
        let text = "<Game><ShellVisuals DefaultDisplayName=\"A > B\" Description='x>y'/>\
                    <TitleId>639AA8DD</TitleId>\
                    <PersistentLocalStorage><SizeMB>64</SizeMB></PersistentLocalStorage></Game>";
        let t = Title::from_manifest(text);
        assert_eq!((t.id, t.storage_bytes), (0x639A_A8DD, 64 << 20));
    }

    #[test]
    fn reads_title_id_store_id_and_storage_quota() {
        let text = "\u{feff}<?xml version=\"1.0\"?>\n<Game configVersion=\"1\">\
                    <!-- <TitleId>00000000</TitleId> -->\
                    <Identity Name=\"x\"/>\n  <TitleId>639AA8DD</TitleId>\n\
                    <![CDATA[ <TitleId>1</TitleId> ]]>                    <StoreId>9&amp;C</StoreId><Other><SizeMB>7</SizeMB></Other>\
                    <PersistentLocalStorage>\n  <SizeMB>1024</SizeMB> <!-- c -->\n\
                    </PersistentLocalStorage></Game>";
        let t = Title::from_manifest(text);
        assert_eq!((t.id, t.storage_bytes), (0x639A_A8DD, 1024 << 20));
        assert_eq!(t.store_id, "9&C");
        assert_eq!(Title::from_manifest("").storage_bytes, 0);
    }

    #[test]
    fn inline_comments_do_not_truncate_a_field() {
        let text = "<Game>\
            <TitleId><!-- generated -->639AA8DD</TitleId>\
            <StoreId>9&amp;<!-- keep -->C</StoreId>\
            <PersistentLocalStorage><SizeMB><!-- n -->1024</SizeMB></PersistentLocalStorage>\
            </Game>";
        let t = Title::from_manifest(text);
        assert_eq!(t.id, 0x639A_A8DD);
        assert_eq!(t.store_id, "9&C");
        assert_eq!(t.storage_bytes, 1024 << 20);
    }

    #[cfg(windows)]
    #[test]
    fn explicit_small_working_set_never_inflates_fallback() {
        const MIB: u64 = 1 << 20;
        const GIB: u64 = 1 << 30;
        let phys32 = 32 * GIB;
        // Default on 32 GiB: hard 22 GiB, soft fallback 16 GiB, floor 1 GiB.
        assert_eq!(
            working_set_plan(phys32, None, None),
            Some(WorkingSetPlan {
                target: 22 * GIB,
                floor: GIB,
                hard: true,
                soft_fallback: 16 * GIB,
            })
        );
        // Explicit sub-1 GiB request (256 MiB): honored directly, floor 64 MiB,
        // soft fallback capped at 256 MiB (never inflated to 16 GiB).
        assert_eq!(
            working_set_plan(phys32, Some("1"), Some("256")),
            Some(WorkingSetPlan {
                target: 256 * MIB,
                floor: 64 * MIB,
                hard: true,
                soft_fallback: 256 * MIB,
            })
        );
        // Explicit 4 GiB request on 32 GiB: soft fallback never exceeds 4 GiB.
        assert_eq!(
            working_set_plan(phys32, Some("1"), Some("4096")),
            Some(WorkingSetPlan {
                target: 4 * GIB,
                floor: 64 * MIB,
                hard: true,
                soft_fallback: 4 * GIB,
            })
        );
        // Explicit 0 disables working-set pinning.
        assert_eq!(working_set_plan(phys32, Some("1"), Some("0")), None);
    }
}
