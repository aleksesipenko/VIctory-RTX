//! Host input arbitration: combines DualSense HID controllers (wired USB,
//! Pico 2 W `DS5Dongle`, and Bluetooth `0x31`) with XInput controllers across
//! host slots `0..4`, deduplicating Steam Input (`0x28DE`) / ViGEm virtual
//! XInput mirrors while preserving genuine Xbox + PlayStation multi-pad play.

use victory_core::config::{InputConfig, VirtualXInputPolicy};

use super::dualsense::{DualSenseManager, MAX_DUALSENSE, SlotSnapshot, XInputTopology};
use super::{BatteryState, KeyState, PADS, PadReading, PadState, Source};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BackendSlot {
    DualSense(usize),
    XInput(usize),
}

/// Sustained active matching polls AND truly distinct non-zero quantized
/// states required before an XInput slot is classified as a virtual mirror of
/// a specific DualSense when both virtual and physical XInput devices exist.
const MIRROR_SUPPRESS_POLLS: u16 = 25;
const MIRROR_DISTINCT_STATES: usize = 3;

/// Decides whether a specific XInput slot should be suppressed to prevent
/// double-input while a DualSense HID controller is active.
///
/// Under `Auto`:
/// - When no virtual XInput wrapper (Steam Input / ViGEm) is enumerated,
///   physical XInput slots are never heuristically suppressed.
/// - When only virtual XInput interfaces are enumerated (no physical XInput
///   device present), virtual XInput is suppressed immediately.
/// - When both virtual and physical XInput interfaces are present, only slots
///   with sustained multi-state correlation to a DualSense pad are suppressed.
pub fn should_suppress_xinput(
    policy: VirtualXInputPolicy,
    any_dualsense_active: bool,
    topology: XInputTopology,
    slot_mirrors_dualsense: bool,
) -> bool {
    if !any_dualsense_active {
        return false;
    }
    match policy {
        VirtualXInputPolicy::Always => true,
        VirtualXInputPolicy::Never => false,
        VirtualXInputPolicy::Auto => {
            if !topology.virtual_present {
                false
            } else if topology.only_virtual() {
                true
            } else {
                slot_mirrors_dualsense
            }
        }
    }
}

fn pad_has_activity(p: &PadState) -> bool {
    p.buttons != 0
        || p.left_trigger > 0.1
        || p.right_trigger > 0.1
        || p.left_x.abs() > 0.25
        || p.left_y.abs() > 0.25
        || p.right_x.abs() > 0.25
        || p.right_y.abs() > 0.25
}

fn pads_mirror(a: &PadState, b: &PadState) -> bool {
    const TOL: f32 = 0.08;
    a.buttons == b.buttons
        && (a.left_trigger - b.left_trigger).abs() <= TOL
        && (a.right_trigger - b.right_trigger).abs() <= TOL
        && (a.left_x - b.left_x).abs() <= TOL
        && (a.left_y - b.left_y).abs() <= TOL
        && (a.right_x - b.right_x).abs() <= TOL
        && (a.right_y - b.right_y).abs() <= TOL
}

/// Coarse quantized fingerprint of an active `PadState` used to track genuinely
/// distinct non-zero input states (so alternating `A -> B -> A` only counts as
/// 2 unique states rather than 3 transitions).
type QuantizedPad = (u32, u8, u8, i8, i8, i8, i8);

fn pad_fingerprint(p: &PadState) -> QuantizedPad {
    let trig = |v: f32| (v.clamp(0.0, 1.0) * 8.0).round() as u8;
    let axis = |v: f32| (v.clamp(-1.0, 1.0) * 4.0).round() as i8;
    (
        p.buttons,
        trig(p.left_trigger),
        trig(p.right_trigger),
        axis(p.left_x),
        axis(p.left_y),
        axis(p.right_x),
        axis(p.right_y),
    )
}

#[derive(Clone, Copy, Default)]
struct MirrorTracker {
    source: Option<usize>,
    matching_polls: u16,
    seen_states: [Option<QuantizedPad>; MIRROR_DISTINCT_STATES],
    seen_count: u8,
    suppressed: bool,
}

impl MirrorTracker {
    fn record_unique_state(&mut self, fp: QuantizedPad) {
        let count = usize::from(self.seen_count);
        if self.seen_states[..count].contains(&Some(fp)) {
            return;
        }
        if count < self.seen_states.len() {
            self.seen_states[count] = Some(fp);
            self.seen_count += 1;
        }
    }
}

/// Assigns DualSense and XInput devices to persistent host slots `0..PADS` so
/// hot-plugging a second controller never swaps Player 1's slot index.
#[derive(Default)]
pub struct SlotArbitrator {
    owners: [Option<BackendSlot>; PADS],
    mirrors: [MirrorTracker; PADS],
}

impl SlotArbitrator {
    pub fn owner(&self, host_slot: usize) -> Option<BackendSlot> {
        self.owners.get(host_slot).copied().flatten()
    }

    pub fn resolve(
        &mut self,
        ds: &[Option<SlotSnapshot>; MAX_DUALSENSE],
        xinput: &[Option<PadState>; PADS],
        policy: VirtualXInputPolicy,
        topology: XInputTopology,
    ) -> [Option<PadReading>; PADS] {
        let any_ds = ds.iter().any(Option::is_some);
        let need_correlation = any_ds && topology.virtual_present && topology.physical_present;

        // Per-slot sustained unique-state correlation (only active when BOTH a
        // virtual XInput wrapper and a physical XInput device are enumerated):
        // - Requires >= 25 continuous matching active polls AND >= 3 genuinely
        //   distinct non-zero quantized states (`seen_states`), so alternating
        //   `A -> B -> A` never counts as 3 states.
        // - While unsuppressed, any idle frame resets `matching_polls` and
        //   `seen_count` to 0 so unrelated taps separated by rest never accumulate.
        // - Once suppressed, an already-confirmed mirror stays suppressed across
        //   idle pauses until the first diverging active poll immediately clears
        //   suppression.
        for (xi, xpad) in xinput.iter().enumerate() {
            let Some(xp) = *xpad else {
                self.mirrors[xi] = MirrorTracker::default();
                continue;
            };
            if !need_correlation {
                self.mirrors[xi] = MirrorTracker::default();
                continue;
            }
            let m = &mut self.mirrors[xi];
            let xp_active = pad_has_activity(&xp);
            if let Some(di) = m.source {
                match ds.get(di).and_then(|s| *s) {
                    Some(d) => {
                        let ds_active = pad_has_activity(&d.pad);
                        if !xp_active && !ds_active {
                            if !m.suppressed {
                                *m = MirrorTracker::default();
                            }
                        } else if xp_active && ds_active {
                            if pads_mirror(&d.pad, &xp) {
                                m.matching_polls = m.matching_polls.saturating_add(1);
                                m.record_unique_state(pad_fingerprint(&xp));
                                if m.matching_polls >= MIRROR_SUPPRESS_POLLS
                                    && usize::from(m.seen_count) >= MIRROR_DISTINCT_STATES
                                {
                                    m.suppressed = true;
                                }
                            } else {
                                *m = MirrorTracker::default();
                            }
                        } else {
                            // Exactly one controller is above the activity threshold while
                            // the other is idle (e.g. `(0.24, 0.26)`): never record/count as
                            // an active match. Clear immediately on divergence or while unsuppressed.
                            if !pads_mirror(&d.pad, &xp) || !m.suppressed {
                                *m = MirrorTracker::default();
                            }
                        }
                    }
                    None => {
                        *m = MirrorTracker::default();
                    }
                }
            } else if xp_active
                && let Some((di, _)) = ds.iter().enumerate().find(|(_, d)| {
                    d.is_some_and(|s| pad_has_activity(&s.pad) && pads_mirror(&s.pad, &xp))
                })
            {
                let mut tracker = MirrorTracker {
                    source: Some(di),
                    matching_polls: 1,
                    seen_states: [None; MIRROR_DISTINCT_STATES],
                    seen_count: 0,
                    suppressed: false,
                };
                tracker.record_unique_state(pad_fingerprint(&xp));
                *m = tracker;
            }
        }

        // Release slots whose backing device disconnected or became suppressed.
        for slot in &mut self.owners {
            let keep = match *slot {
                None => false,
                Some(BackendSlot::DualSense(i)) => ds.get(i).and_then(|x| *x).is_some(),
                Some(BackendSlot::XInput(i)) => {
                    let present = xinput.get(i).and_then(|x| *x).is_some();
                    present
                        && !should_suppress_xinput(
                            policy,
                            any_ds,
                            topology,
                            self.mirrors[i].suppressed,
                        )
                }
            };
            if !keep {
                *slot = None;
            }
        }

        // Bind newly connected DualSense pads first.
        for (i, snap) in ds.iter().enumerate() {
            if snap.is_some()
                && !self.owners.contains(&Some(BackendSlot::DualSense(i)))
                && let Some(free) = self.owners.iter_mut().find(|o| o.is_none())
            {
                *free = Some(BackendSlot::DualSense(i));
            }
        }

        // Bind genuine (non-suppressed) XInput pads to remaining free slots.
        for (i, pad) in xinput.iter().enumerate() {
            if pad.is_some()
                && !should_suppress_xinput(policy, any_ds, topology, self.mirrors[i].suppressed)
                && !self.owners.contains(&Some(BackendSlot::XInput(i)))
                && let Some(free) = self.owners.iter_mut().find(|o| o.is_none())
            {
                *free = Some(BackendSlot::XInput(i));
            }
        }

        std::array::from_fn(|slot| match self.owners[slot]? {
            BackendSlot::DualSense(i) => {
                let s = ds[i]?;
                Some(PadReading {
                    pad: s.pad,
                    battery: s.battery,
                    system_buttons: s.system_buttons,
                })
            }
            BackendSlot::XInput(i) => Some(PadReading {
                pad: xinput[i]?,
                battery: BatteryState::unknown(),
                system_buttons: 0,
            }),
        })
    }
}

#[cfg(windows)]
pub struct Host {
    config: InputConfig,
    dualsense: DualSenseManager,
    arbitrator: SlotArbitrator,
    empty_until: [Option<std::time::Instant>; PADS],
    resolved: [Option<PadReading>; PADS],
    was_focused: bool,
}

#[cfg(windows)]
impl Default for Host {
    fn default() -> Self {
        let cfg = InputConfig::default();
        Self {
            dualsense: DualSenseManager::new(cfg.dualsense.clone()),
            config: cfg,
            arbitrator: SlotArbitrator::default(),
            empty_until: [None; PADS],
            resolved: [None; PADS],
            was_focused: true,
        }
    }
}

#[cfg(windows)]
impl Host {
    fn refresh(&mut self) {
        let focused = imp::focused();
        let active_focus = !self.config.mute_unfocused || focused;
        if self.was_focused && !active_focus {
            for xi in 0..PADS {
                imp::xinput_rumble(xi, 0, 0);
            }
        }
        self.was_focused = active_focus;

        let ds = self.dualsense.poll(active_focus);
        let xinput: [Option<PadState>; PADS] = std::array::from_fn(|xi| {
            let mut pad = imp::poll_xinput(xi, &mut self.empty_until[xi])?;
            if !active_focus {
                pad = PadState::default();
            }
            Some(pad)
        });
        self.resolved = self.arbitrator.resolve(
            &ds,
            &xinput,
            self.config.suppress_virtual_xinput,
            super::dualsense::xinput_topology(),
        );
    }
}

#[cfg(windows)]
impl Source for Host {
    fn begin_poll(&mut self) {
        self.refresh();
    }

    fn pad(&mut self, slot: usize) -> Option<PadReading> {
        self.resolved.get(slot).copied().flatten()
    }

    fn keys(&mut self) -> Vec<KeyState> {
        imp::poll_keys()
    }

    #[cfg(any(test, feature = "diagnostics"))]
    fn scripted_keys(&mut self) -> Vec<KeyState> {
        #[cfg(feature = "diagnostics")]
        {
            imp::poll_scripted_keys()
        }
        #[cfg(not(feature = "diagnostics"))]
        {
            Vec::new()
        }
    }

    fn rumble(&mut self, slot: usize, motors: [f32; 4]) {
        let active_focus = !self.config.mute_unfocused || imp::focused();
        let motors = if active_focus { motors } else { [0.0; 4] };
        match self.arbitrator.owner(slot) {
            Some(BackendSlot::DualSense(i)) => self.dualsense.set_rumble(i, motors),
            Some(BackendSlot::XInput(i)) => {
                let speed = |v: f32| (v.clamp(0.0, 1.0) * f32::from(u16::MAX)) as u16;
                imp::xinput_rumble(i, speed(motors[0]), speed(motors[1]));
            }
            None => {}
        }
    }

    fn set_config(&mut self, config: InputConfig) {
        self.dualsense.set_config(config.dualsense.clone());
        self.config = config;
    }
}

#[cfg(windows)]
mod imp {
    #[cfg(feature = "diagnostics")]
    use std::sync::LazyLock;
    use std::time::{Duration, Instant};

    use windows_sys::Win32::System::Threading::GetCurrentProcessId;
    use windows_sys::Win32::UI::Input::KeyboardAndMouse::{
        GetAsyncKeyState, MAPVK_VK_TO_VSC_EX, MapVirtualKeyW,
    };
    use windows_sys::Win32::UI::Input::XboxController::{
        XINPUT_GAMEPAD, XINPUT_GAMEPAD_A, XINPUT_GAMEPAD_B, XINPUT_GAMEPAD_BACK,
        XINPUT_GAMEPAD_DPAD_DOWN, XINPUT_GAMEPAD_DPAD_LEFT, XINPUT_GAMEPAD_DPAD_RIGHT,
        XINPUT_GAMEPAD_DPAD_UP, XINPUT_GAMEPAD_LEFT_SHOULDER, XINPUT_GAMEPAD_LEFT_THUMB,
        XINPUT_GAMEPAD_RIGHT_SHOULDER, XINPUT_GAMEPAD_RIGHT_THUMB, XINPUT_GAMEPAD_START,
        XINPUT_GAMEPAD_X, XINPUT_GAMEPAD_Y, XINPUT_STATE, XINPUT_VIBRATION, XInputGetState,
        XInputSetState,
    };
    use windows_sys::Win32::UI::WindowsAndMessaging::{
        GetForegroundWindow, GetWindowThreadProcessId,
    };

    use crate::modules::gameinput::{
        A, B, DOWN, KeyState, LEFT, LEFT_SHOULDER, LEFT_STICK, MENU, PadState, RIGHT,
        RIGHT_SHOULDER, RIGHT_STICK, UP, VIEW, X, Y,
    };

    /// Asking XInput about an empty slot is slow; look again this often.
    const EMPTY_SLOT_RECHECK: Duration = Duration::from_secs(1);

    const BUTTONS: [(u16, u32); 14] = [
        (XINPUT_GAMEPAD_START, MENU),
        (XINPUT_GAMEPAD_BACK, VIEW),
        (XINPUT_GAMEPAD_A, A),
        (XINPUT_GAMEPAD_B, B),
        (XINPUT_GAMEPAD_X, X),
        (XINPUT_GAMEPAD_Y, Y),
        (XINPUT_GAMEPAD_DPAD_UP, UP),
        (XINPUT_GAMEPAD_DPAD_DOWN, DOWN),
        (XINPUT_GAMEPAD_DPAD_LEFT, LEFT),
        (XINPUT_GAMEPAD_DPAD_RIGHT, RIGHT),
        (XINPUT_GAMEPAD_LEFT_SHOULDER, LEFT_SHOULDER),
        (XINPUT_GAMEPAD_RIGHT_SHOULDER, RIGHT_SHOULDER),
        (XINPUT_GAMEPAD_LEFT_THUMB, LEFT_STICK),
        (XINPUT_GAMEPAD_RIGHT_THUMB, RIGHT_STICK),
    ];

    /// Raw values, no dead zone: titles apply their own.
    pub(super) fn convert(pad: &XINPUT_GAMEPAD) -> PadState {
        let stick = |v: i16| (f32::from(v) / 32767.0).max(-1.0);
        PadState {
            buttons: BUTTONS
                .iter()
                .filter(|(host, _)| pad.wButtons & host != 0)
                .fold(0, |all, (_, bit)| all | bit),
            left_trigger: f32::from(pad.bLeftTrigger) / 255.0,
            right_trigger: f32::from(pad.bRightTrigger) / 255.0,
            left_x: stick(pad.sThumbLX),
            left_y: stick(pad.sThumbLY),
            right_x: stick(pad.sThumbRX),
            right_y: stick(pad.sThumbRY),
        }
    }

    pub(super) fn poll_xinput(slot: usize, empty_until: &mut Option<Instant>) -> Option<PadState> {
        if empty_until.is_some_and(|until| Instant::now() < until) {
            return None;
        }
        let mut state = XINPUT_STATE::default();
        // SAFETY: room for the state.
        if unsafe { XInputGetState(slot as u32, &mut state) } != 0 {
            *empty_until = Some(Instant::now() + EMPTY_SLOT_RECHECK);
            return None;
        }
        *empty_until = None;
        Some(convert(&state.Gamepad))
    }

    pub(super) fn xinput_rumble(slot: usize, low: u16, high: u16) {
        let speeds = XINPUT_VIBRATION {
            wLeftMotorSpeed: low,
            wRightMotorSpeed: high,
        };
        // SAFETY: a valid vibration; an empty slot just fails.
        unsafe {
            let _ = XInputSetState(slot as u32, &speeds);
        }
    }

    fn key_state(vk: u8) -> KeyState {
        KeyState {
            // SAFETY: plain query.
            scan_code: unsafe { MapVirtualKeyW(u32::from(vk), MAPVK_VK_TO_VSC_EX) },
            virtual_key: vk,
            // shortcut: no characters (code point 0); Grand Theft
            // Auto VI reads scan codes and virtual keys only.
            ..Default::default()
        }
    }

    pub(super) fn poll_keys() -> Vec<KeyState> {
        #[cfg(feature = "diagnostics")]
        let mut down = scripted();
        #[cfg(not(feature = "diagnostics"))]
        let mut down = Vec::new();
        if focused() {
            // Left/right modifier keys only, as devices report them; no
            // mouse buttons (below 0x08).
            let held: Vec<u8> = (0x08..=0xFE_u8)
                .filter(|vk| !(0x10..=0x12).contains(vk) && !down.contains(vk))
                // SAFETY: plain queries.
                .filter(|&vk| unsafe { GetAsyncKeyState(i32::from(vk)) } < 0)
                .collect();
            down.extend(held);
        }
        down.into_iter().map(key_state).collect()
    }

    #[cfg(feature = "diagnostics")]
    pub(super) fn poll_scripted_keys() -> Vec<KeyState> {
        scripted().into_iter().map(key_state).collect()
    }

    /// Debugging aid: `VICTORY_KEYS=<seconds>:<key>,...` holds each key for
    /// 0.2 s that long after input starts (`<from>-<to>:<key>`: all that time),
    /// with or without focus.
    #[cfg(feature = "diagnostics")]
    fn scripted() -> Vec<u8> {
        static SCRIPT: LazyLock<(Instant, Vec<(Duration, Duration, u8)>)> = LazyLock::new(|| {
            let list = std::env::var("VICTORY_KEYS").unwrap_or_default();
            let keys = list.split(',').filter_map(|entry| {
                let (at, key) = entry.split_once(':')?;
                let vk = match key.trim() {
                    "Enter" => 0x0D,
                    "Esc" => 0x1B,
                    "Tab" => 0x09,
                    "Back" => 0x08,
                    "Up" => 0x26,
                    "Down" => 0x28,
                    "Left" => 0x25,
                    "Right" => 0x27,
                    k if k.len() == 1 && k.as_bytes()[0].is_ascii_alphanumeric() => {
                        k.as_bytes()[0].to_ascii_uppercase()
                    }
                    _ => return None,
                };
                let time = |t: &str| Duration::try_from_secs_f64(t.trim().parse().ok()?).ok();
                let (from, to) = match at.split_once('-') {
                    Some((from, to)) => (time(from)?, time(to)?),
                    None => (time(at)?, time(at)? + Duration::from_millis(200)),
                };
                Some((from, to, vk))
            });
            (Instant::now(), keys.collect())
        });
        let (start, keys) = &*SCRIPT;
        let now = start.elapsed();
        keys.iter()
            .filter(|&&(from, to, _)| (from..to).contains(&now))
            .map(|&(_, _, vk)| vk)
            .collect()
    }

    /// The foreground window is one of ours (the title's or our fallback).
    pub(super) fn focused() -> bool {
        let mut process = 0;
        // SAFETY: plain queries; a null window gives process 0.
        unsafe {
            GetWindowThreadProcessId(GetForegroundWindow(), &mut process);
            process == GetCurrentProcessId()
        }
    }

    #[test]
    fn host_pads_convert_to_console_pads() {
        let pad = XINPUT_GAMEPAD {
            wButtons: XINPUT_GAMEPAD_A | XINPUT_GAMEPAD_START | XINPUT_GAMEPAD_DPAD_LEFT,
            bLeftTrigger: 255,
            sThumbLX: -32768,
            sThumbLY: 32767,
            ..Default::default()
        };
        let state = convert(&pad);
        assert_eq!(state.buttons, A | MENU | LEFT);
        assert_eq!(
            (
                state.left_trigger,
                state.left_x,
                state.left_y,
                state.right_x
            ),
            (1.0, -1.0, 1.0, 0.0)
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::modules::gameinput::{A, B, X, Y};

    fn ds_with(pad: PadState) -> Option<SlotSnapshot> {
        Some(SlotSnapshot {
            connected: true,
            transport: Some(super::super::dualsense::protocol::Transport::Usb),
            pad,
            battery: BatteryState::default(),
            system_buttons: 0,
        })
    }

    const ONLY_VIRTUAL: XInputTopology = XInputTopology {
        virtual_present: true,
        physical_present: false,
    };
    const ONLY_PHYSICAL: XInputTopology = XInputTopology {
        virtual_present: false,
        physical_present: true,
    };
    const BOTH_VIRTUAL_AND_PHYSICAL: XInputTopology = XInputTopology {
        virtual_present: true,
        physical_present: true,
    };

    #[test]
    fn suppresses_steam_virtual_xinput_while_allowing_mixed_real_pads() {
        let mut arb = SlotArbitrator::default();
        let mut ds = [None; MAX_DUALSENSE];
        ds[0] = ds_with(PadState {
            buttons: A,
            ..Default::default()
        });

        // When only virtual XInput is present, Steam's virtual mirror is suppressed immediately.
        let mut xi = [None; PADS];
        xi[0] = Some(PadState {
            buttons: A,
            ..Default::default()
        });
        let resolved = arb.resolve(&ds, &xi, VirtualXInputPolicy::Auto, ONLY_VIRTUAL);
        assert!(resolved[0].is_some());
        assert!(
            resolved[1].is_none(),
            "Steam virtual XInput mirror must be suppressed when only virtual XInput exists"
        );

        // When NO virtual wrapper is present, two real players holding identical inputs
        // for 50 polls across multiple states must NEVER suppress the physical Xbox pad.
        let mut phys_arb = SlotArbitrator::default();
        let states = [
            PadState { buttons: A, ..Default::default() },
            PadState { buttons: A | B, left_x: 0.6, ..Default::default() },
            PadState { buttons: X, left_x: -0.8, ..Default::default() },
        ];
        for i in 0..50usize {
            let p = states[i % states.len()];
            ds[0] = ds_with(p);
            xi[0] = Some(p);
            let r = phys_arb.resolve(&ds, &xi, VirtualXInputPolicy::Auto, ONLY_PHYSICAL);
            assert!(r[0].is_some() && r[1].is_some());
        }

        // When BOTH virtual and physical XInput devices exist: sustained multi-state
        // mirroring on XInput 0 suppresses ONLY slot 0 while real Xbox on slot 1 stays bound.
        let mut mixed_arb = SlotArbitrator::default();
        xi[1] = Some(PadState {
            buttons: Y,
            ..Default::default()
        });
        let mut resolved = [None; PADS];
        for i in 0..usize::from(MIRROR_SUPPRESS_POLLS) {
            let mirrored = states[i / 9];
            ds[0] = ds_with(mirrored);
            xi[0] = Some(mirrored);
            resolved = mixed_arb.resolve(
                &ds,
                &xi,
                VirtualXInputPolicy::Auto,
                BOTH_VIRTUAL_AND_PHYSICAL,
            );
        }
        assert_eq!(mixed_arb.owner(0), Some(BackendSlot::DualSense(0)));
        assert!(
            (1..PADS).all(|s| mixed_arb.owner(s) != Some(BackendSlot::XInput(0))),
            "sustained multi-state virtual XInput mirror on slot 0 must be suppressed"
        );
        let xbox_slot = (0..PADS)
            .find(|&s| mixed_arb.owner(s) == Some(BackendSlot::XInput(1)))
            .expect("real Xbox pad on XInput 1 must stay bound alongside DualSense + Steam");
        assert_eq!(resolved[xbox_slot].unwrap().pad.buttons, Y);
    }

    #[test]
    fn brief_or_single_state_or_idle_separated_matches_never_suppress_real_xbox() {
        let mut arb = SlotArbitrator::default();
        let mut ds = [None; MAX_DUALSENSE];
        let mut xi = [None; PADS];

        // 1. Brief 3-poll simultaneous press must NOT suppress real Xbox.
        ds[0] = ds_with(PadState { buttons: A, ..Default::default() });
        xi[0] = Some(PadState { buttons: A, ..Default::default() });
        for _ in 0..3 {
            let r = arb.resolve(&ds, &xi, VirtualXInputPolicy::Auto, BOTH_VIRTUAL_AND_PHYSICAL);
            assert!(r[1].is_some());
        }

        // 2. Holding the exact same single button for 50 polls (only 1 distinct state)
        //    must NOT suppress real Xbox.
        for _ in 0..50 {
            let r = arb.resolve(&ds, &xi, VirtualXInputPolicy::Auto, BOTH_VIRTUAL_AND_PHYSICAL);
            assert!(r[1].is_some());
        }

        // 3. Alternating between only 2 unique states (A -> B -> A -> B...) for 36
        //    continuous polls must NOT suppress real Xbox (requires 3 unique states).
        let state_a = PadState { buttons: A, ..Default::default() };
        let state_b = PadState { buttons: B, ..Default::default() };
        for i in 0..36usize {
            let p = if (i / 4).is_multiple_of(2) { state_a } else { state_b };
            ds[0] = ds_with(p);
            xi[0] = Some(p);
            let r = arb.resolve(&ds, &xi, VirtualXInputPolicy::Auto, BOTH_VIRTUAL_AND_PHYSICAL);
            assert!(
                r[1].is_some(),
                "alternating A <-> B (only 2 unique states) must never suppress real Xbox"
            );
        }

        // 4. Separate coincidental taps separated by idle frames must NOT accumulate
        //    across idle gaps into suppression.
        ds[0] = ds_with(PadState::default());
        xi[0] = Some(PadState::default());
        let _ = arb.resolve(&ds, &xi, VirtualXInputPolicy::Auto, BOTH_VIRTUAL_AND_PHYSICAL);
        for btn in [A, B, X, Y, A | B] {
            ds[0] = ds_with(PadState { buttons: btn, ..Default::default() });
            xi[0] = Some(PadState { buttons: btn, ..Default::default() });
            for _ in 0..10 {
                let r =
                    arb.resolve(&ds, &xi, VirtualXInputPolicy::Auto, BOTH_VIRTUAL_AND_PHYSICAL);
                assert!(r[1].is_some());
            }
            // Both pads return to idle: partial match resets cleanly.
            ds[0] = ds_with(PadState::default());
            xi[0] = Some(PadState::default());
            let r = arb.resolve(&ds, &xi, VirtualXInputPolicy::Auto, BOTH_VIRTUAL_AND_PHYSICAL);
            assert!(r[1].is_some());
        }
    }

    #[test]
    fn real_xbox_pad_recovers_immediately_on_divergence_after_suppression() {
        let mut arb = SlotArbitrator::default();
        let mut ds = [None; MAX_DUALSENSE];
        let mut xi = [None; PADS];

        let states = [
            PadState { buttons: A, ..Default::default() },
            PadState { buttons: B, left_x: 0.75, ..Default::default() },
            PadState { buttons: X, left_y: -0.75, ..Default::default() },
        ];
        for i in 0..usize::from(MIRROR_SUPPRESS_POLLS) {
            let p = states[i / 9];
            ds[0] = ds_with(p);
            xi[0] = Some(p);
            let _ = arb.resolve(&ds, &xi, VirtualXInputPolicy::Auto, BOTH_VIRTUAL_AND_PHYSICAL);
        }
        assert!(arb.owner(1).is_none(), "mirrored slot should be suppressed");

        // Once suppressed, returning to idle keeps the virtual mirror suppressed.
        ds[0] = ds_with(PadState::default());
        xi[0] = Some(PadState::default());
        let idle = arb.resolve(&ds, &xi, VirtualXInputPolicy::Auto, BOTH_VIRTUAL_AND_PHYSICAL);
        assert!(idle[1].is_none());

        // The very first diverging active poll immediately restores the real Xbox pad.
        ds[0] = ds_with(PadState { buttons: X, ..Default::default() });
        xi[0] = Some(PadState { buttons: B, ..Default::default() });
        let recovered =
            arb.resolve(&ds, &xi, VirtualXInputPolicy::Auto, BOTH_VIRTUAL_AND_PHYSICAL);
        assert_eq!(recovered[1].unwrap().pad.buttons, B);
    }

    #[test]
    fn asymmetric_idle_and_active_boundary_never_accumulates_suppression() {
        let mut arb = SlotArbitrator::default();
        let mut ds = [None; MAX_DUALSENSE];
        let mut xi = [None; PADS];

        // Pairs where one pad is just below the activity threshold (`0.24` stick / `0.09` trigger)
        // and the other is just above (`0.26` / `0.11`) are within `TOL = 0.08`, yet only ONE
        // controller is active: they must NEVER accumulate distinct states or suppress real Xbox.
        let boundary_pairs = [
            (
                PadState { left_x: 0.24, ..Default::default() },
                PadState { left_x: 0.26, ..Default::default() },
            ),
            (
                PadState { left_y: 0.26, ..Default::default() },
                PadState { left_y: 0.24, ..Default::default() },
            ),
            (
                PadState { right_x: -0.24, ..Default::default() },
                PadState { right_x: -0.26, ..Default::default() },
            ),
            (
                PadState { left_trigger: 0.09, ..Default::default() },
                PadState { left_trigger: 0.11, ..Default::default() },
            ),
        ];
        for i in 0..48usize {
            let (d_pad, x_pad) = boundary_pairs[(i / 12) % boundary_pairs.len()];
            ds[0] = ds_with(d_pad);
            xi[0] = Some(x_pad);
            let r = arb.resolve(&ds, &xi, VirtualXInputPolicy::Auto, BOTH_VIRTUAL_AND_PHYSICAL);
            assert!(
                r[1].is_some(),
                "asymmetric idle/active boundary pair must never accumulate into suppression (poll {i})"
            );
        }
    }
}
