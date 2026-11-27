//! DualSense (`0x0CE6`) and DualSense Edge (`0x0DF2`) HID report codecs for
//! USB (`0x01`/`0x02`, wired and Pico 2 W `DS5Dongle`) and Bluetooth (`0x31`).

use victory_core::config::{
    DualSenseConfig, EdgePadBinding, EdgePaddleMode, LedBrightness, TouchpadClickMode,
};

use crate::modules::gameinput::{
    A, B, BatteryState, DOWN, LEFT, LEFT_SHOULDER, LEFT_STICK, MENU, PADDLE_LEFT_1, PADDLE_RIGHT_1,
    PadState, RIGHT, RIGHT_SHOULDER, RIGHT_STICK, UP, VIEW, X, Y,
};

pub const SONY_VID: u16 = 0x054C;
pub const PID_DUALSENSE: u16 = 0x0CE6;
pub const PID_DUALSENSE_EDGE: u16 = 0x0DF2;

pub const USAGE_PAGE_GENERIC_DESKTOP: u16 = 0x01;
pub const USAGE_GAMEPAD: u16 = 0x05;

pub const USB_INPUT_REPORT_ID: u8 = 0x01;
pub const USB_OUTPUT_REPORT_ID: u8 = 0x02;
pub const BT_REPORT_ID: u8 = 0x31;
/// Required DualSense Bluetooth `0x31` output report tag (byte 2), matching
/// Linux `hid-playstation.c` (`DS_OUTPUT_TAG`) and SDL `SDL_hidapi_ps5.c`.
pub const BT_OUTPUT_TAG: u8 = 0x10;

pub const USB_INPUT_REPORT_LEN: usize = 64;
pub const USB_OUTPUT_REPORT_LEN: usize = 48;
pub const BT_REPORT_LEN: usize = 78;
pub const OUTPUT_COMMON_LEN: usize = 47;

/// Mode `0x05`: active adaptive trigger off/release command (per Microsoft
/// GameInput DualSense sample and `hid-playstation`).
pub const TRIGGER_MODE_OFF: u8 = 0x05;
/// 11-byte adaptive trigger block that actively disengages stepper resistance.
pub const TRIGGER_OFF: [u8; 11] = [TRIGGER_MODE_OFF, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];

pub const FEATURE_CALIBRATION_ID: u8 = 0x05;
pub const FEATURE_CALIBRATION_LEN: usize = 41;
pub const FEATURE_FIRMWARE_ID: u8 = 0x20;
pub const FEATURE_FIRMWARE_LEN: usize = 64;

/// Firmware threshold introducing `COMPATIBLE_VIBRATION2` (`valid_flag2` bit 2).
pub const FIRMWARE_VIBRATION2_MIN: u16 = 0x0224;

pub const BT_CRC_SEED_INPUT: u8 = 0xA1;
pub const BT_CRC_SEED_OUTPUT: u8 = 0xA2;
pub const BT_CRC_SEED_FEATURE: u8 = 0xA3;

/// `valid_flag0` bits.
const FLAG0_COMPATIBLE_VIBRATION: u8 = 0x01;
const FLAG0_HAPTICS_SELECT: u8 = 0x02;
const FLAG0_RIGHT_TRIGGER_ENABLE: u8 = 0x04;
const FLAG0_LEFT_TRIGGER_ENABLE: u8 = 0x08;

/// `valid_flag1` bits.
const FLAG1_LIGHTBAR_ENABLE: u8 = 0x04;
const FLAG1_RELEASE_LEDS: u8 = 0x08;
const FLAG1_PLAYER_INDICATOR_ENABLE: u8 = 0x10;

/// `valid_flag2` bits.
const FLAG2_LED_BRIGHTNESS_ENABLE: u8 = 0x01;
const FLAG2_LIGHTBAR_SETUP_ENABLE: u8 = 0x02;
const FLAG2_COMPATIBLE_VIBRATION2: u8 = 0x04;

/// PS5-style symmetrical 5-LED player indicator masks for slots 0..=3.
const PLAYER_LED_MASKS: [u8; 4] = [0x04, 0x0A, 0x15, 0x1B];

/// Touchpad horizontal midpoint (1920x1080 panel).
const TOUCHPAD_MID_X: u16 = 960;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ControllerModel {
    Standard,
    Edge,
}

impl ControllerModel {
    pub fn from_pid(pid: u16) -> Option<Self> {
        match pid {
            PID_DUALSENSE => Some(Self::Standard),
            PID_DUALSENSE_EDGE => Some(Self::Edge),
            _ => None,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Transport {
    /// Direct USB cable or Pico 2 W `DS5Dongle` USB-spoofed wireless bridge.
    Usb,
    /// Host motherboard Bluetooth (`0x31` enhanced report with CRC-32).
    Bluetooth,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TouchPoint {
    pub active: bool,
    pub id: u8,
    pub x: u16,
    pub y: u16,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RawReport {
    pub lx: u8,
    pub ly: u8,
    pub rx: u8,
    pub ry: u8,
    pub l2: u8,
    pub r2: u8,
    pub seq: u8,
    pub buttons: [u8; 4],
    pub gyro: [i16; 3],
    pub accel: [i16; 3],
    /// 3 MHz (`0.3333 µs` per tick) hardware sensor clock.
    pub sensor_timestamp: u32,
    pub touch: [TouchPoint; 2],
    /// `0..=10` (10% increments).
    pub battery_level: u8,
    /// `0x0` discharging, `0x1` charging, `0x2` full; `>= 0xA` abnormal/absent.
    pub charging_state: u8,
    /// True when decoded from a 10-byte Bluetooth basic `0x01` fallback report
    /// (which carries no hardware sensor clock, IMU, or battery telemetry).
    pub basic_bt_report: bool,
}

impl RawReport {
    pub fn ps_pressed(&self) -> bool {
        self.buttons[2] & 0x01 != 0
    }

    pub fn touchpad_clicked(&self) -> bool {
        self.buttons[2] & 0x02 != 0
    }

    pub fn any_touch_active(&self) -> bool {
        self.touch[0].active || self.touch[1].active
    }

    /// True if any primary action button (Triangle/Circle/Cross/Square) is down.
    /// Used to avoid intercepting `PS + Action` multi-device Bluetooth slot chords.
    pub fn any_face_button_pressed(&self) -> bool {
        self.buttons[0] & 0xF0 != 0
    }

    /// True when the report carries a plausible live controller state rather
    /// than an uninitialized all-zero `DS5Dongle` standby frame.
    pub fn has_live_telemetry(&self) -> bool {
        if self.basic_bt_report {
            return true;
        }
        let valid_charge = matches!(self.charging_state, 0x0..=0x2);
        valid_charge
            && (self.sensor_timestamp != 0 || self.battery_level > 0 || self.charging_state != 0)
    }

    pub fn battery_state(&self) -> BatteryState {
        const FULL_WH: f32 = 5.7;
        let ratio = (f32::from(self.battery_level.min(10)) / 10.0).clamp(0.0, 1.0);
        let (status, charge_rate) = match self.charging_state {
            0x0 => (1, -0.8), // Discharging
            0x1 => (3, 2.5),  // Charging
            0x2 => (2, 0.0),  // Full / Idle
            _ => (-1, 0.0),   // Unknown
        };
        BatteryState {
            status,
            charge_rate,
            remaining_capacity: if status == 2 {
                FULL_WH
            } else {
                ratio * FULL_WH
            },
            full_charge_capacity: FULL_WH,
        }
    }
}

/// IEEE 802.3 CRC-32 with a single leading seed byte (`0xA1`/`0xA2`/`0xA3`),
/// as required by DualSense Bluetooth HID reports.
pub fn crc32_seeded(seed: u8, bytes: &[u8]) -> u32 {
    fn step(mut crc: u32, byte: u8) -> u32 {
        crc ^= u32::from(byte);
        for _ in 0..8 {
            let mask = (crc & 1).wrapping_neg();
            crc = (crc >> 1) ^ (0xEDB8_8320 & mask);
        }
        crc
    }
    let mut crc = step(0xFFFF_FFFF, seed);
    for &b in bytes {
        crc = step(crc, b);
    }
    !crc
}

/// Extracts firmware update version (`u16`, e.g. `0x0224`, `0x0402`) from
/// Feature Report `0x20` (64 bytes; Bluetooth carries a trailing CRC-32).
pub fn parse_firmware_version(report: &[u8], transport: Transport) -> Option<u16> {
    if report.len() < FEATURE_FIRMWARE_LEN || report[0] != FEATURE_FIRMWARE_ID {
        return None;
    }
    if transport == Transport::Bluetooth {
        let payload = &report[..FEATURE_FIRMWARE_LEN - 4];
        let expected = u32::from_le_bytes(
            report[FEATURE_FIRMWARE_LEN - 4..FEATURE_FIRMWARE_LEN]
                .try_into()
                .ok()?,
        );
        if crc32_seeded(BT_CRC_SEED_FEATURE, payload) != expected {
            return None;
        }
    }
    let version = u16::from_le_bytes([report[44], report[45]]);
    (version != 0).then_some(version)
}

/// Decodes a USB (`0x01`, 64+ bytes), Bluetooth enhanced (`0x31`, 78+ bytes),
/// or Bluetooth basic (`0x01`, 10+ bytes) DualSense input report.
pub fn parse_input_report(packet: &[u8], transport: Transport) -> Option<RawReport> {
    match (transport, packet.first().copied()?) {
        (Transport::Usb, USB_INPUT_REPORT_ID) if packet.len() >= USB_INPUT_REPORT_LEN => {
            parse_common_payload(&packet[1..64])
        }
        (Transport::Bluetooth, BT_REPORT_ID) if packet.len() >= BT_REPORT_LEN => {
            let body = &packet[..BT_REPORT_LEN - 4];
            let got_crc =
                u32::from_le_bytes(packet[BT_REPORT_LEN - 4..BT_REPORT_LEN].try_into().ok()?);
            if crc32_seeded(BT_CRC_SEED_INPUT, body) != got_crc {
                return None;
            }
            // Byte 0 is 0x31, byte 1 is BT header/tag; 63-byte common payload starts at byte 2.
            parse_common_payload(&packet[2..65])
        }
        // Fallback if Bluetooth controller has not yet switched out of basic 0x01 mode.
        (Transport::Bluetooth, USB_INPUT_REPORT_ID) if packet.len() >= 10 => Some(RawReport {
            lx: packet[1],
            ly: packet[2],
            rx: packet[3],
            ry: packet[4],
            buttons: [packet[5], packet[6], packet[7], 0],
            l2: packet[8],
            r2: packet[9],
            sensor_timestamp: 0,
            battery_level: 0,
            charging_state: 0x0F,
            basic_bt_report: true,
            ..Default::default()
        }),
        _ => None,
    }
}

fn parse_touch(raw: &[u8; 4]) -> TouchPoint {
    let active = raw[0] & 0x80 == 0;
    let id = raw[0] & 0x7F;
    let x = u16::from(raw[1]) | (u16::from(raw[2] & 0x0F) << 8);
    let y = (u16::from(raw[2] >> 4)) | (u16::from(raw[3]) << 4);
    TouchPoint { active, id, x, y }
}

fn parse_common_payload(p: &[u8]) -> Option<RawReport> {
    if p.len() < 63 {
        return None;
    }
    let i16_at = |at: usize| i16::from_le_bytes([p[at], p[at + 1]]);
    let t0: [u8; 4] = p[32..36].try_into().ok()?;
    let t1: [u8; 4] = p[36..40].try_into().ok()?;
    let status0 = p[52];
    Some(RawReport {
        lx: p[0],
        ly: p[1],
        rx: p[2],
        ry: p[3],
        l2: p[4],
        r2: p[5],
        seq: p[6],
        buttons: [p[7], p[8], p[9], p[10]],
        gyro: [i16_at(15), i16_at(17), i16_at(19)],
        accel: [i16_at(21), i16_at(23), i16_at(25)],
        sensor_timestamp: u32::from_le_bytes([p[27], p[28], p[29], p[30]]),
        touch: [parse_touch(&t0), parse_touch(&t1)],
        battery_level: status0 & 0x0F,
        charging_state: (status0 >> 4) & 0x0F,
        basic_bt_report: false,
    })
}

fn map_edge_binding(binding: EdgePadBinding) -> u32 {
    match binding {
        EdgePadBinding::None => 0,
        EdgePadBinding::A => A,
        EdgePadBinding::B => B,
        EdgePadBinding::X => X,
        EdgePadBinding::Y => Y,
        EdgePadBinding::LeftShoulder => LEFT_SHOULDER,
        EdgePadBinding::RightShoulder => RIGHT_SHOULDER,
        EdgePadBinding::LeftStick => LEFT_STICK,
        EdgePadBinding::RightStick => RIGHT_STICK,
        EdgePadBinding::Up => UP,
        EdgePadBinding::Down => DOWN,
        EdgePadBinding::Left => LEFT,
        EdgePadBinding::Right => RIGHT,
        EdgePadBinding::View => VIEW,
        EdgePadBinding::Menu => MENU,
        EdgePadBinding::PaddleLeft1 => PADDLE_LEFT_1,
        EdgePadBinding::PaddleRight1 => PADDLE_RIGHT_1,
    }
}

/// Maps a physical touchpad click to `VIEW` or `MENU` according to `cfg`.
pub fn touchpad_button(raw: &RawReport, cfg: &DualSenseConfig) -> u32 {
    match cfg.touchpad_click {
        TouchpadClickMode::View => VIEW,
        TouchpadClickMode::Split => {
            let right_half = raw.touch[0].active && raw.touch[0].x >= TOUCHPAD_MID_X;
            if right_half { MENU } else { VIEW }
        }
    }
}

/// Converts a decoded DualSense report into an Xbox `PadState` (sticks raw,
/// Y inverted so up is `+1.0`, center `128` mapped to exact `0.0`), omitting
/// the physical touchpad click so `SystemButtonFilter` can arbitrate the
/// `PS + Touchpad` gyro-recenter chord across frames.
pub fn to_pad_state_without_touchpad(
    raw: &RawReport,
    model: ControllerModel,
    cfg: &DualSenseConfig,
) -> PadState {
    let stick_x = |v: u8| ((f32::from(v) - 128.0) / 127.0).clamp(-1.0, 1.0);
    let stick_y = |v: u8| ((128.0 - f32::from(v)) / 127.0).clamp(-1.0, 1.0);

    let mut buttons = match raw.buttons[0] & 0x0F {
        0 => UP,
        1 => UP | RIGHT,
        2 => RIGHT,
        3 => DOWN | RIGHT,
        4 => DOWN,
        5 => DOWN | LEFT,
        6 => LEFT,
        7 => UP | LEFT,
        _ => 0,
    };

    if raw.buttons[0] & 0x10 != 0 {
        buttons |= X; // Square
    }
    if raw.buttons[0] & 0x20 != 0 {
        buttons |= A; // Cross
    }
    if raw.buttons[0] & 0x40 != 0 {
        buttons |= B; // Circle
    }
    if raw.buttons[0] & 0x80 != 0 {
        buttons |= Y; // Triangle
    }

    if raw.buttons[1] & 0x01 != 0 {
        buttons |= LEFT_SHOULDER;
    }
    if raw.buttons[1] & 0x02 != 0 {
        buttons |= RIGHT_SHOULDER;
    }
    if raw.buttons[1] & 0x10 != 0 {
        buttons |= VIEW; // Create
    }
    if raw.buttons[1] & 0x20 != 0 {
        buttons |= MENU; // Options
    }
    if raw.buttons[1] & 0x40 != 0 {
        buttons |= LEFT_STICK;
    }
    if raw.buttons[1] & 0x80 != 0 {
        buttons |= RIGHT_STICK;
    }

    if model == ControllerModel::Edge && cfg.edge_paddles == EdgePaddleMode::Custom {
        if raw.buttons[2] & 0x40 != 0 {
            buttons |= map_edge_binding(cfg.edge_left_paddle);
        }
        if raw.buttons[2] & 0x80 != 0 {
            buttons |= map_edge_binding(cfg.edge_right_paddle);
        }
    }

    PadState {
        buttons,
        left_trigger: f32::from(raw.l2) / 255.0,
        right_trigger: f32::from(raw.r2) / 255.0,
        left_x: stick_x(raw.lx),
        left_y: stick_y(raw.ly),
        right_x: stick_x(raw.rx),
        right_y: stick_y(raw.ry),
    }
}

/// Stateless single-frame conversion including touchpad clicks (suppressed
/// whenever PS is currently held). Live HID streaming uses
/// [`to_pad_state_without_touchpad`] together with `SystemButtonFilter` so
/// `PS + Touchpad` chord suppression spans release order and button skew.
pub fn to_pad_state(raw: &RawReport, model: ControllerModel, cfg: &DualSenseConfig) -> PadState {
    let mut pad = to_pad_state_without_touchpad(raw, model, cfg);
    if raw.touchpad_clicked() && !raw.ps_pressed() {
        pad.buttons |= touchpad_button(raw, cfg);
    }
    pad
}

/// Desired actuator, adaptive trigger, and indicator state for an output report.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OutputCommand {
    pub firmware_version: u16,
    pub enable_hid_rumble: bool,
    pub motor_left: u8,
    pub motor_right: u8,
    pub left_trigger: [u8; 11],
    pub right_trigger: [u8; 11],
    pub update_leds: bool,
    pub release_leds: bool,
    pub slot: usize,
    pub lightbar_rgb: [u8; 3],
    pub brightness: LedBrightness,
}

impl OutputCommand {
    /// Clean hardware-release packet sent on focus loss, shutdown, or title crash:
    /// stops rumble, actively disengages both adaptive triggers (`0x05`), and
    /// optionally releases LEDs.
    pub fn neutral(firmware_version: u16, release_leds: bool) -> Self {
        Self {
            firmware_version,
            enable_hid_rumble: true,
            motor_left: 0,
            motor_right: 0,
            left_trigger: TRIGGER_OFF,
            right_trigger: TRIGGER_OFF,
            update_leds: false,
            release_leds,
            slot: 0,
            lightbar_rgb: [0; 3],
            brightness: LedBrightness::Medium,
        }
    }

    pub fn is_actuators_zero(&self) -> bool {
        self.motor_left == 0
            && self.motor_right == 0
            && self.left_trigger == TRIGGER_OFF
            && self.right_trigger == TRIGGER_OFF
    }

    /// Encodes the 47-byte common DualSense output block. Audio volume/routing
    /// bits (`valid_flag0` `0x10..0x80`) and the microphone mute LED are
    /// intentionally never set so `DS5Dongle` v0.8+ volume locks, user headset
    /// levels, and real OS microphone mute state remain truthful and untouched.
    pub fn encode_common(&self) -> [u8; OUTPUT_COMMON_LEN] {
        let mut c = [0u8; OUTPUT_COMMON_LEN];
        let mut flag0 = FLAG0_RIGHT_TRIGGER_ENABLE | FLAG0_LEFT_TRIGGER_ENABLE;
        let mut flag1 = 0u8;
        let mut flag2 = 0u8;

        if self.enable_hid_rumble {
            flag0 |= FLAG0_HAPTICS_SELECT;
            if self.firmware_version >= FIRMWARE_VIBRATION2_MIN {
                flag2 |= FLAG2_COMPATIBLE_VIBRATION2;
            } else {
                flag0 |= FLAG0_COMPATIBLE_VIBRATION;
            }
            c[2] = self.motor_right;
            c[3] = self.motor_left;
        }

        c[10..21].copy_from_slice(&self.right_trigger);
        c[21..32].copy_from_slice(&self.left_trigger);

        if self.release_leds {
            flag1 |= FLAG1_RELEASE_LEDS;
        } else if self.update_leds {
            flag1 |= FLAG1_LIGHTBAR_ENABLE | FLAG1_PLAYER_INDICATOR_ENABLE;
            flag2 |= FLAG2_LED_BRIGHTNESS_ENABLE | FLAG2_LIGHTBAR_SETUP_ENABLE;
            c[41] = 0x02; // Enable lightbar
            c[42] = match self.brightness {
                LedBrightness::High => 0x00,
                LedBrightness::Medium => 0x01,
                LedBrightness::Low | LedBrightness::Off => 0x02,
            };
            if self.brightness != LedBrightness::Off {
                c[43] = PLAYER_LED_MASKS[self.slot.min(3)];
                c[44..47].copy_from_slice(&self.lightbar_rgb);
            }
        }

        c[0] = flag0;
        c[1] = flag1;
        c[38] = flag2;
        c
    }

    /// Encodes a USB Output Report `0x02` sized to `report_len` (minimum 48 bytes;
    /// matches `HIDP_CAPS.OutputReportByteLength` for `DS5Dongle` TinyUSB).
    pub fn encode_usb(&self, report_len: usize) -> Vec<u8> {
        let len = report_len.max(USB_OUTPUT_REPORT_LEN);
        let mut buf = vec![0u8; len];
        buf[0] = USB_OUTPUT_REPORT_ID;
        buf[1..1 + OUTPUT_COMMON_LEN].copy_from_slice(&self.encode_common());
        buf
    }

    /// Encodes a 78-byte Bluetooth Output Report `0x31` matching Linux
    /// `hid-playstation.c` (`struct dualsense_output_report_bt`):
    /// - byte 0: `0x31` (`BT_REPORT_ID`)
    /// - byte 1: `(seq & 0x0F) << 4` (`seq_tag`)
    /// - byte 2: `0x10` (`BT_OUTPUT_TAG`)
    /// - bytes 3..50: 47-byte common payload
    /// - bytes 50..74: reserved zeros
    /// - bytes 74..78: little-endian IEEE 802.3 CRC-32 seeded with `0xA2` over `0..74`
    pub fn encode_bt(&self, seq: u8) -> [u8; BT_REPORT_LEN] {
        let mut buf = [0u8; BT_REPORT_LEN];
        buf[0] = BT_REPORT_ID;
        buf[1] = (seq & 0x0F) << 4;
        buf[2] = BT_OUTPUT_TAG;
        buf[3..3 + OUTPUT_COMMON_LEN].copy_from_slice(&self.encode_common());
        let crc = crc32_seeded(BT_CRC_SEED_OUTPUT, &buf[..BT_REPORT_LEN - 4]);
        buf[BT_REPORT_LEN - 4..].copy_from_slice(&crc.to_le_bytes());
        buf
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_usb_input(ly: u8, buttons0: u8, buttons2: u8, touch_x: u16, ts: u32) -> [u8; 64] {
        let mut pkt = [0u8; 64];
        pkt[0] = USB_INPUT_REPORT_ID;
        pkt[1] = 128; // lx center
        pkt[2] = ly;
        pkt[3] = 128;
        pkt[4] = 128;
        pkt[5] = 255; // L2 full
        pkt[8] = buttons0;
        pkt[10] = buttons2;
        pkt[28..32].copy_from_slice(&ts.to_le_bytes());
        // Active touch point 0 at (touch_x, 500)
        pkt[33] = 0x01; // bit 7 == 0 => active
        pkt[34] = (touch_x & 0xFF) as u8;
        pkt[35] = ((touch_x >> 8) & 0x0F) as u8 | ((500u16 & 0x0F) as u8) << 4;
        pkt[36] = (500u16 >> 4) as u8;
        pkt[53] = 0x18; // charging (0x1), 80% (8)
        pkt
    }

    #[test]
    fn usb_and_bt_reports_decode_and_invert_stick_y() {
        let usb = make_usb_input(0, 0x20 | 0x06, 0x02, 1400, 90_000);
        let raw = parse_input_report(&usb, Transport::Usb).unwrap();
        assert!(raw.has_live_telemetry());
        assert_eq!(raw.sensor_timestamp, 90_000);
        assert_eq!(raw.battery_level, 8);
        assert_eq!(raw.charging_state, 1);

        let cfg = DualSenseConfig::default();
        let pad = to_pad_state(&raw, ControllerModel::Standard, &cfg);
        // ly = 0 (pushed all the way up) must become +1.0; Cross = A, hat 6 = LEFT, right touchpad click = MENU.
        assert_eq!(pad.left_x, 0.0);
        assert_eq!(pad.left_y, 1.0);
        assert_eq!(pad.left_trigger, 1.0);
        assert_eq!(pad.buttons, A | LEFT | MENU);

        // Wrap the exact 63-byte payload in a 78-byte Bluetooth 0x31 packet with CRC-32.
        let mut bt = [0u8; BT_REPORT_LEN];
        bt[0] = BT_REPORT_ID;
        bt[1] = 0x02;
        bt[2..65].copy_from_slice(&usb[1..64]);
        let crc = crc32_seeded(BT_CRC_SEED_INPUT, &bt[..BT_REPORT_LEN - 4]);
        bt[BT_REPORT_LEN - 4..].copy_from_slice(&crc.to_le_bytes());
        assert_eq!(parse_input_report(&bt, Transport::Bluetooth).unwrap(), raw);

        // Corrupting CRC-32 rejects the Bluetooth frame.
        bt[BT_REPORT_LEN - 1] ^= 0x55;
        assert!(parse_input_report(&bt, Transport::Bluetooth).is_none());
    }

    #[test]
    fn firmware_0224_selects_compatible_vibration2() {
        let mut old_cmd = OutputCommand::neutral(0x0210, false);
        old_cmd.motor_left = 200;
        let old = old_cmd.encode_common();
        assert_ne!(old[0] & FLAG0_COMPATIBLE_VIBRATION, 0);
        assert_eq!(old[38] & FLAG2_COMPATIBLE_VIBRATION2, 0);

        let mut new_cmd = OutputCommand::neutral(0x0402, false);
        new_cmd.motor_left = 200;
        let new = new_cmd.encode_common();
        assert_eq!(new[0] & FLAG0_COMPATIBLE_VIBRATION, 0);
        assert_ne!(new[38] & FLAG2_COMPATIBLE_VIBRATION2, 0);
    }

    #[test]
    fn edge_custom_paddles_and_ps_touchpad_chord_suppression() {
        let cfg = DualSenseConfig {
            edge_paddles: EdgePaddleMode::Custom,
            edge_left_paddle: EdgePadBinding::A,
            edge_right_paddle: EdgePadBinding::RightShoulder,
            ..Default::default()
        };

        // Left & right Edge paddles + PS + Touchpad click held together.
        let usb = make_usb_input(128, 0x08, 0x01 | 0x02 | 0x40 | 0x80, 200, 12_345);
        let raw = parse_input_report(&usb, Transport::Usb).unwrap();
        let pad = to_pad_state(&raw, ControllerModel::Edge, &cfg);
        // PS + Touchpad chord must NOT emit VIEW/MENU, while custom Edge paddles emit A | RB.
        assert_eq!(pad.buttons, A | RIGHT_SHOULDER);
    }

    #[test]
    fn bt_output_0x31_layout_seq_tag_crc_and_trigger_off() {
        let mut cmd = OutputCommand::neutral(0x0402, true);
        cmd.motor_left = 120;
        cmd.motor_right = 60;
        let common = cmd.encode_common();

        // Neutral triggers actively send mode 0x05 at both trigger blocks, and
        // mic mute LED bit (valid_flag1 bit 0) is never set.
        assert_eq!(TRIGGER_OFF, [TRIGGER_MODE_OFF, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
        assert_eq!(common[10..21], TRIGGER_OFF);
        assert_eq!(common[21..32], TRIGGER_OFF);
        assert_eq!(common[1] & 0x01, 0, "mic mute LED bit must never be set");

        for seq in [0u8, 5, 15, 0x2A] {
            let bt = cmd.encode_bt(seq);
            assert_eq!(bt.len(), BT_REPORT_LEN);
            assert_eq!(bt[0], BT_REPORT_ID);
            assert_eq!(bt[1], (seq & 0x0F) << 4);
            assert_eq!(bt[2], BT_OUTPUT_TAG);
            assert_eq!(&bt[3..3 + OUTPUT_COMMON_LEN], &common);
            assert!(bt[3 + OUTPUT_COMMON_LEN..BT_REPORT_LEN - 4].iter().all(|&b| b == 0));
            let expected_crc = crc32_seeded(BT_CRC_SEED_OUTPUT, &bt[..BT_REPORT_LEN - 4]);
            let got_crc = u32::from_le_bytes(bt[BT_REPORT_LEN - 4..].try_into().unwrap());
            assert_eq!(got_crc, expected_crc);
        }
    }

    #[test]
    fn disabling_hid_rumble_omits_haptics_select_and_vibration_flags() {
        let mut cmd = OutputCommand::neutral(0x0402, false);
        cmd.enable_hid_rumble = false;
        cmd.motor_left = 255;
        cmd.motor_right = 255;
        let common = cmd.encode_common();
        assert_eq!(common[0] & FLAG0_HAPTICS_SELECT, 0);
        assert_eq!(common[0] & FLAG0_COMPATIBLE_VIBRATION, 0);
        assert_eq!(common[38] & FLAG2_COMPATIBLE_VIBRATION2, 0);
        assert_ne!(common[0] & FLAG0_LEFT_TRIGGER_ENABLE, 0);
        assert_ne!(common[0] & FLAG0_RIGHT_TRIGGER_ENABLE, 0);
        assert_eq!(common[2], 0);
        assert_eq!(common[3], 0);
    }

    #[test]
    fn bt_basic_0x01_report_sets_basic_flag_and_live_telemetry() {
        let pkt = [
            USB_INPUT_REPORT_ID,
            128, // lx
            0,   // ly up
            128, // rx
            128, // ry
            0x20 | 0x08, // Cross (A) + neutral hat
            0x01,        // L1
            0x00, // buttons2
            255, // l2
            128, // r2
        ];
        let raw = parse_input_report(&pkt, Transport::Bluetooth).unwrap();
        assert!(raw.basic_bt_report);
        assert!(raw.has_live_telemetry());
        assert_eq!(raw.sensor_timestamp, 0);
        let pad = to_pad_state(&raw, ControllerModel::Standard, &DualSenseConfig::default());
        assert_eq!(pad.buttons, A | LEFT_SHOULDER);
        assert_eq!(pad.left_y, 1.0);
        assert_eq!(pad.left_trigger, 1.0);
    }
}
