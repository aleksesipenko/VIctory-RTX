//! Which console the title runs on. Titles pick settings (resolution,
//! memory budgets) by it, so the user chooses; Series X unless told.

use std::sync::atomic::{AtomicBool, Ordering};

const GIB: usize = 1 << 30;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum Console {
    #[default]
    SeriesX,
    SeriesS,
}

impl Console {
    /// Title memory in bytes: (GPU-optimal, standard). Series X in the
    /// advanced configuration; Series S has one range, counted as standard
    /// (7,776 MiB, the published console-mode figure).
    // shortcut: later system updates gave Series S titles more (amount not
    // published here); the title sizes nothing from this figure
    // (it goes by the device type), so update it if a title does.
    pub const fn title_memory(self) -> (usize, usize) {
        match self {
            Console::SeriesX => (10 * GIB, 7 * GIB / 2),
            Console::SeriesS => (0, 7_776 << 20),
        }
    }

    pub const fn name(self) -> &'static str {
        match self {
            Console::SeriesX => "Xbox Series X",
            Console::SeriesS => "Xbox Series S",
        }
    }
}

static SERIES_S: AtomicBool = AtomicBool::new(false);

/// Set before the title starts.
pub fn set(console: Console) {
    let series_s = console == Console::SeriesS;
    SERIES_S.store(series_s, Ordering::Relaxed);
    victory_formats::texture::set_series_s_tiling(series_s);
}

pub fn current() -> Console {
    if SERIES_S.load(Ordering::Relaxed) {
        Console::SeriesS
    } else {
        Console::SeriesX
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn set_updates_series_s_tiling_and_restores_series_x() {
        struct Reset;
        impl Drop for Reset {
            fn drop(&mut self) {
                set(Console::SeriesX);
            }
        }
        let _reset = Reset;
        let desc = victory_formats::texture::Desc {
            dimension: victory_formats::texture::Dimension::Tex2d,
            width: 256,
            height: 256,
            depth: 1,
            mips: 1,
            element_bytes: 4,
            block: 1,
            samples: 1,
            swizzle: victory_formats::texture::Swizzle::SW_64KB_R_X,
        };
        set(Console::SeriesS);
        assert_eq!(current(), Console::SeriesS);
        let series_s = victory_formats::texture::Layout::new(&desc).unwrap();
        assert_eq!(series_s.offset(0, 32, 0, 0), Ok(0x2400));

        set(Console::SeriesX);
        assert_eq!(current(), Console::SeriesX);
        let series_x = victory_formats::texture::Layout::new(&desc).unwrap();
        assert_eq!(series_x.offset(0, 32, 0, 0), Ok(0x800));
    }
}
