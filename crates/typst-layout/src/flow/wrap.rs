//! Side-wrapping float support.
//!
//! A side-wrapping float reserves a rectangular region on one side of the
//! column ([`ExclusionBand`]); in-flow paragraph text reflows into the
//! remaining measure for the lines level with it. Because the line breaker
//! commits lines strictly in order, it can ask a [`WrapProfile`] for the
//! available width of each line *by index* — it never needs to know a line's
//! eventual vertical position, which is only resolved later by the distributor.

use ecow::EcoVec;
use typst_library::layout::{Abs, FixedAlignment};

/// A rectangular region that in-flow text must avoid, wrapping into the space
/// beside it.
///
/// Coordinates are paragraph-relative: `y = 0` is the top of the paragraph that
/// wraps around the band. The distributor translates a float's absolute flow
/// position into this space when it builds a [`WrapProfile`].
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub struct ExclusionBand {
    /// Top of the reserved region, paragraph-relative.
    pub y0: Abs,
    /// Bottom of the reserved region, paragraph-relative.
    pub y1: Abs,
    /// Horizontal space reserved on `side`, including the clearance gutter.
    pub inset: Abs,
    /// `Start` for a left float, `End` for a right float.
    pub side: FixedAlignment,
}

/// A per-line-index table of available widths for a paragraph that wraps around
/// one or more [`ExclusionBand`]s.
///
/// The breaker queries [`WrapProfile::at`] by line index rather than by a
/// vertical coordinate, since lines are committed in order while their actual
/// `y` is resolved later. The table is derived from the bands plus an estimate
/// of the line pitch (line height + leading).
///
/// An empty profile (no bands) is equivalent to uniform full-width breaking and
/// hashes identically to a feature-free layout, which preserves the
/// memoization cache for non-wrapping paragraphs.
#[derive(Debug, Clone, Hash, PartialEq, Eq, Default)]
pub struct WrapProfile {
    /// Bands intersecting this paragraph, paragraph-relative.
    bands: EcoVec<ExclusionBand>,
    /// The full text width (the measure with no exclusion).
    width: Abs,
    /// Estimated height of a single line box.
    line_height: Abs,
    /// Leading inserted between consecutive lines.
    leading: Abs,
}

impl WrapProfile {
    /// Create a profile from its bands and line metrics.
    pub fn new(
        bands: EcoVec<ExclusionBand>,
        width: Abs,
        line_height: Abs,
        leading: Abs,
    ) -> Self {
        Self { bands, width, line_height, leading }
    }

    /// Whether this profile narrows no line (equivalent to uniform breaking).
    pub fn is_empty(&self) -> bool {
        self.bands.is_empty()
    }

    /// The full (un-narrowed) text width.
    pub fn width(&self) -> Abs {
        self.width
    }

    /// The horizontal offset and available width for line index `k`.
    ///
    /// Returns `(x_offset, available)`, where `x_offset` is the distance from
    /// the column's start edge at which the line should be placed and
    /// `available` is the line's measure. Lines that clear all bands get the
    /// full width at offset zero.
    pub fn at(&self, k: usize) -> (Abs, Abs) {
        // Vertical extent of line `k`, paragraph-relative. Leading sits between
        // lines, so line 0 starts at the paragraph top.
        let pitch = self.line_height + self.leading;
        let top = pitch * k as f64;
        let bottom = top + self.line_height;

        // Accumulate the insets contributed by overlapping bands on each side.
        let mut left = Abs::zero();
        let mut right = Abs::zero();
        for band in &self.bands {
            // A line level with any part of the band is shortened (matching the
            // CSS float model); a straddling line counts as overlapping.
            if top < band.y1 && bottom > band.y0 {
                match band.side {
                    FixedAlignment::End => right += band.inset,
                    // `Start` (and the unused `Center`) reserve from the left.
                    _ => left += band.inset,
                }
            }
        }

        if left == Abs::zero() && right == Abs::zero() {
            return (Abs::zero(), self.width);
        }

        let available = self.width - left - right;
        // Guard against a float at least as wide as the measure: fall back to
        // the full width so we never hand the breaker a non-positive measure.
        // (Forcing such a line fully below the float is a later refinement.)
        if available <= Abs::zero() {
            return (Abs::zero(), self.width);
        }

        (left, available)
    }
}

/// Whether a drop-cap exclusion band should continue past its opener paragraph.
///
/// An M-line opener consumes `consumed` of a full-N band of height `pitch * n`,
/// leaving `remaining0 = (n − M)·pitch + leading`. The band continues iff the
/// opener is at least one full line short of the cap (`remaining0 ≥ pitch`) AND a
/// continuation paragraph actually follows (the caller checks the latter).
///
/// The `≥ pitch` floor is load-bearing, NOT a fraction of a line: an opener that
/// EXACTLY fills the cap (M = n) still leaves `remaining0 = leading` (the band's
/// `pitch·n` overshoots the last baseline by one leading). `leading` can exceed
/// half a line in loose-leading modes (TTRPG / accessibility), so a `line_height/2`
/// floor would spuriously continue and shrink the gap to the next paragraph from the
/// paragraph spacing down to one leading. Since `leading < pitch` always, `≥ pitch`
/// is off for M = n and on for every M < n.
pub(crate) fn band_should_continue(
    n: usize,
    pitch: Abs,
    consumed: Abs,
    continuation_follows: bool,
) -> bool {
    let remaining0 = pitch * n as f64 - consumed;
    remaining0 >= pitch && continuation_follows
}

/// Whether a still-live band is exhausted after a continuation paragraph
/// consumed `lines_height` of it. A residual under half a line ends the run.
pub(crate) fn band_is_exhausted(remaining: Abs, line_height: Abs) -> bool {
    remaining <= line_height / 2.0
}

/// Supplies the available width for each line to the line breaker.
///
/// `Uniform` is the non-wrapping case and keeps the breaker branch- and
/// allocation-free; `Variable` consults a [`WrapProfile`].
#[derive(Debug, Copy, Clone)]
pub enum WidthProvider<'a> {
    /// A single measure for every line.
    Uniform(Abs),
    /// A per-line width table for a wrapping paragraph.
    Variable(&'a WrapProfile),
}

impl WidthProvider<'_> {
    /// The horizontal offset and available width for line index `k`.
    pub fn at(&self, k: usize) -> (Abs, Abs) {
        match self {
            Self::Uniform(width) => (Abs::zero(), *width),
            Self::Variable(profile) => profile.at(k),
        }
    }

    /// The available width for line index `k` (ignoring the offset).
    pub fn width_at(&self, k: usize) -> Abs {
        self.at(k).1
    }

    /// The full measure with no exclusion. Used to derive how much a band
    /// narrows a given line (`full_width - width_at(k)`).
    pub fn full_width(&self) -> Abs {
        match self {
            Self::Uniform(width) => *width,
            Self::Variable(profile) => profile.width(),
        }
    }
}

#[cfg(test)]
mod tests {
    use ecow::eco_vec;

    use super::*;

    fn profile(side: FixedAlignment) -> WrapProfile {
        // A 120pt float, 90pt tall, in a 300pt measure; 12pt lines, 4pt leading
        // (16pt pitch). Lines with top < 90pt are level with the float.
        WrapProfile::new(
            eco_vec![ExclusionBand {
                y0: Abs::zero(),
                y1: Abs::pt(90.0),
                inset: Abs::pt(120.0),
                side,
            }],
            Abs::pt(300.0),
            Abs::pt(12.0),
            Abs::pt(4.0),
        )
    }

    #[test]
    fn no_bands_is_full_width() {
        let p = WrapProfile::new(EcoVec::new(), Abs::pt(300.0), Abs::pt(12.0), Abs::pt(4.0));
        assert!(p.is_empty());
        assert_eq!(p.at(0), (Abs::zero(), Abs::pt(300.0)));
        assert_eq!(p.at(99), (Abs::zero(), Abs::pt(300.0)));
    }

    #[test]
    fn left_float_narrows_from_left() {
        let p = profile(FixedAlignment::Start);
        assert_eq!(p.at(0), (Abs::pt(120.0), Abs::pt(180.0))); // top 0 — beside
        assert_eq!(p.at(5), (Abs::pt(120.0), Abs::pt(180.0))); // top 80 — beside
        assert_eq!(p.at(6), (Abs::zero(), Abs::pt(300.0))); // top 96 — below
    }

    #[test]
    fn right_float_narrows_from_right() {
        let p = profile(FixedAlignment::End);
        assert_eq!(p.at(0), (Abs::zero(), Abs::pt(180.0))); // beside, no offset
        assert_eq!(p.at(6), (Abs::zero(), Abs::pt(300.0))); // below
    }

    #[test]
    fn straddling_line_counts_as_beside() {
        // Line 5: top 80, bottom 92 straddles y1 = 90 — still narrowed.
        let p = profile(FixedAlignment::Start);
        assert_eq!(p.at(5).1, Abs::pt(180.0));
    }

    #[test]
    fn provider_narrowing_is_reserved_width() {
        // The narrowing `commit` applies is `full_width - width_at(k)`.
        let p = profile(FixedAlignment::Start);
        let provider = WidthProvider::Variable(&p);
        assert_eq!(provider.full_width(), Abs::pt(300.0));
        // Beside the float: narrowed by the 120pt inset.
        assert_eq!(provider.full_width() - provider.width_at(0), Abs::pt(120.0));
        // Below the float: no narrowing.
        assert_eq!(provider.full_width() - provider.width_at(6), Abs::zero());
        // Uniform never narrows.
        let uniform = WidthProvider::Uniform(Abs::pt(300.0));
        assert_eq!(uniform.full_width() - uniform.width_at(0), Abs::zero());
    }

    #[test]
    fn float_wider_than_measure_falls_back_to_full_width() {
        let p = WrapProfile::new(
            eco_vec![ExclusionBand {
                y0: Abs::zero(),
                y1: Abs::pt(50.0),
                inset: Abs::pt(400.0),
                side: FixedAlignment::Start,
            }],
            Abs::pt(300.0),
            Abs::pt(12.0),
            Abs::pt(4.0),
        );
        assert_eq!(p.at(0), (Abs::zero(), Abs::pt(300.0)));
    }

    // --- v1 drop-cap continuation gate + band lifecycle -------------------

    #[test]
    fn gate_off_when_no_continuation_follows() {
        // N=3, pitch=16pt. A short opener (1 line, consumed≈12pt) leaves a tall
        // residual, but with no continuation paragraph following, the gate is
        // off → the opener takes today's clamped path (no active band).
        let pitch = Abs::pt(16.0);
        let consumed = Abs::pt(12.0); // one line
        assert!(!band_should_continue(3, pitch, consumed, false));
    }

    #[test]
    fn gate_on_for_short_opener_with_continuation() {
        // Same short opener, but a continuation follows: gate fires.
        // remaining0 = 48 − 12 = 36 ≥ pitch(16) → on.
        let pitch = Abs::pt(16.0);
        let consumed = Abs::pt(12.0);
        assert!(band_should_continue(3, pitch, consumed, true));
    }

    #[test]
    fn gate_off_when_opener_fills_the_cap() {
        // A ≥N-line opener consumes (nearly) the whole N-line band: remaining0 is
        // one leading (< pitch), so even with a continuation flag the gate stays
        // off → byte-identical to today for the common case.
        let pitch = Abs::pt(16.0);
        // 3 lines + 2 leadings = 36 + 8 = 44pt consumed; band = 48pt;
        // remaining0 = 4pt < pitch(16) → off.
        let consumed = Abs::pt(44.0);
        assert!(!band_should_continue(3, pitch, consumed, true));
    }

    #[test]
    fn gate_off_for_full_opener_in_loose_leading_mode() {
        // Regression guard: an M=N opener leaves remaining0 = leading. With LOOSE
        // leading (line_height=10, leading=8, pitch=18) that leading (8pt) exceeds
        // half a line (5pt) — a `line_height/2` floor would spuriously continue and
        // shrink the next paragraph's gap. The `≥ pitch` floor keeps it off.
        let pitch = Abs::pt(18.0);
        let consumed = Abs::pt(46.0); // 3*10 + 2*8 = three full lines
        // remaining0 = 54 − 46 = 8pt = one leading; 8 < pitch(18) → off.
        assert!(!band_should_continue(3, pitch, consumed, true));
    }

    #[test]
    fn band_exhaustion_threshold_is_half_a_line() {
        let line_height = Abs::pt(12.0);
        assert!(band_is_exhausted(Abs::pt(5.0), line_height)); // < 6pt → done
        assert!(band_is_exhausted(Abs::pt(6.0), line_height)); // == 6pt → done
        assert!(!band_is_exhausted(Abs::pt(7.0), line_height)); // > 6pt → continue
    }

    #[test]
    fn continuation_band_narrows_exactly_the_level_lines() {
        // A continuation paragraph wraps beside a residual band. With pitch=16pt
        // and the residual band y1=32pt, lines 0 and 1 (tops 0, 16) are level
        // with the cap and narrowed; line 2 (top 32) clears it and is full
        // width. This is the "at(k) selects exactly the lines level with the
        // cap" assertion for the continuation profile.
        let residual_y1 = Abs::pt(32.0);
        let p = WrapProfile::new(
            eco_vec![ExclusionBand {
                y0: Abs::zero(),
                y1: residual_y1,
                inset: Abs::pt(60.0),
                side: FixedAlignment::Start,
            }],
            Abs::pt(300.0),
            Abs::pt(12.0),
            Abs::pt(4.0),
        );
        assert_eq!(p.at(0), (Abs::pt(60.0), Abs::pt(240.0))); // top 0 — beside
        assert_eq!(p.at(1), (Abs::pt(60.0), Abs::pt(240.0))); // top 16 — beside
        assert_eq!(p.at(2), (Abs::zero(), Abs::pt(300.0))); // top 32 — below
    }
}
