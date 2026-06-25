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
}
