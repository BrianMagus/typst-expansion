use typst_library::introspection::Tag;
use typst_library::layout::{
    Abs, Axes, FixedAlignment, Fr, Frame, FrameItem, Point, Ratio, Region, Regions,
    Rel, Size, Transform,
};
use typst_utils::Numeric;

use comemo::Tracked;
use typst_library::foundations::StyleChain;
use typst_library::text::{TextElem, families, variant};
use typst_library::World;

use super::collect::build_line_children;
use super::wrap::{
    ExclusionBand, WrapProfile, band_is_exhausted, band_should_continue,
};
use super::{
    Child, Composer, DeferredParChild, FlowResult, LineChild, MultiChild, MultiSpill,
    PlacedChild, SingleChild, Stop, WrapSpill, Work,
};

/// Distributes as many children as fit from `composer.work` into the first
/// region and returns the resulting frame.
pub fn distribute(composer: &mut Composer, regions: Regions) -> FlowResult<Frame> {
    let mut distributor = Distributor {
        composer,
        regions,
        items: vec![],
        sticky: None,
        stickable: None,
        pending_wrap_float: None,
        active_band: None,
    };
    let init = distributor.snapshot();
    let forced = match distributor.run() {
        Ok(()) => distributor.composer.work.done(),
        Err(Stop::Finish(forced)) => forced,
        Err(err) => return Err(err),
    };
    let region = Region::new(regions.size, regions.expand);
    distributor.finalize(region, init, forced)
}

/// State for distribution.
///
/// See [Composer] regarding lifetimes.
struct Distributor<'a, 'b, 'x, 'y, 'z> {
    /// The composer that is used to handle insertions.
    composer: &'z mut Composer<'a, 'b, 'x, 'y>,
    /// Regions which are continuously shrunk as new items are added.
    regions: Regions<'z>,
    /// Already laid out items, not yet aligned.
    items: Vec<Item<'a, 'b>>,
    /// A snapshot which can be restored to migrate a suffix of sticky blocks to
    /// the next region.
    sticky: Option<DistributionSnapshot<'a, 'b>>,
    /// Whether the current group of consecutive sticky blocks are still sticky
    /// and may migrate with the attached frame. This is `None` while we aren't
    /// processing sticky blocks. On the first sticky block, this will become
    /// `Some(true)` if migrating sticky blocks as usual would make a
    /// difference - this is given by `regions.may_progress()`. Otherwise, it
    /// is set to `Some(false)`, which is usually the case when the first
    /// sticky block in the group is at the very top of the page (then,
    /// migrating it would just lead us back to the top of the page, leading
    /// to an infinite loop). In that case, all sticky blocks of the group are
    /// also disabled, until this is reset to `None` on the first non-sticky
    /// frame we find.
    ///
    /// While this behavior of disabling stickiness of sticky blocks at the
    /// very top of the page may seem non-ideal, it is only problematic (that
    /// is, may lead to orphaned sticky blocks / headings) if the combination
    /// of 'sticky blocks + attached frame' doesn't fit in one page, in which
    /// case there is nothing Typst can do to improve the situation, as sticky
    /// blocks are supposed to always be in the same page as the subsequent
    /// frame, but that is impossible in that case, which is thus pathological.
    stickable: Option<bool>,
    /// The side-wrap float laid out by the immediately-preceding `placed()`
    /// call, awaiting the deferred paragraph that wraps beside it. Single-slot:
    /// POC guarantees float and paragraph are adjacent within one pass.
    pending_wrap_float: Option<(Frame, FixedAlignment)>,
    /// A drop-cap exclusion band that outlived its opener paragraph (short
    /// opener, M < N lines) and continues into the following paragraph(s). Set
    /// by `deferred_par` when the gate fires; consumed by `continuation_par`;
    /// cleared the moment the band is exhausted or a non-continuation child
    /// arrives (which first emits a strong clearing gap). Lives on the
    /// per-region distributor — a band never crosses a region break (§10g).
    active_band: Option<ActiveBand>,
}

/// A live drop-cap exclusion band threaded across continuation paragraphs.
///
/// The cap frame is already painted at the opener; continuation paragraphs only
/// narrow their lines beside the band's lower part. `remaining` is the unfilled
/// vertical extent of the band (flow-y); each continuation subtracts the leading
/// it owns plus the height of the lines it emits.
#[derive(Clone, Copy)]
struct ActiveBand {
    /// Unconsumed vertical extent of the band (flow-y), measured from the top of
    /// the next continuation paragraph.
    remaining: Abs,
    /// Horizontal space reserved on `side` (scaled cap width + clearance).
    inset: Abs,
    /// `Start` for a left float, `End` for a right float.
    side: FixedAlignment,
    /// Line pitch (line height + leading), uniform across the run.
    pitch: Abs,
    /// Leading between consecutive lines (intra-paragraph), used to build the
    /// continuation's WrapProfile.
    leading: Abs,
    /// The single inter-paragraph gap each continuation owns. For a drop cap this
    /// is the leading (so the cap sits beside evenly-leaded lines, §10d); for a
    /// general image float it is the paragraph spacing (so wrapped paragraphs keep
    /// their normal vertical rhythm beside the image).
    gap: Abs,
}

/// A snapshot of the distribution state.
struct DistributionSnapshot<'a, 'b> {
    work: Work<'a, 'b>,
    items: usize,
}

/// A laid out item in a distribution.
enum Item<'a, 'b> {
    /// An introspection tag.
    Tag(&'a Tag),
    /// Absolute spacing and its weakness level.
    Abs(Abs, u8),
    /// Fractional spacing or a fractional block.
    Fr(Fr, u8, Option<&'b SingleChild<'a>>),
    /// A frame for a laid out line or block.
    Frame(Frame, Axes<FixedAlignment>),
    /// A frame for an absolutely (not floatingly) placed child.
    Placed(Frame, &'b PlacedChild<'a>),
    /// A side-wrap float: drawn at the current offset WITHOUT advancing it,
    /// so the following deferred paragraph's lines overlap its top band. The
    /// `Abs` is a vertical anchor delta added to the paint offset: zero for an
    /// ordinary wrap float, and `line0_baseline − body_cap_height` for a drop
    /// cap so the (scaled) cap-top registers on line 1's cap-top.
    WrapFloat(Frame, FixedAlignment, Abs),
}

impl Item<'_, '_> {
    /// Whether this item should be migrated to the next region if the region
    /// consists solely of such items.
    fn migratable(&self) -> bool {
        match self {
            Self::Tag(_) => true,
            Self::Frame(frame, _) => {
                frame.size().is_zero()
                    && frame.items().all(|(_, item)| {
                        matches!(item, FrameItem::Link(_, _) | FrameItem::Tag(_))
                    })
            }
            Self::Placed(_, placed) => !placed.float,
            Self::WrapFloat(..) => false,
            _ => false,
        }
    }
}

impl<'a, 'b> Distributor<'a, 'b, '_, '_, '_> {
    /// Distributes content into the region.
    fn run(&mut self) -> FlowResult<()> {
        // First, handle spill of a breakable block.
        if let Some(spill) = self.composer.work.spill.take() {
            self.multi_spill(spill)?;
        }

        // Then handle spilled continuation lines of a wrapped paragraph.
        if let Some(spill) = self.composer.work.wrap_spill.take() {
            self.wrap_spill(spill)?;
        }

        // If spill are taken care of, process children until no space is left
        // or no children are left.
        while let Some(child) = self.composer.work.head() {
            self.child(child)?;
            self.composer.work.advance();
        }

        Ok(())
    }

    /// Processes a single child.
    ///
    /// - Returns `Ok(())` if the child was successfully processed.
    /// - Returns `Err(Stop::Finish)` if a region break should be triggered.
    /// - Returns `Err(Stop::Relayout(_))` if the region needs to be relayouted
    ///   due to an insertion (float/footnote).
    /// - Returns `Err(Stop::Error(_))` if there was a fatal error.
    fn child(&mut self, child: &'b Child<'a>) -> FlowResult<()> {
        // A live drop-cap band that is NOT being continued by this child must be
        // cleared before the child is laid out, reserving the leftover float
        // height with a STRONG gap so the next block can't overlap the cap
        // (§10e). Tags pass through untouched (they collect between paragraphs);
        // a continuation `Deferred` is handled by `continuation_par` below.
        let is_continuation = matches!(child, Child::Deferred(d) if d.continuation);
        if self.active_band.is_some()
            && !is_continuation
            && !matches!(child, Child::Tag(_))
        {
            let band = self.active_band.take().expect("checked is_some");
            self.regions.size.y -= band.remaining;
            self.items.push(Item::Abs(band.remaining, 0));
        }

        match child {
            Child::Tag(tag) => self.tag(tag),
            Child::Rel(amount, weakness) => self.rel(*amount, *weakness),
            Child::Fr(fr, weakness) => self.fr(*fr, *weakness),
            Child::Line(line) => self.line(line)?,
            Child::Single(single) => self.single(single)?,
            Child::Multi(multi) => self.multi(multi)?,
            Child::Placed(placed) => self.placed(placed)?,
            // A continuation paragraph wraps beside the live band; the opener
            // (or a continuation that arrived with no band) takes deferred_par.
            Child::Deferred(d) if d.continuation && self.active_band.is_some() => {
                self.continuation_par(d)?
            }
            Child::Deferred(deferred) => self.deferred_par(deferred)?,
            Child::Flush => self.flush()?,
            Child::Break(weak) => self.break_(*weak)?,
        }
        Ok(())
    }

    /// Processes a tag.
    fn tag(&mut self, tag: &'a Tag) {
        self.composer.work.tags.push(tag);
    }

    /// Generate items for pending tags.
    fn flush_tags(&mut self) {
        if !self.composer.work.tags.is_empty() {
            let tags = &mut self.composer.work.tags;
            self.items.extend(tags.iter().copied().map(Item::Tag));
            tags.clear();
        }
    }

    /// Processes relative spacing.
    fn rel(&mut self, amount: Rel<Abs>, weakness: u8) {
        let amount = amount.relative_to(self.regions.base().y);
        if weakness > 0 && !self.keep_weak_rel_spacing(amount, weakness) {
            return;
        }

        self.regions.size.y -= amount;
        self.items.push(Item::Abs(amount, weakness));
    }

    /// Processes fractional spacing.
    fn fr(&mut self, fr: Fr, weakness: u8) {
        if weakness > 0 && !self.keep_weak_fr_spacing(fr, weakness) {
            return;
        }

        // If we decided to keep the fr spacing, it's safe to trim previous
        // spacing as no stronger fr spacing can exist.
        self.trim_spacing();

        self.items.push(Item::Fr(fr, weakness, None));
    }

    /// Decides whether to keep weak spacing based on previous items. If there
    /// is a preceding weak spacing, it might be patched in place.
    fn keep_weak_rel_spacing(&mut self, amount: Abs, weakness: u8) -> bool {
        for item in self.items.iter_mut().rev() {
            match *item {
                // When previous weak relative spacing exists that's at most as
                // weak, we reuse the old item, set it to the maximum of both,
                // and discard the new item.
                Item::Abs(prev_amount, prev_weakness @ 1..) => {
                    if weakness <= prev_weakness
                        && (weakness < prev_weakness || amount > prev_amount)
                    {
                        self.regions.size.y -= amount - prev_amount;
                        *item = Item::Abs(amount, weakness);
                    }
                    return false;
                }
                // These are "peeked beyond" for spacing collapsing purposes.
                Item::Tag(_) | Item::Abs(_, 0) | Item::Placed(..)
                | Item::WrapFloat(..) => {}
                // Any kind of fractional spacing destructs weak relative
                // spacing.
                Item::Fr(.., None) => return false,
                // These naturally support the spacing.
                Item::Frame(..) | Item::Fr(.., Some(_)) => return true,
            }
        }
        false
    }

    /// Decides whether to keep weak fractional spacing based on previous items.
    /// If there is a preceding weak spacing, it might be patched in place.
    fn keep_weak_fr_spacing(&mut self, fr: Fr, weakness: u8) -> bool {
        for item in self.items.iter_mut().rev() {
            match *item {
                // When previous weak fr spacing exists that's at most as weak,
                // we reuse the old item, set it to the maximum of both, and
                // discard the new item.
                Item::Fr(prev_fr, prev_weakness @ 1.., None) => {
                    if weakness <= prev_weakness
                        && (weakness < prev_weakness || fr > prev_fr)
                    {
                        *item = Item::Fr(fr, weakness, None);
                    }
                    return false;
                }
                // These are "peeked beyond" for spacing collapsing purposes.
                // Weak absolute spacing, in particular, will be trimmed once
                // we push the fractional spacing.
                Item::Tag(_) | Item::Abs(..) | Item::Placed(..)
                | Item::WrapFloat(..) => {}
                // For weak + strong fr spacing, we keep both, same as for
                // weak + strong rel spacing.
                Item::Fr(.., None) => return true,
                // These naturally support the spacing.
                Item::Frame(..) | Item::Fr(.., Some(_)) => return true,
            }
        }
        false
    }

    /// Trims trailing weak spacing from the items.
    fn trim_spacing(&mut self) {
        for (i, item) in self.items.iter().enumerate().rev() {
            match *item {
                Item::Abs(amount, 1..) => {
                    self.regions.size.y += amount;
                    self.items.remove(i);
                    break;
                }
                Item::Fr(_, 1.., None) => {
                    self.items.remove(i);
                    break;
                }
                Item::Tag(_) | Item::Abs(..) | Item::Placed(..)
                | Item::WrapFloat(..) => {}
                Item::Frame(..) | Item::Fr(..) => break,
            }
        }
    }

    /// The amount of trailing weak spacing.
    fn weak_spacing(&mut self) -> Abs {
        for item in self.items.iter().rev() {
            match *item {
                Item::Abs(amount, 1..) => return amount,
                Item::Tag(_) | Item::Abs(..) | Item::Placed(..)
                | Item::WrapFloat(..) => {}
                Item::Frame(..) | Item::Fr(..) => break,
            }
        }
        Abs::zero()
    }

    /// Processes a line of a paragraph.
    fn line(&mut self, line: &'b LineChild) -> FlowResult<()> {
        // If the line doesn't fit and a followup region may improve things,
        // finish the region.
        if !self.regions.size.y.fits(line.frame.height()) && self.regions.may_progress() {
            return Err(Stop::Finish(false));
        }

        // If the line's need, which includes its own height and that of
        // following lines grouped by widow/orphan prevention, does not fit into
        // the current region, but does fit into the next region, finish the
        // region.
        if !self.regions.size.y.fits(line.need)
            && self
                .regions
                .iter()
                .nth(1)
                .is_some_and(|region| region.y.fits(line.need))
        {
            return Err(Stop::Finish(false));
        }

        self.frame(line.frame.clone(), line.align, false, false)
    }

    /// Processes an unbreakable block.
    fn single(&mut self, single: &'b SingleChild<'a>) -> FlowResult<()> {
        // Lay out the block.
        let frame = single.layout(
            self.composer.engine,
            Region::new(self.regions.base(), self.regions.expand),
        )?;

        // Handle fractionally sized blocks.
        if let Some(fr) = single.fr {
            self.composer
                .footnotes(&self.regions, &frame, Abs::zero(), false, true)?;
            self.flush_tags();
            self.items.push(Item::Fr(fr, 0, Some(single)));
            return Ok(());
        }

        // If the block doesn't fit and a followup region may improve things,
        // finish the region.
        if !self.regions.size.y.fits(frame.height()) && self.regions.may_progress() {
            return Err(Stop::Finish(false));
        }

        self.frame(frame, single.align, single.sticky, false)
    }

    /// Processes a breakable block.
    fn multi(&mut self, multi: &'b MultiChild<'a>) -> FlowResult<()> {
        // Skip directly if the region is already (over)full. `line` and
        // `single` implicitly do this through their `fits` checks.
        if self.regions.is_full() {
            return Err(Stop::Finish(false));
        }

        // Lay out the block.
        let (frame, spill) = multi.layout(self.composer.engine, self.regions)?;
        if frame.is_empty()
            && spill.as_ref().is_some_and(|s| s.exist_non_empty_frame)
            && self.regions.may_progress()
        {
            // If the first frame is empty, but there are non-empty frames in
            // the spill, the whole child should be put in the next region to
            // avoid any invisible orphans at the end of this region.
            return Err(Stop::Finish(false));
        }

        self.frame(frame, multi.align, multi.sticky, true)?;

        // If the block didn't fully fit into the current region, save it into
        // the `spill` and finish the region.
        if let Some(spill) = spill {
            self.composer.work.spill = Some(spill);
            self.composer.work.advance();
            return Err(Stop::Finish(false));
        }

        Ok(())
    }

    /// Processes spillover from a breakable block.
    fn multi_spill(&mut self, spill: MultiSpill<'a, 'b>) -> FlowResult<()> {
        // Skip directly if the region is already (over)full.
        if self.regions.is_full() {
            self.composer.work.spill = Some(spill);
            return Err(Stop::Finish(false));
        }

        // Lay out the spilled remains.
        let align = spill.align();
        let (frame, spill) = spill.layout(self.composer.engine, self.regions)?;
        self.frame(frame, align, false, true)?;

        // If there's still more, save it into the `spill` and finish the
        // region.
        if let Some(spill) = spill {
            self.composer.work.spill = Some(spill);
            return Err(Stop::Finish(false));
        }

        Ok(())
    }

    /// Processes an in-flow frame, generated from a line or block.
    fn frame(
        &mut self,
        frame: Frame,
        align: Axes<FixedAlignment>,
        sticky: bool,
        breakable: bool,
    ) -> FlowResult<()> {
        if sticky {
            // If the frame is sticky and we haven't remembered a preceding
            // sticky element, make a checkpoint which we can restore should we
            // end on this sticky element.
            //
            // The first sticky block within consecutive sticky blocks
            // determines whether this group of sticky blocks has stickiness
            // disabled or not.
            //
            // The criteria used here is: if migrating this group of sticky
            // blocks together with the "attached" block can't improve the lack
            // of space, since we're at the start of the region, then we don't
            // do so, and stickiness is disabled (at least, for this region).
            // Otherwise, migration is allowed.
            //
            // Note that, since the whole region is checked, this ensures sticky
            // blocks at the top of a block - but not necessarily of the page -
            // can still be migrated.
            if self.sticky.is_none()
                && *self.stickable.get_or_insert_with(|| self.regions.may_progress())
            {
                self.sticky = Some(self.snapshot());
            }
        } else if !frame.is_empty() {
            // If the frame isn't sticky, we can forget a previous snapshot. We
            // interrupt a group of sticky blocks, if there was one, so we reset
            // the saved stickable check for the next group of sticky blocks.
            self.sticky = None;
            self.stickable = None;
        }

        // Handle footnotes.
        self.composer.footnotes(
            &self.regions,
            &frame,
            frame.height(),
            breakable,
            true,
        )?;

        // Push an item for the frame.
        self.regions.size.y -= frame.height();
        self.flush_tags();
        self.items.push(Item::Frame(frame, align));
        Ok(())
    }

    /// Processes an absolutely or floatingly placed child.
    fn placed(&mut self, placed: &'b PlacedChild<'a>) -> FlowResult<()> {
        if placed.float && placed.wrap && placed.wrap_active.get() {
            // A side-wrapping float that is immediately followed by a wrapping
            // paragraph (`wrap_active` set at collect time): lay it out and draw
            // it flush to its side WITHOUT reserving its height, so the
            // following deferred paragraph flows BESIDE it, not under.
            let frame = placed.layout(self.composer.engine, self.regions.base())?;

            // Like any non-sticky in-flow frame, a wrap float interrupts a run
            // of sticky blocks, so forget any saved snapshot (cf. `frame`).
            if !frame.is_empty() {
                self.sticky = None;
                self.stickable = None;
            }

            // Footnotes still reserve against the float's real height; that is
            // correct regardless of the wrapping.
            self.composer
                .footnotes(&self.regions, &frame, frame.height(), true, true)?;
            self.flush_tags();
            // Stash the frame so the next Child::Deferred can build its
            // exclusion band without scanning the item stack.
            self.pending_wrap_float = Some((frame.clone(), placed.align_x));
            // `y_delta` starts at zero; `deferred_par` overwrites it (and the
            // frame) in place for a drop cap once the first wrapped line's
            // baseline and the body cap-height are known (R3).
            self.items
                .push(Item::WrapFloat(frame, placed.align_x, Abs::zero()));
            return Ok(());
        }
        if placed.float && placed.wrap {
            // A side-wrap float with no immediately-following wrapping paragraph
            // (next child is a heading/list/another float/EOF): fall back to the
            // Step-4 behavior — anchor in place and RESERVE its height so the
            // following content flows cleanly BELOW it (no overlap).
            let frame = placed.layout(self.composer.engine, self.regions.base())?;

            if !frame.is_empty() {
                self.sticky = None;
                self.stickable = None;
            }

            self.composer
                .footnotes(&self.regions, &frame, frame.height(), true, true)?;
            self.flush_tags();
            self.regions.size.y -= frame.height();
            self.items.push(Item::Frame(
                frame,
                Axes::new(placed.align_x, FixedAlignment::Start),
            ));
            return Ok(());
        }
        if placed.float {
            // If the element is floatingly placed, let the composer handle it.
            // It might require relayout because the area available for
            // distribution shrinks. We make the spacing occupied by weak
            // spacing temporarily available again because it can collapse if it
            // ends up at a break due to the float.
            let weak_spacing = self.weak_spacing();
            self.regions.size.y += weak_spacing;
            self.composer.float(
                placed,
                &self.regions,
                self.items.iter().any(|item| matches!(item, Item::Frame(..))),
                true,
            )?;
            self.regions.size.y -= weak_spacing;
        } else {
            let frame = placed.layout(self.composer.engine, self.regions.base())?;
            self.composer
                .footnotes(&self.regions, &frame, Abs::zero(), true, true)?;
            self.flush_tags();
            self.items.push(Item::Placed(frame, placed));
        }
        Ok(())
    }

    /// Break a paragraph beside the immediately-preceding side-wrap float, and
    /// paginate its continuation lines when the region tail runs out.
    fn deferred_par(&mut self, d: &'b DeferredParChild<'a>) -> FlowResult<()> {
        // The float must have been stashed by the immediately-preceding
        // placed() call. If it is absent, the float and paragraph did not stay
        // adjacent in this pass — fall back to a normal full-width break.
        let Some((frame, side)) = self.pending_wrap_float.take() else {
            return self.deferred_par_uniform(d);
        };

        let float_height = frame.height();
        let float_width = frame.width();

        let leading = d.elem.leading.resolve(d.styles);
        let full_width = self.regions.size.x;

        // For a drop cap, the reserved vertical zone is exactly N line pitches
        // (decoupled from the glyph box height — §3a), and the float is scaled
        // so its cap span equals the N-line target. Both require the true line
        // pitch, so measure it up front in that case; the ordinary wrap path
        // keeps its original ordering byte-for-byte.
        let (band_bottom, dropcap_n, pitch, line_height_pre) = match d.dropcap {
            Some(n) if n >= 1 => {
                let line_height = self.measure_line_height(d, full_width)?;
                let pitch = line_height + leading;
                (pitch * n as f64, Some(n), Some(pitch), Some(line_height))
            }
            _ => (float_height + d.clearance, None, None, None),
        };

        // GUARD 0 (cache-consistent region): the float is laid out exactly once
        // by placed() against `regions.base()`; its cached height is only valid
        // where the live `full` still equals the collect-time base height. This
        // excludes only regions whose live `full` differs from that base — e.g.
        // a continuation region after a page break. Normal column 2 keeps the
        // same column height, so `full == d.base.y` holds and wrap proceeds
        // correctly there. The `==` is intentionally exact: `full` is a raw,
        // unmodified field copy of the collect-time base (no arithmetic), so
        // there is no rounding to tolerate.
        let cache_consistent = self.regions.full == d.base.y;

        // GUARD 1 (band-fit): the entire float-adjacent zone must complete in
        // this region's tail, or the band would straddle a region boundary and
        // post-break narrow lines would render with no float beside them (a left
        // gutter with no float on the next page). This must fire WHENEVER the
        // band can't fit the tail, regardless of page position — at the page top
        // (`may_progress()` false, e.g. a float taller than the region) bypassing
        // it would still produce inset lines that overflow and spill a guttered
        // continuation line. Guard 1 is not a break-retry gate: on failure
        // `wrap_fallback_reserve` consumes the `Child::Deferred` (completing or
        // spilling with advance_on_spill), so it is never re-entered for the same
        // paragraph — termination-safe. The float-taller-than-the-whole-page
        // degenerate then degrades to existing over-tall-content overflow rather
        // than phantom-gutter corruption.
        if !cache_consistent || !self.regions.size.y.fits(band_bottom) {
            return self.wrap_fallback_reserve(d, frame, side);
        }

        // For a drop cap, pre-scale the cap to its nominal N-line span so the
        // exclusion band narrows by the *scaled* cap width (§3c). `n_eff` is
        // re-clamped to the actual laid-out line count after breaking (R1), and
        // the painted cap is re-derived from the original frame then — so a
        // short opener never over-reserves. `scaled_cap_width` drives the band
        // inset here; the painted frame + anchor are computed post-layout.
        let scaled_cap_width = match (dropcap_n, pitch) {
            (Some(n), Some(pitch)) => {
                let body_cap_height = body_cap_height(self.composer.engine.world, d.styles);
                let target = pitch * (n.saturating_sub(1)) as f64 + body_cap_height;
                let cap_span = frame.height();
                let scale = if cap_span > Abs::zero() {
                    target / cap_span
                } else {
                    1.0
                };
                float_width * scale
            }
            _ => float_width,
        };

        let band = ExclusionBand {
            y0: Abs::zero(),
            y1: band_bottom,
            inset: scaled_cap_width + d.clearance,
            side, // the stashed float's align_x
        };

        // Measure the true line pitch, then break the paragraph beside the
        // float. The drop-cap path already measured it above; reuse that value.
        let line_height = match line_height_pre {
            Some(h) => h,
            None => self.measure_line_height(d, full_width)?,
        };

        let profile =
            WrapProfile::new(ecow::eco_vec![band], full_width, line_height, leading);

        let frames = crate::inline::layout_par(
            d.elem,
            self.composer.engine,
            d.locator.relayout(),
            d.styles,
            d.base,
            d.expand,
            d.par_situation,
            Some(&profile),
        )?
        .into_frames();

        // Per-line height safety: WrapProfile::at derives which lines straddle
        // the band from a single probed pitch. If any line's true height
        // diverges (a tall interior inline), the band assignment is unreliable —
        // discard the wrapped frames and fall back to anchored reserve-height.
        const EPS: f64 = 0.01;
        let pitch_varies = frames
            .iter()
            .any(|f| (f.height() - line_height).to_pt().abs() > EPS);
        if pitch_varies {
            return self.wrap_fallback_reserve(d, frame, side);
        }

        let spacing = d.elem.spacing.resolve(d.styles);
        let line_children = build_line_children(frames, leading, d.styles);

        // Σ line heights, to emit a compensating gap if the float outlives the
        // paragraph (so following content clears a taller-than-paragraph float),
        // and to compute the band continuation gate. Computed BEFORE the
        // dropcap scaling block so the gate can decide full-`n` vs clamped.
        let lines_height: Abs =
            line_children.iter().map(|l| l.frame.height()).sum::<Abs>()
                + leading * (line_children.len().saturating_sub(1) as f64);

        // CONTINUATION GATE (§10c — load-bearing regression guard): keep the cap
        // at its FULL N lines (no n_eff clamp) and continue the band into the
        // following paragraph(s) ONLY when (a) the cap outlives the opener by at
        // least half a line, AND (b) a continuation `Deferred` actually follows.
        // Both must hold, or every short-opener-with-no-continuation and every
        // ≥N-line opener would diverge from today. The "did not spill" leg is
        // only known post-emit; the spill path returns Err before the trailing
        // rel, so trailing suppression is irrelevant there.
        let continuation_follows = matches!(
            self.composer.work.peek_next(),
            Some(Child::Deferred(nd)) if nd.continuation
        );
        let want_active = continuation_follows
            && match (dropcap_n, pitch) {
                (Some(n), Some(pitch)) => band_should_continue(n, pitch, lines_height, true),
                // General image float (no drop cap): continue while the float
                // outlives the opener by at least half a line, so a tiny overhang
                // doesn't start a one-line continuation beside a near-empty band.
                _ => band_bottom - lines_height >= line_height / 2.0,
            };

        // Drop-cap anchoring + scaling (Tier 1 + Tier 2), computed HERE because
        // the laid-out lines, the body cap-height, the pitch, and N all coexist
        // in this scope (R3). For an ordinary wrap float this whole block is
        // skipped and `band_bottom` is unchanged, keeping that path identical.
        // The active-band branch scales to FULL `n` (band continues); the else
        // branch keeps today's `n_eff` clamp + compensating gap verbatim.
        let mut band_bottom = band_bottom;
        let mut active_band_to_set: Option<ActiveBand> = None;
        if let (Some(n), Some(pitch)) = (dropcap_n, pitch) {
            // n_eff = full `n` when continuing (the cap stays N lines tall beside
            // the continuation); else clamp to the laid-out line count so a short
            // opener without a continuation never reserves phantom lines (R1).
            let n_eff = if want_active {
                n.max(1)
            } else {
                n.min(line_children.len()).max(1)
            };
            // The reserved zone is exactly N_eff pitches (not the glyph box).
            band_bottom = pitch * n_eff as f64;

            // line-0 cap-top anchor + cap span target, from real metrics.
            let line0_baseline = line_children
                .first()
                .map(|l| l.frame.baseline())
                .unwrap_or_default();
            let body_cap_height = body_cap_height(self.composer.engine.world, d.styles);
            let target = pitch * (n_eff.saturating_sub(1)) as f64 + body_cap_height;

            // The cap frame is cap-trimmed (top-edge cap-height / bottom-edge
            // baseline), so `frame.height()` is the cap span and `baseline()`
            // ≈ `height()`. Guard a future markup change that breaks this.
            debug_assert!(
                (frame.baseline() - frame.height()).to_pt().abs() < 0.5,
                "drop-cap frame must be cap-trimmed (baseline ≈ height)",
            );

            let cap_span = frame.height();
            let scale = if cap_span > Abs::zero() {
                target / cap_span
            } else {
                1.0
            };

            // Scale the cap frame so it REPORTS its scaled bounds, following the
            // `layout_scale` precedent (transforms.rs: transform + set_size).
            let mut scaled = frame.clone();
            let ratio = Ratio::new(scale);
            scaled.transform(Transform::scale(ratio, ratio));
            scaled.set_size(Size::new(scaled.width() * scale, scaled.height() * scale));

            // Paint the scaled cap-top at line 1's cap-top; its baseline then
            // lands on line N's baseline.
            let y_delta = line0_baseline - body_cap_height;

            // Stage the band for continuation BEFORE moving `scaled` into the
            // WrapFloat: the part of the (full-N) band the opener did NOT consume
            // continues into the next paragraph(s). The inset is the SCALED cap
            // width + clearance (matches the painted cap and the band the opener
            // narrowed against). Pitch is uniform across the run (§10d), and the
            // continuation owns its leading-sized gap.
            if want_active {
                // The threaded band's physical bottom is the cap's VISUAL baseline
                // (line N's baseline = line0_baseline + (N−1)·pitch), NOT the line-box
                // bottom pitch·N. pitch·N overshoots the cap by ~one leading and lands
                // the continuation's lower boundary on an EXACT line top, where float
                // rounding (`remaining` = 2·pitch + 1e-15) narrows one extra line beside
                // empty space under the cap. Anchoring to the baseline puts the boundary
                // mid-line — robust — and narrows exactly the lines level with the cap.
                let cap_baseline = line0_baseline + pitch * (n_eff.saturating_sub(1)) as f64;
                let remaining = (cap_baseline - lines_height).max(Abs::zero());
                active_band_to_set = Some(ActiveBand {
                    remaining,
                    inset: scaled.width() + d.clearance,
                    side,
                    pitch,
                    leading,
                    gap: leading, // drop cap: leading-tight gap (§10d)
                });
            }

            // Push the scaled frame + anchor onto the trailing WrapFloat that
            // placed() left at the top of the item stack (nothing has been
            // pushed since — same invariant wrap_fallback_reserve relies on).
            if let Some(Item::WrapFloat(f, _, yd)) = self.items.last_mut() {
                *f = scaled;
                *yd = y_delta;
            }
        }

        // General image float (no drop cap): continue the wrap into the following
        // paragraph(s). Unlike a drop cap, the band's physical bottom is the FLOAT's
        // own height (`band_bottom = float_height + clearance`), the inset is the
        // float's own width, and the inter-paragraph gap is the paragraph SPACING
        // (image floats keep their normal vertical rhythm). `pitch·N` boundary
        // rounding isn't a concern here — `band_bottom` is the image's arbitrary
        // height, so the lower edge lands mid-line.
        if want_active && dropcap_n.is_none() {
            let remaining = (band_bottom - lines_height).max(Abs::zero());
            active_band_to_set = Some(ActiveBand {
                remaining,
                inset: float_width + d.clearance,
                side,
                pitch: line_height + leading,
                leading,
                gap: spacing,
            });
        }

        self.flush_tags();
        self.rel(spacing.into(), 4);
        // GUARD 2 (per-line gate): emit through the same fit logic as
        // Child::Line. Narrow band lines are guaranteed to fit (Guard 1); the
        // first full-width continuation line that doesn't fit spills the rest
        // (carrying the trailing `spacing` so it lands on the final region).
        // When continuing the band, SUPPRESS the trailing paragraph spacing so
        // the continuation owns the single inter-paragraph (leading-sized) gap
        // inside the band (§10d).
        self.emit_wrapped_lines(line_children, leading, spacing, true, !want_active)?;

        if let Some(band) = active_band_to_set {
            // The opener completed in-region (no spill — emit_wrapped_lines
            // returns Err on spill, short-circuiting before here) and a
            // continuation follows: keep the band live and SKIP the compensating
            // gap. The continuation paragraph(s) consume `remaining`.
            self.active_band = Some(band);
            return Ok(());
        }

        // No continuation: today's behavior. The lines plus the trailing spacing
        // have advanced the offset by `lines_height + spacing`. If the float is
        // taller than that, emit a compensating gap so the TOTAL advance is
        // exactly `band_bottom` (no extra spacing on top), letting the next child
        // clear a taller-than-text float without overlap.
        let advanced = lines_height + spacing;
        if advanced < band_bottom {
            let gap = band_bottom - advanced;
            self.regions.size.y -= gap;
            self.items.push(Item::Abs(gap, 0));
        }
        Ok(())
    }

    /// Wrap a continuation paragraph beside the lower part of a still-live
    /// drop-cap band (set by `deferred_par` when the opener was too short to
    /// fill the full-N cap). The cap is already painted at the opener — this
    /// only narrows the lines level with the band's residual and updates
    /// `active_band`. Single-region: a spill clears the band (§10g).
    fn continuation_par(&mut self, d: &'b DeferredParChild<'a>) -> FlowResult<()> {
        // Defensive: the dispatch guard guarantees `active_band` is Some, but if
        // it somehow isn't, fall back to a plain full-width break. Copy it out so
        // `self` is free to be borrowed mutably for the rest of the method.
        let Some(mut band) = self.active_band else {
            return self.deferred_par_uniform(d);
        };

        let spacing = d.elem.spacing.resolve(d.styles);
        let full_width = self.regions.size.x;
        let leading = band.leading;

        // Own ONE inter-paragraph gap (`band.gap`): the leading for a drop cap so
        // the cap sits beside evenly-leaded lines (§10d); the paragraph spacing for
        // a general image float so wrapped paragraphs keep their normal rhythm. It
        // consumes band height like a line would, keeping the band aligned to the
        // float.
        self.flush_tags();
        self.rel(band.gap.into(), 4);
        band.remaining = (band.remaining - band.gap).max(Abs::zero());

        // Reuse the run's pitch (uniform across the run); do not re-probe.
        let line_height = band.pitch - leading;

        let profile = WrapProfile::new(
            ecow::eco_vec![ExclusionBand {
                y0: Abs::zero(),
                y1: band.remaining,
                inset: band.inset,
                side: band.side,
            }],
            full_width,
            line_height,
            leading,
        );

        let frames = crate::inline::layout_par(
            d.elem,
            self.composer.engine,
            d.locator.relayout(),
            d.styles,
            d.base,
            d.expand,
            d.par_situation,
            Some(&profile),
        )?
        .into_frames();

        // Same per-line pitch-variance guard as deferred_par: if a line's true
        // height diverges from the probed pitch, the band assignment is
        // unreliable — clear the band, reserve the residual float height, and
        // break this paragraph full-width.
        const EPS: f64 = 0.01;
        let pitch_varies = frames
            .iter()
            .any(|f| (f.height() - line_height).to_pt().abs() > EPS);
        if pitch_varies {
            self.active_band = None;
            self.regions.size.y -= band.remaining;
            self.items.push(Item::Abs(band.remaining, 0));
            return self.deferred_par_uniform(d);
        }

        // Apply the normal path's last-line content-hint tweak before building
        // line children (§10h).
        let mut frames = frames;
        if let Some(line) = frames.last_mut() {
            if line.content_hint() == '\0' {
                line.set_content_hint('\n');
            }
        }
        let frames = frames;
        let line_children = build_line_children(frames, leading, d.styles);

        let lines_height: Abs =
            line_children.iter().map(|l| l.frame.height()).sum::<Abs>()
                + leading * (line_children.len().saturating_sub(1) as f64);

        // The cap is already painted; emit the (possibly narrowed) lines with
        // their normal trailing paragraph spacing. A spill carries the rest
        // full-width and clears the band below.
        if let Err(stop) = self.emit_wrapped_lines(line_children, leading, spacing, true, true)
        {
            // A fatal error propagates unchanged. A region finish is a spill:
            // single-region, the band cannot continue into the spill region (the
            // float is painted here), so clear it (§10g). emit_wrapped_lines has
            // already stashed the wrap_spill and advanced the work head.
            self.active_band = None;
            return Err(stop);
        }

        // Consume the band; if too little remains for another beside-line, end
        // the run so the next paragraph flows full-width.
        band.remaining = (band.remaining - lines_height).max(Abs::zero());
        if band_is_exhausted(band.remaining, line_height) {
            self.active_band = None;
        } else {
            self.active_band = Some(band);
        }
        Ok(())
    }

    /// Fallback when a deferred paragraph has no usable pending wrap float (no
    /// stash, band doesn't fit the tail, or pitch varies). Breaks the paragraph
    /// full width and paginates it, like Collector::par's normal path.
    fn deferred_par_uniform(&mut self, d: &'b DeferredParChild<'a>) -> FlowResult<()> {
        let lines = crate::inline::layout_par(
            d.elem,
            self.composer.engine,
            d.locator.relayout(),
            d.styles,
            d.base,
            d.expand,
            d.par_situation,
            None,
        )?
        .into_frames();
        // Apply the normal path's last-line content-hint tweak (collect.rs
        // :219-222) so an over-deferred paragraph collapses weak spacing /
        // paginates identically to a non-deferred one (§10h).
        let mut lines = lines;
        if let Some(line) = lines.last_mut() {
            if line.content_hint() == '\0' {
                line.set_content_hint('\n');
            }
        }
        let lines = lines;
        let spacing = d.elem.spacing.resolve(d.styles);
        let leading = d.elem.leading.resolve(d.styles);
        let line_children = build_line_children(lines, leading, d.styles);
        self.flush_tags();
        self.rel(spacing.into(), 4);
        self.emit_wrapped_lines(line_children, leading, spacing, true, true)?;
        Ok(())
    }

    /// The Guard-1 fallback: the band can't complete in this region (or the
    /// region is cache-inconsistent / the pitch varies). Convert the trailing
    /// zero-advance `Item::WrapFloat` (pushed by `placed()`) into a
    /// height-reserving `Item::Frame` so following text flows BELOW the float
    /// (Step-4 behavior), then break the paragraph full-width and paginate it.
    fn wrap_fallback_reserve(
        &mut self,
        d: &'b DeferredParChild<'a>,
        frame: Frame,
        side: FixedAlignment,
    ) -> FlowResult<()> {
        // INVARIANT: placed() pushed the WrapFloat as the last item and nothing
        // has been pushed since, so it MUST be at the top of the stack here.
        // This assert makes a future regression that interleaves an item between
        // placed() and this fallback fail loudly, rather than silently leaving a
        // zero-advance WrapFloat in place and overlapping the following text.
        debug_assert!(
            matches!(self.items.last(), Some(Item::WrapFloat(..))),
            "wrap_fallback_reserve expects the WrapFloat to be the trailing item",
        );
        if matches!(self.items.last(), Some(Item::WrapFloat(..))) {
            self.items.pop();
            self.regions.size.y -= frame.height();
            self.items
                .push(Item::Frame(frame, Axes::new(side, FixedAlignment::Start)));
        }
        self.deferred_par_uniform(d)
    }

    /// Emit pre-broken lines through the same two-stage fit gate as
    /// `Child::Line`. On a no-fit, package the remaining lines (plus the
    /// trailing `spacing`) into a `WrapSpill` and finish the region so they
    /// continue (full-width) next region. On completion, emits the trailing
    /// paragraph spacing. Inserts inter-line leading between emitted lines.
    ///
    /// `advance_on_spill` must be `true` when called while a `Child::Deferred`
    /// is the work head (so it is consumed and not re-processed — its remaining
    /// work now lives in `wrap_spill`), and `false` when draining an existing
    /// `wrap_spill` (no work head to consume).
    ///
    /// `emit_trailing` is `true` everywhere except the active-band opener, which
    /// suppresses its trailing paragraph spacing so the continuation owns the
    /// single inter-paragraph gap inside the band (§10d). On a spill the trailing
    /// rel is never reached, so this flag is irrelevant there.
    fn emit_wrapped_lines(
        &mut self,
        lines: Vec<LineChild>,
        leading: Abs,
        spacing: Abs,
        advance_on_spill: bool,
        emit_trailing: bool,
    ) -> FlowResult<()> {
        let mut iter = lines.into_iter().enumerate();
        while let Some((i, line)) = iter.next() {
            if i > 0 {
                self.rel(leading.into(), 5);
            }
            // Same two-stage gate as self.line().
            let raw_no_fit = !self.regions.size.y.fits(line.frame.height())
                && self.regions.may_progress();
            let need_no_fit = !self.regions.size.y.fits(line.need)
                && self.regions.iter().nth(1).is_some_and(|r| r.y.fits(line.need));
            if raw_no_fit || need_no_fit {
                // Carry this line and the rest, full-width, to the next region.
                let mut rest = vec![line];
                rest.extend(iter.map(|(_, l)| l));
                self.composer.work.wrap_spill =
                    Some(WrapSpill { lines: rest, leading, spacing });
                // Consume the deferred work head so the next region drains the
                // spill instead of re-running deferred_par from scratch.
                if advance_on_spill {
                    self.composer.work.advance();
                }
                return Err(Stop::Finish(false));
            }
            self.frame(line.frame.clone(), line.align, false, false)?;
        }
        // All lines emitted on this region: emit the trailing paragraph spacing,
        // unless the caller (the active-band opener) owns it elsewhere.
        if emit_trailing {
            self.rel(spacing.into(), 4);
        }
        Ok(())
    }

    /// Drain carried full-width continuation lines onto a fresh region.
    fn wrap_spill(&mut self, spill: WrapSpill) -> FlowResult<()> {
        if self.regions.is_full() {
            self.composer.work.wrap_spill = Some(spill);
            return Err(Stop::Finish(false));
        }
        // Continuation lines are full-width; no float, no band. Re-gate them.
        // A continuation region starts at the top, so the leading-before-first
        // is suppressed by emit_wrapped_lines's `if i > 0`. The trailing
        // paragraph spacing is emitted by emit_wrapped_lines on completion.
        self.emit_wrapped_lines(spill.lines, spill.leading, spill.spacing, false, true)
    }

    /// Measure the true line pitch by breaking the paragraph once at uniform
    /// full width and reading the first non-empty line's frame height. Using the
    /// em would under-estimate the line box and keep too many lines beside the
    /// float. Costs one extra break; the wrap path is rare.
    fn measure_line_height(
        &mut self,
        d: &DeferredParChild<'a>,
        full_width: Abs,
    ) -> FlowResult<Abs> {
        let probe = crate::inline::layout_par(
            d.elem,
            self.composer.engine,
            d.locator.relayout(),
            d.styles,
            Size::new(full_width, self.regions.size.y),
            d.expand,
            d.par_situation,
            None,
        )?
        .into_frames();
        // Take the first NON-ZERO-height line so a leading tag/label line does
        // not collapse the pitch to zero.
        Ok(probe
            .iter()
            .map(|f| f.height())
            .find(|h| *h > Abs::zero())
            .unwrap_or_else(|| d.styles.resolve(TextElem::size)))
    }

    /// Processes a float flush.
    fn flush(&mut self) -> FlowResult<()> {
        // If there are still pending floats, finish the region instead of
        // adding more content to it.
        if !self.composer.work.floats.is_empty() {
            return Err(Stop::Finish(false));
        }
        Ok(())
    }

    /// Processes a column break.
    fn break_(&mut self, weak: bool) -> FlowResult<()> {
        // If there is a region to break into, break into it.
        if (!weak || !self.items.is_empty())
            && (!self.regions.backlog.is_empty() || self.regions.last.is_some())
        {
            self.composer.work.advance();
            return Err(Stop::Finish(true));
        }
        Ok(())
    }

    /// Arranges the produced items into an output frame.
    ///
    /// This performs alignment and resolves fractional spacing and blocks.
    fn finalize(
        mut self,
        region: Region,
        init: DistributionSnapshot<'a, 'b>,
        forced: bool,
    ) -> FlowResult<Frame> {
        if forced {
            // If this is the very end of the flow, flush pending tags.
            self.flush_tags();
        } else if !self.items.is_empty() && self.items.iter().all(Item::migratable) {
            // Restore the initial state of all items are migratable.
            self.restore(init);
        } else {
            // If we ended on a sticky block, but are not yet at the end of
            // the flow, restore the saved checkpoint to move the sticky
            // suffix to the next region.
            if let Some(snapshot) = self.sticky.take() {
                self.restore(snapshot)
            }
        }

        self.trim_spacing();

        let mut frs = Fr::zero();
        let mut used = Size::zero();
        let mut has_fr_child = false;

        // Determine the amount of used space and the sum of fractionals.
        for item in &self.items {
            match item {
                Item::Abs(v, _) => used.y += *v,
                Item::Fr(v, _, child) => {
                    frs += *v;
                    has_fr_child |= child.is_some();
                }
                Item::Frame(frame, _) => {
                    used.y += frame.height();
                    used.x.set_max(frame.width());
                }
                // A wrap float contributes no height/width; its space belongs
                // to the paragraph that wraps beside it.
                Item::Tag(_) | Item::Placed(..) | Item::WrapFloat(..) => {}
            }
        }

        // When we have fractional spacing, occupy the remaining space with it.
        let mut fr_space = Abs::zero();
        if frs.get() > 0.0 && region.size.y.is_finite() {
            fr_space = region.size.y - used.y;
            used.y = region.size.y;
        }

        // Lay out fractionally sized blocks.
        let mut fr_frames = vec![];
        if has_fr_child {
            for item in &self.items {
                let Item::Fr(v, _, Some(single)) = item else { continue };
                let length = v.share(frs, fr_space);
                let pod = Region::new(Size::new(region.size.x, length), region.expand);
                let frame = single.layout(self.composer.engine, pod)?;
                used.x.set_max(frame.width());
                fr_frames.push(frame);
            }
        }

        // Also consider the width of insertions for alignment.
        if !region.expand.x {
            used.x.set_max(self.composer.insertion_width());
        }

        // Determine the region's size.
        let size = region.expand.select(region.size, used.min(region.size));
        let free = size.y - used.y;

        let mut output = Frame::soft(size);
        let mut ruler = FixedAlignment::Start;
        let mut offset = Abs::zero();
        let mut fr_frames = fr_frames.into_iter();

        // Position all items.
        let mut baseline_set = false;
        for item in self.items {
            match item {
                Item::Tag(tag) => {
                    let y = offset + ruler.position(free);
                    let pos = Point::with_y(y);
                    output.push(pos, FrameItem::Tag(tag.clone()));
                }
                Item::Abs(v, _) => {
                    offset += v;
                }
                Item::Fr(v, _, single) => {
                    let length = v.share(frs, fr_space);
                    if let Some(single) = single {
                        let frame = fr_frames.next().unwrap();
                        let x = single.align.x.position(size.x - frame.width());
                        let pos = Point::new(x, offset);
                        output.push_frame(pos, frame);
                    }
                    offset += length;
                }
                Item::Frame(frame, align) => {
                    ruler = ruler.max(align.y);

                    let x = align.x.position(size.x - frame.width());
                    let y = offset + ruler.position(free);
                    let pos = Point::new(x, y);
                    offset += frame.height();

                    // The baseline of the whole region will be the set to the
                    // baseline of the first in-flow frame. For example, of the
                    // first paragraph, if there is more than one. But also,
                    // inside the paragraph itself, this will be the first line
                    // (since each line is laid out as a separate frame).
                    if !baseline_set {
                        if frame.has_baseline() {
                            output.set_baseline(y + frame.baseline());
                        }
                        baseline_set = true;
                    }

                    output.push_frame(pos, frame);
                }
                Item::Placed(frame, placed) => {
                    let x = placed.align_x.position(size.x - frame.width());
                    let y = match placed.align_y.unwrap_or_default() {
                        Some(align) => align.position(size.y - frame.height()),
                        _ => offset + ruler.position(free),
                    };

                    let pos = Point::new(x, y)
                        + placed.delta.zip_map(size, Rel::relative_to).to_point();

                    output.push_frame(pos, frame);
                }
                Item::WrapFloat(frame, align_x, y_delta) => {
                    // Draw the float at the current flow position WITHOUT
                    // advancing `offset`, so the following deferred paragraph's
                    // line 0 draws at this same y and wraps beside it. For a
                    // drop cap, `y_delta` pushes the (scaled) cap down so its
                    // cap-top registers on line 1's cap-top; it is zero for an
                    // ordinary wrap float, leaving that path byte-identical.
                    let x = align_x.position(size.x - frame.width());
                    let y = offset + y_delta;
                    output.push_frame(Point::new(x, y), frame);
                    // offset unchanged — deliberately.
                }
            }
        }

        Ok(output)
    }

    /// Create a snapshot of the work and items.
    fn snapshot(&self) -> DistributionSnapshot<'a, 'b> {
        DistributionSnapshot {
            work: self.composer.work.clone(),
            items: self.items.len(),
        }
    }

    /// Restore a snapshot of the work and items.
    fn restore(&mut self, snapshot: DistributionSnapshot<'a, 'b>) {
        *self.composer.work = snapshot.work;
        self.items.truncate(snapshot.items);
    }
}

/// Resolve the body cap-height for a paragraph's dominant text size, from the
/// first available font family. Used by the drop-cap anchor/scale math so the
/// registration is font-independent. Mirrors the metric-fetch pattern in
/// `inline/line.rs` (`apply_shift`) and `inline/shaping.rs`. A mixed-font opener
/// line uses the para's resolved (dominant) size — a documented limit.
fn body_cap_height(world: Tracked<dyn World + '_>, styles: StyleChain) -> Abs {
    let size = styles.resolve(TextElem::size);
    let variant = variant(styles);
    let variations = styles.get_cloned(TextElem::variations);
    families(styles)
        .find_map(|family| {
            world
                .book()
                .select(family.as_str(), variant)
                .and_then(|id| world.font(id))
                .map(|font| font.instantiate(variant, size, &variations))
        })
        .map(|font| font.metrics().cap_height.at(size))
        // No usable font: fall back to a typical cap-height ratio so the math
        // stays finite. Drop caps always sit on real prose, so this is a
        // defensive floor, not an expected path.
        .unwrap_or_else(|| size * 0.7)
}
