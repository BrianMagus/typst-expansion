use typst_library::introspection::Tag;
use typst_library::layout::{
    Abs, Axes, FixedAlignment, Fr, Frame, FrameItem, Point, Region, Regions, Rel, Size,
};
use typst_utils::Numeric;

use typst_library::text::TextElem;

use super::collect::build_line_children;
use super::wrap::{ExclusionBand, WrapProfile};
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
    /// so the following deferred paragraph's lines overlap its top band.
    WrapFloat(Frame, FixedAlignment),
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
        match child {
            Child::Tag(tag) => self.tag(tag),
            Child::Rel(amount, weakness) => self.rel(*amount, *weakness),
            Child::Fr(fr, weakness) => self.fr(*fr, *weakness),
            Child::Line(line) => self.line(line)?,
            Child::Single(single) => self.single(single)?,
            Child::Multi(multi) => self.multi(multi)?,
            Child::Placed(placed) => self.placed(placed)?,
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
            self.items.push(Item::WrapFloat(frame, placed.align_x));
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
        let band_bottom = float_height + d.clearance;

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

        let band = ExclusionBand {
            y0: Abs::zero(),
            y1: band_bottom,
            inset: float_width + d.clearance,
            side, // the stashed float's align_x
        };

        let leading = d.elem.leading.resolve(d.styles);
        let full_width = self.regions.size.x;

        // Measure the true line pitch, then break the paragraph beside the float.
        let line_height = self.measure_line_height(d, full_width)?;

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
        // paragraph (so following content clears a taller-than-paragraph float).
        let lines_height: Abs =
            line_children.iter().map(|l| l.frame.height()).sum::<Abs>()
                + leading * (line_children.len().saturating_sub(1) as f64);

        self.flush_tags();
        self.rel(spacing.into(), 4);
        // GUARD 2 (per-line gate): emit through the same fit logic as
        // Child::Line. Narrow band lines are guaranteed to fit (Guard 1); the
        // first full-width continuation line that doesn't fit spills the rest
        // (carrying the trailing `spacing` so it lands on the final region).
        self.emit_wrapped_lines(line_children, leading, spacing, true)?;
        // If we reach here, the whole paragraph fit (no spill). The lines plus
        // the trailing spacing have advanced the offset by `lines_height +
        // spacing`. If the float is taller than that, emit a compensating gap so
        // the TOTAL advance is exactly `band_bottom` (no extra spacing on top),
        // letting the next child clear a taller-than-text float without overlap.
        let advanced = lines_height + spacing;
        if advanced < band_bottom {
            let gap = band_bottom - advanced;
            self.regions.size.y -= gap;
            self.items.push(Item::Abs(gap, 0));
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
        let spacing = d.elem.spacing.resolve(d.styles);
        let leading = d.elem.leading.resolve(d.styles);
        let line_children = build_line_children(lines, leading, d.styles);
        self.flush_tags();
        self.rel(spacing.into(), 4);
        self.emit_wrapped_lines(line_children, leading, spacing, true)?;
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
    fn emit_wrapped_lines(
        &mut self,
        lines: Vec<LineChild>,
        leading: Abs,
        spacing: Abs,
        advance_on_spill: bool,
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
        // All lines emitted on this region: emit the trailing paragraph spacing.
        self.rel(spacing.into(), 4);
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
        self.emit_wrapped_lines(spill.lines, spill.leading, spill.spacing, false)
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
                Item::WrapFloat(frame, align_x) => {
                    // Draw the float at the current flow position WITHOUT
                    // advancing `offset`, so the following deferred paragraph's
                    // line 0 draws at this same y and wraps beside it.
                    let x = align_x.position(size.x - frame.width());
                    let y = offset;
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
