use std::cell::{LazyCell, RefCell};
use std::fmt::{self, Debug, Formatter};
use std::hash::Hash;

use bumpalo::Bump;
use bumpalo::boxed::Box as BumpBox;
use comemo::{Track, Tracked, TrackedMut};
use typst_library::diag::{SourceResult, bail, warning};
use typst_library::engine::{Engine, Route, Sink, Traced};
use typst_library::foundations::{Packed, Resolve, Smart, StyleChain};
use typst_library::introspection::{
    Introspector, Location, Locator, LocatorLink, SplitLocator, Tag, TagElem,
};
use typst_library::layout::{
    Abs, AlignElem, Alignment, Axes, BlockElem, ColbreakElem, FixedAlignment, FlushElem,
    Fr, Fragment, Frame, FrameParent, Inherit, PagebreakElem, PlaceElem, PlacementScope,
    Ratio, Region, Regions, Rel, Size, Sizing, Spacing, VElem,
};
use typst_library::model::ParElem;
use typst_library::routines::Pair;
use typst_library::text::TextElem;
use typst_library::{Library, World};
use typst_utils::{LazyHash, Protected, SliceExt};

use super::{FlowMode, layout_multi_block, layout_single_block};
use crate::inline::ParSituation;
use crate::modifiers::layout_and_modify;

/// Collects all elements of the flow into prepared children. These are much
/// simpler to handle than the raw elements.
#[typst_macros::time]
#[allow(clippy::too_many_arguments)]
pub fn collect<'a>(
    engine: &mut Engine,
    bump: &'a Bump,
    children: &[Pair<'a>],
    locator: Locator<'a>,
    base: Size,
    expand: bool,
    mode: FlowMode,
) -> SourceResult<Vec<Child<'a>>> {
    Collector {
        engine,
        bump,
        children,
        locator: locator.split(),
        base,
        expand,
        output: Vec::with_capacity(children.len()),
        par_situation: ParSituation::First,
        continuation_budget: 0,
    }
    .run(mode)
}

/// How many consecutive paragraphs after a general (non-drop-cap) wrap float may be
/// deferred to continue the wrap. A photo-sized float spans only a handful of
/// paragraphs; the distributor stops at band exhaustion, so this is just a ceiling
/// that bounds re-layout cost (over-deferred paragraphs break full-width). A drop cap
/// uses its exact line span `N` instead.
const GENERAL_WRAP_CONTINUATION_PARS: usize = 8;

/// State for collection.
struct Collector<'a, 'x, 'y> {
    engine: &'x mut Engine<'y>,
    bump: &'a Bump,
    children: &'x [Pair<'a>],
    base: Size,
    expand: bool,
    locator: SplitLocator<'a>,
    output: Vec<Child<'a>>,
    par_situation: ParSituation,
    /// Remaining paragraphs that may be chained as drop-cap continuations after
    /// a short opener. Set to `N` when an opener defers with `dropcap:Some(n)`;
    /// decremented as each consecutive paragraph chains; reset to `0` by any
    /// non-par / non-tag child (or an ordinary, non-dropcap wrap float) so a run
    /// never crosses a block boundary. v1 is drop-cap only.
    continuation_budget: usize,
}

impl<'a> Collector<'a, '_, '_> {
    /// Perform the collection.
    fn run(self, mode: FlowMode) -> SourceResult<Vec<Child<'a>>> {
        match mode {
            FlowMode::Root | FlowMode::Block => self.run_block(),
            FlowMode::Inline => self.run_inline(),
        }
    }

    /// Perform collection for block-level children.
    fn run_block(mut self) -> SourceResult<Vec<Child<'a>>> {
        for &(child, styles) in self.children {
            if let Some(elem) = child.to_packed::<TagElem>() {
                self.output.push(Child::Tag(&elem.tag));
            } else if let Some(elem) = child.to_packed::<VElem>() {
                self.v(elem, styles);
            } else if let Some(elem) = child.to_packed::<ParElem>() {
                self.par(elem, styles)?;
            } else if let Some(elem) = child.to_packed::<BlockElem>() {
                self.block(elem, styles);
            } else if let Some(elem) = child.to_packed::<PlaceElem>() {
                self.place(elem, styles)?;
            } else if child.is::<FlushElem>() {
                self.output.push(Child::Flush);
            } else if let Some(elem) = child.to_packed::<ColbreakElem>() {
                self.output.push(Child::Break(elem.weak.get(styles)));
                self.par_situation = ParSituation::First;
            } else if child.is::<PagebreakElem>() {
                bail!(
                    child.span(), "pagebreaks are not allowed inside of containers";
                    hint: "try using a `#colbreak()` instead";
                );
            } else {
                self.engine.sink.warn(warning!(
                    child.span(),
                    "{} was ignored during paged export",
                    child.func().name(),
                ));
            }
        }

        Ok(self.output)
    }

    /// Perform collection for inline-level children.
    fn run_inline(mut self) -> SourceResult<Vec<Child<'a>>> {
        // Extract leading and trailing tags.
        let (start, end) = self.children.split_prefix_suffix(|(c, _)| c.is::<TagElem>());
        let inner = &self.children[start..end];

        // Compute the shared styles.
        let styles = StyleChain::trunk_from_pairs(inner).unwrap_or_default();

        // Layout the lines.
        let lines = crate::inline::layout_inline(
            self.engine,
            inner,
            &mut self.locator,
            styles,
            self.base,
            self.expand,
        )?
        .into_frames();

        for (c, _) in &self.children[..start] {
            let elem = c.to_packed::<TagElem>().unwrap();
            self.output.push(Child::Tag(&elem.tag));
        }

        let leading = styles.resolve(ParElem::leading);
        self.lines(lines, leading, styles);

        for (c, _) in &self.children[end..] {
            let elem = c.to_packed::<TagElem>().unwrap();
            self.output.push(Child::Tag(&elem.tag));
        }

        Ok(self.output)
    }

    /// Collect vertical spacing into a relative or fractional child.
    fn v(&mut self, elem: &'a Packed<VElem>, styles: StyleChain<'a>) {
        // Any explicit spacing breaks a drop-cap continuation run.
        self.continuation_budget = 0;
        self.output.push(match elem.amount {
            Spacing::Rel(rel) => {
                Child::Rel(rel.resolve(styles), elem.weak.get(styles) as u8)
            }
            Spacing::Fr(fr) => Child::Fr(fr, elem.weak.get(styles) as u8),
        });
    }

    /// Collect a paragraph into [`LineChild`]ren. This already performs line
    /// layout since it is not dependent on the concrete regions.
    fn par(
        &mut self,
        elem: &'a Packed<ParElem>,
        styles: StyleChain<'a>,
    ) -> SourceResult<()> {
        // Defer iff the immediately-preceding child (skipping only Tags) is a
        // side-wrap float. Any intervening Rel/Flush/Frame fails this match and
        // falls through to the normal under-the-float path (POC scope guard).
        let prev_float = self
            .output
            .iter()
            .rev()
            .find(|c| !matches!(c, Child::Tag(_)))
            .and_then(|c| match c {
                Child::Placed(p) if p.float && p.wrap => {
                    // Mark that a deferred wrapping paragraph follows this
                    // float, so the distributor reserves zero height for it
                    // (rather than the Step-4 under-the-float fallback).
                    p.wrap_active.set(true);
                    // Carry the float's drop-cap span (if any) onto the
                    // deferred paragraph so the band builder can reserve N
                    // lines and anchor/scale the cap (Tier 1 + Tier 2).
                    Some((p.clearance, p.elem.dropcap.get(p.styles)))
                }
                _ => None,
            });

        if let Some((clearance, dropcap)) = prev_float {
            // NOTE: this next() MUST stay here, before the push and before the
            // early return — it preserves SplitLocator ordering identical to
            // the non-deferred path so in-paragraph labels/refs resolve to the
            // same Location with or without the float. Do not move it.
            let locator = self.locator.next(&elem.span());
            self.output.push(Child::Deferred(self.boxed(DeferredParChild {
                elem,
                styles,
                locator,
                base: self.base,
                expand: self.expand,
                par_situation: self.par_situation,
                clearance,
                dropcap,
                continuation: false,
            })));
            // Open a wrap-float continuation run. A drop cap bounds it by N (≤ N
            // lines fit in ≤ N paragraphs — exact). A general image float can't know
            // its line-height at collect time, so it gets a small static cap; the
            // distributor stops the moment the band is exhausted (remaining ≤ 0) and
            // any over-deferred tail just breaks full-width, so a float taller than
            // the cap simply clears its remainder (graceful degradation).
            self.continuation_budget = dropcap.unwrap_or(GENERAL_WRAP_CONTINUATION_PARS);
            self.par_situation = ParSituation::Consecutive;
            return Ok(());
        }

        // Drop-cap continuation: this paragraph immediately follows a deferred
        // opener (or an earlier continuation), with budget remaining. Defer it
        // too so the distributor can wrap it beside the lower part of the cap.
        // The immediately-preceding non-Tag child being a `Deferred` guarantees
        // no intervening block ended the run (the normal-par path below resets
        // the budget, and block()/v()/place() reset it as well).
        if self.continuation_budget > 0
            && self
                .output
                .iter()
                .rev()
                .find(|c| !matches!(c, Child::Tag(_)))
                .is_some_and(|c| matches!(c, Child::Deferred(_)))
        {
            let locator = self.locator.next(&elem.span());
            self.output.push(Child::Deferred(self.boxed(DeferredParChild {
                elem,
                styles,
                locator,
                base: self.base,
                expand: self.expand,
                par_situation: self.par_situation,
                // Continuation builds its band from `active_band`, whose `inset`
                // already folds in the opener's clearance; this field is unused
                // on the continuation path, so zero is correct.
                clearance: Abs::zero(),
                dropcap: None,
                continuation: true,
            })));
            self.continuation_budget -= 1;
            self.par_situation = ParSituation::Consecutive;
            return Ok(());
        }

        // Normal paragraph: any non-continuing paragraph ends a drop-cap run.
        self.continuation_budget = 0;

        let lines = crate::inline::layout_par(
            elem,
            self.engine,
            self.locator.next(&elem.span()),
            styles,
            self.base,
            self.expand,
            self.par_situation,
            None,
        )?
        .into_frames();

        let mut lines = lines;
        if let Some(line) = lines.last_mut() {
            if line.content_hint() == '\0' {
                line.set_content_hint('\n');
            }
        }
        let lines = lines;

        let spacing = elem.spacing.resolve(styles);
        let leading = elem.leading.resolve(styles);

        self.output.push(Child::Rel(spacing.into(), 4));

        self.lines(lines, leading, styles);

        self.output.push(Child::Rel(spacing.into(), 4));
        self.par_situation = ParSituation::Consecutive;

        Ok(())
    }

    /// Collect laid-out lines.
    fn lines(&mut self, lines: Vec<Frame>, leading: Abs, styles: StyleChain<'a>) {
        let line_children = build_line_children(lines, leading, styles);
        for (i, line) in line_children.into_iter().enumerate() {
            if i > 0 {
                self.output.push(Child::Rel(leading.into(), 5));
            }
            self.output.push(Child::Line(self.boxed(line)));
        }
    }

    /// Collect a block into a [`SingleChild`] or [`MultiChild`] depending on
    /// whether it is breakable.
    fn block(&mut self, elem: &'a Packed<BlockElem>, styles: StyleChain<'a>) {
        // A block ends any drop-cap continuation run.
        self.continuation_budget = 0;
        let locator = self.locator.next(&elem.span());
        let align = styles.resolve(AlignElem::alignment);
        let alone = self.children.len() == 1;
        let sticky = elem.sticky.get(styles);
        let breakable = elem.breakable.get(styles);
        let fr = match elem.height.get(styles) {
            Sizing::Fr(fr) => Some(fr),
            _ => None,
        };

        let fallback = LazyCell::new(|| styles.resolve(ParElem::spacing));
        let spacing = |amount| match amount {
            Smart::Auto => Child::Rel((*fallback).into(), 4),
            Smart::Custom(Spacing::Rel(rel)) => Child::Rel(rel.resolve(styles), 3),
            Smart::Custom(Spacing::Fr(fr)) => Child::Fr(fr, 2),
        };

        self.output.push(spacing(elem.above.get(styles)));

        if !breakable || fr.is_some() {
            self.output.push(Child::Single(self.boxed(SingleChild {
                align,
                sticky,
                alone,
                fr,
                elem,
                styles,
                locator,
                cell: CachedCell::new(),
            })));
        } else {
            self.output.push(Child::Multi(self.boxed(MultiChild {
                align,
                sticky,
                alone,
                elem,
                styles,
                locator,
                cell: CachedCell::new(),
            })));
        };

        self.output.push(spacing(elem.below.get(styles)));
        self.par_situation = ParSituation::Other;
    }

    /// Collects a placed element into a [`PlacedChild`].
    fn place(
        &mut self,
        elem: &'a Packed<PlaceElem>,
        styles: StyleChain<'a>,
    ) -> SourceResult<()> {
        // A placed element ends any drop-cap continuation run; if it is itself a
        // wrap float, `par()` re-opens the run for its own opener.
        self.continuation_budget = 0;
        let alignment = elem.alignment.get(styles);
        let align_x = alignment.map_or(FixedAlignment::Center, |align| {
            align.x().unwrap_or_default().resolve(styles)
        });
        let align_y = alignment.map(|align| align.y().map(|y| y.resolve(styles)));
        let scope = elem.scope.get(styles);
        let float = elem.float.get(styles);
        let wrap = elem.wrap.get(styles);

        if wrap {
            // Side-wrapping floats reflow surrounding text into a narrowed
            // measure. They anchor in place (no vertical alignment, i.e. at the
            // current flow position) on a definite horizontal side, within a
            // single column. These rules supersede the general float rules
            // below, since an in-place vertical anchor is otherwise rejected.
            if !float {
                bail!(
                    elem.span(),
                    "wrapping placement is only available for floating placement";
                    hint: "you can enable floating placement with `place(float: true, ..)`";
                );
            }
            if !matches!(align_x, FixedAlignment::Start | FixedAlignment::End) {
                bail!(
                    elem.span(),
                    "wrapping placement requires a `left` or `right` alignment"
                );
            }
            if !matches!(align_y, Smart::Custom(None)) {
                bail!(
                    elem.span(),
                    "wrapping placement anchors the float at its position in the text";
                    hint: "remove the `top`/`bottom`/`horizon` vertical alignment";
                );
            }
            if scope == PlacementScope::Parent {
                bail!(
                    elem.span(),
                    "wrapping placement is not supported for parent scope"
                );
            }
        } else {
            match (float, align_y) {
                (true, Smart::Custom(None | Some(FixedAlignment::Center))) => bail!(
                    elem.span(),
                    "vertical floating placement must be `auto`, `top`, or `bottom`"
                ),
                (false, Smart::Auto) => bail!(
                    elem.span(),
                    "automatic positioning is only available for floating placement";
                    hint: "you can enable floating placement with `place(float: true, ..)`";
                ),
                _ => {}
            }

            if !float && scope == PlacementScope::Parent {
                bail!(
                    elem.span(),
                    "parent-scoped positioning is currently only available for floating placement";
                    hint: "you can enable floating placement with `place(float: true, ..)`";
                );
            }
        }

        let locator = self.locator.next(&elem.span());
        let clearance = elem.clearance.resolve(styles);
        let delta = Axes::new(elem.dx.get(styles), elem.dy.get(styles)).resolve(styles);
        self.output.push(Child::Placed(self.boxed(PlacedChild {
            align_x,
            align_y,
            scope,
            float,
            wrap,
            wrap_active: std::cell::Cell::new(false),
            clearance,
            delta,
            elem,
            styles,
            locator,
            alignment,
            cell: CachedCell::new(),
        })));

        Ok(())
    }

    /// Wraps a value in a bump-allocated box to reduce its footprint in the
    /// [`Child`] enum.
    fn boxed<T>(&self, value: T) -> BumpBox<'a, T> {
        BumpBox::new_in(value, self.bump)
    }
}

/// Build [`LineChild`]ren from laid-out line frames, computing the widow/orphan
/// `need` for each line. Shared by `Collector::lines` (the normal paragraph
/// path) and the distributor's deferred-wrap paths so that the `need` ladder is
/// byte-identical across both. The caller is responsible for inserting the
/// inter-line `Child::Rel`/spacing between consecutive children.
pub(crate) fn build_line_children(
    lines: Vec<Frame>,
    leading: Abs,
    styles: StyleChain,
) -> Vec<LineChild> {
    let align = styles.resolve(AlignElem::alignment);
    let costs = styles.get(TextElem::costs);

    // Determine whether to prevent widow and orphans.
    let len = lines.len();
    let prevent_orphans =
        costs.orphan() > Ratio::zero() && len >= 2 && !lines[1].is_empty();
    let prevent_widows =
        costs.widow() > Ratio::zero() && len >= 2 && !lines[len - 2].is_empty();
    let prevent_all = len == 3 && prevent_orphans && prevent_widows;

    // Store the heights of lines at the edges because we'll potentially
    // need these later when `lines` is already moved.
    let height_at = |i| lines.get(i).map(Frame::height).unwrap_or_default();
    let front_1 = height_at(0);
    let front_2 = height_at(1);
    let back_2 = height_at(len.saturating_sub(2));
    let back_1 = height_at(len.saturating_sub(1));

    let mut output = Vec::with_capacity(len);
    for (i, frame) in lines.into_iter().enumerate() {
        // To prevent widows and orphans, we require enough space for
        // - all lines if it's just three
        // - the first two lines if we're at the first line
        // - the last two lines if we're at the second to last line
        let need = if prevent_all && i == 0 {
            front_1 + leading + front_2 + leading + back_1
        } else if prevent_orphans && i == 0 {
            front_1 + leading + front_2
        } else if prevent_widows && i >= 2 && i + 2 == len {
            back_2 + leading + back_1
        } else {
            frame.height()
        };

        output.push(LineChild { frame, align, need });
    }
    output
}

/// A prepared child in flow layout.
///
/// The larger variants are bump-boxed to keep the enum size down.
#[derive(Debug)]
pub enum Child<'a> {
    /// An introspection tag.
    Tag(&'a Tag),
    /// Relative spacing with a specific weakness level.
    Rel(Rel<Abs>, u8),
    /// Fractional spacing with a specific weakness level.
    Fr(Fr, u8),
    /// An already layouted line of a paragraph.
    Line(BumpBox<'a, LineChild>),
    /// An unbreakable block.
    Single(BumpBox<'a, SingleChild<'a>>),
    /// A breakable block.
    Multi(BumpBox<'a, MultiChild<'a>>),
    /// An absolutely or floatingly placed element.
    Placed(BumpBox<'a, PlacedChild<'a>>),
    /// A paragraph deferred for break-beside-float in the distributor.
    Deferred(BumpBox<'a, DeferredParChild<'a>>),
    /// A place flush.
    Flush,
    /// An explicit column break.
    Break(bool),
}

/// A child that encapsulates a layouted line of a paragraph.
#[derive(Debug, Clone)]
pub struct LineChild {
    pub frame: Frame,
    pub align: Axes<FixedAlignment>,
    pub need: Abs,
}

/// A paragraph deferred because it immediately follows a side-wrap float.
/// Broken in the distributor once the float's height is known. POC scope:
/// nothing but tags sits between the float and this paragraph.
#[derive(Debug)]
pub struct DeferredParChild<'a> {
    pub elem: &'a Packed<ParElem>,
    pub styles: StyleChain<'a>,
    pub locator: Locator<'a>,
    pub base: Size,
    pub expand: bool,
    pub par_situation: ParSituation,
    /// Clearance copied from the preceding float.
    pub clearance: Abs,
    /// Drop-cap span (number of lines) copied from the preceding wrap float, if
    /// it was a drop cap. `None` for ordinary side-wrap floats.
    pub dropcap: Option<usize>,
    /// `false` for the opener paragraph (immediately after the float, sets the
    /// band); `true` for a chained continuation paragraph that wraps beside the
    /// lower part of the same (full-N-tall) cap. The distributor routes a
    /// continuation through `continuation_par` (band from `active_band`), the
    /// opener through `deferred_par` (band from `pending_wrap_float`).
    pub continuation: bool,
}

/// A child that encapsulates a prepared unbreakable block.
#[derive(Debug)]
pub struct SingleChild<'a> {
    pub align: Axes<FixedAlignment>,
    pub sticky: bool,
    pub alone: bool,
    pub fr: Option<Fr>,
    elem: &'a Packed<BlockElem>,
    styles: StyleChain<'a>,
    locator: Locator<'a>,
    cell: CachedCell<SourceResult<Frame>>,
}

impl SingleChild<'_> {
    /// Build the child's frame given the region's base size.
    pub fn layout(&self, engine: &mut Engine, region: Region) -> SourceResult<Frame> {
        self.cell.get_or_init(region, |mut region| {
            // Vertical expansion is only kept if this block is the only child.
            region.expand.y &= self.alone;
            layout_single_impl(
                engine.world,
                engine.library,
                engine.introspector.into_raw(),
                engine.traced,
                TrackedMut::reborrow_mut(&mut engine.sink),
                engine.route.track(),
                self.elem,
                self.locator.track(),
                self.styles,
                region,
            )
        })
    }
}

/// The cached, internal implementation of [`SingleChild::layout`].
#[comemo::memoize]
#[allow(clippy::too_many_arguments)]
fn layout_single_impl(
    world: Tracked<dyn World + '_>,
    library: &LazyHash<Library>,
    introspector: Tracked<dyn Introspector + '_>,
    traced: Tracked<Traced>,
    sink: TrackedMut<Sink>,
    route: Tracked<Route>,
    elem: &Packed<BlockElem>,
    locator: Tracked<Locator>,
    styles: StyleChain,
    region: Region,
) -> SourceResult<Frame> {
    let introspector = Protected::from_raw(introspector);
    let link = LocatorLink::new(locator);
    let locator = Locator::link(&link);
    let mut engine = Engine {
        library,
        world,
        introspector,
        traced,
        sink,
        route: Route::extend(route),
    };

    layout_and_modify(styles, |styles| {
        layout_single_block(elem, &mut engine, locator, styles, region).map(
            |mut frame| {
                if !frame.is_empty() {
                    frame.set_content_hint('\n');
                }
                frame
            },
        )
    })
}

/// A child that encapsulates a prepared breakable block.
#[derive(Debug)]
pub struct MultiChild<'a> {
    pub align: Axes<FixedAlignment>,
    pub sticky: bool,
    alone: bool,
    elem: &'a Packed<BlockElem>,
    styles: StyleChain<'a>,
    locator: Locator<'a>,
    cell: CachedCell<SourceResult<Fragment>>,
}

impl<'a> MultiChild<'a> {
    /// Build the child's frames given regions.
    pub fn layout<'b>(
        &'b self,
        engine: &mut Engine,
        regions: Regions,
    ) -> SourceResult<(Frame, Option<MultiSpill<'a, 'b>>)> {
        let fragment = self.layout_full(engine, regions)?;
        let exist_non_empty_frame = fragment.iter().any(|f| !f.is_empty());

        // Extract the first frame.
        let mut frames = fragment.into_iter();
        let frame = frames.next().unwrap();

        // If there's more, return a `spill`.
        let mut spill = None;
        if frames.next().is_some() {
            spill = Some(MultiSpill {
                exist_non_empty_frame,
                multi: self,
                full: regions.full,
                first: regions.size.y,
                backlog: vec![],
                min_backlog_len: regions.backlog.len(),
            });
        }

        Ok((frame, spill))
    }

    /// The shared internal implementation of [`Self::layout`] and
    /// [`MultiSpill::layout`].
    fn layout_full(
        &self,
        engine: &mut Engine,
        regions: Regions,
    ) -> SourceResult<Fragment> {
        self.cell.get_or_init(regions, |mut regions| {
            // Vertical expansion is only kept if this block is the only child.
            regions.expand.y &= self.alone;
            layout_multi_impl(
                engine.world,
                engine.library,
                engine.introspector.into_raw(),
                engine.traced,
                TrackedMut::reborrow_mut(&mut engine.sink),
                engine.route.track(),
                self.elem,
                self.locator.track(),
                self.styles,
                regions,
            )
        })
    }
}

/// The cached, internal implementation of [`MultiChild::layout_full`].
#[comemo::memoize]
#[allow(clippy::too_many_arguments)]
fn layout_multi_impl(
    world: Tracked<dyn World + '_>,
    library: &LazyHash<Library>,
    introspector: Tracked<dyn Introspector + '_>,
    traced: Tracked<Traced>,
    sink: TrackedMut<Sink>,
    route: Tracked<Route>,
    elem: &Packed<BlockElem>,
    locator: Tracked<Locator>,
    styles: StyleChain,
    regions: Regions,
) -> SourceResult<Fragment> {
    let introspector = Protected::from_raw(introspector);
    let link = LocatorLink::new(locator);
    let locator = Locator::link(&link);
    let mut engine = Engine {
        library,
        world,
        introspector,
        traced,
        sink,
        route: Route::extend(route),
    };

    layout_and_modify(styles, |styles| {
        layout_multi_block(elem, &mut engine, locator, styles, regions)
    })
}

/// The spilled remains of a `MultiChild` that broke across two regions.
#[derive(Debug, Clone)]
pub struct MultiSpill<'a, 'b> {
    pub(super) exist_non_empty_frame: bool,
    multi: &'b MultiChild<'a>,
    first: Abs,
    full: Abs,
    backlog: Vec<Abs>,
    min_backlog_len: usize,
}

impl MultiSpill<'_, '_> {
    /// Build the spill's frames given regions.
    pub fn layout(
        mut self,
        engine: &mut Engine,
        regions: Regions,
    ) -> SourceResult<(Frame, Option<Self>)> {
        // The first region becomes unchangeable and committed to our backlog.
        self.backlog.push(regions.size.y);

        // The remaining regions are ephemeral and may be replaced.
        let mut backlog: Vec<_> =
            self.backlog.iter().chain(regions.backlog).copied().collect();

        // Remove unnecessary backlog items to prevent it from growing
        // unnecessarily, changing the region's hash.
        while backlog.len() > self.min_backlog_len
            && backlog.last().copied() == regions.last
        {
            backlog.pop();
        }

        // Build the pod with the merged regions.
        let pod = Regions {
            size: Size::new(regions.size.x, self.first),
            expand: regions.expand,
            full: self.full,
            backlog: &backlog,
            last: regions.last,
        };

        // Extract the not-yet-processed frames.
        let mut frames = self
            .multi
            .layout_full(engine, pod)?
            .into_iter()
            .skip(self.backlog.len());

        // Ensure that the backlog never shrinks, so that unwrapping below is at
        // least fairly safe. Note that the whole region juggling here is
        // fundamentally not ideal: It is a compatibility layer between the old
        // (all regions provided upfront) & new (each region provided on-demand,
        // like an iterator) layout model. This approach is not 100% correct, as
        // in the old model later regions could have an effect on earlier
        // frames, but it's the best we can do for now, until the multi
        // layouters are refactored to the new model.
        self.min_backlog_len = self.min_backlog_len.max(backlog.len());

        // Save the first frame.
        let frame = frames.next().unwrap();

        // If there's more, return a `spill`.
        let mut spill = None;
        if frames.next().is_some() {
            spill = Some(self);
        }

        Ok((frame, spill))
    }

    /// The alignment of the breakable block.
    pub fn align(&self) -> Axes<FixedAlignment> {
        self.multi.align
    }
}

/// Remaining full-width lines of a wrapped paragraph, carried to the next
/// region. Lines are already broken (continuation lines are full-width by
/// construction); we only need to re-gate them against the new region. Mirrors
/// `MultiSpill`'s role for the deferred side-wrap-float paragraph path.
#[derive(Debug, Clone)]
pub struct WrapSpill {
    /// Pre-broken, full-width continuation line frames (in order).
    pub lines: Vec<LineChild>,
    /// Leading to insert between consecutive lines (`Rel` weakness 5).
    pub leading: Abs,
    /// Trailing paragraph spacing to emit after the last line (`Rel` weakness 4).
    pub spacing: Abs,
}

/// A child that encapsulates a prepared placed element.
#[derive(Debug)]
pub struct PlacedChild<'a> {
    pub align_x: FixedAlignment,
    pub align_y: Smart<Option<FixedAlignment>>,
    pub scope: PlacementScope,
    pub float: bool,
    pub wrap: bool,
    /// Set during collection iff a `Child::Deferred` paragraph was produced for
    /// this side-wrap float (i.e. a wrapping paragraph immediately follows it).
    /// Interior-mutable so the immutable `Collector::par` scan can set it; it is
    /// never hashed (PlacedChild already holds a non-hashed `CachedCell`).
    pub wrap_active: std::cell::Cell<bool>,
    pub clearance: Abs,
    pub delta: Axes<Rel<Abs>>,
    elem: &'a Packed<PlaceElem>,
    styles: StyleChain<'a>,
    locator: Locator<'a>,
    alignment: Smart<Alignment>,
    cell: CachedCell<SourceResult<Frame>>,
}

impl PlacedChild<'_> {
    /// Build the child's frame given the region's base size.
    pub fn layout(&self, engine: &mut Engine, base: Size) -> SourceResult<Frame> {
        self.cell.get_or_init(base, |base| {
            let align = self.alignment.unwrap_or_else(|| Alignment::CENTER);
            let aligned = AlignElem::alignment.set(align).wrap();
            let styles = self.styles.chain(&aligned);

            let mut frame = layout_and_modify(styles, |styles| {
                crate::layout_frame(
                    engine,
                    &self.elem.body,
                    self.locator.relayout(),
                    styles,
                    Region::new(base, Axes::splat(false)),
                )
            })?;

            if self.float {
                frame.set_parent(FrameParent::new(
                    self.elem.location().unwrap(),
                    Inherit::Yes,
                ));
            }

            Ok(frame)
        })
    }

    /// The element's location.
    pub fn location(&self) -> Location {
        self.elem.location().unwrap()
    }
}

/// Wraps a parameterized computation and caches its latest output.
///
/// - When the computation is performed multiple times consecutively with the
///   same argument, reuses the cache.
/// - When the argument changes, the new output is cached.
#[derive(Clone)]
struct CachedCell<T>(RefCell<Option<(u128, T)>>);

impl<T> CachedCell<T> {
    /// Create an empty cached cell.
    fn new() -> Self {
        Self(RefCell::new(None))
    }

    /// Perform the computation `f` with caching.
    fn get_or_init<F, I>(&self, input: I, f: F) -> T
    where
        I: Hash,
        T: Clone,
        F: FnOnce(I) -> T,
    {
        let input_hash = typst_utils::hash128(&input);

        let mut slot = self.0.borrow_mut();
        if let Some((hash, output)) = &*slot
            && *hash == input_hash
        {
            return output.clone();
        }

        let output = f(input);
        *slot = Some((input_hash, output.clone()));
        output
    }
}

impl<T> Default for CachedCell<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> Debug for CachedCell<T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.pad("CachedCell(..)")
    }
}
