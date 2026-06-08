//! Live-tunable forest settings, matching `tunables` / `tunableSpecs` in
//! `minigames/tree.lua` (the 13 original sliders + the "New forest" button) —
//! plus this port's additions: **Wind** (the Lua forest is static; 0 = static,
//! the original's behaviour), **Leaves** + their size/density/hue, and
//! **Size variation** (per-tree scale jitter for depth; 0 = uniform, the
//! original). Every addition zeroes back to the faithful original.
//!
//! The tunables fall into three cost buckets (see `sim`):
//! - **structural** (growth + the 5 rewrite probs) ⇒ regrow the L-system token
//!   streams (throttled while dragging, the original's `REGROW_THROTTLE`);
//! - **geometry** (branch angle/length/width, size variation, leaf size/density)
//!   ⇒ re-walk the turtle and re-emit triangles (no regrow, also throttled);
//! - **colour/wind** (hue, hue spread, brightness, leaf hue, wind) ⇒ pure shader
//!   uniforms, so those sliders update live for free — no regrow, no re-upload.

use bevy::prelude::*;

use crate::ui::SliderBinding;

/// One forest's worth of tunables — a single *kind* of tree, grown many times.
/// The scene holds a list of these (see [`ForestSettings`]); each new forest
/// gets a [randomised](TreeParams::random) set so it's a visibly different tree.
/// Wind is NOT here — one breeze blows through the whole scene (see
/// [`ForestSettings::wind`]).
#[derive(Clone)]
pub struct TreeParams {
    // Structure (regrow the L-system token stream when changed).
    pub count: f32,         // number of trees
    pub growth: f32,        // L-system iterations (the string grows ~2x/level)
    pub no_expand: f32,     // chance a node rests this iteration
    pub forward: f32,       // chance an F segment extends (F -> FF)
    pub branch_left: f32,   // relative weight of the left-leaning branch shape
    pub branch_right: f32,  // relative weight of the right-leaning shape
    pub branch_center: f32, // relative weight of the symmetric three-way shape
    // Geometry (re-walk the turtle when changed; no regrow).
    pub branch_angle: f32,    // angle between child branches (x pi)
    pub branch_length: f32,   // segment length (px)
    pub trunk_width: f32,     // base branch width (px); tapers toward the tips
    pub size_variation: f32,  // per-tree scale jitter (port addition; 0 = uniform)
    // Colour (per-forest shader uniform; free live update — one small draw each).
    pub hue: f32,        // base hue at the trunk (degrees) — original red default
    pub hue_spread: f32, // hue rotation per branch level (degrees)
    pub brightness: f32, // base lightness at the trunk (%); brightens outward
    // Leaves (port addition; OFF preserves the faithful bare L-system).
    pub leaves: bool,
    pub leaf_size: f32,    // leaf radius (px)
    pub leaf_density: f32, // fraction of twig tips that sprout a leaf (0..1)
    pub leaf_hue: f32,     // leaf hue (degrees)
}

impl Default for TreeParams {
    fn default() -> Self {
        Self {
            // The original `tree.lua` defaults.
            count: 5.0,
            growth: 12.0,
            no_expand: 0.10,
            forward: 0.40,
            branch_left: 0.50,
            branch_right: 0.50,
            branch_center: 0.00,
            branch_angle: 0.10,
            branch_length: 5.0,
            trunk_width: 15.0,
            hue: 0.0,
            hue_spread: 1.0,
            brightness: 10.0,
            // Port additions, all at their faithful zero-position.
            size_variation: 0.0,
            leaves: false,
            leaf_size: 6.0,
            leaf_density: 0.6,
            leaf_hue: 110.0,
        }
    }
}

impl TreeParams {
    /// A randomised-but-sane parameter set for a freshly added forest, so the
    /// "+" button always yields a visibly different *kind* of tree the user then
    /// tunes. Every field lands inside its slider [`range`](ForestParam::range)
    /// so nothing is degenerate or un-editable; the structure stays plausible (a
    /// real-ish tree, not a hairball). `seed` makes it deterministic for tests.
    pub fn random(seed: u64) -> Self {
        // A tiny local splitmix stream — no dependency on the sim's RNG, and
        // self-contained so the UI layer needn't reach into `sim`.
        let mut state = seed | 1;
        let mut next = || {
            state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = state;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            ((z ^ (z >> 31)) >> 40) as f32 / (1u64 << 24) as f32 // [0,1)
        };
        let lerp = |min: f32, max: f32, t: f32| min + t * (max - min);
        Self {
            count: lerp(3.0, 40.0, next()).round(),
            growth: lerp(9.0, 14.0, next()).round(),
            no_expand: lerp(0.0, 0.35, next()),
            forward: lerp(0.2, 0.7, next()),
            branch_left: lerp(0.2, 1.0, next()),
            branch_right: lerp(0.2, 1.0, next()),
            // Bias toward the two-way branches; a centre weight only sometimes.
            branch_center: if next() < 0.4 { lerp(0.2, 1.0, next()) } else { 0.0 },
            branch_angle: lerp(0.06, 0.22, next()),
            branch_length: lerp(3.0, 8.0, next()),
            trunk_width: lerp(6.0, 28.0, next()),
            size_variation: lerp(0.0, 0.4, next()),
            hue: lerp(0.0, 360.0, next()),
            hue_spread: lerp(0.0, 40.0, next()),
            brightness: lerp(8.0, 22.0, next()),
            // Leaves on ~half the time, with a matching-ish hue near the trunk.
            leaves: next() < 0.5,
            leaf_size: lerp(3.0, 10.0, next()),
            leaf_density: lerp(0.3, 1.0, next()),
            leaf_hue: lerp(0.0, 360.0, next()),
        }
    }
}

/// The whole scene's forest settings: a list of [`TreeParams`] (one per forest
/// kind), which one the UI is editing, and the single shared wind. The
/// `SliderBinding` routes every tunable except `Wind` to the *selected* forest,
/// so the existing slider panel edits whichever forest is current; `Wind` is
/// global. (One Resource keeps the shared slider machinery — which keys its
/// resync off `is_changed` — working unchanged: selecting a forest mutates this
/// resource and every slider snaps to the new forest's values.)
#[derive(Resource, Clone)]
pub struct ForestSettings {
    pub forests: Vec<TreeParams>,
    pub selected: usize,
    pub wind: f32, // shared by all forests: one breeze through the whole scene
}

impl Default for ForestSettings {
    fn default() -> Self {
        Self {
            forests: vec![TreeParams::default()],
            selected: 0,
            wind: 0.0,
        }
    }
}

impl ForestSettings {
    /// The forest the sliders currently edit (selection is always kept in range
    /// by the add/remove systems, but clamp defensively).
    pub fn current(&self) -> &TreeParams {
        &self.forests[self.selected.min(self.forests.len() - 1)]
    }

    pub fn current_mut(&mut self) -> &mut TreeParams {
        let i = self.selected.min(self.forests.len() - 1);
        &mut self.forests[i]
    }

    /// Add a new forest with [randomised](TreeParams::random) params and select
    /// it. `seed` keeps the choice deterministic for tests.
    pub fn add_forest(&mut self, seed: u64) {
        self.forests.push(TreeParams::random(seed));
        self.selected = self.forests.len() - 1;
    }

    /// Remove the selected forest, keeping at least one (the selector hides the
    /// remove button at one forest, but guard anyway). Selection clamps down.
    pub fn remove_selected(&mut self) {
        if self.forests.len() <= 1 {
            return;
        }
        let i = self.selected.min(self.forests.len() - 1);
        self.forests.remove(i);
        self.selected = self.selected.min(self.forests.len() - 1);
    }

    /// Step the selection by `delta` (the ◀ / ▶ buttons), wrapping around.
    pub fn select_step(&mut self, delta: i32) {
        let n = self.forests.len() as i32;
        self.selected = (self.selected as i32 + delta).rem_euclid(n) as usize;
    }
}

/// The forest's slider-bound tunables (the Leaves checkbox is a separate
/// widget). UI widgets carry this component to bind to a field of
/// [`ForestSettings`].
#[derive(Component, Clone, Copy, PartialEq, Eq, Debug)]
pub enum ForestParam {
    Count,
    Growth,
    NoExpand,
    Forward,
    BranchLeft,
    BranchRight,
    BranchCenter,
    BranchAngle,
    BranchLength,
    TrunkWidth,
    SizeVariation,
    Hue,
    HueSpread,
    Brightness,
    Wind,
    LeafSize,
    LeafDensity,
    LeafHue,
}

impl ForestParam {
    /// Structure + shape + colour sliders, always visible — the original's
    /// `tunableSpecs` order, with the port's Size variation / Wind folded in
    /// near their kin.
    pub const ALWAYS: [Self; 15] = [
        Self::Count,
        Self::Growth,
        Self::NoExpand,
        Self::Forward,
        Self::BranchLeft,
        Self::BranchRight,
        Self::BranchCenter,
        Self::BranchAngle,
        Self::BranchLength,
        Self::TrunkWidth,
        Self::SizeVariation,
        Self::Hue,
        Self::HueSpread,
        Self::Brightness,
        Self::Wind,
    ];
    /// Leaf sliders — shown only while Leaves is on (the `visibleIf` pattern).
    pub const LEAF: [Self; 3] = [Self::LeafSize, Self::LeafDensity, Self::LeafHue];

    /// Hover help shown by the shared tooltip layer (the original explained
    /// nothing).
    pub fn tip(self) -> &'static str {
        match self {
            Self::Count => "How many trees grow, spread across the ground.",
            Self::Growth => {
                "L-system iterations. Each level roughly doubles the string, so \
                 high values grow dense, detailed trees (and cost more to regrow)."
            }
            Self::NoExpand => "Chance a node rests an iteration - gnarlier, less regular trees.",
            Self::Forward => "Chance a branch segment lengthens (F -> FF). Higher = longer limbs.",
            Self::BranchLeft => "Relative weight of the left-leaning two-way branch.",
            Self::BranchRight => "Relative weight of the right-leaning two-way branch.",
            Self::BranchCenter => "Relative weight of the symmetric three-way branch.",
            Self::BranchAngle => "Angle between child branches (x pi). 0 = a straight broom.",
            Self::BranchLength => "Length of each branch segment, in pixels.",
            Self::TrunkWidth => "Trunk width in pixels; branches taper to 0.8x of it per level.",
            Self::SizeVariation => {
                "Random size spread between trees, for depth. 0 = every tree the \
                 same size (the original)."
            }
            Self::Hue => "Trunk hue in degrees (0 = red, the original default).",
            Self::HueSpread => "How far the hue rotates per branch level.",
            Self::Brightness => "Trunk lightness in percent; branches brighten 1.2x per level.",
            Self::Wind => {
                "Sway strength. 0 = a static forest (the original); higher = a \
                 stronger breeze through the canopy."
            }
            Self::LeafSize => "Leaf radius in pixels.",
            Self::LeafDensity => "Fraction of twig tips that sprout a leaf.",
            Self::LeafHue => "Leaf hue in degrees (110 = green).",
        }
    }
}

impl SliderBinding for ForestParam {
    type Settings = ForestSettings;

    fn label(self) -> &'static str {
        match self {
            Self::Count => "Trees",
            Self::Growth => "Growth",
            Self::NoExpand => "No-expand",
            Self::Forward => "Forward",
            Self::BranchLeft => "Branch left",
            Self::BranchRight => "Branch right",
            Self::BranchCenter => "Branch center",
            Self::BranchAngle => "Branch angle",
            Self::BranchLength => "Branch length",
            Self::TrunkWidth => "Trunk width",
            Self::SizeVariation => "Size variation",
            Self::Hue => "Base hue",
            Self::HueSpread => "Hue spread",
            Self::Brightness => "Brightness",
            Self::Wind => "Wind",
            Self::LeafSize => "Leaf size",
            Self::LeafDensity => "Leaf density",
            Self::LeafHue => "Leaf hue",
        }
    }

    /// The original's spec ranges, with port additions and per-line notes
    /// wherever a cap is the port's rather than the original's. `Count`'s max
    /// is the certified perf cap (see `sim::tests` / ARCHITECTURE.md); a CLI
    /// perf run sets it raw, bypassing the clamp.
    fn range(self) -> (f32, f32) {
        match self {
            // Certified at the compound worst case (see sim::MAX_SEGMENTS and
            // ARCHITECTURE.md): at the 700 max, dense probs + leaves + wind hold
            // ~110 fps (M4 Pro, headless) — and the worst case plateaus ~100 fps
            // at any higher count, because the segment budget bounds total
            // geometry and the canopy saturates the screen. Normal use is
            // 1000+ fps. The original's 30 was arbitrary.
            Self::Count => (1.0, 700.0),
            // The original's 4..15. The per-tree segment budget (sim) clamps
            // the actual work, so 15 can't blow up even with heavy probs.
            Self::Growth => (4.0, 15.0),
            Self::NoExpand => (0.0, 0.9),
            Self::Forward => (0.0, 1.0),
            Self::BranchLeft => (0.0, 1.0),
            Self::BranchRight => (0.0, 1.0),
            Self::BranchCenter => (0.0, 1.0),
            Self::BranchAngle => (0.0, 0.60),
            Self::BranchLength => (0.5, 12.0),
            Self::TrunkWidth => (1.0, 60.0),
            // Port addition: 0 = uniform; 0.8 spreads scales over [0.2, 1.8].
            Self::SizeVariation => (0.0, 0.8),
            // The full hue wheel (applied modulo, so wider just repeats).
            Self::Hue => (0.0, 360.0),
            Self::HueSpread => (0.0, 120.0),
            Self::Brightness => (1.0, 60.0),
            // Port addition: a unitless sway strength.
            Self::Wind => (0.0, 1.0),
            Self::LeafSize => (1.0, 20.0),
            Self::LeafDensity => (0.0, 1.0),
            Self::LeafHue => (0.0, 360.0),
        }
    }

    fn get(self, s: &ForestSettings) -> f32 {
        // Wind is the one shared (global) tunable; everything else reads from
        // the forest currently being edited.
        if self == Self::Wind {
            return s.wind;
        }
        let p = s.current();
        match self {
            Self::Count => p.count,
            Self::Growth => p.growth,
            Self::NoExpand => p.no_expand,
            Self::Forward => p.forward,
            Self::BranchLeft => p.branch_left,
            Self::BranchRight => p.branch_right,
            Self::BranchCenter => p.branch_center,
            Self::BranchAngle => p.branch_angle,
            Self::BranchLength => p.branch_length,
            Self::TrunkWidth => p.trunk_width,
            Self::SizeVariation => p.size_variation,
            Self::Hue => p.hue,
            Self::HueSpread => p.hue_spread,
            Self::Brightness => p.brightness,
            Self::LeafSize => p.leaf_size,
            Self::LeafDensity => p.leaf_density,
            Self::LeafHue => p.leaf_hue,
            Self::Wind => unreachable!("handled above"),
        }
    }

    fn set(self, s: &mut ForestSettings, value: f32) {
        let (min, max) = self.range();
        let value = value.clamp(min, max);
        if self == Self::Wind {
            s.wind = value;
            return;
        }
        let p = s.current_mut();
        match self {
            Self::Count => p.count = value,
            Self::Growth => p.growth = value,
            Self::NoExpand => p.no_expand = value,
            Self::Forward => p.forward = value,
            Self::BranchLeft => p.branch_left = value,
            Self::BranchRight => p.branch_right = value,
            Self::BranchCenter => p.branch_center = value,
            Self::BranchAngle => p.branch_angle = value,
            Self::BranchLength => p.branch_length = value,
            Self::TrunkWidth => p.trunk_width = value,
            Self::SizeVariation => p.size_variation = value,
            Self::Hue => p.hue = value,
            Self::HueSpread => p.hue_spread = value,
            Self::Brightness => p.brightness = value,
            Self::LeafSize => p.leaf_size = value,
            Self::LeafDensity => p.leaf_density = value,
            Self::LeafHue => p.leaf_hue = value,
            Self::Wind => unreachable!("handled above"),
        }
    }

    /// The original's format strings (`%d` / `%.2f` / `%.1f`).
    fn format(self, value: f32) -> String {
        match self {
            Self::Count
            | Self::Growth
            | Self::TrunkWidth
            | Self::Hue
            | Self::HueSpread
            | Self::Brightness
            | Self::LeafHue => format!("{}", value.round() as i32),
            Self::BranchLength | Self::LeafSize => format!("{value:.1}"),
            _ => format!("{value:.2}"),
        }
    }

    /// `Count` is log-scaled (every doubling gets equal track room — the
    /// flock/fish/flow convention for counts that span hundreds); the rest are
    /// linear.
    fn t(self, s: &ForestSettings) -> f32 {
        let (min, max) = self.range();
        let value = self.get(s).clamp(min, max);
        match self {
            Self::Count => (value / min).ln() / (max / min).ln(),
            _ => (value - min) / (max - min),
        }
    }

    fn value_from_t(self, t: f32) -> f32 {
        let (min, max) = self.range();
        let t = t.clamp(0.0, 1.0);
        match self {
            Self::Count => min * (max / min).powf(t),
            _ => min + t * (max - min),
        }
    }
}

pub fn plugin(app: &mut App) {
    let mut settings = ForestSettings::default();
    // Perf-harness overrides (`cargo run --release -- <count> forest
    // [growth=N] [wind] [leaves] [dense] ...`). Guarded on the `forest` flag so
    // other experiments' counts don't leak in here (the fish/flow pattern). They
    // configure the single default forest (forests[0]); `wind` is the shared,
    // scene-global setting. Count is set RAW, not through `ForestParam::set` —
    // the slider range must not clamp a perf run.
    let flag = |name: &str| std::env::args().skip(1).any(|arg| arg == name);
    if flag("forest") {
        let p = &mut settings.forests[0];
        if let Some(count) = std::env::args().nth(1).and_then(|arg| arg.parse().ok()) {
            p.count = count;
        }
        if flag("leaves") {
            p.leaves = true;
            p.leaf_density = 1.0;
        }
        // The experiment's structural worst case: every node branches three
        // ways and extends, nothing rests, and segments are tiny (the most,
        // shortest, most-overlapping segments per pixel — the overdraw wall).
        if flag("dense") {
            p.growth = 15.0;
            p.no_expand = 0.0;
            p.forward = 1.0;
            p.branch_left = 1.0;
            p.branch_right = 1.0;
            p.branch_center = 1.0;
            p.branch_length = 1.0;
            p.branch_angle = 0.25;
        }
        // Probe overrides (`key=value`), so a stress grid needs no rebuild
        // per point.
        let kv = |name: &str| {
            std::env::args().skip(1).find_map(|arg| {
                arg.strip_prefix(name)
                    .and_then(|rest| rest.strip_prefix('='))
                    .and_then(|value| value.parse::<f32>().ok())
            })
        };
        if let Some(v) = kv("growth") {
            p.growth = v;
        }
        if let Some(v) = kv("length") {
            p.branch_length = v;
        }
        if let Some(v) = kv("angle") {
            p.branch_angle = v;
        }
        if let Some(v) = kv("forward") {
            p.forward = v;
        }
        if let Some(v) = kv("leafdensity") {
            p.leaf_density = v;
        }
        // Wind is shared (scene-global), so it lives on the container.
        if flag("wind") {
            settings.wind = 0.5;
        }
        if let Some(v) = kv("wind") {
            settings.wind = v;
        }
        // `forests=N` pre-populates the scene with N forests of different random
        // trees (forest[0] keeps the flags above), for the multi-forest perf
        // harness and headless visual checks where there's no UI to press [+].
        if let Some(n) = kv("forests") {
            let n = (n.round() as usize).clamp(1, 12);
            let mut i = 1u64;
            while settings.forests.len() < n {
                settings.add_forest(i.wrapping_mul(0x9E37_79B9_7F4A_7C15));
                i += 1;
            }
            settings.selected = 0;
        }
    }
    app.insert_resource(settings);
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `t` and `value_from_t` must be inverses across every param's range,
    /// or sliders would jump on grab.
    #[test]
    fn slider_mappings_roundtrip() {
        let mut settings = ForestSettings::default();
        let all = ForestParam::ALWAYS.iter().chain(ForestParam::LEAF.iter());
        for &param in all {
            for i in 0..=20 {
                let t = i as f32 / 20.0;
                let value = param.value_from_t(t);
                param.set(&mut settings, value);
                let back = param.t(&settings);
                assert!(
                    (back - t).abs() < 1e-4,
                    "{param:?}: t {t} -> value {value} -> t {back}"
                );
            }
        }
    }

    /// The log-scaled tree count keeps the low end usable: mid-track must be a
    /// modest count, not half the maximum.
    #[test]
    fn count_log_scale_keeps_low_end_usable() {
        let mid = ForestParam::Count.value_from_t(0.5);
        assert!(
            (10.0..=40.0).contains(&mid),
            "mid-track count = {mid}, low end unusable"
        );
        assert!((ForestParam::Count.value_from_t(0.0) - 1.0).abs() < 1e-3);
    }
}
