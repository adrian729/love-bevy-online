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

#[derive(Resource, Clone)]
pub struct ForestSettings {
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
    // Colour / wind (pure shader uniforms; free live updates).
    pub hue: f32,        // base hue at the trunk (degrees) — original red default
    pub hue_spread: f32, // hue rotation per branch level (degrees)
    pub brightness: f32, // base lightness at the trunk (%); brightens outward
    pub wind: f32,       // sway speed/strength (port addition; 0 = static original)
    // Leaves (port addition; OFF preserves the faithful bare L-system).
    pub leaves: bool,
    pub leaf_size: f32,    // leaf radius (px)
    pub leaf_density: f32, // fraction of twig tips that sprout a leaf (0..1)
    pub leaf_hue: f32,     // leaf hue (degrees)
}

impl Default for ForestSettings {
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
            wind: 0.0,
            leaves: false,
            leaf_size: 6.0,
            leaf_density: 0.6,
            leaf_hue: 110.0,
        }
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
        match self {
            Self::Count => s.count,
            Self::Growth => s.growth,
            Self::NoExpand => s.no_expand,
            Self::Forward => s.forward,
            Self::BranchLeft => s.branch_left,
            Self::BranchRight => s.branch_right,
            Self::BranchCenter => s.branch_center,
            Self::BranchAngle => s.branch_angle,
            Self::BranchLength => s.branch_length,
            Self::TrunkWidth => s.trunk_width,
            Self::SizeVariation => s.size_variation,
            Self::Hue => s.hue,
            Self::HueSpread => s.hue_spread,
            Self::Brightness => s.brightness,
            Self::Wind => s.wind,
            Self::LeafSize => s.leaf_size,
            Self::LeafDensity => s.leaf_density,
            Self::LeafHue => s.leaf_hue,
        }
    }

    fn set(self, s: &mut ForestSettings, value: f32) {
        let (min, max) = self.range();
        let value = value.clamp(min, max);
        match self {
            Self::Count => s.count = value,
            Self::Growth => s.growth = value,
            Self::NoExpand => s.no_expand = value,
            Self::Forward => s.forward = value,
            Self::BranchLeft => s.branch_left = value,
            Self::BranchRight => s.branch_right = value,
            Self::BranchCenter => s.branch_center = value,
            Self::BranchAngle => s.branch_angle = value,
            Self::BranchLength => s.branch_length = value,
            Self::TrunkWidth => s.trunk_width = value,
            Self::SizeVariation => s.size_variation = value,
            Self::Hue => s.hue = value,
            Self::HueSpread => s.hue_spread = value,
            Self::Brightness => s.brightness = value,
            Self::Wind => s.wind = value,
            Self::LeafSize => s.leaf_size = value,
            Self::LeafDensity => s.leaf_density = value,
            Self::LeafHue => s.leaf_hue = value,
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
    // other experiments' counts don't leak in here (the fish/flow pattern).
    // Count is set RAW, not through `ForestParam::set` — the slider range must
    // not clamp a perf run.
    let flag = |name: &str| std::env::args().skip(1).any(|arg| arg == name);
    if flag("forest") {
        if let Some(count) = std::env::args().nth(1).and_then(|arg| arg.parse().ok()) {
            settings.count = count;
        }
        if flag("wind") {
            settings.wind = 0.5;
        }
        if flag("leaves") {
            settings.leaves = true;
            settings.leaf_density = 1.0;
        }
        // The experiment's structural worst case: every node branches three
        // ways and extends, nothing rests, and segments are tiny (the most,
        // shortest, most-overlapping segments per pixel — the overdraw wall).
        if flag("dense") {
            settings.growth = 15.0;
            settings.no_expand = 0.0;
            settings.forward = 1.0;
            settings.branch_left = 1.0;
            settings.branch_right = 1.0;
            settings.branch_center = 1.0;
            settings.branch_length = 1.0;
            settings.branch_angle = 0.25;
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
            settings.growth = v;
        }
        if let Some(v) = kv("length") {
            settings.branch_length = v;
        }
        if let Some(v) = kv("angle") {
            settings.branch_angle = v;
        }
        if let Some(v) = kv("forward") {
            settings.forward = v;
        }
        if let Some(v) = kv("wind") {
            settings.wind = v;
        }
        if let Some(v) = kv("leafdensity") {
            settings.leaf_density = v;
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
