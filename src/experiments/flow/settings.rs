//! Live-tunable flow-field settings, matching `tunables` / `tunableSpecs`
//! in `minigames/flow.lua` — plus `evolve`, this port's addition (the field
//! drifts through a third noise dimension over time; 0 = static, the
//! original's behavior). Everything applies live; rebuild-affecting values
//! re-generate the field as they change (throttled while dragging, like the
//! original's `REBUILD_THROTTLE`).

use bevy::prelude::*;

use super::sim::SEED_PERIOD;
use crate::ui::{CyclerBinding, SliderBinding};

/// How the field is visualised — the original's `mode` options tunable.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum FlowMode {
    /// Lines traced through the field ('illustrate' in the original).
    Streamlines,
    /// One direction arrow per grid cell ('line').
    Arrows,
    /// A per-cell colour fill keyed to the field's direction ('gradient') —
    /// smoothed in this port: colours interpolate across cells.
    Gradient,
    /// Animated particles riding the field with glowing trails.
    #[default]
    Particles,
}

impl FlowMode {
    pub const ALL: [Self; 4] = [
        Self::Streamlines,
        Self::Arrows,
        Self::Gradient,
        Self::Particles,
    ];
}

/// Angle → colour scheme — the original's `palette` options tunable.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum FlowPalette {
    #[default]
    Rainbow,
    Ocean,
    Fire,
    Forest,
    Mono,
}

impl FlowPalette {
    pub const ALL: [Self; 5] = [
        Self::Rainbow,
        Self::Ocean,
        Self::Fire,
        Self::Forest,
        Self::Mono,
    ];
}

#[derive(Resource, Clone)]
pub struct FlowSettings {
    // Field shape.
    pub seed: f32,        // which field; scrubbing pans smoothly through noise space
    pub scale: f32,       // grid cell size in px (smaller = finer & denser)
    pub noise_scale: f32, // noise frequency (larger = more chaotic swirls)
    pub octaves: f32,     // fractal noise layers (1 = plain noise)
    pub roughness: f32,   // finer-octave contribution (persistence)
    pub warp: f32,        // domain-warp strength (0 = none; higher = turbulent)
    pub swirl: f32,       // constant rotation bias in degrees (laminar <-> swirly)
    pub evolve: f32,      // field drift speed over time (port addition; 0 = static)
    // Visualisation.
    pub mode: FlowMode,
    pub palette: FlowPalette,
    pub detail: f32,     // streamline count (Streamlines only)
    pub length: f32,     // max streamline length in steps (Streamlines only)
    pub line_width: f32, // stroke width (Streamlines / Arrows)
    pub opacity: f32,    // stroke alpha (Streamlines / Arrows)
    pub arrowheads: bool, // draw arrowheads (Arrows only)
    pub background: bool, // paint a dimmed gradient behind the lines/particles
    // Particle overlay (animated over the static field).
    pub animate: bool, // particle layer on the non-Particles views
    pub particle_count: f32,
    pub particle_speed: f32, // px/sec
    pub trail_fade: f32,     // per-sample trail fade (higher = shorter trails)
}

impl Default for FlowSettings {
    fn default() -> Self {
        Self {
            seed: 1234.0,
            scale: 20.0,
            noise_scale: 0.02,
            octaves: 1.0,
            roughness: 0.5,
            warp: 0.0,
            swirl: -100.0,
            evolve: 0.0,
            mode: FlowMode::Particles,
            palette: FlowPalette::Rainbow,
            detail: 1200.0,
            length: 50.0,
            line_width: 1.0,
            opacity: 1.0,
            arrowheads: false,
            background: false,
            animate: false,
            particle_count: 2000.0,
            particle_speed: 150.0,
            trail_fade: 0.15,
        }
    }
}

/// The flow's slider-bound tunables (mode/palette are cyclers, the booleans
/// checkboxes). UI widgets carry this component to bind themselves to a
/// field of [`FlowSettings`].
#[derive(Component, Clone, Copy, PartialEq, Eq, Debug)]
pub enum FlowParam {
    Seed,
    Scale,
    NoiseScale,
    Octaves,
    Roughness,
    Warp,
    Swirl,
    Evolve,
    Detail,
    Length,
    LineWidth,
    Opacity,
    ParticleCount,
    ParticleSpeed,
    TrailFade,
}

impl FlowParam {
    /// The field-shape sliders, always visible — the original's spec order.
    pub const FIELD: [Self; 8] = [
        Self::Seed,
        Self::Scale,
        Self::NoiseScale,
        Self::Octaves,
        Self::Roughness,
        Self::Warp,
        Self::Swirl,
        Self::Evolve,
    ];
    /// Streamlines-only sliders (`visibleIf mode == 'illustrate'`).
    pub const STREAM: [Self; 2] = [Self::Detail, Self::Length];
    /// Stroke sliders (`visibleIf mode in {'illustrate', 'line'}`).
    pub const STROKE: [Self; 2] = [Self::LineWidth, Self::Opacity];
    /// Particle sliders (`visibleIf animate or mode == 'particles'`).
    pub const PARTICLE: [Self; 3] = [Self::ParticleCount, Self::ParticleSpeed, Self::TrailFade];
}

impl SliderBinding for FlowParam {
    type Settings = FlowSettings;

    fn label(self) -> &'static str {
        match self {
            Self::Seed => "Seed",
            Self::Scale => "Scale",
            Self::NoiseScale => "Noise",
            Self::Octaves => "Octaves",
            Self::Roughness => "Roughness",
            Self::Warp => "Warp",
            Self::Swirl => "Swirl",
            Self::Evolve => "Evolve",
            Self::Detail => "Detail",
            Self::Length => "Length",
            Self::LineWidth => "Width",
            Self::Opacity => "Opacity",
            Self::ParticleCount => "Particles",
            Self::ParticleSpeed => "Speed",
            Self::TrailFade => "Trail fade",
        }
    }

    /// The original's spec ranges, widened wherever its cap was arbitrary
    /// rather than physical (per-line notes). Every cap that stays has a
    /// real reason.
    fn range(self) -> (f32, f32) {
        match self {
            // Every integer seed below SEED_PERIOD (256,000) is a distinct
            // field; past it they repeat exactly (sim::SEED_PERIOD's
            // derivation). The original's 9999 was arbitrary. Tip: type or
            // paste an exact seed via its value label.
            Self::Seed => (0.0, SEED_PERIOD),
            // Min 4 is a perf cap (a fullscreen grid at 4 is ~100k noise
            // cells per rebuild — the certified worst case); the original's
            // max 40 was arbitrary — coarser is *cheaper*, and huge smooth
            // cells are a look of their own.
            Self::Scale => (4.0, 100.0),
            // Pure look, both ends arbitrary in the original: lower is
            // near-laminar, higher dissolves into per-cell chaos.
            Self::NoiseScale => (0.002, 0.3),
            // Real cap: octave n samples at 2^(n-1)× frequency, and past 6
            // the finest octave is already sub-cell at typical noise
            // scales — deeper costs rebuild time for invisible detail.
            Self::Octaves => (1.0, 6.0),
            // 1.0 = every octave equally loud (the fbm normalizes, so
            // nothing diverges); the original's 0.9 was shy of it.
            Self::Roughness => (0.05, 1.0),
            // Warp strength is a free multiplier (the two extra fbm
            // evaluations run whenever warp > 0, whatever the value);
            // past ~5 the field is noise soup.
            Self::Warp => (0.0, 5.0),
            // Real cap: swirl is a constant rotation, so ±180° already
            // covers the full circle — anything wider repeats.
            Self::Swirl => (-180.0, 180.0),
            // Rate multiplier on the evolve drift; the rebuild cadence is
            // fixed (30 Hz) so higher costs nothing. Past ~3 the field
            // churns faster than it reads.
            Self::Evolve => (0.0, 3.0),
            // Real cap, measured: 2500 retraces + a worst-case field
            // rebuild fit inside one 30 Hz evolve tick (~205 fps at
            // `worst streamlines`); at 5000 they no longer do, so every
            // frame pays the full rebuild and fps cliffs to ~24.
            Self::Detail => (200.0, 2500.0),
            // Longer is free: lines terminate at the window border, which
            // bounds the traced work long before 300 steps (measured: no
            // fps change doubling 150 → 300 at the worst case).
            Self::Length => (5.0, 300.0),
            // Same geometry either way; the original's 6 was arbitrary.
            Self::LineWidth => (0.5, 12.0),
            // Real cap: alpha.
            Self::Opacity => (0.1, 1.0),
            // The original capped at 6000. Max ≈ 2x the certified count
            // (140k ≥ ~120 fps on the M4 Pro with the vertex-pull +
            // fragment-feathered trails; the max itself measures ~55 fps),
            // the same convention as the flock/fish counts.
            Self::ParticleCount => (50.0, 300_000.0),
            // Trails sample at a fixed 60 Hz of sim time, so speed only
            // stretches them; the original's 1000 was arbitrary.
            Self::ParticleSpeed => (10.0, 2000.0),
            // 0.02 already keeps every ring sample visible (the cutoff
            // math saturates — lower changes nothing); 0.6 ≈ 5-point
            // comet stubs, the useful extreme.
            Self::TrailFade => (0.02, 0.6),
        }
    }

    fn get(self, s: &FlowSettings) -> f32 {
        match self {
            Self::Seed => s.seed,
            Self::Scale => s.scale,
            Self::NoiseScale => s.noise_scale,
            Self::Octaves => s.octaves,
            Self::Roughness => s.roughness,
            Self::Warp => s.warp,
            Self::Swirl => s.swirl,
            Self::Evolve => s.evolve,
            Self::Detail => s.detail,
            Self::Length => s.length,
            Self::LineWidth => s.line_width,
            Self::Opacity => s.opacity,
            Self::ParticleCount => s.particle_count,
            Self::ParticleSpeed => s.particle_speed,
            Self::TrailFade => s.trail_fade,
        }
    }

    fn set(self, s: &mut FlowSettings, value: f32) {
        let (min, max) = self.range();
        let value = value.clamp(min, max);
        match self {
            Self::Seed => s.seed = value,
            Self::Scale => s.scale = value,
            Self::NoiseScale => s.noise_scale = value,
            Self::Octaves => s.octaves = value,
            Self::Roughness => s.roughness = value,
            Self::Warp => s.warp = value,
            Self::Swirl => s.swirl = value,
            Self::Evolve => s.evolve = value,
            Self::Detail => s.detail = value,
            Self::Length => s.length = value,
            Self::LineWidth => s.line_width = value,
            Self::Opacity => s.opacity = value,
            Self::ParticleCount => s.particle_count = value,
            Self::ParticleSpeed => s.particle_speed = value,
            Self::TrailFade => s.trail_fade = value,
        }
    }

    /// The original's format strings.
    fn format(self, value: f32) -> String {
        match self {
            Self::Seed
            | Self::Scale
            | Self::Octaves
            | Self::Swirl
            | Self::Detail
            | Self::Length
            | Self::ParticleCount
            | Self::ParticleSpeed => format!("{}", value.round() as i32),
            Self::NoiseScale => format!("{value:.3}"),
            Self::Roughness | Self::Warp | Self::Evolve | Self::Opacity | Self::TrailFade => {
                format!("{value:.2}")
            }
            Self::LineWidth => format!("{value:.1}"),
        }
    }

    /// The particle count uses a log scale — every doubling gets equal
    /// track room, the flock/fish convention for counts in the thousands.
    fn t(self, s: &FlowSettings) -> f32 {
        let (min, max) = self.range();
        let value = self.get(s).clamp(min, max);
        match self {
            Self::ParticleCount => (value / min).ln() / (max / min).ln(),
            _ => (value - min) / (max - min),
        }
    }

    /// Inverse of [`Self::t`].
    fn value_from_t(self, t: f32) -> f32 {
        let (min, max) = self.range();
        let t = t.clamp(0.0, 1.0);
        match self {
            Self::ParticleCount => min * (max / min).powf(t),
            _ => min + t * (max - min),
        }
    }
}

/// The flow's two option cyclers — the original's `type = 'options'` specs.
#[derive(Component, Clone, Copy, PartialEq, Eq, Debug)]
pub enum FlowCycler {
    Mode,
    Palette,
}

impl CyclerBinding for FlowCycler {
    type Settings = FlowSettings;

    fn label(self) -> &'static str {
        match self {
            Self::Mode => "View",
            Self::Palette => "Palette",
        }
    }

    fn count(self) -> usize {
        match self {
            Self::Mode => FlowMode::ALL.len(),
            Self::Palette => FlowPalette::ALL.len(),
        }
    }

    fn get(self, s: &FlowSettings) -> usize {
        match self {
            Self::Mode => FlowMode::ALL.iter().position(|m| *m == s.mode).unwrap_or(0),
            Self::Palette => FlowPalette::ALL
                .iter()
                .position(|p| *p == s.palette)
                .unwrap_or(0),
        }
    }

    fn set(self, s: &mut FlowSettings, index: usize) {
        match self {
            Self::Mode => s.mode = FlowMode::ALL[index % FlowMode::ALL.len()],
            Self::Palette => s.palette = FlowPalette::ALL[index % FlowPalette::ALL.len()],
        }
    }

    /// The original's option labels.
    fn option_label(self, index: usize) -> &'static str {
        match self {
            Self::Mode => match FlowMode::ALL[index % FlowMode::ALL.len()] {
                FlowMode::Streamlines => "Streamlines",
                FlowMode::Arrows => "Arrows",
                FlowMode::Gradient => "Gradient",
                FlowMode::Particles => "Particles",
            },
            Self::Palette => match FlowPalette::ALL[index % FlowPalette::ALL.len()] {
                FlowPalette::Rainbow => "Rainbow",
                FlowPalette::Ocean => "Ocean",
                FlowPalette::Fire => "Fire",
                FlowPalette::Forest => "Forest",
                FlowPalette::Mono => "Mono",
            },
        }
    }
}

pub fn plugin(app: &mut App) {
    let mut settings = FlowSettings::default();
    // Perf-harness overrides from the CLI (`cargo run --release --
    // <count> flow [streamlines|arrows|gradient] [evolve] [worst]`).
    // Guarded on the `flow` flag so other experiments' counts don't leak
    // in here (the fish pattern). Count is set raw, NOT through
    // `FlowParam::set` — the slider range must not clamp a perf run.
    let flag = |name: &str| std::env::args().skip(1).any(|arg| arg == name);
    if flag("flow") {
        if let Some(count) = std::env::args().nth(1).and_then(|arg| arg.parse().ok()) {
            settings.particle_count = count;
        }
        if flag("streamlines") {
            settings.mode = FlowMode::Streamlines;
        } else if flag("arrows") {
            settings.mode = FlowMode::Arrows;
        } else if flag("gradient") {
            settings.mode = FlowMode::Gradient;
        }
        if flag("evolve") {
            settings.evolve = 1.0;
        }
        // Stress probe: the shortest trails (7 visible points vs 20).
        if flag("fade04") {
            settings.trail_fade = 0.4;
        }
        // The experiment's worst case: finest grid, deepest fbm, full warp,
        // evolving every tick — and, with `streamlines`, the longest
        // densest retrace. Kept in sync with the slider maxima (warp and
        // evolve only cost when > 0; their magnitudes are free).
        if flag("worst") {
            settings.scale = 4.0;
            settings.octaves = 6.0;
            settings.warp = 5.0;
            settings.evolve = 1.0;
            settings.detail = FlowParam::Detail.range().1;
            settings.length = FlowParam::Length.range().1;
        }
        // Probe overrides (`detail=N length=N seed=N`), so a stress grid
        // doesn't need a rebuild per point.
        let kv = |name: &str| {
            std::env::args().skip(1).find_map(|arg| {
                arg.strip_prefix(name)
                    .and_then(|rest| rest.strip_prefix('='))
                    .and_then(|value| value.parse::<f32>().ok())
            })
        };
        if let Some(value) = kv("detail") {
            settings.detail = value;
        }
        if let Some(value) = kv("length") {
            settings.length = value;
        }
        if let Some(value) = kv("seed") {
            settings.seed = value;
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
        let params = [
            FlowParam::Seed,
            FlowParam::Scale,
            FlowParam::NoiseScale,
            FlowParam::Octaves,
            FlowParam::Roughness,
            FlowParam::Warp,
            FlowParam::Swirl,
            FlowParam::Evolve,
            FlowParam::Detail,
            FlowParam::Length,
            FlowParam::LineWidth,
            FlowParam::Opacity,
            FlowParam::ParticleCount,
            FlowParam::ParticleSpeed,
            FlowParam::TrailFade,
        ];
        let mut settings = FlowSettings::default();
        for param in params {
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

    /// The log-scaled particle count keeps the low end usable: half the
    /// track must still be a modest count, not half the maximum.
    #[test]
    fn count_log_scale_keeps_low_end_usable() {
        let mid = FlowParam::ParticleCount.value_from_t(0.5);
        assert!(
            (500.0..=4000.0).contains(&mid),
            "mid-track count = {mid}, low end unusable"
        );
        assert!((FlowParam::ParticleCount.value_from_t(0.0) - 50.0).abs() < 1e-3);
    }

    /// Both cyclers step through every option and wrap; get/set agree.
    #[test]
    fn cyclers_roundtrip() {
        let mut settings = FlowSettings::default();
        for cycler in [FlowCycler::Mode, FlowCycler::Palette] {
            for index in 0..cycler.count() {
                cycler.set(&mut settings, index);
                assert_eq!(cycler.get(&settings), index, "{cycler:?} index {index}");
                assert!(!cycler.option_label(index).is_empty());
            }
        }
        assert_eq!(settings.mode, FlowMode::Particles);
        assert_eq!(settings.palette, FlowPalette::Mono);
    }
}
