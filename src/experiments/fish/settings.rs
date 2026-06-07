//! Live-tunable fish settings, matching `tunables` / `tunableSpecs` in
//! `minigames/fish.lua` — plus the fish count, which the original kept in
//! a separate "school" minigame and we fold in as one more tunable (the
//! count and speed *defaults* deviate: 5 fish at 320 instead of the
//! original's lone fish at 200). Speed,
//! the wave knobs, and the count apply live; start size and growth apply
//! from the next restart (growth changes the running fish's future
//! growth, not its current size — like the original).

use bevy::prelude::*;

use crate::ui::SliderBinding;

#[derive(Resource, Clone)]
pub struct FishSettings {
    pub count: f32,       // how many fish; > 1 swims as a school of boids
    pub speed: f32,       // how fast the fish chases the cursor
    pub start_scale: f32, // size the fish starts (and resets) at
    pub growth_rate: f32, // how much the fish grows per food eaten (0 = none)
    pub separation: f32,  // school: avoid crowding neighbours (separate_k)
    pub alignment: f32,   // school: match neighbours' heading (align_k)
    pub cohesion: f32,    // school: steer toward the local centre (coherence_k)
    pub wave: bool,       // swim in a sine path instead of straight to the cursor
    pub wave_freq: f32,   // wiggles per second
    pub wave_amp: f32,    // lateral wiggle size (px)
    // The pond (water.rs) — render-only, no original to be faithful to.
    // Top-down framing: caustics live on the pond bed, ripples and
    // sparkle on the surface, bubbles rise toward the camera.
    pub water: bool,           // the whole water layer (bed + surface + bubbles)
    pub caustics: f32,         // bed light-web intensity (0 = plain bed)
    pub sparkle: f32,          // ambient surface sun-sparkle intensity
    pub ripples: bool,         // fish wakes / food plops disturb the surface
    pub ripple_strength: f32,  // how hard the fish churn the water
    pub bubbles: bool,         // ambient bubbles rising to the surface
    pub bubble_count: f32,     // how many bubbles cycle at once
}

impl Default for FishSettings {
    fn default() -> Self {
        Self {
            // Open with a small school already in motion, a touch quicker
            // than the original's lone fish at 200 — a deliberate default
            // change (the parity tests pin speed explicitly).
            count: 5.0,
            speed: 320.0,
            start_scale: 0.05,
            growth_rate: 0.01,
            // The school minigame's steering-weight defaults (lib/school.lua).
            separation: 2.8,
            alignment: 1.0,
            cohesion: 1.0,
            wave: true,
            wave_freq: 4.5,
            wave_amp: 15.0,
            water: true,
            caustics: 0.55,
            sparkle: 0.35,
            ripples: true,
            ripple_strength: 0.5,
            bubbles: true,
            bubble_count: 128.0,
        }
    }
}

/// The fish's slider-bound tunables (the wave checkbox is a separate
/// widget). UI widgets carry this component to bind themselves to a field
/// of [`FishSettings`].
#[derive(Component, Clone, Copy, PartialEq, Eq, Debug)]
pub enum FishParam {
    Count,
    Speed,
    StartScale,
    GrowthRate,
    Separation,
    Alignment,
    Cohesion,
    WaveFreq,
    WaveAmp,
    Caustics,
    Sparkle,
    RippleStrength,
    BubbleCount,
}

impl FishParam {
    /// Sliders always visible in the popup. Count first — the order the
    /// original school minigame lists its specs.
    pub const MAIN: [Self; 4] = [
        Self::Count,
        Self::Speed,
        Self::StartScale,
        Self::GrowthRate,
    ];
    /// The school's steering weights (the original school minigame's
    /// separation/alignment/cohesion specs) — only shown while more than
    /// one fish swims, since they do nothing to a lone fish.
    pub const SCHOOL: [Self; 3] = [Self::Separation, Self::Alignment, Self::Cohesion];
    /// Sliders only shown while the wiggle checkbox is on — the original's
    /// `visibleIf = 'wave'`.
    pub const WAVE: [Self; 2] = [Self::WaveFreq, Self::WaveAmp];
    /// The water layer's always-relevant dials, shown while Water is on.
    pub const WATER: [Self; 2] = [Self::Caustics, Self::Sparkle];
    /// Shown while Water AND its sub-checkbox are on.
    pub const RIPPLE: [Self; 1] = [Self::RippleStrength];
    pub const BUBBLE: [Self; 1] = [Self::BubbleCount];
}

impl SliderBinding for FishParam {
    type Settings = FishSettings;

    fn label(self) -> &'static str {
        match self {
            Self::Count => "Fish",
            Self::Speed => "Speed",
            Self::StartScale => "Start size",
            Self::GrowthRate => "Growth rate",
            Self::Separation => "Separation",
            Self::Alignment => "Alignment",
            Self::Cohesion => "Cohesion",
            Self::WaveFreq => "Wave freq",
            Self::WaveAmp => "Wave amp",
            Self::Caustics => "Caustics",
            Self::Sparkle => "Sparkle",
            Self::RippleStrength => "Ripple strength",
            Self::BubbleCount => "Bubble count",
        }
    }

    fn range(self) -> (f32, f32) {
        match self {
            // Max = 2x the certified school size (4096 ≥ ~100 fps on the
            // M4 Pro), the same convention as the flock's count slider.
            Self::Count => (1.0, 8192.0),
            Self::Speed => (50.0, 900.0),
            Self::StartScale => (0.05, 0.5),
            Self::GrowthRate => (0.0, 0.1),
            // The school minigame's spec ranges.
            Self::Separation => (0.0, 8.0),
            Self::Alignment | Self::Cohesion => (0.0, 6.0),
            Self::WaveFreq => (1.0, 20.0),
            Self::WaveAmp => (5.0, 120.0),
            // Pure shader dials: intensity multipliers, free at any value.
            Self::Caustics | Self::Sparkle => (0.0, 1.0),
            // Scales the CPU splat amplitude; the wave grid stays the
            // same size, so cost is flat across the range.
            Self::RippleStrength => (0.0, 3.0),
            // Each bubble is one vertex-pull quad — 512 of them is ~3k
            // vertices, nothing next to a single fish's outline.
            Self::BubbleCount => (1.0, 512.0),
        }
    }

    fn get(self, s: &FishSettings) -> f32 {
        match self {
            Self::Count => s.count,
            Self::Speed => s.speed,
            Self::StartScale => s.start_scale,
            Self::GrowthRate => s.growth_rate,
            Self::Separation => s.separation,
            Self::Alignment => s.alignment,
            Self::Cohesion => s.cohesion,
            Self::WaveFreq => s.wave_freq,
            Self::WaveAmp => s.wave_amp,
            Self::Caustics => s.caustics,
            Self::Sparkle => s.sparkle,
            Self::RippleStrength => s.ripple_strength,
            Self::BubbleCount => s.bubble_count,
        }
    }

    fn set(self, s: &mut FishSettings, value: f32) {
        let (min, max) = self.range();
        let value = value.clamp(min, max);
        match self {
            Self::Count => s.count = value,
            Self::Speed => s.speed = value,
            Self::StartScale => s.start_scale = value,
            Self::GrowthRate => s.growth_rate = value,
            Self::Separation => s.separation = value,
            Self::Alignment => s.alignment = value,
            Self::Cohesion => s.cohesion = value,
            Self::WaveFreq => s.wave_freq = value,
            Self::WaveAmp => s.wave_amp = value,
            Self::Caustics => s.caustics = value,
            Self::Sparkle => s.sparkle = value,
            Self::RippleStrength => s.ripple_strength = value,
            Self::BubbleCount => s.bubble_count = value,
        }
    }

    /// The original's format strings: `%d`, `%.2f`, `%.3f`, `%.1f`, `%d`
    /// (and the school's `%.2f` steering weights; the water dials follow
    /// the same conventions).
    fn format(self, value: f32) -> String {
        match self {
            Self::Count | Self::Speed | Self::WaveAmp | Self::BubbleCount => {
                format!("{}", value.round() as i32)
            }
            Self::StartScale
            | Self::Separation
            | Self::Alignment
            | Self::Cohesion
            | Self::Caustics
            | Self::Sparkle
            | Self::RippleStrength => {
                format!("{value:.2}")
            }
            Self::GrowthRate => format!("{value:.3}"),
            Self::WaveFreq => format!("{value:.1}"),
        }
    }

    /// Count uses a log scale — every doubling gets equal track room, so
    /// the 1-fish default and small schools stay draggable under a max in
    /// the thousands. The rest are linear (the trait default).
    fn t(self, s: &FishSettings) -> f32 {
        let (min, max) = self.range();
        let value = self.get(s).clamp(min, max);
        match self {
            Self::Count => (value / min).ln() / (max / min).ln(),
            _ => (value - min) / (max - min),
        }
    }

    /// Inverse of [`Self::t`].
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
    let mut settings = FishSettings::default();
    // Perf-harness count from the CLI (`cargo run --release -- <count>
    // fish ...`). Set raw, NOT through `FishParam::set` — the slider
    // range must not clamp a perf run's count. Guarded on the `fish` flag
    // so a flock perf run's count doesn't leak in here. (On wasm, args
    // are empty and this never fires.)
    if std::env::args().skip(1).any(|arg| arg == "fish")
        && let Some(count) = std::env::args().nth(1).and_then(|arg| arg.parse().ok())
    {
        settings.count = count;
    }
    // Perf A/B switch: `nowater` measures the fish exactly as before the
    // water layer existed (the water defaults on).
    if std::env::args().skip(1).any(|arg| arg == "nowater") {
        settings.water = false;
    }
    // Water probe overrides (the flow CLI's `key=value` convention):
    // `ripple=3 caustics=0.2 sparkle=1` — tuning/perf probes only.
    for arg in std::env::args().skip(1) {
        if let Some((key, value)) = arg.split_once('=')
            && let Ok(value) = value.parse::<f32>()
        {
            match key {
                "ripple" => settings.ripple_strength = value,
                "caustics" => settings.caustics = value,
                "sparkle" => settings.sparkle = value,
                "bubbles" => settings.bubble_count = value,
                _ => {}
            }
        }
    }
    app.insert_resource(settings);
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `t` and `value_from_t` must be inverses across every param's range,
    /// or sliders would jump on grab. (Derived for the fish ranges — the
    /// flock has its own version of this test.)
    #[test]
    fn slider_mappings_roundtrip() {
        let params = [
            FishParam::Count,
            FishParam::Speed,
            FishParam::StartScale,
            FishParam::GrowthRate,
            FishParam::Separation,
            FishParam::Alignment,
            FishParam::Cohesion,
            FishParam::WaveFreq,
            FishParam::WaveAmp,
            FishParam::Caustics,
            FishParam::Sparkle,
            FishParam::RippleStrength,
            FishParam::BubbleCount,
        ];
        let mut settings = FishSettings::default();
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

    /// The log-scaled count keeps the small-school end usable: half the
    /// track must still be a modest school, not half the maximum.
    #[test]
    fn count_log_scale_keeps_low_end_usable() {
        let mid = FishParam::Count.value_from_t(0.5);
        assert!(
            (8.0..=128.0).contains(&mid),
            "mid-track count = {mid}, low end unusable"
        );
        assert!((FishParam::Count.value_from_t(0.0) - 1.0).abs() < 1e-6);
    }
}
