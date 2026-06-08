//! Live-tunable lizard settings. The body shape is fixed (the reference
//! lizard — see `sim::BodyPlan::reference`), so the only knobs are the
//! behavioural ones: how fast it chases the cursor, the size it starts at,
//! and how much it grows per food eaten. Start size and growth keep the LÖVE
//! test-lizard's values (0.5 / 0.002); speed is uncapped, since the gait
//! clock runs on distance travelled. Speed and growth apply while playing;
//! Start size rebuilds the body in place the moment it changes (see
//! reshape_lizard). The Skeleton view swaps the skin for the procedural rig.

use bevy::prelude::*;

use crate::ui::SliderBinding;

#[derive(Resource, Clone)]
pub struct LizardSettings {
    pub start_scale: f32, // size the lizard starts (and resets) at
    pub growth_rate: f32, // how much it grows per food eaten (0 = none)
    pub skeleton: bool,   // draw the rig (lines and circles), not the skin
}

impl Default for LizardSettings {
    fn default() -> Self {
        Self {
            // Starts tiny and slow; chase speed is coupled to size
            // (sim::SPEED_PER_SCALE), so 0.1 → 50 px/s, growing as it eats.
            start_scale: 0.1,
            growth_rate: 0.002,
            skeleton: false,
        }
    }
}

/// The lizard's slider-bound tunables (the skeleton checkbox is a separate
/// widget). Speed isn't here — it's a function of size, not an independent
/// control.
#[derive(Component, Clone, Copy, PartialEq, Eq, Debug)]
pub enum LizardParam {
    StartScale,
    GrowthRate,
}

impl LizardParam {
    /// The behavioural sliders shown in the options popup.
    pub const ALL: [Self; 2] = [Self::StartScale, Self::GrowthRate];
}

impl SliderBinding for LizardParam {
    type Settings = LizardSettings;

    fn label(self) -> &'static str {
        match self {
            Self::StartScale => "Start size",
            Self::GrowthRate => "Growth rate",
        }
    }

    fn range(self) -> (f32, f32) {
        match self {
            Self::StartScale => (0.1, 1.5),
            Self::GrowthRate => (0.0, 0.05),
        }
    }

    fn get(self, s: &LizardSettings) -> f32 {
        match self {
            Self::StartScale => s.start_scale,
            Self::GrowthRate => s.growth_rate,
        }
    }

    fn set(self, s: &mut LizardSettings, value: f32) {
        let (min, max) = self.range();
        let value = value.clamp(min, max);
        match self {
            Self::StartScale => s.start_scale = value,
            Self::GrowthRate => s.growth_rate = value,
        }
    }

    fn format(self, value: f32) -> String {
        match self {
            Self::StartScale => format!("{value:.2}"),
            Self::GrowthRate => format!("{value:.3}"),
        }
    }
}

pub fn plugin(app: &mut App) {
    let mut settings = LizardSettings::default();
    // Probe overrides (the flow/fish CLI `key=value` convention), guarded on
    // the `lizard` flag so other experiments' probe runs don't retune the
    // lizard. The perf `<count>` argument is deliberately ignored: there is
    // always exactly one lizard, and the perf number is the pipeline's
    // single-creature floor.
    if std::env::args().skip(1).any(|arg| arg == "lizard") {
        for arg in std::env::args().skip(1) {
            if arg == "skeleton" {
                settings.skeleton = true;
            }
            if let Some((key, value)) = arg.split_once('=')
                && let Ok(value) = value.parse::<f32>()
            {
                match key {
                    "scale" => settings.start_scale = value,
                    "growth" => settings.growth_rate = value,
                    _ => {}
                }
            }
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
        let mut settings = LizardSettings::default();
        for param in LizardParam::ALL {
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
}
