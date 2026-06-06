//! Live-tunable fish settings, matching `tunables` / `tunableSpecs` in
//! `minigames/fish.lua`. Speed and the wave knobs apply live; start size
//! and growth apply from the next restart (growth changes the running
//! fish's future growth, not its current size — like the original).

use bevy::prelude::*;

use crate::ui::SliderBinding;

#[derive(Resource, Clone)]
pub struct FishSettings {
    pub speed: f32,       // how fast the fish chases the cursor
    pub start_scale: f32, // size the fish starts (and resets) at
    pub growth_rate: f32, // how much the fish grows per food eaten (0 = none)
    pub wave: bool,       // swim in a sine path instead of straight to the cursor
    pub wave_freq: f32,   // wiggles per second
    pub wave_amp: f32,    // lateral wiggle size (px)
}

impl Default for FishSettings {
    fn default() -> Self {
        Self {
            speed: 200.0,
            start_scale: 0.1,
            growth_rate: 0.01,
            wave: true,
            wave_freq: 4.5,
            wave_amp: 15.0,
        }
    }
}

/// The fish's slider-bound tunables (the wave checkbox is a separate
/// widget). UI widgets carry this component to bind themselves to a field
/// of [`FishSettings`].
#[derive(Component, Clone, Copy, PartialEq, Eq, Debug)]
pub enum FishParam {
    Speed,
    StartScale,
    GrowthRate,
    WaveFreq,
    WaveAmp,
}

impl FishParam {
    /// Sliders always visible in the popup.
    pub const MAIN: [Self; 3] = [Self::Speed, Self::StartScale, Self::GrowthRate];
    /// Sliders only shown while the wiggle checkbox is on — the original's
    /// `visibleIf = 'wave'`.
    pub const WAVE: [Self; 2] = [Self::WaveFreq, Self::WaveAmp];
}

impl SliderBinding for FishParam {
    type Settings = FishSettings;

    fn label(self) -> &'static str {
        match self {
            Self::Speed => "Speed",
            Self::StartScale => "Start size",
            Self::GrowthRate => "Growth rate",
            Self::WaveFreq => "Wave freq",
            Self::WaveAmp => "Wave amp",
        }
    }

    fn range(self) -> (f32, f32) {
        match self {
            Self::Speed => (50.0, 900.0),
            Self::StartScale => (0.05, 0.5),
            Self::GrowthRate => (0.0, 0.1),
            Self::WaveFreq => (1.0, 20.0),
            Self::WaveAmp => (5.0, 120.0),
        }
    }

    fn get(self, s: &FishSettings) -> f32 {
        match self {
            Self::Speed => s.speed,
            Self::StartScale => s.start_scale,
            Self::GrowthRate => s.growth_rate,
            Self::WaveFreq => s.wave_freq,
            Self::WaveAmp => s.wave_amp,
        }
    }

    fn set(self, s: &mut FishSettings, value: f32) {
        let (min, max) = self.range();
        let value = value.clamp(min, max);
        match self {
            Self::Speed => s.speed = value,
            Self::StartScale => s.start_scale = value,
            Self::GrowthRate => s.growth_rate = value,
            Self::WaveFreq => s.wave_freq = value,
            Self::WaveAmp => s.wave_amp = value,
        }
    }

    /// The original's format strings: `%d`, `%.2f`, `%.3f`, `%.1f`, `%d`.
    fn format(self, value: f32) -> String {
        match self {
            Self::Speed | Self::WaveAmp => format!("{}", value.round() as i32),
            Self::StartScale => format!("{value:.2}"),
            Self::GrowthRate => format!("{value:.3}"),
            Self::WaveFreq => format!("{value:.1}"),
        }
    }
}

pub fn plugin(app: &mut App) {
    app.init_resource::<FishSettings>();
}
