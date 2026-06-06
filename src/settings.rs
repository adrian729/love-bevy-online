//! Live-tunable simulation settings and the fixed steering constants,
//! matching `lib/flock.lua` + `minigames/flock.lua` from the original.

use bevy::prelude::*;

/// Per-frame steering clamp (at the 60 fps reference, see [`REF_FPS`]).
pub const MAX_FORCE: f32 = 0.3;
/// Radius within which boids actively avoid each other.
pub const SEPARATE_DIST: f32 = 50.0;
/// Radius within which boids align and cohere (also the spatial-hash cell size).
pub const NEIGHBOUR_DIST: f32 = 100.0;
/// The mouse repels boids closer than this and attracts the rest.
pub const MOUSE_NEAR: f32 = 100.0;
pub const MOUSE_ATTRACT_K: f32 = 4.0;
pub const MOUSE_REPEL_K: f32 = -6.0;
/// The LÖVE original integrates per frame with constants tuned around 60 fps.
/// We convert its per-frame units to per-second against this reference so the
/// simulation feels identical at any frame rate.
pub const REF_FPS: f32 = 60.0;

/// The tunables panel state. Mirrors the original's `tunables` table.
#[derive(Resource, Clone)]
pub struct SimSettings {
    pub count: f32,      // number of boids
    pub speed: f32,      // max boid speed, px/s
    pub separation: f32, // weight: avoid crowding neighbours
    pub alignment: f32,  // weight: match neighbours' heading
    pub cohesion: f32,   // weight: steer toward the local centre of mass
}

impl Default for SimSettings {
    fn default() -> Self {
        Self {
            count: 50.0,
            speed: 400.0,
            separation: 1.8,
            alignment: 1.0,
            cohesion: 1.0,
        }
    }
}

/// One tunable parameter. UI widgets carry this component to bind themselves
/// to a field of [`SimSettings`] (the original's `tunableSpecs`).
#[derive(Component, Clone, Copy, PartialEq, Eq, Debug)]
pub enum Param {
    Count,
    Speed,
    Separation,
    Alignment,
    Cohesion,
}

impl Param {
    pub const ALL: [Self; 5] = [
        Self::Count,
        Self::Speed,
        Self::Separation,
        Self::Alignment,
        Self::Cohesion,
    ];

    pub fn label(self) -> &'static str {
        match self {
            Self::Count => "Boids",
            Self::Speed => "Speed",
            Self::Separation => "Separation",
            Self::Alignment => "Alignment",
            Self::Cohesion => "Cohesion",
        }
    }

    pub fn range(self) -> (f32, f32) {
        match self {
            // With the neighbour-sampling cap the sim cost is linear in n;
            // 20k measured at ~120 fps (release, M4 Pro). The LÖVE original
            // capped at 300 for browser performance.
            Self::Count => (10.0, 20_000.0),
            Self::Speed => (50.0, 1500.0),
            Self::Separation => (0.0, 8.0),
            Self::Alignment => (0.0, 6.0),
            Self::Cohesion => (0.0, 6.0),
        }
    }

    pub fn get(self, s: &SimSettings) -> f32 {
        match self {
            Self::Count => s.count,
            Self::Speed => s.speed,
            Self::Separation => s.separation,
            Self::Alignment => s.alignment,
            Self::Cohesion => s.cohesion,
        }
    }

    pub fn set(self, s: &mut SimSettings, value: f32) {
        let (min, max) = self.range();
        let value = value.clamp(min, max);
        match self {
            Self::Count => s.count = value,
            Self::Speed => s.speed = value,
            Self::Separation => s.separation = value,
            Self::Alignment => s.alignment = value,
            Self::Cohesion => s.cohesion = value,
        }
    }

    /// Display format, matching the original's `%d` / `%.2f` specs.
    pub fn format(self, value: f32) -> String {
        match self {
            Self::Count | Self::Speed => format!("{}", value.round() as i32),
            _ => format!("{value:.2}"),
        }
    }

    /// Normalized 0..1 slider position for the current value. Count uses a
    /// log scale so the whole 10..10000 range stays draggable with useful
    /// precision at the low end.
    pub fn t(self, s: &SimSettings) -> f32 {
        let (min, max) = self.range();
        let value = self.get(s).clamp(min, max);
        match self {
            Self::Count => (value / min).ln() / (max / min).ln(),
            _ => (value - min) / (max - min),
        }
    }

    /// Inverse of [`Self::t`]: the value for a 0..1 slider position.
    pub fn value_from_t(self, t: f32) -> f32 {
        let (min, max) = self.range();
        let t = t.clamp(0.0, 1.0);
        match self {
            Self::Count => min * (max / min).powf(t),
            _ => min + t * (max - min),
        }
    }
}

pub fn plugin(app: &mut App) {
    let mut settings = SimSettings::default();
    // Optional initial boid count from the CLI, handy for perf testing:
    // `cargo run --release -- 10000`
    if let Some(count) = std::env::args().nth(1).and_then(|arg| arg.parse().ok()) {
        Param::Count.set(&mut settings, count);
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
        for param in Param::ALL {
            let (min, max) = param.range();
            for i in 0..=20 {
                let t = i as f32 / 20.0;
                let value = param.value_from_t(t);
                assert!((min..=max).contains(&value), "{param:?} out of range");
                let mut settings = SimSettings::default();
                param.set(&mut settings, value);
                assert!(
                    (param.t(&settings) - t).abs() < 1e-3,
                    "{param:?}: t={t} round-tripped to {}",
                    param.t(&settings)
                );
            }
        }
    }

    /// The log scale exists to keep small flocks draggable: half the track
    /// should cover counts up to a few hundred, not up to half the max.
    #[test]
    fn count_log_scale_keeps_low_end_usable() {
        let mid = Param::Count.value_from_t(0.5);
        assert!((100.0..600.0).contains(&mid), "track midpoint = {mid}");
    }
}
