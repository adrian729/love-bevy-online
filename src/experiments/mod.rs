//! The experiment registry — the original's `minigames` list in `main.lua`.
//!
//! Each experiment is a set of Bevy plugins added once at startup (plugins
//! can't be loaded at runtime). All experiments stay registered side by
//! side; [`CurrentExperiment`] (set by the menu, or the perf CLI) decides
//! which one's systems actually run, via [`experiment_active`]. The menu
//! builds one button per entry here and picks a random backdrop per visit —
//! the original's `menuBgPool`.

pub mod fish;
pub mod flock;
pub mod flow;
pub mod lizard;

use bevy::prelude::*;

/// Identifies one experiment in [`EXPERIMENTS`].
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ExperimentId {
    Flock,
    Fish,
    Flow,
    Lizard,
}

/// A registry entry: what the menu shows.
pub struct Experiment {
    pub id: ExperimentId,
    pub title: &'static str,
}

pub const EXPERIMENTS: &[Experiment] = &[
    Experiment {
        id: ExperimentId::Flock,
        title: "Flock",
    },
    Experiment {
        id: ExperimentId::Fish,
        title: "Fish",
    },
    Experiment {
        id: ExperimentId::Flow,
        title: "Flow Field",
    },
    Experiment {
        id: ExperimentId::Lizard,
        title: "Lizard",
    },
];

/// Which experiment owns the screen — the original's `current` (and, on the
/// menu, `menuBg`). Inactive experiments keep their plugins registered but
/// their systems gated off.
#[derive(Resource, Clone, Copy, PartialEq, Eq, Debug)]
pub struct CurrentExperiment(pub ExperimentId);

impl Default for CurrentExperiment {
    fn default() -> Self {
        Self(ExperimentId::Flock)
    }
}

/// Run condition: this experiment is the current one.
pub fn experiment_active(id: ExperimentId) -> impl FnMut(Res<CurrentExperiment>) -> bool {
    move |current: Res<CurrentExperiment>| current.0 == id
}
