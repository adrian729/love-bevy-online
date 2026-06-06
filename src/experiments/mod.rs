//! The experiment registry — the original's `minigames` list in `main.lua`.
//!
//! Each experiment is a set of Bevy plugins added once at startup (plugins
//! can't be loaded at runtime), and the menu builds one button per entry
//! here. When a second experiment lands: add a `CurrentExperiment` resource
//! set by the menu, gate each experiment's systems on it (their plugins stay
//! registered side by side), and have the menu pick a random backdrop per
//! visit — the original's `menuBgPool`.

pub mod flock;

/// Identifies one experiment in [`EXPERIMENTS`].
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ExperimentId {
    Flock,
}

/// A registry entry: what the menu shows.
pub struct Experiment {
    pub id: ExperimentId,
    pub title: &'static str,
}

pub const EXPERIMENTS: &[Experiment] = &[Experiment {
    id: ExperimentId::Flock,
    title: "Flock",
}];
