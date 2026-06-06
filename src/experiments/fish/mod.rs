//! The fish experiment — a Bevy/Rust port of the LÖVE "fish" minigame from
//! `love-online`: a procedural 12-joint FABRIK fish that chases the cursor,
//! orbits it when it rests, and grows by eating food. Same spine solver,
//! same spline-outlined body, same tunables.

pub mod render;
pub mod settings;
pub mod sim;
pub mod ui;

use bevy::prelude::*;

/// Everything the fish experiment registers. The sim is plain CPU work (a
/// 12-joint chain per fish needs no GPU compute), and the renderer is one
/// dynamic vertex-colored mesh — deliberately not the flock's instanced
/// pipeline, which solves a problem the fish doesn't have.
pub struct FishPlugin {
    /// Headless perf runs have no window: skip the experiment's UI.
    pub headless: bool,
}

impl Plugin for FishPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins((settings::plugin, sim::plugin, render::plugin));
        if !self.headless {
            app.add_plugins(ui::plugin);
        }
    }
}
