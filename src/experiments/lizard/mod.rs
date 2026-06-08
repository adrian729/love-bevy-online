//! The lizard experiment — the LÖVE prototype's procedural lizard,
//! finished: the fish's 14-joint FABRIK spine technique with the lizard's
//! measurements, four legs that plant in the world and trot in diagonal
//! pairs (2-bone IK elbows, step-when-too-far feet — the part the
//! prototype never got right), and a green flat-color body in the
//! collection's outline art language. Chases the cursor, orbits it at
//! rest, grazes for food while the pointer is away, eats and grows.

pub mod render;
pub mod settings;
pub mod sim;
pub mod ui;

use bevy::prelude::*;

/// Everything the lizard experiment registers. One creature: plain CPU
/// sim, one dynamic vertex-colored draw — the fish's pipeline shape
/// without its school-scale machinery.
pub struct LizardPlugin {
    /// Headless perf runs have no window: skip the experiment's UI.
    pub headless: bool,
}

impl Plugin for LizardPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins((settings::plugin, sim::plugin, render::plugin));
        if !self.headless {
            app.add_plugins(ui::plugin);
        }
    }
}
