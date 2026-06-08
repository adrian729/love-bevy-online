//! The forest experiment — a port of the LÖVE prototype's `minigames/tree.lua`
//! and `lib/tree.lua`: a forest of procedural L-system trees grown by randomised
//! string rewriting, baked once and redrawn each frame (the original's
//! render-once-to-canvas). The port keeps the 13 original tunables and the
//! two-signature regrow/repaint scheme, and adds, in the experiment alone:
//! feathered branches (the collection's line art language), a wind sway that
//! animates the canopy entirely in the vertex shader (0 = the original's static
//! forest), optional leaves, per-tree size variation, and shader-side colour so
//! the hue/brightness/wind sliders update live for free.

pub mod render;
pub mod settings;
pub mod sim;
pub mod ui;

use bevy::prelude::*;

/// Everything the forest experiment registers. The forest is static geometry,
/// so there is no per-frame simulation step — just the (throttled) regrow/repaint
/// and one baked, version-gated draw.
pub struct ForestPlugin {
    /// Headless perf runs have no window: skip the experiment's UI.
    pub headless: bool,
}

impl Plugin for ForestPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins((settings::plugin, sim::plugin, render::plugin));
        if !self.headless {
            app.add_plugins(ui::plugin);
        }
    }
}
