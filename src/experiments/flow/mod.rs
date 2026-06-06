//! The flow-field experiment — a Bevy/Rust port of the LÖVE "flow"
//! minigame from `love-online`: a Perlin-noise vector field visualised as
//! streamlines, arrows, a colour gradient, or particles riding the field
//! with glowing trails. Deliberately *better* than the original rather
//! than 1:1 (the user's request): bilinearly smooth sampling, RK2
//! advection, an evolving field, tapered ribbon strokes, and a far higher
//! particle ceiling — the tunables and palettes are the original's.

pub mod render;
pub mod settings;
pub mod sim;
pub mod ui;

use bevy::prelude::*;

/// Everything the flow experiment registers. The sim is CPU work (a noise
/// field plus independent particles parallelize fine on the compute
/// pool), and the renderer is two dynamic vertex-colored layers through
/// one premultiplied-alpha pipeline — not the fish's (alpha-blended, one
/// layer) nor the flock's (instanced) path, which solve different
/// problems.
pub struct FlowPlugin {
    /// Headless perf runs have no window: skip the experiment's UI.
    pub headless: bool,
}

impl Plugin for FlowPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins((settings::plugin, sim::plugin, render::plugin));
        if !self.headless {
            app.add_plugins(ui::plugin);
        }
    }
}
