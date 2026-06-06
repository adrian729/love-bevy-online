//! The flock experiment — a Bevy/Rust port of the LÖVE "flock" boids
//! experiment from `love-online`, pushed to its limits. Same simulation
//! rules and tunables; see ARCHITECTURE.md for the design.

pub mod gpu_sim;
pub mod render;
pub mod settings;
pub mod sim;
pub mod ui;

use bevy::prelude::*;

/// Everything the flock experiment registers. Whether the GPU compute sim
/// (default) or the CPU reference sim drives it is read from the CLI
/// internally (the `cpu` / `nosim` perf flags).
pub struct FlockPlugin {
    /// `false` = the original 12-vertex geometry renderer (the `geo` flag).
    pub quads: bool,
    /// Headless perf runs have no window: skip the experiment's UI.
    pub headless: bool,
}

impl Plugin for FlockPlugin {
    fn build(&self, app: &mut App) {
        let cpu = sim::cpu_sim_selected();
        app.insert_resource(if cpu {
            render::SimMode::Cpu
        } else {
            render::SimMode::Gpu
        })
        .add_plugins((
            settings::plugin,
            sim::plugin,
            render::FlockRenderPlugin { quads: self.quads },
        ));
        if !cpu {
            app.add_plugins(gpu_sim::FlockGpuSimPlugin);
        }
        if !self.headless {
            app.add_plugins(ui::plugin);
        }
    }
}
