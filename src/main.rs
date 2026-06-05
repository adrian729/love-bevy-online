//! Flock — a Bevy/Rust port of the LÖVE "flock" boids experiment from
//! `love-online`. Same simulation rules and tunables, idiomatic Bevy ECS.

// Bevy queries routinely trip this lint; Bevy itself allows it.
#![allow(clippy::type_complexity)]

use bevy::diagnostic::{DiagnosticsStore, FrameTimeDiagnosticsPlugin};
use bevy::prelude::*;

mod boids;
mod settings;
mod ui;

/// Top-level game state, mirroring the original's `playing` / `options`.
#[derive(States, Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum AppState {
    #[default]
    Playing,
    Options,
}

fn main() {
    let mut app = App::new();
    app.add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "Flock — Reynolds boids (Bevy)".into(),
                resolution: (1280, 800).into(),
                resize_constraints: WindowResizeConstraints {
                    min_width: 640.0,
                    min_height: 480.0,
                    ..default()
                },
                ..default()
            }),
            ..default()
        }))
        // Background tint from the original: rgb(0.1, 0.1, 0.12).
        .insert_resource(ClearColor(Color::srgb(0.10, 0.10, 0.12)))
        .init_state::<AppState>()
        .add_plugins((settings::plugin, boids::plugin, ui::plugin));
    // Perf-test mode (`boids <count>`): print fps to stdout once a second.
    // Uses println! rather than Bevy's LogDiagnosticsPlugin — the latter logs
    // through the `log` facade, which Cargo.toml statically caps at WARN in
    // release builds (`release_max_level_warn`).
    if std::env::args().nth(1).is_some() {
        app.add_systems(Update, print_fps);
    }
    app.run();
}

fn print_fps(diagnostics: Res<DiagnosticsStore>, time: Res<Time>, mut elapsed: Local<f32>) {
    *elapsed += time.delta_secs();
    if *elapsed < 1.0 {
        return;
    }
    *elapsed = 0.0;
    if let Some(fps) = diagnostics
        .get(&FrameTimeDiagnosticsPlugin::FPS)
        .and_then(|fps| fps.smoothed())
    {
        println!("fps: {fps:.1}");
    }
}
