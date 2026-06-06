//! Flock — a Bevy/Rust port of the LÖVE "flock" boids experiment from
//! `love-online`. Same simulation rules and tunables, idiomatic Bevy ECS.

// Bevy queries routinely trip this lint; Bevy itself allows it.
#![allow(clippy::type_complexity)]

use std::time::Duration;

use bevy::app::ScheduleRunnerPlugin;
use bevy::diagnostic::{DiagnosticsStore, FrameTimeDiagnosticsPlugin};
use bevy::prelude::*;
use bevy::window::{ExitCondition, PresentMode};
use bevy::winit::WinitPlugin;

mod boids;
mod gpu_sim;
mod render;
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
    // Perf-test mode (`boids <count> [pin] [headless] [nosim]`): print fps to
    // stdout once a second, with vsync off so readings show real headroom
    // past the display's refresh rate.
    //   pin      — fake mouse attractor at screen centre: the sustained worst
    //              case (the whole flock piled into a ring).
    //   headless — no window at all; the camera renders to an offscreen
    //              texture and the schedule free-runs. Exercises the full
    //              sim + extract + batch + GPU pipeline, immune to display
    //              sleep / occlusion throttling (macOS caps presentation on
    //              occluded or sleeping displays, poisoning fps readings).
    //   nosim    — skip steering, isolating the render/ECS floor.
    //   geo      — render the original 12-vertex boid geometry instead of
    //              the baked-texture quads (the visual reference).
    let perf_mode = std::env::args().nth(1).is_some();
    let flag = |name: &str| std::env::args().skip(2).any(|arg| arg == name);
    let pin = flag("pin");
    let headless = flag("headless");
    let quads = !flag("geo");

    let mut app = App::new();
    if headless {
        app.add_plugins(
            DefaultPlugins
                .set(WindowPlugin {
                    primary_window: None,
                    exit_condition: ExitCondition::DontExit,
                    ..default()
                })
                .disable::<WinitPlugin>(),
        )
        .add_plugins(ScheduleRunnerPlugin::run_loop(Duration::ZERO));
    } else {
        app.add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "Flock — Reynolds boids (Bevy)".into(),
                resolution: (1280, 800).into(),
                resize_constraints: WindowResizeConstraints {
                    min_width: 640.0,
                    min_height: 480.0,
                    ..default()
                },
                present_mode: if perf_mode {
                    PresentMode::AutoNoVsync
                } else {
                    PresentMode::default()
                },
                ..default()
            }),
            ..default()
        }));
    }
    // The GPU compute sim is the default; `cpu` (or `nosim`) selects the
    // CPU path, kept as the reference implementation and render-floor probe.
    let cpu = boids::cpu_sim_selected();

    // Background tint from the original: rgb(0.1, 0.1, 0.12).
    app.insert_resource(ClearColor(Color::srgb(0.10, 0.10, 0.12)))
        .insert_resource(boids::PinnedAttractor(pin))
        .insert_resource(boids::HeadlessRender(headless))
        .insert_resource(if cpu {
            render::SimMode::Cpu
        } else {
            render::SimMode::Gpu
        })
        .init_state::<AppState>()
        .add_plugins((
            settings::plugin,
            boids::plugin,
            render::FlockRenderPlugin { quads },
        ));
    if !cpu {
        app.add_plugins(gpu_sim::FlockGpuSimPlugin);
    }
    // The UI needs a window; everything else runs the same headless. The UI
    // plugin owns FrameTimeDiagnosticsPlugin (for its fps readout), so the
    // headless path registers it itself for `print_fps`.
    if headless {
        app.add_plugins(FrameTimeDiagnosticsPlugin::default());
    } else {
        app.add_plugins(ui::plugin);
    }
    // Uses println! rather than Bevy's LogDiagnosticsPlugin — the latter logs
    // through the `log` facade, which Cargo.toml statically caps at WARN in
    // release builds (`release_max_level_warn`).
    if perf_mode {
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
