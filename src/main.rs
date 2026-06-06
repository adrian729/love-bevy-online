//! love-bevy-online — a Bevy/Rust port of the LÖVE `love-online`
//! experiments. The menu lists every experiment in [`experiments`]; the
//! first (and so far only) one is **flock**, the Reynolds boids experiment,
//! ported with the same simulation rules and tunables and pushed to its
//! limits.

// Bevy queries routinely trip this lint; Bevy itself allows it.
#![allow(clippy::type_complexity)]

use std::time::Duration;

use bevy::app::ScheduleRunnerPlugin;
use bevy::diagnostic::{DiagnosticsStore, FrameTimeDiagnosticsPlugin};
use bevy::prelude::*;
use bevy::window::{ExitCondition, PresentMode};
use bevy::winit::WinitPlugin;

mod app;
mod experiments;
mod menu;
mod ui;

use app::AppState;
use experiments::{CurrentExperiment, ExperimentId, fish, flock};

fn main() {
    // Perf-test mode (`boids <count> [fish] [pin] [headless] [nosim]`):
    // print fps to stdout once a second, with vsync off so readings show
    // real headroom past the display's refresh rate. The count is boids by
    // default, fish with the `fish` flag.
    //   fish     — perf-test the fish experiment instead of the flock.
    //   pin      — fake mouse attractor at screen centre: the sustained worst
    //              case (the whole flock piled into a ring / fish stacked on
    //              one spot).
    //   headless — no window at all; the camera renders to an offscreen
    //              texture and the schedule free-runs. Exercises the full
    //              sim + extract + batch + GPU pipeline, immune to display
    //              sleep / occlusion throttling (macOS caps presentation on
    //              occluded or sleeping displays, poisoning fps readings).
    //   nosim    — skip steering, isolating the render/ECS floor (flock).
    //   geo      — render the original 12-vertex boid geometry instead of
    //              the baked-texture quads (the visual reference, flock).
    let perf_mode = std::env::args().nth(1).is_some();
    let flag = |name: &str| std::env::args().skip(1).any(|arg| arg == name);
    let pin = flag("pin");
    let headless = flag("headless");
    let quads = !flag("geo");
    let experiment = if flag("fish") {
        ExperimentId::Fish
    } else {
        ExperimentId::Flock
    };

    let mut bevy_app = App::new();
    if headless {
        bevy_app.add_plugins(
            DefaultPlugins
                .set(WindowPlugin {
                    primary_window: None,
                    exit_condition: ExitCondition::DontExit,
                    ..default()
                })
                .disable::<WinitPlugin>(),
        );
        bevy_app.add_plugins(ScheduleRunnerPlugin::run_loop(Duration::ZERO));
    } else {
        bevy_app.add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "love-bevy-online".into(),
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

    // Background tint from the original: rgb(0.1, 0.1, 0.12).
    bevy_app
        .insert_resource(ClearColor(Color::srgb(0.10, 0.10, 0.12)))
        // Mirrors the window's starting present mode (perf runs uncap).
        .insert_resource(app::VsyncEnabled(!perf_mode))
        .insert_resource(app::PinnedAttractor(pin))
        .insert_resource(app::HeadlessRender(headless))
        // Perf runs pin the chosen experiment; a normal launch starts on
        // the menu, which re-picks a random backdrop on entry anyway.
        .insert_resource(CurrentExperiment(experiment))
        .add_plugins((
            app::plugin,
            flock::FlockPlugin { quads, headless },
            fish::FishPlugin { headless },
        ));

    // Perf runs (any CLI arg) skip the menu and boot straight into the
    // experiment, so the harness numbers stay comparable across versions —
    // the original's `--state=playing` dev arg. A normal launch starts on
    // the menu.
    if perf_mode {
        bevy_app.insert_state(AppState::Playing);
    } else {
        bevy_app.init_state::<AppState>();
    }

    // The UI and the menu need a window; everything else runs the same
    // headless. Frame diagnostics feed both the HUD's fps readout and
    // `print_fps`, so they register once here.
    bevy_app.add_plugins(FrameTimeDiagnosticsPlugin::default());
    if !headless {
        bevy_app.add_plugins((ui::plugin, menu::plugin));
    }
    // Uses println! rather than Bevy's LogDiagnosticsPlugin — the latter logs
    // through the `log` facade, which Cargo.toml statically caps at WARN in
    // release builds (`release_max_level_warn`).
    if perf_mode {
        bevy_app.add_systems(Update, print_fps);
    }
    bevy_app.run();
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
