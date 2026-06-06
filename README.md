# Flock — Reynolds boids in Bevy/Rust

A Bevy port of the **flock** experiment from the LÖVE/Lua `love-online` project.
Same simulation rules, same tunables, same UI behaviour — rebuilt idiomatically
on Bevy's ECS to compare the two stacks on the same experiment.

```sh
cargo run            # dev profile: fast iteration (dynamic linking, opt-level 1)
cargo run --release  # optimized build
```

## Controls

| Input | Action |
|---|---|
| Mouse | Attracts the flock from afar, scatters it within 100 px |
| `Esc` / `O` | Pause + options popup (instructions, sliders, Reset/Resume/Restart) |
| `R` | Restart (respawn the flock, keep settings) |
| Top-right panel | Live tuning while playing; `Hide`/`Show` collapses it |

## Tunables

| Setting | Range | Default | |
|---|---|---|---|
| Boids | 10 – 20,000 | 50 | log-scale slider; the LÖVE original capped at 300 |
| Speed | 50 – 1500 | 400 | |
| Separation | 0 – 8 | 1.8 | |
| Alignment | 0 – 6 | 1.0 | |
| Cohesion | 0 – 6 | 1.0 | |

All apply live, including the boid count (the flock grows/shrinks on the fly).

## Performance

Steering runs in parallel on the compute task pool (`Query::par_iter_mut`
against a frame-start snapshot + spatial hash). Measured on an M4 Pro,
release build: **10,000 boids at ~120 fps (vsync-limited)**, **20,000 at
~85–95 fps**, and **~100–115 fps sustained with all 20,000 held in a ring on
a pinned attractor** (the worst case — the whole flock parked on the cursor;
neighbour sampling bounds it, see below). Past ~20k the cost is per-entity
overhead (transform propagation, extraction, batching), which grows linearly;
the next big jump would be a GPU-compute sim.

The worst case is bounded by neighbour sampling: each boid examines at most
512 candidates per frame, stride-sampled across its 3x3 cells. Steering only
uses the *direction* of the neighbour aggregate, so the sample is
statistically unbiased; with ≤512 candidates the scan is exhaustive and
bit-for-bit exact, which covers the original's entire 10–300 range.

Perf-test mode — pass an initial count and fps prints to stdout once a second:

```sh
cargo run --release -- 10000
```

## How the LÖVE code maps to Bevy

| LÖVE original | Bevy port |
|---|---|
| `Flock` table with parallel `positions`/`velocities` arrays | One entity per boid: `Boid` + `Velocity` + `Transform` (`src/boids.rs`) |
| `lib/grid.lua` spatial hash | `HashMap<IVec2, Vec<usize>>` rebuilt per frame, cell = neighbour radius; steering+integration parallelized per boid |
| `target_force` = `k * limit(normalize(dir)*max_speed - vel, 0.3)` | Same formula, converted from per-frame (60 fps) units to px/s² so it is frame-rate independent |
| `(pos + screen) % screen` wrap | `rem_euclid` wrap in world space |
| `love.graphics.circle` + `polygon` per boid | One shared vertex-colored `Mesh2d` (red dot + white heading triangle), instanced by all boids |
| SUIT immediate-mode sliders/buttons | Native retained `bevy_ui` nodes; sliders are hand-rolled — `Interaction::Pressed` persists while the mouse is held, so a drag is just "map cursor x onto the track" (`src/ui.rs`) |
| `pointerOverUI` → `ignore_mouse` | `PointerOverUi` resource: true while any UI node reports interaction |
| `state = 'playing' / 'options'` | `States` enum; sim systems gated on `in_state(Playing)`, popup spawned on `OnEnter(Options)` |

Steering constants kept from the original: `max_force 0.3`, separation radius
50 px, neighbour radius 100 px, mouse attract `k = 4` / repel `k = -6` inside
100 px.

Deliberate differences:

- No multi-game main menu — this port is only the flock experiment.
- An FPS readout was added under the score, to compare runtimes (the point of
  the experiment).
- Physics are frame-rate independent (the original integrates per frame).
- The Boids slider is log-scaled so the 10..10,000 range stays draggable.

## Cargo profile notes

`Cargo.toml` follows the "fast iterative compiles" setup: dynamic linking,
`opt-level = 1` for our code / `3` for dependencies, and wgpu-types debug
assertions disabled.

- Dynamic linking is the default `dynamic` feature. The resulting binary only
  runs via `cargo run` (it needs Rust's dylibs). Build a standalone binary
  with `cargo build --release --no-default-features`.
- Heads-up: `log`'s `release_max_level_warn` feature statically strips
  info-level logging from every crate using the `log` facade in release
  builds — including Bevy's own `LogDiagnosticsPlugin`. That's why perf-test
  mode prints fps with `println!`.
