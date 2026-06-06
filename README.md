# love-bevy-online — the LÖVE experiments in Bevy/Rust

A Bevy port of the LÖVE/Lua `love-online` project: a start menu with a
live experiment animating behind it, and three experiments so far —
**flock** (Reynolds boids), **fish** (a procedural FABRIK fish that
grows by eating, swimming as a whole school of boids once the Fish slider
goes past 1), and **flow field** (a Perlin-noise vector field shown as
streamlines, arrows, a colour gradient, or particles riding it with
glowing trails). Same simulation rules, same tunables, same UI behaviour —
rebuilt on Bevy to see how far the same experiments can be pushed in Rust.

Answer so far: **the LÖVE original capped at 300 boids; this port holds
~100+ fps at 640,000**, with the slider allowing 1.28 million. The fish
school holds ~100+ fps at 4,096 spline-rendered fish (the original's
school suggested raising past ~30 "with care"), slider allowing 8,192.
The flow field holds ~120 fps at 140,000 glowing-trail particles (the
original capped its slider at 6,000), slider allowing 300,000.

```sh
cargo run            # dev profile: fast iteration (dynamic linking, opt-level 1)
cargo run --release  # optimized build
```

A normal launch starts on the menu, with the flock running ambiently
(mouse ignored) behind a dim overlay — like the original's animated menu
backdrop. Click an experiment to play it.

## Controls

| Input | Action |
|---|---|
| Menu: click an experiment | Start it fresh (`Esc` on the menu quits) |
| Mouse | Attracts the flock from afar, scatters it within 100 px |
| `Esc` / `O` | Pause + options popup (instructions, sliders, VSync checkbox, Reset/Resume/Restart/Main Menu) |
| `R` | Restart (respawn the flock, keep settings) |
| Top-right panel | Live tuning while playing; `Hide`/`Show` collapses it |

## Tunables

| Setting | Range | Default | |
|---|---|---|---|
| Boids | 10 – 1,280,000 | 50 | log-scale slider; the LÖVE original capped at 300 |
| Speed | 50 – 1500 | 400 | |
| Separation | 0 – 8 | 1.8 | |
| Alignment | 0 – 6 | 1.0 | |
| Cohesion | 0 – 6 | 1.0 | |

All apply live, including the boid count (the flock grows/shrinks on the fly).

## Architecture

(Design rationale — what each piece is and why it exists — lives in
[ARCHITECTURE.md](ARCHITECTURE.md).)

The crate is laid out for more experiments to land beside the flock,
mirroring the original's `minigames/` registry:

- `src/app.rs` — the shared shell: the `Menu`/`Playing`/`Options` state
  machine, the camera, the simulation bounds, the perf-harness plumbing.
- `src/menu.rs` — the start menu; `src/ui.rs` — the shared SUIT theme and
  widgets.
- `src/experiments/` — the registry (`EXPERIMENTS`) the menu lists; each
  experiment is a folder of plugins under it (`src/experiments/flock/`).

Within the flock, two simulations share one renderer; both implement exactly
the same rules (`target_force = k * limit(normalize(dir) * max_speed - vel,
0.3)`, converted from the original's per-frame units to px/s² so physics are
frame-rate independent).

- **GPU compute sim** (default, `src/experiments/flock/gpu_sim.rs`) — six
  WGSL dispatches per frame: clear grid → atomic histogram → prefix scan →
  scatter into cell-major order → steer + integrate → reverse-copy the
  render instances. State lives in storage buffers; the CPU only updates a
  uniform (and uploads spawns) — per-boid data never crosses the bus.
- **CPU sim** (`cargo run --release -- <count> cpu`,
  `src/experiments/flock/sim.rs`) — the reference implementation: parallel
  counting sort into SoA arrays, then a branchless, hand-SIMD (NEON `Vec4`)
  steering kernel on the compute task pool. Kept because it's the
  behavioural baseline the GPU port is checked against, and it's an
  instructive artifact on its own (~320k boids at ~125 fps).
- **Renderer** (`src/experiments/flock/render.rs`) — the whole flock is *one* instanced draw
  call into Bevy's `Transparent2d` phase. The boid shape (red dot + white
  triangle) is baked once, on the CPU, into a 20x12 coverage texture, and
  each boid is an alpha-tested quad: two triangles, six shader-generated
  vertices, `[pos, cos/sin]` per boid in the only vertex buffer. Opaque, no
  MSAA (matching the LÖVE original), depth-written. Alpha-tested fragments
  forfeit the tile GPU's hidden-surface removal, so the instance buffer is
  reversed and z decreases across it: the draw runs front-to-back and
  early-z culls a dense pile's overdraw instead (~78 → ~140 fps at 640k
  pinned), while later boids still draw on top like the original. The
  pre-bake 12-vertex geometry pipeline is kept behind the `geo` flag as the
  visual reference — the baked quad is pixel-equivalent at game scale.
  One entity per boid was the original design — it collapsed past ~40k
  under per-entity engine bookkeeping (transform propagation, visibility,
  extraction), which is why boids are plain data, not entities.

Neighbour search is a flat counting-sort grid (cell = the 100 px neighbour
radius), so each boid's 3x3 neighbourhood is three contiguous memory spans.
Above `MAX_NEIGHBOUR_SAMPLES = 128` candidates, each span contributes a
proportional contiguous block at a per-boid, per-frame pseudo-random offset
(circular): steering uses only the *direction* of neighbour aggregates, so
uniform sampling is statistically transparent — and below the budget the
scan is exhaustive, covering the original's 10–300 range in ordinary play.

## Performance

Measured on an M4 Pro, release, 1280x800, via the headless perf harness
(below). "Pinned" is the sustained worst case: a permanent attractor at
screen centre piles the whole flock into a dense ring.

| Boids | spread | pinned ring |
|---|---|---|
| 320,000 | ~315 fps | ~180 fps |
| **640,000** | **~165 fps** | **~135 fps** |
| 1,280,000 | ~85 fps | ~59 fps |
| 2,097,152 | ~80 fps | — |

Past 640k the frame is pinned by two floors, and the next doubling misses
~100 fps on its worst case (1.28M pinned: ~59 fps) by more than any tuning
can recover:

- **Simulation**: 1.28M boids × up to 128 neighbour samples ≈ 164M
  steering samples per frame. The sample budget *is* the behavioural
  contract — cutting it changes how the flock moves.
- **Rendering**: 2 triangles per boid is the floor for a textured boid
  (≈ 7.7M vertex invocations at 1.28M; the render-only floor is ~127 fps,
  up from ~93 with the 12-vertex geometry).

That's where the optimize-without-changing-behaviour loop stops.

### Perf harness

```sh
cargo run --release -- <count> [fish|flow] [pin] [headless] [cpu] [nosim] [geo]
```

Prints fps once a second (vsync off). Any CLI argument skips the menu and
boots straight into the experiment, so harness numbers stay comparable
across versions. Flags compose:

- `fish` — perf-test the fish experiment instead of the flock: `<count>`
  fish swim as the real school-of-boids path (`pin`: the cursor parks at
  the centre and the whole school piles onto it — the sustained worst
  case; without `pin`, headless runs have no pointer and the school
  flocks freely). Certified: a school of 4096 at ≥~100 fps on the M4
  Pro; see ARCHITECTURE.md.
- `flow` — perf-test the flow field: `<count>` particles ride the field.
  Extra flags pick the other views (`streamlines`, `arrows`, `gradient`),
  `evolve` animates the field, `worst` sets the field-rebuild worst case
  (scale 4, octaves 6, warp 5, evolve, maxed streamline detail/length),
  `fade04` the shortest trails; `detail=N`/`length=N`/`seed=N` override
  single tunables for probe grids. Certified: 140,000 particles at
  ~120 fps on the M4 Pro.
- `pin` — fake mouse attractor at screen centre (sustained worst case).
- `headless` — no window: renders to an offscreen texture, schedule
  free-runs, and a snapshot lands in `/tmp/boids_headless_{0,1,2}.png`
  every 5 s. Immune to macOS display-sleep/occlusion throttling, which
  silently caps presentation (and poisons fps numbers) on a sleeping or
  covered display.
- `cpu` — use the CPU reference sim.
- `nosim` — spawn and render only: isolates the render floor.
- `geo` — render the original 12-vertex boid geometry instead of the
  baked-texture quads (the visual reference the bake is compared against).

## Running in the browser

The game also builds for the web — full circle, since the LÖVE original was
capped at 300 boids *because* it ran in a browser. It requires **WebGPU**
(current Chrome/Edge/Firefox, Safari 26+): the compute sim has no WebGL2
fallback, because WebGL2 has no compute shaders. The sim was written against
baseline WebGPU limits from the start, so it ports verbatim; with no CLI
args the web build boots the right defaults (GPU sim + quad renderer), and
the native-only paths (perf flags, CPU sim) are simply unreachable.

```sh
rustup target add wasm32-unknown-unknown   # one-time
web/build.sh                               # → dist/ (installs a matching wasm-bindgen-cli if needed)
python3 -m http.server -d dist 8080        # open http://localhost:8080
```

Deployment is a static site: the `Dockerfile` builds `dist/` and serves it
with nginx on port 80 (precompressed; in Coolify, add the repo as an
application with the Dockerfile build pack). Native builds are unaffected
by any of this — the wasm bits in `Cargo.toml` / `.cargo/config.toml` are
target-gated.

## How the LÖVE code maps to Bevy

| LÖVE original | Bevy port |
|---|---|
| `Flock` table with parallel `positions`/`velocities` arrays | `Flock` resource (CPU) / storage buffer (GPU); boids are data, not entities |
| `lib/grid.lua` spatial hash | Flat counting-sort grid, rebuilt per frame (parallel on CPU, atomics on GPU) |
| `target_force` = `k * limit(normalize(dir)*max_speed - vel, 0.3)` | Same formula, per-frame units converted to px/s² (frame-rate independent) |
| `(pos + screen) % screen` wrap | Single conditional wrap per axis (boids move px per frame, never a screen) |
| `love.graphics.circle` + `polygon` per boid | One instanced draw of alpha-tested quads sampling the baked shape (`geo` flag: shared 12-vertex mesh) |
| `Flock:setSize` truncating arrays | Same: grow appends random boids, shrink truncates (GPU) / removes uniformly (CPU) |
| SUIT immediate-mode sliders/buttons | Hand-rolled retained `bevy_ui` sliders (`Interaction::Pressed` persists through a drag) |
| `pointerOverUI` → `ignore_mouse` | `PointerOverUi` resource |
| `state = 'menu' / 'playing' / 'options'` | `States` enum; sim steps in `Menu` (the live backdrop) and `Playing`, popup on `OnEnter(Options)` |
| `minigames` registry + menu backdrop pool | `src/experiments/` registry; flock + fish + flow, random backdrop per menu visit |
| `school.lua` (boids of fish, its own minigame) | The fish experiment's "Fish" slider: count > 1 swims the school's boids rules, food objective kept |
| `lib/flow.lua` angle grid + canvas blit + particle trails | `FlowField` resource + a re-emitted-on-rebuild static layer; trails expand from raw ring buffers in the vertex shader |

Steering constants kept from the original: `max_force 0.3`, separation radius
50 px, neighbour radius 100 px, mouse attract `k = 4` / repel `k = -6` inside
100 px.

Deliberate differences:

- The menu lists three experiments so far (the original had five), with a
  random backdrop picked per visit like the original's pool.
- The original kept "fish" and "school" as separate minigames; here the
  school is the fish experiment's count slider — and unlike the original's
  school, the fish still eat (the food gently pulls fish within 100 px,
  or a cursor-parked school could never reach it through its own mouse
  repulsion). The school minigame's separation/alignment/cohesion
  tunables appear in the popup whenever more than one fish swims.
- School fish never snap. The original's forces can reverse a boid's
  whole velocity between two frames (the bang-bang attract/repel ring,
  separation conflicts in any close pass) — invisible on triangles,
  a flickering whip on spline fish bodies. Here every school velocity
  runs through a slew limiter (headings turn at most ~5 rad/s, speeds
  relax through a short low-pass): the steering rules are still the
  ported ones, but the fish respond to them like creatures with
  momentum. On top of that, once the pointer rests, fish that have
  reached it settle into a slow mill around it — the single fish's
  stationary-pointer orbit, schooled. Pointer movement releases the
  mill instantly, and the food pull rides outside it, so a milling
  school still dives and eats.
- The flow field is deliberately *better* than the original rather than
  1:1 (the rest of its tunables, palettes, and view modes are the
  original's). The original samples the nearest grid cell (trajectories
  kink at every cell edge) and steps with Euler; this port interpolates
  the field bilinearly (direction-vector lerp, so the angle wrap is
  seamless) and advects with RK2 — streamlines and particle paths are
  smooth curves. New `Evolve` tunable: the field drifts through a third
  noise dimension over time (0 = static, the original). The gradient view
  interpolates colours across cells instead of flat rects; streamlines
  are tapered, feathered ribbons instead of uniform hairlines; trails
  record at a fixed 60 Hz of simulated time (the original recorded per
  frame, so its trails shrank as fps rose); and the seed pans the noise
  offset linearly instead of reshuffling (each integer seed used to
  re-roll an RNG) — every integer seed below 256,000 is a distinct
  field (the slider spans them all; the offset wraps the Perlin
  lattice's 256-unit period, so any seed keeps full float precision),
  and clicking the Seed value label (or any flow value label) types or
  pastes an exact value — Cmd/Ctrl+C/V — to share a field. Particle
  trails are expanded on the GPU from each particle's raw ring buffer
  (a vertex-pull shader; the glow's cross profile is computed per
  fragment, halving the vertex work), which is what buys the
  140k-particle ceiling.
- An FPS readout under the score, to compare runtimes (the point of the
  experiment) — plus a VSync checkbox in the options popup, so a normal run
  can uncap presentation without the perf harness. Heads-up when reading
  the number with VSync on: it tracks the display's *current* refresh rate
  (a MacBook on battery drops ProMotion to 60 Hz; perf flags disable vsync
  entirely).
- Physics are frame-rate independent (the original integrates per frame).
- The Boids slider is log-scaled so the huge range stays draggable.
- No antialiasing — same as LÖVE's default, and 4x MSAA is pure fill-rate
  cost once the flock piles up.

## Cargo profile notes

`Cargo.toml` follows the "fast iterative compiles" setup: dynamic linking,
`opt-level = 1` for our code / `3` for dependencies, and wgpu-types debug
assertions disabled.

- Dynamic linking is the default `dynamic` feature. The resulting binary only
  runs via `cargo run` (it needs Rust's dylibs). Build a standalone binary
  with `cargo build --release --no-default-features`.
- Heads-up: `log`'s `release_max_level_warn` feature statically strips
  info-level logging from every crate using the `log` facade in release
  builds — including Bevy's own `LogDiagnosticsPlugin`. That's why the perf
  harness prints fps with `println!`.
