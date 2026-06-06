# Architecture

What this codebase currently is and why each piece exists. (The README covers
how to run it and how it maps to the LÖVE original; this document explains the
design.)

## The shape of the thing

The crate is an experiment collection, mirroring the original's `minigames/`
registry. A thin shared shell wraps the experiments:

- `src/app.rs` — the `Menu` / `Playing` / `Options` state machine, the
  camera, `SimBounds`, and the perf-harness plumbing (headless render
  target, pinned attractor). The simulation steps in `Menu` too: the menu's
  live backdrop, the original's `menuBg` (with the mouse ignored, like
  `menuBg:update(dt, true)`). Perf runs (any CLI args) boot straight into
  `Playing`, so harness numbers never involve the menu.
- `src/menu.rs` + `src/ui.rs` — the start menu, and the SUIT theme/widgets
  shared between it and the experiments' own UI.
- `src/experiments/mod.rs` — the registry the menu lists. Each experiment is
  a folder of plugins under `src/experiments/`; plugins register once at
  startup, so a second experiment will gate its systems on a
  `CurrentExperiment` resource rather than being loaded on demand.

The flock experiment itself (`src/experiments/flock/`):

```
              ┌─────────────────────────────┐
 settings.rs  │ SimSettings (live tunables) │
              └──────────┬──────────────────┘
                         │
          ┌──────────────┴───────────────┐
          │                              │
   gpu_sim.rs (default)           sim.rs (`cpu` flag)
   6 compute dispatches           counting sort + NEON kernel
   state never leaves GPU         on the compute task pool
          │                              │
          │ instance buffer              │ FlockRenderData (Vec)
          │ (GPU-resident)               │ extracted + uploaded
          └──────────────┬───────────────┘
                         │
                   render.rs
        ONE instanced draw of the whole flock
        (alpha-tested baked-texture quads;
         `geo` flag: 12-vertex geometry)
```

Two interchangeable simulations feed one renderer. Both implement exactly the
same rules; the CPU one is the readable reference the GPU one is verified
against, and survives as its own artifact (~320k boids at ~125 fps).

## Boids are plain data, not entities

A boid is 16 bytes of state (`vec4(pos, vel)`) in a flat array — not an ECS
entity. One entity per boid was the original design and it collapsed past
~40k boids: transform propagation, visibility culling, extraction and
batching all walk every entity every frame, and that bookkeeping — not the
steering math — dominated the frame. The flock is two resources (state +
instance records) and the renderer is a single custom draw call, so per-boid
engine overhead is zero.

Consequences:

- Boids carry no identity. Growth appends, shrink truncates (GPU) /
  removes uniformly (CPU) — matching the LÖVE `Flock:setSize` semantics.
- The renderer consumes a raw `[pos.x, pos.y, cos, sin]` record per boid;
  nothing else about a boid exists outside the simulation arrays.

## The simulation contract (shared by both sims)

The rules are the LÖVE original's, made frame-rate independent:

- `target_force = k * limit(normalize(dir) * max_speed - vel, MAX_FORCE)`
  for each of: separation (< 50 px), alignment + cohesion (< 100 px), and
  the mouse (attract `k=4` beyond 100 px, repel `k=-6` inside).
- The original integrates per frame with constants tuned at 60 fps; we scale
  forces by `REF_FPS` so the same constants produce the same trajectories at
  any frame rate.
- Toroidal wrap via one conditional add per axis (boids move pixels per
  frame, never a full screen — cheaper than `rem_euclid`, same result).

**Neighbour search.** A flat counting-sort grid, rebuilt from scratch every
frame, with cell size = the 100 px neighbour radius. A boid's 3x3 cell
neighbourhood is then *three contiguous spans* of a cell-major array — the
whole inner loop is linear scans over memory, which both CPUs and GPUs love.
No persistent spatial structure, no incremental updates: rebuilding is a
counting sort, O(n), and trivially parallel.

**The sampling budget.** Above `MAX_NEIGHBOUR_SAMPLES = 128` candidates, each
span contributes a proportional *contiguous block* at a per-boid, per-frame
pseudo-random offset (circular within the span). Why this is sound: steering
uses only the direction of the neighbour aggregates, so a uniform sample is
statistically transparent; why blocks instead of strides: contiguous reads
vectorize and prefetch. Below the budget the scan is exhaustive — which
covers the original's entire 10–300 range, so at original scale this port
computes exactly what LÖVE computed. The budget is part of the behavioural
contract: both sims implement the identical scheme, and the certified
performance numbers assume 128.

## GPU sim (`gpu_sim.rs`) — the default

Six dispatches in one compute pass per frame (dispatch boundaries are
implicit barriers, one pass keeps it cheap):

1. `clear_grid` — zero the per-cell counters.
2. `count_boids` — atomic histogram: each boid increments its cell.
3. `scan_cells` — exclusive prefix sum turning counts into cursors. Runs on
   a *single thread*: there are only ~104 cells (1280x800 / 100px), so a
   parallel scan would be all overhead.
4. `scatter` — each boid `atomicAdd`s its cell cursor and writes itself into
   cell-major order. Positions and velocities go to *separate* arrays so the
   steer pass's distance test streams half the bytes (velocity is only read
   for accepted neighbours). `id_of` records which boid landed where.
5. `steer` — the actual flocking: three spans, block sampling, forces,
   integration, wrap. Writes the new state *and* the render instance
   (`vec4(pos, cos/sin)`). Storing the heading as `(cos, sin)` is free here
   (normalized velocity) and saves an `atan2`/`sincos` per boid per frame
   downstream.
6. `reverse_instances` — copies the instance buffer into reverse boid order
   for the renderer (see "front-to-back" below). A bandwidth-bound copy,
   deliberately separate: in `steer`, ids run roughly ascending within each
   cell span, so writing `instances[count-1-id]` there would turn
   near-coalesced ascending writes into descending ones — measured at ~25%
   of the *whole frame* on Apple silicon. In the copy pass the descending
   side is the read, which caches fine. This pass also runs while paused, so
   count changes made from the options menu stay consistent with what a
   truncation of the authoritative state would show.

The authoritative state lives in storage buffers in insertion order, exactly
like the LÖVE arrays. The CPU's per-frame work is: write one uniform
(settings, dt, mouse, a frame salt for the sampler), and upload freshly
spawned boids' states when the flock grows. Per-boid data never crosses the
bus in either direction — the renderer binds the sim's instance buffer
directly as its vertex buffer.

## CPU sim (`sim.rs`) — the reference

Same algorithm, tuned for a many-core CPU:

- **Parallel counting sort**: per-task histograms over chunks of boids,
  serially combined (the combine is ~104 numbers per task — nothing), then a
  parallel scatter through raw pointers into disjoint, pre-reserved ranges.
- **The steering kernel** processes span candidates four at a time with
  `glam::Vec4` (NEON on aarch64, with a scalar tail), in one flat loop with
  plain local accumulators. Acceptance tests are *branchless* — candidates
  get a 0-or-1 lane weight instead of an `if` — because acceptance is
  effectively random and mispredicted branches cost more than the arithmetic
  they skip (measured 2.3x).
- All four forces (separate / align / cohere / mouse) are then resolved in
  one SIMD evaluation per boid, a zero `k` lane disabling a force.
- Work is spread over the compute task pool in chunks *derived from the
  pool size* (`chunk_sizes`): ~a dozen steering chunks per worker — enough
  for the work-stealing pool to balance the density-skewed work, not enough
  to drown in task-spawn overhead — and 4x bigger sort chunks (uniform work,
  and fewer tasks keep the serial histogram combine short). The ratios are
  the tuning; the absolute sizes adapt to the machine (on the 6-thread pool
  this was tuned on, the derivation reproduces the hand-found 4096/16384).

Why keep it at all: it is the behavioural baseline the GPU port is checked
against (same constants, same sampling scheme, comparable snapshots), it
runs the game fine to ~320k boids, and it documents the algorithm in
ordinary Rust rather than WGSL.

## Renderer (`render.rs`)

One instanced draw call for the entire flock, queued as a custom
`Transparent2d` phase item that reuses Bevy's standard 2D view bind group.
There is no mesh asset, no per-boid entity, no batching — the draw is fully
described by resources.

**Baked-shape quads.** The boid (red dot, radius 3; white heading triangle,
tip 14 px forward) is rasterized once on the CPU into a 20x12 RGBA texture:
8x8 supersampled coverage per texel, RGB premultiplied by coverage so
bilinear filtering at the silhouette doesn't bleed background black, sRGB
storage. Each boid is then a quad — two triangles, six vertices generated
from `vertex_index` in the shader (the only vertex buffer is the per-boid
instance record). The fragment shader samples the texture, `discard`s below
0.5 coverage, and un-premultiplies. One texel maps to one screen pixel, so
no mipmaps are needed, and the bilinear-interpolated 0.5 iso-contour
reproduces the hard, unantialiased edge the rasterized geometry had —
verified pixel-equivalent at game scale against the `geo` reference path.

What it accomplishes: 6 vertex invocations per boid instead of 12 (vertex
rate is the wall at these counts — the render-only floor at 1.28M went from
~93 to ~127 fps), with the same image.

**Front-to-back + early-z.** Alpha-tested ("punch-through") fragments defeat
the hidden-surface removal of tile-based GPUs: depth can only be committed
after the shader decides not to discard, so a dense pile of boids — the
pinned-attractor worst case piles the *whole flock* into a 100 px ring —
would be fragment-shaded thousands of layers deep. The fix is ordering: the
instance buffer is reversed and the per-boid z *decreases* across it, so the
draw runs front-to-back and the GPU's early-z test rejects occluded
fragments against already-committed depth (640k pinned: ~78 fps
back-to-front, ~140 front-to-back). Reversed buffer x reversed z means
later-in-the-flock boids still draw on top, exactly like the LÖVE original's
draw order.

**Pipeline state, and why:**

- *Opaque (no blending)* — boids are solid; skipping the blend
  read-modify-write matters when a pile overdraws the same pixels.
- *Depth-written, `GreaterEqual`* — gives the GPU something to cull with
  (HSR for the `geo` path, early-z for the quads). The z values are tiny
  per-instance offsets; the flock is the only depth-written 2D content.
- *No MSAA* — matches the LÖVE original's look, and 4x multisampling is pure
  fill-rate cost once the flock piles up.
- The `geo` flag switches to the original 12-vertex octagon + triangle
  mesh pipeline (two vertex buffers, no texture). It exists as the visual
  reference for the bake and as the A/B baseline.

**Feeding the draw:** in GPU-sim mode, vertex slot 0 *is* the compute sim's
reversed instance buffer — zero copies. In CPU mode, the sim publishes a
`Vec<BoidInstance>` which extraction reverses into a `RawBufferVec` and
uploads (one memcpy each side, ~5 MB at 320k).

## Where the limits are

Certified (~100+ fps including the pinned worst case, headless, M4 Pro):
**640k boids**. The next doubling (1.28M) lands at ~85 spread / ~59 pinned
because two floors meet:

- **Simulation**: 1.28M boids x 128 samples ≈ 164M steering samples per
  frame. The budget is the behavioural contract; cutting it changes the
  flock.
- **Rendering**: 2 triangles per boid is the floor for a textured boid
  (~7.7M vertex invocations at 1.28M), and the render-only floor is ~127 fps
  before any simulation runs.

Neither can shrink without changing what the game computes or how it looks,
which is the agreed stopping line.

## Portability

Nothing in the architecture is Apple-specific; some tuning is.

- **Runs anywhere Bevy/wgpu runs** (Metal / Vulkan / DX12 — and the browser
  via WebGPU, see the README's web section; WebGL2 is out, it has no compute
  shaders). The WGSL uses only baseline WebGPU features (storage buffers,
  atomics, compute), so the browser enforces nothing the sim doesn't already
  respect. The web build also never hits the one readback-shaped hazard:
  per-boid data never crosses the bus.
- **Universal by design**: no per-boid entities, single instanced draw,
  counting-sort grid, contiguous-span sampling, branchless inner loops,
  opaque + depth-written rendering, fewer vertices per boid. These help on
  every GPU/CPU vendor.
- **Front-to-back ordering** is *required* on tile-based GPUs (Apple, ARM,
  Qualcomm) for the alpha-test path, and is the classic optimal order on
  immediate-mode GPUs (NVIDIA/AMD) too — early-z exists everywhere.
- **Apple-tuned but harmless elsewhere**: the separate reverse-copy pass
  works around Apple's descending-write penalty; other GPUs may not need it,
  but it costs ~nothing. Hoisted SIMD constants in the CPU kernel work
  around an LLVM/macOS pattern (`memset_pattern16` calls in hot loops);
  hoisting is free everywhere.
- **aarch64-specific**: the CPU kernel's `vsqrtq_f32` use is behind
  `cfg(target_arch = "aarch64")` with a scalar fallback, and `glam`'s
  `Vec4` maps to SSE on x86 — the CPU sim compiles and vectorizes on x86,
  just without the hand-NEON sqrt.
- **Self-tuning where it matters**: the CPU sim's task chunk sizes are
  derived from the compute pool's thread count at runtime (the tasks-per-
  worker *ratio* is the tuning; the absolute sizes adapt — see
  `chunk_sizes` in `sim.rs`, pinned by a unit test to the values certified
  here). The GPU workgroup size (256) needs no adaptation: it's the WebGPU
  baseline limit and a multiple of every vendor's SIMD width — the portable
  optimum, not a machine-specific choice.
- **Machine-specific numbers**: every fps figure was measured on an M4 Pro.
  The certified 640k and the 1.28M slider cap describe *this* machine; a
  different GPU shifts where ~100 fps lands, not whether the game works.
  To recalibrate on new hardware: run the perf harness doublings
  (`cargo run --release -- <count> pin headless`), find the largest count
  whose pinned worst case holds ~100 fps, and set the `Count` range max in
  `settings.rs` to 2x it — that one constant is the only calibration.
