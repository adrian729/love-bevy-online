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
  startup and gate their systems on the `CurrentExperiment` resource (set by
  the menu, which also picks a random backdrop per visit — `menuBgPool`).
  The chrome the original kept in `main.lua` — HUD (score/fps/hint), the
  options-popup shell with its nav buttons, [Esc]/[O] pause, the generic
  slider (`SliderBinding`), checkbox, and option-cycler (`CyclerBinding`,
  the original's `type = 'options'` tunables) widgets — lives in
  `src/ui.rs`; experiments contribute only their own content. The typed
  value entry on slider value labels (`value_entry_plugin`) and the
  hover tooltips (`Tooltip` components over a shared bubble system) are
  opt-in per experiment — only flow registers/attaches them; the
  flock/fish sliders are exactly as they were.

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

## The fish experiment (`src/experiments/fish/`)

A port of the original's `minigames/fish.lua` + `lib/{fish,chain,joint,
vec2,splines}.lua`: a procedural 12-joint FABRIK fish that chases the
cursor, orbits it when it rests, and grows by eating food — plus a "Fish"
count tunable that folds in the original's *school* minigame
(`lib/school.lua`): more than one fish swims as a school of boids while
the food rules keep working. Deliberately NOT built on the flock's
machinery — different problem, different shape:

- `sim.rs` — the FABRIK spine, the minigame rules, and the school, all in
  **window coordinates** (top-left, y-down) so every Lua formula ports
  sign-identical and the cursor needs no transform. Port-parity is pinned
  by tests whose expected values come from running the actual Lua under
  LuaJIT (a 180-frame spine trajectory matches every joint within 0.05px;
  the `renderV2` spline and `constrainAngle` match to 1e-3; the school
  boids match `lib/school.lua` across 60 per-frame-resynced steps — a
  free-run comparison is meaningless for a chaotic system). State is a
  plain `Vec<Fish>`: at count 1 the game drives it from the mouse
  (byte-identical to the pre-school fish game); at count > 1 a `School`
  of boids (the school's own constants: max_force 0.6, separate 75,
  neighbour 150, mouse attract/repel 4/−6 at 50px, no screen wrap; the
  separation/alignment/cohesion weights are the school minigame's live
  tunables, surfaced as popup sliders whenever count > 1) drives each
  fish via `Fish::set_target(pos, vel)` — the original
  `school.lua`'s contract; spine heads are never written back. Any fish
  that touches the food eats it and grows. One addition the original
  never needed: the food pulls fish within 100px (`steer_to_target`'s
  unused target mechanism, k = 10) — without it the mouse's −6 repulsion
  makes a cursor-parked school physically unable to reach the food
  (`school_reaches_the_food` proves the contract, and
  `school_stays_steerable_away_from_food` the player's control). A
  second addition: **smoothing**, in two layers, because the Lua forces
  routinely exceed the speed cap per frame (the bang-bang mouse ring,
  separation conflicts in close passes) and can reverse a boid's full
  velocity between two frames — invisible on triangles, a flickering
  whip on spline fish bodies whose wave offset also flips sides with
  the velocity. Layer one, always on: a slew limiter — headings turn at
  most 5 rad/s (a U-turn is a ~0.6s arc; the flicker needs ~π per
  frame), speeds relax through a 0.1s low-pass; the forces stay as
  ported, only the velocity's response is made continuous
  (`school_never_snaps_headings` stress-drives a packed blob through a
  sweeping, resting, then bolting cursor and asserts the bound on every
  fish every frame). Layer two: once the pointer has rested ~¼s, fish
  within 200px (fully inside 100px) aim for a slow tangential mill
  around it (0.6·max_speed on a ~70px ring — fast enough to keep FABRIK
  targets past the chain's 2px freeze threshold), capped at an 0.85
  blend so separation keeps spacing the pile. The food impulse is
  integrated outside the mill blend, so a calmed school still dives and
  eats; pointer movement resets the blend instantly
  (`school_calms_into_a_mill_at_idle_pointer`,
  `moving_pointer_keeps_the_school_raw`). The parity tests run the
  integration with both layers off — that path is the Lua one. Above
  128 neighbour candidates the 3x3 scan samples proportional salted
  blocks per bucket — the flock's `MAX_NEIGHBOUR_SAMPLES` trick,
  re-derived (the steering only uses aggregate directions); below the
  cap the scan is exhaustive and Lua-exact. A third addition (LÖVE
  could never observe it: its cursor always has a position): while the
  pointer is **out of the window**, fish go for the food instead — the
  lone fish swims to it directly (orbit state reset, so the pointer's
  return starts a plain chase), and the school enters a *graze* mode:
  the food takes over the mouse's far-field attraction (k = 4, no repel
  ring) and the approach speed is capped at 0.8·turn_rate·distance.
  The cap is what makes the eat land: the slew limiter gives every fish
  a minimum turn radius of speed/turn_rate (64 px at the default 320 —
  over three times the 20 px eat radius), and a turn-limited pursuer
  chasing a point it cannot out-turn settles on a *stable orbit* at that
  radius — the school circled the food forever. Capped, holding any
  circle needs less turning than the limiter allows, so the orbit
  contracts into a strike at every speed setting
  (`grazing_school_breaks_the_orbit_and_eats` starts the school on the
  stuck orbit itself, at the default speed and the slider max).
  In-window behaviour is byte-identical to before; headless perf runs
  have no window at all and stay pure flocking
  (`lone_fish_grazes_while_the_pointer_is_away`,
  `school_goes_for_the_food_while_the_pointer_is_away` drive the real
  system against a scripted `Window` cursor).
- `render.rs` — per-frame vector geometry in the original's painter order
  (food, lateral fins, tail, body, dorsal, eyes), each part splined with
  the original's Hermite "v2" algorithm (NOT Catmull-Rom), 0.5px-deduped,
  filled by ear clipping (`love.math.triangulate`) and outlined like
  LÖVE's "smooth" 1px line (solid core + 1px feather). The tail outline is
  intentionally open (hidden under the body) and the body outline closes
  at the nose, exactly like the Lua.

Perf shape (the scaling loop's findings, M4 Pro, headless): geometry
generation is the cost — per fish ~3.2µs single-threaded after the loop's
optimizations (spline sampling capped at ~1.1/px of chord, sub-pixel
polyline simplification with a bounded merge window, reflex-aware O(n)
ear clipping, pooled scratch buffers, fish-parallel build + parallel
concatenation on the compute pool). Triangles reach the GPU through a
custom 12-byte vertex (pos f32x2 + unorm color) and persistent
`RawBufferVec`s — the `Assets<Mesh>` path re-interleaved and re-copied
the full mesh every frame and dominated past ~2k fish. The school's
boids pass is parallel and sample-capped (uncapped, a pinned 4096-fish
pile is O(n²): ~12 fps; capped + chunked it reads ~157 — and ~170–211
once the calm regime spreads a pinned pile into its even milling
annulus). Certified:
**a school of 4096 ≥ ~100 fps** (~143–178 pinned in this loop's windows);
the count slider allows 8192 (2x certified, the flock's convention),
which reads ~77–99 pinned. The floor there is per-fish CPU geometry plus
~78MB/frame of vertex traffic; the identified next lever is GPU-side
geometry generation from uploaded spines (~1MB/frame), an investment of
the same shape as the flock's compute sim.

## The flow-field experiment (`src/experiments/flow/`)

A port of the original's `minigames/flow.lua` + `lib/flow.lua` — a grid
of flow angles from fractal Perlin noise (octaves, domain warp, swirl
bias), shown as streamlines, per-cell arrows, a colour gradient, or
particles riding the field with glowing trails. Unlike the other two,
this port is deliberately **better, not 1:1** (the explicit request):
tunable ranges, defaults, palettes, and view semantics are the
original's; the behavior upgrades are documented as a list in
`sim.rs`'s module docs. The headline ones: bilinear field sampling
(direction-*vector* lerp, wrap-safe — the Lua reads the nearest cell, so
every trajectory kinks at cell edges), RK2 advection with the trace
step capped at 10 px (a half-cell step is huge on coarse grids and drew
visibly polygonal lines; reach stays length × scale), an `Evolve`
tunable (the field drifts through a third noise dimension at a fixed
30 Hz tick; 0 = static = the original), a per-pixel GPU gradient
view, tapered feathered streamline ribbons, frame-rate-independent
trails (fixed 60 Hz of simulated time; the Lua recorded one sample per
rendered frame), and continuous seeds (the seed pans the noise offset
linearly instead of re-rolling an RNG, and the raw f32 slider value
sits in the rebuild signature, so small seed steps morph). The Seed
slider spans all **256,000 distinct fields** — the joint period of the
two per-seed pans over the Perlin lattice's 256-unit repeat — and the
offset wraps into that lattice period (`seed_base`), so every seed
keeps full f32 noise precision. Exact seeds are shareable: flow's
value labels accept typed input (click → type → [Enter], Cmd/Ctrl+C/V
to copy/paste) via the opt-in `value_entry_plugin`. Every tunable
range was audited: caps are either widened past the original's
arbitrary ones or kept with a stated physical/measured reason
(`FlowParam::range`'s per-line notes).

- `sim.rs` — its own deterministic 3D Perlin (fixed permutation table,
  like `love.math.noise`'s global one; the seed moves the sample offsets
  instead — that's what makes scrubbing continuous), fbm with the Lua's
  octave/persistence shape and its sequential warp dependency kept,
  `FlowField` (angles + unit dirs, row-major), streamline tracing and
  the particle SoA (positions, lifetimes, 20-sample trail ring buffers),
  all in window coordinates. Rebuilds are chunk-parallel on the compute
  pool and throttled to ~15/s while a slider is held with an exact
  rebuild on release (the original's `REBUILD_THROTTLE`). The Lua's
  `signature()` is a `PartialEq` struct of the build-affecting tunables.
  Everything random is a tiny xorshift64* seeded from the field seed —
  fields, streamline starts, and particle spawns reproduce exactly in
  tests, no `rand` in the sim path.
- `render.rs` — three layers, three pipelines, all blending
  premultiplied alpha (one blend state serves the normal layers,
  `(rgb·a, a)`, and the additive trail glow, `(rgb·a, 0)`, chosen per
  vertex). The **gradient layer** (the Gradient view and the dimmed
  background of the other views) is per-fragment: one window quad whose
  fragment shader bilinearly samples the field's direction grid
  (uploaded only on rebuild) and runs the Lua palette formulas per
  pixel — a vertex-color grid creased along quad diagonals and mosaicked
  sharp warp fronts. The **static layer** (the stroke views' geometry)
  is the original's render-once-to-a-canvas: CPU-built 12-byte-vertex
  triangles re-emitted and re-uploaded only when the field's version
  bumps. The
  **trail layer** is a GPU vertex-pull pipeline: the CPU uploads each
  particle's raw ring buffer plus a 24-byte meta record (head position,
  ring state, packed palette colour) and the vertex shader expands the
  tapered glow ribbons and head dots from `vertex_index` alone — no
  vertex or index buffer. One quad per segment: the glow's cross
  profile (a tent — full at the spine, zero at the edges) is computed
  per fragment from an interpolated coordinate, which halved the
  vertex count of the original two-quads-per-segment expansion (and
  made the head dot round instead of a diamond). CPU-expanded trails
  were measured geometry-throughput-bound (~170MB/frame at 100k
  particles through emit → merge → extract → upload; ~48 fps); the
  ring buffers are ~18MB/frame, and with the fragment-profile quads
  the same count reads ~168 fps — still bounded by GPU geometry work
  (the known next lever is incremental ring-buffer uploads).
  Angle→colour goes through a 256-entry linear-rgb LUT per palette
  (the hsv conversion would otherwise run per particle per frame).
- `ui.rs` — the original declares `onscreenControls = 'all'`, so the
  top-right panel carries every tunable (gated rows included); the
  options popup lays the same ~20 controls out in a two-column wrap
  (flow-local — the shared popup shell is untouched). View/Palette are
  the shared cycler widget's two bindings; `visibleIf` rows are marker
  components toggling `Display`, the fish pattern. "New field" re-rolls
  the seed only (the original's `regenerate`); [R]/Restart also respawns
  the particles (`reset`). Flow opts its value labels into the shared
  typed-entry systems (an `Interaction` marks them clickable) and
  cancels any open edit the moment the screen changes, so an edit
  never outlives flow being front and center — while one is open,
  [Esc] cancels it instead of toggling the popup, and a typed "r"
  doesn't restart. Every flow control also carries a hover tooltip
  (`FlowParam::tip` etc. — the original explained nothing): hold the
  cursor on a control's name, a checkbox, or a button for ~half a
  second and a bubble explains what it does.

Perf (M4 Pro, headless): **140,000 particles certified at ~120 fps**
(50k ≈ 340, 100k ≈ 168, 200k ≈ 88; the slider allows 300k ≈ 55 fps —
≈2x certified, the flock/fish convention). The field-rebuild worst case
(scale 4, octaves 6, warp 5, evolving at 30 Hz — ~64k cells × ~13 fbm
samples each) reads ~880 fps thanks to the chunk-parallel rebuild; the
streamline worst case (detail 2500 × length 300, retraced per evolve
tick) holds ~160–215 fps. Detail's 2500 cap is *measured*, not
inherited: at 5000 the retrace no longer fits inside one 30 Hz evolve
tick, so every frame pays a full rebuild and fps cliffs to ~24. Length
is border-bounded (lines leave the window long before the step cap), so
doubling it to 300 was free.

## The forest experiment (`src/experiments/forest/`)

A port of the original's `minigames/tree.lua` + `lib/tree.lua`: a forest of
procedural **L-system** trees grown by randomised string rewriting and walked
by a turtle into line geometry, baked once and redrawn each frame (the
original's render-once-to-a-canvas). Like flow, it declares
`onscreenControls = 'all'` and re-uploads its baked geometry only on a version
bump. Like flow's flow-field port, it is deliberately **better, not 1:1** — the
13 original tunables and their defaults are kept exactly (a red, brightening-
outward L-system tree at the defaults), and every addition zeroes back to the
faithful original.

- `sim.rs` — the model, in window coordinates (y-flip at vertex emission, the
  flow/fish convention). The L-system rewrites **ping-pong `u8` token buffers**
  (`expand_once` matches `lib/tree.lua`'s rewrite order and per-character RNG
  draw count exactly, so the morph-in-place behaviour holds) rather than the
  Lua's O(n²) string concat. A turtle then walks the tokens, **coalescing each
  straight same-level run of `F`s into one feathered quad** (visually identical
  to the Lua's per-segment rectangles, far fewer vertices when `forward` makes
  long limbs). The two-signature scheme is the original's: a **structural**
  change (growth + the 5 rewrite probabilities) regrows the token streams; a
  **geometry** change (branch angle/length/width, size variation, leaf
  size/density, the window size) only re-walks the turtle — both throttled to
  ~15/s while a slider is held, exact on release. Each tree is seeded per index
  (`forest_seed ^ splitmix(i)`), so a structural tweak morphs every tree in
  place and adding/removing trees leaves the rest untouched; regrow + geometry
  build both run **parallel across trees** on the compute pool, then concatenate
  into one merged buffer. The load-bearing safety is a **segment budget**: total
  built `F`-segments are hard-capped (`MAX_SEGMENTS`), enforced *during*
  expansion (a per-tree budget = `MAX_SEGMENTS / count` stops the rewrite early),
  so a pathological slider setting can neither OOM the CPU token buffers nor blow
  up the GPU vertex buffer — it is the cap on *built geometry*, not tokens.
- `render.rs` — one baked vertex buffer over the standard 2D view bind group, a
  single `Transparent2d` indexed draw, premultiplied-alpha (the feather edges
  emit alpha 0), the flow static-layer shape. Two deliberate departures, both
  driven by the workload being **fill/overdraw-bound** (a dense canopy of
  blended feathered branches can't early-z): the **fragment shader stays
  trivial** (it returns the interpolated colour — the 1px feather is CPU-
  expanded, *not* fragment-computed, since more fragment work would only worsen
  the bottleneck), and **colour + wind live in a uniform, not the vertices**.
  The 12-byte vertex carries only `{pos, packed}` where `packed` is
  `{branch level, leaf flag, coreness/alpha, sway weight, per-tree wind phase}`;
  the **vertex shader** computes the branch colour from `lib/color.lua`'s
  `hsl2rgb` (`base_hue + hue_spread·level`, brightness ×1.2^level, clamped after
  the HSL like LÖVE's draw-time clamp, then sRGB→linear), so the hue/spread/
  brightness/leaf-hue sliders are free live updates — no regrow, no re-upload.
  The same uniform carries the wind: a **height-weighted horizontal shear** (0 at
  the ground, concentrated toward the tips) times one oscillation **per tree**
  (keyed off the packed per-tree phase, NOT the x position), animated entirely in
  the vertex shader so the geometry stays baked while the canopy moves. Because
  the displacement is a continuous function of height alone, the whole tree leans
  back and forth as one body — horizontal limbs keep their length and branch
  junctions never crack — and the per-tree phase (not x) gives the variety; an
  earlier per-x phase made a travelling wave that visibly stretched horizontal
  branches. Wind 0 = the original's static forest.
- New, additive controls (all at a faithful zero-position by default): **Wind**
  (the headline upgrade — the Lua forest was static; every other experiment
  moves), **Leaves** + leaf size/density/hue (soft leaf fans at the twig tips,
  baked into the same buffer), and **Size variation** (per-tree scale jitter for
  depth). Feathered branches replace the original's hard aliased rectangles (the
  collection's fish/flow line art language).

Perf (M4 Pro, headless): the per-frame cost is **fill/overdraw**, not vertex
throughput or the (one-time, parallel, budget-bounded) regrow. The compound
worst case — `dense` probs (growth 15, every node branching three ways and
extending) with tiny segments for maximum overdraw, plus leaves and wind — holds
**~110 fps at the 700-tree slider max**, and *plateaus at ~100 fps* at any higher
count: the segment budget bounds total geometry and the canopy saturates the
1280×800 screen, so fps cannot fall much further. Ordinary forests (default
probabilities, even at growth 15 and hundreds of trees) run 1,000+ fps. The
budget (`MAX_SEGMENTS`) is the governor, tuned so the *pathological* worst case
sits in the ~100–120 band while normal use never approaches it. The one
remaining lever — opaque branch cores with depth-written early-z (the flock's
trick) to cull a dense canopy's overdraw — is deliberately *not* taken: it would
trade away the smooth feathered edges that are the port's visual upgrade, and the
worst case already meets target. That is the agreed stopping line.

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
