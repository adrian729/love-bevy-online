//! The pond — a water layer under (and over) the fish. Render-only and
//! original-free: the LÖVE fish swam on a flat clear color, so there is no
//! fidelity contract here, only the experiment's procedural style.
//!
//! Top-down framing (a pond watched from above, outside the water):
//! - **bed** — one window-covering quad whose fragment shader builds
//!   everything procedurally (the flow experiment's gradient-layer
//!   pattern): large-scale depth patches over a deep blue-green bed, and
//!   animated Voronoi caustic webs playing across it.
//! - **surface** — a coarse CPU wave-equation grid (~6 px cells,
//!   sub-millisecond) splatted by fish head movement and food plops; the
//!   fragment shader samples it bilinearly and uses the surface slope to
//!   refract the bed (the caustics swim under the rings) and to catch sun
//!   glints. An analytic micro-ripple field adds ambient sparkle so the
//!   surface never reads dead-flat.
//! - **bubbles** — flow's vertex-pull trick at its extreme: every bubble's
//!   position, size and life phase are pure functions of `(hash(i), time)`
//!   in the vertex shader. No sim, no buffers, no upload. Top-down,
//!   bubbles rise *toward the camera*: they grow as they ascend, then pop
//!   into a brief expanding ring at the surface and reseed elsewhere.
//!
//! Sort keys: the bed quad draws at -1.0 (under the fish at 0.0 — it is
//! the pond's opaque background, the global `ClearColor` stays untouched),
//! the bubbles at 0.5 (the surface is above the fish). Geometry lives in
//! the sim's window coordinates; the y-flip happens in the shaders via the
//! uniform's `origin` (the fish convention).
//!
//! Three looks, one machine: the popup's Style button cycles the pond
//! through Natural / Sketch / Glossy. The pipelines specialize on
//! `(key, style)`, so the natural style's shaders stay byte-identical.
//!
//! - **Sketch** — the pond as a cartoonist doodles it (the fish's own
//!   flat-fill / white-stroke language): one flat color, and white
//!   hand-drawn ring outlines that wobble, break into dashes and fade as
//!   they expand. Rings are first-class vector events (a small CPU pool,
//!   one vertex-pulled quad each), NOT a wave-grid readout — clean strokes
//!   need parametric circles, not thresholded heights.
//! - **Glossy** — a storybook pond: soft gradients, gentle highlight
//!   contours, the same ring events drawn as clean double-stroke circles.
//!
//! Both new styles ignore the height grid entirely (it only steps while
//! the natural style shows it); the Ripples toggle gates ring spawning
//! instead, and Ripple strength scales ring size.

use bevy::core_pipeline::core_2d::{CORE_2D_DEPTH_FORMAT, Transparent2d};
use bevy::ecs::query::ROQueryItem;
use bevy::ecs::system::SystemParamItem;
use bevy::ecs::system::lifetimeless::SRes;
use bevy::math::FloatOrd;
use bevy::prelude::*;
use bevy::render::render_phase::{
    AddRenderCommand, DrawFunctions, PhaseItem, PhaseItemExtraIndex, RenderCommand,
    RenderCommandResult, SetItemPipeline, TrackedRenderPass, ViewSortedRenderPhases,
};
use bevy::render::render_resource::binding_types::{
    storage_buffer_read_only_sized, uniform_buffer,
};
use bevy::render::render_resource::{
    BindGroup, BindGroupEntries, BindGroupLayoutDescriptor, BindGroupLayoutEntries, BlendState,
    BufferUsages, ColorTargetState, ColorWrites, CompareFunction, DepthBiasState,
    DepthStencilState, FragmentState, MultisampleState, PipelineCache, PrimitiveState,
    PrimitiveTopology, RawBufferVec, RenderPipelineDescriptor, ShaderStages, ShaderType,
    SpecializedRenderPipeline, SpecializedRenderPipelines, StencilFaceState, StencilState,
    TextureFormat, UniformBuffer, VertexState,
};
use bevy::render::renderer::{RenderDevice, RenderQueue};
use bevy::render::sync_world::MainEntity;
use bevy::render::view::{ExtractedView, ViewTarget};
use bevy::render::{Extract, Render, RenderApp, RenderStartup, RenderSystems};
use bevy::sprite_render::{
    Mesh2dPipeline, Mesh2dPipelineKey, SetMesh2dViewBindGroup, init_mesh_2d_pipeline,
};
use rand::Rng;

use super::settings::{FishSettings, WaterStyle};
use super::sim::{FishGame, FishSimSet, Fishes};
use crate::app::{SimBounds, sim_active};
use crate::experiments::{CurrentExperiment, ExperimentId, experiment_active};

/// Surface grid cell size, px. ~6 px keeps a 1280x800 window under ~30k
/// cells (sub-ms CPU step, ~115 KB/frame upload — noise next to the fish's
/// own vertex traffic) while rings still read round.
const RIPPLE_CELL: f32 = 6.0;
/// Wave speed factor c² in grid units. The 2D CFL stability bound is 0.5.
const WAVE_C2: f32 = 0.42;
/// Per-step energy retention — rings fade out in a couple of seconds.
const WAVE_DAMP: f32 = 0.986;
/// Fixed surface step. Frame-rate independent: vsync-off perf runs must
/// not fast-forward the pond.
const WAVE_STEP: f32 = 1.0 / 60.0;
/// Catch-up cap; past this the surface slows down instead of spiralling.
const WAVE_MAX_STEPS: f32 = 4.0;
/// Heights stay bounded no matter how dense the school piles up — a
/// churning pile saturates into white water instead of running away.
const SPLAT_CLAMP: f32 = 2.0;
/// Wake amplitude per pixel of head travel, before the strength dial.
const WAKE_AMP_PER_PX: f32 = 0.09;
/// One frame's wake deposit cap (a 900-speed fish still leaves a wake,
/// not a trench).
const WAKE_AMP_MAX: f32 = 0.8;
/// Fish-size factor: a grown lone fish churns more than a school minnow.
const WAKE_SCALE_GAIN: f32 = 6.0;
/// A head jumping further than this teleported (restart/respawn): no splat.
const WAKE_TELEPORT: f32 = 60.0;
/// The plop where the food was eaten, and the smaller one where the fresh
/// dot lands.
const PLOP_EAT_AMP: f32 = 1.4;
const PLOP_DROP_AMP: f32 = 0.6;
/// Vertices per vertex-pulled bubble: one quad.
const BUBBLE_VERTS: u32 = 6;

// The sketch/glossy styles' ripple rings — vector ring events, one
// vertex-pulled quad each (see RING_SHADER).
/// Live-ring pool cap: bounds the buffer and the overdraw no matter how
/// big the school gets.
const RING_MAX: usize = 512;
/// A swimming head spawns a wake ring every this many px of travel.
const RING_WAKE_EVERY: f32 = 26.0;
/// Ring lifetimes by kind, seconds (wake, plop, ambient squiggle).
const RING_LIFE: [f32; 3] = [0.9, 1.6, 2.6];
/// Ambient squiggle spawns per second at Caustics = 1 — the idle "pond
/// life" linework of the sketch/glossy styles.
const RING_AMBIENT_RATE: f32 = 9.0;
/// Vertices per vertex-pulled ring: one quad.
const RING_VERTS: u32 = 6;

pub fn plugin(app: &mut App) {
    // Shaders only exist in the main world; hand the handles into the
    // render app.
    let mut shaders = app.world_mut().resource_mut::<Assets<Shader>>();
    let mut add = |source: &'static str, name: &'static str| {
        shaders.add(Shader::from_wgsl(source, format!("{}#{name}", file!())))
    };
    let bubbles_natural = add(BUBBLE_SHADER, "bubbles");
    let shader = WaterShader {
        water: StyleShaders {
            natural: add(WATER_SHADER, "water"),
            sketch: add(SKETCH_WATER_SHADER, "sketch_water"),
            glossy: add(GLOSSY_WATER_SHADER, "glossy_water"),
        },
        surface: StyleShaders {
            natural: add(SURFACE_SHADER, "surface"),
            sketch: add(SKETCH_SURFACE_SHADER, "sketch_surface"),
            glossy: add(GLOSSY_SURFACE_SHADER, "glossy_surface"),
        },
        bubbles: StyleShaders {
            natural: bubbles_natural.clone(),
            sketch: add(SKETCH_BUBBLE_SHADER, "sketch_bubbles"),
            glossy: bubbles_natural,
        },
        rings: add(RING_SHADER, "rings"),
    };

    app.init_resource::<RippleGrid>()
        .init_resource::<RippleRings>()
        .init_resource::<WaterClock>()
        .add_systems(
            Update,
            (
                // After the sim so wakes follow this frame's heads. Not in
                // Options: the pond freezes behind the popup with the fish.
                step_ripples
                    .after(FishSimSet)
                    .run_if(experiment_active(ExperimentId::Fish))
                    .run_if(sim_active),
                // Ungated: must fire on the frame fish stops being current.
                clear_when_inactive,
            ),
        );

    app.sub_app_mut(RenderApp)
        .insert_resource(shader)
        .add_render_command::<Transparent2d, DrawWater>()
        .add_render_command::<Transparent2d, DrawBubbles>()
        .add_render_command::<Transparent2d, DrawRings>()
        .init_resource::<SpecializedRenderPipelines<WaterPipeline>>()
        .init_resource::<SpecializedRenderPipelines<SurfacePipeline>>()
        .init_resource::<SpecializedRenderPipelines<BubblePipeline>>()
        .init_resource::<SpecializedRenderPipelines<RingPipeline>>()
        .add_systems(
            RenderStartup,
            (
                init_water_pipelines.after(init_mesh_2d_pipeline),
                |mut commands: Commands| {
                    commands.init_resource::<WaterBuffers>();
                },
            ),
        )
        .add_systems(ExtractSchedule, extract_water)
        .add_systems(
            Render,
            (
                prepare_water.in_set(RenderSystems::PrepareResources),
                queue_water.in_set(RenderSystems::Queue),
            ),
        );
}

/// The water's animation time. Advanced only while the sim runs, so the
/// caustics, sparkle and bubbles freeze behind the options popup exactly
/// like the fish do. Wrapped to keep f32 trig arguments precise.
#[derive(Resource, Default)]
struct WaterClock(f32);

// ---------------------------------------------------------------------------
// The surface: a damped wave equation on a coarse grid, in the sim's
// window coordinates. Node (c, r) sits at window (c·CELL, r·CELL).

/// The CPU surface-height grid (double-buffered for the wave equation).
#[derive(Resource, Default)]
pub struct RippleGrid {
    cols: usize,
    rows: usize,
    curr: Vec<f32>,
    prev: Vec<f32>,
    /// Fixed-step accumulator.
    acc: f32,
}

impl RippleGrid {
    /// Size the grid to the window; any resize restarts the surface calm.
    fn resize_for(&mut self, bounds: Vec2) {
        let cols = (bounds.x / RIPPLE_CELL).ceil() as usize + 1;
        let rows = (bounds.y / RIPPLE_CELL).ceil() as usize + 1;
        if cols == self.cols && rows == self.rows {
            return;
        }
        self.cols = cols;
        self.rows = rows;
        self.curr.clear();
        self.curr.resize(cols * rows, 0.0);
        self.prev.clear();
        self.prev.resize(cols * rows, 0.0);
        self.acc = 0.0;
    }

    /// Flatten the surface (still sized).
    fn calm(&mut self) {
        self.curr.fill(0.0);
        self.prev.fill(0.0);
        self.acc = 0.0;
    }

    /// Deposit a disturbance, bilinearly split over the four surrounding
    /// nodes and clamped so pile-ups saturate instead of exploding.
    fn splat(&mut self, pos: Vec2, amp: f32) {
        if self.cols < 2 || self.rows < 2 {
            return;
        }
        let gx = (pos.x / RIPPLE_CELL).clamp(0.0, (self.cols - 1) as f32);
        let gy = (pos.y / RIPPLE_CELL).clamp(0.0, (self.rows - 1) as f32);
        let c0 = gx as usize;
        let r0 = gy as usize;
        let c1 = (c0 + 1).min(self.cols - 1);
        let r1 = (r0 + 1).min(self.rows - 1);
        let fx = gx - c0 as f32;
        let fy = gy - r0 as f32;
        for (index, w) in [
            (r0 * self.cols + c0, (1.0 - fx) * (1.0 - fy)),
            (r0 * self.cols + c1, fx * (1.0 - fy)),
            (r1 * self.cols + c0, (1.0 - fx) * fy),
            (r1 * self.cols + c1, fx * fy),
        ] {
            self.curr[index] = (self.curr[index] + amp * w).clamp(-SPLAT_CLAMP, SPLAT_CLAMP);
        }
    }

    /// One fixed step of the damped wave equation
    /// (u′ = 2u − u₋₁ + c²·∇²u, then damp). Edges clamp (zero-gradient),
    /// so rings reflect off the pond walls. Writes the new field over
    /// `prev` in place, then swaps — no third buffer.
    fn step(&mut self) {
        let (cols, rows) = (self.cols, self.rows);
        if cols < 2 || rows < 2 {
            return;
        }
        let curr = &self.curr;
        let prev = &mut self.prev;
        for r in 0..rows {
            let up = r.saturating_sub(1) * cols;
            let down = (r + 1).min(rows - 1) * cols;
            let row = r * cols;
            for c in 0..cols {
                let left = row + c.saturating_sub(1);
                let right = row + (c + 1).min(cols - 1);
                let centre = curr[row + c];
                let lap = curr[up + c] + curr[down + c] + curr[left] + curr[right] - 4.0 * centre;
                prev[row + c] = (2.0 * centre - prev[row + c] + WAVE_C2 * lap) * WAVE_DAMP;
            }
        }
        std::mem::swap(&mut self.curr, &mut self.prev);
    }
}

// ---------------------------------------------------------------------------
// Ring events: the sketch/glossy styles' ripples. A fixed pool of short
// vector events — each one becomes a hand-drawn (or glossy) stroked
// circle expanding from where something disturbed the pond.

/// One ripple ring. `kind` indexes [`RING_LIFE`] and picks the shader's
/// shaping: 0 = wake, 1 = plop, 2 = ambient squiggle.
#[derive(Clone, Copy)]
struct RingEvent {
    pos: Vec2,
    /// WaterClock time of birth — may sit slightly in the future (plops
    /// stagger their concentric rings); the shader sleeps until then.
    born: f32,
    amp: f32,
    seed: u32,
    kind: u32,
}

/// The live-ring pool. Grows to [`RING_MAX`] then recycles round-robin —
/// but only steals slots from rings past their prime, so a 4096-fish
/// churn saturates into steady ring chatter instead of strobing.
#[derive(Resource, Default)]
pub struct RippleRings {
    rings: Vec<RingEvent>,
    cursor: usize,
    /// Fractional ambient-spawn budget.
    ambient_acc: f32,
    /// Seed counter.
    counter: u32,
}

impl RippleRings {
    fn life(kind: u32) -> f32 {
        RING_LIFE[(kind as usize).min(RING_LIFE.len() - 1)]
    }

    fn spawn(&mut self, now: f32, pos: Vec2, amp: f32, kind: u32, delay: f32) {
        self.counter = self.counter.wrapping_add(1);
        let ring = RingEvent {
            pos,
            born: now + delay,
            amp,
            seed: self.counter.wrapping_mul(2654435761),
            kind,
        };
        if self.rings.len() < RING_MAX {
            self.rings.push(ring);
            return;
        }
        let slot = self.cursor % RING_MAX;
        self.cursor = self.cursor.wrapping_add(1);
        if now - self.rings[slot].born > 0.45 * Self::life(self.rings[slot].kind) {
            self.rings[slot] = ring;
        }
    }

    fn clear(&mut self) {
        self.rings.clear();
        self.cursor = 0;
        self.ambient_acc = 0.0;
    }
}

/// Frame-to-frame memory for the disturbance sources: last head positions
/// (wake deltas), per-fish travel accumulators (wake-ring spawning), the
/// last food state (plops), and whether the wave grid was live (to calm
/// it once on style/toggle changes instead of every frame).
#[derive(Default)]
struct RippleMemo {
    heads: Vec<Vec2>,
    travel: Vec<f32>,
    food: Option<Vec2>,
    eaten: u32,
    grid_was_live: bool,
}

/// Advance the water clock and feed this frame's disturbances into the
/// style's medium: the natural style splats the wave grid; sketch/glossy
/// spawn vector ring events (plus their ambient squiggles).
#[allow(clippy::too_many_arguments)]
fn step_ripples(
    time: Res<Time>,
    settings: Res<FishSettings>,
    fishes: Res<Fishes>,
    game: Res<FishGame>,
    bounds: Res<SimBounds>,
    mut grid: ResMut<RippleGrid>,
    mut rings: ResMut<RippleRings>,
    mut clock: ResMut<WaterClock>,
    mut memo: Local<RippleMemo>,
) {
    // One deref so the heads/travel zip below can borrow fields disjointly.
    let memo = &mut *memo;
    let dt = time.delta_secs();
    let before = clock.0;
    clock.0 = (clock.0 + dt) % 3600.0;
    let now = clock.0;
    // The hourly clock wrap would strand every live ring at a far-future
    // birth time (unkillable, unstealable); restart the pool instead.
    if now < before {
        rings.clear();
    }

    let natural = settings.style == WaterStyle::Natural;
    let disturb = settings.water && settings.ripples;
    // Each style's ripple medium lives only while it is the one shown:
    // the wave grid calms once when it stops feeding the natural surface
    // (style flips return to a fresh pond, not a stale one), and the ring
    // pool empties whenever the sketch/glossy styles aren't current.
    let grid_live = disturb && natural;
    if !grid_live && memo.grid_was_live {
        grid.calm();
    }
    memo.grid_was_live = grid_live;
    if grid_live {
        grid.resize_for(bounds.0);
    }
    if (natural || !settings.water) && !rings.rings.is_empty() {
        rings.clear();
    }

    if disturb {
        // Fish wakes. Natural: each head's travel this frame deposits
        // into the grid where it now is (dt-scaled, so wake energy is
        // frame-rate independent). Sketch/glossy: travel accumulates and
        // every RING_WAKE_EVERY px drops a small ring. Teleports
        // (restart, count changes) deposit nothing.
        let count = fishes.0.len();
        if memo.heads.len() != count {
            memo.heads.clear();
            memo.heads.extend(fishes.0.iter().map(|fish| fish.head()));
            memo.travel.clear();
            memo.travel.resize(count, 0.0);
        }
        for (fish, (prev, travel)) in fishes
            .0
            .iter()
            .zip(memo.heads.iter_mut().zip(memo.travel.iter_mut()))
        {
            let head = fish.head();
            let delta = head.distance(*prev);
            if delta > 1e-3 && delta < WAKE_TELEPORT {
                let size = (fish.scale * WAKE_SCALE_GAIN).clamp(0.2, 2.0);
                if natural {
                    let amp = settings.ripple_strength
                        * (delta * WAKE_AMP_PER_PX).min(WAKE_AMP_MAX)
                        * size;
                    grid.splat(head, amp);
                } else {
                    *travel += delta;
                    if *travel >= RING_WAKE_EVERY {
                        *travel = 0.0;
                        rings.spawn(now, head, settings.ripple_strength * size, 0, 0.0);
                    }
                }
            }
            *prev = head;
        }

        // Food plops: a big one where the dot was eaten, a small one
        // where the fresh dot lands. First frame only records (no
        // startup splash). A sketch/glossy eat-plop is the reference
        // look: three concentric rings, staggered.
        if let Some(last) = memo.food {
            if game.eaten != memo.eaten {
                if natural {
                    grid.splat(last, settings.ripple_strength * PLOP_EAT_AMP);
                } else {
                    for (i, delay) in [0.0, 0.18, 0.36].into_iter().enumerate() {
                        let amp = settings.ripple_strength * (1.0 - 0.2 * i as f32);
                        rings.spawn(now, last, amp, 1, delay);
                    }
                }
            }
            if game.food != last {
                if natural {
                    grid.splat(game.food, settings.ripple_strength * PLOP_DROP_AMP);
                } else {
                    rings.spawn(now, game.food, settings.ripple_strength * 0.5, 1, 0.0);
                }
            }
        }
        memo.food = Some(game.food);
        memo.eaten = game.eaten;
    } else {
        memo.heads.clear();
        memo.travel.clear();
        memo.food = None;
    }

    // Ambient squiggles — the sketch/glossy styles' idle pond life,
    // independent of the Ripples toggle (they are light play, not
    // disturbances). The Caustics dial is the density.
    if settings.water && !natural && settings.caustics > 0.0 {
        let mut rng = rand::rng();
        rings.ambient_acc += dt * RING_AMBIENT_RATE * settings.caustics;
        while rings.ambient_acc >= 1.0 {
            rings.ambient_acc -= 1.0;
            let pos = Vec2::new(
                rng.random::<f32>() * bounds.0.x,
                rng.random::<f32>() * bounds.0.y,
            );
            rings.spawn(now, pos, 1.0, 2, 0.0);
        }
    }

    if grid_live {
        grid.acc = (grid.acc + dt).min(WAVE_STEP * WAVE_MAX_STEPS);
        while grid.acc >= WAVE_STEP {
            grid.acc -= WAVE_STEP;
            grid.step();
        }
    }
}

/// Calm the pond when another experiment takes over; returning starts
/// fresh. (The draw itself is gated per frame in extract, so nothing can
/// draw stale over the next experiment.)
fn clear_when_inactive(
    current: Res<CurrentExperiment>,
    mut grid: ResMut<RippleGrid>,
    mut rings: ResMut<RippleRings>,
) {
    if !current.is_changed() || current.0 == ExperimentId::Fish {
        return;
    }
    grid.calm();
    rings.clear();
}

// ---------------------------------------------------------------------------
// Render world: the bed quad's uniform + height storage buffer, the
// bubbles' uniform, two pipelines over the standard 2D view uniform.

/// One layer's shader per style. Glossy bubbles alias the natural ones —
/// the soft additive look already fits the storybook style.
#[derive(Clone)]
struct StyleShaders {
    natural: Handle<Shader>,
    sketch: Handle<Shader>,
    glossy: Handle<Shader>,
}

impl StyleShaders {
    fn pick(&self, style: WaterStyle) -> &Handle<Shader> {
        match style {
            WaterStyle::Natural => &self.natural,
            WaterStyle::Sketch => &self.sketch,
            WaterStyle::Glossy => &self.glossy,
        }
    }
}

/// Resource holding the shader handles for the pipelines to take.
#[derive(Resource)]
struct WaterShader {
    water: StyleShaders,
    surface: StyleShaders,
    bubbles: StyleShaders,
    rings: Handle<Shader>,
}

/// The bed/surface shader's uniforms. Mirrors the WGSL `WaterParams`.
#[derive(Clone, Copy, Default, ShaderType)]
struct WaterParams {
    /// Window center, for the window→world y-flip.
    origin: Vec2,
    /// Window size — the quad the vertex shader spans.
    bounds: Vec2,
    time: f32,
    /// The Caustics dial (0..1).
    caustics: f32,
    /// The Sparkle dial (0..1).
    sparkle: f32,
    /// Surface grid cell size in px (1.0 when the grid is the 1-cell calm
    /// placeholder).
    cell: f32,
    cols: u32,
    rows: u32,
}

/// The bubble shader's uniforms. Mirrors the WGSL `BubbleParams`.
#[derive(Clone, Copy, Default, ShaderType)]
struct BubbleParams {
    origin: Vec2,
    bounds: Vec2,
    time: f32,
}

/// The ring shader's uniforms. Mirrors the WGSL `RingParams`. One shader
/// serves both ring looks; `style` (1 = sketch, 2 = glossy) branches the
/// fragment — uniform-coherent, so it costs nothing.
#[derive(Clone, Copy, Default, ShaderType)]
struct RingParams {
    origin: Vec2,
    bounds: Vec2,
    time: f32,
    style: u32,
}

/// One live ring on the GPU. Mirrors the WGSL `Ring` (24-byte stride).
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuRing {
    pos: [f32; 2],
    born: f32,
    amp: f32,
    seed: u32,
    kind: u32,
}

/// GPU buffers for all the water layers.
#[derive(Resource)]
struct WaterBuffers {
    /// The surface heights, row-major — or a single calm node when the
    /// ripples are off (every bilinear read then lands on a flat 0).
    heights: RawBufferVec<f32>,
    params: UniformBuffer<WaterParams>,
    bubble_params: UniformBuffer<BubbleParams>,
    /// The live rings (sketch/glossy styles only).
    rings: RawBufferVec<GpuRing>,
    ring_params: UniformBuffer<RingParams>,
    water_on: bool,
    bubbles_on: bool,
    rings_on: bool,
    /// The popup's Style button — picks each layer's pipeline variant.
    style: WaterStyle,
    bubble_verts: u32,
    ring_verts: u32,
    water_bind_group: Option<BindGroup>,
    bubble_bind_group: Option<BindGroup>,
    ring_bind_group: Option<BindGroup>,
}

impl Default for WaterBuffers {
    fn default() -> Self {
        Self {
            heights: RawBufferVec::new(BufferUsages::STORAGE),
            params: UniformBuffer::default(),
            bubble_params: UniformBuffer::default(),
            rings: RawBufferVec::new(BufferUsages::STORAGE),
            ring_params: UniformBuffer::default(),
            water_on: false,
            bubbles_on: false,
            rings_on: false,
            style: WaterStyle::Natural,
            bubble_verts: 0,
            ring_verts: 0,
            water_bind_group: None,
            bubble_bind_group: None,
            ring_bind_group: None,
        }
    }
}

/// Copy this frame's water state into the render world. Reads the main
/// world's resources directly — the only per-frame payload is the height
/// grid. (`Option`: the buffers resource is created in `RenderStartup`,
/// which may not have run yet.)
#[allow(clippy::too_many_arguments)]
fn extract_water(
    current: Extract<Res<CurrentExperiment>>,
    settings: Extract<Res<FishSettings>>,
    clock: Extract<Res<WaterClock>>,
    grid: Extract<Res<RippleGrid>>,
    ring_events: Extract<Res<RippleRings>>,
    bounds: Extract<Res<SimBounds>>,
    buffers: Option<ResMut<WaterBuffers>>,
) {
    let Some(mut buffers) = buffers else { return };
    buffers.water_on = current.0 == ExperimentId::Fish && settings.water;
    buffers.style = settings.style;
    if !buffers.water_on {
        buffers.bubbles_on = false;
        buffers.rings_on = false;
        return;
    }
    let origin = bounds.0 / 2.0;
    let natural = settings.style == WaterStyle::Natural;

    let ripples_live = natural && settings.ripples && grid.cols >= 2 && !grid.curr.is_empty();
    let heights = buffers.heights.values_mut();
    heights.clear();
    if ripples_live {
        heights.extend_from_slice(&grid.curr);
    } else {
        heights.push(0.0);
    }
    let (cols, rows, cell) = if ripples_live {
        (grid.cols as u32, grid.rows as u32, RIPPLE_CELL)
    } else {
        (1, 1, 1.0)
    };
    buffers.params.set(WaterParams {
        origin,
        bounds: bounds.0,
        time: clock.0,
        caustics: settings.caustics,
        sparkle: settings.sparkle,
        cell,
        cols,
        rows,
    });

    // The live rings (sketch/glossy): upload only the ones still inside
    // (or waiting on) their lifetime — at most RING_MAX * 24 B.
    let now = clock.0;
    let rings = buffers.rings.values_mut();
    rings.clear();
    if !natural {
        for ring in &ring_events.rings {
            let age = now - ring.born;
            if age < RippleRings::life(ring.kind) {
                rings.push(GpuRing {
                    pos: ring.pos.to_array(),
                    born: ring.born,
                    amp: ring.amp,
                    seed: ring.seed,
                    kind: ring.kind,
                });
            }
        }
    }
    let ring_count = buffers.rings.len() as u32;
    buffers.rings_on = ring_count > 0;
    buffers.ring_verts = ring_count * RING_VERTS;
    buffers.ring_params.set(RingParams {
        origin,
        bounds: bounds.0,
        time: clock.0,
        style: settings.style as u32,
    });

    buffers.bubbles_on = settings.bubbles && settings.bubble_count >= 1.0;
    buffers.bubble_verts = settings.bubble_count.round() as u32 * BUBBLE_VERTS;
    buffers.bubble_params.set(BubbleParams {
        origin,
        bounds: bounds.0,
        time: clock.0,
    });
}

/// Upload and (re)build the bind groups — every frame, like the flow's:
/// a `RawBufferVec` reallocation invalidates the old one.
#[allow(clippy::too_many_arguments)]
fn prepare_water(
    mut buffers: ResMut<WaterBuffers>,
    water_pipeline: Option<Res<WaterPipeline>>,
    bubble_pipeline: Option<Res<BubblePipeline>>,
    ring_pipeline: Option<Res<RingPipeline>>,
    pipeline_cache: Res<PipelineCache>,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
) {
    buffers.water_bind_group = None;
    buffers.bubble_bind_group = None;
    buffers.ring_bind_group = None;
    if !buffers.water_on {
        return;
    }
    buffers.heights.write_buffer(&render_device, &render_queue);
    buffers.params.write_buffer(&render_device, &render_queue);
    if let Some(pipeline) = water_pipeline
        && let Some(heights) = buffers.heights.buffer()
    {
        buffers.water_bind_group = Some(render_device.create_bind_group(
            "fish_water",
            &pipeline_cache.get_bind_group_layout(&pipeline.layout),
            &BindGroupEntries::sequential((&buffers.params, heights.as_entire_binding())),
        ));
    }
    if buffers.bubbles_on
        && let Some(pipeline) = bubble_pipeline
    {
        buffers
            .bubble_params
            .write_buffer(&render_device, &render_queue);
        buffers.bubble_bind_group = Some(render_device.create_bind_group(
            "fish_bubbles",
            &pipeline_cache.get_bind_group_layout(&pipeline.layout),
            &BindGroupEntries::sequential((&buffers.bubble_params,)),
        ));
    }
    if buffers.rings_on
        && let Some(pipeline) = ring_pipeline
    {
        buffers.rings.write_buffer(&render_device, &render_queue);
        buffers
            .ring_params
            .write_buffer(&render_device, &render_queue);
        if let Some(rings) = buffers.rings.buffer() {
            buffers.ring_bind_group = Some(render_device.create_bind_group(
                "fish_rings",
                &pipeline_cache.get_bind_group_layout(&pipeline.layout),
                &BindGroupEntries::sequential((&buffers.ring_params, rings.as_entire_binding())),
            ));
        }
    }
}

/// The bed/surface pipeline: one window-covering quad from `vertex_index`;
/// the fragment shader does everything. Group 1: uniform + height grid.
/// Each pipeline carries every style's shader; the specialize key's style
/// picks one (a style's variant only ever compiles if it gets used).
#[derive(Resource)]
struct WaterPipeline {
    mesh2d_pipeline: Mesh2dPipeline,
    shaders: StyleShaders,
    layout: BindGroupLayoutDescriptor,
}

/// The surface pipeline — the same quad and bind group as the bed, a
/// different fragment: the translucent layer drawn over the fish.
#[derive(Resource)]
struct SurfacePipeline {
    mesh2d_pipeline: Mesh2dPipeline,
    shaders: StyleShaders,
    /// A clone of the bed pipeline's descriptor: the pipeline cache
    /// dedupes identical descriptors, so the bed's bind group fits both
    /// (and every style — the descriptor doesn't mention the shader).
    layout: BindGroupLayoutDescriptor,
}

/// The bubble pipeline: vertex-pull from `vertex_index` alone. Group 1:
/// the uniform.
#[derive(Resource)]
struct BubblePipeline {
    mesh2d_pipeline: Mesh2dPipeline,
    shaders: StyleShaders,
    layout: BindGroupLayoutDescriptor,
}

/// The ring pipeline (sketch/glossy): vertex-pull, one quad per live
/// ring. Group 1: uniform + ring storage. One shader for both looks —
/// the uniform's `style` branches the fragment.
#[derive(Resource)]
struct RingPipeline {
    mesh2d_pipeline: Mesh2dPipeline,
    shader: Handle<Shader>,
    layout: BindGroupLayoutDescriptor,
}

fn init_water_pipelines(
    mut commands: Commands,
    mesh2d_pipeline: Res<Mesh2dPipeline>,
    shader: Res<WaterShader>,
) {
    let water_layout = BindGroupLayoutDescriptor::new(
        "fish_water_layout",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::VERTEX_FRAGMENT,
            (
                uniform_buffer::<WaterParams>(false),
                storage_buffer_read_only_sized(false, None), // heights
            ),
        ),
    );
    commands.insert_resource(WaterPipeline {
        mesh2d_pipeline: mesh2d_pipeline.clone(),
        shaders: shader.water.clone(),
        layout: water_layout.clone(),
    });
    commands.insert_resource(SurfacePipeline {
        mesh2d_pipeline: mesh2d_pipeline.clone(),
        shaders: shader.surface.clone(),
        layout: water_layout,
    });
    let bubble_layout = BindGroupLayoutDescriptor::new(
        "fish_bubble_layout",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::VERTEX_FRAGMENT,
            (uniform_buffer::<BubbleParams>(false),),
        ),
    );
    commands.insert_resource(BubblePipeline {
        mesh2d_pipeline: mesh2d_pipeline.clone(),
        shaders: shader.bubbles.clone(),
        layout: bubble_layout,
    });
    let ring_layout = BindGroupLayoutDescriptor::new(
        "fish_ring_layout",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::VERTEX_FRAGMENT,
            (
                uniform_buffer::<RingParams>(false),
                storage_buffer_read_only_sized(false, None), // rings
            ),
        ),
    );
    commands.insert_resource(RingPipeline {
        mesh2d_pipeline: mesh2d_pipeline.clone(),
        shader: shader.rings.clone(),
        layout: ring_layout,
    });
}

/// Shared pipeline descriptor shape for both water pipelines (no vertex
/// buffers, premultiplied blending, the fish experiment's depth settings).
fn water_pipeline_descriptor(
    label: &'static str,
    shader: &Handle<Shader>,
    view_layout: &BindGroupLayoutDescriptor,
    layout: &BindGroupLayoutDescriptor,
    key: Mesh2dPipelineKey,
) -> RenderPipelineDescriptor {
    let format = match key.contains(Mesh2dPipelineKey::HDR) {
        true => ViewTarget::TEXTURE_FORMAT_HDR,
        false => TextureFormat::bevy_default(),
    };
    RenderPipelineDescriptor {
        label: Some(label.into()),
        vertex: VertexState {
            shader: shader.clone(),
            // Vertex-pull: everything comes from vertex_index + group 1.
            buffers: vec![],
            ..default()
        },
        fragment: Some(FragmentState {
            shader: shader.clone(),
            targets: vec![Some(ColorTargetState {
                format,
                // Premultiplied: the bed writes alpha 1 (opaque base
                // layer), the bubbles alpha 0 (pure additive light).
                blend: Some(BlendState::PREMULTIPLIED_ALPHA_BLENDING),
                write_mask: ColorWrites::ALL,
            })],
            ..default()
        }),
        layout: vec![view_layout.clone(), layout.clone()],
        primitive: PrimitiveState {
            topology: PrimitiveTopology::TriangleList,
            ..default()
        },
        depth_stencil: Some(DepthStencilState {
            format: CORE_2D_DEPTH_FORMAT,
            // Blended layers: no depth interaction, pure paint order.
            depth_write_enabled: false,
            depth_compare: CompareFunction::Always,
            stencil: StencilState {
                front: StencilFaceState::IGNORE,
                back: StencilFaceState::IGNORE,
                read_mask: 0,
                write_mask: 0,
            },
            bias: DepthBiasState {
                constant: 0,
                slope_scale: 0.0,
                clamp: 0.0,
            },
        }),
        multisample: MultisampleState {
            count: key.msaa_samples(),
            mask: !0,
            alpha_to_coverage_enabled: false,
        },
        ..default()
    }
}

// The keys carry the style: `(mesh key, style)`. Natural reproduces the
// original descriptors exactly; the others swap in their fragment program.

impl SpecializedRenderPipeline for WaterPipeline {
    type Key = (Mesh2dPipelineKey, WaterStyle);

    fn specialize(&self, (key, style): Self::Key) -> RenderPipelineDescriptor {
        water_pipeline_descriptor(
            match style {
                WaterStyle::Natural => "fish_water_pipeline",
                WaterStyle::Sketch => "fish_water_sketch_pipeline",
                WaterStyle::Glossy => "fish_water_glossy_pipeline",
            },
            self.shaders.pick(style),
            &self.mesh2d_pipeline.view_layout,
            &self.layout,
            key,
        )
    }
}

impl SpecializedRenderPipeline for SurfacePipeline {
    type Key = (Mesh2dPipelineKey, WaterStyle);

    fn specialize(&self, (key, style): Self::Key) -> RenderPipelineDescriptor {
        water_pipeline_descriptor(
            match style {
                WaterStyle::Natural => "fish_water_surface_pipeline",
                WaterStyle::Sketch => "fish_water_surface_sketch_pipeline",
                WaterStyle::Glossy => "fish_water_surface_glossy_pipeline",
            },
            self.shaders.pick(style),
            &self.mesh2d_pipeline.view_layout,
            &self.layout,
            key,
        )
    }
}

impl SpecializedRenderPipeline for BubblePipeline {
    type Key = (Mesh2dPipelineKey, WaterStyle);

    fn specialize(&self, (key, style): Self::Key) -> RenderPipelineDescriptor {
        water_pipeline_descriptor(
            match style {
                WaterStyle::Natural => "fish_bubble_pipeline",
                WaterStyle::Sketch => "fish_bubble_sketch_pipeline",
                WaterStyle::Glossy => "fish_bubble_glossy_pipeline",
            },
            self.shaders.pick(style),
            &self.mesh2d_pipeline.view_layout,
            &self.layout,
            key,
        )
    }
}

impl SpecializedRenderPipeline for RingPipeline {
    type Key = Mesh2dPipelineKey;

    fn specialize(&self, key: Self::Key) -> RenderPipelineDescriptor {
        water_pipeline_descriptor(
            "fish_ring_pipeline",
            &self.shader,
            &self.mesh2d_pipeline.view_layout,
            &self.layout,
            key,
        )
    }
}

/// Draws the bed/surface quad.
struct DrawWaterQuad;

impl<P: PhaseItem> RenderCommand<P> for DrawWaterQuad {
    type Param = SRes<WaterBuffers>;
    type ViewQuery = ();
    type ItemQuery = ();

    fn render<'w>(
        _: &P,
        _: ROQueryItem<'w, '_, Self::ViewQuery>,
        _: Option<ROQueryItem<'w, '_, Self::ItemQuery>>,
        buffers: SystemParamItem<'w, '_, Self::Param>,
        pass: &mut TrackedRenderPass<'w>,
    ) -> RenderCommandResult {
        let buffers = buffers.into_inner();
        if !buffers.water_on {
            return RenderCommandResult::Success;
        }
        let Some(bind_group) = &buffers.water_bind_group else {
            return RenderCommandResult::Failure("water bind group not prepared");
        };
        pass.set_bind_group(1, bind_group, &[]);
        pass.draw(0..6, 0..1);
        RenderCommandResult::Success
    }
}

/// Draws every bubble from `vertex_index` alone.
struct DrawBubbleQuads;

impl<P: PhaseItem> RenderCommand<P> for DrawBubbleQuads {
    type Param = SRes<WaterBuffers>;
    type ViewQuery = ();
    type ItemQuery = ();

    fn render<'w>(
        _: &P,
        _: ROQueryItem<'w, '_, Self::ViewQuery>,
        _: Option<ROQueryItem<'w, '_, Self::ItemQuery>>,
        buffers: SystemParamItem<'w, '_, Self::Param>,
        pass: &mut TrackedRenderPass<'w>,
    ) -> RenderCommandResult {
        let buffers = buffers.into_inner();
        if !buffers.bubbles_on || buffers.bubble_verts == 0 {
            return RenderCommandResult::Success;
        }
        let Some(bind_group) = &buffers.bubble_bind_group else {
            return RenderCommandResult::Failure("bubble bind group not prepared");
        };
        pass.set_bind_group(1, bind_group, &[]);
        pass.draw(0..buffers.bubble_verts, 0..1);
        RenderCommandResult::Success
    }
}

/// Draws every live ring from `vertex_index` alone.
struct DrawRingQuads;

impl<P: PhaseItem> RenderCommand<P> for DrawRingQuads {
    type Param = SRes<WaterBuffers>;
    type ViewQuery = ();
    type ItemQuery = ();

    fn render<'w>(
        _: &P,
        _: ROQueryItem<'w, '_, Self::ViewQuery>,
        _: Option<ROQueryItem<'w, '_, Self::ItemQuery>>,
        buffers: SystemParamItem<'w, '_, Self::Param>,
        pass: &mut TrackedRenderPass<'w>,
    ) -> RenderCommandResult {
        let buffers = buffers.into_inner();
        if !buffers.rings_on || buffers.ring_verts == 0 {
            return RenderCommandResult::Success;
        }
        let Some(bind_group) = &buffers.ring_bind_group else {
            return RenderCommandResult::Failure("ring bind group not prepared");
        };
        pass.set_bind_group(1, bind_group, &[]);
        pass.draw(0..buffers.ring_verts, 0..1);
        RenderCommandResult::Success
    }
}

type DrawWater = (SetItemPipeline, SetMesh2dViewBindGroup<0>, DrawWaterQuad);

type DrawBubbles = (SetItemPipeline, SetMesh2dViewBindGroup<0>, DrawBubbleQuads);

type DrawRings = (SetItemPipeline, SetMesh2dViewBindGroup<0>, DrawRingQuads);

/// Queue the water draws into every 2D view: the bed under the fish (the
/// fish queues at 0.0), the bubbles over them, and the translucent
/// surface on top of everything — fish and bubbles live IN the water.
/// The surface item reuses the bed's draw function (same quad, same bind
/// group); only the pipeline differs.
#[allow(clippy::too_many_arguments)]
fn queue_water(
    transparent_draw_functions: Res<DrawFunctions<Transparent2d>>,
    water_pipeline: Option<Res<WaterPipeline>>,
    surface_pipeline: Option<Res<SurfacePipeline>>,
    bubble_pipeline: Option<Res<BubblePipeline>>,
    ring_pipeline: Option<Res<RingPipeline>>,
    mut water_pipelines: ResMut<SpecializedRenderPipelines<WaterPipeline>>,
    mut surface_pipelines: ResMut<SpecializedRenderPipelines<SurfacePipeline>>,
    mut bubble_pipelines: ResMut<SpecializedRenderPipelines<BubblePipeline>>,
    mut ring_pipelines: ResMut<SpecializedRenderPipelines<RingPipeline>>,
    pipeline_cache: Res<PipelineCache>,
    buffers: Option<Res<WaterBuffers>>,
    mut transparent_render_phases: ResMut<ViewSortedRenderPhases<Transparent2d>>,
    views: Query<(&ExtractedView, &Msaa)>,
) {
    let (
        Some(water_pipeline),
        Some(surface_pipeline),
        Some(bubble_pipeline),
        Some(ring_pipeline),
        Some(buffers),
    ) = (
        water_pipeline,
        surface_pipeline,
        bubble_pipeline,
        ring_pipeline,
        buffers,
    )
    else {
        return;
    };
    if !buffers.water_on {
        return;
    }
    let draw_water = transparent_draw_functions.read().id::<DrawWater>();
    let draw_bubbles = transparent_draw_functions.read().id::<DrawBubbles>();
    let draw_rings = transparent_draw_functions.read().id::<DrawRings>();

    for (view, msaa) in &views {
        let Some(transparent_phase) = transparent_render_phases.get_mut(&view.retained_view_entity)
        else {
            continue;
        };
        let mesh_key = Mesh2dPipelineKey::from_msaa_samples(msaa.samples())
            | Mesh2dPipelineKey::from_hdr(view.hdr)
            | Mesh2dPipelineKey::from_primitive_topology(PrimitiveTopology::TriangleList);
        let key = (mesh_key, buffers.style);

        let mut item = |draw_function, pipeline, sort_key| {
            transparent_phase.add(Transparent2d {
                // The draw is fully described by resources; no entity involved.
                entity: (Entity::PLACEHOLDER, MainEntity::from(Entity::PLACEHOLDER)),
                draw_function,
                pipeline,
                sort_key: FloatOrd(sort_key),
                batch_range: 0..1,
                extra_index: PhaseItemExtraIndex::None,
                extracted_index: usize::MAX,
                indexed: false,
            });
        };
        let pipeline_id = water_pipelines.specialize(&pipeline_cache, &water_pipeline, key);
        item(draw_water, pipeline_id, -1.0);
        if buffers.bubbles_on {
            let pipeline_id = bubble_pipelines.specialize(&pipeline_cache, &bubble_pipeline, key);
            item(draw_bubbles, pipeline_id, 0.5);
        }
        let pipeline_id = surface_pipelines.specialize(&pipeline_cache, &surface_pipeline, key);
        item(draw_water, pipeline_id, 0.75);
        // The rings ride above the surface wash — they are its marks.
        if buffers.rings_on {
            let pipeline_id = ring_pipelines.specialize(&pipeline_cache, &ring_pipeline, mesh_key);
            item(draw_rings, pipeline_id, 0.8);
        }
    }
}

// The bed/surface shader: one window quad; every pixel composes the pond
// bottom (depth patches + two animated Voronoi caustic webs) seen through
// the surface (wave-grid slope refraction + analytic micro-ripples), plus
// sun glints off the total slope. Colors are authored in sRGB and
// converted once at the end; output is premultiplied with alpha 1 (the
// opaque base layer under everything).
const WATER_SHADER: &str = r"
#import bevy_sprite::mesh2d_view_bindings::view

struct WaterParams {
    origin: vec2<f32>,
    bounds: vec2<f32>,
    time: f32,
    caustics: f32,
    sparkle: f32,
    cell: f32,
    cols: u32,
    rows: u32,
};

@group(1) @binding(0) var<uniform> params: WaterParams;
@group(1) @binding(1) var<storage, read> heights: array<f32>;

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    // Window-space position, interpolated per fragment.
    @location(0) window: vec2<f32>,
};

@vertex
fn vertex(@builtin(vertex_index) vi: u32) -> VertexOutput {
    // Two triangles covering the window.
    var corner: vec2<f32>;
    switch vi {
        case 0u: { corner = vec2<f32>(0.0, 0.0); }
        case 1u: { corner = vec2<f32>(1.0, 0.0); }
        case 2u: { corner = vec2<f32>(0.0, 1.0); }
        case 3u: { corner = vec2<f32>(1.0, 0.0); }
        case 4u: { corner = vec2<f32>(1.0, 1.0); }
        default: { corner = vec2<f32>(0.0, 1.0); }
    }
    let window = corner * params.bounds;
    let world = vec2<f32>(window.x - params.origin.x, params.origin.y - window.y);
    var out: VertexOutput;
    out.clip_position = view.clip_from_world * vec4<f32>(world, 0.0, 1.0);
    out.window = window;
    return out;
}

const TAU: f32 = 6.28318530718;
// The sun sits off toward the window's upper-left; surface slopes facing
// it glint. Window coords are y-down.
const SUN: vec2<f32> = vec2<f32>(-0.6, -0.8);

fn hash21(p: vec2<f32>) -> f32 {
    var q = fract(p * vec2<f32>(123.34, 345.45));
    q += dot(q, q + 34.345);
    return fract(q.x * q.y);
}

fn hash22(p: vec2<f32>) -> vec2<f32> {
    var p3 = fract(vec3<f32>(p.x, p.y, p.x) * vec3<f32>(0.1031, 0.1030, 0.0973));
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.xx + p3.yz) * p3.zy);
}

// Smooth value noise — the static depth patches of the pond bed.
fn vnoise(p: vec2<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let u = f * f * (3.0 - 2.0 * f);
    let a = hash21(i);
    let b = hash21(i + vec2<f32>(1.0, 0.0));
    let c = hash21(i + vec2<f32>(0.0, 1.0));
    let d = hash21(i + vec2<f32>(1.0, 1.0));
    return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}

// Animated Voronoi cell borders: the bright filament web of light a wavy
// surface focuses onto the bed. F2−F1 small = on a border.
fn caustic_web(uv: vec2<f32>, t: f32) -> f32 {
    let id = floor(uv);
    let f = fract(uv);
    var d1 = 8.0;
    var d2 = 8.0;
    for (var y = -1; y <= 1; y = y + 1) {
        for (var x = -1; x <= 1; x = x + 1) {
            let off = vec2<f32>(f32(x), f32(y));
            let h = hash22(id + off);
            let p = off + 0.5 + 0.4 * sin(t + TAU * h) - f;
            let d = dot(p, p);
            if (d < d1) {
                d2 = d1;
                d1 = d;
            } else if (d < d2) {
                d2 = d;
            }
        }
    }
    let border = sqrt(d2) - sqrt(d1);
    let w = 1.0 - smoothstep(0.02, 0.18, border);
    return w * w;
}

// Bilinear surface height at a window position. With the calm 1x1
// placeholder grid every read lands on heights[0] = 0.
fn height_at(p: vec2<f32>) -> f32 {
    let gx = clamp(p.x / params.cell, 0.0, f32(params.cols - 1u));
    let gy = clamp(p.y / params.cell, 0.0, f32(params.rows - 1u));
    let c0 = u32(gx);
    let r0 = u32(gy);
    let c1 = min(c0 + 1u, params.cols - 1u);
    let r1 = min(r0 + 1u, params.rows - 1u);
    let fx = gx - f32(c0);
    let fy = gy - f32(r0);
    let top = mix(heights[r0 * params.cols + c0], heights[r0 * params.cols + c1], fx);
    let bottom = mix(heights[r1 * params.cols + c0], heights[r1 * params.cols + c1], fx);
    return mix(top, bottom, fy);
}

// Analytic ambient micro-ripples: three directional waves' summed
// gradient. Keeps the surface sparkling when nothing disturbs it.
fn micro_slope(p: vec2<f32>, t: f32) -> vec2<f32> {
    let d1 = vec2<f32>(0.866, 0.5);
    let d2 = vec2<f32>(-0.259, 0.966);
    let d3 = vec2<f32>(-0.643, -0.766);
    var g = d1 * cos(dot(p, d1) * 0.20 + t * 2.1);
    g += d2 * cos(dot(p, d2) * 0.34 + t * 2.9);
    g += d3 * cos(dot(p, d3) * 0.51 - t * 3.7);
    return g;
}

// Exact sRGB -> linear (colors are authored in sRGB; the framebuffer
// encodes back on write).
fn srgb_to_linear(c: vec3<f32>) -> vec3<f32> {
    let lo = c / 12.92;
    let hi = pow((c + vec3<f32>(0.055)) / 1.055, vec3<f32>(2.4));
    return select(hi, lo, c <= vec3<f32>(0.04045));
}

@fragment
fn fragment(in: VertexOutput) -> @location(0) vec4<f32> {
    let p = in.window;
    let t = params.time;

    // Surface slope: the wave grid (forward differences of the bilinear
    // height) plus the ambient micro-ripples, scaled by the Sparkle dial.
    let h = height_at(p);
    let hx = height_at(p + vec2<f32>(params.cell, 0.0));
    let hy = height_at(p + vec2<f32>(0.0, params.cell));
    let rip_grad = vec2<f32>(hx - h, hy - h) / params.cell;
    let micro = micro_slope(p, t);

    // The bed, seen through the surface: ripples and micro-waves refract
    // the line of sight, so the bed samples shift with the slope.
    let bed_p = p + rip_grad * 40.0 + micro * 2.5;

    // Depth patches: large, static, darker where the pond is deeper. The
    // smoothstep stretches the noise's mid-heavy histogram into real
    // shallows and real deeps.
    let dn = 0.65 * vnoise(bed_p * 0.004 + 3.1) + 0.35 * vnoise(bed_p * 0.012 + 9.7);
    let depth = smoothstep(0.2, 0.8, dn);
    let bed = mix(vec3<f32>(0.012, 0.085, 0.115), vec3<f32>(0.07, 0.26, 0.27), depth);

    // Two caustic webs drifting at different scales and tempos; the fine
    // one modulates the coarse one so filaments braid instead of tile.
    let drift = vec2<f32>(t * 3.0, t * 1.9);
    let c1 = caustic_web((bed_p + drift) / 130.0, t * 0.55);
    let c2 = caustic_web((bed_p - drift) / 60.0 + vec2<f32>(13.7, 41.3), t * 0.8 + 2.0);
    let caustic = c1 * (0.4 + 0.6 * c2);

    // The bed is everything UNDER the fish; the glints, wave shading and
    // dapple live in the surface pass drawn above them (the fish swim in
    // the water, not on it).
    var col = bed;
    col += vec3<f32>(0.50, 0.80, 0.75) * caustic * params.caustics * (0.15 + 0.55 * depth);

    return vec4<f32>(srgb_to_linear(clamp(col, vec3<f32>(0.0), vec3<f32>(1.0))), 1.0);
}
";

// The surface pass — drawn ABOVE the fish (and the bubbles): a soft
// water-tint wash that pushes everything visually under the surface, the
// wave-height shading (crests brighten, troughs absorb — wakes pass over
// the fish), the sun glints, and a faint caustic dapple playing across
// the fish's backs. Shares the bed pass's uniform + height grid bind
// group; output is premultiplied translucent.
const SURFACE_SHADER: &str = r"
#import bevy_sprite::mesh2d_view_bindings::view

struct WaterParams {
    origin: vec2<f32>,
    bounds: vec2<f32>,
    time: f32,
    caustics: f32,
    sparkle: f32,
    cell: f32,
    cols: u32,
    rows: u32,
};

@group(1) @binding(0) var<uniform> params: WaterParams;
@group(1) @binding(1) var<storage, read> heights: array<f32>;

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) window: vec2<f32>,
};

@vertex
fn vertex(@builtin(vertex_index) vi: u32) -> VertexOutput {
    var corner: vec2<f32>;
    switch vi {
        case 0u: { corner = vec2<f32>(0.0, 0.0); }
        case 1u: { corner = vec2<f32>(1.0, 0.0); }
        case 2u: { corner = vec2<f32>(0.0, 1.0); }
        case 3u: { corner = vec2<f32>(1.0, 0.0); }
        case 4u: { corner = vec2<f32>(1.0, 1.0); }
        default: { corner = vec2<f32>(0.0, 1.0); }
    }
    let window = corner * params.bounds;
    let world = vec2<f32>(window.x - params.origin.x, params.origin.y - window.y);
    var out: VertexOutput;
    out.clip_position = view.clip_from_world * vec4<f32>(world, 0.0, 1.0);
    out.window = window;
    return out;
}

const TAU: f32 = 6.28318530718;
const SUN: vec2<f32> = vec2<f32>(-0.6, -0.8);

fn hash22(p: vec2<f32>) -> vec2<f32> {
    var p3 = fract(vec3<f32>(p.x, p.y, p.x) * vec3<f32>(0.1031, 0.1030, 0.0973));
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.xx + p3.yz) * p3.zy);
}

fn caustic_web(uv: vec2<f32>, t: f32) -> f32 {
    let id = floor(uv);
    let f = fract(uv);
    var d1 = 8.0;
    var d2 = 8.0;
    for (var y = -1; y <= 1; y = y + 1) {
        for (var x = -1; x <= 1; x = x + 1) {
            let off = vec2<f32>(f32(x), f32(y));
            let h = hash22(id + off);
            let p = off + 0.5 + 0.4 * sin(t + TAU * h) - f;
            let d = dot(p, p);
            if (d < d1) {
                d2 = d1;
                d1 = d;
            } else if (d < d2) {
                d2 = d;
            }
        }
    }
    let border = sqrt(d2) - sqrt(d1);
    let w = 1.0 - smoothstep(0.02, 0.18, border);
    return w * w;
}

fn height_at(p: vec2<f32>) -> f32 {
    let gx = clamp(p.x / params.cell, 0.0, f32(params.cols - 1u));
    let gy = clamp(p.y / params.cell, 0.0, f32(params.rows - 1u));
    let c0 = u32(gx);
    let r0 = u32(gy);
    let c1 = min(c0 + 1u, params.cols - 1u);
    let r1 = min(r0 + 1u, params.rows - 1u);
    let fx = gx - f32(c0);
    let fy = gy - f32(r0);
    let top = mix(heights[r0 * params.cols + c0], heights[r0 * params.cols + c1], fx);
    let bottom = mix(heights[r1 * params.cols + c0], heights[r1 * params.cols + c1], fx);
    return mix(top, bottom, fy);
}

fn micro_slope(p: vec2<f32>, t: f32) -> vec2<f32> {
    let d1 = vec2<f32>(0.866, 0.5);
    let d2 = vec2<f32>(-0.259, 0.966);
    let d3 = vec2<f32>(-0.643, -0.766);
    var g = d1 * cos(dot(p, d1) * 0.20 + t * 2.1);
    g += d2 * cos(dot(p, d2) * 0.34 + t * 2.9);
    g += d3 * cos(dot(p, d3) * 0.51 - t * 3.7);
    return g;
}

fn srgb_to_linear(c: vec3<f32>) -> vec3<f32> {
    let lo = c / 12.92;
    let hi = pow((c + vec3<f32>(0.055)) / 1.055, vec3<f32>(2.4));
    return select(hi, lo, c <= vec3<f32>(0.04045));
}

@fragment
fn fragment(in: VertexOutput) -> @location(0) vec4<f32> {
    let p = in.window;
    let t = params.time;

    let h = height_at(p);
    let hx = height_at(p + vec2<f32>(params.cell, 0.0));
    let hy = height_at(p + vec2<f32>(0.0, params.cell));
    let rip_grad = vec2<f32>(hx - h, hy - h) / params.cell;
    let micro = micro_slope(p, t);

    // Sun glints off the total slope — surface reflections, so they
    // belong above everything in the water.
    let slope = rip_grad * 30.0 + micro * (0.45 * params.sparkle);
    let facing = clamp(dot(slope, SUN), 0.0, 1.0);
    let glint = pow(facing, 6.0);

    // A faint caustic dapple on whatever swims below, refracted like the
    // bed's webs so the two layers stay in register.
    let drift = vec2<f32>(t * 3.0, t * 1.9);
    let dapple = caustic_web((p + rip_grad * 40.0 + drift) / 130.0, t * 0.55);

    // Wave shading: crests brighten, troughs deepen the wash — the rings
    // visibly travel over the fish.
    let shade = 0.30 * h;
    let alpha = clamp(0.16 + max(-shade, 0.0) * 0.5, 0.0, 0.8);
    var rgb = srgb_to_linear(vec3<f32>(0.05, 0.22, 0.26)) * alpha;
    rgb += srgb_to_linear(vec3<f32>(0.85, 1.0, 0.98)) * max(shade, 0.0);
    rgb += srgb_to_linear(vec3<f32>(0.50, 0.80, 0.75)) * dapple * params.caustics * 0.22;
    rgb += srgb_to_linear(vec3<f32>(1.0, 0.98, 0.9)) * glint;

    return vec4<f32>(rgb, alpha);
}
";

// The bubble shader: vertex-pull — bubble i's whole life is a pure
// function of (hash, time). Top-down, a bubble rises toward the camera:
// it fades in small, grows and sways on the way up, then pops into a
// brief expanding ring at the surface and reseeds somewhere else for the
// next cycle. One quad each; the fragment shapes rim, fill and a sun-side
// highlight from the interpolated profile coordinate. Premultiplied with
// alpha 0 — pure additive light over the pond.
const BUBBLE_SHADER: &str = r"
#import bevy_sprite::mesh2d_view_bindings::view

struct BubbleParams {
    origin: vec2<f32>,
    bounds: vec2<f32>,
    time: f32,
};

@group(1) @binding(0) var<uniform> params: BubbleParams;

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    // |prof| = 0 at the bubble's centre, 1 at the quad edge.
    @location(0) prof: vec2<f32>,
    @location(1) alpha: f32,
    // 0 while rising, 1 during the surface pop.
    @location(2) pop: f32,
};

const TAU: f32 = 6.28318530718;

fn pcg(v: u32) -> u32 {
    let s = v * 747796405u + 2891336453u;
    let w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
    return (w >> 22u) ^ w;
}

fn hash01(v: u32) -> f32 {
    return f32(pcg(v)) / 4294967295.0;
}

@vertex
fn vertex(@builtin(vertex_index) vi: u32) -> VertexOutput {
    let bubble = vi / 6u;
    let corner = vi % 6u;
    var prof: vec2<f32>;
    switch corner {
        case 0u: { prof = vec2<f32>(-1.0, -1.0); }
        case 1u: { prof = vec2<f32>(1.0, -1.0); }
        case 2u: { prof = vec2<f32>(-1.0, 1.0); }
        case 3u: { prof = vec2<f32>(1.0, -1.0); }
        case 4u: { prof = vec2<f32>(1.0, 1.0); }
        default: { prof = vec2<f32>(-1.0, 1.0); }
    }

    // Each bubble cycles on its own period/phase; the anchor re-rolls
    // every cycle so pops reseed elsewhere.
    let period = mix(5.0, 11.0, hash01(bubble * 4u + 1u));
    let life = params.time / period + hash01(bubble * 4u + 2u);
    let cycle = u32(floor(life));
    let t = fract(life);
    let seed = bubble * 747u + cycle * 2654435761u;
    let anchor = vec2<f32>(
        hash01(seed ^ 0x68bc21ebu) * params.bounds.x,
        hash01(seed ^ 0x02e5be93u) * params.bounds.y,
    );
    let r_max = mix(1.6, 4.2, hash01(seed ^ 0x967a889bu));

    var r: f32;
    var alpha: f32;
    var pop: f32;
    if (t < 0.85) {
        // Rising: grow toward the camera, fade in, sway more as the
        // surface (and its currents) get close.
        let rise = t / 0.85;
        r = mix(0.5, r_max, rise);
        alpha = 0.55 * smoothstep(0.0, 0.12, t) * mix(0.4, 1.0, rise);
        pop = 0.0;
    } else {
        // The pop: a ring snaps outward and fades.
        let pt = (t - 0.85) / 0.15;
        r = r_max * (1.0 + 2.5 * pt);
        alpha = 0.8 * (1.0 - pt);
        pop = 1.0;
    }
    let rise = min(t / 0.85, 1.0);
    let fb = f32(bubble);
    let sway = vec2<f32>(
        sin(params.time * 1.7 + fb * 2.39),
        cos(params.time * 1.3 + fb * 5.71),
    ) * (2.5 * rise);
    let centre = anchor + sway;
    let window = centre + prof * r;
    let world = vec2<f32>(window.x - params.origin.x, params.origin.y - window.y);

    var out: VertexOutput;
    out.clip_position = view.clip_from_world * vec4<f32>(world, 0.0, 1.0);
    out.prof = prof;
    out.alpha = alpha;
    out.pop = pop;
    return out;
}

@fragment
fn fragment(in: VertexOutput) -> @location(0) vec4<f32> {
    let d = length(in.prof);
    // A bubble from above: a bright rim, a faint fill, and a small
    // sun-side highlight. The pop keeps only the (thinning) rim.
    let rim = smoothstep(0.45, 0.8, d) * (1.0 - smoothstep(0.8, 1.0, d));
    let fill = (1.0 - smoothstep(0.0, 0.9, d)) * 0.10 * (1.0 - in.pop);
    let hl = (1.0 - smoothstep(0.0, 0.3, length(in.prof - vec2<f32>(-0.32, -0.32))))
        * 0.7 * (1.0 - in.pop);
    let i = (rim * 0.55 + fill + hl) * in.alpha;
    return vec4<f32>(vec3<f32>(0.75, 0.92, 1.0) * i, 0.0);
}
";

// ---------------------------------------------------------------------------
// The sketch style: the pond as a cartoonist doodles it — the fish's own
// flat-fill / white-stroke language. One flat color underfoot; all the
// life is white linework drawn by the ring pipeline, plus sparse diamond
// twinkles. Same uniform structs and bindings as the natural shaders
// (the one bind group serves every style); the height grid goes unread.

// The sketch bed: one flat cartoon teal with the gentlest radial fade so
// big windows don't read as a paint-bucket fill. Nothing else — the
// linework above carries the style.
const SKETCH_WATER_SHADER: &str = r"
#import bevy_sprite::mesh2d_view_bindings::view

struct WaterParams {
    origin: vec2<f32>,
    bounds: vec2<f32>,
    time: f32,
    caustics: f32,
    sparkle: f32,
    cell: f32,
    cols: u32,
    rows: u32,
};

@group(1) @binding(0) var<uniform> params: WaterParams;

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) window: vec2<f32>,
};

@vertex
fn vertex(@builtin(vertex_index) vi: u32) -> VertexOutput {
    var corner: vec2<f32>;
    switch vi {
        case 0u: { corner = vec2<f32>(0.0, 0.0); }
        case 1u: { corner = vec2<f32>(1.0, 0.0); }
        case 2u: { corner = vec2<f32>(0.0, 1.0); }
        case 3u: { corner = vec2<f32>(1.0, 0.0); }
        case 4u: { corner = vec2<f32>(1.0, 1.0); }
        default: { corner = vec2<f32>(0.0, 1.0); }
    }
    let window = corner * params.bounds;
    let world = vec2<f32>(window.x - params.origin.x, params.origin.y - window.y);
    var out: VertexOutput;
    out.clip_position = view.clip_from_world * vec4<f32>(world, 0.0, 1.0);
    out.window = window;
    return out;
}

fn srgb_to_linear(c: vec3<f32>) -> vec3<f32> {
    let lo = c / 12.92;
    let hi = pow((c + vec3<f32>(0.055)) / 1.055, vec3<f32>(2.4));
    return select(hi, lo, c <= vec3<f32>(0.04045));
}

@fragment
fn fragment(in: VertexOutput) -> @location(0) vec4<f32> {
    let p = in.window;
    let edge = distance(p, params.origin) / max(0.5 * length(params.bounds), 1.0);
    let g = smoothstep(0.25, 1.0, edge);
    let col = mix(vec3<f32>(0.11, 0.38, 0.45), vec3<f32>(0.07, 0.28, 0.36), g);
    return vec4<f32>(srgb_to_linear(col), 1.0);
}
";

// The sketch surface — over the fish: a whisper of flat wash so they sit
// under the water, and sparse hand-placed diamond twinkles. The pond's
// linework (rings, squiggles) is the ring pipeline's job.
const SKETCH_SURFACE_SHADER: &str = r"
#import bevy_sprite::mesh2d_view_bindings::view

struct WaterParams {
    origin: vec2<f32>,
    bounds: vec2<f32>,
    time: f32,
    caustics: f32,
    sparkle: f32,
    cell: f32,
    cols: u32,
    rows: u32,
};

@group(1) @binding(0) var<uniform> params: WaterParams;

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) window: vec2<f32>,
};

@vertex
fn vertex(@builtin(vertex_index) vi: u32) -> VertexOutput {
    var corner: vec2<f32>;
    switch vi {
        case 0u: { corner = vec2<f32>(0.0, 0.0); }
        case 1u: { corner = vec2<f32>(1.0, 0.0); }
        case 2u: { corner = vec2<f32>(0.0, 1.0); }
        case 3u: { corner = vec2<f32>(1.0, 0.0); }
        case 4u: { corner = vec2<f32>(1.0, 1.0); }
        default: { corner = vec2<f32>(0.0, 1.0); }
    }
    let window = corner * params.bounds;
    let world = vec2<f32>(window.x - params.origin.x, params.origin.y - window.y);
    var out: VertexOutput;
    out.clip_position = view.clip_from_world * vec4<f32>(world, 0.0, 1.0);
    out.window = window;
    return out;
}

fn hash21(p: vec2<f32>) -> f32 {
    var q = fract(p * vec2<f32>(123.34, 345.45));
    q += dot(q, q + 34.345);
    return fract(q.x * q.y);
}

fn hash22(p: vec2<f32>) -> vec2<f32> {
    var p3 = fract(vec3<f32>(p.x, p.y, p.x) * vec3<f32>(0.1031, 0.1030, 0.0973));
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.xx + p3.yz) * p3.zy);
}

// Hand-placed flat twinkles: the window is cut into coarse cells, each on
// its own clock; when a cell's turn comes up (the Sparkle dial sets how
// often) it shows a small flat diamond at a re-rolled spot for a beat.
fn twinkle(p: vec2<f32>, t: f32, density: f32) -> f32 {
    let cellp = p / 56.0;
    let id = floor(cellp);
    let clock = t * 0.8 + hash21(id) * 7.0;
    let bucket = floor(clock);
    let phase = fract(clock);
    let h = hash21(id + vec2<f32>(bucket * 13.7, bucket * 5.3));
    // `active` is a WGSL reserved word, hence `lit`.
    let lit = step(1.0 - 0.22 * density, h);
    let held = step(abs(phase - 0.5), 0.32);
    let pos = id + vec2<f32>(0.2) + 0.6 * hash22(id + vec2<f32>(bucket * 31.7, bucket * 17.3));
    let dd = cellp - pos;
    let star = 1.0 - smoothstep(0.10, 0.16, abs(dd.x) + abs(dd.y));
    return lit * held * star;
}

fn srgb_to_linear(c: vec3<f32>) -> vec3<f32> {
    let lo = c / 12.92;
    let hi = pow((c + vec3<f32>(0.055)) / 1.055, vec3<f32>(2.4));
    return select(hi, lo, c <= vec3<f32>(0.04045));
}

@fragment
fn fragment(in: VertexOutput) -> @location(0) vec4<f32> {
    let spark = twinkle(in.window, params.time, params.sparkle);
    let a_wash = 0.06;
    let a_spark = spark * 0.60;
    let alpha = clamp(a_wash + a_spark, 0.0, 0.9);
    var rgb = srgb_to_linear(vec3<f32>(0.10, 0.34, 0.40)) * a_wash;
    rgb += srgb_to_linear(vec3<f32>(0.97, 1.0, 0.99)) * a_spark;
    return vec4<f32>(rgb, alpha);
}
";

// The sketch bubbles: the natural shader's vertex life-cycle (rise toward
// the camera, pop, reseed) with a doodled fragment — a bold white outline
// circle, a short highlight arc on the sun side, the faintest fill; the
// pop erodes the outline into dashes, the same hand-drawn break-up the
// rings use.
const SKETCH_BUBBLE_SHADER: &str = r"
#import bevy_sprite::mesh2d_view_bindings::view

struct BubbleParams {
    origin: vec2<f32>,
    bounds: vec2<f32>,
    time: f32,
};

@group(1) @binding(0) var<uniform> params: BubbleParams;

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) prof: vec2<f32>,
    @location(1) alpha: f32,
    // 0 while rising; ramps 0..1 across the pop.
    @location(2) prog: f32,
    // Per-bubble noise offset for the pop's dash break-up.
    @location(3) sv: vec2<f32>,
};

const TAU: f32 = 6.28318530718;

fn pcg(v: u32) -> u32 {
    let s = v * 747796405u + 2891336453u;
    let w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
    return (w >> 22u) ^ w;
}

fn hash01(v: u32) -> f32 {
    return f32(pcg(v)) / 4294967295.0;
}

@vertex
fn vertex(@builtin(vertex_index) vi: u32) -> VertexOutput {
    let bubble = vi / 6u;
    let corner = vi % 6u;
    var prof: vec2<f32>;
    switch corner {
        case 0u: { prof = vec2<f32>(-1.0, -1.0); }
        case 1u: { prof = vec2<f32>(1.0, -1.0); }
        case 2u: { prof = vec2<f32>(-1.0, 1.0); }
        case 3u: { prof = vec2<f32>(1.0, -1.0); }
        case 4u: { prof = vec2<f32>(1.0, 1.0); }
        default: { prof = vec2<f32>(-1.0, 1.0); }
    }

    let period = mix(5.0, 11.0, hash01(bubble * 4u + 1u));
    let life = params.time / period + hash01(bubble * 4u + 2u);
    let cycle = u32(floor(life));
    let t = fract(life);
    let seed = bubble * 747u + cycle * 2654435761u;
    let anchor = vec2<f32>(
        hash01(seed ^ 0x68bc21ebu) * params.bounds.x,
        hash01(seed ^ 0x02e5be93u) * params.bounds.y,
    );
    let r_max = mix(1.8, 4.6, hash01(seed ^ 0x967a889bu));

    var r: f32;
    var alpha: f32;
    var prog: f32;
    if (t < 0.85) {
        let rise = t / 0.85;
        r = mix(0.6, r_max, rise);
        alpha = 0.9 * smoothstep(0.0, 0.12, t) * mix(0.5, 1.0, rise);
        prog = 0.0;
    } else {
        let pt = (t - 0.85) / 0.15;
        r = r_max * (1.0 + 2.2 * pt);
        alpha = 0.9 * (1.0 - pt);
        prog = pt;
    }
    let rise = min(t / 0.85, 1.0);
    let fb = f32(bubble);
    let sway = vec2<f32>(
        sin(params.time * 1.7 + fb * 2.39),
        cos(params.time * 1.3 + fb * 5.71),
    ) * (2.5 * rise);
    let centre = anchor + sway;
    let window = centre + prof * r;
    let world = vec2<f32>(window.x - params.origin.x, params.origin.y - window.y);

    var out: VertexOutput;
    out.clip_position = view.clip_from_world * vec4<f32>(world, 0.0, 1.0);
    out.prof = prof;
    out.alpha = alpha;
    out.prog = prog;
    out.sv = vec2<f32>(
        f32(seed & 1023u) * 0.317,
        f32((seed >> 10u) & 1023u) * 0.173,
    );
    return out;
}

fn hash21(p: vec2<f32>) -> f32 {
    var q = fract(p * vec2<f32>(123.34, 345.45));
    q += dot(q, q + 34.345);
    return fract(q.x * q.y);
}

fn vnoise(p: vec2<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let u = f * f * (3.0 - 2.0 * f);
    let a = hash21(i);
    let b = hash21(i + vec2<f32>(1.0, 0.0));
    let c = hash21(i + vec2<f32>(0.0, 1.0));
    let d = hash21(i + vec2<f32>(1.0, 1.0));
    return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}

fn srgb_to_linear(c: vec3<f32>) -> vec3<f32> {
    let lo = c / 12.92;
    let hi = pow((c + vec3<f32>(0.055)) / 1.055, vec3<f32>(2.4));
    return select(hi, lo, c <= vec3<f32>(0.04045));
}

@fragment
fn fragment(in: VertexOutput) -> @location(0) vec4<f32> {
    let d = length(in.prof);
    let dir = in.prof / max(d, 0.001);

    // The outline circle, eroding into dashes as the pop progresses.
    let ring = 1.0 - smoothstep(0.10, 0.20, abs(d - 0.74));
    let e = in.prog * 0.8;
    let cover = smoothstep(e, e + 0.22, vnoise(dir * 2.6 + in.sv));

    // A short hand-drawn highlight arc on the sun side (upper-left),
    // and the faintest fill — gone the instant the pop starts.
    let solid = 1.0 - step(0.001, in.prog);
    let arc_band = 1.0 - smoothstep(0.07, 0.15, abs(d - 0.46));
    let arc_side = smoothstep(0.72, 0.92, dot(dir, vec2<f32>(-0.707, -0.707)));
    let arc = arc_band * arc_side * solid;
    let fill = (1.0 - smoothstep(0.0, 0.74, d)) * 0.05 * solid;

    let a_ring = ring * cover * 0.9;
    let a_arc = arc * 0.8;
    let a_fill = fill;
    let alpha = clamp(a_ring + a_arc + a_fill, 0.0, 1.0) * in.alpha;
    var rgb = srgb_to_linear(vec3<f32>(0.97, 1.0, 0.99)) * (a_ring + a_arc);
    rgb += srgb_to_linear(vec3<f32>(0.62, 0.86, 0.88)) * a_fill;
    return vec4<f32>(rgb * in.alpha, alpha);
}
";

// ---------------------------------------------------------------------------
// The glossy style: a storybook pond — soft gradients, gentle drifting
// highlight contours, soft pulsing sparkles; the ring pipeline draws its
// ripples as clean double-stroke circles. (Glossy bubbles reuse the
// natural shader — already soft.)

// The glossy bed: an airy teal gradient with soft highlight contours
// playing across it (the Caustics dial), like light through gentle waves.
const GLOSSY_WATER_SHADER: &str = r"
#import bevy_sprite::mesh2d_view_bindings::view

struct WaterParams {
    origin: vec2<f32>,
    bounds: vec2<f32>,
    time: f32,
    caustics: f32,
    sparkle: f32,
    cell: f32,
    cols: u32,
    rows: u32,
};

@group(1) @binding(0) var<uniform> params: WaterParams;

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) window: vec2<f32>,
};

@vertex
fn vertex(@builtin(vertex_index) vi: u32) -> VertexOutput {
    var corner: vec2<f32>;
    switch vi {
        case 0u: { corner = vec2<f32>(0.0, 0.0); }
        case 1u: { corner = vec2<f32>(1.0, 0.0); }
        case 2u: { corner = vec2<f32>(0.0, 1.0); }
        case 3u: { corner = vec2<f32>(1.0, 0.0); }
        case 4u: { corner = vec2<f32>(1.0, 1.0); }
        default: { corner = vec2<f32>(0.0, 1.0); }
    }
    let window = corner * params.bounds;
    let world = vec2<f32>(window.x - params.origin.x, params.origin.y - window.y);
    var out: VertexOutput;
    out.clip_position = view.clip_from_world * vec4<f32>(world, 0.0, 1.0);
    out.window = window;
    return out;
}

fn hash21(p: vec2<f32>) -> f32 {
    var q = fract(p * vec2<f32>(123.34, 345.45));
    q += dot(q, q + 34.345);
    return fract(q.x * q.y);
}

fn vnoise(p: vec2<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let u = f * f * (3.0 - 2.0 * f);
    let a = hash21(i);
    let b = hash21(i + vec2<f32>(1.0, 0.0));
    let c = hash21(i + vec2<f32>(0.0, 1.0));
    let d = hash21(i + vec2<f32>(1.0, 1.0));
    return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}

fn srgb_to_linear(c: vec3<f32>) -> vec3<f32> {
    let lo = c / 12.92;
    let hi = pow((c + vec3<f32>(0.055)) / 1.055, vec3<f32>(2.4));
    return select(hi, lo, c <= vec3<f32>(0.04045));
}

@fragment
fn fragment(in: VertexOutput) -> @location(0) vec4<f32> {
    let p = in.window;
    let t = params.time;

    let edge = distance(p, params.origin) / max(0.5 * length(params.bounds), 1.0);
    let g = smoothstep(0.15, 1.05, edge);
    var col = mix(vec3<f32>(0.15, 0.47, 0.55), vec3<f32>(0.09, 0.35, 0.45), g);

    // Two slim contour lines through a slow two-octave noise field — the
    // thin wavy light lines of a storybook pond, drifting gently. The
    // narrow band around each iso-level is what keeps them lines, not
    // blobs.
    let drift = vec2<f32>(t * 6.0, t * 3.8);
    let n = 0.7 * vnoise((p + drift) * 0.004) + 0.3 * vnoise((p - drift) * 0.009 + 4.7);
    let band1 = 1.0 - smoothstep(0.008, 0.022, abs(n - 0.50));
    let band2 = (1.0 - smoothstep(0.006, 0.018, abs(n - 0.38))) * 0.6;
    col += vec3<f32>(0.30, 0.52, 0.52) * (band1 + band2) * params.caustics * 0.6;

    return vec4<f32>(srgb_to_linear(clamp(col, vec3<f32>(0.0), vec3<f32>(1.0))), 1.0);
}
";

// The glossy surface — over the fish: a soft wash and round sparkles that
// breathe in and out at hash-picked spots (the Sparkle dial).
const GLOSSY_SURFACE_SHADER: &str = r"
#import bevy_sprite::mesh2d_view_bindings::view

struct WaterParams {
    origin: vec2<f32>,
    bounds: vec2<f32>,
    time: f32,
    caustics: f32,
    sparkle: f32,
    cell: f32,
    cols: u32,
    rows: u32,
};

@group(1) @binding(0) var<uniform> params: WaterParams;

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) window: vec2<f32>,
};

@vertex
fn vertex(@builtin(vertex_index) vi: u32) -> VertexOutput {
    var corner: vec2<f32>;
    switch vi {
        case 0u: { corner = vec2<f32>(0.0, 0.0); }
        case 1u: { corner = vec2<f32>(1.0, 0.0); }
        case 2u: { corner = vec2<f32>(0.0, 1.0); }
        case 3u: { corner = vec2<f32>(1.0, 0.0); }
        case 4u: { corner = vec2<f32>(1.0, 1.0); }
        default: { corner = vec2<f32>(0.0, 1.0); }
    }
    let window = corner * params.bounds;
    let world = vec2<f32>(window.x - params.origin.x, params.origin.y - window.y);
    var out: VertexOutput;
    out.clip_position = view.clip_from_world * vec4<f32>(world, 0.0, 1.0);
    out.window = window;
    return out;
}

const TAU: f32 = 6.28318530718;

fn hash21(p: vec2<f32>) -> f32 {
    var q = fract(p * vec2<f32>(123.34, 345.45));
    q += dot(q, q + 34.345);
    return fract(q.x * q.y);
}

fn hash22(p: vec2<f32>) -> vec2<f32> {
    var p3 = fract(vec3<f32>(p.x, p.y, p.x) * vec3<f32>(0.1031, 0.1030, 0.0973));
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.xx + p3.yz) * p3.zy);
}

fn srgb_to_linear(c: vec3<f32>) -> vec3<f32> {
    let lo = c / 12.92;
    let hi = pow((c + vec3<f32>(0.055)) / 1.055, vec3<f32>(2.4));
    return select(hi, lo, c <= vec3<f32>(0.04045));
}

@fragment
fn fragment(in: VertexOutput) -> @location(0) vec4<f32> {
    let p = in.window;
    let t = params.time;

    // Soft round sparkles, one per coarse cell, breathing on their own
    // phase. Static homes, gentle pulse — glossy, not glittery.
    let cellp = p / 64.0;
    let id = floor(cellp);
    let h = hash21(id);
    let lit = step(1.0 - 0.30 * params.sparkle, h);
    let pos = id + vec2<f32>(0.15) + 0.7 * hash22(id * 1.31 + 2.7);
    let dd = cellp - pos;
    let pulse = 0.5 + 0.5 * sin(t * 1.4 + h * TAU * 7.0);
    let spark = (1.0 - smoothstep(0.02, 0.18, length(dd))) * pulse * pulse * lit;

    let a_wash = 0.08;
    let a_spark = spark * 0.28;
    let alpha = clamp(a_wash + a_spark, 0.0, 0.9);
    var rgb = srgb_to_linear(vec3<f32>(0.16, 0.45, 0.52)) * a_wash;
    rgb += srgb_to_linear(vec3<f32>(0.95, 1.0, 1.0)) * a_spark;
    return vec4<f32>(rgb, alpha);
}
";

// ---------------------------------------------------------------------------
// The ring shader — the sketch/glossy ripples. One vertex-pulled quad per
// live ring event; the fragment draws a stroked circle expanding from the
// event. Sketch: a hand-drawn wobbly outline that erodes into dashes as
// it ages (noise sampled on the unit direction, so the break-up is
// seamless around the circle). Glossy: a clean double stroke. The
// uniform's `style` picks the look (1 = sketch, 2 = glossy).
const RING_SHADER: &str = r"
#import bevy_sprite::mesh2d_view_bindings::view

struct RingParams {
    origin: vec2<f32>,
    bounds: vec2<f32>,
    time: f32,
    style: u32,
};

// Mirrors the Rust GpuRing (24-byte stride).
struct Ring {
    pos: vec2<f32>,
    born: f32,
    amp: f32,
    seed: u32,
    kind: u32,
};

@group(1) @binding(0) var<uniform> params: RingParams;
@group(1) @binding(1) var<storage, read> rings: array<Ring>;

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    // Offset from the ring's centre, px.
    @location(0) local: vec2<f32>,
    // (radius px, age 0..1, stroke half-width px, erosion base).
    @location(1) shape: vec4<f32>,
    // Per-ring noise offset.
    @location(2) sv: vec2<f32>,
};

fn pcg(v: u32) -> u32 {
    let s = v * 747796405u + 2891336453u;
    let w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
    return (w >> 22u) ^ w;
}

fn hash01(v: u32) -> f32 {
    return f32(pcg(v)) / 4294967295.0;
}

@vertex
fn vertex(@builtin(vertex_index) vi: u32) -> VertexOutput {
    let index = vi / 6u;
    let corner = vi % 6u;
    var prof: vec2<f32>;
    switch corner {
        case 0u: { prof = vec2<f32>(-1.0, -1.0); }
        case 1u: { prof = vec2<f32>(1.0, -1.0); }
        case 2u: { prof = vec2<f32>(-1.0, 1.0); }
        case 3u: { prof = vec2<f32>(1.0, -1.0); }
        case 4u: { prof = vec2<f32>(1.0, 1.0); }
        default: { prof = vec2<f32>(-1.0, 1.0); }
    }
    let ring = rings[index];

    // Kind shaping — lifetimes mirror the Rust RING_LIFE.
    let life = select(select(0.9, 1.6, ring.kind == 1u), 2.6, ring.kind == 2u);
    let age = params.time - ring.born;
    let a01 = age / life;

    var out: VertexOutput;
    if (age < 0.0 || a01 >= 1.0) {
        // Asleep (staggered birth) or expired: park the quad off-screen.
        out.clip_position = vec4<f32>(2.0, 2.0, 2.0, 1.0);
        out.local = vec2<f32>(0.0);
        out.shape = vec4<f32>(0.0);
        out.sv = vec2<f32>(0.0);
        return out;
    }

    var r_max: f32;
    var radius: f32;
    var w: f32;
    var erosion: f32;
    let ease = 1.0 - (1.0 - a01) * (1.0 - a01);
    if (ring.kind == 0u) {
        // Wake: small, quick, breaks up early.
        r_max = 8.0 + 18.0 * ring.amp;
        radius = mix(3.0, r_max, ease);
        w = 1.4;
        erosion = 0.15;
    } else if (ring.kind == 1u) {
        // Plop: the reference look — big, bold, holds together longest.
        r_max = 26.0 + 30.0 * ring.amp;
        radius = mix(4.0, r_max, ease);
        w = 1.8;
        erosion = 0.05;
    } else {
        // Ambient squiggle: barely grows, mostly dashes from birth.
        r_max = 16.0 + 22.0 * hash01(ring.seed);
        radius = r_max * (0.92 + 0.08 * a01);
        w = 1.2;
        erosion = 0.42;
    }

    let half = radius * 1.16 + w + 4.0;
    let window = ring.pos + prof * half;
    let world = vec2<f32>(window.x - params.origin.x, params.origin.y - window.y);
    out.clip_position = view.clip_from_world * vec4<f32>(world, 0.0, 1.0);
    out.local = prof * half;
    out.shape = vec4<f32>(radius, a01, w, erosion);
    out.sv = vec2<f32>(
        f32(ring.seed & 1023u) * 0.317,
        f32((ring.seed >> 10u) & 1023u) * 0.173,
    );
    return out;
}

fn hash21(p: vec2<f32>) -> f32 {
    var q = fract(p * vec2<f32>(123.34, 345.45));
    q += dot(q, q + 34.345);
    return fract(q.x * q.y);
}

fn vnoise(p: vec2<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let u = f * f * (3.0 - 2.0 * f);
    let a = hash21(i);
    let b = hash21(i + vec2<f32>(1.0, 0.0));
    let c = hash21(i + vec2<f32>(0.0, 1.0));
    let d = hash21(i + vec2<f32>(1.0, 1.0));
    return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}

fn srgb_to_linear(c: vec3<f32>) -> vec3<f32> {
    let lo = c / 12.92;
    let hi = pow((c + vec3<f32>(0.055)) / 1.055, vec3<f32>(2.4));
    return select(hi, lo, c <= vec3<f32>(0.04045));
}

@fragment
fn fragment(in: VertexOutput) -> @location(0) vec4<f32> {
    let radius = in.shape.x;
    let a01 = in.shape.y;
    let w = in.shape.z;
    let erosion = in.shape.w;
    let rad = length(in.local);
    let dir = in.local / max(rad, 0.001);

    var alpha: f32;
    var color: vec3<f32>;
    if (params.style == 1u) {
        // Sketch: wobble the radius around the circumference, erode the
        // stroke into dashes as it ages.
        let wob = (vnoise(dir * 1.7 + in.sv) - 0.5) * 0.22 * radius;
        let stroke = 1.0 - smoothstep(w, w + 1.4, abs(rad - (radius + wob)));
        let e = erosion + a01 * a01 * 0.65;
        let cover = smoothstep(e, e + 0.22, vnoise(dir * 3.1 + in.sv * 1.9 + 7.7));
        let fade = 1.0 - smoothstep(0.55, 1.0, a01);
        alpha = stroke * cover * fade * 0.85;
        color = vec3<f32>(0.97, 1.0, 0.99);
    } else {
        // Glossy: a clean stroke with a fainter inner echo.
        let s1 = 1.0 - smoothstep(w * 0.8, w * 0.8 + 1.4, abs(rad - radius));
        let s2 = (1.0 - smoothstep(w * 0.5, w * 0.5 + 1.2, abs(rad - radius * 0.72))) * 0.5;
        let fade = 1.0 - smoothstep(0.45, 1.0, a01);
        alpha = clamp(s1 + s2, 0.0, 1.0) * fade * 0.6;
        color = vec3<f32>(0.85, 0.97, 1.0);
    }

    return vec4<f32>(srgb_to_linear(color) * alpha, alpha);
}
";

#[cfg(test)]
mod tests {
    use super::*;

    fn splat_centre(grid: &mut RippleGrid) -> (usize, usize) {
        let (c, r) = (grid.cols / 2, grid.rows / 2);
        grid.splat(
            Vec2::new(c as f32 * RIPPLE_CELL, r as f32 * RIPPLE_CELL),
            1.0,
        );
        (c, r)
    }

    /// A node-aligned splat deposits its full amplitude on that node.
    #[test]
    fn splat_deposits_bilinearly() {
        let mut grid = RippleGrid::default();
        grid.resize_for(Vec2::new(120.0, 120.0));
        let (c, r) = splat_centre(&mut grid);
        assert!((grid.curr[r * grid.cols + c] - 1.0).abs() < 1e-6);
        let total: f32 = grid.curr.iter().sum();
        assert!((total - 1.0).abs() < 1e-6, "deposit sums to amp: {total}");

        // Off-node splits over the four neighbours, conserving the total.
        grid.calm();
        grid.splat(Vec2::new(RIPPLE_CELL * 1.5, RIPPLE_CELL * 2.25), 1.0);
        let total: f32 = grid.curr.iter().sum();
        assert!((total - 1.0).abs() < 1e-6, "bilinear sums to amp: {total}");
        assert!(grid.curr.iter().filter(|h| **h > 0.0).count() == 4);
    }

    /// The wave equation spreads a splat outward — after a few steps the
    /// centre has shed height to a surrounding ring.
    #[test]
    fn wave_spreads_into_a_ring() {
        let mut grid = RippleGrid::default();
        grid.resize_for(Vec2::new(300.0, 300.0));
        let (c, r) = splat_centre(&mut grid);
        let centre = r * grid.cols + c;
        for _ in 0..8 {
            grid.step();
        }
        // The peak moved off the centre node...
        let peak = grid
            .curr
            .iter()
            .cloned()
            .fold(f32::MIN, f32::max);
        assert!(grid.curr[centre] < peak, "centre shed its height");
        // ...to nodes a few cells away (the expanding ring).
        let ring = grid.curr[centre + 4];
        assert!(ring.abs() > 1e-4, "ring carries energy: {ring}");
    }

    /// Damping drains the surface: after enough steps everything decays
    /// toward calm.
    #[test]
    fn wave_decays_to_calm() {
        let mut grid = RippleGrid::default();
        grid.resize_for(Vec2::new(120.0, 120.0));
        splat_centre(&mut grid);
        for _ in 0..600 {
            grid.step();
        }
        let max = grid.curr.iter().fold(0.0f32, |m, h| m.max(h.abs()));
        assert!(max < 0.01, "surface calmed: {max}");
    }

    /// Splats clamp instead of running away under a dense pile.
    #[test]
    fn splats_saturate() {
        let mut grid = RippleGrid::default();
        grid.resize_for(Vec2::new(120.0, 120.0));
        for _ in 0..1000 {
            splat_centre(&mut grid);
        }
        let max = grid.curr.iter().fold(0.0f32, |m, h| m.max(h.abs()));
        assert!(max <= SPLAT_CLAMP + 1e-6, "clamped: {max}");
    }

    /// Resizing restarts calm; same-size calls keep the surface.
    #[test]
    fn resize_resets_only_on_change() {
        let mut grid = RippleGrid::default();
        grid.resize_for(Vec2::new(300.0, 300.0));
        splat_centre(&mut grid);
        grid.resize_for(Vec2::new(300.0, 300.0));
        assert!(grid.curr.iter().any(|h| *h != 0.0), "same size kept");
        grid.resize_for(Vec2::new(600.0, 300.0));
        assert!(grid.curr.iter().all(|h| *h == 0.0), "resize calmed");
        assert_eq!(grid.cols, (600.0 / RIPPLE_CELL).ceil() as usize + 1);
    }

    /// The ring pool grows to its cap, then recycles — but only slots
    /// whose ring is past its prime, so a saturated pool drops spawns
    /// instead of strobing fresh rings.
    #[test]
    fn ring_pool_caps_and_recycles() {
        let mut rings = RippleRings::default();
        for i in 0..(RING_MAX + 50) {
            rings.spawn(0.0, Vec2::new(i as f32, 0.0), 1.0, 0, 0.0);
        }
        assert_eq!(rings.rings.len(), RING_MAX, "pool capped");
        // All slots are brand new: nothing may be stolen.
        let before: Vec<f32> = rings.rings.iter().map(|r| r.pos.x).collect();
        rings.spawn(0.1, Vec2::new(-1.0, 0.0), 1.0, 0, 0.0);
        let after: Vec<f32> = rings.rings.iter().map(|r| r.pos.x).collect();
        assert_eq!(before, after, "young rings keep their slots");
        // Once the pool ages past its prime, spawns recycle slots.
        rings.spawn(RING_LIFE[0], Vec2::new(-2.0, 0.0), 1.0, 0, 0.0);
        assert!(
            rings.rings.iter().any(|r| r.pos.x == -2.0),
            "an aged slot was recycled"
        );
        rings.clear();
        assert!(rings.rings.is_empty());
    }

    /// All the shader strings must be valid WGSL. They only reach naga at
    /// runtime, so a typo — or a helper function missing from one shader,
    /// which slipped through once — passes clippy and every other test,
    /// then kills that layer live. Parse and validate each, with the one
    /// bevy `#import` stubbed to what the shaders actually use of it.
    #[test]
    fn shaders_compile() {
        let stub = "struct View { clip_from_world: mat4x4<f32> }\n\
                    @group(0) @binding(0) var<uniform> view: View;";
        for (name, src) in [
            ("water", WATER_SHADER),
            ("surface", SURFACE_SHADER),
            ("bubbles", BUBBLE_SHADER),
            ("sketch_water", SKETCH_WATER_SHADER),
            ("sketch_surface", SKETCH_SURFACE_SHADER),
            ("sketch_bubbles", SKETCH_BUBBLE_SHADER),
            ("glossy_water", GLOSSY_WATER_SHADER),
            ("glossy_surface", GLOSSY_SURFACE_SHADER),
            ("rings", RING_SHADER),
        ] {
            let src = src.replace("#import bevy_sprite::mesh2d_view_bindings::view", stub);
            let module = naga::front::wgsl::parse_str(&src)
                .unwrap_or_else(|e| panic!("{name}: parse: {e}"));
            naga::valid::Validator::new(
                naga::valid::ValidationFlags::all(),
                naga::valid::Capabilities::all(),
            )
            .validate(&module)
            .unwrap_or_else(|e| panic!("{name}: validate: {e}"));
        }
    }

    /// The CFL bound the constants must respect for the step to be stable.
    #[test]
    fn wave_constants_are_stable() {
        assert!(WAVE_C2 <= 0.5, "2D CFL bound");
        assert!(WAVE_DAMP < 1.0);
        // A long, undisturbed run must not blow up (stability smoke test).
        let mut grid = RippleGrid::default();
        grid.resize_for(Vec2::new(240.0, 240.0));
        splat_centre(&mut grid);
        for _ in 0..2000 {
            grid.step();
        }
        assert!(grid.curr.iter().all(|h| h.is_finite()));
    }
}
