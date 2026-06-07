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

use super::settings::FishSettings;
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

pub fn plugin(app: &mut App) {
    // Shaders only exist in the main world; hand the handles into the
    // render app.
    let mut shaders = app.world_mut().resource_mut::<Assets<Shader>>();
    let water = shaders.add(Shader::from_wgsl(WATER_SHADER, concat!(file!(), "#water")));
    let surface = shaders.add(Shader::from_wgsl(
        SURFACE_SHADER,
        concat!(file!(), "#surface"),
    ));
    let bubbles = shaders.add(Shader::from_wgsl(BUBBLE_SHADER, concat!(file!(), "#bubbles")));

    app.init_resource::<RippleGrid>()
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
        .insert_resource(WaterShader {
            water,
            surface,
            bubbles,
        })
        .add_render_command::<Transparent2d, DrawWater>()
        .add_render_command::<Transparent2d, DrawBubbles>()
        .init_resource::<SpecializedRenderPipelines<WaterPipeline>>()
        .init_resource::<SpecializedRenderPipelines<SurfacePipeline>>()
        .init_resource::<SpecializedRenderPipelines<BubblePipeline>>()
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

/// Frame-to-frame memory for the splat sources: last head positions (for
/// wake deltas) and the last food state (for plops).
#[derive(Default)]
struct RippleMemo {
    heads: Vec<Vec2>,
    food: Option<Vec2>,
    eaten: u32,
}

/// Advance the water clock, splat this frame's disturbances, and run the
/// surface's fixed-step wave equation.
#[allow(clippy::too_many_arguments)]
fn step_ripples(
    time: Res<Time>,
    settings: Res<FishSettings>,
    fishes: Res<Fishes>,
    game: Res<FishGame>,
    bounds: Res<SimBounds>,
    mut grid: ResMut<RippleGrid>,
    mut clock: ResMut<WaterClock>,
    mut memo: Local<RippleMemo>,
) {
    let dt = time.delta_secs();
    clock.0 = (clock.0 + dt) % 3600.0;

    if !settings.water || !settings.ripples {
        // Toggling ripples back on starts from a calm pond.
        if !grid.curr.is_empty() {
            grid.calm();
        }
        memo.heads.clear();
        memo.food = None;
        return;
    }
    grid.resize_for(bounds.0);

    // Fish wakes: each head's travel this frame deposits where it now is.
    // The per-frame delta scales with dt, so the wake energy is frame-rate
    // independent; teleports (restart, count changes) deposit nothing.
    let count = fishes.0.len();
    if memo.heads.len() != count {
        memo.heads.clear();
        memo.heads.extend(fishes.0.iter().map(|fish| fish.head()));
    }
    for (fish, prev) in fishes.0.iter().zip(memo.heads.iter_mut()) {
        let head = fish.head();
        let delta = head.distance(*prev);
        if delta > 1e-3 && delta < WAKE_TELEPORT {
            let size = (fish.scale * WAKE_SCALE_GAIN).clamp(0.2, 2.0);
            let amp = settings.ripple_strength * (delta * WAKE_AMP_PER_PX).min(WAKE_AMP_MAX) * size;
            grid.splat(head, amp);
        }
        *prev = head;
    }

    // Food plops: a big one where the dot was eaten, a small one where the
    // fresh dot lands. First frame only records (no startup splash).
    if let Some(last) = memo.food {
        if game.eaten != memo.eaten {
            grid.splat(last, settings.ripple_strength * PLOP_EAT_AMP);
        }
        if game.food != last {
            grid.splat(game.food, settings.ripple_strength * PLOP_DROP_AMP);
        }
    }
    memo.food = Some(game.food);
    memo.eaten = game.eaten;

    grid.acc = (grid.acc + dt).min(WAVE_STEP * WAVE_MAX_STEPS);
    while grid.acc >= WAVE_STEP {
        grid.acc -= WAVE_STEP;
        grid.step();
    }
}

/// Calm the pond when another experiment takes over; returning starts
/// fresh. (The draw itself is gated per frame in extract, so nothing can
/// draw stale over the next experiment.)
fn clear_when_inactive(current: Res<CurrentExperiment>, mut grid: ResMut<RippleGrid>) {
    if !current.is_changed() || current.0 == ExperimentId::Fish {
        return;
    }
    grid.calm();
}

// ---------------------------------------------------------------------------
// Render world: the bed quad's uniform + height storage buffer, the
// bubbles' uniform, two pipelines over the standard 2D view uniform.

/// Resource holding the shader handles for the pipelines to take.
#[derive(Resource)]
struct WaterShader {
    water: Handle<Shader>,
    surface: Handle<Shader>,
    bubbles: Handle<Shader>,
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

/// GPU buffers for both layers.
#[derive(Resource)]
struct WaterBuffers {
    /// The surface heights, row-major — or a single calm node when the
    /// ripples are off (every bilinear read then lands on a flat 0).
    heights: RawBufferVec<f32>,
    params: UniformBuffer<WaterParams>,
    bubble_params: UniformBuffer<BubbleParams>,
    water_on: bool,
    bubbles_on: bool,
    bubble_verts: u32,
    water_bind_group: Option<BindGroup>,
    bubble_bind_group: Option<BindGroup>,
}

impl Default for WaterBuffers {
    fn default() -> Self {
        Self {
            heights: RawBufferVec::new(BufferUsages::STORAGE),
            params: UniformBuffer::default(),
            bubble_params: UniformBuffer::default(),
            water_on: false,
            bubbles_on: false,
            bubble_verts: 0,
            water_bind_group: None,
            bubble_bind_group: None,
        }
    }
}

/// Copy this frame's water state into the render world. Reads the main
/// world's resources directly — the only per-frame payload is the height
/// grid. (`Option`: the buffers resource is created in `RenderStartup`,
/// which may not have run yet.)
fn extract_water(
    current: Extract<Res<CurrentExperiment>>,
    settings: Extract<Res<FishSettings>>,
    clock: Extract<Res<WaterClock>>,
    grid: Extract<Res<RippleGrid>>,
    bounds: Extract<Res<SimBounds>>,
    buffers: Option<ResMut<WaterBuffers>>,
) {
    let Some(mut buffers) = buffers else { return };
    buffers.water_on = current.0 == ExperimentId::Fish && settings.water;
    if !buffers.water_on {
        buffers.bubbles_on = false;
        return;
    }
    let origin = bounds.0 / 2.0;

    let ripples_live = settings.ripples && grid.cols >= 2 && !grid.curr.is_empty();
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
fn prepare_water(
    mut buffers: ResMut<WaterBuffers>,
    water_pipeline: Option<Res<WaterPipeline>>,
    bubble_pipeline: Option<Res<BubblePipeline>>,
    pipeline_cache: Res<PipelineCache>,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
) {
    buffers.water_bind_group = None;
    buffers.bubble_bind_group = None;
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
}

/// The bed/surface pipeline: one window-covering quad from `vertex_index`;
/// the fragment shader does everything. Group 1: uniform + height grid.
#[derive(Resource)]
struct WaterPipeline {
    mesh2d_pipeline: Mesh2dPipeline,
    shader: Handle<Shader>,
    layout: BindGroupLayoutDescriptor,
}

/// The surface pipeline — the same quad and bind group as the bed, a
/// different fragment: the translucent layer drawn over the fish.
#[derive(Resource)]
struct SurfacePipeline {
    mesh2d_pipeline: Mesh2dPipeline,
    shader: Handle<Shader>,
    /// A clone of the bed pipeline's descriptor: the pipeline cache
    /// dedupes identical descriptors, so the bed's bind group fits both.
    layout: BindGroupLayoutDescriptor,
}

/// The bubble pipeline: vertex-pull from `vertex_index` alone. Group 1:
/// the uniform.
#[derive(Resource)]
struct BubblePipeline {
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
        shader: shader.water.clone(),
        layout: water_layout.clone(),
    });
    commands.insert_resource(SurfacePipeline {
        mesh2d_pipeline: mesh2d_pipeline.clone(),
        shader: shader.surface.clone(),
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
        shader: shader.bubbles.clone(),
        layout: bubble_layout,
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

impl SpecializedRenderPipeline for WaterPipeline {
    type Key = Mesh2dPipelineKey;

    fn specialize(&self, key: Self::Key) -> RenderPipelineDescriptor {
        water_pipeline_descriptor(
            "fish_water_pipeline",
            &self.shader,
            &self.mesh2d_pipeline.view_layout,
            &self.layout,
            key,
        )
    }
}

impl SpecializedRenderPipeline for SurfacePipeline {
    type Key = Mesh2dPipelineKey;

    fn specialize(&self, key: Self::Key) -> RenderPipelineDescriptor {
        water_pipeline_descriptor(
            "fish_water_surface_pipeline",
            &self.shader,
            &self.mesh2d_pipeline.view_layout,
            &self.layout,
            key,
        )
    }
}

impl SpecializedRenderPipeline for BubblePipeline {
    type Key = Mesh2dPipelineKey;

    fn specialize(&self, key: Self::Key) -> RenderPipelineDescriptor {
        water_pipeline_descriptor(
            "fish_bubble_pipeline",
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

type DrawWater = (SetItemPipeline, SetMesh2dViewBindGroup<0>, DrawWaterQuad);

type DrawBubbles = (SetItemPipeline, SetMesh2dViewBindGroup<0>, DrawBubbleQuads);

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
    mut water_pipelines: ResMut<SpecializedRenderPipelines<WaterPipeline>>,
    mut surface_pipelines: ResMut<SpecializedRenderPipelines<SurfacePipeline>>,
    mut bubble_pipelines: ResMut<SpecializedRenderPipelines<BubblePipeline>>,
    pipeline_cache: Res<PipelineCache>,
    buffers: Option<Res<WaterBuffers>>,
    mut transparent_render_phases: ResMut<ViewSortedRenderPhases<Transparent2d>>,
    views: Query<(&ExtractedView, &Msaa)>,
) {
    let (Some(water_pipeline), Some(surface_pipeline), Some(bubble_pipeline), Some(buffers)) =
        (water_pipeline, surface_pipeline, bubble_pipeline, buffers)
    else {
        return;
    };
    if !buffers.water_on {
        return;
    }
    let draw_water = transparent_draw_functions.read().id::<DrawWater>();
    let draw_bubbles = transparent_draw_functions.read().id::<DrawBubbles>();

    for (view, msaa) in &views {
        let Some(transparent_phase) = transparent_render_phases.get_mut(&view.retained_view_entity)
        else {
            continue;
        };
        let key = Mesh2dPipelineKey::from_msaa_samples(msaa.samples())
            | Mesh2dPipelineKey::from_hdr(view.hdr)
            | Mesh2dPipelineKey::from_primitive_topology(PrimitiveTopology::TriangleList);

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
