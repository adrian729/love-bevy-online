//! GPU-compute flock simulation: the same Reynolds rules as `sim.rs`,
//! ported to WGSL and run entirely on the GPU. Per frame, one compute pass
//! with five dispatches:
//!
//! 1. `clear_grid` — zero the per-cell counters
//! 2. `count`      — atomically histogram boids into grid cells
//! 3. `scan`       — prefix-sum the (~hundred) cells into span starts
//! 4. `scatter`    — copy the flock into cell-major order (+ id map)
//! 5. `steer`      — neighbour rules, steering, integration; writes the new
//!    state *and* the render instances
//!
//! The authoritative state buffer stays in insertion order (`id_of` maps
//! sorted slots back), so boid identity is stable: growing appends, and
//! shrinking truncates — exactly the LÖVE original's `Flock:setSize`. The
//! CPU's only per-frame work is a uniform update; on grow/restart it uploads
//! the new boids' states. The render instances never leave the GPU.
//!
//! The CPU simulation in `sim.rs` remains available behind the `cpu`
//! perf-flag, and is the reference for behaviour parity: the WGSL mirrors
//! its math constant-for-constant, including the `MAX_NEIGHBOUR_SAMPLES`
//! circular-window sampling.

use std::borrow::Cow;
use std::f32::consts::TAU;

use bevy::prelude::*;
use bevy::render::extract_resource::{ExtractResource, ExtractResourcePlugin};
use bevy::render::render_graph::{self, RenderGraph, RenderLabel};
use bevy::render::render_resource::binding_types::{storage_buffer_sized, uniform_buffer};
use bevy::render::render_resource::{
    BindGroup, BindGroupEntries, BindGroupLayoutDescriptor, BindGroupLayoutEntries, Buffer,
    BufferDescriptor, BufferUsages, CachedComputePipelineId, CachedPipelineState,
    ComputePassDescriptor, ComputePipelineDescriptor, PipelineCache, ShaderStages, ShaderType,
    UniformBuffer,
};
use bevy::render::renderer::{RenderContext, RenderDevice, RenderQueue};
use bevy::render::{Render, RenderApp, RenderStartup, RenderSystems};
use bevy::shader::PipelineCacheError;
use bevy::window::PrimaryWindow;
use rand::Rng;

use super::settings::{NEIGHBOUR_DIST, SimSettings};
use crate::app::{AppState, PinnedAttractor, PointerOverUi, RestartRequested, SimBounds};

/// Buffer capacity in boids — two more doublings of headroom past the
/// current slider maximum. ~150 MB of GPU memory all-in at this size.
pub const MAX_BOIDS: usize = 2_097_152;
/// Grid-cell buffer capacity (supports windows up to ~12k x 12k px).
const MAX_CELLS: usize = 16_384;

/// Everything the WGSL needs per frame. Field order mirrors the shader's
/// `Params` struct.
#[derive(Resource, Clone, Default, ExtractResource, ShaderType)]
pub struct GpuFlockParams {
    size: Vec2,
    half: Vec2,
    dt: f32,
    max_speed: f32,
    separation: f32,
    alignment: f32,
    cohesion: f32,
    mouse_active: f32,
    frame_salt: u32,
    count: u32,
    mouse_pos: Vec2,
    cols: u32,
    rows: u32,
}

impl GpuFlockParams {
    /// Current flock size (the renderer's instance count in GPU mode).
    pub fn count(&self) -> u32 {
        self.count
    }
}

/// Whether the GPU sim should run this frame (false: paused, `cpu`/`nosim`
/// perf modes, or empty flock).
#[derive(Resource, Clone, Default, ExtractResource)]
pub struct GpuSimRun(pub bool);

/// The CPU-side mirror of how many boids live on the GPU, plus the pending
/// state uploads for newly spawned ones.
#[derive(Resource, Clone, Default, ExtractResource)]
pub struct GpuSpawns {
    /// Index of the first pending boid (`upload` starts here).
    base: u32,
    /// `[pos.x, pos.y, vel.x, vel.y]` per new boid.
    upload: Vec<Vec4>,
}

/// CPU mirror of the GPU flock size — also what the HUD shows in GPU mode.
#[derive(Resource, Default)]
pub struct GpuFlockCount(pub usize);

/// Grow/shrink the GPU flock to the slider, and feed the per-frame uniform.
/// Mirrors `sync_flock_size` + the cursor logic of the CPU `flocking`.
#[allow(clippy::too_many_arguments)]
fn gpu_sync_flock(
    time: Res<Time>,
    settings: Res<SimSettings>,
    bounds: Res<SimBounds>,
    pointer_over_ui: Res<PointerOverUi>,
    pinned: Res<PinnedAttractor>,
    state: Res<State<AppState>>,
    keys: Res<ButtonInput<KeyCode>>,
    mut restart: ResMut<RestartRequested>,
    mut count: ResMut<GpuFlockCount>,
    mut spawns: ResMut<GpuSpawns>,
    mut params: ResMut<GpuFlockParams>,
    mut run: ResMut<GpuSimRun>,
    window: Query<&Window, With<PrimaryWindow>>,
    camera: Query<(&Camera, &GlobalTransform), With<Camera2d>>,
) {
    // Last frame's uploads have been extracted; start fresh.
    spawns.upload.clear();
    spawns.base = count.0 as u32;

    // The sim also steps behind the menu (the live backdrop); the R key
    // only restarts during actual play.
    let playing = *state.get() == AppState::Playing;
    let active = state.get().sim_runs();
    if (restart.0 && active) || (keys.just_pressed(KeyCode::KeyR) && playing) {
        restart.0 = false;
        count.0 = 0;
        spawns.base = 0;
    }

    if active {
        let target = (settings.count.round().max(1.0) as usize).min(MAX_BOIDS);
        if count.0 < target {
            let half = bounds.0 / 2.0;
            let mut rng = rand::rng();
            spawns.upload.reserve(target - count.0);
            for _ in count.0..target {
                let pos = Vec2::new(
                    rng.random_range(-half.x..=half.x),
                    rng.random_range(-half.y..=half.y),
                );
                let vel = Vec2::from_angle(rng.random_range(0.0..TAU))
                    * rng.random_range(0.0..=settings.speed);
                spawns.upload.push(Vec4::new(pos.x, pos.y, vel.x, vel.y));
            }
        }
        // Shrinking just truncates — the state buffer is in insertion order,
        // exactly like the arrays in the original's `Flock:setSize`.
        count.0 = target;
    }

    let size = bounds.0.max(Vec2::ONE);
    // No mouse force behind the menu — the original's `menuBg:update(dt, true)`.
    let mouse = if *state.get() == AppState::Menu {
        None
    } else if pinned.0 {
        Some(Vec2::ZERO)
    } else if pointer_over_ui.0 {
        None
    } else {
        window
            .single()
            .ok()
            .and_then(|window| window.cursor_position())
            .and_then(|screen| {
                let (cam, cam_tf) = camera.single().ok()?;
                cam.viewport_to_world_2d(cam_tf, screen).ok()
            })
    };
    *params = GpuFlockParams {
        size,
        half: size / 2.0,
        dt: time.delta_secs(),
        max_speed: settings.speed,
        separation: settings.separation,
        alignment: settings.alignment,
        cohesion: settings.cohesion,
        mouse_active: mouse.is_some() as u32 as f32,
        frame_salt: time.elapsed().as_millis() as u32,
        count: count.0 as u32,
        mouse_pos: mouse.unwrap_or(Vec2::ZERO),
        cols: (size.x / NEIGHBOUR_DIST).ceil().max(1.0) as u32,
        rows: (size.y / NEIGHBOUR_DIST).ceil().max(1.0) as u32,
    };
    run.0 = active && count.0 > 0;
}

/// The GPU-side buffers. `instances_rev` doubles as the renderer's vertex
/// slot 1.
#[derive(Resource)]
pub struct FlockGpuBuffers {
    /// `vec4(pos, vel)` per boid, insertion order — the authoritative state.
    state: Buffer,
    /// Cell-major copies, split so the steer pass's distance test streams
    /// half the bytes (velocity is only read for accepted neighbours).
    sorted_pos: Buffer,
    sorted_vel: Buffer,
    cell_of: Buffer,
    id_of: Buffer,
    counts: Buffer,
    starts: Buffer,
    /// `vec4(pos, rot)` per boid, insertion order — written by `steer`.
    instances: Buffer,
    /// `instances` reversed by the `reverse_instances` pass — what the
    /// instanced draw reads (see that kernel for why).
    pub instances_rev: Buffer,
    uniform: UniformBuffer<GpuFlockParams>,
}

fn init_gpu_buffers(mut commands: Commands, render_device: Res<RenderDevice>) {
    let storage = |label: &'static str, size: u64, extra: BufferUsages| {
        render_device.create_buffer(&BufferDescriptor {
            label: Some(label),
            size,
            usage: BufferUsages::STORAGE | extra,
            mapped_at_creation: false,
        })
    };
    let n = MAX_BOIDS as u64;
    commands.insert_resource(FlockGpuBuffers {
        state: storage("flock_state", n * 16, BufferUsages::COPY_DST),
        sorted_pos: storage("flock_sorted_pos", n * 8, BufferUsages::empty()),
        sorted_vel: storage("flock_sorted_vel", n * 8, BufferUsages::empty()),
        cell_of: storage("flock_cell_of", n * 4, BufferUsages::empty()),
        id_of: storage("flock_id_of", n * 4, BufferUsages::empty()),
        counts: storage("flock_counts", MAX_CELLS as u64 * 4, BufferUsages::empty()),
        starts: storage(
            "flock_starts",
            (MAX_CELLS as u64 + 1) * 4,
            BufferUsages::empty(),
        ),
        instances: storage("flock_instances", n * 16, BufferUsages::empty()),
        instances_rev: storage("flock_instances_rev", n * 16, BufferUsages::VERTEX),
        uniform: UniformBuffer::default(),
    });
}

#[derive(Resource)]
struct FlockSimPipelines {
    layout: BindGroupLayoutDescriptor,
    clear_grid: CachedComputePipelineId,
    count: CachedComputePipelineId,
    scan: CachedComputePipelineId,
    scatter: CachedComputePipelineId,
    steer: CachedComputePipelineId,
    reverse: CachedComputePipelineId,
}

/// Handle to the compute shader, created in the main world at plugin build
/// (`Assets<Shader>` does not exist in the render world).
#[derive(Resource)]
struct SimShader(Handle<Shader>);

fn init_sim_pipelines(
    mut commands: Commands,
    pipeline_cache: Res<PipelineCache>,
    shader: Res<SimShader>,
) {
    let layout = BindGroupLayoutDescriptor::new(
        "flock_sim",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::COMPUTE,
            (
                uniform_buffer::<GpuFlockParams>(false),
                storage_buffer_sized(false, None), // state
                storage_buffer_sized(false, None), // sorted_pos
                storage_buffer_sized(false, None), // sorted_vel
                storage_buffer_sized(false, None), // cell_of
                storage_buffer_sized(false, None), // id_of
                storage_buffer_sized(false, None), // counts (atomic)
                storage_buffer_sized(false, None), // starts
                storage_buffer_sized(false, None), // instances
                storage_buffer_sized(false, None), // instances_rev
            ),
        ),
    );
    let pipeline = |entry: &'static str| {
        pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some(format!("flock_{entry}").into()),
            layout: vec![layout.clone()],
            shader: shader.0.clone(),
            entry_point: Some(Cow::from(entry)),
            ..default()
        })
    };
    commands.insert_resource(FlockSimPipelines {
        clear_grid: pipeline("clear_grid"),
        count: pipeline("count_boids"),
        scan: pipeline("scan_cells"),
        scatter: pipeline("scatter"),
        steer: pipeline("steer"),
        reverse: pipeline("reverse_instances"),
        layout,
    });
}

#[derive(Resource)]
struct FlockSimBindGroup(BindGroup);

/// Upload pending spawns + this frame's params; (re)build the bind group.
/// (`Option`s: the first render frames can run before the first extract.)
#[allow(clippy::too_many_arguments)]
fn prepare_sim(
    mut commands: Commands,
    buffers: Option<ResMut<FlockGpuBuffers>>,
    pipelines: Option<Res<FlockSimPipelines>>,
    pipeline_cache: Res<PipelineCache>,
    params: Option<Res<GpuFlockParams>>,
    spawns: Option<Res<GpuSpawns>>,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
) {
    let (Some(mut buffers), Some(pipelines), Some(params), Some(spawns)) =
        (buffers, pipelines, params, spawns)
    else {
        return;
    };
    if !spawns.upload.is_empty() {
        let base = (spawns.base as usize).min(MAX_BOIDS) as u64;
        render_queue.write_buffer(
            &buffers.state,
            base * 16,
            bytemuck::cast_slice(&spawns.upload),
        );
    }
    buffers.uniform.set(params.clone());
    buffers.uniform.write_buffer(&render_device, &render_queue);

    let bind_group = render_device.create_bind_group(
        "flock_sim",
        &pipeline_cache.get_bind_group_layout(&pipelines.layout),
        &BindGroupEntries::sequential((
            &buffers.uniform,
            buffers.state.as_entire_binding(),
            buffers.sorted_pos.as_entire_binding(),
            buffers.sorted_vel.as_entire_binding(),
            buffers.cell_of.as_entire_binding(),
            buffers.id_of.as_entire_binding(),
            buffers.counts.as_entire_binding(),
            buffers.starts.as_entire_binding(),
            buffers.instances.as_entire_binding(),
            buffers.instances_rev.as_entire_binding(),
        )),
    );
    commands.insert_resource(FlockSimBindGroup(bind_group));
}

#[derive(Debug, Hash, PartialEq, Eq, Clone, RenderLabel)]
struct FlockSimLabel;

#[derive(Default)]
struct FlockSimNode {
    ready: bool,
}

impl render_graph::Node for FlockSimNode {
    fn update(&mut self, world: &mut World) {
        if self.ready {
            return;
        }
        let Some(pipelines) = world.get_resource::<FlockSimPipelines>() else {
            return;
        };
        let cache = world.resource::<PipelineCache>();
        self.ready = [
            pipelines.clear_grid,
            pipelines.count,
            pipelines.scan,
            pipelines.scatter,
            pipelines.steer,
            pipelines.reverse,
        ]
        .iter()
        .all(|&id| match cache.get_compute_pipeline_state(id) {
            CachedPipelineState::Ok(_) => true,
            CachedPipelineState::Err(PipelineCacheError::ShaderNotLoaded(_)) => false,
            CachedPipelineState::Err(err) => panic!("flock sim shader: {err}"),
            _ => false,
        });
    }

    fn run(
        &self,
        _graph: &mut render_graph::RenderGraphContext,
        render_context: &mut RenderContext,
        world: &World,
    ) -> Result<(), render_graph::NodeRunError> {
        if !self.ready {
            return Ok(());
        }
        // The simulation proper only steps while playing…
        let stepping = world.get_resource::<GpuSimRun>().is_some_and(|run| run.0);
        let Some(params) = world.get_resource::<GpuFlockParams>() else {
            return Ok(());
        };
        if params.count == 0 {
            return Ok(());
        }
        let Some(bind_group) = world.get_resource::<FlockSimBindGroup>().map(|b| &b.0) else {
            return Ok(());
        };
        let pipelines = world.resource::<FlockSimPipelines>();
        let cache = world.resource::<PipelineCache>();

        let ncells = params.cols * params.rows;
        // Workgroup size 256 (here and in the WGSL) is the portable choice,
        // not a machine-specific tuning: it's the WebGPU baseline limit —
        // guaranteed everywhere — and a multiple of every vendor's SIMD
        // width (32/64), so no lane sits idle on any GPU.
        let boid_groups = params.count.div_ceil(256);
        let mut pass =
            render_context
                .command_encoder()
                .begin_compute_pass(&ComputePassDescriptor {
                    label: Some("flock_sim"),
                    ..default()
                });
        pass.set_bind_group(0, bind_group, &[]);
        // Sequential dispatches in one pass: WebGPU guarantees the writes of
        // each dispatch are visible to the next.
        let sim = [
            (pipelines.clear_grid, ncells.div_ceil(256)),
            (pipelines.count, boid_groups),
            (pipelines.scan, 1),
            (pipelines.scatter, boid_groups),
            (pipelines.steer, boid_groups),
        ];
        // …but the render copy refreshes even when paused, so count changes
        // made from the options menu show the same boids a truncation of the
        // authoritative buffer would.
        let always = [(pipelines.reverse, boid_groups)];
        for &(pipeline, groups) in sim.iter().filter(|_| stepping).chain(&always) {
            let Some(pipeline) = cache.get_compute_pipeline(pipeline) else {
                return Ok(());
            };
            pass.set_pipeline(pipeline);
            pass.dispatch_workgroups(groups, 1, 1);
        }
        Ok(())
    }
}

pub struct FlockGpuSimPlugin;

impl Plugin for FlockGpuSimPlugin {
    fn build(&self, app: &mut App) {
        let shader = app
            .world_mut()
            .resource_mut::<Assets<Shader>>()
            .add(Shader::from_wgsl(SIM_SHADER, file!()));

        app.init_resource::<GpuFlockParams>()
            .init_resource::<GpuSpawns>()
            .init_resource::<GpuFlockCount>()
            .init_resource::<GpuSimRun>()
            .add_plugins((
                ExtractResourcePlugin::<GpuFlockParams>::default(),
                ExtractResourcePlugin::<GpuSpawns>::default(),
                ExtractResourcePlugin::<GpuSimRun>::default(),
            ))
            .add_systems(Update, gpu_sync_flock.after(crate::app::update_sim_bounds));

        let render_app = app.sub_app_mut(RenderApp);
        render_app
            .insert_resource(SimShader(shader))
            .add_systems(RenderStartup, (init_gpu_buffers, init_sim_pipelines))
            .add_systems(Render, prepare_sim.in_set(RenderSystems::PrepareBindGroups));
        let mut render_graph = render_app.world_mut().resource_mut::<RenderGraph>();
        render_graph.add_node(FlockSimLabel, FlockSimNode::default());
        render_graph.add_node_edge(FlockSimLabel, bevy::render::graph::CameraDriverLabel);
    }
}

const SIM_SHADER: &str = r"
// Constants mirrored from src/experiments/flock/settings.rs — keep in sync.
const NEIGHBOUR_DIST: f32 = 100.0;
const NEIGHBOUR_D2: f32 = 10000.0;
const SEPARATE_D2: f32 = 2500.0;
const MAX_FORCE: f32 = 0.3;
const REF_FPS: f32 = 60.0;
const MOUSE_NEAR2: f32 = 10000.0;
const MOUSE_ATTRACT_K: f32 = 4.0;
const MOUSE_REPEL_K: f32 = -6.0;
// Mirrors MAX_NEIGHBOUR_SAMPLES in src/experiments/flock/sim.rs.
const MAX_SAMPLES: u32 = 128u;

struct Params {
    size: vec2<f32>,
    half: vec2<f32>,
    dt: f32,
    max_speed: f32,
    separation: f32,
    alignment: f32,
    cohesion: f32,
    mouse_active: f32,
    frame_salt: u32,
    count: u32,
    mouse_pos: vec2<f32>,
    cols: u32,
    rows: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
// vec4(pos.xy, vel.xy), insertion order — authoritative.
@group(0) @binding(1) var<storage, read_write> state: array<vec4<f32>>;
// Cell-major position/velocity copies (split: the distance test only
// streams positions; velocities load on acceptance).
@group(0) @binding(2) var<storage, read_write> sorted_pos: array<vec2<f32>>;
@group(0) @binding(3) var<storage, read_write> sorted_vel: array<vec2<f32>>;
@group(0) @binding(4) var<storage, read_write> cell_of: array<u32>;
// sorted slot -> boid id.
@group(0) @binding(5) var<storage, read_write> id_of: array<u32>;
// per-cell counters, then write cursors after the scan.
@group(0) @binding(6) var<storage, read_write> counts: array<atomic<u32>>;
// exclusive prefix sums: cell c spans sorted[starts[c]..starts[c+1]].
@group(0) @binding(7) var<storage, read_write> starts: array<u32>;
// vec4(pos.xy, rot.xy) — the instanced draw's vertex slot 1.
@group(0) @binding(8) var<storage, read_write> instances: array<vec4<f32>>;
@group(0) @binding(9) var<storage, read_write> instances_rev: array<vec4<f32>>;

fn cell_x(x: f32) -> u32 {
    return min(u32(max((x + params.half.x) / NEIGHBOUR_DIST, 0.0)), params.cols - 1u);
}
fn cell_y(y: f32) -> u32 {
    return min(u32(max((y + params.half.y) / NEIGHBOUR_DIST, 0.0)), params.rows - 1u);
}

@compute @workgroup_size(256)
fn clear_grid(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x < params.cols * params.rows) {
        atomicStore(&counts[gid.x], 0u);
    }
}

@compute @workgroup_size(256)
fn count_boids(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.count) { return; }
    let p = state[i].xy;
    let c = cell_y(p.y) * params.cols + cell_x(p.x);
    cell_of[i] = c;
    atomicAdd(&counts[c], 1u);
}

// The grid is ~a hundred cells; a single thread scans it in microseconds.
// The counters become each cell's write cursor for the scatter.
@compute @workgroup_size(1)
fn scan_cells(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x != 0u) { return; }
    let ncells = params.cols * params.rows;
    var running = 0u;
    for (var c = 0u; c < ncells; c++) {
        starts[c] = running;
        let n = atomicLoad(&counts[c]);
        atomicStore(&counts[c], running);
        running += n;
    }
    starts[ncells] = running;
}

@compute @workgroup_size(256)
fn scatter(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.count) { return; }
    let dst = atomicAdd(&counts[cell_of[i]], 1u);
    let boid = state[i];
    sorted_pos[dst] = boid.xy;
    sorted_vel[dst] = boid.zw;
    id_of[dst] = i;
}

// The original's target_force: k * limit(normalize(dir) * max_speed -
// velocity, MAX_FORCE), per-frame units at the 60 fps reference -> px/s^2.
fn target_force(k: f32, dir: vec2<f32>, vel: vec2<f32>) -> vec2<f32> {
    var n = vec2<f32>(0.0);
    let len2 = dot(dir, dir);
    if (len2 > 0.0) { n = dir / sqrt(len2); }
    let steer = n * params.max_speed - vel / REF_FPS;
    let sl2 = dot(steer, steer);
    var clamped = steer;
    if (sl2 > MAX_FORCE * MAX_FORCE) { clamped = steer * (MAX_FORCE / sqrt(sl2)); }
    return k * clamped * REF_FPS * REF_FPS;
}

@compute @workgroup_size(256)
fn steer(@builtin(global_invocation_id) gid: vec3<u32>) {
    let slot = gid.x;
    if (slot >= params.count) { return; }
    let p = sorted_pos[slot];
    let v = sorted_vel[slot];

    // The 3x3 candidate cells: one contiguous sorted-array span per row.
    let cx = cell_x(p.x);
    let cy = cell_y(p.y);
    let x0 = select(cx - 1u, 0u, cx == 0u);
    let x1 = min(cx + 2u, params.cols);
    let y0 = select(cy - 1u, 0u, cy == 0u);
    let y1 = min(cy + 2u, params.rows);
    var span_s = array<u32, 3>(0u, 0u, 0u);
    var span_e = array<u32, 3>(0u, 0u, 0u);
    var nspans = 0u;
    var candidates = 0u;
    for (var y = y0; y < y1; y++) {
        let s = starts[y * params.cols + x0];
        let e = starts[y * params.cols + x1];
        span_s[nspans] = s;
        span_e[nspans] = e;
        nspans++;
        candidates += e - s;
    }

    // Identical sampling to the CPU kernel: exhaustive under the budget,
    // otherwise each span contributes a proportional contiguous block at a
    // per-boid per-frame pseudo-random offset, wrapping circularly.
    let stride = max((candidates + MAX_SAMPLES - 1u) / MAX_SAMPLES, 1u);
    let salt = slot * 0x9E3779B9u + params.frame_salt;
    var sum_sep = vec2<f32>(0.0);
    var sum_align = vec2<f32>(0.0);
    var sum_coh = vec2<f32>(0.0);
    var cnt_n = 0.0;
    var cnt_s = 0.0;
    for (var si = 0u; si < nspans; si++) {
        let s = span_s[si];
        let len = span_e[si] - s;
        if (len == 0u) { continue; }
        var take = len;
        var start = 0u;
        if (stride > 1u) {
            take = (len + stride - 1u) / stride;
            start = salt % len;
        }
        for (var t = 0u; t < take; t++) {
            var idx = start + t;
            if (idx >= len) { idx -= len; }
            let q = sorted_pos[s + idx];
            let d = p - q;
            let d2 = dot(d, d);
            // d2 > 0 excludes self, like the original's d > 0.
            if (d2 > 0.0 && d2 < NEIGHBOUR_D2) {
                sum_align += sorted_vel[s + idx];
                sum_coh += q;
                cnt_n += 1.0;
                if (d2 < SEPARATE_D2) {
                    // normalize(d)/dist == d/d2: away-vector, 1/d falloff.
                    sum_sep += d / d2;
                    cnt_s += 1.0;
                }
            }
        }
    }

    var acc = vec2<f32>(0.0);
    if (cnt_s > 0.0) { acc += target_force(params.separation, sum_sep, v); }
    if (cnt_n > 0.0) {
        acc += target_force(params.alignment, sum_align, v);
        acc += target_force(params.cohesion, sum_coh / cnt_n - p, v);
    }
    if (params.mouse_active > 0.5) {
        let diff = params.mouse_pos - p;
        var k = MOUSE_ATTRACT_K;
        if (dot(diff, diff) < MOUSE_NEAR2) { k = MOUSE_REPEL_K; }
        acc += target_force(k, diff, v);
    }

    // Integrate, clamp speed, wrap the toroidal screen.
    var new_v = v + acc * params.dt;
    let sp2 = dot(new_v, new_v);
    if (sp2 > params.max_speed * params.max_speed) {
        new_v *= params.max_speed / sqrt(sp2);
    }
    var pos = p + new_v * params.dt;
    if (pos.x < -params.half.x) { pos.x += params.size.x; }
    else if (pos.x >= params.half.x) { pos.x -= params.size.x; }
    if (pos.y < -params.half.y) { pos.y += params.size.y; }
    else if (pos.y >= params.half.y) { pos.y -= params.size.y; }

    let id = id_of[slot];
    var rot = instances[id].zw; // keep the old heading at a standstill
    if (sp2 > 0.0) { rot = new_v / sqrt(sp2); }
    state[id] = vec4<f32>(pos, new_v);
    instances[id] = vec4<f32>(pos, rot);
}

// The renderer wants the instance records in REVERSE boid order: it walks
// the buffer front-to-back (its z decreases with the GPU instance index) so
// early-z can reject the alpha-tested quads of occluded boids instead of
// shading a dense pile thousands of layers deep; reversed buffer x reversed
// z keeps the later-boids-draw-on-top layering of the LOVE original.
//
// Done as its own bandwidth-bound pass rather than in steer: id_of runs
// roughly ascending within each cell span, so steer writing
// instances[count - 1 - id] turns near-coalesced ascending writes into
// descending ones, which this GPU punishes (~25% off the whole frame at
// 640k pinned). Here the descending side is the READ, which caches fine.
@compute @workgroup_size(256)
fn reverse_instances(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.count) { return; }
    instances_rev[i] = instances[params.count - 1u - i];
}
";
