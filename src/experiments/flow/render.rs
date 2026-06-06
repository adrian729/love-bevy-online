//! Flow rendering: the static field visualization (the Lua's offscreen
//! canvas) plus the animated particle trails.
//!
//! Two layers, two pipelines, both blending **premultiplied alpha** (src
//! ONE, dst ONE_MINUS_SRC_ALPHA — a vertex emitting `(rgb·a, a)` blends
//! normally, one emitting `(rgb·a, 0)` adds pure light):
//! - **static** — background gradient + the current view (streamline
//!   ribbons / arrows / gradient fill) as CPU-built 12-byte-vertex
//!   triangles. Re-emitted and re-uploaded only when the field rebuilds
//!   (`FlowState::version`), the equivalent of the original's
//!   render-once-to-canvas.
//! - **trails** — GPU **vertex-pull**: the CPU uploads only each
//!   particle's raw trail ring buffer and a 24-byte meta record; the
//!   vertex shader expands them into tapered glow ribbons + head dots from
//!   `vertex_index` alone (no vertex buffer, no index buffer). Expanding
//!   on the CPU was measured geometry-throughput-bound: at 100k particles
//!   the triangles are ~170 MB/frame through emit → merge → extract →
//!   upload, vs ~18 MB/frame of ring buffers — the same lever as the
//!   flock's GPU sim.
//!
//! Geometry lives in the sim's window coordinates; the y-flip to world
//! coordinates happens at CPU vertex emission (static) or in the trail
//! shader via the uniform's `origin` (the fish convention).

use std::f32::consts::TAU;

use bevy::core_pipeline::core_2d::{CORE_2D_DEPTH_FORMAT, Transparent2d};
use bevy::ecs::query::ROQueryItem;
use bevy::ecs::system::SystemParamItem;
use bevy::ecs::system::lifetimeless::SRes;
use bevy::math::FloatOrd;
use bevy::mesh::VertexBufferLayout;
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
    DepthStencilState, FragmentState, IndexFormat, MultisampleState, PipelineCache,
    PrimitiveState, PrimitiveTopology, RawBufferVec, RenderPipelineDescriptor, ShaderStages,
    ShaderType, SpecializedRenderPipeline, SpecializedRenderPipelines, StencilFaceState,
    StencilState, TextureFormat, UniformBuffer, VertexAttribute, VertexFormat, VertexState,
    VertexStepMode,
};
use bevy::render::renderer::{RenderDevice, RenderQueue};
use bevy::render::sync_world::MainEntity;
use bevy::render::view::{ExtractedView, ViewTarget};
use bevy::render::{Extract, Render, RenderApp, RenderStartup, RenderSystems};
use bevy::sprite_render::{
    Mesh2dPipeline, Mesh2dPipelineKey, SetMesh2dViewBindGroup, init_mesh_2d_pipeline,
};
use bevy::tasks::ComputeTaskPool;
use bytemuck::{Pod, Zeroable};

use super::settings::{FlowMode, FlowPalette, FlowSettings};
use super::sim::{
    BuildSig, FlowField, FlowParticles, FlowSimSet, FlowState, FlowStreamlines, TRAIL_CAP,
    TRAIL_CUTOFF,
};
use crate::app::SimBounds;
use crate::experiments::{CurrentExperiment, ExperimentId, experiment_active};

/// The dimmed-background factors from minigames/flow.lua: the gradient
/// behind the particles view dims by 0.4, behind the stroke views by 0.55.
const BG_DIM_PARTICLES: f32 = 0.4;
const BG_DIM_STROKES: f32 = 0.55;
/// Streamline ribbons taper (width and alpha) over this share of each end.
const STREAM_TAPER: f32 = 0.2;
/// Trail ribbon half-widths, head → tail (the glow tent's zero edge).
const TRAIL_HEAD_W: f32 = 2.4;
const TRAIL_TAIL_W: f32 = 0.8;
/// Radius of the soft head glow (the Lua's 1.5px head dot).
const HEAD_GLOW_R: f32 = 2.4;
/// Particle counts at and past this pack their meta records chunk-parallel.
const PARALLEL_TRAILS: usize = 2048;
/// Shader vertices per trail segment: ONE quad spanning the ribbon's full
/// width; the glow tent (full alpha at the spine, zero at the edges) is a
/// per-fragment `1 − |v|` over an interpolated cross-axis coordinate —
/// the identical profile the old two-quads-per-segment expansion built
/// from vertex alphas, at half the vertex count (the trail layer is
/// geometry-throughput-bound; this was the second 'fewer bytes through
/// the geometry stage' lever after vertex pull itself).
const SEG_VERTS: u32 = 6;
/// Shader vertices for the head dot: one quad, a radial cone per fragment
/// (round — the old 12-vertex expansion was a diamond).
const DOT_VERTS: u32 = 6;

/// One triangle vertex: world position + **premultiplied** linear unorm
/// color. Alpha 0 with non-zero rgb = pure additive light.
#[derive(Clone, Copy, Pod, Zeroable)]
#[repr(C)]
struct FlowVertex {
    pos: [f32; 2],
    color: [u8; 4],
}

pub fn plugin(app: &mut App) {
    // Shaders only exist in the main world; hand the handles into the
    // render app.
    let mut shaders = app.world_mut().resource_mut::<Assets<Shader>>();
    let static_layer = shaders.add(Shader::from_wgsl(FLOW_SHADER, file!()));
    let trails = shaders.add(Shader::from_wgsl(FLOW_TRAIL_SHADER, concat!(file!(), "#trails")));

    app.init_resource::<FlowDrawData>()
        .init_resource::<FlowPaletteLut>()
        .add_systems(
            Update,
            (
                (emit_static, pack_trails)
                    .after(FlowSimSet)
                    .run_if(experiment_active(ExperimentId::Flow)),
                // Ungated: it must fire on the frame flow STOPS being
                // current, or its last buffers would draw over the next
                // experiment (all experiments share one sort key).
                clear_when_inactive,
            ),
        );

    app.sub_app_mut(RenderApp)
        .insert_resource(FlowShader {
            static_layer,
            trails,
        })
        .add_render_command::<Transparent2d, DrawFlow>()
        .add_render_command::<Transparent2d, DrawFlowTrails>()
        .init_resource::<SpecializedRenderPipelines<FlowPipeline>>()
        .init_resource::<SpecializedRenderPipelines<FlowTrailPipeline>>()
        .add_systems(
            RenderStartup,
            (
                init_flow_pipeline.after(init_mesh_2d_pipeline),
                |mut commands: Commands| {
                    commands.init_resource::<FlowBuffers>();
                },
            ),
        )
        .add_systems(ExtractSchedule, extract_flow)
        .add_systems(
            Render,
            (
                prepare_flow.in_set(RenderSystems::PrepareResources),
                queue_flow.in_set(RenderSystems::Queue),
            ),
        );
}

/// One particle's per-frame trail record for the vertex-pull shader: the
/// live head position, the ring-buffer state, and the packed linear color.
/// Mirrors the WGSL `Meta` struct (24 bytes, vec2-aligned).
#[derive(Clone, Copy, Pod, Zeroable)]
#[repr(C)]
struct TrailMeta {
    pos: [f32; 2],
    head: u32,
    len: u32,
    /// Linear rgb packed little-endian (unpack4x8unorm in the shader).
    color: u32,
    _pad: u32,
}

/// The trail shader's uniforms. Mirrors the WGSL `TrailParams` struct.
#[derive(Clone, Copy, ShaderType)]
struct TrailParams {
    /// Window center, for the window→world y-flip in the shader.
    origin: Vec2,
    /// Per-sample alpha falloff (1 − trail_fade).
    fade_k: f32,
    head_w: f32,
    tail_w: f32,
    glow_r: f32,
    /// How many trail samples stay above the alpha cutoff.
    max_points: u32,
    trail_cap: u32,
}

impl Default for TrailParams {
    fn default() -> Self {
        Self {
            origin: Vec2::ZERO,
            fade_k: 0.85,
            head_w: TRAIL_HEAD_W,
            tail_w: TRAIL_TAIL_W,
            glow_r: HEAD_GLOW_R,
            max_points: TRAIL_CAP as u32,
            trail_cap: TRAIL_CAP as u32,
        }
    }
}

/// How many trail points stay above the alpha cutoff at this fade — the
/// shader stops there, like the CPU cutoff in the original.
fn trail_max_points(trail_fade: f32) -> u32 {
    let fade_k = 1.0 - trail_fade;
    if fade_k >= 0.999 {
        TRAIL_CAP as u32
    } else {
        (((TRAIL_CUTOFF.ln() / fade_k.ln()).ceil() as usize + 1).min(TRAIL_CAP)) as u32
    }
}

/// Shader vertex count for one particle at this `max_points`.
fn trail_verts_per_particle(max_points: u32) -> u32 {
    (max_points.max(2) - 1) * SEG_VERTS + DOT_VERTS
}

/// Main-world handoff: the static layer (re-emitted on rebuild) and the
/// per-particle trail meta records (re-packed every frame; the trail
/// *samples* go straight from `FlowParticles` to the render world in
/// extract). `static_version` is the `FlowState::version` the static
/// geometry was built from — `None` until the first build and after a
/// clear.
#[derive(Resource, Default)]
struct FlowDrawData {
    static_vertices: Vec<FlowVertex>,
    static_indices: Vec<u32>,
    static_version: Option<u64>,
    meta: Vec<TrailMeta>,
    fade_k: f32,
    max_points: u32,
    origin: Vec2,
}

// ---------------------------------------------------------------------------
// Palettes — lib/flow.lua's PALETTES through lib/color.lua's hsv2rgb,
// pre-baked into a 256-entry angle→linear-rgb LUT (the hot trail loop
// would otherwise pay an sRGB conversion per particle per frame).

/// The Lua hsv2rgb: h in degrees, s/v in percent, sRGB floats out.
fn hsv2rgb(h: f32, s: f32, v: f32) -> [f32; 3] {
    let h = (h / 360.0).rem_euclid(1.0);
    let s = s / 100.0;
    let v = v / 100.0;
    let i = (h * 6.0).floor();
    let f = h * 6.0 - i;
    let p = v * (1.0 - s);
    let q = v * (1.0 - f * s);
    let t = v * (1.0 - (1.0 - f) * s);
    match (i as i32).rem_euclid(6) {
        0 => [v, t, p],
        1 => [q, v, p],
        2 => [p, v, t],
        3 => [p, q, v],
        4 => [t, p, v],
        _ => [v, p, q],
    }
}

/// One palette sample: `t` is the normalized flow angle. sRGB floats.
fn palette_srgb(palette: FlowPalette, t: f32) -> [f32; 3] {
    match palette {
        FlowPalette::Rainbow => hsv2rgb(t * 360.0, 80.0, 95.0),
        FlowPalette::Ocean => hsv2rgb(175.0 + t * 85.0, 65.0, 45.0 + t * 50.0),
        FlowPalette::Fire => hsv2rgb(t * 55.0, 90.0, 55.0 + t * 45.0),
        FlowPalette::Forest => hsv2rgb(75.0 + t * 95.0, 65.0, 45.0 + t * 50.0),
        FlowPalette::Mono => {
            let v = 0.2 + 0.75 * t;
            [v, v, v]
        }
    }
}

/// Angle → linear rgb, 256 buckets (invisible quantization, no per-sample
/// sRGB pow).
#[derive(Resource)]
struct FlowPaletteLut {
    palette: Option<FlowPalette>,
    lut: Vec<[f32; 3]>,
}

impl Default for FlowPaletteLut {
    fn default() -> Self {
        Self {
            palette: None,
            lut: vec![[0.0; 3]; 256],
        }
    }
}

impl FlowPaletteLut {
    fn ensure(&mut self, palette: FlowPalette) {
        if self.palette == Some(palette) {
            return;
        }
        for (i, entry) in self.lut.iter_mut().enumerate() {
            let srgb = palette_srgb(palette, i as f32 / 256.0);
            let linear = Color::srgb(srgb[0], srgb[1], srgb[2]).to_linear();
            *entry = [linear.red, linear.green, linear.blue];
        }
        self.palette = Some(palette);
    }

    /// Linear rgb for a flow angle (wrapped to [0, 2π) like the Lua's
    /// `colorForAngle`).
    fn linear(&self, angle: f32) -> [f32; 3] {
        let index = (angle.rem_euclid(TAU) * (256.0 / TAU)) as usize;
        self.lut[index.min(255)]
    }
}

/// Premultiplied vertex color: `(rgb·a, a)` blends normally, `(rgb·a, 0)`
/// adds pure light.
fn premul(rgb: [f32; 3], alpha: f32, additive: bool) -> [u8; 4] {
    let a = alpha.clamp(0.0, 1.0);
    [
        (rgb[0].clamp(0.0, 1.0) * a * 255.0).round() as u8,
        (rgb[1].clamp(0.0, 1.0) * a * 255.0).round() as u8,
        (rgb[2].clamp(0.0, 1.0) * a * 255.0).round() as u8,
        if additive { 0 } else { (a * 255.0).round() as u8 },
    ]
}

// ---------------------------------------------------------------------------
// Geometry emitters — flat-colored quads and ribbons in window
// coordinates, y-flipped at vertex emission.

#[derive(Default)]
struct FlowEmitter {
    vertices: Vec<FlowVertex>,
    indices: Vec<u32>,
    origin: Vec2,
}

impl FlowEmitter {
    fn clear(&mut self, origin: Vec2) {
        self.vertices.clear();
        self.indices.clear();
        self.origin = origin;
    }

    fn vertex(&mut self, p: Vec2, color: [u8; 4]) -> u32 {
        let index = self.vertices.len() as u32;
        self.vertices.push(FlowVertex {
            pos: [p.x - self.origin.x, self.origin.y - p.y],
            color,
        });
        index
    }

    /// A solid-core stroke with a 1px feather to transparent on each side —
    /// LÖVE's "smooth" line profile (lib note: a plain tent reads dimmer).
    /// `attrs(i)` gives each point's (half-core-width, alpha, rgb).
    fn ribbon_feathered(
        &mut self,
        points: &[Vec2],
        additive: bool,
        mut attrs: impl FnMut(usize) -> (f32, f32, [f32; 3]),
    ) {
        if points.len() < 2 {
            return;
        }
        let mut last_normal = Vec2::Y;
        let mut prev: Option<[u32; 4]> = None;
        for i in 0..points.len() {
            let dir = if i == 0 {
                points[1] - points[0]
            } else if i == points.len() - 1 {
                points[i] - points[i - 1]
            } else {
                points[i + 1] - points[i - 1]
            };
            let normal = if dir.length_squared() > 1e-8 {
                dir.perp().normalize() // window coords; flipped with the y at emit
            } else {
                last_normal
            };
            last_normal = normal;
            let (half, alpha, rgb) = attrs(i);
            let core = premul(rgb, alpha, additive);
            let edge = premul(rgb, 0.0, additive);
            let p = points[i];
            let row = [
                self.vertex(p + normal * (half + 1.0), edge),
                self.vertex(p + normal * half, core),
                self.vertex(p - normal * half, core),
                self.vertex(p - normal * (half + 1.0), edge),
            ];
            if let Some(last) = prev {
                for s in 0..3 {
                    self.indices.extend_from_slice(&[
                        last[s],
                        last[s + 1],
                        row[s],
                        last[s + 1],
                        row[s + 1],
                        row[s],
                    ]);
                }
            }
            prev = Some(row);
        }
    }

}

/// The gradient view / dimmed background: a vertex grid over the whole
/// viewport, one corner per cell boundary, colored by the smooth field
/// angle there. The GPU interpolates — the original's flat per-cell rects,
/// smoothed (a deliberate difference).
fn emit_gradient(
    emitter: &mut FlowEmitter,
    field: &FlowField,
    lut: &FlowPaletteLut,
    bounds: Vec2,
    brightness: f32,
) {
    let (cols, rows) = (field.cols, field.rows);
    let cw = bounds.x / cols as f32;
    let ch = bounds.y / rows as f32;
    let base = emitter.vertices.len() as u32;
    for i in 0..=rows {
        for j in 0..=cols {
            // Vertices span the viewport exactly (the Lua's cw = vw/cols
            // stretch); colors sample the field at the matching *grid*
            // coordinate, so a window that isn't a multiple of `scale`
            // stretches the field uniformly instead of skewing it.
            let p = Vec2::new(j as f32 * cw, i as f32 * ch);
            let grid_p = Vec2::new(j as f32 * field.scale, i as f32 * field.scale);
            let rgb = lut.linear(field.angle_at(grid_p));
            let color = premul(
                [rgb[0] * brightness, rgb[1] * brightness, rgb[2] * brightness],
                1.0,
                false,
            );
            emitter.vertex(p, color);
        }
    }
    let stride = (cols + 1) as u32;
    for i in 0..rows as u32 {
        for j in 0..cols as u32 {
            let a = base + i * stride + j;
            let b = a + 1;
            let c = a + stride;
            let d = c + 1;
            emitter.indices.extend_from_slice(&[a, b, c, b, d, c]);
        }
    }
}

/// The arrows view: one feathered stroke of length 3·scale per cell,
/// rotated to the cell angle, optional arrowheads (lib/flow.lua's
/// `draw_lines`).
fn emit_arrows(emitter: &mut FlowEmitter, field: &FlowField, lut: &FlowPaletteLut, sig: &BuildSig) {
    let len = 3.0 * field.scale;
    let head = field.scale * 0.7;
    let half = sig.line_width * 0.5;
    for r in 0..field.rows {
        for c in 0..field.cols {
            let angle = field.angles[r * field.cols + c];
            let rgb = lut.linear(angle);
            let anchor = Vec2::new((c as f32 + 0.5) * field.scale, (r as f32 + 0.5) * field.scale);
            let dir = Vec2::new(angle.cos(), angle.sin());
            let tip = anchor + dir * len;
            let attrs = |_: usize| (half, sig.opacity, rgb);
            emitter.ribbon_feathered(&[anchor, tip], false, attrs);
            if sig.arrowheads {
                let perp = dir.perp();
                let back = tip - dir * head;
                emitter.ribbon_feathered(&[tip, back + perp * (head * 0.45)], false, attrs);
                emitter.ribbon_feathered(&[tip, back - perp * (head * 0.45)], false, attrs);
            }
        }
    }
}

/// The streamlines view: every traced line as a feathered ribbon whose
/// width and alpha taper toward both ends (the Lua drew uniform hairline
/// segments), colored by the local field angle.
fn emit_streamlines(
    emitter: &mut FlowEmitter,
    field: &FlowField,
    lines: &FlowStreamlines,
    lut: &FlowPaletteLut,
    sig: &BuildSig,
) {
    let half = (sig.line_width * 0.5).max(0.1);
    for line in &lines.0 {
        if line.len() < 2 {
            continue;
        }
        let last = (line.len() - 1) as f32;
        emitter.ribbon_feathered(line, false, |i| {
            let t = i as f32 / last;
            let taper = (t / STREAM_TAPER).min((1.0 - t) / STREAM_TAPER).min(1.0);
            let rgb = lut.linear(field.angle_at(line[i]));
            (half * (0.25 + 0.75 * taper), sig.opacity * taper, rgb)
        });
    }
}

/// Re-emit the static layer when the field was rebuilt. Reads the
/// *applied* build signature, so the geometry always matches the field it
/// draws (settings may already be a throttle-step ahead mid-drag).
fn emit_static(
    flow: Res<FlowState>,
    field: Res<FlowField>,
    lines: Res<FlowStreamlines>,
    bounds: Res<SimBounds>,
    mut lut: ResMut<FlowPaletteLut>,
    mut data: ResMut<FlowDrawData>,
    mut emitter: Local<FlowEmitter>,
) {
    if data.static_version == Some(flow.version) {
        return;
    }
    let Some(sig) = flow.applied.as_ref() else {
        return;
    };
    if field.cols == 0 {
        return;
    }
    lut.ensure(sig.palette);
    emitter.clear(bounds.0 / 2.0);

    match sig.mode {
        FlowMode::Particles => {
            // The particles themselves are the dynamic layer; the static
            // canvas is blank unless the dimmed gradient is on.
            if sig.background {
                emit_gradient(&mut emitter, &field, &lut, bounds.0, 1.0 - BG_DIM_PARTICLES);
            }
        }
        FlowMode::Gradient => emit_gradient(&mut emitter, &field, &lut, bounds.0, 1.0),
        FlowMode::Streamlines => {
            if sig.background {
                emit_gradient(&mut emitter, &field, &lut, bounds.0, 1.0 - BG_DIM_STROKES);
            }
            emit_streamlines(&mut emitter, &field, &lines, &lut, sig);
        }
        FlowMode::Arrows => {
            if sig.background {
                emit_gradient(&mut emitter, &field, &lut, bounds.0, 1.0 - BG_DIM_STROKES);
            }
            emit_arrows(&mut emitter, &field, &lut, sig);
        }
    }

    // Swap, don't copy: the emitter inherits the old capacity back.
    std::mem::swap(&mut data.static_vertices, &mut emitter.vertices);
    std::mem::swap(&mut data.static_indices, &mut emitter.indices);
    data.static_version = Some(flow.version);
}

/// Pack each particle's trail meta record (head position, ring state,
/// palette color) for the vertex-pull shader — every frame, chunk-parallel
/// at high counts. Runs in Options too: the particles freeze there (the
/// sim is paused) but a palette change recolors the frozen trails live.
/// Cleared when the particle layer is off.
fn pack_trails(
    settings: Res<FlowSettings>,
    particles: Res<FlowParticles>,
    field: Res<FlowField>,
    bounds: Res<SimBounds>,
    mut lut: ResMut<FlowPaletteLut>,
    mut data: ResMut<FlowDrawData>,
) {
    let on = settings.animate || settings.mode == FlowMode::Particles;
    if !on || particles.count() == 0 || field.cols == 0 {
        data.meta.clear();
        return;
    }
    lut.ensure(settings.palette);
    data.origin = bounds.0 / 2.0;
    data.fade_k = 1.0 - settings.trail_fade;
    data.max_points = trail_max_points(settings.trail_fade);

    let count = particles.count();
    data.meta.resize(count, TrailMeta::zeroed());
    let pack_one = |particles: &FlowParticles, field: &FlowField, lut: &FlowPaletteLut, i: usize| {
        let pos = particles.pos[i];
        let rgb = lut.linear(field.angle_at(pos));
        TrailMeta {
            pos: pos.to_array(),
            head: particles.head[i] as u32,
            len: particles.len[i] as u32,
            color: u32::from_le_bytes([
                (rgb[0] * 255.0).round() as u8,
                (rgb[1] * 255.0).round() as u8,
                (rgb[2] * 255.0).round() as u8,
                255,
            ]),
            _pad: 0,
        }
    };
    if count < PARALLEL_TRAILS {
        for i in 0..count {
            data.meta[i] = pack_one(&particles, &field, &lut, i);
        }
    } else {
        let pool = ComputeTaskPool::get_or_init(Default::default);
        let chunk = count.div_ceil((pool.thread_num().max(1) * 3).min(count));
        let particles = &*particles;
        let field = &*field;
        let lut = &*lut;
        pool.scope(|scope| {
            for (chunk_index, metas) in data.meta.chunks_mut(chunk).enumerate() {
                let start = chunk_index * chunk;
                scope.spawn(async move {
                    for (offset, meta) in metas.iter_mut().enumerate() {
                        *meta = pack_one(particles, field, lut, start + offset);
                    }
                });
            }
        });
    }
}

/// Drop both layers (and the particles) when another experiment takes
/// over; returning rebuilds fresh (the cleared applied-signature forces a
/// field rebuild, which re-versions the static layer).
fn clear_when_inactive(
    current: Res<CurrentExperiment>,
    mut data: ResMut<FlowDrawData>,
    mut particles: ResMut<FlowParticles>,
    mut state: ResMut<FlowState>,
) {
    if !current.is_changed() || current.0 == ExperimentId::Flow {
        return;
    }
    data.static_vertices.clear();
    data.static_indices.clear();
    data.static_version = None;
    data.meta.clear();
    particles.clear();
    state.applied = None;
}

// ---------------------------------------------------------------------------
// Render world: the static layer's persistent vertex/index pair (uploaded
// only when its version moves), the trail layer's ring-buffer storage
// buffers (uploaded every frame), and two premultiplied-alpha pipelines
// over the standard 2D view uniform.

/// Resource holding the shader handles for the pipelines to take.
#[derive(Resource)]
struct FlowShader {
    static_layer: Handle<Shader>,
    trails: Handle<Shader>,
}

/// GPU buffers for both layers.
#[derive(Resource)]
struct FlowBuffers {
    static_vertices: RawBufferVec<FlowVertex>,
    static_indices: RawBufferVec<u32>,
    static_index_count: u32,
    static_seen: Option<u64>,
    static_dirty: bool,
    /// Every particle's trail ring buffer, raw (window coords).
    samples: RawBufferVec<[f32; 2]>,
    /// Per-particle head/len/color records.
    meta: RawBufferVec<TrailMeta>,
    params: UniformBuffer<TrailParams>,
    trail_count: u32,
    trail_max_points: u32,
    trail_bind_group: Option<BindGroup>,
}

impl Default for FlowBuffers {
    fn default() -> Self {
        Self {
            static_vertices: RawBufferVec::new(BufferUsages::VERTEX),
            static_indices: RawBufferVec::new(BufferUsages::INDEX),
            static_index_count: 0,
            static_seen: None,
            static_dirty: false,
            samples: RawBufferVec::new(BufferUsages::STORAGE),
            meta: RawBufferVec::new(BufferUsages::STORAGE),
            params: UniformBuffer::default(),
            trail_count: 0,
            trail_max_points: TRAIL_CAP as u32,
            trail_bind_group: None,
        }
    }
}

/// Copy this frame's data into the render world — the trail ring buffers
/// and meta records every frame (straight from the sim's `FlowParticles`,
/// no expanded geometry), the static layer only when its version moved.
/// (`Option`: the buffers resource is created in `RenderStartup`, which
/// may not have run yet.)
fn extract_flow(
    data: Extract<Res<FlowDrawData>>,
    particles: Extract<Res<FlowParticles>>,
    buffers: Option<ResMut<FlowBuffers>>,
) {
    let Some(mut buffers) = buffers else { return };
    if data.meta.is_empty() {
        buffers.trail_count = 0;
    } else {
        buffers.meta.values_mut().clone_from(&data.meta);
        let samples = buffers.samples.values_mut();
        samples.clear();
        samples.extend(particles.trail.iter().map(|p| p.to_array()));
        buffers.trail_count = data.meta.len() as u32;
        buffers.trail_max_points = data.max_points;
        let params = TrailParams {
            origin: data.origin,
            fade_k: data.fade_k,
            max_points: data.max_points,
            ..default()
        };
        buffers.params.set(params);
    }
    if buffers.static_seen != data.static_version {
        buffers
            .static_vertices
            .values_mut()
            .clone_from(&data.static_vertices);
        buffers
            .static_indices
            .values_mut()
            .clone_from(&data.static_indices);
        buffers.static_index_count = data.static_indices.len() as u32;
        buffers.static_seen = data.static_version;
        buffers.static_dirty = true;
    }
}

/// Upload — the static pair only when freshly extracted, the trail
/// buffers every frame, plus the trail bind group (rebuilt each frame:
/// `RawBufferVec` reallocations invalidate the old one).
fn prepare_flow(
    mut buffers: ResMut<FlowBuffers>,
    pipeline: Option<Res<FlowTrailPipeline>>,
    pipeline_cache: Res<PipelineCache>,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
) {
    if buffers.static_dirty {
        if buffers.static_index_count > 0 {
            buffers
                .static_vertices
                .write_buffer(&render_device, &render_queue);
            buffers
                .static_indices
                .write_buffer(&render_device, &render_queue);
        }
        buffers.static_dirty = false;
    }
    buffers.trail_bind_group = None;
    let Some(pipeline) = pipeline else { return };
    if buffers.trail_count == 0 {
        return;
    }
    buffers.samples.write_buffer(&render_device, &render_queue);
    buffers.meta.write_buffer(&render_device, &render_queue);
    buffers.params.write_buffer(&render_device, &render_queue);
    let (Some(samples), Some(meta)) = (buffers.samples.buffer(), buffers.meta.buffer()) else {
        return;
    };
    buffers.trail_bind_group = Some(render_device.create_bind_group(
        "flow_trails",
        &pipeline_cache.get_bind_group_layout(&pipeline.trail_layout),
        &BindGroupEntries::sequential((
            &buffers.params,
            samples.as_entire_binding(),
            meta.as_entire_binding(),
        )),
    ));
}

#[derive(Resource)]
struct FlowPipeline {
    mesh2d_pipeline: Mesh2dPipeline,
    shader: Handle<Shader>,
}

/// The trail layer's vertex-pull pipeline: no vertex buffers — group 1
/// carries the uniform + the two storage buffers the shader expands.
#[derive(Resource)]
struct FlowTrailPipeline {
    mesh2d_pipeline: Mesh2dPipeline,
    shader: Handle<Shader>,
    trail_layout: BindGroupLayoutDescriptor,
}

fn init_flow_pipeline(
    mut commands: Commands,
    mesh2d_pipeline: Res<Mesh2dPipeline>,
    shader: Res<FlowShader>,
) {
    commands.insert_resource(FlowPipeline {
        mesh2d_pipeline: mesh2d_pipeline.clone(),
        shader: shader.static_layer.clone(),
    });
    let trail_layout = BindGroupLayoutDescriptor::new(
        "flow_trail_layout",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::VERTEX,
            (
                uniform_buffer::<TrailParams>(false),
                storage_buffer_read_only_sized(false, None), // samples
                storage_buffer_read_only_sized(false, None), // meta
            ),
        ),
    );
    commands.insert_resource(FlowTrailPipeline {
        mesh2d_pipeline: mesh2d_pipeline.clone(),
        shader: shader.trails.clone(),
        trail_layout,
    });
}

impl SpecializedRenderPipeline for FlowPipeline {
    type Key = Mesh2dPipelineKey;

    fn specialize(&self, key: Self::Key) -> RenderPipelineDescriptor {
        let format = match key.contains(Mesh2dPipelineKey::HDR) {
            true => ViewTarget::TEXTURE_FORMAT_HDR,
            false => TextureFormat::bevy_default(),
        };

        RenderPipelineDescriptor {
            label: Some("flow_pipeline".into()),
            vertex: VertexState {
                shader: self.shader.clone(),
                buffers: vec![VertexBufferLayout {
                    array_stride: size_of::<FlowVertex>() as u64,
                    step_mode: VertexStepMode::Vertex,
                    attributes: vec![
                        VertexAttribute {
                            format: VertexFormat::Float32x2,
                            offset: 0,
                            shader_location: 0,
                        },
                        VertexAttribute {
                            format: VertexFormat::Unorm8x4,
                            offset: 8,
                            shader_location: 1,
                        },
                    ],
                }],
                ..default()
            },
            fragment: Some(FragmentState {
                shader: self.shader.clone(),
                targets: vec![Some(ColorTargetState {
                    format,
                    // Premultiplied: vertex alpha 0 with non-zero rgb is
                    // pure additive light (the trails); alpha > 0 blends
                    // normally (the static layers). One pipeline for both.
                    blend: Some(BlendState::PREMULTIPLIED_ALPHA_BLENDING),
                    write_mask: ColorWrites::ALL,
                })],
                ..default()
            }),
            // Group 0: the standard 2D view uniform.
            layout: vec![self.mesh2d_pipeline.view_layout.clone()],
            primitive: PrimitiveState {
                topology: PrimitiveTopology::TriangleList,
                ..default()
            },
            depth_stencil: Some(DepthStencilState {
                format: CORE_2D_DEPTH_FORMAT,
                // Blended vector art: no depth interaction, pure paint order.
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
}

impl SpecializedRenderPipeline for FlowTrailPipeline {
    type Key = Mesh2dPipelineKey;

    fn specialize(&self, key: Self::Key) -> RenderPipelineDescriptor {
        let format = match key.contains(Mesh2dPipelineKey::HDR) {
            true => ViewTarget::TEXTURE_FORMAT_HDR,
            false => TextureFormat::bevy_default(),
        };

        RenderPipelineDescriptor {
            label: Some("flow_trail_pipeline".into()),
            vertex: VertexState {
                shader: self.shader.clone(),
                // Vertex-pull: no vertex buffers; everything comes from
                // the storage buffers + vertex_index.
                buffers: vec![],
                ..default()
            },
            fragment: Some(FragmentState {
                shader: self.shader.clone(),
                targets: vec![Some(ColorTargetState {
                    format,
                    // Premultiplied with alpha 0 = pure additive glow.
                    blend: Some(BlendState::PREMULTIPLIED_ALPHA_BLENDING),
                    write_mask: ColorWrites::ALL,
                })],
                ..default()
            }),
            // Group 0: the standard 2D view uniform; group 1: the trail
            // uniform + storage buffers.
            layout: vec![
                self.mesh2d_pipeline.view_layout.clone(),
                self.trail_layout.clone(),
            ],
            primitive: PrimitiveState {
                topology: PrimitiveTopology::TriangleList,
                ..default()
            },
            depth_stencil: Some(DepthStencilState {
                format: CORE_2D_DEPTH_FORMAT,
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
}

/// Draws the static layer (the canvas equivalent).
struct DrawFlowGeometry;

impl<P: PhaseItem> RenderCommand<P> for DrawFlowGeometry {
    type Param = SRes<FlowBuffers>;
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
        if buffers.static_index_count == 0 {
            return RenderCommandResult::Success;
        }
        let (Some(vertices), Some(indices)) = (
            buffers.static_vertices.buffer(),
            buffers.static_indices.buffer(),
        ) else {
            return RenderCommandResult::Failure("flow static buffers not uploaded");
        };
        pass.set_vertex_buffer(0, vertices.slice(..));
        pass.set_index_buffer(indices.slice(..), IndexFormat::Uint32);
        pass.draw_indexed(0..buffers.static_index_count, 0, 0..1);
        RenderCommandResult::Success
    }
}

/// Draws the trail layer: bind the ring-buffer storage and let the vertex
/// shader expand every particle's glow ribbon — no vertex/index buffers.
struct DrawFlowTrailsPull;

impl<P: PhaseItem> RenderCommand<P> for DrawFlowTrailsPull {
    type Param = SRes<FlowBuffers>;
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
        if buffers.trail_count == 0 {
            return RenderCommandResult::Success;
        }
        let Some(bind_group) = &buffers.trail_bind_group else {
            return RenderCommandResult::Failure("flow trail bind group not prepared");
        };
        pass.set_bind_group(1, bind_group, &[]);
        let verts = buffers.trail_count * trail_verts_per_particle(buffers.trail_max_points);
        pass.draw(0..verts, 0..1);
        RenderCommandResult::Success
    }
}

type DrawFlow = (
    SetItemPipeline,
    SetMesh2dViewBindGroup<0>,
    DrawFlowGeometry,
);

type DrawFlowTrails = (
    SetItemPipeline,
    SetMesh2dViewBindGroup<0>,
    DrawFlowTrailsPull,
);

/// Queue the flow draws into every 2D view: the static layer, then the
/// trails over it (a fractionally higher sort key keeps the order
/// deterministic; no other experiment queues while flow owns the screen).
#[allow(clippy::too_many_arguments)]
fn queue_flow(
    transparent_draw_functions: Res<DrawFunctions<Transparent2d>>,
    flow_pipeline: Option<Res<FlowPipeline>>,
    trail_pipeline: Option<Res<FlowTrailPipeline>>,
    mut pipelines: ResMut<SpecializedRenderPipelines<FlowPipeline>>,
    mut trail_pipelines: ResMut<SpecializedRenderPipelines<FlowTrailPipeline>>,
    pipeline_cache: Res<PipelineCache>,
    buffers: Option<Res<FlowBuffers>>,
    mut transparent_render_phases: ResMut<ViewSortedRenderPhases<Transparent2d>>,
    views: Query<(&ExtractedView, &Msaa)>,
) {
    let (Some(flow_pipeline), Some(trail_pipeline), Some(buffers)) =
        (flow_pipeline, trail_pipeline, buffers)
    else {
        return;
    };
    if buffers.static_index_count == 0 && buffers.trail_count == 0 {
        return;
    }
    let draw_flow = transparent_draw_functions.read().id::<DrawFlow>();
    let draw_trails = transparent_draw_functions.read().id::<DrawFlowTrails>();

    for (view, msaa) in &views {
        let Some(transparent_phase) = transparent_render_phases.get_mut(&view.retained_view_entity)
        else {
            continue;
        };

        let key = Mesh2dPipelineKey::from_msaa_samples(msaa.samples())
            | Mesh2dPipelineKey::from_hdr(view.hdr)
            | Mesh2dPipelineKey::from_primitive_topology(PrimitiveTopology::TriangleList);

        let mut item = |draw_function, pipeline, sort_key, indexed| {
            transparent_phase.add(Transparent2d {
                // The draw is fully described by resources; no entity involved.
                entity: (Entity::PLACEHOLDER, MainEntity::from(Entity::PLACEHOLDER)),
                draw_function,
                pipeline,
                sort_key: FloatOrd(sort_key),
                batch_range: 0..1,
                extra_index: PhaseItemExtraIndex::None,
                extracted_index: usize::MAX,
                indexed,
            });
        };
        if buffers.static_index_count > 0 {
            let pipeline_id = pipelines.specialize(&pipeline_cache, &flow_pipeline, key);
            item(draw_flow, pipeline_id, 0.0, true);
        }
        if buffers.trail_count > 0 {
            let pipeline_id = trail_pipelines.specialize(&pipeline_cache, &trail_pipeline, key);
            item(draw_trails, pipeline_id, 0.001, false);
        }
    }
}

const FLOW_SHADER: &str = r"
#import bevy_sprite::mesh2d_view_bindings::view

struct Vertex {
    @location(0) pos: vec2<f32>,
    @location(1) color: vec4<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec4<f32>,
};

@vertex
fn vertex(vertex: Vertex) -> VertexOutput {
    var out: VertexOutput;
    out.clip_position = view.clip_from_world * vec4<f32>(vertex.pos, 0.0, 1.0);
    out.color = vertex.color;
    return out;
}

@fragment
fn fragment(in: VertexOutput) -> @location(0) vec4<f32> {
    return in.color;
}
";

// The trail layer's vertex-pull shader: expands every particle's trail
// ring buffer into a tapered additive glow ribbon plus a soft round head
// dot, entirely from `vertex_index`. One quad per segment; the glow's
// cross profile (full at the spine, zero at the edges) is computed per
// fragment from an interpolated profile coordinate — same tent the old
// two-quad expansion built from vertex alphas, at half the vertices (the
// layer is geometry-throughput-bound). Mirrors the Rust
// `TrailMeta`/`TrailParams` layouts and the sim's ring-buffer arithmetic
// (`trail_iter`). Segments past a trail's recorded length collapse to
// zero area (clamped samples); per-sample alpha falls off by fade_k just
// like the CPU original.
const FLOW_TRAIL_SHADER: &str = r"
#import bevy_sprite::mesh2d_view_bindings::view

struct TrailParams {
    origin: vec2<f32>,
    fade_k: f32,
    head_w: f32,
    tail_w: f32,
    glow_r: f32,
    max_points: u32,
    trail_cap: u32,
};

struct Meta {
    pos: vec2<f32>,
    head: u32,
    len: u32,
    color: u32,
    _pad: u32,
};

@group(1) @binding(0) var<uniform> params: TrailParams;
@group(1) @binding(1) var<storage, read> samples: array<vec2<f32>>;
@group(1) @binding(2) var<storage, read> metas: array<Meta>;

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec4<f32>,
    // Profile coordinate: |prof| = 0 full glow, 1 the transparent edge.
    // Segments use (±1, 0) across the ribbon (a linear tent per fragment);
    // the head dot (±1, ±1) corners (a radial cone — a round dot).
    @location(1) prof: vec2<f32>,
};

const SEG_VERTS: u32 = 6u;
const DOT_VERTS: u32 = 6u;

// Window coords -> world coords (the y-flip every layer uses).
fn world_of(p: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(p.x - params.origin.x, params.origin.y - p.y);
}

// The j-th newest trail sample of one particle, clamped to its length —
// the sim's ring-buffer read, ages walking back from `head`.
fn sample_at(particle: u32, j: u32, head: u32, len: u32) -> vec2<f32> {
    let jj = min(j, max(len, 1u) - 1u);
    let idx = (head + params.trail_cap - (jj % params.trail_cap)) % params.trail_cap;
    return samples[particle * params.trail_cap + idx];
}

@vertex
fn vertex(@builtin(vertex_index) vi: u32) -> VertexOutput {
    let segs = max(params.max_points, 2u) - 1u;
    let vpp = segs * SEG_VERTS + DOT_VERTS;
    let particle = vi / vpp;
    let r = vi % vpp;
    let m = metas[particle];
    let rgb = unpack4x8unorm(m.color).rgb;

    var world: vec2<f32>;
    var alpha: f32;
    var prof: vec2<f32>;

    if (r < segs * SEG_VERTS) {
        let s = r / SEG_VERTS;   // segment, newest end of the trail first
        let corner = r % SEG_VERTS;
        // One quad: corners are (end, side) — triangles (0,+)(1,+)(0,−)
        // and (1,+)(1,−)(0,−).
        var end: u32;
        var side: f32;
        switch corner {
            case 0u: { end = 0u; side = 1.0; }
            case 1u: { end = 1u; side = 1.0; }
            case 2u: { end = 0u; side = -1.0; }
            case 3u: { end = 1u; side = 1.0; }
            case 4u: { end = 1u; side = -1.0; }
            default: { end = 0u; side = -1.0; }
        }
        let j = s + end;
        let a = sample_at(particle, j, m.head, m.len);
        let older = sample_at(particle, j + 1u, m.head, m.len);
        let newer = sample_at(particle, max(j, 1u) - 1u, m.head, m.len);
        var dir = newer - older;
        if (dot(dir, dir) < 1e-12) {
            dir = vec2<f32>(0.0, 1.0);
        }
        let n = normalize(vec2<f32>(-dir.y, dir.x));
        let t = f32(j) / f32(max(params.max_points - 1u, 1u));
        let half_w = mix(params.head_w, params.tail_w, t);
        world = world_of(a + n * half_w * side);
        alpha = pow(params.fade_k, f32(j));
        prof = vec2<f32>(side, 0.0);
    } else {
        // The soft head dot at the live position: one quad, the cone
        // profile rounds it per fragment.
        let corner = r - segs * SEG_VERTS;
        var cx: f32;
        var cy: f32;
        switch corner {
            case 0u: { cx = -1.0; cy = -1.0; }
            case 1u: { cx = 1.0; cy = -1.0; }
            case 2u: { cx = -1.0; cy = 1.0; }
            case 3u: { cx = 1.0; cy = -1.0; }
            case 4u: { cx = 1.0; cy = 1.0; }
            default: { cx = -1.0; cy = 1.0; }
        }
        prof = vec2<f32>(cx, cy);
        world = world_of(m.pos + prof * params.glow_r);
        alpha = 1.0;
    }

    var out: VertexOutput;
    out.clip_position = view.clip_from_world * vec4<f32>(world, 0.0, 1.0);
    // Premultiplied additive: rgb scaled by alpha, alpha 0. The fragment
    // scales by the glow profile on top.
    out.color = vec4<f32>(rgb * alpha, 0.0);
    out.prof = prof;
    return out;
}

@fragment
fn fragment(in: VertexOutput) -> @location(0) vec4<f32> {
    // Tent across a segment (prof.y = 0), cone over the head dot. Alpha
    // stays 0 (additive), so scaling the whole color is exact.
    return in.color * max(0.0, 1.0 - length(in.prof));
}
";

#[cfg(test)]
mod tests {
    use super::*;

    /// Premultiplied encoding: normal blending keeps alpha, additive
    /// zeroes it; rgb is always scaled by alpha.
    #[test]
    fn premul_encodes_normal_and_additive() {
        let normal = premul([1.0, 0.5, 0.0], 0.5, false);
        assert_eq!(normal, [128, 64, 0, 128]);
        let additive = premul([1.0, 0.5, 0.0], 0.5, true);
        assert_eq!(additive, [128, 64, 0, 0]);
        let invisible = premul([1.0, 1.0, 1.0], 0.0, false);
        assert_eq!(invisible, [0, 0, 0, 0]);
    }

    /// hsv2rgb spot values, hand-computed through the Lua formula.
    #[test]
    fn hsv2rgb_matches_lua_spot_values() {
        // h=0 (m=0): (v, t, p) with f=0 → t=p=v(1-s).
        let [r, g, b] = hsv2rgb(0.0, 80.0, 95.0);
        assert!((r - 0.95).abs() < 1e-5);
        assert!((g - 0.19).abs() < 1e-5);
        assert!((b - 0.19).abs() < 1e-5);
        // h=120 (m=2): (p, v, t) with f=0 → green peak.
        let [r, g, b] = hsv2rgb(120.0, 100.0, 100.0);
        assert!((r - 0.0).abs() < 1e-5);
        assert!((g - 1.0).abs() < 1e-5);
        assert!((b - 0.0).abs() < 1e-5);
        // Achromatic.
        let [r, g, b] = hsv2rgb(200.0, 0.0, 50.0);
        assert!((r - 0.5).abs() < 1e-5 && (g - 0.5).abs() < 1e-5 && (b - 0.5).abs() < 1e-5);
    }

    /// Every palette stays inside [0,1]³ across the full angle range, and
    /// mono is the Lua's linear ramp, not an hsv ramp.
    #[test]
    fn palettes_are_bounded() {
        for palette in FlowPalette::ALL {
            for i in 0..=100 {
                let t = i as f32 / 100.0;
                let rgb = palette_srgb(palette, t);
                for c in rgb {
                    assert!((0.0..=1.0).contains(&c), "{palette:?} t={t}: {rgb:?}");
                }
            }
        }
        let m = palette_srgb(FlowPalette::Mono, 0.4);
        assert!((m[0] - 0.5).abs() < 1e-5 && m[0] == m[1] && m[1] == m[2]);
    }

    /// A feathered ribbon emits 4 vertices per point and 18 indices per
    /// segment.
    #[test]
    fn ribbon_vertex_and_index_counts() {
        let points = [
            Vec2::new(0.0, 0.0),
            Vec2::new(10.0, 0.0),
            Vec2::new(20.0, 5.0),
        ];
        let mut emitter = FlowEmitter::default();
        emitter.clear(Vec2::new(640.0, 400.0));
        emitter.ribbon_feathered(&points, false, |_| (1.0, 1.0, [1.0, 1.0, 1.0]));
        assert_eq!(emitter.vertices.len(), 3 * 4);
        assert_eq!(emitter.indices.len(), 2 * 18);
    }

    /// The vertex-pull layout math the trail shader mirrors: the GPU-side
    /// struct is 24 bytes (vec2-aligned), the cutoff-derived point count
    /// matches the CPU original's, and the per-particle vertex count
    /// follows it.
    #[test]
    fn trail_vertex_pull_layout() {
        assert_eq!(size_of::<TrailMeta>(), 24);
        // fade 0.15 → k 0.85 → all 20 ring samples stay visible.
        assert_eq!(trail_max_points(0.15), 20);
        // fade 0.4 → k 0.6 → ceil(ln 0.05 / ln 0.6) + 1 = 7.
        assert_eq!(trail_max_points(0.4), 7);
        // 19 segments × one 6-vertex quad + the 6-vertex head dot.
        assert_eq!(trail_verts_per_particle(20), 19 * 6 + 6);
        assert_eq!(trail_verts_per_particle(7), 6 * 6 + 6);
        // Degenerate minimum still draws the dot.
        assert_eq!(trail_verts_per_particle(1), 6 + 6);
    }

    /// The packed meta color is little-endian rgba with full alpha — what
    /// the shader's unpack4x8unorm expects.
    #[test]
    fn trail_meta_color_packs_little_endian() {
        let color = u32::from_le_bytes([10, 20, 30, 255]);
        assert_eq!(color & 0xFF, 10);
        assert_eq!((color >> 8) & 0xFF, 20);
        assert_eq!((color >> 16) & 0xFF, 30);
        assert_eq!(color >> 24, 255);
    }

    /// The gradient grid covers the viewport corners exactly and indexes
    /// every cell.
    #[test]
    fn gradient_grid_covers_the_viewport() {
        let field = FlowField {
            cols: 4,
            rows: 3,
            scale: 10.0,
            angles: vec![0.0; 12],
            dirs: vec![Vec2::X; 12],
        };
        let mut lut = FlowPaletteLut::default();
        lut.ensure(FlowPalette::Rainbow);
        let mut emitter = FlowEmitter::default();
        let bounds = Vec2::new(100.0, 60.0);
        emitter.clear(bounds / 2.0);
        emit_gradient(&mut emitter, &field, &lut, bounds, 1.0);
        assert_eq!(emitter.vertices.len(), 5 * 4);
        assert_eq!(emitter.indices.len(), 4 * 3 * 6);
        // Window corners → world corners (y-flipped around the center).
        let first = emitter.vertices.first().unwrap().pos;
        let last = emitter.vertices.last().unwrap().pos;
        assert_eq!(first, [-50.0, 30.0]);
        assert_eq!(last, [50.0, -30.0]);
    }
}
