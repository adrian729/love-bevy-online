//! Fish rendering: rebuild the scene's triangles every frame from the
//! spine, in the original's painter order — food, lateral fins, tail, body,
//! dorsal fin, eyes. Each spline-smoothed part is filled (the original's
//! `love.math.triangulate`) and outlined in white (`love.graphics.line`).
//!
//! Dynamic vector geometry is the right tool here (unlike the flock's
//! per-boid instancing): the fish is a handful of organic polygons whose
//! shape changes every frame, and triangle order gives the exact LÖVE draw
//! order in one draw call. The triangles go to the GPU through a custom
//! 12-byte vertex (position + unorm color) and persistent buffers — the
//! `Assets<Mesh>` path re-interleaves and re-copies a meshes' full data
//! every frame, which dominated the frame at thousands of fish.
//!
//! Geometry is computed in the sim's window coordinates and y-flipped to
//! world coordinates at vertex emission.

use std::f32::consts::{PI, TAU};
use std::sync::OnceLock;

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
use bevy::render::render_resource::{
    BlendState, BufferUsages, ColorTargetState, ColorWrites, CompareFunction, DepthBiasState,
    DepthStencilState, FragmentState, IndexFormat, MultisampleState, PipelineCache,
    PrimitiveState, PrimitiveTopology, RawBufferVec, RenderPipelineDescriptor,
    SpecializedRenderPipeline, SpecializedRenderPipelines, StencilFaceState, StencilState,
    TextureFormat, VertexAttribute, VertexFormat, VertexState, VertexStepMode,
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

use super::sim::{Fish, FishGame, Fishes, FishSimSet, JOINTS, orthogonal};
use crate::app::{SimBounds, sim_active};
use crate::experiments::{CurrentExperiment, ExperimentId, experiment_active};

/// Spline detail per part, from lib/fish.lua's `render` calls.
const BODY_DETAIL: f32 = 500.0;
const DORSAL_DETAIL: f32 = 200.0;
const TAIL_DETAIL: f32 = 100.0;

/// One triangle vertex: world position + linear unorm color. 12 bytes —
/// bandwidth is what scales with fish count.
#[derive(Clone, Copy, Pod, Zeroable)]
#[repr(C)]
struct FishVertex {
    pos: [f32; 2],
    color: [u8; 4],
}

/// The LÖVE colors (sRGB values, gammacorrect off), pre-converted to
/// linear unorm once.
struct Palette {
    body: [u8; 4],
    fin: [u8; 4],
    /// Flake-food tones: the original's amber plus an orange and a
    /// toasted brown, like a pinch from a tub of fish flakes.
    flakes: [[u8; 4]; 3],
    white: [u8; 4],
    eye: [u8; 4],
    pupil: [u8; 4],
}

fn linear_u8(r: f32, g: f32, b: f32) -> [u8; 4] {
    let c = Color::srgb(r, g, b).to_linear();
    [
        (c.red * 255.0).round() as u8,
        (c.green * 255.0).round() as u8,
        (c.blue * 255.0).round() as u8,
        255,
    ]
}

fn palette() -> &'static Palette {
    static PALETTE: OnceLock<Palette> = OnceLock::new();
    PALETTE.get_or_init(|| Palette {
        body: linear_u8(58.0 / 255.0, 124.0 / 255.0, 165.0 / 255.0),
        fin: linear_u8(129.0 / 255.0, 195.0 / 255.0, 215.0 / 255.0),
        flakes: [
            linear_u8(219.0 / 255.0, 182.0 / 255.0, 0.0),
            linear_u8(228.0 / 255.0, 134.0 / 255.0, 36.0 / 255.0),
            linear_u8(178.0 / 255.0, 98.0 / 255.0, 30.0 / 255.0),
        ],
        white: [255, 255, 255, 255],
        eye: linear_u8(0.7, 0.7, 0.7),
        pupil: linear_u8(0.05, 0.05, 0.05),
    })
}

pub fn plugin(app: &mut App) {
    // Shaders only exist in the main world; hand the handle into the
    // render app.
    let shader = app
        .world_mut()
        .resource_mut::<Assets<Shader>>()
        .add(Shader::from_wgsl(FISH_SHADER, file!()));

    app.init_resource::<FishDrawData>().add_systems(
        Update,
        (
            rebuild_geometry
                .after(FishSimSet)
                .run_if(experiment_active(ExperimentId::Fish))
                // Not in Options: the last frame stays visible, frozen
                // behind the popup like the original.
                .run_if(sim_active),
            clear_when_inactive,
        ),
    );

    app.sub_app_mut(RenderApp)
        .insert_resource(FishShader(shader))
        .add_render_command::<Transparent2d, DrawFish>()
        .init_resource::<SpecializedRenderPipelines<FishPipeline>>()
        .add_systems(
            RenderStartup,
            (init_fish_pipeline.after(init_mesh_2d_pipeline), |mut commands: Commands| {
                commands.init_resource::<FishBuffers>();
            }),
        )
        .add_systems(ExtractSchedule, extract_fish)
        .add_systems(
            Render,
            (
                prepare_fish.in_set(RenderSystems::PrepareResources),
                queue_fish.in_set(RenderSystems::Queue),
            ),
        );
}

/// Main-world handoff: this frame's triangles, swapped out of the build
/// emitter (no copy) and picked up by the render world's extract.
#[derive(Resource, Default)]
struct FishDrawData {
    vertices: Vec<FishVertex>,
    indices: Vec<u32>,
}

/// Reused per-worker geometry buffers for the parallel build.
#[derive(Default)]
struct ChunkBuffers(Vec<(Emitter, Scratch)>);

/// Per-frame rebuild: emit food then every fish into the scratch buffers
/// and hand them to the renderer. Many fish (the perf harness / a future
/// school) build their geometry in parallel on the compute pool — each
/// fish's part list is independent; only the final concatenation is
/// ordered.
fn rebuild_geometry(
    fishes: Res<Fishes>,
    game: Res<FishGame>,
    bounds: Res<SimBounds>,
    mut data: ResMut<FishDrawData>,
    mut emitter: Local<Emitter>,
    mut scratch: Local<Scratch>,
    mut chunks: Local<ChunkBuffers>,
) {
    let origin = bounds.0 / 2.0;
    emitter.clear(origin);

    // The food draws first (minigames/fish.lua draw order). Any fish can
    // eat it — one or a whole school — so its footprint tracks the
    // biggest fish (the original dot's radius rule).
    if let Some(max_scale) = fishes.0.iter().map(|fish| fish.scale).reduce(f32::max) {
        let r = (max_scale * 20.0).min(10.0);
        emit_food(&mut emitter, game.food, r);
    }

    let count = fishes.0.len();
    if count < 16 {
        for fish in &fishes.0 {
            emit_fish(&mut emitter, &mut scratch, fish);
        }
        // Swap, don't copy: the emitter inherits last frame's capacity back.
        std::mem::swap(&mut data.vertices, &mut emitter.vertices);
        std::mem::swap(&mut data.indices, &mut emitter.indices);
    } else {
        // A few chunks per worker: pool threads are shared with other
        // systems, so coarser chunks straggle the frame (measured: one
        // chunk per worker cost ~25%).
        let pool = ComputeTaskPool::get();
        let chunk_count = (pool.thread_num().max(1) * 3).min(count);
        let chunk_size = count.div_ceil(chunk_count);
        let buffers = &mut chunks.0;
        buffers.resize_with(chunk_count, Default::default);
        pool.scope(|scope| {
            for (buffer, chunk) in buffers.iter_mut().zip(fishes.0.chunks(chunk_size)) {
                scope.spawn(async move {
                    let (emitter, scratch) = buffer;
                    emitter.clear(origin);
                    for fish in chunk {
                        emit_fish(emitter, scratch, fish);
                    }
                });
            }
        });
        // Concatenate in parallel too: at thousands of fish this is tens of
        // megabytes, and a serial merge was a measurable slice of the frame.
        // Each chunk copies into its own disjoint slice of the output,
        // offsetting its indices on the way. The main emitter (the food,
        // emitted above) is the prefix: copied verbatim — its indices are
        // already 0-based — while every fish chunk's indices shift past
        // its vertices via the `base` starting value.
        let total_vertices: usize =
            emitter.vertices.len() + buffers.iter().map(|(e, _)| e.vertices.len()).sum::<usize>();
        let total_indices: usize =
            emitter.indices.len() + buffers.iter().map(|(e, _)| e.indices.len()).sum::<usize>();
        let FishDrawData { vertices, indices } = &mut *data;
        vertices.resize(total_vertices, FishVertex::zeroed());
        indices.resize(total_indices, 0);
        pool.scope(|scope| {
            let (vertex_prefix, mut vertex_rest) = vertices.split_at_mut(emitter.vertices.len());
            let (index_prefix, mut index_rest) = indices.split_at_mut(emitter.indices.len());
            vertex_prefix.copy_from_slice(&emitter.vertices);
            index_prefix.copy_from_slice(&emitter.indices);
            let mut base = emitter.vertices.len() as u32;
            for (chunk_emitter, _) in buffers.iter() {
                let (vertex_slice, rest) = vertex_rest.split_at_mut(chunk_emitter.vertices.len());
                vertex_rest = rest;
                let (index_slice, rest) = index_rest.split_at_mut(chunk_emitter.indices.len());
                index_rest = rest;
                let chunk_base = base;
                base += chunk_emitter.vertices.len() as u32;
                scope.spawn(async move {
                    vertex_slice.copy_from_slice(&chunk_emitter.vertices);
                    for (dst, src) in index_slice.iter_mut().zip(&chunk_emitter.indices) {
                        *dst = src + chunk_base;
                    }
                });
            }
        });
    }
}

/// Drop the triangles (and the fish) when another experiment takes over;
/// returning re-spawns fresh via the sim's restart path.
fn clear_when_inactive(
    current: Res<CurrentExperiment>,
    mut data: ResMut<FishDrawData>,
    mut fishes: ResMut<Fishes>,
) {
    if !current.is_changed() || current.0 == ExperimentId::Fish {
        return;
    }
    fishes.0.clear();
    data.vertices.clear();
    data.indices.clear();
}

// ---------------------------------------------------------------------------
// The fish's parts — lib/fish.lua's draw functions, building point lists in
// window coordinates.

/// Reused per-part point buffers.
#[derive(Default)]
struct Scratch {
    left: Vec<Vec2>,
    right: Vec<Vec2>,
    shape: Vec<Vec2>,
    spline: Vec<Vec2>,
    cleaned: Vec<Vec2>,
    simplified: Vec<Vec2>,
}

impl Scratch {
    fn begin_part(&mut self) {
        self.left.clear();
        self.right.clear();
        self.shape.clear();
    }

    /// left + reversed right, the closed-ish outline every part uses.
    fn join_sides(&mut self) {
        self.shape.extend_from_slice(&self.left);
        self.shape.extend(self.right.iter().rev());
    }

    /// Spline the shape, dedupe, then fill + outline it (`paintPolygon`).
    fn paint(&mut self, emitter: &mut Emitter, detail: f32, fill: [u8; 4]) {
        spline_v2(&self.shape, detail, &mut self.spline);
        cleanup(&self.spline, &mut self.cleaned);
        simplify(&self.cleaned, 0.25, &mut self.simplified);
        // triangulate needs at least 3 vertices; skip degenerate shapes.
        if self.simplified.len() < 3 {
            return;
        }
        emitter.fill_polygon(&self.simplified, fill);
        // The outline polyline is NOT auto-closed (love.graphics.line):
        // the body closes because its shape starts and ends at the nose
        // tip; the tail's base edge stays open, hidden under the body.
        emitter.stroke_polyline(&self.simplified, palette().white, false);
    }
}

/// A pinch of flake food instead of the original's plain amber dot: a
/// loose scatter of small flat ellipses in warm tones. The arrangement
/// hashes off the drop position — stable while the food sits, fresh on
/// every respawn. No linework: white rings swamped the warm fills at this
/// size and the scatter read as a smudge.
fn emit_food(emitter: &mut Emitter, center: Vec2, r: f32) {
    // PCG hash seeded from the drop position's bits.
    let mut state =
        center.x.to_bits().wrapping_mul(0x9E37_79B9) ^ center.y.to_bits().rotate_left(16);
    let mut rand = move || {
        state = state.wrapping_mul(747_796_405).wrapping_add(2_891_336_453);
        let word = ((state >> ((state >> 28) + 4)) ^ state).wrapping_mul(277_803_737);
        ((word >> 22) ^ word) as f32 * (1.0 / u32::MAX as f32)
    };
    // The scatter radius and flake size are floored in pixels so even the
    // game-start drop (r = 1 while every fish is small) shows a visible
    // sprinkle, not a lone sub-pixel speck.
    let spread = r.max(4.0);
    let flakes = ((r * 1.2) as usize).clamp(4, 10);
    let spin = rand() * TAU;
    for i in 0..flakes {
        // Stratified angles so the sprinkle spreads instead of clumping.
        let angle = spin + (i as f32 + 0.7 * rand()) * (TAU / flakes as f32);
        let dist = spread * (0.35 + 0.85 * rand());
        let pos = center + dist * Vec2::from_angle(angle);
        let rx = (r * (0.24 + 0.12 * rand())).max(1.2);
        let ry = rx * (0.55 + 0.30 * rand());
        let rot = rand() * TAU;
        let tone = palette().flakes[(rand() * 3.0) as usize % 3];
        emitter.fill_ellipse(pos, rx, ry, rot, tone);
    }
}

fn emit_fish(emitter: &mut Emitter, scratch: &mut Scratch, fish: &Fish) {
    emit_lateral_fins(emitter, fish);
    emit_tail(emitter, scratch, fish);
    emit_body(emitter, scratch, fish);
    emit_dorsal_fin(emitter, scratch, fish);
    emit_eyes(emitter, fish);
}

/// Two ellipses at joint 3's flanks, rotated ±60° off the spine.
fn emit_lateral_fins(emitter: &mut Emitter, fish: &Fish) {
    let j = &fish.spine.joints;
    let r_x = fish.scale * 75.0;
    let r_y = r_x / 2.0;
    let angle = (j[1] - j[2]).to_angle();
    let (right, left) = orthogonal(j[2], j[3], fish.body_width(2));
    for (side, sign) in [(left, 1.0), (right, -1.0)] {
        let rot = angle + sign * PI / 3.0;
        emitter.fill_ellipse(side, r_x, r_y, rot, palette().fin);
        emitter.stroke_ellipse(side, r_x, r_y, rot, palette().white);
    }
}

/// The tail fin: a 5-point shape over the last joints, splined. Its base
/// edge is left open (the body draws over it).
fn emit_tail(emitter: &mut Emitter, scratch: &mut Scratch, fish: &Fish) {
    let j = &fish.spine.joints;
    let n = JOINTS; // Lua 1-based: joints[n-2], [n-1], [n] = indices n-3, n-2, n-1
    scratch.begin_part();

    let (right_side, left_side) = orthogonal(j[n - 3], j[n - 2], 0.8 * fish.body_width(n - 2));
    scratch.left.push(left_side);
    scratch.right.push(right_side);

    // Orthogonal taken from the last joint *backwards*, so the pair is
    // mirrored relative to the body's convention — as in the original.
    let (tail_1, tail_2) = orthogonal(j[n - 1], j[n - 2], 1.2 * fish.body_width(n - 1));
    scratch.left.push(tail_1);
    scratch.right.push(tail_2);

    let tail_dir = j[n - 1] - j[n - 2];
    let tail_end =
        set_magnitude_or_zero(tail_dir, fish.body_width(n - 1) + fish.spine.link) + j[n - 2];
    scratch.left.push(tail_end);

    scratch.join_sides();
    scratch.paint(emitter, TAIL_DETAIL, palette().fin);
}

/// The main body: a pointed nose cone, then orthogonal flanks down the
/// spine, splined into one closed outline.
fn emit_body(emitter: &mut Emitter, scratch: &mut Scratch, fish: &Fish) {
    let j = &fish.spine.joints;
    let bw0 = fish.body_width(0);
    scratch.begin_part();

    // Nose tip, extended ahead of the head; first point of BOTH sides, so
    // the outline closes back onto it.
    let front = set_magnitude_or_zero(j[0] - j[1], bw0 + fish.spine.link) + j[1];
    scratch.left.push(front);
    scratch.right.push(front);
    let angle = (j[0] - j[1]).to_angle();
    for (side, sign) in [(&mut scratch.left, -1.0), (&mut scratch.right, 1.0)] {
        side.push(j[0] + Vec2::from_angle(angle + sign * PI / 8.0) * bw0);
        side.push(j[0] + Vec2::from_angle(angle + sign * PI / 4.0) * bw0);
    }

    // Flanks at each joint's width, head to joint 10 (the tail fin covers
    // the rest).
    for i in 0..JOINTS - 2 {
        let (right_side, left_side) = orthogonal(j[i], j[i + 1], fish.body_width(i));
        scratch.left.push(left_side);
        scratch.right.push(right_side);
    }

    scratch.join_sides();
    scratch.paint(emitter, BODY_DETAIL, palette().body);
}

/// A thin sliver along joints 3..7 that bows out when the fish bends.
fn emit_dorsal_fin(emitter: &mut Emitter, scratch: &mut Scratch, fish: &Fish) {
    let j = &fish.spine.joints;
    // Lua 1-based start_idx = 3, end_idx = 7.
    let (start, end) = (2, 6);
    scratch.begin_part();

    scratch.left.push(j[start]);
    scratch.right.push(j[start]);
    let (side_r, side_l) = orthogonal(j[start], j[start + 1], 0.02 * fish.body_width(start));
    scratch.left.push(side_l);
    scratch.right.push(side_r);

    for i in start + 1..end {
        let (side_r, side_l) = orthogonal(j[i], j[i + 1], 0.1 * fish.body_width(i));
        scratch.left.push(side_l);
        scratch.right.push(side_r);
    }

    let (side_r, side_l) = orthogonal(j[end], j[end + 1], 0.03 * fish.body_width(end));
    scratch.left.push(side_l);
    scratch.right.push(side_r);
    scratch.left.push(j[end]);
    scratch.right.push(j[end]);

    scratch.join_sides();
    scratch.paint(emitter, DORSAL_DETAIL, palette().fin);
}

/// Outward-looking eyes: gray whites with darker pupils seated slightly
/// further out (no outlines).
fn emit_eyes(emitter: &mut Emitter, fish: &Fish) {
    let j = &fish.spine.joints;
    let eye_size = 24.0 * fish.scale;
    let (eye_right, eye_left) = orthogonal(j[0], j[1], fish.body_width(0) - 0.6 * eye_size);
    emitter.fill_circle(eye_left, eye_size, palette().eye);
    emitter.fill_circle(eye_right, eye_size, palette().eye);
    let pupil_size = 18.0 * fish.scale;
    let (pupil_left, pupil_right) = orthogonal(j[0], j[1], fish.body_width(0) - 0.4 * eye_size);
    emitter.fill_circle(pupil_left, pupil_size, palette().pupil);
    emitter.fill_circle(pupil_right, pupil_size, palette().pupil);
}

fn set_magnitude_or_zero(v: Vec2, m: f32) -> Vec2 {
    let len = v.length();
    if len > 1e-12 { v * (m / len) } else { Vec2::ZERO }
}

// ---------------------------------------------------------------------------
// Splines — lib/splines.lua's `renderV2` (the fish renders with
// `type = 'v2'`, NOT Catmull-Rom): a Hermite spline with tangents
// 0.5·(p2−p0), whose per-segment point count shrinks as the three points
// around it approach collinearity.

/// cos² of the turn angle at p2 — 0 for turns past 90°. 0/0 (duplicate
/// points) yields NaN, which poisons the segment into emitting nothing,
/// exactly like the Lua.
fn colinearity(p1: Vec2, p2: Vec2, p3: Vec2) -> f32 {
    let u = p2 - p1;
    let v = p3 - p2;
    let udv = u.dot(v);
    if udv < 0.0 {
        return 0.0;
    }
    (udv * udv) / (u.length_squared() * v.length_squared())
}

fn spline_v2(points: &[Vec2], detail: f32, out: &mut Vec<Vec2>) {
    out.clear();
    // Fewer than 4 points: returned as-is (Splines:render's early out).
    if points.len() < 4 {
        out.extend_from_slice(points);
        return;
    }
    for i in 0..points.len() - 1 {
        let p1 = points[i];
        let p2 = points[i + 1];
        let p0 = i.checked_sub(1).map(|i| points[i]);
        let p3 = points.get(i + 2).copied();

        let mut t1 = Vec2::ZERO;
        let mut t2 = Vec2::ZERO;
        let c1 = p0.map(|p0| {
            t1 = 0.5 * (p2 - p0);
            colinearity(p0, p1, p2)
        });
        let c2 = p3.map(|p3| {
            t2 = 0.5 * (p3 - p1);
            colinearity(p1, p2, p3)
        });
        let colin = match (c1, c2) {
            (Some(a), Some(b)) => (a + b) / 2.0,
            (Some(a), None) | (None, Some(a)) => a,
            (None, None) => 0.0,
        };

        let rdetail = detail * (1.0 - colin);
        if !rdetail.is_finite() {
            // Duplicate-point NaN: the Lua loop emits nothing usable here.
            continue;
        }
        if rdetail <= 0.0 {
            // Perfectly straight: Lua emits one NaN point that the dedupe
            // drops; the segment start is already covered by its
            // predecessor, so emitting p1 is equivalent.
            out.push(p1);
            continue;
        }
        // At most ~1.1 samples per pixel of chord: the chord error at the
        // fish's tightest curvature is hundredths of a pixel, far below
        // the AA feather, so the rendered outline stays identical while
        // skipping the original's thousands of immediately-discarded
        // evaluations (Lua asks for up to 500 per segment regardless of
        // how small the segment is on screen).
        let rdetail = rdetail.min((p1.distance(p2) * 1.1).max(4.0));
        // Lua's inclusive `for j = 0, rdetail` with a possibly fractional
        // bound: j runs to floor(rdetail).
        let last = rdetail.floor() as i32;
        for j in 0..=last {
            let s = j as f32 / rdetail;
            let s2 = s * s;
            let s3 = s2 * s;
            let h1 = 2.0 * s3 - 3.0 * s2 + 1.0;
            let h2 = -2.0 * s3 + 3.0 * s2;
            let h3 = s3 - 2.0 * s2 + s;
            let h4 = s3 - s2;
            out.push(h1 * p1 + h2 * p2 + h3 * t1 + h4 * t2);
        }
        if rdetail.ceil() > rdetail {
            out.push(p2);
        }
    }
}

/// `cleanupPoints`: keep the first point, then only points further than
/// 0.5px from the last kept one. Bounds the vertex count by arc length —
/// at small fish scales most spline points collapse away.
fn cleanup(points: &[Vec2], out: &mut Vec<Vec2>) {
    out.clear();
    let Some(&first) = points.first() else {
        return;
    };
    out.push(first);
    for &point in points {
        if point.distance(*out.last().unwrap()) > 0.5 {
            out.push(point);
        }
    }
}

/// Greedy sub-pixel run merging: drop points whose removal keeps every
/// intermediate point within `eps` of the replacement chord. At eps well
/// under half a pixel the stroked/filled result is indistinguishable —
/// the spline outline is smooth, so straight-ish runs collapse to a few
/// vertices and triangle counts drop severalfold.
fn simplify(points: &[Vec2], eps: f32, out: &mut Vec<Vec2>) {
    out.clear();
    let n = points.len();
    if n <= 2 {
        out.extend_from_slice(points);
        return;
    }
    // Bounded merge window: the re-scan per grown candidate is quadratic in
    // run length, and near-straight runs (the dorsal sliver) otherwise merge
    // half their points in one go. A couple of extra vertices on long
    // straights is invisible; the cost bound is what matters.
    const MAX_WINDOW: usize = 16;
    let mut anchor = 0;
    out.push(points[0]);
    while anchor < n - 1 {
        // Furthest endpoint whose chord keeps all skipped points inside eps.
        let mut end = anchor + 1;
        'grow: for candidate in anchor + 2..n.min(anchor + 2 + MAX_WINDOW) {
            let a = points[anchor];
            let b = points[candidate];
            let ab = b - a;
            let len_sq = ab.length_squared();
            for &p in &points[anchor + 1..candidate] {
                // Perpendicular distance from p to chord a→b.
                let t = if len_sq > 0.0 {
                    ((p - a).dot(ab) / len_sq).clamp(0.0, 1.0)
                } else {
                    0.0
                };
                if p.distance_squared(a + ab * t) > eps * eps {
                    break 'grow;
                }
            }
            end = candidate;
        }
        out.push(points[end]);
        anchor = end;
    }
}

// ---------------------------------------------------------------------------
// The triangle emitter: every primitive appends vertices (window coords,
// y-flipped to world here) and indices; triangle order = draw order.

#[derive(Default)]
struct Emitter {
    vertices: Vec<FishVertex>,
    indices: Vec<u32>,
    /// Half the window, for the window→world flip.
    origin: Vec2,
    // Pooled per-primitive scratch: at thousands of fish, per-part
    // allocations were a measurable slice of the frame.
    rows: Vec<[u32; 4]>,
    ring: Vec<Vec2>,
    ear: EarScratch,
}

impl Emitter {
    fn clear(&mut self, origin: Vec2) {
        self.vertices.clear();
        self.indices.clear();
        self.origin = origin;
    }

    fn vertex(&mut self, p: Vec2, color: [u8; 4]) -> u32 {
        let index = self.vertices.len() as u32;
        // Window (top-left, y-down) → world (centered, y-up).
        self.vertices.push(FishVertex {
            pos: [p.x - self.origin.x, self.origin.y - p.y],
            color,
        });
        index
    }

    fn triangle(&mut self, a: u32, b: u32, c: u32) {
        self.indices.extend([a, b, c]);
    }

    /// Fill a simple polygon by ear clipping — `love.math.triangulate`.
    fn fill_polygon(&mut self, points: &[Vec2], color: [u8; 4]) {
        let mut n = points.len();
        // A duplicated closing point (the body outline ends where it
        // starts) would make a degenerate ear; drop it for the fill only.
        while n >= 2 && points[0].distance(points[n - 1]) < 1e-3 {
            n -= 1;
        }
        if n < 3 {
            return;
        }
        let base = self.vertices.len() as u32;
        for &p in &points[..n] {
            self.vertex(p, color);
        }
        let mut ear = std::mem::take(&mut self.ear);
        ear_clip(&points[..n], &mut ear, |a, b, c| {
            self.indices.extend([base + a, base + b, base + c]);
        });
        self.ear = ear;
    }

    /// Stroke a polyline like LÖVE's "smooth" line style — a solid core
    /// at full alpha with a feather fading to transparent on each side —
    /// but slimmer than the original's width-1 profile (±0.5px core,
    /// ±1.5px feather): a deliberate deviation, the user found the ported
    /// outlines too thick over the water.
    fn stroke_polyline(&mut self, points: &[Vec2], color: [u8; 4], closed: bool) {
        let n = points.len();
        if n < 2 {
            return;
        }
        let transparent = [color[0], color[1], color[2], 0];
        let dir = |a: Vec2, b: Vec2| (b - a).normalize_or(Vec2::X);
        let mut rows = std::mem::take(&mut self.rows);
        rows.clear();
        for i in 0..n {
            // Per-point normal: average of the adjacent segment normals.
            let incoming = if i > 0 {
                Some(dir(points[i - 1], points[i]))
            } else if closed {
                Some(dir(points[n - 1], points[i]))
            } else {
                None
            };
            let outgoing = if i + 1 < n {
                Some(dir(points[i], points[i + 1]))
            } else if closed {
                Some(dir(points[i], points[0]))
            } else {
                None
            };
            let tangent = match (incoming, outgoing) {
                (Some(a), Some(b)) => (a + b).normalize_or(a),
                (Some(a), None) | (None, Some(a)) => a,
                (None, None) => Vec2::X,
            };
            let normal = tangent.perp();
            let p = points[i];
            rows.push([
                self.vertex(p + 1.1 * normal, transparent),
                self.vertex(p + 0.35 * normal, color),
                self.vertex(p - 0.35 * normal, color),
                self.vertex(p - 1.1 * normal, transparent),
            ]);
        }
        let segments = if closed { n } else { n - 1 };
        for i in 0..segments {
            let a = rows[i];
            let b = rows[(i + 1) % n];
            for k in 0..3 {
                self.triangle(a[k], b[k], b[k + 1]);
                self.triangle(a[k], b[k + 1], a[k + 1]);
            }
        }
        self.rows = rows;
    }

    fn fill_circle(&mut self, center: Vec2, r: f32, color: [u8; 4]) {
        self.fill_ellipse(center, r, r, 0.0, color);
    }

    fn fill_ellipse(&mut self, center: Vec2, rx: f32, ry: f32, rot: f32, color: [u8; 4]) {
        let segments = ellipse_segments(rx, ry);
        let mut walk = EllipseWalk::new(rx, ry, rot, segments);
        let center_index = self.vertex(center, color);
        let first = self.vertex(center + walk.next_point(), color);
        let mut previous = first;
        for _ in 1..segments {
            let next = self.vertex(center + walk.next_point(), color);
            self.triangle(center_index, previous, next);
            previous = next;
        }
        self.triangle(center_index, previous, first);
    }

    fn stroke_ellipse(&mut self, center: Vec2, rx: f32, ry: f32, rot: f32, color: [u8; 4]) {
        let segments = ellipse_segments(rx, ry);
        let mut walk = EllipseWalk::new(rx, ry, rot, segments);
        let mut ring = std::mem::take(&mut self.ring);
        ring.clear();
        ring.extend((0..segments).map(|_| center + walk.next_point()));
        // LÖVE's circle/ellipse 'line' closes the loop.
        self.stroke_polyline(&ring, color, true);
        self.ring = ring;
    }
}

/// Marches around an ellipse with a rotation recurrence — two `sin_cos`
/// calls total instead of one per point (the round parts were a visible
/// slice of the per-fish cost at thousands of fish).
struct EllipseWalk {
    rx: f32,
    ry: f32,
    rot: Vec2,
    dir: Vec2,
    step: Vec2,
}

impl EllipseWalk {
    fn new(rx: f32, ry: f32, rot: f32, segments: usize) -> Self {
        Self {
            rx,
            ry,
            rot: Vec2::from_angle(rot),
            dir: Vec2::X,
            step: Vec2::from_angle(TAU / segments as f32),
        }
    }

    fn next_point(&mut self) -> Vec2 {
        let point = self
            .rot
            .rotate(Vec2::new(self.rx * self.dir.x, self.ry * self.dir.y));
        self.dir = self.step.rotate(self.dir);
        point
    }
}

/// LÖVE 11's `calculateEllipsePoints`: enough segments that the chord
/// error stays under half a pixel.
fn ellipse_segments(rx: f32, ry: f32) -> usize {
    let r = rx.max(ry).max(0.3);
    let arg = (1.0 - 0.5 / r).clamp(-1.0, 1.0);
    let points = (PI / arg.acos()).ceil();
    if points.is_finite() {
        (points as usize).max(8)
    } else {
        8
    }
}

/// Reused ear-clipping tables.
#[derive(Default)]
struct EarScratch {
    next: Vec<u32>,
    prev: Vec<u32>,
    reflex: Vec<bool>,
    reflex_list: Vec<u32>,
}

/// Ear-clipping triangulation of a simple polygon (either winding).
/// Degenerate input degrades to a fan instead of failing — the original
/// would throw from `love.math.triangulate`, but it never does in play.
///
/// Linked-list walk with a reflex-vertex list: only reflex vertices can
/// invalidate an ear, and unlinking a vertex can only turn its neighbours
/// convex, never reflex. Convex outlines (a straight fish) triangulate in
/// O(n); a bent fish pays only for its few inflection points.
fn ear_clip(points: &[Vec2], scratch: &mut EarScratch, mut emit: impl FnMut(u32, u32, u32)) {
    let n = points.len();
    if n < 3 {
        return;
    }
    // Shoelace orientation.
    let mut doubled_area = 0.0;
    for i in 0..n {
        let a = points[i];
        let b = points[(i + 1) % n];
        doubled_area += a.x * b.y - b.x * a.y;
    }
    let sign = if doubled_area > 0.0 { 1.0 } else { -1.0 };

    let EarScratch {
        next,
        prev,
        reflex,
        reflex_list,
    } = scratch;
    let n32 = n as u32;
    next.clear();
    next.extend((0..n32).map(|i| (i + 1) % n32));
    prev.clear();
    prev.extend((0..n32).map(|i| (i + n32 - 1) % n32));
    let convex = |prev: &[u32], next: &[u32], i: u32| {
        let (a, b, c) = (
            points[prev[i as usize] as usize],
            points[i as usize],
            points[next[i as usize] as usize],
        );
        sign * (b - a).perp_dot(c - a) > 0.0
    };
    reflex.clear();
    reflex.extend((0..n32).map(|i| !convex(prev, next, i)));
    reflex_list.clear();
    reflex_list.extend((0..n32).filter(|&i| reflex[i as usize]));

    let mut remaining = n;
    let mut current = 0u32;
    let mut stalled = 0;
    while remaining > 3 {
        let (ia, ib, ic) = (prev[current as usize], current, next[current as usize]);
        let is_ear = !reflex[ib as usize] && {
            let (a, b, c) = (
                points[ia as usize],
                points[ib as usize],
                points[ic as usize],
            );
            !reflex_list.iter().any(|&j| {
                reflex[j as usize]
                    && j != ia
                    && j != ic
                    && next[prev[j as usize] as usize] == j // still linked
                    && point_in_triangle(points[j as usize], a, b, c)
            })
        };
        if is_ear {
            emit(ia, ib, ic);
            next[ia as usize] = ic;
            prev[ic as usize] = ia;
            remaining -= 1;
            stalled = 0;
            // Unlinking can only relax the neighbours toward convex.
            for i in [ia, ic] {
                if reflex[i as usize] && convex(prev, next, i) {
                    reflex[i as usize] = false;
                }
            }
            current = ic;
            continue;
        }
        current = next[current as usize];
        stalled += 1;
        if stalled > remaining {
            // No ear found (self-intersecting outline): fan the rest.
            let start = current;
            let mut walk = next[start as usize];
            while next[walk as usize] != start {
                emit(start, walk, next[walk as usize]);
                walk = next[walk as usize];
            }
            return;
        }
    }
    emit(prev[current as usize], current, next[current as usize]);
}

fn point_in_triangle(p: Vec2, a: Vec2, b: Vec2, c: Vec2) -> bool {
    let d1 = (b - a).perp_dot(p - a);
    let d2 = (c - b).perp_dot(p - b);
    let d3 = (a - c).perp_dot(p - c);
    let has_neg = d1 < 0.0 || d2 < 0.0 || d3 < 0.0;
    let has_pos = d1 > 0.0 || d2 > 0.0 || d3 > 0.0;
    !(has_neg && has_pos)
}

// ---------------------------------------------------------------------------
// Render world: persistent vertex/index buffers, a pipeline over the
// standard 2D view uniform, and one indexed draw in the transparent phase.
// Same shape as the flock's custom path (see flock/render.rs), minus
// instancing — the fish's triangles are unique every frame.

/// Resource holding the shader handle for the pipeline to take.
#[derive(Resource)]
struct FishShader(Handle<Shader>);

/// GPU buffers, re-filled from [`FishDrawData`] every frame.
#[derive(Resource)]
struct FishBuffers {
    vertices: RawBufferVec<FishVertex>,
    indices: RawBufferVec<u32>,
    index_count: u32,
}

impl Default for FishBuffers {
    fn default() -> Self {
        Self {
            vertices: RawBufferVec::new(BufferUsages::VERTEX),
            indices: RawBufferVec::new(BufferUsages::INDEX),
            index_count: 0,
        }
    }
}

/// Copy this frame's triangles into the render world. (`Option`: the
/// buffers resource is created in `RenderStartup`, which may not have run
/// yet.)
fn extract_fish(data: Extract<Res<FishDrawData>>, buffers: Option<ResMut<FishBuffers>>) {
    let Some(mut buffers) = buffers else { return };
    buffers.vertices.values_mut().clone_from(&data.vertices);
    buffers.indices.values_mut().clone_from(&data.indices);
    buffers.index_count = data.indices.len() as u32;
}

/// Upload the triangles.
fn prepare_fish(
    mut buffers: ResMut<FishBuffers>,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
) {
    if buffers.index_count == 0 {
        return;
    }
    buffers.vertices.write_buffer(&render_device, &render_queue);
    buffers.indices.write_buffer(&render_device, &render_queue);
}

#[derive(Resource)]
struct FishPipeline {
    mesh2d_pipeline: Mesh2dPipeline,
    shader: Handle<Shader>,
}

fn init_fish_pipeline(
    mut commands: Commands,
    mesh2d_pipeline: Res<Mesh2dPipeline>,
    shader: Res<FishShader>,
) {
    commands.insert_resource(FishPipeline {
        mesh2d_pipeline: mesh2d_pipeline.clone(),
        shader: shader.0.clone(),
    });
}

impl SpecializedRenderPipeline for FishPipeline {
    type Key = Mesh2dPipelineKey;

    fn specialize(&self, key: Self::Key) -> RenderPipelineDescriptor {
        let format = match key.contains(Mesh2dPipelineKey::HDR) {
            true => ViewTarget::TEXTURE_FORMAT_HDR,
            false => TextureFormat::bevy_default(),
        };

        RenderPipelineDescriptor {
            label: Some("fish_pipeline".into()),
            vertex: VertexState {
                shader: self.shader.clone(),
                buffers: vec![VertexBufferLayout {
                    array_stride: size_of::<FishVertex>() as u64,
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
                    // The outline feather needs real alpha blending; draw
                    // order inside the index buffer is the painter order.
                    blend: Some(BlendState::ALPHA_BLENDING),
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

/// Binds the buffers and issues the one indexed draw.
struct DrawFishGeometry;

impl<P: PhaseItem> RenderCommand<P> for DrawFishGeometry {
    type Param = SRes<FishBuffers>;
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
        if buffers.index_count == 0 {
            return RenderCommandResult::Success;
        }
        let (Some(vertices), Some(indices)) =
            (buffers.vertices.buffer(), buffers.indices.buffer())
        else {
            return RenderCommandResult::Failure("fish buffers not uploaded");
        };
        pass.set_vertex_buffer(0, vertices.slice(..));
        pass.set_index_buffer(indices.slice(..), IndexFormat::Uint32);
        pass.draw_indexed(0..buffers.index_count, 0, 0..1);
        RenderCommandResult::Success
    }
}

type DrawFish = (
    SetItemPipeline,
    SetMesh2dViewBindGroup<0>,
    DrawFishGeometry,
);

/// Queue the one fish draw into every 2D view.
fn queue_fish(
    transparent_draw_functions: Res<DrawFunctions<Transparent2d>>,
    fish_pipeline: Option<Res<FishPipeline>>,
    mut pipelines: ResMut<SpecializedRenderPipelines<FishPipeline>>,
    pipeline_cache: Res<PipelineCache>,
    buffers: Option<Res<FishBuffers>>,
    mut transparent_render_phases: ResMut<ViewSortedRenderPhases<Transparent2d>>,
    views: Query<(&ExtractedView, &Msaa)>,
) {
    let (Some(fish_pipeline), Some(buffers)) = (fish_pipeline, buffers) else {
        return;
    };
    if buffers.index_count == 0 {
        return;
    }
    let draw_fish = transparent_draw_functions.read().id::<DrawFish>();

    for (view, msaa) in &views {
        let Some(transparent_phase) = transparent_render_phases.get_mut(&view.retained_view_entity)
        else {
            continue;
        };

        let key = Mesh2dPipelineKey::from_msaa_samples(msaa.samples())
            | Mesh2dPipelineKey::from_hdr(view.hdr)
            | Mesh2dPipelineKey::from_primitive_topology(PrimitiveTopology::TriangleList);
        let pipeline_id = pipelines.specialize(&pipeline_cache, &fish_pipeline, key);

        transparent_phase.add(Transparent2d {
            // The draw is fully described by resources; no entity involved.
            entity: (Entity::PLACEHOLDER, MainEntity::from(Entity::PLACEHOLDER)),
            draw_function: draw_fish,
            pipeline: pipeline_id,
            sort_key: FloatOrd(0.0),
            batch_range: 0..1,
            extra_index: PhaseItemExtraIndex::None,
            extracted_index: usize::MAX,
            indexed: true,
        });
    }
}

const FISH_SHADER: &str = r"
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

#[cfg(test)]
mod tests {
    use super::*;

    /// Straight runs collapse: colinearity 1 → rdetail 0 → each segment
    /// emits only its start point, and the final point of the run is never
    /// emitted — exactly the Lua behaviour (its NaN point gets dropped by
    /// the dedupe).
    #[test]
    fn spline_collapses_straight_runs() {
        let points = [
            Vec2::new(0.0, 0.0),
            Vec2::new(10.0, 0.0),
            Vec2::new(20.0, 0.0),
            Vec2::new(30.0, 0.0),
        ];
        let mut out = Vec::new();
        spline_v2(&points, 500.0, &mut out);
        assert_eq!(
            out,
            vec![
                Vec2::new(0.0, 0.0),
                Vec2::new(10.0, 0.0),
                Vec2::new(20.0, 0.0)
            ]
        );
    }

    /// Full output parity with the Lua renderV2 under LuaJIT
    /// (/tmp/fish_truth.lua): same shape, detail 7 — covers fractional
    /// rdetail, the inclusive loop, the p2 append, and the duplicated
    /// segment-boundary points the dedupe later removes. (The sample cap
    /// never binds at this detail/scale.)
    #[test]
    fn spline_matches_lua_output() {
        const LUA_OUT: &[(f32, f32)] = &[
            (0.00000, 0.00000),
            (2.57253, 1.23907),
            (8.96883, 4.10691),
            (17.20699, 7.32941),
            (25.30509, 9.63249),
            (30.00000, 10.00000),
            (30.00000, 10.00000),
            (33.99098, 8.34245),
            (37.37952, 4.95943),
            (40.49302, 0.91500),
            (43.65889, -2.72676),
            (47.20454, -4.90181),
            (50.00000, -5.00000),
            (50.00000, -5.00000),
            (53.97816, -3.04170),
            (58.41584, 0.81530),
            (63.12100, 5.75476),
            (67.90159, 10.96046),
            (72.56553, 15.61617),
            (76.92079, 18.90566),
            (80.00000, 20.00000),
            (80.00000, 20.00000),
            (83.76359, 19.13855),
            (87.67316, 16.34203),
            (91.45978, 12.37247),
            (94.85448, 7.99189),
            (97.58832, 3.96232),
            (99.39234, 1.04578),
            (100.00000, 0.00000),
        ];
        let points = [
            Vec2::new(0.0, 0.0),
            Vec2::new(30.0, 10.0),
            Vec2::new(50.0, -5.0),
            Vec2::new(80.0, 20.0),
            Vec2::new(100.0, 0.0),
        ];
        let mut out = Vec::new();
        spline_v2(&points, 7.0, &mut out);
        assert_eq!(out.len(), LUA_OUT.len(), "point count");
        for (i, (lua, got)) in LUA_OUT.iter().zip(&out).enumerate() {
            let d = Vec2::new(lua.0, lua.1).distance(*got);
            assert!(d < 1e-3, "point {i}: {got} vs Lua {lua:?}");
        }
    }

    /// Fewer than 4 points pass through untouched (Splines:render).
    #[test]
    fn spline_short_input_passthrough() {
        let points = [Vec2::ZERO, Vec2::new(5.0, 5.0), Vec2::new(10.0, 0.0)];
        let mut out = Vec::new();
        spline_v2(&points, 100.0, &mut out);
        assert_eq!(out, points.to_vec());
    }

    /// A curved segment interpolates between its endpoints and starts/ends
    /// on them (Hermite h-basis sanity).
    #[test]
    fn spline_hits_segment_endpoints() {
        let points = [
            Vec2::new(0.0, 0.0),
            Vec2::new(10.0, 10.0),
            Vec2::new(20.0, -3.0),
            Vec2::new(30.0, 8.0),
            Vec2::new(40.0, 0.0),
        ];
        let mut out = Vec::new();
        spline_v2(&points, 50.0, &mut out);
        assert_eq!(out[0], points[0]);
        // Each interior point appears in the output (s=0 of its segment).
        for p in &points[1..points.len() - 1] {
            assert!(
                out.iter().any(|q| q.distance(*p) < 1e-4),
                "missing {p:?}"
            );
        }
        // Final point reached via the last segment's append/inclusive end.
        assert!(out.last().unwrap().distance(points[4]) < 1e-4);
    }

    /// Duplicate points NaN-poison their segment into emitting nothing,
    /// like the Lua.
    #[test]
    fn spline_duplicate_points_emit_nothing_for_segment() {
        let points = [
            Vec2::new(0.0, 0.0),
            Vec2::new(10.0, 2.0),
            Vec2::new(10.0, 2.0),
            Vec2::new(20.0, 0.0),
            Vec2::new(30.0, 5.0),
        ];
        let mut out = Vec::new();
        spline_v2(&points, 50.0, &mut out);
        assert!(out.iter().all(|p| p.x.is_finite() && p.y.is_finite()));
    }

    #[test]
    fn cleanup_dedupes_below_half_pixel() {
        let points = [
            Vec2::new(0.0, 0.0),
            Vec2::new(0.2, 0.0),
            Vec2::new(0.6, 0.0),
            Vec2::new(0.7, 0.0),
            Vec2::new(2.0, 0.0),
        ];
        let mut out = Vec::new();
        cleanup(&points, &mut out);
        assert_eq!(
            out,
            vec![Vec2::new(0.0, 0.0), Vec2::new(0.6, 0.0), Vec2::new(2.0, 0.0)]
        );
    }

    /// Sub-pixel simplification keeps endpoints and every surviving point
    /// within eps of the original polyline.
    #[test]
    fn simplify_is_subpixel_faithful() {
        // A gentle arc sampled densely.
        let points: Vec<Vec2> = (0..100)
            .map(|i| {
                let t = i as f32 / 99.0 * PI;
                Vec2::new(t.cos() * 50.0, t.sin() * 50.0)
            })
            .collect();
        let mut out = Vec::new();
        simplify(&points, 0.15, &mut out);
        assert!(out.len() < points.len() / 2, "no reduction: {}", out.len());
        assert_eq!(out[0], points[0]);
        assert_eq!(*out.last().unwrap(), *points.last().unwrap());
        // Every original point stays within eps of the simplified chain.
        for &p in &points {
            let mut best = f32::MAX;
            for pair in out.windows(2) {
                let (a, b) = (pair[0], pair[1]);
                let ab = b - a;
                let t = ((p - a).dot(ab) / ab.length_squared()).clamp(0.0, 1.0);
                best = best.min(p.distance(a + ab * t));
            }
            assert!(best <= 0.15 + 1e-4, "point {p} off by {best}");
        }
    }

    /// Ear clipping covers the polygon exactly (area-preserving), for both
    /// windings and concave shapes.
    #[test]
    fn ear_clip_preserves_area() {
        let l_shape = [
            Vec2::new(0.0, 0.0),
            Vec2::new(4.0, 0.0),
            Vec2::new(4.0, 1.0),
            Vec2::new(1.0, 1.0),
            Vec2::new(1.0, 3.0),
            Vec2::new(0.0, 3.0),
        ];
        for flip in [false, true] {
            let pts: Vec<Vec2> = if flip {
                l_shape.iter().rev().copied().collect()
            } else {
                l_shape.to_vec()
            };
            let mut tri_area = 0.0;
            let mut count = 0;
            ear_clip(&pts, &mut EarScratch::default(), |a, b, c| {
                let (a, b, c) = (pts[a as usize], pts[b as usize], pts[c as usize]);
                tri_area += ((b - a).perp_dot(c - a) / 2.0).abs();
                count += 1;
            });
            assert_eq!(count, pts.len() - 2);
            // 4x1 horizontal bar + 1x2 vertical leg.
            assert!((tri_area - 6.0).abs() < 1e-4, "area {tri_area}");
        }
    }

    /// Window→world emission: the screen centre lands at the world origin,
    /// y flipped.
    #[test]
    fn emitter_flips_to_world_coords() {
        let mut emitter = Emitter::default();
        emitter.clear(Vec2::new(640.0, 400.0));
        emitter.vertex(Vec2::new(640.0, 400.0), [255; 4]);
        emitter.vertex(Vec2::new(0.0, 0.0), [255; 4]);
        emitter.vertex(Vec2::new(1280.0, 800.0), [255; 4]);
        assert_eq!(emitter.vertices[0].pos, [0.0, 0.0]);
        assert_eq!(emitter.vertices[1].pos, [-640.0, 400.0]);
        assert_eq!(emitter.vertices[2].pos, [640.0, -400.0]);
    }

    /// LÖVE's segment-count formula: big radii get enough segments to stay
    /// smooth, tiny ones the floor of 8.
    #[test]
    fn ellipse_segment_counts() {
        assert_eq!(ellipse_segments(1.0, 1.0), 8);
        assert!(ellipse_segments(75.0, 37.5) > 20);
    }

    /// Stage timings for the perf loop. Run explicitly:
    /// `cargo test --release bench_emit_stages -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn bench_emit_stages() {
        use super::super::sim::Fish;
        use std::time::Instant;

        let mut rng = rand::rng();
        let mut fish = Fish::new(Vec2::new(640.0, 400.0), 0.1, 200.0, &mut rng);
        fish.update(0.0);
        // Bend it like a swimming pose.
        for frame in 0..120 {
            let t = frame as f32 / 20.0;
            fish.set_target_at_speed(
                Vec2::new(640.0 + 300.0 * t.cos(), 400.0 + 300.0 * t.sin()),
                1.0 / 60.0,
            );
            fish.update(1.0 / 60.0);
        }

        const N: usize = 20_000;
        let mut emitter = Emitter::default();
        let mut scratch = Scratch::default();

        let t = Instant::now();
        for _ in 0..N {
            emitter.clear(Vec2::new(640.0, 400.0));
            emit_fish(&mut emitter, &mut scratch, &fish);
        }
        println!("emit_fish      {:>8.1?}/fish", t.elapsed() / N as u32);

        let parts: [(&str, fn(&mut Emitter, &mut Scratch, &Fish)); 4] = [
            ("lateral_fins", |e, _, f| emit_lateral_fins(e, f)),
            ("tail", emit_tail),
            ("body", emit_body),
            ("dorsal", emit_dorsal_fin),
        ];
        for (name, part) in parts {
            let t = Instant::now();
            for _ in 0..N {
                emitter.clear(Vec2::new(640.0, 400.0));
                part(&mut emitter, &mut scratch, &fish);
            }
            println!("{name:<14} {:>8.1?}/fish", t.elapsed() / N as u32);
        }
        let t = Instant::now();
        for _ in 0..N {
            emitter.clear(Vec2::new(640.0, 400.0));
            emit_eyes(&mut emitter, &fish);
        }
        println!("eyes           {:>8.1?}/fish", t.elapsed() / N as u32);

        // Body sub-stages.
        let mut s = Scratch::default();
        s.begin_part();
        // Rebuild the body shape exactly as emit_body does.
        let j = &fish.spine.joints;
        let bw0 = fish.body_width(0);
        let front = set_magnitude_or_zero(j[0] - j[1], bw0 + fish.spine.link) + j[1];
        s.left.push(front);
        s.right.push(front);
        let angle = (j[0] - j[1]).to_angle();
        for (side, sign) in [(&mut s.left, -1.0f32), (&mut s.right, 1.0)] {
            side.push(j[0] + Vec2::from_angle(angle + sign * PI / 8.0) * bw0);
            side.push(j[0] + Vec2::from_angle(angle + sign * PI / 4.0) * bw0);
        }
        for i in 0..JOINTS - 2 {
            let (right_side, left_side) = orthogonal(j[i], j[i + 1], fish.body_width(i));
            s.left.push(left_side);
            s.right.push(right_side);
        }
        s.join_sides();

        let t = Instant::now();
        for _ in 0..N {
            spline_v2(&s.shape, BODY_DETAIL, &mut s.spline);
        }
        println!(
            "body spline    {:>8.1?}/fish ({} pts)",
            t.elapsed() / N as u32,
            s.spline.len()
        );
        let t = Instant::now();
        for _ in 0..N {
            cleanup(&s.spline, &mut s.cleaned);
        }
        println!(
            "body cleanup   {:>8.1?}/fish ({} pts)",
            t.elapsed() / N as u32,
            s.cleaned.len()
        );
        let t = Instant::now();
        for _ in 0..N {
            simplify(&s.cleaned, 0.15, &mut s.simplified);
        }
        println!(
            "body simplify  {:>8.1?}/fish ({} pts)",
            t.elapsed() / N as u32,
            s.simplified.len()
        );
        let t = Instant::now();
        for _ in 0..N {
            emitter.clear(Vec2::new(640.0, 400.0));
            emitter.fill_polygon(&s.simplified, [255; 4]);
        }
        println!("body fill      {:>8.1?}/fish", t.elapsed() / N as u32);
        let t = Instant::now();
        for _ in 0..N {
            emitter.clear(Vec2::new(640.0, 400.0));
            emitter.stroke_polyline(&s.simplified, [255; 4], false);
        }
        println!("body stroke    {:>8.1?}/fish", t.elapsed() / N as u32);
    }
}
