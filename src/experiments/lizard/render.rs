//! Lizard rendering: rebuild the scene's triangles every frame from the
//! sim, painter-ordered — food, feet, legs, body, belly, dorsal stripe,
//! eyes. Each spline-smoothed part is filled and outlined in white, the
//! collection's flat-color art language; each foot draws under its leg and
//! the legs under the body, so the limbs anchor into the torso (the fish
//! does the same with its fins).
//!
//! The pipeline is the fish's proven custom path, copied rather than
//! shared (the finished fish stays untouched): a 12-byte vertex (position
//! plus unorm color), persistent buffers, one indexed draw in the
//! transparent phase — the only path that gives multi-color painter order
//! plus the per-vertex alpha the feathered outlines need in a single
//! draw. The fish's parallel-build machinery is NOT copied: one lizard.
//!
//! Geometry is computed in the sim's window coordinates and y-flipped to
//! world coordinates at vertex emission.

use std::f32::consts::{FRAC_PI_2, PI, TAU};
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
use bytemuck::{Pod, Zeroable};

use super::settings::LizardSettings;
use super::sim::{Leg, Lizard, LizardEntity, LizardGame, LizardSimSet};
use crate::app::SimBounds;
use crate::experiments::{CurrentExperiment, ExperimentId, experiment_active};

/// Spline detail per part — the body is sampled finely, the limbs less so.
const BODY_DETAIL: f32 = 500.0;
const LEG_DETAIL: f32 = 120.0;
/// End-cap reach as a multiple of the local half-width. The reference width
/// profile (snout 52 → cheek 58 → neck pinch 40) already shapes the skull, so
/// the snout cap just rounds the front off (≈ a semicircle of the snout
/// half-width, like the reference's ±π/6 head vertices); the tail runs long
/// so its fine 7px tip tapers to a soft point.
const HEAD_CAP: f32 = 1.1;
const TAIL_CAP: f32 = 2.2;

/// One triangle vertex: world position + linear unorm color.
#[derive(Clone, Copy, Pod, Zeroable)]
#[repr(C)]
struct LizardVertex {
    pos: [f32; 2],
    color: [u8; 4],
}

/// The lizard's greens (sRGB, pre-converted to linear unorm once) — the
/// fish's blue sibling at the same saturation level, per the collection's
/// flat-fill + white-outline language. The food flakes are the fish's.
struct Palette {
    body: [u8; 4],
    limb: [u8; 4],
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
        body: linear_u8(78.0 / 255.0, 138.0 / 255.0, 58.0 / 255.0),
        // Darker than the body by a clear step: the limbs draw OVER the
        // torso, and at the old near-body green a leg lying on the flank
        // all but vanished.
        limb: linear_u8(56.0 / 255.0, 106.0 / 255.0, 42.0 / 255.0),
        flakes: [
            linear_u8(219.0 / 255.0, 182.0 / 255.0, 0.0),
            linear_u8(228.0 / 255.0, 134.0 / 255.0, 36.0 / 255.0),
            linear_u8(178.0 / 255.0, 98.0 / 255.0, 30.0 / 255.0),
        ],
        white: [255, 255, 255, 255],
        eye: linear_u8(0.93, 0.90, 0.72),
        pupil: linear_u8(0.05, 0.05, 0.05),
    })
}

pub fn plugin(app: &mut App) {
    // Shaders only exist in the main world; hand the handle into the
    // render app.
    let shader = app
        .world_mut()
        .resource_mut::<Assets<Shader>>()
        .add(Shader::from_wgsl(LIZARD_SHADER, file!()));

    app.init_resource::<LizardDrawData>().add_systems(
        Update,
        (
            rebuild_geometry
                .after(LizardSimSet)
                // Runs in Options too (no sim_active gate): the sim is frozen
                // there, but a live reshape from a wonky slider must redraw so
                // the popup previews the new body instead of a stale frame.
                .run_if(experiment_active(ExperimentId::Lizard)),
            clear_when_inactive,
        ),
    );

    app.sub_app_mut(RenderApp)
        .insert_resource(LizardShader(shader))
        .add_render_command::<Transparent2d, DrawLizard>()
        .init_resource::<SpecializedRenderPipelines<LizardPipeline>>()
        .add_systems(
            RenderStartup,
            (init_lizard_pipeline.after(init_mesh_2d_pipeline), |mut commands: Commands| {
                commands.init_resource::<LizardBuffers>();
            }),
        )
        .add_systems(ExtractSchedule, extract_lizard)
        .add_systems(
            Render,
            (
                prepare_lizard.in_set(RenderSystems::PrepareResources),
                queue_lizard.in_set(RenderSystems::Queue),
            ),
        );
}

/// Main-world handoff: this frame's triangles, swapped out of the build
/// emitter (no copy) and picked up by the render world's extract.
#[derive(Resource, Default)]
struct LizardDrawData {
    vertices: Vec<LizardVertex>,
    indices: Vec<u32>,
}

/// Per-frame rebuild: food, then the lizard, serially — one creature.
fn rebuild_geometry(
    lizard: Res<LizardEntity>,
    game: Res<LizardGame>,
    settings: Res<LizardSettings>,
    bounds: Res<SimBounds>,
    mut data: ResMut<LizardDrawData>,
    mut emitter: Local<Emitter>,
    mut scratch: Local<Scratch>,
) {
    let origin = bounds.0 / 2.0;
    emitter.clear(origin);

    if let Some(lizard) = lizard.0.as_ref() {
        // The food draws first; its footprint tracks the lizard, the
        // original dot's radius rule (test-lizard.lua).
        let r = (lizard.scale * 20.0).min(10.0);
        emit_food(&mut emitter, game.food, r);
        if settings.skeleton {
            emit_skeleton(&mut emitter, lizard);
        } else {
            emit_lizard(&mut emitter, &mut scratch, lizard);
        }
    }

    // Swap, don't copy: the emitter inherits last frame's capacity back.
    std::mem::swap(&mut data.vertices, &mut emitter.vertices);
    std::mem::swap(&mut data.indices, &mut emitter.indices);
}

/// Drop the triangles (and the lizard) when another experiment takes
/// over; returning re-spawns fresh via the sim's restart path.
fn clear_when_inactive(
    current: Res<CurrentExperiment>,
    mut data: ResMut<LizardDrawData>,
    mut lizard: ResMut<LizardEntity>,
) {
    if !current.is_changed() || current.0 == ExperimentId::Lizard {
        return;
    }
    lizard.0 = None;
    data.vertices.clear();
    data.indices.clear();
}

// ---------------------------------------------------------------------------
// The lizard's parts, building point lists in window coordinates.

/// Reused per-part point buffers.
#[derive(Default)]
struct Scratch {
    left: Vec<Vec2>,
    right: Vec<Vec2>,
    shape: Vec<Vec2>,
    spline: Vec<Vec2>,
    cleaned: Vec<Vec2>,
    simplified: Vec<Vec2>,
    /// Ribbon control stations after merging near-duplicates (a folded
    /// limb's coincident points used to poison the spline — see
    /// [`emit_tapered_ribbon`]).
    ctrl: Vec<Vec2>,
    ctrl_w: Vec<f32>,
}

/// A pinch of flake food — the fish's look, copied verbatim: a loose
/// scatter of small flat ellipses in warm tones, hashed off the drop
/// position (stable while the food sits, fresh on every respawn).
fn emit_food(emitter: &mut Emitter, center: Vec2, r: f32) {
    // PCG hash seeded from the drop position's bits.
    let mut state =
        center.x.to_bits().wrapping_mul(0x9E37_79B9) ^ center.y.to_bits().rotate_left(16);
    let mut rand = move || {
        state = state.wrapping_mul(747_796_405).wrapping_add(2_891_336_453);
        let word = ((state >> ((state >> 28) + 4)) ^ state).wrapping_mul(277_803_737);
        ((word >> 22) ^ word) as f32 * (1.0 / u32::MAX as f32)
    };
    let spread = r.max(4.0);
    let flakes = ((r * 1.2) as usize).clamp(4, 10);
    let spin = rand() * TAU;
    for i in 0..flakes {
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

/// The guide's rig, live: a white circle per spine joint at the body's
/// width with the spine line threaded through, two-bone leg lines with
/// dots at shoulder and elbow, a ring per foot — and each leg's desired
/// step position as a small amber ring (the tutorial's red dot: "where
/// the foot would be if it had just finished a step").
fn emit_skeleton(emitter: &mut Emitter, lizard: &Lizard) {
    let white = palette().white;
    let amber = palette().flakes[0];
    let s = lizard.scale;

    for leg in &lizard.legs {
        let (_, target) = lizard.leg_frame(leg);
        emitter.stroke_circle(target, (6.0 * s).max(2.5), amber);
        emitter.stroke_polyline(&[leg.shoulder, leg.elbow, leg.foot], white, false);
        emitter.fill_circle(leg.shoulder, 2.5, white);
        emitter.fill_circle(leg.elbow, 2.5, white);
        emitter.stroke_circle(leg.foot, (9.0 * s * leg.girth).max(3.0), white);
    }

    let joints = lizard.joints();
    for (i, &joint) in joints.iter().enumerate() {
        emitter.stroke_circle(joint, lizard.body_width(i), white);
    }
    emitter.stroke_polyline(joints, white, false);
    for &joint in joints {
        emitter.fill_circle(joint, 2.5, white);
    }

    let (left, right, eye_size) = eye_geometry(lizard);
    emitter.stroke_circle(left, eye_size, white);
    emitter.stroke_circle(right, eye_size, white);
}

fn emit_lizard(emitter: &mut Emitter, scratch: &mut Scratch, lizard: &Lizard) {
    // Anatomical layering, back to front: each foot under its own leg, the
    // legs under the body. The toes tuck beneath the limb that grows from
    // them, and the trunk sweeps cleanly over the shoulder roots so the
    // limbs read as anchored INTO the flanks rather than pasted on top.
    // (The old "floating feet on turns" was never a layering problem — it
    // was the spline dropping a collinear tail point, fixed in
    // emit_tapered_ribbon — so the legs can sit under the body again.)
    for leg in &lizard.legs {
        emit_foot(emitter, scratch, lizard, leg);
    }
    for leg in &lizard.legs {
        emit_leg(emitter, scratch, lizard, leg);
    }
    emit_body(emitter, scratch, lizard);
    emit_eyes(emitter, lizard);
}

/// Monotone-cubic (PCHIP / Fritsch–Carlson) tangents for the half-width
/// profile at stations `knots` (ascending) with values `widths`, written
/// into `out`. This is the key to a smooth silhouette: the rails are the
/// splined centerline offset by this width, so the width must itself be a
/// smooth curve. Plain linear width faceted the rails into chords between
/// stations; a basic smoothstep removed the facets but scalloped any
/// multi-point ramp (the snout). A monotone cubic is C1-smooth AND cannot
/// overshoot — so a flat run stays flat, the snout's rising ramp is a clean
/// curve with no scallop, and a step between regions transitions without a
/// bulge.
fn pchip_tangents(knots: &[f32], widths: &[f32], out: &mut Vec<f32>) {
    let n = knots.len();
    out.clear();
    if n == 0 {
        return;
    }
    if n == 1 {
        out.push(0.0);
        return;
    }
    let secant = |k: usize| (widths[k + 1] - widths[k]) / (knots[k + 1] - knots[k]).max(1e-9);
    for k in 0..n {
        let m = if k == 0 {
            secant(0)
        } else if k == n - 1 {
            secant(n - 2)
        } else {
            let (s_prev, s_next) = (secant(k - 1), secant(k));
            if s_prev * s_next <= 0.0 {
                // Local extremum (or a flat shoulder): zero tangent keeps
                // the curve monotone — no overshoot bulge at region steps.
                0.0
            } else {
                // Weighted harmonic mean of the neighbouring secants.
                let h_prev = (knots[k] - knots[k - 1]).max(1e-9);
                let h_next = (knots[k + 1] - knots[k]).max(1e-9);
                let w1 = 2.0 * h_next + h_prev;
                let w2 = h_next + 2.0 * h_prev;
                (w1 + w2) / (w1 / s_prev + w2 / s_next)
            }
        };
        out.push(m);
    }
}

/// Half-width at arc-fraction `f`, evaluated on the monotone cubic Hermite
/// defined by `knots`/`widths`/`tangents` (from [`pchip_tangents`]). Clamps
/// past the ends, and never returns negative.
fn smooth_width(knots: &[f32], widths: &[f32], tangents: &[f32], f: f32) -> f32 {
    let m = knots.len();
    if m == 0 {
        return 0.0;
    }
    if f <= knots[0] {
        return widths[0];
    }
    if f >= knots[m - 1] {
        return widths[m - 1];
    }
    let mut i = 0;
    while i + 1 < m && knots[i + 1] < f {
        i += 1;
    }
    let h = (knots[i + 1] - knots[i]).max(1e-9);
    let t = (f - knots[i]) / h;
    let t2 = t * t;
    let t3 = t2 * t;
    let h00 = 2.0 * t3 - 3.0 * t2 + 1.0;
    let h10 = t3 - 2.0 * t2 + t;
    let h01 = -2.0 * t3 + 3.0 * t2;
    let h11 = t3 - t2;
    (h00 * widths[i] + h10 * h * tangents[i] + h01 * widths[i + 1] + h11 * h * tangents[i + 1])
        .max(0.0)
}

/// How a ribbon is painted: its fill, an optional outline color, and
/// whether that outline closes across the far end. `closed` suits the
/// body's tapered tips; open suits the leg's buried shoulder root;
/// `outline: None` is the flat markings (belly, stripe) — fill only.
struct RibbonStyle {
    fill: [u8; 4],
    outline: Option<[u8; 4]>,
    closed: bool,
}

/// A tapered ribbon down a control centerline: spline the centerline, offset
/// it into two rails by the per-station half-width (interpolated by
/// arc-length so it tracks the spline), fill the band as a quad STRIP, and
/// stroke the outline. The strip is the robust core — where a fat or sharply
/// bent ribbon makes the inner rail cross itself, a strip just overlaps into
/// a clean bulge, while ear-clipping the concatenated outline fanned a
/// spanning garbage triangle (the reported green blob).
fn emit_tapered_ribbon(
    emitter: &mut Emitter,
    scratch: &mut Scratch,
    control: &[Vec2],
    widths: &[f32],
    detail: f32,
    style: RibbonStyle,
) {
    if !build_rails(scratch, control, widths, detail) {
        return;
    }
    emitter.fill_strip(&scratch.left, &scratch.right, style.fill);
    if let Some(color) = style.outline {
        let Scratch {
            left,
            right,
            simplified,
            ..
        } = scratch;
        simplified.clear();
        simplified.extend_from_slice(left);
        simplified.extend(right.iter().rev());
        emitter.stroke_polyline(simplified, color, style.closed);
    }
}

/// Spline a control centerline (centripetal Catmull-Rom) and offset it into
/// two rails by the per-station half-width (monotone-cubic in arc-length so
/// it tracks the curve smoothly), leaving the centerline in `scratch.cleaned`
/// and the rails in `scratch.left`/`scratch.right`. Returns false if the
/// whole part spans under a pixel. The shared core of every fleshy part.
fn build_rails(scratch: &mut Scratch, control: &[Vec2], widths: &[f32], detail: f32) -> bool {
    // Merge near-coincident control stations first (keeping the wider
    // half-width, so the silhouette never thins). A fully folded limb puts
    // its elbow ON the shoulder→foot midpoints; duplicate points poison
    // the spline's tangents into NaN and its guard then DROPS those
    // segments — the limb used to vanish for a frame at sharp folds (the
    // reported leg flicker).
    let Scratch {
        ctrl,
        ctrl_w,
        spline,
        cleaned,
        left,
        right,
        ..
    } = scratch;
    ctrl.clear();
    ctrl_w.clear();
    for (k, &p) in control.iter().enumerate() {
        if let Some(&last) = ctrl.last()
            && p.distance(last) < 0.75
        {
            let w = ctrl_w.last_mut().expect("widths track stations");
            *w = w.max(widths[k]);
            continue;
        }
        ctrl.push(p);
        ctrl_w.push(widths[k]);
    }
    let m = ctrl.len();
    if m < 2 {
        return false;
    }

    // Arc-fraction of each control station, so widths interpolate along the
    // splined path even when stations are unevenly spaced (snout vs torso).
    let ctrl_len: f32 = (1..m).map(|k| ctrl[k - 1].distance(ctrl[k])).sum();
    let mut knots = Vec::with_capacity(m);
    let mut acc = 0.0;
    for k in 0..m {
        if k > 0 {
            acc += ctrl[k - 1].distance(ctrl[k]);
        }
        knots.push(if ctrl_len > 1e-6 {
            acc / ctrl_len
        } else {
            k as f32 / (m - 1) as f32
        });
    }
    let mut wtan = Vec::with_capacity(m);
    pchip_tangents(&knots, ctrl_w, &mut wtan);

    spline_v2(ctrl, detail, spline);
    cleanup(spline, cleaned);
    // The spline can drop the final point when the path's tail is collinear
    // (a limb's lower bone is straight by construction — its elbow→foot
    // midpoint sits on the line). Pin the path's end back on so the foot
    // doesn't float half a bone away.
    if let (Some(&last), Some(&want)) = (cleaned.last(), ctrl.last())
        && last.distance(want) > 0.5
    {
        cleaned.push(want);
    }
    let n = cleaned.len();
    if n < 2 {
        return false;
    }
    let path_len: f32 = (1..n).map(|k| cleaned[k - 1].distance(cleaned[k])).sum();
    left.clear();
    right.clear();
    let mut cacc = 0.0;
    for k in 0..n {
        if k > 0 {
            cacc += cleaned[k - 1].distance(cleaned[k]);
        }
        let f = if path_len > 1e-6 { cacc / path_len } else { 0.0 };
        let w = smooth_width(&knots, ctrl_w, &wtan, f);
        let tangent = if k == 0 {
            cleaned[1] - cleaned[0]
        } else if k == n - 1 {
            cleaned[n - 1] - cleaned[n - 2]
        } else {
            cleaned[k + 1] - cleaned[k - 1]
        };
        let normal = tangent.perp().normalize_or(Vec2::Y);
        left.push(cleaned[k] + normal * w);
        right.push(cleaned[k] - normal * w);
    }
    true
}

/// A smooth half-ellipse end cap, from `from` around the end to `to`,
/// bulging by `reach` along `axis` (pointing away from the body). Meets the
/// rails tangentially (its minor axis is the rail half-width), so the
/// silhouette stays C1-smooth across the join — no cone-meets-cylinder
/// shoulder. Appends the arc (excluding the endpoints) to `out`.
fn cap_arc(center: Vec2, axis: Vec2, side: Vec2, reach: f32, half: f32, out: &mut Vec<Vec2>) {
    // `side` points to `from` (the +half rail end); sweep θ from +π/2 (from)
    // through 0 (the tip, along axis) to −π/2 (to). Resolution is fixed by
    // ANGLE (≈2° a step), not pixels, so the cap stays a smooth arc even on
    // a small lizard zoomed way in.
    let segments = ((reach.max(half) * 1.2) as usize).clamp(90, 256);
    for s in 1..segments {
        let theta = FRAC_PI_2 - PI * (s as f32 / segments as f32);
        out.push(center + axis * (reach * theta.cos()) + side * (half * theta.sin()));
    }
}

/// One limb as a slender tapered ribbon shoulder → elbow → foot. The
/// mid-bone control points let the Hermite round the elbow into a smooth
/// knee while the bones stay straight; the gentle taper thins the limb to
/// the ankle (the old widths made fat stumpy tubes — the legs are long
/// now, so they can be slim like the reference's). Open at the shoulder
/// root (buried under the body); the ankle rounds off under the paw.
fn emit_leg(emitter: &mut Emitter, scratch: &mut Scratch, lizard: &Lizard, leg: &Leg) {
    let s = lizard.scale * leg.girth;
    let control = [
        leg.shoulder,
        leg.shoulder.midpoint(leg.elbow),
        leg.elbow,
        leg.elbow.midpoint(leg.foot),
        leg.foot,
    ];
    let widths = [20.0 * s, 16.0 * s, 12.5 * s, 9.5 * s, 7.0 * s];
    emit_tapered_ribbon(
        emitter,
        scratch,
        &control,
        &widths,
        LEG_DETAIL,
        RibbonStyle {
            fill: palette().limb,
            outline: Some(palette().white),
            closed: false,
        },
    );
}

/// The foot: a hand-shaped paw — a rounded palm with four toe lobes
/// fanning forward — built as a STAR-SHAPED radial polygon around the
/// ankle: radius(θ) = palm, raised through each toe's bump. Star-shaped
/// means it can never self-intersect, so the fill can never hit the
/// ear-clip bail (the old splined web outline looped at sharp headings —
/// the paw flashed in and out and its stray white outline read as
/// "little birds"). Points down the leg's heading (eased toward the
/// travel direction, so the toes turn with the body); the step's lift
/// arc swells it — the render-only step cue.
fn emit_foot(emitter: &mut Emitter, scratch: &mut Scratch, lizard: &Lizard, leg: &Leg) {
    let s = lizard.scale * leg.girth;
    let swell = 1.0 + 0.25 * leg.lift();
    let base = leg.heading.to_angle();
    let palm = 8.5 * s * swell;
    let toe = 5.5 * s * swell; // bump height past the palm — short, compact toes
    const TOES: usize = 4;
    // Total half-fan of the toe spread, and each toe lobe's half-width. A
    // tight fan + short toes reads as a rounded reptile foot, not a splayed
    // bird hand.
    const FAN: f32 = 0.55;
    const LOBE: f32 = 0.34;
    const SAMPLES: usize = 44;

    scratch.shape.clear();
    for q in 0..SAMPLES {
        // March the full circle in foot-relative angles, so the toe bumps
        // sit symmetrically around the heading.
        let rel = -PI + TAU * (q as f32 / SAMPLES as f32);
        let mut r = palm;
        for k in 0..TOES {
            let center = -FAN + (2.0 * FAN) * (k as f32 / (TOES - 1) as f32);
            let d = (rel - center).abs() / LOBE;
            if d < 1.0 {
                // A rounded lobe: cosine bell, max() so neighbours merge
                // into webbing instead of notching each other.
                let bump = (d * PI / 2.0).cos().powf(1.4);
                r = r.max(palm + toe * bump);
            }
        }
        scratch
            .shape
            .push(leg.foot + Vec2::from_angle(base + rel) * r);
    }
    emitter.fill_polygon(&scratch.shape, palette().limb);
    emitter.stroke_polyline(&scratch.shape, palette().white, true);
}

/// The whole body — flat green flesh and a white outline — drawn in ONE
/// depth-ordered sweep so a self-overlapping (curled) lizard layers
/// consistently and the silhouette stays smooth.
///
/// Construction: spline the spine joints (centripetal Catmull-Rom) and offset
/// by the reference per-joint half-width into a smooth rail, the ends capped
/// by tangent half-ellipses (a rounded snout, a fine tapering tail tip).
///
/// The OUTLINE is not a stroked polyline (which traces a self-crossing loop
/// wherever a tight curl folds the inner rail) but a fattened white UNDERLAY
/// drawn first: every green slice paints over it, so the only white that
/// survives is the true silhouette margin — and a fold just overlaps into
/// solid colour. The green flesh is then emitted slice by slice from TAIL to
/// HEAD, so where the body crosses itself the head-end always wins (a single
/// consistent order). Flat green + outline only — the reference has no
/// belly wash or dorsal stripe.
fn emit_body(emitter: &mut Emitter, scratch: &mut Scratch, lizard: &Lizard) {
    let j = lizard.joints();
    let n = j.len();
    let control: Vec<Vec2> = j.to_vec();
    // The reference width profile shapes the whole body — snout, cheek
    // bulge, neck pinch, barrel swell, tapering tail — so the rail simply
    // follows it; no render-time reshaping.
    let widths: Vec<f32> = (0..n).map(|i| lizard.body_width(i)).collect();
    if !build_rails(scratch, &control, &widths, BODY_DETAIL) {
        return;
    }

    // Per-sample frame: centre, unit outward normal, half-width.
    let cleaned = scratch.cleaned.clone();
    let left = scratch.left.clone();
    let m = cleaned.len();
    let last = m - 1;
    let mut nrm = Vec::with_capacity(m);
    let mut wid = Vec::with_capacity(m);
    for k in 0..m {
        let off = left[k] - cleaned[k];
        wid.push(off.length());
        nrm.push(off.normalize_or(Vec2::Y));
    }

    let p = palette();
    let out = (1.6 * lizard.scale).max(1.0);

    // End caps (a green cap and a white one fattened by the outline width).
    let cap = |center: Vec2, axis: Vec2, side: Vec2, reach: f32, half: f32| -> Vec<Vec2> {
        let mut arc = vec![center + side * half];
        cap_arc(center, axis, side, reach, half, &mut arc);
        arc.push(center - side * half);
        arc
    };
    let t0 = (cleaned[1] - cleaned[0]).normalize_or(Vec2::X);
    let tl = (cleaned[last] - cleaned[last - 1]).normalize_or(t0);
    let (w0, wl) = (wid[0], wid[last]);
    let head_white = cap(cleaned[0], -t0, -nrm[0], HEAD_CAP * w0 + out, w0 + out);
    let head_green = cap(cleaned[0], -t0, -nrm[0], HEAD_CAP * w0, w0);
    let tail_white = cap(cleaned[last], tl, nrm[last], TAIL_CAP * wl + out, wl + out);
    let tail_green = cap(cleaned[last], tl, nrm[last], TAIL_CAP * wl, wl);

    // 1) White underlay first — under all the green, so only the silhouette
    //    margin shows and a folded inner rail just overlaps cleanly.
    let lf: Vec<Vec2> = (0..m).map(|k| cleaned[k] + nrm[k] * (wid[k] + out)).collect();
    let rf: Vec<Vec2> = (0..m).map(|k| cleaned[k] - nrm[k] * (wid[k] + out)).collect();
    emitter.fill_strip(&lf, &rf, p.white);
    emitter.fill_fan(cleaned[0], &head_white, p.white);
    emitter.fill_fan(cleaned[last], &tail_white, p.white);

    // 2) Flat green flesh, TAIL → HEAD so overlaps resolve head-over-tail.
    emitter.fill_fan(cleaned[last], &tail_green, p.body);
    for q in (0..last).rev() {
        let (a, b) = (q, q + 1);
        emitter.fill_strip(
            &[cleaned[a] + nrm[a] * wid[a], cleaned[b] + nrm[b] * wid[b]],
            &[cleaned[a] - nrm[a] * wid[a], cleaned[b] - nrm[b] * wid[b]],
            p.body,
        );
    }
    emitter.fill_fan(cleaned[0], &head_green, p.body);
}

/// Eye centres and radius, the reference's: two circles at **±3π/5** off the
/// head joint's heading, **width[0] − 7** out (just inside the snout edge,
/// near the back corners of the skull) and **diameter 24** (radius 12). All
/// scaled. Shared by the skinned eyes and the skeleton overlay.
fn eye_geometry(lizard: &Lizard) -> (Vec2, Vec2, f32) {
    let j = lizard.joints();
    let s = lizard.scale;
    let bw0 = lizard.body_width(0);
    let heading = (j[0] - j[1]).to_angle();
    let reach = bw0 - 7.0 * s;
    const EYE_ANGLE: f32 = 3.0 * PI / 5.0;
    let left = j[0] + Vec2::from_angle(heading + EYE_ANGLE) * reach;
    let right = j[0] + Vec2::from_angle(heading - EYE_ANGLE) * reach;
    (left, right, 12.0 * s)
}

/// Bulging temple eyes: pale sclera ringed white so the bulge reads
/// against the body outline it crosses, near-black pupil seated slightly
/// outward-forward — where the lizard is looking.
fn emit_eyes(emitter: &mut Emitter, lizard: &Lizard) {
    let j = lizard.joints();
    let (eye_left, eye_right, eye_size) = eye_geometry(lizard);
    let p = palette();
    emitter.fill_circle(eye_left, eye_size, p.eye);
    emitter.fill_circle(eye_right, eye_size, p.eye);
    emitter.stroke_circle(eye_left, eye_size, p.white);
    emitter.stroke_circle(eye_right, eye_size, p.white);
    // A round pupil seated slightly forward — where the lizard is looking.
    let fwd = (j[0] - j[1]).normalize_or(Vec2::X);
    let pupil_off = fwd * (0.28 * eye_size);
    let pupil_size = 0.5 * eye_size;
    emitter.fill_circle(eye_left + pupil_off, pupil_size, p.pupil);
    emitter.fill_circle(eye_right + pupil_off, pupil_size, p.pupil);
}

// ---------------------------------------------------------------------------
// Splines — a CENTRIPETAL Catmull-Rom (Barry–Goldman pyramid). This
// replaces the fish's uniform Hermite (tangents 0.5·(p2−p0)): uniform
// Catmull-Rom overshoots and kinks wherever the control points are
// unevenly spaced, because the tangent at a joint is scaled by the
// distance to its FAR neighbour while the local segment may be tiny — and
// the lizard's snout (a cluster of close profile points) running into the
// far-apart spine joints did exactly that (a 17.7° kink at the muzzle).
// Centripetal parameterisation (knot spacing = √chord) is the standard
// cure: it provably never cusps or self-intersects, so the silhouette is
// a genuinely smooth curve at every spacing.

/// Centripetal Catmull-Rom through `points`, ~1.1 samples per pixel of
/// chord (bounded by `detail`), into `out`. Passes through every control
/// point; ends use a reflected phantom so the curve doesn't flatten there.
fn spline_v2(points: &[Vec2], detail: f32, out: &mut Vec<Vec2>) {
    out.clear();
    let n = points.len();
    if n < 2 {
        out.extend_from_slice(points);
        return;
    }
    // Reflected phantom points extend the curve cleanly past the ends.
    let get = |i: i32| -> Vec2 {
        if i < 0 {
            2.0 * points[0] - points[1]
        } else if i as usize >= n {
            2.0 * points[n - 1] - points[n - 2]
        } else {
            points[i as usize]
        }
    };
    // Centripetal knot from a chord (α = 0.5); never zero, so coincident
    // control points can't divide by zero.
    let knot = |a: Vec2, b: Vec2| (a - b).length().sqrt().max(1e-4);
    for i in 0..n - 1 {
        let (p0, p1, p2, p3) = (get(i as i32 - 1), get(i as i32), get(i as i32 + 1), get(i as i32 + 2));
        let t0 = 0.0;
        let t1 = t0 + knot(p1, p0);
        let t2 = t1 + knot(p2, p1);
        let t3 = t2 + knot(p3, p2);
        let steps = (p1.distance(p2) * 2.0).clamp(6.0, detail);
        let last = steps.floor() as i32;
        for k in 0..=last {
            let s = t1 + (k as f32 / steps) * (t2 - t1);
            // Barry–Goldman repeated linear interpolation.
            let a1 = ((t1 - s) * p0 + (s - t0) * p1) / (t1 - t0);
            let a2 = ((t2 - s) * p1 + (s - t1) * p2) / (t2 - t1);
            let a3 = ((t3 - s) * p2 + (s - t2) * p3) / (t3 - t2);
            let b1 = ((t2 - s) * a1 + (s - t0) * a2) / (t2 - t0);
            let b2 = ((t3 - s) * a2 + (s - t1) * a3) / (t3 - t1);
            out.push(((t2 - s) * b1 + (s - t1) * b2) / (t2 - t1));
        }
        if steps.ceil() > steps {
            out.push(p2);
        }
    }
}

/// `cleanupPoints`: keep the first point, then only points further than
/// 0.5px from the last kept one.
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

// ---------------------------------------------------------------------------
// The triangle emitter — the fish's, copied: every primitive appends
// vertices (window coords, y-flipped to world here) and indices; triangle
// order = draw order.

#[derive(Default)]
struct Emitter {
    vertices: Vec<LizardVertex>,
    indices: Vec<u32>,
    /// Half the window, for the window→world flip.
    origin: Vec2,
    rows: Vec<[u32; 4]>,
    /// Reused point loop for [`Self::stroke_circle`].
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
        self.vertices.push(LizardVertex {
            pos: [p.x - self.origin.x, self.origin.y - p.y],
            color,
        });
        index
    }

    fn triangle(&mut self, a: u32, b: u32, c: u32) {
        self.indices.extend([a, b, c]);
    }

    /// Fill a simple polygon by ear clipping.
    fn fill_polygon(&mut self, points: &[Vec2], color: [u8; 4]) {
        let mut n = points.len();
        // A duplicated closing point would make a degenerate ear; drop it
        // for the fill only.
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
    /// at full alpha with a feather fading to transparent on each side
    /// (the fish's slimmed ±0.35px core, ±1.1px feather).
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

    /// Fill the band between two equal-length rails as a quad strip — one
    /// quad per pair of stations. Unlike ear-clipping a concatenated
    /// outline, a strip CANNOT produce a spanning garbage triangle when the
    /// inner rail crosses itself at a sharp bend: the quads just overlap, so
    /// a folded limb reads as a slightly bulged elbow, never a blob. This is
    /// the robust fill for the leg ribbon (whose elbow routinely folds to a
    /// sharp V mid-stride).
    fn fill_strip(&mut self, left: &[Vec2], right: &[Vec2], color: [u8; 4]) {
        let n = left.len().min(right.len());
        if n < 2 {
            return;
        }
        let mut prev_l = self.vertex(left[0], color);
        let mut prev_r = self.vertex(right[0], color);
        for k in 1..n {
            let l = self.vertex(left[k], color);
            let r = self.vertex(right[k], color);
            self.triangle(prev_l, prev_r, r);
            self.triangle(prev_l, r, l);
            prev_l = l;
            prev_r = r;
        }
    }

    /// Triangle-fan a cap: `center` to each consecutive pair of `arc`.
    fn fill_fan(&mut self, center: Vec2, arc: &[Vec2], color: [u8; 4]) {
        if arc.len() < 2 {
            return;
        }
        let c = self.vertex(center, color);
        let mut prev = self.vertex(arc[0], color);
        for &p in &arc[1..] {
            let cur = self.vertex(p, color);
            self.triangle(c, prev, cur);
            prev = cur;
        }
    }

    fn fill_circle(&mut self, center: Vec2, r: f32, color: [u8; 4]) {
        self.fill_ellipse(center, r, r, 0.0, color);
    }

    /// A feathered circle outline — the skeleton view's joints. Built as a
    /// closed stroked polyline so it matches the linework everywhere else.
    fn stroke_circle(&mut self, center: Vec2, r: f32, color: [u8; 4]) {
        let segments = ellipse_segments(r, r);
        let mut walk = EllipseWalk::new(r, r, 0.0, segments);
        let mut ring = std::mem::take(&mut self.ring);
        ring.clear();
        for _ in 0..segments {
            ring.push(center + walk.next_point());
        }
        self.stroke_polyline(&ring, color, true);
        self.ring = ring;
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

}

/// Marches around an ellipse with a rotation recurrence — two `sin_cos`
/// calls total instead of one per point.
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
/// Degenerate input degrades to a fan instead of failing.
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
            // No ear found — the outline crosses itself, which has no valid
            // triangulation. Bail rather than fan the remaining vertices
            // (the old behaviour spanned the crossing and painted a big
            // garbage triangle — the reported "green blob"). Leaving the
            // sliver unfilled is invisible next to that. The ribbon fills
            // (legs, body) avoid the situation entirely via fill_strip; this
            // is the backstop for the simple shapes (feet, food, eyes).
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
// standard 2D view uniform, and one indexed draw in the transparent phase
// — the fish's plumbing, renamed.

/// Resource holding the shader handle for the pipeline to take.
#[derive(Resource)]
struct LizardShader(Handle<Shader>);

/// GPU buffers, re-filled from [`LizardDrawData`] every frame.
#[derive(Resource)]
struct LizardBuffers {
    vertices: RawBufferVec<LizardVertex>,
    indices: RawBufferVec<u32>,
    index_count: u32,
}

impl Default for LizardBuffers {
    fn default() -> Self {
        Self {
            vertices: RawBufferVec::new(BufferUsages::VERTEX),
            indices: RawBufferVec::new(BufferUsages::INDEX),
            index_count: 0,
        }
    }
}

/// Copy this frame's triangles into the render world.
fn extract_lizard(data: Extract<Res<LizardDrawData>>, buffers: Option<ResMut<LizardBuffers>>) {
    let Some(mut buffers) = buffers else { return };
    buffers.vertices.values_mut().clone_from(&data.vertices);
    buffers.indices.values_mut().clone_from(&data.indices);
    buffers.index_count = data.indices.len() as u32;
}

/// Upload the triangles.
fn prepare_lizard(
    mut buffers: ResMut<LizardBuffers>,
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
struct LizardPipeline {
    mesh2d_pipeline: Mesh2dPipeline,
    shader: Handle<Shader>,
}

fn init_lizard_pipeline(
    mut commands: Commands,
    mesh2d_pipeline: Res<Mesh2dPipeline>,
    shader: Res<LizardShader>,
) {
    commands.insert_resource(LizardPipeline {
        mesh2d_pipeline: mesh2d_pipeline.clone(),
        shader: shader.0.clone(),
    });
}

impl SpecializedRenderPipeline for LizardPipeline {
    type Key = Mesh2dPipelineKey;

    fn specialize(&self, key: Self::Key) -> RenderPipelineDescriptor {
        let format = match key.contains(Mesh2dPipelineKey::HDR) {
            true => ViewTarget::TEXTURE_FORMAT_HDR,
            false => TextureFormat::bevy_default(),
        };

        RenderPipelineDescriptor {
            label: Some("lizard_pipeline".into()),
            vertex: VertexState {
                shader: self.shader.clone(),
                buffers: vec![VertexBufferLayout {
                    array_stride: size_of::<LizardVertex>() as u64,
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
struct DrawLizardGeometry;

impl<P: PhaseItem> RenderCommand<P> for DrawLizardGeometry {
    type Param = SRes<LizardBuffers>;
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
            return RenderCommandResult::Failure("lizard buffers not uploaded");
        };
        pass.set_vertex_buffer(0, vertices.slice(..));
        pass.set_index_buffer(indices.slice(..), IndexFormat::Uint32);
        pass.draw_indexed(0..buffers.index_count, 0, 0..1);
        RenderCommandResult::Success
    }
}

type DrawLizard = (
    SetItemPipeline,
    SetMesh2dViewBindGroup<0>,
    DrawLizardGeometry,
);

/// Queue the one lizard draw into every 2D view.
fn queue_lizard(
    transparent_draw_functions: Res<DrawFunctions<Transparent2d>>,
    lizard_pipeline: Option<Res<LizardPipeline>>,
    mut pipelines: ResMut<SpecializedRenderPipelines<LizardPipeline>>,
    pipeline_cache: Res<PipelineCache>,
    buffers: Option<Res<LizardBuffers>>,
    mut transparent_render_phases: ResMut<ViewSortedRenderPhases<Transparent2d>>,
    views: Query<(&ExtractedView, &Msaa)>,
) {
    let (Some(lizard_pipeline), Some(buffers)) = (lizard_pipeline, buffers) else {
        return;
    };
    if buffers.index_count == 0 {
        return;
    }
    let draw_lizard = transparent_draw_functions.read().id::<DrawLizard>();

    for (view, msaa) in &views {
        let Some(transparent_phase) = transparent_render_phases.get_mut(&view.retained_view_entity)
        else {
            continue;
        };

        let key = Mesh2dPipelineKey::from_msaa_samples(msaa.samples())
            | Mesh2dPipelineKey::from_hdr(view.hdr)
            | Mesh2dPipelineKey::from_primitive_topology(PrimitiveTopology::TriangleList);
        let pipeline_id = pipelines.specialize(&pipeline_cache, &lizard_pipeline, key);

        transparent_phase.add(Transparent2d {
            // The draw is fully described by resources; no entity involved.
            entity: (Entity::PLACEHOLDER, MainEntity::from(Entity::PLACEHOLDER)),
            draw_function: draw_lizard,
            pipeline: pipeline_id,
            sort_key: FloatOrd(0.0),
            batch_range: 0..1,
            extra_index: PhaseItemExtraIndex::None,
            extracted_index: usize::MAX,
            indexed: true,
        });
    }
}

const LIZARD_SHADER: &str = r"
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

    /// The shader string must be valid WGSL — it only reaches naga at
    /// runtime, so a typo passes every other test then kills the lizard
    /// live. Parse and validate, with the one bevy `#import` stubbed.
    #[test]
    fn shaders_compile() {
        let stub = "struct View { clip_from_world: mat4x4<f32> }\n\
                    @group(0) @binding(0) var<uniform> view: View;";
        let src = LIZARD_SHADER.replace("#import bevy_sprite::mesh2d_view_bindings::view", stub);
        let module = naga::front::wgsl::parse_str(&src)
            .unwrap_or_else(|e| panic!("lizard shader: parse: {e}"));
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .unwrap_or_else(|e| panic!("lizard shader: validate: {e}"));
    }

    /// Straight runs are now fully subdivided (no Lua detail-dropping) so a
    /// gently-curved body reads as a smooth arc, not a chord between joints
    /// — but the samples stay ON the line and span it end to end.
    #[test]
    fn spline_subdivides_straight_runs_on_the_line() {
        let points = [
            Vec2::new(0.0, 0.0),
            Vec2::new(10.0, 0.0),
            Vec2::new(20.0, 0.0),
            Vec2::new(30.0, 0.0),
        ];
        let mut out = Vec::new();
        spline_v2(&points, 500.0, &mut out);
        assert!(out.len() > 8, "run wasn't subdivided: {} points", out.len());
        assert_eq!(out.first().copied(), Some(Vec2::new(0.0, 0.0)));
        assert!(
            (out.last().unwrap().x - 30.0).abs() < 1e-3,
            "spline didn't reach the run's end: {:?}",
            out.last()
        );
        assert!(
            out.iter().all(|p| p.y.abs() < 1e-4),
            "a straight run drifted off the line"
        );
        assert!(
            out.windows(2).all(|w| w[1].x >= w[0].x - 1e-4),
            "samples backtracked (a facet)"
        );
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

    /// Ear clipping covers the polygon exactly (area-preserving), for both
    /// windings and concave shapes — the legs bend concave at the elbow.
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

    /// Every part of a walking lizard emits sane, finite geometry — the
    /// whole emit path, exercised over a curving walk with steps mid-air.
    #[test]
    fn emit_lizard_produces_finite_geometry() {
        use super::super::sim::{BodyPlan, Lizard};

        let mut lizard = Lizard::new(Vec2::new(640.0, 400.0), 0.5, 240.0, &BodyPlan::reference());
        let mut emitter = Emitter::default();
        let mut scratch = Scratch::default();
        for frame in 0..240 {
            let t = frame as f32 / 40.0;
            let goal = Vec2::new(640.0, 400.0) + Vec2::from_angle(t) * 300.0;
            lizard.set_target_at_speed(goal, 1.0 / 60.0);
            lizard.update(1.0 / 60.0);

            emitter.clear(Vec2::new(640.0, 400.0));
            emit_lizard(&mut emitter, &mut scratch, &lizard);
            assert!(!emitter.vertices.is_empty(), "frame {frame}: nothing emitted");
            assert!(
                emitter
                    .vertices
                    .iter()
                    .all(|v| v.pos[0].is_finite() && v.pos[1].is_finite()),
                "frame {frame}: non-finite vertex"
            );
            let max = emitter.vertices.len() as u32;
            assert!(
                emitter.indices.iter().all(|&i| i < max),
                "frame {frame}: index out of range"
            );
        }
    }

    /// SMOOTHNESS GATE: the splined body centerline — which the silhouette
    /// rails trace — must be a smooth curve, never a chain of chords between
    /// joints (the original facets came from the spline DROPPING points on
    /// near-straight runs). We assert that across a hard curving walk no two
    /// consecutive centerline samples turn more than a couple of degrees.
    /// (The width is C1 by construction — the monotone cubic — and the end
    /// caps are analytic ellipses, so a smooth centerline gives a smooth
    /// silhouette; the inner rail of a tight curl may still fold, but that
    /// overlaps into solid fill, by design.)
    #[test]
    fn body_centerline_is_smooth() {
        use super::super::sim::{BodyPlan, Lizard};

        let mut lizard = Lizard::new(Vec2::new(640.0, 400.0), 1.0, 240.0, &BodyPlan::reference());
        let mut emitter = Emitter::default();
        let mut scratch = Scratch::default();
        let mut overall = 0.0_f32;
        for frame in 0..240 {
            let t = frame as f32 / 40.0;
            let goal = Vec2::new(640.0, 400.0) + Vec2::from_angle(t) * 300.0;
            lizard.set_target_at_speed(goal, 1.0 / 60.0);
            lizard.update(1.0 / 60.0);
            emitter.clear(Vec2::new(640.0, 400.0));
            emit_body(&mut emitter, &mut scratch, &lizard);
            let c = &scratch.cleaned;
            for k in 1..c.len() - 1 {
                // Skip sub-pixel steps: a coincident pair is not a turn.
                if c[k - 1].distance(c[k]) < 0.3 || c[k].distance(c[k + 1]) < 0.3 {
                    continue;
                }
                let a = (c[k] - c[k - 1]).normalize_or_zero();
                let b = (c[k + 1] - c[k]).normalize_or_zero();
                overall = overall.max(a.angle_to(b).abs().to_degrees());
            }
        }
        eprintln!("worst centerline turn across the walk: {overall:.1} deg");
        assert!(
            overall < 3.0,
            "body centerline faceted: a sample turns {overall:.1}deg (want < 3)"
        );
    }

    /// The whole emit path stays well-formed at every size through hard fast
    /// direction reversals — the turning case from the green-blob
    /// screenshots. Finite vertices and in-range indices on every frame is
    /// the guard that the ribbon strips (body, legs) and the ear-clip bail
    /// never emit garbage as the body folds on itself. (The old
    /// single-outline fills self-intersected here and ear-clipped a spanning
    /// triangle — the blob; strips can't, by construction.)
    #[test]
    fn emit_path_is_robust_on_hard_turns() {
        use super::super::sim::{BodyPlan, Lizard};

        let mut emitter = Emitter::default();
        let mut scratch = Scratch::default();
        for scale in [0.2_f32, 0.5, 1.0, 1.5] {
            let mut lizard = Lizard::new(Vec2::new(640.0, 400.0), scale, 240.0, &BodyPlan::reference());
            for frame in 0..360 {
                let t = frame as f32 / 60.0;
                let goal = Vec2::new(640.0, 400.0) + Vec2::from_angle(8.0 * t) * 260.0;
                lizard.set_target_at_speed(goal, 1.0 / 60.0);
                lizard.update(1.0 / 60.0);

                emitter.clear(Vec2::new(640.0, 400.0));
                emit_lizard(&mut emitter, &mut scratch, &lizard);
                assert!(
                    emitter
                        .vertices
                        .iter()
                        .all(|v| v.pos[0].is_finite() && v.pos[1].is_finite()),
                    "@scale {scale} frame {frame}: non-finite vertex"
                );
                let max = emitter.vertices.len() as u32;
                assert!(
                    emitter.indices.iter().all(|&i| i < max),
                    "@scale {scale} frame {frame}: index out of range"
                );
            }
        }
    }
}
