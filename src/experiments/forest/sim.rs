//! The forest model: a set of procedural **L-system** trees (the original's
//! `lib/tree.lua`), grown by randomised string rewriting and walked by a turtle
//! into triangle geometry. Window coordinates (top-left, y-down) so every Lua
//! formula ports sign-identical; the y-flip to world space happens at vertex
//! emission (the flow/fish convention).
//!
//! Unlike the LÖVE original — which builds literal strings (O(n^2) concat) and
//! paints once to a canvas — this port:
//! - rewrites **ping-pong `u8` token buffers** (memory = final token count),
//! - **caps the built geometry** by a per-tree segment budget so a mis-set
//!   slider can neither OOM the CPU nor blow up VRAM (the audit's load-bearing
//!   fix: the cap is on segments, not tokens),
//! - builds every tree's geometry **in parallel** on the compute pool, then
//!   concatenates into one merged buffer the renderer uploads on a version bump,
//! - moves **colour and wind into shader uniforms** (see `render`): the
//!   hue/spread/brightness/leaf-hue/wind sliders never touch the geometry, so
//!   they update live for free.
//!
//! The original's two-signature scheme is kept: a **structural** change (growth
//! and the 5 rewrite probabilities) regrows the token streams; a **geometry**
//! change (branch angle/length/width, size variation, leaf size/density, the
//! window size) only re-walks the turtle. Both are throttled to ~15/s while a
//! slider is held (the original's `REGROW_THROTTLE`), with an exact rebuild on
//! release. Each tree is seeded per index, so structural tweaks morph trees in
//! place and adding/removing trees leaves the rest untouched.

use std::f32::consts::PI;

use bevy::prelude::*;
use bevy::tasks::ComputeTaskPool;
use bytemuck::{Pod, Zeroable};

use super::settings::ForestSettings;
use crate::app::{AppState, RestartRequested, SimBounds, sim_active, update_sim_bounds};
use crate::experiments::{CurrentExperiment, ExperimentId, experiment_active};
use crate::ui::ValueEdit;

/// Cap live geometry rebuilds while a slider is held (~15/s), with an exact
/// rebuild on release — the original's `REGROW_THROTTLE`.
const REGROW_THROTTLE: f32 = 1.0 / 15.0;

/// Hardcoded `lib/tree.lua` library defaults the minigame never exposed:
/// branch width tapers x0.8 per level, the branch angle tightens x0.95, and
/// (in `render`) brightness brightens x1.2. Not sliders — matching the original.
const WIDTH_K: f32 = 0.8;
const ANGLE_K: f32 = 0.95;

/// The 1px feather to transparent on each edge of a branch — LÖVE's "smooth"
/// line profile (the fish/flow line art language), in place of the original's
/// hard aliased rectangles.
const FEATHER: f32 = 1.0;

/// Total built F-segments across the whole forest, hard-capped. Bounds BOTH the
/// GPU vertex buffer (~`MAX_SEGMENTS` x 8 verts x 12 B) AND a single fat tree's
/// turtle/regrow time (the two perf walls the audit identified). A per-tree
/// budget = `MAX_SEGMENTS / count` stops token expansion early. Measured —
/// tuned in the perf loop (see ARCHITECTURE.md); the `Count`/`Growth` slider
/// maxes are set so the typical case never approaches it, but heavy probs can.
const MAX_SEGMENTS: u64 = 2_000_000;

// L-system tokens (the original's S/F/B and the l/r/c/[/] turtle commands).
const TOK_S: u8 = 0;
const TOK_F: u8 = 1;
const TOK_B: u8 = 2;
const TOK_L: u8 = 3;
const TOK_R: u8 = 4;
const TOK_C: u8 = 5;
const TOK_PUSH: u8 = 6;
const TOK_POP: u8 = 7;

/// A tiny deterministic xorshift64* — reproducible in tests, no `rand` in the
/// model path (the flow convention). Lua RNG-parity is waived (per the brief);
/// the user-visible *morph-in-place* property comes from the per-tree seed.
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed | 1) // avoid the all-zero fixed point
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    /// A float in [0, 1).
    fn f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }
}

/// SplitMix64 finalizer — decorrelates the per-tree seeds derived from the
/// forest seed so adjacent trees don't share an RNG stream.
fn splitmix(mut z: u64) -> u64 {
    z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// The five per-character rewrite probabilities (the original's `probs`).
#[derive(Clone, Copy)]
struct Probs {
    no_expand: f32,
    forward: f32,
    left: f32,
    right: f32,
    center: f32,
}

impl Probs {
    fn of(s: &ForestSettings) -> Self {
        Self {
            no_expand: s.no_expand,
            forward: s.forward,
            left: s.branch_left,
            right: s.branch_right,
            center: s.branch_center,
        }
    }
}

/// Append `B`'s replacement: one of three branch shapes chosen by the relative
/// left/right/centre weights (the original's `pickBranch` + `BRANCH`). Returns
/// how many `F` segments it added (for the budget). `l+r+c <= 0` ⇒ `B` rests.
fn pick_branch(out: &mut Vec<u8>, rng: &mut Rng, p: &Probs) -> u64 {
    let total = p.left + p.right + p.center;
    if total <= 0.0 {
        out.push(TOK_B); // no branching configured; the node rests
        return 0;
    }
    let x = rng.f32() * total;
    if x < p.left {
        // [llFB][rFB] — weighted left
        out.extend_from_slice(&[
            TOK_PUSH, TOK_L, TOK_L, TOK_F, TOK_B, TOK_POP, TOK_PUSH, TOK_R, TOK_F, TOK_B, TOK_POP,
        ]);
        2
    } else if x < p.left + p.right {
        // [lFB][rrFB] — weighted right
        out.extend_from_slice(&[
            TOK_PUSH, TOK_L, TOK_F, TOK_B, TOK_POP, TOK_PUSH, TOK_R, TOK_R, TOK_F, TOK_B, TOK_POP,
        ]);
        2
    } else {
        // [llFB][cFB][rrFB] — symmetric three-way
        out.extend_from_slice(&[
            TOK_PUSH, TOK_L, TOK_L, TOK_F, TOK_B, TOK_POP, TOK_PUSH, TOK_C, TOK_F, TOK_B, TOK_POP,
            TOK_PUSH, TOK_R, TOK_R, TOK_F, TOK_B, TOK_POP,
        ]);
        3
    }
}

/// One L-system iteration: rewrite `tokens` into `scratch`, swap, return the new
/// `F`-segment count. The rewrite ORDER matches `lib/tree.lua:147-155` exactly —
/// `S` first (no `no_expand` draw), then every other character draws `no_expand`
/// (and may rest), then `F`/`B` specialise. Structural characters consume the
/// `no_expand` draw and copy through, keeping the RNG stream shape (so the
/// morph-in-place property holds across a growth change).
fn expand_once(tokens: &mut Vec<u8>, scratch: &mut Vec<u8>, rng: &mut Rng, p: &Probs) -> u64 {
    scratch.clear();
    let mut f_count = 0u64;
    for &c in tokens.iter() {
        match c {
            TOK_S => {
                scratch.push(TOK_F);
                scratch.push(TOK_B);
                f_count += 1;
            }
            TOK_F => {
                if rng.f32() < p.no_expand {
                    scratch.push(TOK_F); // rests
                    f_count += 1;
                } else if rng.f32() < p.forward {
                    scratch.push(TOK_F);
                    scratch.push(TOK_F);
                    f_count += 2;
                } else {
                    scratch.push(TOK_F);
                    f_count += 1;
                }
            }
            TOK_B => {
                if rng.f32() < p.no_expand {
                    scratch.push(TOK_B); // rests
                } else {
                    f_count += pick_branch(scratch, rng, p);
                }
            }
            other => {
                // l r c [ ] — Lua draws one no_expand test then copies through.
                let _ = rng.f32();
                scratch.push(other);
            }
        }
    }
    std::mem::swap(tokens, scratch);
    f_count
}

/// Grow a tree from the axiom `S` for `growth` iterations, stopping early once
/// the projected `F`-segment count exceeds `budget`. Returns the final segment
/// count. Always a full re-expand (never an incremental extend) so a per-tree
/// re-seed reproduces the same stream — the morph-in-place contract.
fn expand(
    tokens: &mut Vec<u8>,
    scratch: &mut Vec<u8>,
    rng: &mut Rng,
    p: &Probs,
    growth: u32,
    budget: u64,
) -> u64 {
    tokens.clear();
    tokens.push(TOK_S);
    let mut f_count = 0u64;
    for _ in 0..growth {
        f_count = expand_once(tokens, scratch, rng, p);
        if f_count > budget {
            break;
        }
    }
    f_count
}

/// One triangle vertex: world position + a packed `u32` the vertex shader
/// unpacks (the colour and wind are computed there, not stored). 12 bytes — the
/// flock/fish budget. `packed` layout (little-endian):
/// - bits  0..6: branch level (depth; the shader's hue/brightness step)
/// - bit      7: leaf flag (the shader uses the leaf hue instead)
/// - bits 8..15: coreness 0/255 (premultiplied alpha — 0 at the feather edge)
/// - bits 16..23: sway weight (u8 unorm — the wind displacement scale)
/// - bits 24..31: per-tree wind phase (u8 — so trees don't sway in lockstep)
#[derive(Clone, Copy, Pod, Zeroable)]
#[repr(C)]
pub struct ForestVertex {
    pub pos: [f32; 2],
    pub packed: u32,
}

fn pack(level: u8, leaf: bool, core: u8, sway: f32, phase: u8) -> u32 {
    // Clamp to 7 bits so a deep nesting can never spill into the leaf flag
    // (the budget caps nesting far below 127 today; this is belt-and-braces).
    let level = level.min(0x7f) as u32 | if leaf { 0x80 } else { 0 };
    let sway = (sway.clamp(0.0, 1.0) * 255.0).round() as u32;
    level | ((core as u32) << 8) | (sway << 16) | ((phase as u32) << 24)
}

/// Sway weight for a window-space point: how high up the window it sits (the
/// tree base is on the bottom edge), so the canopy sways and the trunk barely
/// does. The shader concentrates the bend further toward the tips. It is a
/// function of height ALONE — the wind is a height-weighted horizontal shear, so
/// two coincident points (a branch junction) always move together (no cracks),
/// and horizontal limbs keep their length (the whole tree leans, it doesn't
/// ripple). The per-tree phase, not the x position, gives the variety.
fn sway_weight(p: Vec2, vh: f32) -> f32 {
    (1.0 - p.y / vh.max(1.0)).clamp(0.0, 1.0)
}

/// A tree's wind phase byte (0..255 → 0..2π in the shader), derived from its
/// seed so each tree leans on its own clock — the variety the old per-x phase
/// faked with a travelling wave that visibly stretched horizontal branches.
fn wind_phase(seed: u64) -> u8 {
    (splitmix(seed ^ 0x117D_C0DE_57AB_CDEF) >> 56) as u8
}

/// A tree's built geometry (one parallel build task fills one of these).
#[derive(Default)]
struct TreeGeo {
    verts: Vec<ForestVertex>,
    idx: Vec<u32>,
}

/// Window→world y-flip (origin at the window centre — the flow/fish emit
/// convention) and push a vertex; returns its index.
fn push_vertex(geo: &mut TreeGeo, p: Vec2, bounds: Vec2, packed: u32) -> u32 {
    let index = geo.verts.len() as u32;
    geo.verts.push(ForestVertex {
        pos: [p.x - bounds.x * 0.5, bounds.y * 0.5 - p.y],
        packed,
    });
    index
}

/// Emit one branch run (a maximal straight, same-level chain of `F`s coalesced
/// into a single quad) as a feathered ribbon: edge / core / core / edge across
/// each end, alpha 0 at the feather edge, 1 at the solid core.
fn emit_segment(geo: &mut TreeGeo, p0: Vec2, p1: Vec2, width: f32, level: u8, bounds: Vec2, phase: u8) {
    let dir = p1 - p0;
    if dir.length_squared() < 1e-9 {
        return;
    }
    let normal = dir.perp().normalize(); // window coords; flipped with y at emit
    let hw = (width * 0.5).max(0.3);
    let row = |geo: &mut TreeGeo, p: Vec2| -> [u32; 4] {
        let sway = sway_weight(p, bounds.y);
        [
            push_vertex(geo, p + normal * (hw + FEATHER), bounds, pack(level, false, 0, sway, phase)),
            push_vertex(geo, p + normal * hw, bounds, pack(level, false, 255, sway, phase)),
            push_vertex(geo, p - normal * hw, bounds, pack(level, false, 255, sway, phase)),
            push_vertex(geo, p - normal * (hw + FEATHER), bounds, pack(level, false, 0, sway, phase)),
        ]
    };
    let a = row(geo, p0);
    let b = row(geo, p1);
    for s in 0..3 {
        geo.idx
            .extend_from_slice(&[a[s], a[s + 1], b[s], a[s + 1], b[s + 1], b[s]]);
    }
}

/// Emit a soft round leaf at a twig tip: a 4-triangle fan, opaque core centre
/// fading to transparent at the rim, swaying with the tip.
fn emit_leaf(geo: &mut TreeGeo, p: Vec2, radius: f32, bounds: Vec2, phase: u8) {
    let sway = sway_weight(p, bounds.y);
    let c = push_vertex(geo, p, bounds, pack(0, true, 255, sway, phase));
    let rim: Vec<u32> = [Vec2::Y, Vec2::X, Vec2::NEG_Y, Vec2::NEG_X]
        .iter()
        .map(|&d| push_vertex(geo, p + d * radius, bounds, pack(0, true, 0, sway, phase)))
        .collect();
    for i in 0..4 {
        geo.idx.extend_from_slice(&[c, rim[i], rim[(i + 1) % 4]]);
    }
}

/// Deterministic [0,1) hash for a tip — so leaves don't flicker between
/// rebuilds (a per-tree, per-tip-index value, independent of the wall clock).
fn tip_hash(seed: u64, tip: u64) -> f32 {
    let h = splitmix(seed ^ splitmix(tip.wrapping_mul(0x100_0001)));
    (h >> 40) as f32 / (1u64 << 24) as f32
}

/// Walk a tree's token stream with a turtle, emitting feathered branch quads
/// (straight runs coalesced) and, if enabled, leaves at the twig tips. Pure of
/// the tree's mutable state: placement + per-tree scale come from `seed`, so a
/// geometry rebuild never teleports a tree.
fn build_geometry(
    tokens: &[u8],
    seed: u64,
    s: &ForestSettings,
    bounds: Vec2,
    geo: &mut TreeGeo,
) {
    geo.verts.clear();
    geo.idx.clear();
    let (vw, vh) = (bounds.x, bounds.y);

    // Placement RNG, a stream independent of the growth RNG so changing growth
    // never moves a tree (the original seeds placement off the same per-tree
    // seed; here it's a decorrelated sub-stream).
    let mut place = Rng::new(splitmix(seed ^ 0x5EED_F04E_57AB_CDEF));
    let base = Vec2::new(20.0 + place.f32() * (vw - 40.0).max(0.0), vh);
    let scale = 1.0 + (place.f32() * 2.0 - 1.0) * s.size_variation;

    let branch_len = s.branch_length * scale;
    let trunk_w = s.trunk_width * scale;
    let leaf_r = s.leaf_size * scale;
    let phase = wind_phase(seed);

    // Turtle state. The heading starts pointing up (the original's `angle - pi`,
    // with forward = local +y). `[`/`]` push/pop the whole state, so the
    // width/angle taper and the level restore automatically on pop.
    #[derive(Clone, Copy)]
    struct T {
        pos: Vec2,
        angle: f32,
        width: f32,
        bangle: f32,
        level: u8,
    }
    let mut t = T {
        pos: base,
        angle: -PI,
        width: trunk_w,
        bangle: s.branch_angle * PI,
        level: 0,
    };
    let mut stack: Vec<T> = Vec::new();

    // Straight-run coalescing: consecutive `F`s with no turn/branch between them
    // are one quad (visually identical to the Lua's per-segment rectangles, far
    // fewer vertices when `forward` makes long limbs).
    let mut run_start: Option<Vec2> = None;
    let mut run_width = 0.0;
    let mut run_level = 0u8;
    let mut tip = 0u64;

    for &c in tokens {
        match c {
            TOK_F => {
                if run_start.is_none() {
                    run_start = Some(t.pos);
                    run_width = t.width;
                    run_level = t.level;
                }
                let heading = Vec2::new(-t.angle.sin(), t.angle.cos());
                t.pos += heading * branch_len;
            }
            _ => {
                // Any non-F flushes the current straight run.
                if let Some(p0) = run_start.take() {
                    emit_segment(geo, p0, t.pos, run_width, run_level, bounds, phase);
                }
                match c {
                    TOK_L => t.angle -= t.bangle,
                    TOK_R => t.angle += t.bangle,
                    TOK_C => {}
                    TOK_PUSH => {
                        stack.push(t);
                        t.width *= WIDTH_K;
                        t.bangle *= ANGLE_K;
                        t.level = t.level.saturating_add(1);
                    }
                    TOK_POP => {
                        if let Some(prev) = stack.pop() {
                            t = prev;
                        }
                    }
                    TOK_B => {
                        // An un-expanded `B` is a growth tip — a leaf candidate.
                        tip += 1;
                        if s.leaves && tip_hash(seed, tip) < s.leaf_density {
                            emit_leaf(geo, t.pos, leaf_r, bounds, phase);
                        }
                    }
                    _ => {}
                }
            }
        }
    }
    if let Some(p0) = run_start.take() {
        emit_segment(geo, p0, t.pos, run_width, run_level, bounds, phase);
    }
}

/// What forces a regrow (the original's `structSig`): growth + the 5 rewrite
/// probabilities. Bit-exact f32 equality is fine — settings only change to
/// discrete slider values.
#[derive(Clone, Copy, PartialEq, Debug)]
struct StructSig {
    growth: u32,
    no_expand: f32,
    forward: f32,
    left: f32,
    right: f32,
    center: f32,
}

impl StructSig {
    fn of(s: &ForestSettings) -> Self {
        Self {
            growth: s.growth.floor().max(0.0) as u32,
            no_expand: s.no_expand,
            forward: s.forward,
            left: s.branch_left,
            right: s.branch_right,
            center: s.branch_center,
        }
    }
}

/// What forces a turtle re-walk (no regrow): branch geometry, per-tree scale,
/// the leaf geometry, and the window size (tree placement depends on it). Colour
/// (hue/spread/brightness/leaf-hue) and wind are NOT here — they're uniforms.
#[derive(Clone, Copy, PartialEq, Debug)]
struct GeoSig {
    branch_angle: f32,
    branch_length: f32,
    trunk_width: f32,
    size_variation: f32,
    leaves: bool,
    leaf_size: f32,
    leaf_density: f32,
    bounds: [f32; 2],
}

impl GeoSig {
    fn of(s: &ForestSettings, bounds: Vec2) -> Self {
        Self {
            branch_angle: s.branch_angle,
            branch_length: s.branch_length,
            trunk_width: s.trunk_width,
            size_variation: s.size_variation,
            leaves: s.leaves,
            leaf_size: s.leaf_size,
            leaf_density: s.leaf_density,
            bounds: [bounds.x, bounds.y],
        }
    }
}

/// One tree: its seed and cached token stream (the expensive thing to rebuild),
/// plus the structural signature the tokens were grown at. Geometry is NOT
/// cached per tree — it's rebuilt into a shared merged buffer each change (cheap
/// relative to a regrow, and a fraction of the memory).
struct Tree {
    seed: u64,
    tokens: Vec<u8>,
    token_sig: Option<StructSig>,
    segments: u64,
}

impl Tree {
    fn new(seed: u64) -> Self {
        Self {
            seed,
            tokens: Vec::new(),
            token_sig: None,
            segments: 0,
        }
    }
}

/// The whole forest: its trees, the merged geometry the renderer uploads on a
/// `version` bump, the wind clock, and the applied signatures that gate rebuilds.
#[derive(Resource, Default)]
pub struct Forest {
    trees: Vec<Tree>,
    seed: u64,
    seeded: bool,
    pub vertices: Vec<ForestVertex>,
    pub indices: Vec<u32>,
    pub version: u64,
    pub total_segments: u64,
    /// Accumulated wind time (advances in Menu + Playing, freezes in Options).
    pub wind_time: f32,
    applied_struct: Option<StructSig>,
    applied_geo: Option<GeoSig>,
    applied_count: usize,
    last_build: f32,
}

impl Forest {
    /// Pick a fresh seed and drop the trees so the next update grows a
    /// brand-new forest (the original's `reseed` + `regrow`).
    fn reseed(&mut self) {
        self.seed = rand::Rng::random::<u64>(&mut rand::rng());
        self.seeded = true;
        self.trees.clear();
        self.applied_struct = None;
        self.applied_geo = None;
    }

    /// Number of trees the forest currently holds (the score).
    pub fn tree_count(&self) -> usize {
        self.trees.len()
    }

    /// Grow/build the forest to the current settings: resize the tree list,
    /// regrow only the trees whose structure changed (parallel), re-walk every
    /// tree's turtle into geometry (parallel), concatenate, and bump the version.
    fn rebuild(&mut self, s: &ForestSettings, bounds: Vec2, n: usize, ssig: StructSig, gsig: GeoSig) {
        while self.trees.len() < n {
            let i = self.trees.len() as u64;
            let seed = self.seed ^ splitmix(i + 1);
            self.trees.push(Tree::new(seed));
        }
        self.trees.truncate(n);

        let budget = (MAX_SEGMENTS / n.max(1) as u64).max(1024);
        let probs = Probs::of(s);
        let growth = ssig.growth.min(64);

        // Regrow pass: re-expand the token stream of any tree whose structural
        // signature is stale (a fresh tree's is `None`). Parallel — each tree's
        // RNG stream is independent.
        let pool = ComputeTaskPool::get_or_init(Default::default);
        pool.scope(|scope| {
            for tree in self.trees.iter_mut() {
                scope.spawn(async move {
                    if tree.token_sig != Some(ssig) {
                        let mut scratch = Vec::new();
                        tree.segments = expand(
                            &mut tree.tokens,
                            &mut scratch,
                            &mut Rng::new(tree.seed),
                            &probs,
                            growth,
                            budget,
                        );
                        tree.token_sig = Some(ssig);
                    }
                });
            }
        });
        self.total_segments = self.trees.iter().map(|t| t.segments).sum();

        // Build pass: walk every tree's turtle into its own geometry buffer.
        let mut outs: Vec<TreeGeo> = (0..n).map(|_| TreeGeo::default()).collect();
        pool.scope(|scope| {
            for (tree, out) in self.trees.iter().zip(outs.iter_mut()) {
                scope.spawn(async move {
                    build_geometry(&tree.tokens, tree.seed, s, bounds, out);
                });
            }
        });

        // Concatenate into the merged buffer (offset each tree's indices).
        self.vertices.clear();
        self.indices.clear();
        for out in &outs {
            let base = self.vertices.len() as u32;
            self.vertices.extend_from_slice(&out.verts);
            self.indices.extend(out.idx.iter().map(|i| i + base));
        }

        self.version = self.version.wrapping_add(1);
        self.applied_struct = Some(ssig);
        self.applied_geo = Some(gsig);
        self.applied_count = n;
    }
}

#[derive(SystemSet, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ForestSimSet;

pub fn plugin(app: &mut App) {
    app.init_resource::<Forest>()
        .add_systems(
            Update,
            (
                // handle_restart runs in every screen (the popup's New-forest /
                // Restart must reseed from Options too); the R key inside it is
                // Playing-only. The geometry rebuilds in Options as well (the
                // popup sliders are the controls, redrawing live, like flow).
                // Wind advances in Menu + Playing, freezing in Options.
                handle_restart,
                update_forest,
                advance_wind.run_if(sim_active),
            )
                .chain()
                .in_set(ForestSimSet)
                .after(update_sim_bounds)
                .run_if(experiment_active(ExperimentId::Forest)),
        )
        // Ungated: must fire on the frame forest STOPS being current, or its
        // baked geometry would draw over the next experiment (all experiments
        // share one Transparent2d sort key) — the flow pattern.
        .add_systems(Update, clear_when_inactive);
}

/// [R] / the popup's Restart / the menu's entry: a brand-new forest (a fresh
/// seed, then a full regrow on the next `update_forest`).
fn handle_restart(
    keys: Res<ButtonInput<KeyCode>>,
    state: Res<State<AppState>>,
    text_edits: Query<(), With<ValueEdit>>,
    mut request: ResMut<RestartRequested>,
    mut forest: ResMut<Forest>,
) {
    // An "r" typed into a value edit is not a restart.
    let key_restart = *state.get() == AppState::Playing
        && keys.just_pressed(KeyCode::KeyR)
        && text_edits.is_empty();
    if !(request.0 || key_restart) {
        return;
    }
    request.0 = false;
    forest.reseed();
}

/// Rebuild the forest when a structural / geometry tunable, the tree count, or
/// the window size changed — throttled while a slider is held, exact on release.
/// Colour and wind never reach here (they're shader uniforms).
fn update_forest(
    settings: Res<ForestSettings>,
    bounds: Res<SimBounds>,
    time: Res<Time>,
    mouse: Res<ButtonInput<MouseButton>>,
    mut forest: ResMut<Forest>,
) {
    if !forest.seeded {
        forest.reseed();
    }
    let n = settings.count.round().max(1.0) as usize;
    let ssig = StructSig::of(&settings);
    let gsig = GeoSig::of(&settings, bounds.0);

    let unchanged = forest.applied_struct == Some(ssig)
        && forest.applied_geo == Some(gsig)
        && forest.applied_count == n
        && !forest.trees.is_empty();
    if unchanged {
        return;
    }

    let now = time.elapsed_secs();
    // Live preview while dragging, throttled so a heavy forest isn't rebuilt
    // every frame (the first build is never throttled).
    if !forest.trees.is_empty()
        && mouse.pressed(MouseButton::Left)
        && now - forest.last_build < REGROW_THROTTLE
    {
        return;
    }

    forest.rebuild(&settings, bounds.0, n, ssig, gsig);
    forest.last_build = now;
}

/// Advance the wind clock (Playing + behind the menu; frozen in Options).
fn advance_wind(time: Res<Time>, mut forest: ResMut<Forest>) {
    forest.wind_time += time.delta_secs();
}

/// Drop the baked geometry when another experiment takes over; returning
/// rebuilds fresh (the cleared signatures force it). The version bump makes the
/// renderer upload the empty buffer, so nothing of the forest lingers.
fn clear_when_inactive(current: Res<CurrentExperiment>, mut forest: ResMut<Forest>) {
    if !current.is_changed() || current.0 == ExperimentId::Forest {
        return;
    }
    forest.vertices.clear();
    forest.indices.clear();
    forest.trees.clear();
    forest.total_segments = 0;
    forest.applied_struct = None;
    forest.applied_geo = None;
    forest.applied_count = 0;
    forest.version = forest.version.wrapping_add(1);
}

/// Port of `lib/color.lua`'s `hsl2rgb` (h in degrees, s/l in percent), used by
/// the tests and any CPU rasterizer dump; the live renderer runs the identical
/// formula in WGSL. Channels are NOT clamped here — LÖVE clamps at draw time, so
/// callers clamp after (brightness x1.2^level routinely exceeds l=100%).
#[cfg(test)]
fn hsl2rgb(h: f32, s: f32, l: f32) -> [f32; 3] {
    let h = (h / 360.0).rem_euclid(1.0);
    let s = s / 100.0;
    let l = l / 100.0;
    if s == 0.0 {
        return [l, l, l];
    }
    let q = if l < 0.5 { l * (1.0 + s) } else { l + s - l * s };
    let p = 2.0 * l - q;
    let hue = |t: f32| {
        let t = t.rem_euclid(1.0);
        if t < 1.0 / 6.0 {
            p + (q - p) * 6.0 * t
        } else if t < 1.0 / 2.0 {
            q
        } else if t < 2.0 / 3.0 {
            p + (q - p) * (2.0 / 3.0 - t) * 6.0
        } else {
            p
        }
    };
    [hue(h + 1.0 / 3.0), hue(h), hue(h - 1.0 / 3.0)]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn probs() -> Probs {
        Probs {
            no_expand: 0.1,
            forward: 0.4,
            left: 0.5,
            right: 0.5,
            center: 0.0,
        }
    }

    fn count(tokens: &[u8], tok: u8) -> usize {
        tokens.iter().filter(|&&c| c == tok).count()
    }

    /// The axiom expands `S -> FB` on the first iteration, no `no_expand` draw
    /// for S (it's checked before the rest branch, like the Lua).
    #[test]
    fn axiom_expands_to_fb() {
        let mut t = vec![TOK_S];
        let mut scratch = Vec::new();
        let mut rng = Rng::new(1);
        let p = probs();
        expand_once(&mut t, &mut scratch, &mut rng, &p);
        assert_eq!(t, vec![TOK_F, TOK_B]);
    }

    /// `pick_branch` honours the weights: left-only always emits the left shape
    /// (2 F, opens with PUSH L L), and `l+r+c <= 0` leaves a resting `B`.
    #[test]
    fn pick_branch_weighting() {
        let mut rng = Rng::new(42);
        let left = Probs { left: 1.0, right: 0.0, center: 0.0, ..probs() };
        for _ in 0..50 {
            let mut out = Vec::new();
            let f = pick_branch(&mut out, &mut rng, &left);
            assert_eq!(f, 2);
            assert_eq!(&out[..3], &[TOK_PUSH, TOK_L, TOK_L]);
        }
        let none = Probs { left: 0.0, right: 0.0, center: 0.0, ..probs() };
        let mut out = Vec::new();
        assert_eq!(pick_branch(&mut out, &mut rng, &none), 0);
        assert_eq!(out, vec![TOK_B]);
    }

    /// Same seed ⇒ identical tokens; different tree indices ⇒ independent
    /// streams (so trees don't clone each other).
    #[test]
    fn expansion_is_deterministic_and_independent() {
        let p = probs();
        let build = |seed: u64| {
            let mut t = Vec::new();
            let mut s = Vec::new();
            expand(&mut t, &mut s, &mut Rng::new(seed), &p, 8, MAX_SEGMENTS);
            t
        };
        let seed_a = 100u64 ^ splitmix(1);
        let seed_b = 100u64 ^ splitmix(2);
        assert_eq!(build(seed_a), build(seed_a));
        assert_ne!(build(seed_a), build(seed_b));
    }

    /// Morph-in-place: the first N iterations of a deeper grow are byte-identical
    /// to the shallower grow (the audit's FD5 — assert draw-sequence prefix
    /// stability, not token-suffix equality).
    #[test]
    fn growth_morphs_in_place() {
        let p = probs();
        let seed = 7u64;
        let mut a = vec![TOK_S];
        let (mut sa, mut ra) = (Vec::new(), Rng::new(seed));
        for _ in 0..8 {
            expand_once(&mut a, &mut sa, &mut ra, &p);
        }
        let mut b = vec![TOK_S];
        let (mut sb, mut rb) = (Vec::new(), Rng::new(seed));
        let mut snapshot_at_8 = Vec::new();
        for i in 0..9 {
            expand_once(&mut b, &mut sb, &mut rb, &p);
            if i == 7 {
                snapshot_at_8 = b.clone();
            }
        }
        assert_eq!(a, snapshot_at_8, "first 8 iterations must be identical");
        assert!(b.len() >= a.len(), "a deeper grow is at least as long");
    }

    /// The segment budget hard-stops expansion — a heavy-prob deep grow can't
    /// run away (the audit's load-bearing geometry cap).
    #[test]
    fn budget_clamps_expansion() {
        let heavy = Probs { no_expand: 0.0, forward: 1.0, left: 1.0, right: 1.0, center: 1.0 };
        let mut t = Vec::new();
        let mut s = Vec::new();
        let segs = expand(&mut t, &mut s, &mut Rng::new(3), &heavy, 30, 10_000);
        // Stops the iteration that first crosses the budget, so it overshoots by
        // at most one iteration's growth — bounded, never unbounded.
        assert!(segs <= 10_000 * 4, "segments {segs} ran away past the budget");
        assert!(count(&t, TOK_F) > 0);
    }

    /// A default tree builds non-empty, finite geometry rooted on the ground.
    #[test]
    fn geometry_is_finite_and_grounded() {
        let s = ForestSettings::default();
        let bounds = Vec2::new(1280.0, 800.0);
        let mut t = Vec::new();
        let mut scratch = Vec::new();
        expand(&mut t, &mut scratch, &mut Rng::new(123), &Probs::of(&s), 12, MAX_SEGMENTS);
        let mut geo = TreeGeo::default();
        build_geometry(&t, 123, &s, bounds, &mut geo);
        assert!(!geo.verts.is_empty() && !geo.idx.is_empty());
        for v in &geo.verts {
            assert!(v.pos[0].is_finite() && v.pos[1].is_finite());
        }
        // Every index is in range.
        let n = geo.verts.len() as u32;
        assert!(geo.idx.iter().all(|&i| i < n));
    }

    /// Structural sliders move `StructSig`; colour sliders move neither
    /// signature (they're uniforms — no rebuild).
    #[test]
    fn signatures_track_the_right_fields() {
        let base = ForestSettings::default();
        let bounds = Vec2::new(1280.0, 800.0);
        let (bs, bg) = (StructSig::of(&base), GeoSig::of(&base, bounds));

        let mut grown = base.clone();
        grown.growth += 1.0;
        assert_ne!(StructSig::of(&grown), bs);

        let mut shaped = base.clone();
        shaped.branch_angle += 0.1;
        assert_ne!(GeoSig::of(&shaped, bounds), bg);
        assert_eq!(StructSig::of(&shaped), bs);

        let mut recolored = base.clone();
        recolored.hue = 200.0;
        recolored.brightness = 40.0;
        recolored.wind = 0.7;
        assert_eq!(StructSig::of(&recolored), bs);
        assert_eq!(GeoSig::of(&recolored, bounds), bg);
    }

    /// The ported `hsl2rgb` matches `lib/color.lua` on the trunk default
    /// (hue 0, full sat, 10% light = a dark red) and greys at zero saturation.
    #[test]
    fn hsl_matches_lua() {
        let red = hsl2rgb(0.0, 100.0, 10.0);
        assert!((red[0] - 0.2).abs() < 1e-3 && red[1].abs() < 1e-3 && red[2].abs() < 1e-3);
        let grey = hsl2rgb(123.0, 0.0, 50.0);
        assert!((grey[0] - 0.5).abs() < 1e-3 && (grey[1] - 0.5).abs() < 1e-3);
    }
}
