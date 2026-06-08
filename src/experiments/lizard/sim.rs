//! The lizard simulation: a 14-joint FABRIK spine chasing the cursor (the
//! fish's proven solver with the lizard's measurements), plus the part the
//! LÖVE prototype never got right — four legs that PLANT in the world and
//! step like a lizard's. The reference `lizard.lua` dragged every foot
//! toward a drifting target with a per-frame lerp, so feet skated and never
//! planted; here each foot follows the classic procedural-walk recipe:
//!
//!  1. a shared **gait clock driven by distance travelled** — the body
//!     advances one stride per clock cycle, each foot is planted for the
//!     first [`DUTY`] of its cycle and swings through the rest, and the
//!     two trot groups run half a cycle apart. Step *rate* therefore
//!     scales with speed automatically (the Speed slider works at any
//!     value), and the diagonal pairs can never double-swing — antiphase
//!     by construction, not by a gate that could starve;
//!  2. a step target planted forward-and-out from its body segment —
//!     "where the foot would be if it had just finished a step" — led by
//!     half the stance's sweep, so the stance carries the foot
//!     front-to-back symmetrically and the extension stays bounded for
//!     any proportions, stride or speed;
//!  3. an analytic 2-bone elbow with a fixed bend side per leg — FABRIK
//!     gave the prototype uncontrolled, flippy elbows;
//!  4. the trunk bows toward the planted diagonal — the guide's "torque",
//!     as real salamander lateral undulation, layered over a clean spine.
//!
//! The body is built from a [`BodyPlan`] — per-region joint counts,
//! lengths and girths, plus leg-pair count and leg length, are live knobs
//! (the guide's "wonky salamanders"), and every gait quantity is
//! reach-relative so all of them walk correctly.
//!
//! Everything works in **window coordinates** (top-left origin, y-down),
//! like the fish; the renderer flips to world coordinates when it emits
//! vertices. The minigame mirrors the fish's UX: chase the pointer, orbit
//! it when it rests, graze for the food while it is out of the window, eat
//! and grow.

use std::f32::consts::{PI, TAU};

use bevy::prelude::*;
use bevy::window::PrimaryWindow;
use rand::Rng;

use super::settings::LizardSettings;
use crate::app::{
    AppState, PinnedAttractor, RestartRequested, SimBounds, sim_active, update_sim_bounds,
};
use crate::experiments::{ExperimentId, experiment_active};

/// The reference lizard's per-joint half-width profile — argonautcode's
/// `Lizard.pde`, recreating TheRujiK's gecko. A single CONTINUOUS curve:
/// rounded snout (52) → cheek bulge (58) → neck pinch (40) → barrel swell
/// (60·68·**71**·65·50) → long thin tail tapering to a fine point
/// (28·15·11·9·7·7). This IS the lizard's silhouette: the renderer splines
/// a smooth rail through these widths and rounds the snout and tail caps.
/// There are no per-region knobs — one lizard, the reference's exact shape.
pub const REFERENCE_WIDTHS: [f32; 14] = [
    52.0, 58.0, 40.0, 60.0, 68.0, 71.0, 65.0, 50.0, 28.0, 15.0, 11.0, 9.0, 7.0, 7.0,
];
/// Unscaled distance between consecutive joints (`link_size = 64`) — 13
/// links for the 14 joints.
pub const DEFAULT_LINK_SIZE: f32 = 64.0;
/// How sharply consecutive segments may bend (`angle_constraint = π/8`).
pub const ANGLE_CONSTRAINT: f32 = PI / 8.0;
/// The four legs, the reference's: a front pair hung off joint 3 with 52px
/// bones (full reach 104) and a hind pair off joint 7 with 36px bones
/// (reach 72) — short, sprawled, the hind pair splaying wider.
pub const FRONT_ATTACH: usize = 3;
pub const HIND_ATTACH: usize = 7;
pub const FRONT_BONE: f32 = 52.0;
pub const HIND_BONE: f32 = 36.0;

/// Shoulder sockets tuck this far inside the body edge (lizard.lua's -20).
const SHOULDER_INSET: f32 = -20.0;
/// A step target is clamped within this fraction of the leg's full reach
/// from the live shoulder, so every plant is reachable whatever the body
/// plan says (the prototype's targets sat *outside* reach — one reason its
/// legs never worked).
const REACH_CLAMP_K: f32 = 0.8;
/// The gait clock: each leg is planted for the first `DUTY` of its cycle
/// and swings through the rest. With the two trot groups half a cycle
/// apart and `DUTY > 0.5`, the swing windows cannot overlap — the
/// diagonal alternation is antiphase by construction, at ANY speed,
/// stride or proportion. The old per-leg drift triggers, diagonal gate
/// and overreach valve starved one group whenever speed or reach grew
/// past the defaults; the clock replaces all three.
const DUTY: f32 = 0.65;
/// How far the body travels per gait cycle, in units of the shortest
/// leg's reach (times the Stride slider). The foot plants half the
/// stance's sweep AHEAD of its socket, so the stance carries it
/// front-to-back symmetrically: peak extension ≈
/// √((DUTY·stride/2)² + (splay·reach)²), which stays inside the reach
/// across the whole slider range — the legs can't be outrun.
const STRIDE_K: f32 = 1.3;
/// Stance backstop: a planted foot is pulled onto this radius if a hard
/// turn at maximum stride ever stretches it past it (a momentary
/// millimetre of slide beats a leg pinned straight). Steady walks never
/// touch it — the symmetric sweep peaks well inside.
const STANCE_CLAMP_K: f32 = 0.98;
/// When the body stalls mid-swing the distance clock freezes; the foot
/// finishes its swing over this many seconds instead, so the lizard
/// settles with every foot planted, never hanging mid-air.
const SETTLE_SWING_TIME: f32 = 0.12;
/// The clock never winds more than this per frame — just under the swing
/// window (1-DUTY), so no window can be skipped outright. Only a
/// degenerate speed/scale ratio hits it (a millimetre-legged lizard told
/// to sprint); the gait then slows gracefully and the reach clamp slides
/// the feet, instead of legs silently never stepping.
const MAX_CLOCK_STEP: f32 = 0.33;
/// Turning winds the clock too, at two strides per full revolution: a
/// hairpin pivot swings the leg sockets a long way while the body's
/// centre barely travels, and on distance alone the feet stayed pinned
/// behind the swing — legs stretched straight up or folded across the
/// trunk mid-turn. With turn fuel the lizard steps its feet around the
/// pivot like the real animal.
const TURN_FUEL: f32 = 2.0 / TAU;
/// The guide's step-5 "torque", made anatomically real: a walking
/// salamander's trunk bends *concave toward the diagonal pair that is
/// planted* (fire-salamander kinematics — when the right fore is in
/// stance the trunk is concave left, and it flips each half-stride). We
/// drive a single signed `bend` ∈ [-1, 1] off which trot group is
/// swinging and lay it over the spine as a standing lateral bow. The
/// shoulders rock, the tail counter-swings, and footfalls and body-bends
/// stay locked in phase — the side-to-side roll argonaut's recreation
/// only gets weakly (and emergently) from chain lag. Kept GENTLE: the
/// reference's sway is mostly the head chasing the cursor through FABRIK
/// lag; this just adds a subtle, tail-weighted wag over it (not the old
/// violent S).
const UNDULATION_AMP: f32 = 16.0;
/// How fast `bend` eases toward the stance side (per second). Quick
/// enough to track the trot, slow enough to read as weight shifting.
const BEND_RATE: f32 = 9.0;
/// Shapes the bow along the body: the lateral offset grows as
/// `(s)^UNDULATION_SHAPE` from head (0) to tail (1). >1 keeps the trunk
/// between the leg girdles nearly straight — so the planted shoulders
/// barely move and the gait stays locked — while the tail carries most of
/// the swing, the lively part a walking lizard actually wags. The whole
/// bow is then scaled by the live Wiggle knob (`0` = a rigid spine, no
/// side-to-side at all).
const UNDULATION_SHAPE: f32 = 1.8;
/// How far the toes splay outward from the body's travel direction.
const TOE_SPLAY: f32 = 0.35;
/// How fast the toes ease toward the travel direction (per second) — even
/// while planted, so a foot turns with the body through a turn instead of
/// staying frozen pointing backward.
const TOE_TURN_RATE: f32 = 9.0;

/// Chase speed is coupled LINEARLY to size: `speed = SPEED_PER_SCALE × scale`.
/// So a freshly-spawned size-0.1 lizard chases at 50 px/s, and as it eats and
/// grows its speed rises with it (size 0.5 → 250 px/s, and so on). Speed is no
/// longer an independent control — it's a pure function of how big the lizard
/// has grown.
pub const SPEED_PER_SCALE: f32 = 500.0;

/// Food spawn inset from the screen edges (the fish's convention).
const FOOD_PAD: f32 = 50.0;
/// Head-to-food distance that counts as eating (test-lizard.lua's 20).
const EAT_DIST: f32 = 20.0;

// ---------------------------------------------------------------------------
// Vec2 helpers — the fish's ports of lib/vec2.lua, copied: the lizard owns
// its math so the finished fish stays untouched.

/// `v:setMagnitude(m)` with a zero-vector guard.
fn set_magnitude(v: Vec2, m: f32) -> Vec2 {
    let len = v.length();
    if len > 1e-12 { v * (m / len) } else { Vec2::ZERO }
}

/// `target:deltaTarget(goal, m)`: step from `from` toward `goal` by `m` —
/// overshooting if `m` exceeds the distance, like the original.
fn delta_target(from: Vec2, goal: Vec2, m: f32) -> Vec2 {
    let d = goal - from;
    if d.length_squared() > 1e-12 {
        from + set_magnitude(d, m)
    } else {
        goal
    }
}

/// Re-seat `next` at exactly `d` from `curr`, along `curr → next`.
fn constrain_distance(curr: Vec2, next: Vec2, d: f32) -> Vec2 {
    curr - set_magnitude(curr - next, d)
}

fn simplify_angle(angle: f32) -> f32 {
    angle.rem_euclid(TAU)
}

/// How many radians to turn `angle` to reach `target`, in [-π, π].
fn relative_angle_diff(angle: f32, target: f32) -> f32 {
    PI - simplify_angle(angle + PI - target)
}

/// Clamp `angle` to within `constraint` of `target`.
fn constrain_angle(angle: f32, target: f32, constraint: f32) -> f32 {
    let diff = relative_angle_diff(angle, target);
    if diff.abs() <= constraint {
        return simplify_angle(angle);
    }
    if diff > constraint {
        return simplify_angle(target - constraint);
    }
    simplify_angle(target + constraint)
}

fn smoothstep(t: f32) -> f32 {
    let t = t.clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

// ---------------------------------------------------------------------------
// The spine — the fish's forward-pass FABRIK generalized to a Vec length
// and a per-spine bend limit, so the wonky body plans stay one config away.

#[derive(Clone)]
pub struct Spine {
    /// Joint positions, head first. All start at (0,0) like the Lua; the
    /// settle pass on reset spreads them.
    pub joints: Vec<Vec2>,
    /// Scaled distance between consecutive joints, one per segment — the
    /// per-region length knobs (long neck, short torso, long tail…) make
    /// these differ down the body.
    pub links: Vec<f32>,
    pub angle_constraint: f32,
    pub target: Vec2,
}

impl Spine {
    fn new(target: Vec2, links: Vec<f32>, angle_constraint: f32) -> Self {
        Self {
            joints: vec![Vec2::ZERO; (links.len() + 1).max(2)],
            links,
            angle_constraint,
            target,
        }
    }

    /// `fabrikForward`: snap the head to the target and re-seat each joint
    /// behind it, clamping the bend per joint. Skipped entirely while the
    /// head is within 2px of the target — the chain freezes when the
    /// lizard arrives (the orbit keeps the target moving).
    fn update(&mut self) {
        if self.joints[0].distance(self.target) < 2.0 {
            return;
        }
        self.joints[0] = self.target;
        for i in 0..self.joints.len() - 1 {
            if i > 0 {
                let prev_angle = (self.joints[i - 1] - self.joints[i]).to_angle();
                let next_angle = (self.joints[i] - self.joints[i + 1]).to_angle();
                let constrained = constrain_angle(next_angle, prev_angle, self.angle_constraint);
                self.joints[i + 1] = self.joints[i] - Vec2::from_angle(constrained);
            }
            self.joints[i + 1] =
                constrain_distance(self.joints[i], self.joints[i + 1], self.links[i]);
        }
    }

    /// A point offset from joint `i`: take the spine's heading there,
    /// rotate it by `angle_offset`, and step out by `width + len_offset` —
    /// the port of lizard.lua's `getPos`. `i` must be ≥ 1 (the heading
    /// needs the joint ahead). Used for the leg sockets/targets, which
    /// hang off the clean centerline so the lateral bow (a render-time
    /// deformation) never perturbs the gait.
    fn pos_offset(&self, i: usize, width: f32, angle_offset: f32, len_offset: f32) -> Vec2 {
        let heading = (self.joints[i - 1] - self.joints[i]).to_angle();
        self.joints[i] + Vec2::from_angle(heading + angle_offset) * (width + len_offset)
    }
}

// ---------------------------------------------------------------------------
// The body plan — the recipe a lizard is built from. The default is the
// classic salamander; everything is Vec-based so the "wonky" stretch knobs
// (leg pairs, joint count, proportions) are a settings change, not a
// rewrite.

#[derive(Clone)]
pub struct LegSpec {
    /// Spine joint the leg hangs from (0-based; the Lua's 1-based 4 / 8).
    pub attach: usize,
    /// +1 right flank, -1 left.
    pub side: f32,
    /// Unscaled bone lengths, shoulder→elbow and elbow→foot.
    pub upper: f32,
    pub lower: f32,
    /// How far out to the side the foot plants, as a fraction of reach
    /// (front legs tuck in, hind legs splay wider — the salamander
    /// sprawl). The forward lead is the shared [`LEG_LEAD`].
    pub splay: f32,
    /// Trot group: 0 = {LF, RH}, 1 = {RF, LH}.
    pub group: u8,
    /// Which side the elbow bows: front elbows bow backward-outward
    /// (`side`), hind knees forward-outward (`-side`).
    pub bend_sign: f32,
    /// Limb thickness factor for the renderer (hind legs run slimmer).
    pub girth: f32,
}

#[derive(Clone)]
pub struct BodyPlan {
    pub joint_count: usize,
    /// Unscaled per-segment spacing (joint_count - 1 entries).
    pub links: Vec<f32>,
    pub angle_constraint: f32,
    pub body_width: Vec<f32>,
    pub legs: Vec<LegSpec>,
}

impl BodyPlan {
    /// The reference lizard, fixed: 14 joints at 64px, a π/8 bend limit, the
    /// [`REFERENCE_WIDTHS`] silhouette and the four reference legs. There are
    /// no tunables — this is the one lizard, argonautcode's `Lizard.pde`.
    pub fn reference() -> Self {
        let joint_count = REFERENCE_WIDTHS.len();
        Self {
            joint_count,
            links: vec![DEFAULT_LINK_SIZE; joint_count - 1],
            angle_constraint: ANGLE_CONSTRAINT,
            body_width: REFERENCE_WIDTHS.to_vec(),
            legs: reference_legs(),
        }
    }
}

/// The reference's four legs: a front pair off joint [`FRONT_ATTACH`] with
/// 52px bones (reach 104) and a hind pair off joint [`HIND_ATTACH`] with 36px
/// bones (reach 72). The two sides of each pair sit in opposite trot groups so
/// the diagonals step together; the front elbows bow backward-outward, the
/// hind knees forward-outward — the salamander sprawl. The hind pair splays
/// wider and runs a touch slimmer.
fn reference_legs() -> Vec<LegSpec> {
    let mut legs = Vec::with_capacity(4);
    // (attach joint, bone length, splay fraction, limb girth, elbow bow dir)
    for &(attach, bone, splay, girth, bow) in &[
        (FRONT_ATTACH, FRONT_BONE, 0.45_f32, 1.0_f32, 1.0_f32),
        (HIND_ATTACH, HIND_BONE, 0.55, 0.85, -1.0),
    ] {
        for side in [-1.0_f32, 1.0] {
            // Front-left and hind-right share group 0; front-right and
            // hind-left group 1 — the diagonal pairs step together.
            let group: u8 = if (attach == FRONT_ATTACH) == (side < 0.0) {
                0
            } else {
                1
            };
            legs.push(LegSpec {
                attach,
                side,
                upper: bone,
                lower: bone,
                splay,
                group,
                bend_sign: bow * side,
                girth,
            });
        }
    }
    legs
}

// ---------------------------------------------------------------------------
// One leg.

/// Where a foot is in its gait cycle. `Stepping` commits to the target
/// captured at lift-off — re-aiming mid-step is exactly the prototype's
/// feet-never-plant bug.
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum StepState {
    Planted,
    Stepping { from: Vec2, to: Vec2, t: f32 },
}

pub struct Leg {
    pub attach: usize,
    pub side: f32,
    /// Scaled bone lengths (set_scale keeps them in step with the body).
    pub upper: f32,
    pub lower: f32,
    pub splay: f32,
    pub group: u8,
    pub bend_sign: f32,
    pub girth: f32,
    /// The foot's world position — planted (fixed) or mid-step.
    pub foot: Vec2,
    pub state: StepState,
    /// Toe direction for the renderer: frozen while planted (planted toes
    /// must not swivel as the body passes over), eased during a step.
    pub heading: Vec2,
    /// Per-frame IK outputs the renderer reads.
    pub shoulder: Vec2,
    pub elbow: Vec2,
}

impl Leg {
    fn from_spec(spec: &LegSpec) -> Self {
        Self {
            attach: spec.attach,
            side: spec.side,
            upper: spec.upper,
            lower: spec.lower,
            splay: spec.splay,
            group: spec.group,
            bend_sign: spec.bend_sign,
            girth: spec.girth,
            foot: Vec2::ZERO,
            state: StepState::Planted,
            heading: Vec2::X,
            shoulder: Vec2::ZERO,
            elbow: Vec2::ZERO,
        }
    }

    /// Full straight-line reach, shoulder to toe.
    pub fn reach(&self) -> f32 {
        self.upper + self.lower
    }

    pub fn planted(&self) -> bool {
        matches!(self.state, StepState::Planted)
    }

    /// 0..1 lift arc while stepping — a render-only cue (the sim foot
    /// stays in-plane so the 2-bone lengths always hold).
    pub fn lift(&self) -> f32 {
        match self.state {
            StepState::Planted => 0.0,
            StepState::Stepping { t, .. } => (PI * t.clamp(0.0, 1.0)).sin(),
        }
    }
}

/// Analytic 2-bone IK: the elbow by law of cosines, bowing to the
/// `bend` side of the shoulder→foot line. The distance is clamped into
/// the reachable annulus so `acos` never NaNs; past full reach the elbow
/// lies on the shoulder→foot line (a straight leg).
fn solve_elbow(shoulder: Vec2, foot: Vec2, l1: f32, l2: f32, bend: f32) -> Vec2 {
    let d_vec = foot - shoulder;
    let len = d_vec.length();
    let dir = if len > 1e-6 { d_vec / len } else { Vec2::X };
    let d_min = (l1 - l2).abs() + 1e-3;
    let d = len.clamp(d_min, (l1 + l2 - 1e-3).max(d_min));
    let cos_a = ((l1 * l1 + d * d - l2 * l2) / (2.0 * l1 * d)).clamp(-1.0, 1.0);
    shoulder + Vec2::from_angle(dir.to_angle() + bend * cos_a.acos()) * l1
}

// ---------------------------------------------------------------------------
// The lizard.

pub struct Lizard {
    pub spine: Spine,
    pub scale: f32,
    /// The "Start size" the body was last (re)built at — a live edit of
    /// the slider rebuilds at the new size; eating grows `scale` past it.
    pub spawn_scale: f32,
    pub speed: f32,
    /// Stride length: how far the body travels per gait cycle. Fixed at the
    /// authored 1.0 now (the lizard has no tuning); kept as a field so the
    /// gait tests can still drive a quick scuttle vs a long stride.
    pub stride_mul: f32,
    /// Lateral undulation amount (`0` = a rigid spine, `1` = the authored
    /// gentle wag). Fixed at 1.0; the visible amplitude lives in
    /// [`UNDULATION_AMP`].
    pub wiggle: f32,
    /// Unscaled half-widths per joint (scaled in [`Self::body_width`]).
    body_width: Vec<f32>,
    pub legs: Vec<Leg>,
    /// Persistent centerline destination the FABRIK spine chases; the
    /// lateral bow is layered on top into [`Self::ribbon`], never folded
    /// back into the centerline (the fish's base_target+wave structure).
    pub base_target: Vec2,
    /// The shared gait clock, in cycles ∈ [0, 1): advances by distance
    /// travelled over [`Self::stride`], so the cadence scales with speed
    /// by construction. Group 0 swings over the last `1-DUTY` of the
    /// cycle; group 1 half a cycle later.
    pub gait_phase: f32,
    /// Last frame's base_target — the clock's odometer.
    last_base: Vec2,
    /// Last frame's head heading — the clock's turn fuel (see
    /// [`TURN_FUEL`]).
    last_heading: f32,
    /// Signed lateral body bow, eased toward ±1 (the stance side) or 0 at
    /// rest — the gait-locked "torque".
    pub bend: f32,
    /// The spine after the lateral bow is applied: what the renderer draws
    /// and the legs hang from. The clean centerline stays in `spine`.
    pub ribbon: Vec<Vec2>,
}

impl Lizard {
    pub fn new(origin: Vec2, scale: f32, speed: f32, plan: &BodyPlan) -> Self {
        let mut lizard = Self {
            spine: Spine::new(origin, plan.links.clone(), plan.angle_constraint),
            scale: 1.0,
            spawn_scale: scale,
            speed,
            stride_mul: 1.0,
            wiggle: 1.0,
            body_width: plan.body_width.clone(),
            legs: plan.legs.iter().map(Leg::from_spec).collect(),
            base_target: origin,
            gait_phase: 0.0,
            last_base: origin,
            last_heading: 0.0,
            bend: 0.0,
            ribbon: vec![origin; plan.joint_count.max(2)],
        };
        lizard.set_scale(scale);
        // Settle the spine so the first frame has spread-out joints, then
        // plant every foot — the prototype left its arms at (0,0) ("TODO:
        // put from the start in the proper position"); here frame 1 is
        // already a coherent stance. The clock starts at 0: group 0 feet
        // plant fresh on their step targets (start of stance), group 1
        // plants half a cycle into its stance — half a stride swept back —
        // so the trot alternates from the very first step.
        lizard.spine.update();
        lizard.build_ribbon();
        lizard.last_heading =
            (lizard.spine.joints[0] - lizard.spine.joints[1]).to_angle();
        let stride = lizard.stride();
        for i in 0..lizard.legs.len() {
            let (shoulder, target) = lizard.leg_frame(&lizard.legs[i]);
            let attach = lizard.legs[i].attach;
            let travel = (lizard.spine.joints[attach - 1] - lizard.spine.joints[attach])
                .normalize_or(Vec2::X);
            let back = travel * (0.5 * stride);
            let raw = if lizard.legs[i].group & 1 == 1 {
                target - back
            } else {
                target
            };
            let reach = lizard.legs[i].reach();
            let foot = shoulder + (raw - shoulder).clamp_length_max(REACH_CLAMP_K * reach);
            let leg = &mut lizard.legs[i];
            leg.shoulder = shoulder;
            leg.foot = foot;
            leg.state = StepState::Planted;
            leg.elbow = solve_elbow(shoulder, foot, leg.upper, leg.lower, leg.bend_sign);
            leg.heading = Vec2::from_angle(TOE_SPLAY * leg.side).rotate(travel);
        }
        lizard
    }

    pub fn head(&self) -> Vec2 {
        self.spine.joints[0]
    }

    /// Scaled body half-width at joint `i`.
    pub fn body_width(&self, i: usize) -> f32 {
        self.body_width[i] * self.scale
    }

    pub fn set_scale(&mut self, scale: f32) {
        if self.scale == scale {
            return;
        }
        // Chain:setScale multiplies link sizes by the ratio; the legs'
        // bones scale the same way. Planted feet stay put — growth per
        // food is far too small to strand one.
        let factor = scale / self.scale;
        for link in &mut self.spine.links {
            *link *= factor;
        }
        for leg in &mut self.legs {
            leg.upper *= factor;
            leg.lower *= factor;
        }
        self.scale = scale;
    }

    /// How far the body travels per gait cycle: proportional to the
    /// shortest leg's reach, so the cadence scales with speed (the clock
    /// runs on distance) and the foot geometry stays bounded.
    pub fn stride(&self) -> f32 {
        let min_reach = self
            .legs
            .iter()
            .map(Leg::reach)
            .fold(f32::INFINITY, f32::min);
        STRIDE_K * self.stride_mul * min_reach
    }

    /// Move the centerline destination toward `target` by `speed·dt`,
    /// with the Lua's 1px arrival guard (lizard.lua's resolve). The gait
    /// clock runs on distance, so the cadence follows any speed — no cap.
    pub fn set_target_at_speed(&mut self, target: Vec2, dt: f32) {
        if self.base_target.distance(target) > 1.0 {
            self.base_target = delta_target(self.base_target, target, self.speed * dt);
        }
    }

    /// The bowed spine the renderer draws and the legs hang from (the
    /// clean FABRIK centerline lives in `spine`).
    pub fn joints(&self) -> &[Vec2] {
        &self.ribbon
    }

    /// One frame: wind the gait clock by the distance travelled, ease the
    /// lateral bow toward the stance side, solve the clean centerline at
    /// the target, lay the bow over it, resolve legs.
    pub fn update(&mut self, dt: f32) {
        let moved = self.base_target.distance(self.last_base);
        self.last_base = self.base_target;
        // Turn fuel: the head's swing since last frame (measured on the
        // pre-solve pose, so both samples are like-for-like).
        let heading = (self.spine.joints[0] - self.spine.joints[1]).to_angle();
        let stride = self.stride().max(1e-3);
        let turned = relative_angle_diff(self.last_heading, heading).abs();
        self.last_heading = heading;
        let fuel = moved + turned * TURN_FUEL * stride;
        let moving = fuel > 1e-4;
        if moving {
            self.gait_phase = (self.gait_phase + (fuel / stride).min(MAX_CLOCK_STEP)).fract();
        }

        // The *swinging* trot group sets the bow's sign — the trunk goes
        // concave toward the planted (stance) diagonal, the real
        // salamander's weight shift. At rest (nothing swinging) it eases
        // back to straight.
        let stepping = [0usize, 1].map(|g| {
            self.legs
                .iter()
                .any(|leg| (leg.group as usize) & 1 == g && !leg.planted())
        });
        let target_bend = match (stepping[0], stepping[1]) {
            (true, false) => 1.0,
            (false, true) => -1.0,
            _ => 0.0,
        };
        self.bend += (target_bend - self.bend) * (1.0 - (-BEND_RATE * dt).exp());

        self.spine.target = self.base_target;
        self.spine.update();
        self.build_ribbon();
        self.resolve_legs(dt, moving);
    }

    /// Lay the signed lateral bow over the clean centerline: each joint
    /// shifts to its body-relative side by the bow envelope (zero at the
    /// head, growing as `s^UNDULATION_SHAPE` toward the tail), so the
    /// shoulders stay still on the cursor while the tail wags gently. The
    /// big sway is the head chasing the cursor through FABRIK lag; this is
    /// the subtle gait-locked roll over it.
    fn build_ribbon(&mut self) {
        let j = &self.spine.joints;
        let n = j.len();
        if self.ribbon.len() != n {
            self.ribbon.resize(n, Vec2::ZERO);
        }
        let amp = UNDULATION_AMP * self.scale * self.wiggle;
        for i in 0..n {
            let s = i as f32 / (n - 1).max(1) as f32;
            let fwd = if i == 0 { j[0] - j[1] } else { j[i - 1] - j[i] };
            let perp = fwd.perp().normalize_or(Vec2::Y);
            // The trunk bows as one, its sign set by the swinging stance
            // side and its envelope leaving the head still and the tail
            // carrying the wag.
            let lateral = s.powf(UNDULATION_SHAPE) * self.bend;
            self.ribbon[i] = j[i] + perp * (amp * lateral);
        }
    }

    /// The live shoulder socket and step target of a leg, off the clean
    /// centerline. The shoulder rides the body perpendicular at the attach
    /// joint, tucked inward; the target is "where the foot would be if it
    /// had just finished a step" — planted half the stance's sweep
    /// (`DUTY·stride/2`) ahead in the travel direction and `splay` out to
    /// the side, so the stance carries the foot front-to-back
    /// symmetrically with bounded extension at any stride or speed.
    /// Public so the skeleton view can draw the guide's
    /// desired-position dots.
    pub fn leg_frame(&self, leg: &Leg) -> (Vec2, Vec2) {
        let bw = self.body_width(leg.attach);
        let shoulder = self.spine.pos_offset(
            leg.attach,
            bw,
            (PI / 2.0) * leg.side,
            SHOULDER_INSET * self.scale,
        );
        let travel = (self.spine.joints[leg.attach - 1] - self.spine.joints[leg.attach])
            .normalize_or(Vec2::X);
        // Body-lateral toward this leg's side (travel rotated +90° · side).
        let out = Vec2::new(-travel.y, travel.x) * leg.side;
        let reach = leg.reach();
        let lead = 0.5 * DUTY * self.stride();
        let raw = shoulder + travel * lead + out * (leg.splay * reach);
        // The clamp is a backstop for wonky body plans; the default
        // lead/splay keep the nominal plant near 0.6·reach, inside it.
        let target = shoulder + (raw - shoulder).clamp_length_max(REACH_CLAMP_K * reach);
        (shoulder, target)
    }

    /// The shoulder socket on the *drawn* body — the bowed [`Self::ribbon`],
    /// not the clean centerline. The legs render and solve their elbows from
    /// here so they always emerge from the body the player sees; without it
    /// the lateral bow slides the drawn body off the centerline sockets and
    /// buries one flank's legs (only the feet peek out). The gait still
    /// plants feet and tests overreach against the clean-centerline shoulder
    /// ([`Self::leg_frame`]) — the bow at the leg girdles is tail-weighted
    /// small, so the trot stays decoupled from the wag.
    fn ribbon_shoulder(&self, leg: &Leg) -> Vec2 {
        let i = leg.attach;
        let bw = self.body_width(i);
        let heading = (self.ribbon[i - 1] - self.ribbon[i]).to_angle();
        self.ribbon[i]
            + Vec2::from_angle(heading + (PI / 2.0) * leg.side) * (bw + SHOULDER_INSET * self.scale)
    }

    /// Per leg: anchor the shoulder, read the gait clock, plant or swing,
    /// solve the elbow, steer the toes.
    ///
    /// Each leg's clock is the shared phase plus half a cycle per trot
    /// group: planted through the first [`DUTY`] of its cycle, swinging
    /// through the rest. A swing chases the LIVE step target — the
    /// guide's "move the foot to the desired position", a point that
    /// travels with the body — and its progress is the clock itself, so
    /// the foot lands exactly ON the target as its stance begins, at any
    /// speed. (The prototype's skating came from a per-frame lerp with no
    /// completion, not from the target moving.)
    fn resolve_legs(&mut self, dt: f32, moving: bool) {
        for i in 0..self.legs.len() {
            // Clean-centerline shoulder/target drive the gait (plant point,
            // stance clamp); the ribbon shoulder drives what's drawn.
            let (shoulder, target) = self.leg_frame(&self.legs[i]);
            let render_shoulder = self.ribbon_shoulder(&self.legs[i]);
            let attach = self.legs[i].attach;
            let travel = (self.spine.joints[attach - 1] - self.spine.joints[attach])
                .normalize_or(Vec2::X);
            let leg = &mut self.legs[i];
            let group = (leg.group as usize) & 1;
            let reach = leg.reach();
            leg.shoulder = render_shoulder;
            // This leg's position in its own cycle.
            let lp = (self.gait_phase + 0.5 * group as f32).fract();

            match leg.state {
                StepState::Planted => {
                    if moving && lp >= DUTY {
                        leg.state = StepState::Stepping {
                            from: leg.foot,
                            to: target,
                            t: 0.0,
                        };
                    }
                }
                StepState::Stepping { from, t, .. } => {
                    // Swing progress IS the clock (u); a stalled body
                    // freezes the clock, so the foot finishes on a timer
                    // instead and the lizard settles fully planted.
                    let u = if lp >= DUTY {
                        (lp - DUTY) / (1.0 - DUTY)
                    } else {
                        1.0
                    };
                    let t = if moving {
                        u.max(t)
                    } else {
                        t + dt / SETTLE_SWING_TIME
                    };
                    if t >= 1.0 {
                        leg.foot = target;
                        leg.state = StepState::Planted;
                    } else {
                        leg.foot = from.lerp(target, smoothstep(t));
                        leg.state = StepState::Stepping {
                            from,
                            to: target,
                            t,
                        };
                    }
                }
            }

            // Whatever the state, the foot never leaves the leg's physical
            // reach — mid-swing included. A hard turn at maximum stride, or
            // a degenerate speed/scale ratio that outruns the clock, would
            // otherwise stretch it; sliding a hair beats a leg pinned
            // straight. Steady walks never touch this — the symmetric
            // sweep peaks well inside.
            let ext = shoulder.distance(leg.foot);
            let max_ext = STANCE_CLAMP_K * reach;
            if ext > max_ext {
                leg.foot = shoulder + (leg.foot - shoulder) * (max_ext / ext);
            }

            // The elbow re-solves from the live (ribbon) shoulder every
            // frame — that is what makes a planted foot read as anchored
            // while the body slides over it.
            leg.elbow = solve_elbow(render_shoulder, leg.foot, leg.upper, leg.lower, leg.bend_sign);
            // Toes point down the body's travel direction, splayed outward,
            // and ease toward it every frame (planted feet included) so they
            // turn WITH the lizard instead of staying frozen pointing
            // backward through a turn — the "feet don't rotate" report. Fast
            // enough to track a turn, slow enough to read as the foot
            // pivoting rather than snapping.
            let splayed = Vec2::from_angle(TOE_SPLAY * leg.side).rotate(travel);
            let blend = 1.0 - (-TOE_TURN_RATE * dt).exp();
            leg.heading = leg.heading.lerp(splayed, blend).normalize_or(splayed);
        }
    }
}

// ---------------------------------------------------------------------------
// The minigame — the fish's UX with the lizard walking it.

/// The one lizard (None until the first restart settles it in).
#[derive(Resource, Default)]
pub struct LizardEntity(pub Option<Lizard>);

/// Game state around the lizard: the food dot and the orbit-a-stationary-
/// pointer behaviour.
#[derive(Resource, Default)]
pub struct LizardGame {
    pub food: Vec2,
    pub eaten: u32,
    /// Orbiting a stationary pointer it has reached.
    circling: bool,
    orbit_angle: f32,
    orbit_dir: f32,
    /// Previous frame's pointer, to detect movement (while the cursor is
    /// out of the window the lizard grazes for the food instead).
    last_mouse: Option<Vec2>,
}

/// Random food position in the padded screen rect, integer coordinates
/// like the original's `love.math.random`.
fn random_food(bounds: Vec2, rng: &mut impl Rng) -> Vec2 {
    let max_x = (bounds.x - FOOD_PAD).max(FOOD_PAD) as i32;
    let max_y = (bounds.y - FOOD_PAD).max(FOOD_PAD) as i32;
    Vec2::new(
        rng.random_range(FOOD_PAD as i32..=max_x) as f32,
        rng.random_range(FOOD_PAD as i32..=max_y) as f32,
    )
}

/// Label for the lizard sim systems; the renderer rebuilds after.
#[derive(SystemSet, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct LizardSimSet;

pub fn plugin(app: &mut App) {
    app.init_resource::<LizardEntity>()
        .init_resource::<LizardGame>()
        .add_systems(
            Update,
            (handle_restart, drive_lizard)
                .chain()
                .in_set(LizardSimSet)
                .after(update_sim_bounds)
                .run_if(experiment_active(ExperimentId::Lizard))
                // Steps while playing AND behind the menu (the live backdrop).
                .run_if(sim_active),
        )
        // Live structural edits apply in EVERY state — including the options
        // popup, where the sim is frozen — so dragging a wonky slider re-poses
        // the lizard immediately instead of doing nothing until a restart.
        .add_systems(
            Update,
            reshape_lizard
                .after(LizardSimSet)
                .run_if(experiment_active(ExperimentId::Lizard)),
        );
}

/// Apply a live edit of the structural ("wonky") knobs — or of Start
/// size — the moment it happens: when the per-region joint counts,
/// lengths, girths, leg knobs or start size no longer match what the
/// lizard was built from, rebuild it in place — same position and
/// heading. This is what makes those sliders feel connected; before,
/// they only took effect on a restart the paused popup could not
/// trigger, so they read as dead.
fn reshape_lizard(settings: Res<LizardSettings>, mut lizard: ResMut<LizardEntity>) {
    let Some(current) = lizard.0.as_mut() else {
        return;
    };
    // Only Start size rebuilds in place now (the body shape is fixed): a
    // drag previews the new size live, same position and heading, instead
    // of applying only on a restart the paused popup can't trigger.
    if current.spawn_scale == settings.start_scale {
        return;
    }
    let at = current.head();
    let base = current.base_target;
    let speed = SPEED_PER_SCALE * settings.start_scale;
    let mut rebuilt = Lizard::new(at, settings.start_scale, speed, &BodyPlan::reference());
    rebuilt.base_target = base;
    rebuilt.spawn_scale = settings.start_scale;
    *current = rebuilt;
}

/// Rebuild the lizard on [R], on a UI restart request, or on first
/// activation: one lizard at screen centre, spine settled and feet
/// planted; fresh food.
fn handle_restart(
    keys: Res<ButtonInput<KeyCode>>,
    state: Res<State<AppState>>,
    settings: Res<LizardSettings>,
    bounds: Res<SimBounds>,
    mut request: ResMut<RestartRequested>,
    mut lizard: ResMut<LizardEntity>,
    mut game: ResMut<LizardGame>,
) {
    let key_restart = *state.get() == AppState::Playing && keys.just_pressed(KeyCode::KeyR);
    if !(request.0 || key_restart || lizard.0.is_none()) {
        return;
    }
    request.0 = false;

    let centre = bounds.0 / 2.0;
    lizard.0 = Some(Lizard::new(
        centre,
        settings.start_scale,
        SPEED_PER_SCALE * settings.start_scale,
        &BodyPlan::reference(),
    ));
    *game = LizardGame {
        food: random_food(bounds.0, &mut rand::rng()),
        orbit_dir: 1.0,
        ..default()
    };
}

/// The per-frame minigame update, the fish game's exact order: sync live
/// speed, eat, track the pointer, orbit it when reached and stationary,
/// then move and solve — legs included.
fn drive_lizard(
    time: Res<Time>,
    settings: Res<LizardSettings>,
    bounds: Res<SimBounds>,
    pinned: Res<PinnedAttractor>,
    window: Query<&Window, With<PrimaryWindow>>,
    mut lizard: ResMut<LizardEntity>,
    mut game: ResMut<LizardGame>,
) {
    let dt = time.delta_secs();
    let Some(lizard) = lizard.0.as_mut() else {
        return;
    };

    // Speed tracks size, every frame — so the moment a meal grows the lizard,
    // it speeds up to match (see [`SPEED_PER_SCALE`]).
    lizard.speed = SPEED_PER_SCALE * lizard.scale;

    if lizard.head().distance(game.food) < EAT_DIST {
        game.eaten += 1;

        if settings.growth_rate > 0.0 {
            let scale = lizard.scale + settings.growth_rate;
            lizard.set_scale(scale);
        }

        // Respawn at least 200px away from where it was eaten.
        let mut rng = rand::rng();
        let old_food = game.food;
        for _ in 0..1000 {
            if game.food.distance(old_food) >= 200.0 {
                break;
            }
            game.food = random_food(bounds.0, &mut rng);
        }
    }

    // Perf pins walk to the screen centre — a steady, comparable case.
    if pinned.0 {
        game.circling = false;
        lizard.set_target_at_speed(bounds.0 / 2.0, dt);
        lizard.update(dt);
        return;
    }

    // The cursor can leave the window in Bevy (LÖVE always reports a
    // position). With nobody to follow, the lizard helps itself: walk
    // straight for the food — eating respawns it afar, so an unattended
    // lizard grazes from food to food until the pointer returns.
    let Some(mouse) = window
        .single()
        .ok()
        .and_then(|window| window.cursor_position())
    else {
        game.circling = false;
        lizard.set_target_at_speed(game.food, dt);
        lizard.update(dt);
        return;
    };

    // When the pointer is still and the lizard has reached it, orbit it in
    // a small circle instead of stopping on top of it (the fish's rule —
    // it also keeps the trot alive at a resting pointer).
    let moved = game
        .last_mouse
        .is_some_and(|last| (mouse - last).length() > 0.5);
    game.last_mouse = Some(mouse);

    let head = lizard.head();
    let orbit_radius = (160.0 * lizard.scale).max(50.0);
    let reach_dist = (lizard.speed * dt).max(8.0);
    let approach = lizard.base_target;

    if moved {
        game.circling = false;
    } else if !game.circling && approach.distance(mouse) <= reach_dist {
        game.circling = true;
        game.orbit_angle = (head - lizard.spine.joints[1]).to_angle();
        game.orbit_dir = 1.0;
    }

    let mut target = mouse;
    if game.circling {
        game.orbit_angle += game.orbit_dir * (lizard.speed / orbit_radius) * dt;
        target = mouse + Vec2::from_angle(game.orbit_angle) * orbit_radius;
    }

    lizard.set_target_at_speed(target, dt);
    lizard.update(dt);
}

#[cfg(test)]
mod tests {
    use super::*;

    const DT: f32 = 1.0 / 60.0;

    fn test_lizard(origin: Vec2) -> Lizard {
        Lizard::new(origin, 0.5, 240.0, &BodyPlan::reference())
    }

    /// Walk the lizard toward a goal for `frames`, like drive_lizard's
    /// move-then-solve order.
    fn walk(lizard: &mut Lizard, goal: Vec2, frames: usize) {
        for _ in 0..frames {
            lizard.set_target_at_speed(goal, DT);
            lizard.update(DT);
        }
    }

    /// Port-parity values for the angle clamp, from the fish's LuaJIT
    /// table — the copied helper must stay character-faithful.
    #[test]
    fn constrain_angle_matches_lua() {
        for (angle, target, constraint, expected) in [
            (0.1, 0.2, 0.5, 0.100000),
            (1.0, 0.2, 0.5, 0.700000),
            (-1.0, 0.2, 0.5, 5.983185),
            (3.0, -3.0, 0.1, 3.183185),
            (6.2, 0.05, 0.2, 6.2),
            (4.0, 1.0, 0.3, 1.3),
        ] {
            let got = constrain_angle(angle, target, constraint);
            assert!(
                (got - expected).abs() < 1e-4,
                "constrain_angle({angle}, {target}, {constraint}) = {got}, Lua says {expected}"
            );
        }
    }

    /// The settle pass from all-zero joints spreads the Vec spine at its
    /// per-segment spacing — the fish's structural test on the
    /// generalized solver, now with mixed link lengths (the per-region
    /// length knobs).
    #[test]
    fn settle_spreads_joints() {
        let links: Vec<f32> = (0..13).map(|i| 24.0 + 2.0 * i as f32).collect();
        let mut spine = Spine::new(Vec2::new(640.0, 400.0), links.clone(), ANGLE_CONSTRAINT);
        spine.update();
        assert_eq!(spine.joints[0], Vec2::new(640.0, 400.0));
        for (i, &link) in links.iter().enumerate() {
            let d = spine.joints[i].distance(spine.joints[i + 1]);
            assert!((d - link).abs() < 1e-3, "link {i} = {d}");
        }
    }

    /// Within 2px of the target the whole solve is skipped.
    #[test]
    fn fabrik_early_out_freezes_chain() {
        let mut spine = Spine::new(Vec2::new(100.0, 100.0), vec![32.0; 13], ANGLE_CONSTRAINT);
        spine.update();
        let before = spine.joints.clone();
        spine.target = Vec2::new(101.0, 100.0);
        spine.update();
        assert_eq!(before, spine.joints);
    }

    /// Bend clamping: a hairpin target can't fold the spine past π/8 per
    /// joint.
    #[test]
    fn angle_constraint_limits_bend() {
        let mut spine = Spine::new(Vec2::ZERO, vec![10.0; 13], ANGLE_CONSTRAINT);
        for (i, joint) in spine.joints.iter_mut().enumerate() {
            *joint = Vec2::new(10.0 * i as f32, 0.0);
        }
        spine.target = Vec2::new(25.0, 0.5);
        spine.update();
        for i in 1..spine.joints.len() - 1 {
            let a = (spine.joints[i - 1] - spine.joints[i]).to_angle();
            let b = (spine.joints[i] - spine.joints[i + 1]).to_angle();
            let diff = relative_angle_diff(b, a).abs();
            assert!(diff <= ANGLE_CONSTRAINT + 1e-4, "joint {i} bent {diff}");
        }
    }

    /// Frame 1 is a coherent stance: every foot planted at a reachable
    /// spot, both bones exact — the prototype's arms-at-(0,0) TODO, fixed.
    #[test]
    fn init_plants_every_foot_within_reach() {
        let lizard = test_lizard(Vec2::new(640.0, 400.0));
        for (i, leg) in lizard.legs.iter().enumerate() {
            assert!(leg.planted(), "leg {i} not planted at spawn");
            let extension = leg.shoulder.distance(leg.foot);
            assert!(
                extension <= REACH_CLAMP_K * leg.reach() + 1e-3,
                "leg {i} spawned overextended: {extension} of {}",
                leg.reach()
            );
            assert!((leg.shoulder.distance(leg.elbow) - leg.upper).abs() < 1e-3);
            assert!((leg.elbow.distance(leg.foot) - leg.lower).abs() < 1e-3);
        }
    }

    /// Planted feet are world-fixed — the rule the prototype's per-frame
    /// lerp broke. Walk a long way and check every frame: a leg that is
    /// planted before and after a frame must not have moved its foot
    /// (except the stance backstop, which never fires on a steady walk).
    #[test]
    fn feet_stay_world_fixed_while_planted() {
        let mut lizard = test_lizard(Vec2::new(200.0, 400.0));
        let shoulders: Vec<Vec2> = lizard.legs.iter().map(|leg| leg.shoulder).collect();
        let mut planted_at: Vec<Option<Vec2>> = lizard
            .legs
            .iter()
            .map(|leg| leg.planted().then_some(leg.foot))
            .collect();
        let mut stance_frames = 0usize;
        for _ in 0..600 {
            lizard.set_target_at_speed(Vec2::new(100_000.0, 400.0), DT);
            lizard.update(DT);
            for (i, leg) in lizard.legs.iter().enumerate() {
                match (planted_at[i], leg.planted()) {
                    (Some(before), true) => {
                        assert_eq!(leg.foot, before, "leg {i} foot drifted while planted");
                        stance_frames += 1;
                    }
                    (_, true) => planted_at[i] = Some(leg.foot),
                    (_, false) => planted_at[i] = None,
                }
            }
        }
        assert!(stance_frames > 600, "feet barely ever planted");
        for (i, leg) in lizard.legs.iter().enumerate() {
            assert!(
                leg.shoulder.distance(shoulders[i]) > 1.0,
                "leg {i} shoulder never moved with the body"
            );
        }
    }

    /// A straight walk triggers steps, every swing finishes within the
    /// clock's swing window, and the foot lands exactly ON the live step
    /// target — the guide's "move the foot to the desired position", a
    /// point that travels with the body.
    #[test]
    fn step_triggers_and_completes() {
        let mut lizard = test_lizard(Vec2::new(200.0, 400.0));
        // One swing covers (1-DUTY) of a cycle; a cycle is stride/speed
        // seconds of travel.
        let swing_secs = (1.0 - DUTY) * lizard.stride() / lizard.speed;
        let budget = (swing_secs / DT).ceil() as usize + 2;
        let mut completed = 0;
        let mut mid_step: Vec<Option<usize>> = vec![None; lizard.legs.len()];
        for _ in 0..600 {
            lizard.set_target_at_speed(Vec2::new(100_000.0, 400.0), DT);
            lizard.update(DT);
            for (i, leg) in lizard.legs.iter().enumerate() {
                match (leg.state, mid_step[i]) {
                    (StepState::Stepping { .. }, None) => mid_step[i] = Some(0),
                    (StepState::Stepping { .. }, Some(frames)) => {
                        mid_step[i] = Some(frames + 1);
                        assert!(frames < budget, "leg {i} step overran {budget} frames");
                    }
                    (StepState::Planted, Some(_)) => {
                        let (_, target) = lizard.leg_frame(leg);
                        assert!(
                            leg.foot.distance(target) < 1e-3,
                            "leg {i} landed {} px off the live target",
                            leg.foot.distance(target)
                        );
                        mid_step[i] = None;
                        completed += 1;
                    }
                    (StepState::Planted, None) => {}
                }
            }
        }
        assert!(completed > 8, "only {completed} steps in a 10s walk");
    }

    /// The diagonal trot: opposite groups are never mid-step together —
    /// with the swing window at the last 1-DUTY (< half) of each cycle
    /// and the groups half a cycle apart, that holds by construction at
    /// any speed — and both groups keep stepping. The first seconds are
    /// excluded: the spawn settle points the body off-axis and the turn
    /// onto the walk line stalls the clock briefly.
    #[test]
    fn diagonal_groups_never_both_stepping() {
        let mut lizard = test_lizard(Vec2::new(200.0, 400.0));
        walk(&mut lizard, Vec2::new(100_000.0, 400.0), 240);
        let mut group_steps = [0usize; 2];
        let mut planted = [true; 4];
        for frame in 0..900 {
            lizard.set_target_at_speed(Vec2::new(100_000.0, 400.0), DT);
            lizard.update(DT);
            let stepping = [0, 1].map(|g| {
                lizard
                    .legs
                    .iter()
                    .any(|leg| (leg.group as usize) & 1 == g && !leg.planted())
            });
            assert!(
                !(stepping[0] && stepping[1]),
                "both groups mid-step at frame {frame}"
            );
            for (i, leg) in lizard.legs.iter().enumerate() {
                if planted[i] && !leg.planted() {
                    group_steps[(leg.group as usize) & 1] += 1;
                }
                planted[i] = leg.planted();
            }
        }
        assert!(
            group_steps[0] > 5 && group_steps[1] > 5,
            "a group starved: {group_steps:?}"
        );
    }

    /// A higher chase speed must actually move the body faster (the old gait
    /// capped chase speed at ≈238 px/s, so faster settings did nothing) AND
    /// step proportionally more often — the clock ties cadence to distance.
    /// (Speed is now coupled to size, [`SPEED_PER_SCALE`], but the underlying
    /// `speed` field still drives pace/cadence, which is what this checks.)
    #[test]
    fn speed_scales_pace_and_cadence() {
        let mut results = Vec::new();
        for speed in [120.0_f32, 480.0] {
            let mut lizard = test_lizard(Vec2::new(200.0, 400.0));
            lizard.speed = speed;
            walk(&mut lizard, Vec2::new(100_000.0, 400.0), 240);
            let start = lizard.head();
            let mut steps = 0usize;
            let mut planted: Vec<bool> = lizard.legs.iter().map(Leg::planted).collect();
            for _ in 0..600 {
                lizard.set_target_at_speed(Vec2::new(100_000.0, 400.0), DT);
                lizard.update(DT);
                for (i, leg) in lizard.legs.iter().enumerate() {
                    if planted[i] && !leg.planted() {
                        steps += 1;
                    }
                    planted[i] = leg.planted();
                }
            }
            results.push((lizard.head().distance(start), steps));
        }
        let (slow_dist, slow_steps) = results[0];
        let (fast_dist, fast_steps) = results[1];
        assert!(
            fast_dist > slow_dist * 3.0,
            "4x speed only moved {fast_dist} vs {slow_dist}"
        );
        assert!(
            fast_steps > slow_steps * 2,
            "4x speed only stepped {fast_steps} vs {slow_steps}"
        );
    }

    /// The Stride slider trades step length for cadence: shorter strides
    /// at the same speed mean more steps over the same walk.
    #[test]
    fn stride_slider_scales_cadence() {
        let mut counts = Vec::new();
        for stride in [0.6_f32, 1.8] {
            let mut lizard = test_lizard(Vec2::new(200.0, 400.0));
            lizard.stride_mul = stride;
            walk(&mut lizard, Vec2::new(100_000.0, 400.0), 240);
            let mut steps = 0usize;
            let mut planted: Vec<bool> = lizard.legs.iter().map(Leg::planted).collect();
            for _ in 0..600 {
                lizard.set_target_at_speed(Vec2::new(100_000.0, 400.0), DT);
                lizard.update(DT);
                for (i, leg) in lizard.legs.iter().enumerate() {
                    if planted[i] && !leg.planted() {
                        steps += 1;
                    }
                    planted[i] = leg.planted();
                }
            }
            counts.push(steps);
        }
        assert!(
            counts[0] > counts[1] * 2,
            "short strides should step far more often: {counts:?}"
        );
    }

    /// The reference lizard walks at every size and speed: a steady
    /// straight walk keeps all four legs reachable (no foot ever pinned past
    /// full reach) and every leg keeps stepping. Reach-relative geometry +
    /// the distance clock are what hold this across the range.
    #[test]
    fn reference_lizard_walks_at_every_scale_and_speed() {
        for scale in [0.2_f32, 0.5, 1.0, 1.5] {
            for speed in [80.0_f32, 240.0, 600.0] {
                let mut lizard = Lizard::new(Vec2::new(200.0, 400.0), scale, speed, &BodyPlan::reference());
                assert_eq!(lizard.legs.len(), 4, "the reference lizard has four legs");
                let mut total_steps = 0usize;
                // Warm up past the spawn turn, then walk straight and watch
                // every leg stay inside its reach.
                walk(&mut lizard, Vec2::new(100_000.0, 400.0), 600);
                let mut planted: Vec<bool> = lizard.legs.iter().map(Leg::planted).collect();
                for _ in 0..900 {
                    lizard.set_target_at_speed(Vec2::new(100_000.0, 400.0), DT);
                    lizard.update(DT);
                    for (i, leg) in lizard.legs.iter().enumerate() {
                        // The gait is bounded against the clean-centerline
                        // shoulder; leg.shoulder is the bowed render one.
                        let (clean_shoulder, _) = lizard.leg_frame(leg);
                        let ext = clean_shoulder.distance(leg.foot) / leg.reach();
                        assert!(
                            ext <= 1.001,
                            "@scale {scale} speed {speed}: leg {i} pinned past reach ({ext:.3})"
                        );
                        if planted[i] && !leg.planted() {
                            total_steps += 1;
                        }
                        planted[i] = leg.planted();
                    }
                }
                assert!(
                    total_steps >= 4,
                    "@scale {scale} speed {speed}: only {total_steps} steps — a leg never moved"
                );
            }
        }
    }

    /// The 2-bone elbow keeps both bone lengths across the reachable
    /// annulus, and lies on the shoulder→foot line past full reach.
    #[test]
    fn ik_satisfies_bone_lengths() {
        let shoulder = Vec2::new(100.0, 100.0);
        let (l1, l2): (f32, f32) = (26.0, 22.0);
        for i in 0..40 {
            let angle = i as f32 * 0.157;
            let d = (l1 - l2).abs() + 0.5 + (i as f32 * 0.997) % (l1 + l2 - (l1 - l2).abs() - 1.0);
            let foot = shoulder + Vec2::from_angle(angle) * d;
            let elbow = solve_elbow(shoulder, foot, l1, l2, 1.0);
            assert!((shoulder.distance(elbow) - l1).abs() < 1e-3, "upper at d={d}");
            assert!((elbow.distance(foot) - l2).abs() < 1e-3, "lower at d={d}");
        }
        // Beyond reach: straight leg — the elbow sits on the line at l1.
        let foot = shoulder + Vec2::new(l1 + l2 + 30.0, 0.0);
        let elbow = solve_elbow(shoulder, foot, l1, l2, 1.0);
        assert!((shoulder.distance(elbow) - l1).abs() < 1e-2);
        let sin = (foot - shoulder).perp_dot(elbow - shoulder).abs()
            / ((foot - shoulder).length() * (elbow - shoulder).length());
        assert!(sin < 0.02, "overreached elbow off the line by sin {sin}");
    }

    /// Mirrored legs bow their elbows to opposite sides of the
    /// shoulder→foot line, and no elbow ever flips sides over a long walk
    /// — the FABRIK prototype's flip, gone.
    #[test]
    fn elbow_bend_sign_mirrors_per_side() {
        let shoulder = Vec2::new(100.0, 100.0);
        let foot = shoulder + Vec2::new(30.0, 10.0);
        let plus = solve_elbow(shoulder, foot, 26.0, 26.0, 1.0);
        let minus = solve_elbow(shoulder, foot, 26.0, 26.0, -1.0);
        let side = |elbow: Vec2| (foot - shoulder).perp_dot(elbow - shoulder);
        assert!(side(plus) * side(minus) < 0.0, "bend signs landed one side");

        let mut lizard = test_lizard(Vec2::new(300.0, 600.0));
        let mut signs = [0.0f32; 4];
        for frame in 0..600 {
            // A curving walk: the hard case for elbow stability.
            let t = frame as f32 / 60.0;
            let goal = Vec2::new(640.0, 400.0) + Vec2::from_angle(t) * 300.0;
            lizard.set_target_at_speed(goal, DT);
            lizard.update(DT);
            for (i, leg) in lizard.legs.iter().enumerate() {
                let bow = (leg.foot - leg.shoulder).perp_dot(leg.elbow - leg.shoulder);
                // Near-straight legs sit on the line; only count real bows.
                if bow.abs() > 1.0 {
                    if signs[i] != 0.0 {
                        assert!(
                            signs[i] * bow > 0.0,
                            "leg {i} elbow flipped sides at frame {frame}"
                        );
                    }
                    signs[i] = bow;
                }
            }
        }
    }

    /// The reference body: 14 joints at the exact `Lizard.pde` widths —
    /// rounded snout, cheek bulge, neck pinch, barrel swell (widest at
    /// joint 5), then a long thin tail tapering to a fine 7px point. This is
    /// the silhouette; if it drifts the lizard reads as a sausage again.
    #[test]
    fn reference_body_is_the_lizard_silhouette() {
        let plan = BodyPlan::reference();
        assert_eq!(plan.joint_count, 14, "reference joint count drifted");
        assert_eq!(
            plan.body_width,
            vec![52.0, 58.0, 40.0, 60.0, 68.0, 71.0, 65.0, 50.0, 28.0, 15.0, 11.0, 9.0, 7.0, 7.0],
            "the reference width profile changed"
        );
        let w = &plan.body_width;
        assert!(w[1] > w[0], "no cheek bulge behind the snout");
        assert!(w[2] < w[1] && w[2] < w[3], "neck not pinched");
        let widest = (0..14).max_by(|&a, &b| w[a].total_cmp(&w[b])).unwrap();
        assert_eq!(widest, 5, "barrel should peak at joint 5");
        assert!(w[13] < w[8], "tail not tapering to a point");
        // Every spine link is the reference 64px.
        assert!(
            plan.links.iter().all(|&l| (l - DEFAULT_LINK_SIZE).abs() < 1e-3),
            "links drifted from the reference 64px spacing"
        );
    }

    /// Wiggle 0 means a rigid spine — the drawn ribbon collapses exactly
    /// onto the clean centerline, no side-to-side at all.
    #[test]
    fn wiggle_zero_holds_the_spine_straight() {
        let mut lizard = test_lizard(Vec2::new(200.0, 400.0));
        lizard.wiggle = 0.0;
        walk(&mut lizard, Vec2::new(100_000.0, 400.0), 120);
        for (j, r) in lizard.spine.joints.iter().zip(&lizard.ribbon) {
            assert!(
                j.distance(*r) < 1e-4,
                "wiggle 0 still bowed the ribbon ({j} vs {r})"
            );
        }
    }

    /// The stance backstop: even with a foot planted impossibly far away
    /// (a teleported plant no walk produces), the next frame pulls it
    /// inside the reach instead of leaving the leg pinned straight.
    #[test]
    fn stance_clamp_keeps_extension_bounded() {
        let mut lizard = test_lizard(Vec2::new(200.0, 400.0));
        let (shoulder, _) = lizard.leg_frame(&lizard.legs[0]);
        let reach = lizard.legs[0].reach();
        lizard.legs[0].foot = shoulder + Vec2::new(3.0 * reach, 0.0);
        lizard.legs[0].state = StepState::Planted;
        lizard.set_target_at_speed(Vec2::new(100_000.0, 400.0), DT);
        lizard.update(DT);
        let (clean_shoulder, _) = lizard.leg_frame(&lizard.legs[0]);
        let ext = clean_shoulder.distance(lizard.legs[0].foot) / lizard.legs[0].reach();
        assert!(ext <= 1.001, "stance clamp left the leg pinned at {ext:.3}");
    }

    /// Start size applies LIVE: dragging it rebuilds the lizard in place the
    /// same frame — same position and heading — instead of doing nothing
    /// until a restart the paused popup can't trigger. Run through the real
    /// `reshape_lizard` system so the wiring is covered too.
    #[test]
    fn start_size_reshapes_in_place_live() {
        let mut app = App::new();
        let settings = LizardSettings::default();
        let lizard = Lizard::new(
            Vec2::new(640.0, 400.0),
            settings.start_scale,
            240.0,
            &BodyPlan::reference(),
        );
        let head = lizard.head();
        app.insert_resource(settings)
            .insert_resource(LizardEntity(Some(lizard)))
            .add_systems(Update, reshape_lizard);

        app.update();
        let entity = app.world().resource::<LizardEntity>();
        assert_eq!(entity.0.as_ref().unwrap().legs.len(), 4, "the reference is four legs");

        // Drag "Start size" to 0.9 → the lizard resizes live, same place (it
        // used to apply only on a restart, so the slider read as dead).
        app.world_mut().resource_mut::<LizardSettings>().start_scale = 0.9;
        app.update();
        let entity = app.world().resource::<LizardEntity>();
        let lizard = entity.0.as_ref().unwrap();
        assert!(
            (lizard.scale - 0.9).abs() < 1e-6,
            "live Start-size edit did not resize ({})",
            lizard.scale
        );
        assert!(
            lizard.head().distance(head) < 1.0,
            "resize jumped the lizard instead of rebuilding in place"
        );
        // …but growth must NOT retrigger the rebuild (spawn anchor, not a
        // per-frame force).
        let grown = {
            let mut entity = app.world_mut().resource_mut::<LizardEntity>();
            let lizard = entity.0.as_mut().unwrap();
            let grown = lizard.scale + 0.002;
            lizard.set_scale(grown);
            grown
        };
        app.update();
        let entity = app.world().resource::<LizardEntity>();
        assert!(
            (entity.0.as_ref().unwrap().scale - grown).abs() < 1e-6,
            "reshape clobbered growth back to start size"
        );
    }

    /// The gait-locked body bow swings while walking and straightens at
    /// rest — the "torque" now driven off which trot group is swinging.
    #[test]
    fn body_bend_swings_with_gait_and_settles() {
        let mut lizard = test_lizard(Vec2::new(200.0, 400.0));
        let mut min_bend = 0.0f32;
        let mut max_bend = 0.0f32;
        for _ in 0..600 {
            lizard.set_target_at_speed(Vec2::new(100_000.0, 400.0), DT);
            lizard.update(DT);
            min_bend = min_bend.min(lizard.bend);
            max_bend = max_bend.max(lizard.bend);
        }
        // A real trot drives the bow to BOTH sides over the walk.
        assert!(
            max_bend > 0.3 && min_bend < -0.3,
            "bow never swung both ways: [{min_bend}, {max_bend}]"
        );
        // Stop steering: the legs plant, the body eases back to straight.
        for _ in 0..120 {
            lizard.update(DT);
        }
        assert!(
            lizard.bend.abs() < 0.05,
            "bow {} never settled at rest",
            lizard.bend
        );
        // And the bowed ribbon must visibly leave the centerline mid-trot
        // but ride it when straight (the tail carries the most swing).
        let n = lizard.ribbon.len();
        assert!(
            (lizard.ribbon[n - 1] - lizard.spine.joints[n - 1]).length() < 1.0,
            "tail still bowed after settling"
        );
    }

    /// Growth on eat is the Lua test's: tiny per food, feet stay planted.
    #[test]
    fn growth_scales_bones_and_keeps_feet() {
        let mut lizard = test_lizard(Vec2::new(640.0, 400.0));
        let feet: Vec<Vec2> = lizard.legs.iter().map(|leg| leg.foot).collect();
        let upper = lizard.legs[0].upper;
        lizard.set_scale(lizard.scale + 0.002);
        assert!((lizard.legs[0].upper - upper * (0.502 / 0.5)).abs() < 1e-4);
        for (i, leg) in lizard.legs.iter().enumerate() {
            assert_eq!(leg.foot, feet[i], "growth moved a planted foot");
        }
    }

    // -----------------------------------------------------------------
    // The pointer-resolution layer, driven through the real system in a
    // headless App with a scripted cursor — the fish's recipe.

    const POINTER_BOUNDS: Vec2 = Vec2::new(1280.0, 720.0);

    fn pointer_app(settings: LizardSettings, food: Vec2) -> App {
        let mut app = App::new();
        let lizard = Lizard::new(
            POINTER_BOUNDS / 2.0,
            settings.start_scale,
            SPEED_PER_SCALE * settings.start_scale,
            &BodyPlan::reference(),
        );
        app.insert_resource(Time::<()>::default())
            .insert_resource(settings)
            .insert_resource(SimBounds(POINTER_BOUNDS))
            .insert_resource(PinnedAttractor(false))
            .insert_resource(LizardEntity(Some(lizard)))
            .insert_resource(LizardGame {
                food,
                orbit_dir: 1.0,
                ..Default::default()
            })
            .add_systems(Update, drive_lizard);
        // A fresh `Window` reports no cursor — the pointer starts "away".
        app.world_mut().spawn((Window::default(), PrimaryWindow));
        app
    }

    fn pointer_frame(app: &mut App) {
        app.world_mut()
            .resource_mut::<Time>()
            .advance_by(std::time::Duration::from_micros(16_667));
        app.update();
    }

    fn set_cursor(app: &mut App, pos: Vec2) {
        let mut windows = app
            .world_mut()
            .query_filtered::<&mut Window, With<PrimaryWindow>>();
        windows
            .single_mut(app.world_mut())
            .unwrap()
            .set_physical_cursor_position(Some(bevy::math::DVec2::new(
                pos.x as f64,
                pos.y as f64,
            )));
    }

    /// An unattended lizard feeds itself — eating, growing, food
    /// respawning afar — and the chase—orbit rules take back over the
    /// moment the pointer returns.
    #[test]
    fn lizard_grazes_while_the_pointer_is_away() {
        let settings = LizardSettings::default();
        let growth = settings.growth_rate;
        let start_scale = settings.start_scale;
        let mut app = pointer_app(settings, Vec2::new(300.0, 300.0));
        let old_food = Vec2::new(300.0, 300.0);
        let mut ate = false;
        for _ in 0..900 {
            pointer_frame(&mut app);
            if app.world().resource::<LizardGame>().eaten > 0 {
                ate = true;
                break;
            }
        }
        assert!(ate, "an unattended lizard never grazed its way to the food");
        let game = app.world().resource::<LizardGame>();
        assert!(
            game.food.distance(old_food) >= 200.0,
            "food respawned too close"
        );
        let lizard = app.world().resource::<LizardEntity>();
        let scale = lizard.0.as_ref().unwrap().scale;
        assert!(
            (scale - (start_scale + growth)).abs() < 1e-6,
            "one meal should grow start {start_scale} by {growth}, got {scale}"
        );

        // The pointer returns, still: the lizard must reach it and engage
        // the orbit — the in-window behaviour, food ignored beyond eats.
        let mouse = Vec2::new(900.0, 360.0);
        set_cursor(&mut app, mouse);
        for _ in 0..900 {
            pointer_frame(&mut app);
        }
        let game = app.world().resource::<LizardGame>();
        assert!(game.circling, "a still pointer, reached: the orbit never engaged");
        let lizard = app.world().resource::<LizardEntity>();
        let d = lizard.0.as_ref().unwrap().base_target.distance(mouse);
        assert!(d < 160.0, "lizard settled {d} px from the returned pointer");
    }

    /// Chase speed is coupled to size: a fresh size-0.1 lizard runs at 50, a
    /// size-0.5 one at 250, and in general `speed == SPEED_PER_SCALE·scale`
    /// every frame — so it speeds up exactly as it grows.
    #[test]
    fn speed_tracks_size_linearly() {
        // Food parked in the far corner so it can't eat (and grow) this frame.
        let mut app = pointer_app(LizardSettings::default(), Vec2::new(1240.0, 700.0));
        pointer_frame(&mut app);
        {
            let entity = app.world().resource::<LizardEntity>();
            let lizard = entity.0.as_ref().unwrap();
            assert!((lizard.scale - 0.1).abs() < 1e-6, "default start size is 0.1");
            assert!(
                (lizard.speed - SPEED_PER_SCALE * lizard.scale).abs() < 1e-3,
                "speed not coupled to size"
            );
            assert!(
                (lizard.speed - 50.0).abs() < 1e-3,
                "a size-0.1 lizard should chase at 50, got {}",
                lizard.speed
            );
        }
        // Grow it to 0.5 by hand (a meal's worth) — next frame its speed must
        // rise to 250 to match.
        {
            let mut entity = app.world_mut().resource_mut::<LizardEntity>();
            entity.0.as_mut().unwrap().set_scale(0.5);
        }
        pointer_frame(&mut app);
        let entity = app.world().resource::<LizardEntity>();
        let speed = entity.0.as_ref().unwrap().speed;
        assert!(
            (speed - 250.0).abs() < 1e-3,
            "a size-0.5 lizard should chase at 250, got {speed}"
        );
    }
}
