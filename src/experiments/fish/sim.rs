//! The fish simulation: a 12-joint FABRIK spine chasing a (possibly
//! sine-offset) target, plus the minigame rules — eat food, grow, orbit a
//! stationary pointer. A direct behavioural port of `lib/fish.lua` +
//! `lib/chain.lua` + `minigames/fish.lua`.
//!
//! Everything here works in **window coordinates** (top-left origin,
//! y-down), exactly like the LÖVE original — `window.cursor_position()`
//! needs no transform and every Lua formula ports sign-identical. The
//! renderer flips to world coordinates when it emits vertices.
//!
//! The state is a plain `Vec<Fish>`. One fish is the original fish game,
//! driven from the mouse. More than one (the "Fish" tunable, or the perf
//! harness's `<count> fish`) swim as a [`School`] of boids ported from
//! the original's *school* minigame (`lib/school.lua`), each boid driving
//! its fish via [`Fish::set_target`] — while the food rules keep working:
//! any fish that touches the food eats it and grows.

use std::collections::HashMap;
use std::f32::consts::{PI, TAU};

use bevy::prelude::*;
use bevy::window::PrimaryWindow;
use rand::Rng;

use super::settings::FishSettings;
use crate::app::{
    AppState, PinnedAttractor, PointerOverUi, RestartRequested, SimBounds, sim_active,
    update_sim_bounds,
};
use crate::experiments::{ExperimentId, experiment_active};

/// Joints in the spine (`joint_count = 12`).
pub const JOINTS: usize = 12;
/// Unscaled distance between consecutive joints (`link_size = 64`).
pub const LINK_SIZE: f32 = 64.0;
/// How sharply consecutive segments may bend (`angle_constraint = π/8`).
pub const ANGLE_CONSTRAINT: f32 = PI / 8.0;
/// Unscaled body half-width at each joint, head to tail.
pub const BODY_WIDTH: [f32; JOINTS] = [
    68.0, 81.0, 84.0, 83.0, 77.0, 64.0, 51.0, 38.0, 32.0, 19.0, 19.0, 19.0,
];

// ---------------------------------------------------------------------------
// Vec2 helpers ported from lib/vec2.lua. Lua's `%` on numbers matches
// `rem_euclid` for a positive divisor, and `math.atan2`'s [-π, π] range
// matches glam's `to_angle`.

/// `v:setMagnitude(m)` with a guard: the Lua version NaN-poisons on a zero
/// vector, which never happens on its call paths; we keep a defensive
/// fallback instead.
fn set_magnitude(v: Vec2, m: f32) -> Vec2 {
    let len = v.length();
    if len > 1e-12 { v * (m / len) } else { Vec2::ZERO }
}

/// `a:orthogonal(dir, m)`: the two points at distance `m` from `a`,
/// perpendicular to `dir - a`. Returns them in the Lua order:
/// `(dy, -dx)`-side first, `(-dy, dx)`-side second.
pub fn orthogonal(a: Vec2, dir: Vec2, m: f32) -> (Vec2, Vec2) {
    let d = dir - a;
    (
        a + set_magnitude(Vec2::new(d.y, -d.x), m),
        a + set_magnitude(Vec2::new(-d.y, d.x), m),
    )
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

/// `pos:constrainDistance(anchor, d)` as used by the chain: re-seat `next`
/// at exactly `d` from `curr`, along the `curr → next` direction.
fn constrain_distance(curr: Vec2, next: Vec2, d: f32) -> Vec2 {
    curr - set_magnitude(curr - next, d)
}

fn simplify_angle(angle: f32) -> f32 {
    angle.rem_euclid(TAU)
}

/// How many radians to turn `angle` to reach `target`, in [-π, π].
fn relative_angle_diff(angle: f32, target: f32) -> f32 {
    // Rotate the space so π sits at the target, avoiding the 0/2π seam.
    PI - simplify_angle(angle + PI - target)
}

/// Clamp `angle` to within `constraint` of `target`.
pub fn constrain_angle(angle: f32, target: f32, constraint: f32) -> f32 {
    let diff = relative_angle_diff(angle, target);
    if diff.abs() <= constraint {
        return simplify_angle(angle);
    }
    if diff > constraint {
        return simplify_angle(target - constraint);
    }
    simplify_angle(target + constraint)
}

// ---------------------------------------------------------------------------
// The spine — lib/chain.lua with the fish's parameters: forward-pass FABRIK
// only (the fish sets no anchor, so the backward pass is a no-op).

#[derive(Clone)]
pub struct Spine {
    /// Joint positions, head first. All start at (0,0) like the Lua
    /// (`Joint:new` defaults `pos` to the zero vector); the settle pass on
    /// reset spreads them.
    pub joints: [Vec2; JOINTS],
    /// Scaled distance between consecutive joints.
    pub link: f32,
    pub target: Vec2,
}

impl Spine {
    fn new(target: Vec2) -> Self {
        Self {
            joints: [Vec2::ZERO; JOINTS],
            link: LINK_SIZE,
            target,
        }
    }

    /// `fabrikForward`: snap the head to the target and re-seat each joint
    /// behind it, clamping the bend per joint. Skipped entirely while the
    /// head is within 2px of the target — the chain freezes when the fish
    /// arrives (which is why the orbit keeps the target moving).
    fn update(&mut self) {
        if self.joints[0].distance(self.target) < 2.0 {
            return;
        }
        self.joints[0] = self.target;
        for i in 0..JOINTS - 1 {
            if i > 0 {
                let prev_angle = (self.joints[i - 1] - self.joints[i]).to_angle();
                let next_angle = (self.joints[i] - self.joints[i + 1]).to_angle();
                let constrained = constrain_angle(next_angle, prev_angle, ANGLE_CONSTRAINT);
                // Unit offset establishes the direction; the distance
                // constraint below fixes the length.
                self.joints[i + 1] = self.joints[i] - Vec2::from_angle(constrained);
            }
            self.joints[i + 1] = constrain_distance(self.joints[i], self.joints[i + 1], self.link);
        }
    }
}

// ---------------------------------------------------------------------------
// One fish — lib/fish.lua.

pub struct Fish {
    pub spine: Spine,
    pub scale: f32,
    pub speed: f32,
    /// Centerline destination; the wave offset is applied on top in
    /// [`Self::update`].
    pub base_target: Option<Vec2>,
    /// Direction the wiggle is taken perpendicular to.
    pub travel_dir: Option<Vec2>,
    pub wave: bool,
    pub wave_freq: f32,
    pub wave_amp: f32,
    pub wave_phase: f32,
}

impl Fish {
    pub fn new(origin: Vec2, scale: f32, speed: f32, rng: &mut impl Rng) -> Self {
        let mut fish = Self {
            spine: Spine::new(origin),
            scale: 1.0,
            speed,
            base_target: None,
            travel_dir: None,
            wave: true,
            wave_freq: 4.5,
            wave_amp: 15.0,
            // Random so a school of fish don't all weave in lockstep.
            wave_phase: rng.random::<f32>() * TAU,
        };
        fish.set_scale(scale);
        fish
    }

    pub fn head(&self) -> Vec2 {
        self.spine.joints[0]
    }

    /// Scaled body half-width at joint `i`. The Lua rebuilds a scaled copy
    /// of the width table; computing from the base is identical
    /// (`scaleBodyWidth` always starts from the base table).
    pub fn body_width(&self, i: usize) -> f32 {
        BODY_WIDTH[i] * self.scale
    }

    pub fn set_scale(&mut self, scale: f32) {
        if self.scale == scale {
            return;
        }
        // Chain:setScale multiplies link sizes by the ratio.
        self.spine.link *= scale / self.scale;
        self.scale = scale;
    }

    /// Set the centerline destination directly, with an optional stable
    /// travel direction override — how the [`School`] drives its fish: the
    /// boid position is the centerline and the boid velocity the wave
    /// direction (the head-to-target vector is too short/noisy there).
    /// The original `school.lua`'s contract.
    pub fn set_target(&mut self, target: Vec2, dir: Option<Vec2>) {
        self.travel_dir = Some(dir.unwrap_or(target - self.spine.joints[0]));
        self.base_target = Some(target);
    }

    /// Move the centerline destination toward `target` by `speed·dt` —
    /// overshoot included, like the original.
    pub fn set_target_at_speed(&mut self, target: Vec2, dt: f32) {
        let base = self.base_target.unwrap_or(self.spine.target);
        let base = delta_target(base, target, self.speed * dt);
        self.base_target = Some(base);
        self.travel_dir = Some(target - base);
    }

    /// Advance the wiggle and solve the spine.
    pub fn update(&mut self, dt: f32) {
        let mut target = self.base_target.unwrap_or(self.spine.target);
        if self.wave && let Some(d) = self.travel_dir {
            // Offset the target perpendicular to the travel direction by a
            // sine wave, so the head weaves toward its destination.
            self.wave_phase += dt * self.wave_freq;
            let m = d.length();
            if m > 1e-6 {
                let s = self.wave_amp * self.wave_phase.sin();
                target += Vec2::new(-d.y / m * s, d.x / m * s);
            }
        }
        self.spine.target = target;
        self.spine.update();
    }
}

// ---------------------------------------------------------------------------
// The minigame — minigames/fish.lua.

/// All fish alive. The game keeps exactly one; the perf harness many.
#[derive(Resource, Default)]
pub struct Fishes(pub Vec<Fish>);

/// Game state around the fish: the food dot and the orbit-a-stationary-
/// pointer behaviour.
#[derive(Resource, Default)]
pub struct FishGame {
    pub food: Vec2,
    pub eaten: u32,
    /// Orbiting a stationary pointer it has reached.
    circling: bool,
    orbit_angle: f32,
    orbit_dir: f32,
    /// Previous frame's pointer, to detect movement. Also the fallback
    /// target while the cursor is outside the window (LÖVE always reports
    /// a position; Bevy doesn't).
    last_mouse: Option<Vec2>,
}

// ---------------------------------------------------------------------------
// The school — lib/school.lua: the same Reynolds rules as the flock
// experiment but with the school's own constants, no screen wrap, and a
// fish spine riding every boid. Deliberately NOT sharing the flock's
// implementation: that one is a hand-SIMD kernel tuned for hundreds of
// thousands of plain dots, and coupling to it could cost the flock its
// performance. The school is its own small sim.

/// Per-frame steering clamp, at the 60 fps reference (`max_force = 0.6`
/// — twice the flock's; a school turns harder).
const SCHOOL_MAX_FORCE: f32 = 0.6;
/// Crowding radius (`separate_dist = 75`).
const SCHOOL_SEPARATE_DIST: f32 = 75.0;
/// Align/cohere radius and the grid cell size (`neighbour_dist = 150`).
const SCHOOL_NEIGHBOUR_DIST: f32 = 150.0;
// The steering weights (`separate_k`, `align_k`, `coherence_k`) live in
// [`FishSettings`] — the original school minigame tunes them live.
/// The mouse attracts from afar and repels within 50 px (the school's
/// `steer_to_target(mouse, 4.0, -6.0, 50)` — note: 50, not the flock's 100).
const SCHOOL_MOUSE_ATTRACT_K: f32 = 4.0;
const SCHOOL_MOUSE_REPEL_K: f32 = -6.0;
const SCHOOL_MOUSE_NEAR: f32 = 50.0;
/// No mouse steering while the pointer sits within 5 px of a window edge.
const SCHOOL_EDGE_MARGIN: f32 = 5.0;
/// Food homing — `lib/school.lua`'s own optional `target` mechanism
/// (`steer_to_target(target, k_far, k_close, range_close)`), which the
/// original school minigame left unused. Without it a school can NEVER
/// eat: parking the cursor on the food repels every boid at 50 px hard
/// enough (Δv up to 3.6 px/frame vs the 3.33 px/frame speed cap) that
/// heads stall past the 20 px eat radius — measured by the
/// `school_reaches_the_food` test. The k must beat the worst case
/// *structurally*: inside the ring the mouse repels at 6·max_force and
/// separation shoves another 2.8·max_force outward, so the net pull
/// (k − 6 − 2.8)·max_force must stay positive — k = 10 converges fish
/// onto the food instead of leaving it to lucky geometry. The range is
/// the lib's `range_close` default; outside 100 px the food exerts
/// nothing and the player steers freely (fish that wander close peel
/// off, snatch it, and rejoin — see `school_stays_steerable_away_from_food`).
const SCHOOL_FOOD_K: f32 = 10.0;
const SCHOOL_FOOD_NEAR: f32 = 100.0;
/// Smoothing — our deliberate fix on top of the Lua rules, in two
/// layers. The original's forces routinely exceed the speed cap *per
/// frame*: the mouse force alone is bang-bang at the 50 px ring (attract
/// k = 4 outside, repel k = -6 inside, up to 216 px/s of velocity change
/// per frame — 6 · 0.6 · 3600 / 60 — against a 200 px/s default cap),
/// and separation conflicts in any close pass do the same. A boid's
/// velocity can fully reverse between two frames; the wave offset rides
/// perpendicular to it, so the spine target also snaps up to 2·amp px
/// sideways. Plain triangles hide all that; a spline fish body whips and
/// flickers. The layers:
///
/// 1. **The slew limiter** ([`school_slew`]) — always on in the shipped
///    game (the parity tests opt out to pin the ported force kernel):
///    fish are heading-constrained, so per-frame heading change is
///    capped at [`SCHOOL_TURN_RATE`]·dt and speed relaxes through a
///    short low-pass. Forces stay as ported — only the velocity's
///    response to them is made continuous, everywhere: near the mouse
///    ring, in fish-fish close passes, on food dives.
/// 2. **The calm mill** — once the pointer rests and a fish has arrived
///    near it, its desired velocity blends toward a slow tangential mill
///    around the pointer (the single fish's stationary-pointer orbit,
///    schooled — milling is what real schools do), so an idle pile reads
///    as a calm vortex instead of a force standoff. The blend is capped
///    below 1 so a fraction of the raw forces (separation above all)
///    keeps spacing the pile, and the food impulse stays outside the
///    blend entirely: a calmed school still dives for food.
///
/// Max heading change, radians per second: a full U-turn takes ~0.6 s
/// (a natural swerve), while the flicker this kills needs ~π per FRAME.
const SCHOOL_TURN_RATE: f32 = 5.0;
/// Speed low-pass time constant (seconds): braking/accelerating reads
/// smooth but stays responsive (63% of a change lands in 0.1 s).
const SCHOOL_SPEED_TAU: f32 = 0.1;
/// Below this speed (px/s) a boid has no meaningful heading to slew
/// from — it adopts the target heading outright (a near-rest body
/// barely moves, so nothing visibly snaps).
const SCHOOL_SLEW_MIN_SPEED: f32 = 1.0;
/// Pointer movement under this many px/frame counts as resting (the
/// single fish's own `moved` threshold).
const SCHOOL_IDLE_MOVE: f32 = 0.5;
/// The pointer must rest this long (seconds) before calming begins…
const SCHOOL_CALM_DELAY: f32 = 0.25;
/// …then the calm blend fades in over this many seconds (and snaps back
/// to zero the moment the pointer moves).
const SCHOOL_CALM_RAMP: f32 = 0.75;
/// Blend ceiling: 15% of the raw forces always leak through, so the
/// milling pile keeps separating instead of stacking up.
const SCHOOL_CALM_MAX: f32 = 0.85;
/// Per-fish proximity gate: fully calm within NEAR px of the pointer,
/// fading to raw behaviour at FAR (fish still approaching chase at full
/// force).
const SCHOOL_CALM_NEAR: f32 = 100.0;
const SCHOOL_CALM_FAR: f32 = 200.0;
/// Mill speed as a fraction of max_speed — fast enough to keep every
/// spine's FABRIK target moving past its 2 px freeze threshold (the
/// chain stutters below ~2 px/frame), slow enough to read as resting.
const SCHOOL_CALM_SPEED_K: f32 = 0.6;
/// Preferred mill ring radius (px) — just outside the mouse-repel ring —
/// with a gentle radial correction toward it.
const SCHOOL_CALM_RADIUS: f32 = 70.0;
/// The original integrates per frame at 60 fps; we store px/s and convert
/// (the flock port's proven scheme — local on purpose, not imported).
const SCHOOL_REF_FPS: f32 = 60.0;
/// Per-boid cap on neighbour candidates examined per frame — the same
/// trick as the flock's `MAX_NEIGHBOUR_SAMPLES`, re-derived here for the
/// school's constants. A pinned school piles every fish into one
/// neighbourhood and the per-neighbour rules degenerate to O(n²) (4096
/// fish: ~12 fps). All three steering forces only use the *direction* of
/// the neighbour aggregates (`target_force` normalizes), so above the cap
/// each 3x3 bucket contributes a proportional contiguous block at a
/// per-fish, per-frame pseudo-random offset — statistically identical for
/// a dense pile. At or below the cap the scan is exhaustive and exact,
/// which covers ordinary school sizes (the LÖVE original topped out at 30).
const SCHOOL_MAX_NEIGHBOUR_SAMPLES: usize = 128;

/// The original's `target_force`: `k * limit(normalize(dir) * max_speed -
/// velocity, max_force)` in per-frame units, converted to px/s². A zero
/// `dir` contributes zero force — the Lua would NaN-poison the boid on
/// `normalize`; this deliberate deviation covers all three aggregates and
/// the boid-exactly-on-the-mouse case (same call the flock port made).
fn school_target_force(k: f32, dir: Vec2, vel: Vec2, max_speed: f32) -> Vec2 {
    let len = dir.length();
    if len <= 1e-12 {
        return Vec2::ZERO;
    }
    let steer = dir * (max_speed / len) - vel / SCHOOL_REF_FPS;
    k * steer.clamp_length_max(SCHOOL_MAX_FORCE) * (SCHOOL_REF_FPS * SCHOOL_REF_FPS)
}

/// The no-flicker contract: move `current` toward `target` like a
/// creature that can't teleport its heading — turn at most
/// [`SCHOOL_TURN_RATE`]·dt radians and relax the speed with
/// [`SCHOOL_SPEED_TAU`]. Whatever the forces demand (the bang-bang
/// mouse ring, a separation standoff, a fresh food dive), the resulting
/// velocity is continuous frame to frame — which is exactly what the
/// spline bodies need to never snap.
fn school_slew(current: Vec2, target: Vec2, dt: f32) -> Vec2 {
    let cur_len = current.length();
    let tgt_len = target.length();
    let speed = cur_len + (tgt_len - cur_len) * (dt / (dt + SCHOOL_SPEED_TAU));
    if cur_len < SCHOOL_SLEW_MIN_SPEED {
        return if tgt_len > 1e-6 {
            target * (speed / tgt_len)
        } else {
            current
        };
    }
    if tgt_len <= 1e-6 {
        // Nothing asked for: brake along the current heading.
        return current * (speed / cur_len);
    }
    let max_turn = SCHOOL_TURN_RATE * dt;
    let angle = current.perp_dot(target).atan2(current.dot(target));
    let turn = angle.clamp(-max_turn, max_turn);
    Vec2::from_angle(turn).rotate(current) * (speed / cur_len)
}

/// Grid cell coordinate — floor division like Lua's `math.floor(x / size)`:
/// the school is not screen-wrapped, so positions go negative and
/// `as i32` truncation would fold cells -1 and 0 together.
fn school_cell(v: f32) -> i32 {
    (v / SCHOOL_NEIGHBOUR_DIST).floor() as i32
}

/// Boid state for `count > 1`, parallel to [`Fishes`] index-for-index —
/// `lib/school.lua`'s positions/velocities/accelerations arrays.
/// Positions are the smooth centerlines the spines chase; the FABRIK
/// heads are never written back (the original's "fish go crazy with
/// wiggle on" bug).
#[derive(Resource, Default)]
pub struct School {
    pos: Vec<Vec2>,
    vel: Vec<Vec2>,
    acc: Vec<Vec2>,
    /// The food impulse, kept apart from the flocking + mouse forces so
    /// the calm blend can leave it at full strength (see [`Self::step`]);
    /// sized lazily by `step`, since only it reads them together.
    acc_food: Vec<Vec2>,
    /// The school's own pointer memory for cursor-left-the-window frames.
    /// Never [`FishGame::last_mouse`] — that one belongs to the single
    /// fish's moved/circling logic, which a count flip must not disturb.
    last_mouse: Option<Vec2>,
    /// Last frame's pointer + how long it has rested, for the calm ramp.
    idle_mouse: Option<Vec2>,
    idle_secs: f32,
    /// Spatial hash, rebuilt per frame (`lib/grid.lua`): cell size =
    /// neighbour radius, so each boid's neighbours live in 3x3 cells.
    grid: HashMap<(i32, i32), Vec<u32>>,
    /// Frame salt for the over-budget neighbour sampling offsets.
    frame: u32,
}

/// The flocking + steering forces for boid `i`, computed from the frozen
/// pre-update snapshot — returned as `(school forces, food impulse)` so
/// the integrate pass can calm the former without touching the latter.
/// Candidate iteration matches Lua's `Grid.collect`
/// order (ox outer, oy inner, ascending index per bucket) so the f32
/// accumulation order mirrors the original's — which keeps the LuaJIT
/// parity tests honest. Only when the 3x3 block exceeds
/// [`SCHOOL_MAX_NEIGHBOUR_SAMPLES`] does each bucket degrade to a
/// proportional contiguous block at a salted offset.
#[allow(clippy::too_many_arguments)]
fn school_steer(
    pos: &[Vec2],
    vel: &[Vec2],
    grid: &HashMap<(i32, i32), Vec<u32>>,
    i: usize,
    mouse: Option<Vec2>,
    food: Option<Vec2>,
    settings: &FishSettings,
    frame: u32,
) -> (Vec2, Vec2) {
    let max_speed = settings.speed;
    let p = pos[i];
    let v = vel[i];
    let (cx, cy) = (school_cell(p.x), school_cell(p.y));
    let mut buckets: [&[u32]; 9] = [&[]; 9];
    let mut total = 0usize;
    for ox in 0..3 {
        for oy in 0..3 {
            if let Some(bucket) = grid.get(&(cx + ox - 1, cy + oy - 1)) {
                buckets[(ox * 3 + oy) as usize] = bucket;
                total += bucket.len();
            }
        }
    }

    let mut sum_align = Vec2::ZERO;
    let mut sum_cohere = Vec2::ZERO;
    let mut sum_separate = Vec2::ZERO;
    let mut n_align = 0u32;
    let mut n_avoid = 0u32;
    let mut tally = |j: u32| {
        let q = pos[j as usize];
        let d = p.distance(q);
        if d > 0.0 && d < SCHOOL_NEIGHBOUR_DIST {
            sum_align += vel[j as usize];
            sum_cohere += q;
            n_align += 1;
        }
        if d > 0.0 && d < SCHOOL_SEPARATE_DIST {
            // (p - q):normalize() / d — a 1/d falloff.
            sum_separate += (p - q) / d / d;
            n_avoid += 1;
        }
    };
    if total <= SCHOOL_MAX_NEIGHBOUR_SAMPLES {
        // Exhaustive and exact — every candidate, in Lua's order.
        for bucket in buckets {
            for &j in bucket {
                tally(j);
            }
        }
    } else {
        // Over budget: each bucket contributes a proportional contiguous
        // block at a per-fish, per-frame offset (wrapping). The steering
        // only uses aggregate directions, so uniform sampling of a dense
        // pile is statistically transparent.
        let salt = (i as u32)
            .wrapping_mul(0x9E37_79B9)
            .wrapping_add(frame.wrapping_mul(0x85EB_CA6B));
        for (b, bucket) in buckets.into_iter().enumerate() {
            if bucket.is_empty() {
                continue;
            }
            let take = (bucket.len() * SCHOOL_MAX_NEIGHBOUR_SAMPLES / total).max(1);
            let start = salt.wrapping_add(b as u32).wrapping_mul(0x9E37_79B9) as usize % bucket.len();
            for k in 0..take.min(bucket.len()) {
                let idx = start + k;
                let idx = if idx >= bucket.len() {
                    idx - bucket.len()
                } else {
                    idx
                };
                tally(bucket[idx]);
            }
        }
    }

    let mut acc = Vec2::ZERO;
    if n_avoid > 0 {
        acc += school_target_force(settings.separation, sum_separate, v, max_speed);
    }
    if n_align > 0 {
        acc += school_target_force(settings.alignment, sum_align, v, max_speed);
        let cohere = sum_cohere / n_align as f32 - p;
        acc += school_target_force(settings.cohesion, cohere, v, max_speed);
    }
    let mut acc_food = Vec2::ZERO;
    if let Some(f) = food
        && p.distance(f) < SCHOOL_FOOD_NEAR
    {
        acc_food = school_target_force(SCHOOL_FOOD_K, f - p, v, max_speed);
    }
    if let Some(m) = mouse {
        let k = if p.distance(m) >= SCHOOL_MOUSE_NEAR {
            SCHOOL_MOUSE_ATTRACT_K
        } else {
            SCHOOL_MOUSE_REPEL_K
        };
        acc += school_target_force(k, m - p, v, max_speed);
    }
    (acc, acc_food)
}

impl School {
    /// Track the pointer's rest time and return this frame's global calm
    /// blend in `[0, SCHOOL_CALM_MAX]` — zero while the pointer moves (or
    /// is absent), ramping in once it has rested [`SCHOOL_CALM_DELAY`].
    fn idle_calm(&mut self, mouse: Option<Vec2>, dt: f32) -> f32 {
        let resting = match (self.idle_mouse, mouse) {
            (Some(prev), Some(m)) => (m - prev).length() <= SCHOOL_IDLE_MOVE,
            _ => false,
        };
        self.idle_mouse = mouse;
        self.idle_secs = if resting { self.idle_secs + dt } else { 0.0 };
        ((self.idle_secs - SCHOOL_CALM_DELAY) / SCHOOL_CALM_RAMP).clamp(0.0, 1.0) * SCHOOL_CALM_MAX
    }

    /// One boids frame: build the grid, compute every acceleration from
    /// the pre-update snapshot, then integrate — `school.lua`'s
    /// `guide` + `update` two-pass structure, force order included
    /// (flocking, then the optional target, then the mouse). `mouse` is
    /// the resolved pointer (None = no mouse steering), `food` the homing
    /// target (None in pure-school parity tests); velocities are px/s,
    /// max speed and the steering weights come from `settings`. `calm`
    /// is the idle-pointer blend from [`Self::idle_calm`], and `smooth`
    /// runs every velocity through the [`school_slew`] limiter — the
    /// shipped game always passes true; the parity tests pass false
    /// (and calm 0), which is exactly the Lua integration.
    /// The accel pass is chunked across the compute pool — each boid's
    /// forces read only the frozen snapshot, so the result is identical
    /// to the serial pass regardless of thread count.
    fn step(
        &mut self,
        mouse: Option<Vec2>,
        food: Option<Vec2>,
        settings: &FishSettings,
        calm: f32,
        smooth: bool,
        dt: f32,
    ) {
        let max_speed = settings.speed;
        let n = self.pos.len();
        self.frame = self.frame.wrapping_add(1);
        self.acc_food.resize(n, Vec2::ZERO);
        self.grid.clear();
        for (i, p) in self.pos.iter().enumerate() {
            self.grid
                .entry((school_cell(p.x), school_cell(p.y)))
                .or_default()
                .push(i as u32);
        }

        let School {
            pos,
            vel,
            acc,
            acc_food,
            grid,
            frame,
            ..
        } = self;
        let (pos, vel, grid, frame) = (&*pos, &*vel, &*grid, *frame);
        let pool = bevy::tasks::ComputeTaskPool::get_or_init(Default::default);
        let chunk_size = n.div_ceil((pool.thread_num().max(1) * 3).min(n).max(1));
        pool.scope(|scope| {
            for (c, (acc_chunk, food_chunk)) in acc
                .chunks_mut(chunk_size)
                .zip(acc_food.chunks_mut(chunk_size))
                .enumerate()
            {
                scope.spawn(async move {
                    let base = c * chunk_size;
                    for (k, (a, af)) in acc_chunk.iter_mut().zip(food_chunk.iter_mut()).enumerate()
                    {
                        (*a, *af) =
                            school_steer(pos, vel, grid, base + k, mouse, food, settings, frame);
                    }
                });
            }
        });

        // Integrate pass. No screen wrap — the original school isn't
        // toroidal; the mouse attraction is what keeps it on screen. The
        // food impulse rides outside the calm blend (a calmed school
        // still dives for food); with `c == 0` the target is numerically
        // the Lua integration (adding the zero food impulse can at most
        // flip a -0.0 sign bit), and the slew limiter then makes the
        // velocity's response continuous (skipped only by parity tests).
        for i in 0..n {
            let dv_school = self.acc[i] * dt;
            let dv_food = self.acc_food[i] * dt;
            let c = match mouse {
                Some(m) if calm > 0.0 => {
                    let d = self.pos[i].distance(m);
                    calm * ((SCHOOL_CALM_FAR - d) / (SCHOOL_CALM_FAR - SCHOOL_CALM_NEAR))
                        .clamp(0.0, 1.0)
                }
                _ => 0.0,
            };
            let target = if c > 0.0
                && let Some(m) = mouse
            {
                // Calm: aim for a slow mill around the pointer — a
                // tangential swim plus a gentle pull onto the mill ring.
                let base = (self.vel[i] + dv_school).clamp_length_max(max_speed);
                let r = self.pos[i] - m;
                let len = r.length();
                let (r_hat, tangent) = if len > 1e-3 {
                    (r / len, Vec2::new(-r.y, r.x) / len)
                } else {
                    (Vec2::X, Vec2::Y)
                };
                let mill_speed = SCHOOL_CALM_SPEED_K * max_speed;
                let radial = ((SCHOOL_CALM_RADIUS - len) / SCHOOL_CALM_RADIUS).clamp(-1.0, 1.0)
                    * 0.5
                    * mill_speed;
                let mill = tangent * mill_speed + r_hat * radial;
                (base.lerp(mill, c) + dv_food).clamp_length_max(max_speed)
            } else {
                (self.vel[i] + dv_school + dv_food).clamp_length_max(max_speed)
            };
            let v = if smooth {
                school_slew(self.vel[i], target, dt)
            } else {
                target
            };
            self.vel[i] = v;
            self.pos[i] += v * dt;
        }
    }
}

/// A fresh boid's velocity — `Vec2:random(love.math.random(max_speed))`:
/// random direction, random magnitude up to the (already-synced) speed.
/// The original quantizes both to integers (a Lua 5.1 `math.random`
/// artifact); the first integrate clamps it all to `max_speed·dt` anyway,
/// so we port the intent: uniform floats.
fn random_school_velocity(max_speed: f32, rng: &mut impl Rng) -> Vec2 {
    Vec2::from_angle(rng.random_range(0.0..TAU)) * rng.random_range(1.0..max_speed.max(2.0))
}

/// `School:setSize(n)`: grow with settled fish at random positions, shrink
/// by popping the tail, floor at 1 — live, no restart. The fish vec and
/// the boid arrays stay aligned index-for-index.
fn set_size(
    fishes: &mut Vec<Fish>,
    school: &mut School,
    n: usize,
    speed: f32,
    start_scale: f32,
    bounds: Vec2,
    rng: &mut impl Rng,
) {
    let n = n.max(1);
    if fishes.len() == 1 && n > 1 {
        // The lone game fish joins the school where it actually swims —
        // its boid slot is stale (last touched on restart). From rest:
        // the boids accelerate it out.
        school.pos[0] = fishes[0].base_target.unwrap_or(fishes[0].head());
        school.vel[0] = Vec2::ZERO;
        // Stale idle time from before a shrink must not pop the calm
        // blend in fully grown — let it ramp from zero again.
        school.idle_mouse = None;
        school.idle_secs = 0.0;
    }
    while fishes.len() < n {
        let origin = Vec2::new(
            rng.random_range(0.0..bounds.x),
            rng.random_range(0.0..bounds.y),
        );
        let mut fish = Fish::new(origin, start_scale, speed, rng);
        // Settle like the original's setSize, so the new fish is
        // drawable this frame.
        fish.update(0.0);
        fishes.push(fish);
        school.pos.push(origin);
        school.vel.push(random_school_velocity(speed, rng));
        school.acc.push(Vec2::ZERO);
    }
    while fishes.len() > n {
        fishes.pop();
        school.pos.pop();
        school.vel.pop();
        school.acc.pop();
    }
}

/// Food spawn inset from the screen edges.
const FOOD_PAD: f32 = 50.0;
/// Head-to-food distance that counts as eating (unscaled, like the original).
const EAT_DIST: f32 = 20.0;

/// Random food position at integer coordinates in the padded screen rect —
/// `love.math.random(PAD, w - PAD)` returns integers.
fn random_food(bounds: Vec2, rng: &mut impl Rng) -> Vec2 {
    let max_x = (bounds.x - FOOD_PAD).max(FOOD_PAD) as i32;
    let max_y = (bounds.y - FOOD_PAD).max(FOOD_PAD) as i32;
    Vec2::new(
        rng.random_range(FOOD_PAD as i32..=max_x) as f32,
        rng.random_range(FOOD_PAD as i32..=max_y) as f32,
    )
}

/// Label for the fish sim systems; the renderer rebuilds its mesh after.
#[derive(SystemSet, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct FishSimSet;

pub fn plugin(app: &mut App) {
    app.init_resource::<Fishes>()
        .init_resource::<FishGame>()
        .init_resource::<School>()
        .add_systems(
            Update,
            (handle_restart, drive_fish)
                .chain()
                .in_set(FishSimSet)
                .after(update_sim_bounds)
                .run_if(experiment_active(ExperimentId::Fish))
                // Steps while playing AND behind the menu (the live backdrop).
                .run_if(sim_active),
        );
}

/// Rebuild the fish on [R], on a UI restart request, or on first
/// activation — the originals' `reset()`: one fish spawns at screen
/// centre (the fish game); a school spawns spread across the screen
/// (the school game), each spine settled with one dt=0 pass; fresh food.
#[allow(clippy::too_many_arguments)]
fn handle_restart(
    keys: Res<ButtonInput<KeyCode>>,
    state: Res<State<AppState>>,
    settings: Res<FishSettings>,
    bounds: Res<SimBounds>,
    mut request: ResMut<RestartRequested>,
    mut fishes: ResMut<Fishes>,
    mut game: ResMut<FishGame>,
    mut school: ResMut<School>,
) {
    let key_restart = *state.get() == AppState::Playing && keys.just_pressed(KeyCode::KeyR);
    if !(request.0 || key_restart || fishes.0.is_empty()) {
        return;
    }
    request.0 = false;

    let mut rng = rand::rng();
    let count = (settings.count.round() as usize).max(1);
    let centre = bounds.0 / 2.0;
    fishes.0.clear();
    school.pos.clear();
    school.vel.clear();
    school.acc.clear();
    school.acc_food.clear();
    school.last_mouse = None;
    school.idle_mouse = None;
    school.idle_secs = 0.0;
    for _ in 0..count {
        let origin = if count == 1 {
            centre
        } else {
            Vec2::new(
                rng.random_range(0.0..bounds.0.x),
                rng.random_range(0.0..bounds.0.y),
            )
        };
        let mut fish = Fish::new(origin, settings.start_scale, settings.speed, &mut rng);
        // Settle the spine so the first frame has spread-out joints;
        // otherwise they all sit at (0,0) and the body collapses. Settle
        // per fish — a school step at dt=0 would zero every velocity.
        fish.update(0.0);
        fishes.0.push(fish);
        school.pos.push(origin);
        school.vel.push(random_school_velocity(settings.speed, &mut rng));
        school.acc.push(Vec2::ZERO);
    }
    *game = FishGame {
        food: random_food(bounds.0, &mut rng),
        orbit_dir: 1.0,
        ..default()
    };
}

/// One frame of the school game (`count > 1`): the fish game's food rules
/// — any fish that touches the food eats it and grows — then
/// `school.lua`'s frame: the boids step, and every spine rides its boid.
/// A plain function so the sim tests can drive it without an `App`.
#[allow(clippy::too_many_arguments)]
fn drive_school(
    fishes: &mut [Fish],
    school: &mut School,
    game: &mut FishGame,
    settings: &FishSettings,
    bounds: Vec2,
    mouse: Option<Vec2>,
    dt: f32,
    rng: &mut impl Rng,
) {
    // A stale orbit must not survive a later shrink back to one fish.
    game.circling = false;

    // Eat first, on last frame's heads — the single fish's order. The
    // first fish within reach scores and grows; the food respawns afar.
    if let Some(eater) = fishes
        .iter_mut()
        .find(|fish| fish.head().distance(game.food) < EAT_DIST)
    {
        game.eaten += 1;
        if settings.growth_rate > 0.0 {
            let scale = eater.scale + settings.growth_rate;
            eater.set_scale(scale);
        }
        // Respawn at least 200px away from where it was eaten.
        let old_food = game.food;
        for _ in 0..1000 {
            if game.food.distance(old_food) >= 200.0 {
                break;
            }
            game.food = random_food(bounds, rng);
        }
    }

    // The calm blend ramps in while the pointer rests (zero otherwise);
    // the slew limiter is always on — the shipped school never snaps.
    let calm = school.idle_calm(mouse, dt);
    school.step(mouse, Some(game.food), settings, calm, true, dt);

    // Every spine rides its boid: the position is the centerline, the
    // velocity the wave direction; the head is never written back (the
    // original's "fish go crazy with wiggle on" bug). Chunked across the
    // compute pool — each fish's solve is independent.
    let pool = bevy::tasks::ComputeTaskPool::get_or_init(Default::default);
    let count = fishes.len();
    let chunk_size = count.div_ceil((pool.thread_num().max(1) * 3).min(count).max(1));
    pool.scope(|scope| {
        for ((fish_chunk, pos_chunk), vel_chunk) in fishes
            .chunks_mut(chunk_size)
            .zip(school.pos.chunks(chunk_size))
            .zip(school.vel.chunks(chunk_size))
        {
            scope.spawn(async move {
                for ((fish, pos), vel) in fish_chunk.iter_mut().zip(pos_chunk).zip(vel_chunk) {
                    fish.speed = settings.speed;
                    fish.wave = settings.wave;
                    fish.wave_freq = settings.wave_freq;
                    fish.wave_amp = settings.wave_amp;
                    fish.set_target(*pos, Some(*vel));
                    fish.update(dt);
                }
            });
        }
    });
}

/// The per-frame minigame update. One fish runs the original fish game's
/// exact order: sync live tunables, eat, track the pointer, orbit it when
/// reached and stationary, then move and solve. More than one fish run
/// the school game instead ([`drive_school`]).
#[allow(clippy::too_many_arguments)]
fn drive_fish(
    time: Res<Time>,
    settings: Res<FishSettings>,
    bounds: Res<SimBounds>,
    pinned: Res<PinnedAttractor>,
    over_ui: Res<PointerOverUi>,
    window: Query<&Window, With<PrimaryWindow>>,
    mut fishes: ResMut<Fishes>,
    mut game: ResMut<FishGame>,
    mut school: ResMut<School>,
) {
    let dt = time.delta_secs();
    let centre = bounds.0 / 2.0;

    // The fish count applies live — the original school syncs max_speed
    // and then calls setSize every update, in that order (a slider-grown
    // fish's init velocity uses the fresh speed). At count == 1 this is
    // a strict no-op and the single-fish game below runs untouched.
    let count = (settings.count.round() as usize).max(1);
    set_size(
        &mut fishes.0,
        &mut school,
        count,
        settings.speed,
        settings.start_scale,
        bounds.0,
        &mut rand::rng(),
    );

    if fishes.0.len() > 1 {
        // The school's pointer. LÖVE always has a concrete position;
        // Bevy doesn't: perf pins aim at the centre, windowed runs fall
        // back to the last known spot and then the centre (the menu
        // backdrop must not let the school drift off-screen), headless
        // unpinned runs have none at all — pure flocking, the spread
        // perf case. The original's guards then apply to the resolved
        // point: no chasing while the pointer drives the UI (the school
        // respects it, unlike the single fish — each faithful to its
        // original) or rests within 5 px of a window edge.
        let mouse = if pinned.0 {
            Some(centre)
        } else if let Ok(window) = window.single() {
            let resolved = window
                .cursor_position()
                .or(school.last_mouse)
                .unwrap_or(centre);
            school.last_mouse = Some(resolved);
            let margin = Vec2::splat(SCHOOL_EDGE_MARGIN);
            let inside =
                resolved.cmpgt(margin).all() && resolved.cmplt(bounds.0 - margin).all();
            (!over_ui.0 && inside).then_some(resolved)
        } else {
            None
        };
        drive_school(
            &mut fishes.0,
            &mut school,
            &mut game,
            &settings,
            bounds.0,
            mouse,
            dt,
            &mut rand::rng(),
        );
        return;
    }

    let Some(fish) = fishes.0.first_mut() else {
        return;
    };

    // Speed and the sine-path wiggle are tunable live from the popup. The
    // fish ignores pointer-over-UI entirely — the original's fish update
    // discards that flag.
    fish.speed = settings.speed;
    fish.wave = settings.wave;
    fish.wave_freq = settings.wave_freq;
    fish.wave_amp = settings.wave_amp;

    if fish.head().distance(game.food) < EAT_DIST {
        game.eaten += 1;

        if settings.growth_rate > 0.0 {
            let scale = fish.scale + settings.growth_rate;
            fish.set_scale(scale);
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

    // When the pointer is still and the fish has reached it, orbit it in a
    // small circle instead of stopping on top of it. The cursor can leave
    // the window in Bevy; keep targeting its last known spot then.
    let mouse = window
        .single()
        .ok()
        .and_then(|window| window.cursor_position())
        .or(game.last_mouse)
        .unwrap_or(centre);
    let moved = game
        .last_mouse
        .is_some_and(|last| (mouse - last).length() > 0.5);
    game.last_mouse = Some(mouse);

    let head = fish.head();
    // Radius scales with the fish; the angular speed keeps the orbit point
    // moving at the fish's own speed.
    let orbit_radius = (160.0 * fish.scale).max(50.0);
    // "Reached" = close enough to touch the pointer this frame; with the
    // wiggle on, start sooner by the wave amplitude. Measured on the smooth
    // centerline, not the wiggling head, so the trigger isn't jittery.
    let mut reach_dist = (fish.speed * dt).max(8.0);
    if settings.wave {
        reach_dist += settings.wave_amp;
    }
    let approach = fish.base_target.unwrap_or(head);

    if moved {
        game.circling = false;
    } else if !game.circling && approach.distance(mouse) <= reach_dist {
        // Start the loop in the direction the fish is already facing so it
        // curves out smoothly.
        game.circling = true;
        game.orbit_angle = (head - fish.spine.joints[1]).to_angle();
        game.orbit_dir = 1.0;
    }

    // Target the pointer until reached; then a point swept around a circle.
    let mut target = mouse;
    if game.circling {
        // The wiggle only applies while swimming to the pointer — on the
        // circle it reads as speeding up / slowing down.
        fish.wave = false;
        game.orbit_angle += game.orbit_dir * (fish.speed / orbit_radius) * dt;
        target = mouse + Vec2::from_angle(game.orbit_angle) * orbit_radius;
    }

    fish.set_target_at_speed(target, dt);
    fish.update(dt);
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Port-parity values computed by running the actual Lua originals
    /// (lib/vec2.lua) under LuaJIT — /tmp/fish_truth.lua.
    #[test]
    fn constrain_angle_matches_lua() {
        for (angle, target, constraint, expected) in [
            (0.1, 0.2, 0.5, 0.100000),
            (1.0, 0.2, 0.5, 0.700000),
            (-1.0, 0.2, 0.5, 5.983185),
            (3.0, -3.0, 0.1, 3.183185),
            (6.2, 0.05, 0.2, 6.200000),
            (4.0, 1.0, 0.3, 1.300000),
        ] {
            let got = constrain_angle(angle, target, constraint);
            assert!(
                (got - expected).abs() < 1e-4,
                "constrain_angle({angle}, {target}, {constraint}) = {got}, Lua says {expected}"
            );
        }
    }

    /// Drive the fish exactly like the Lua harness (settle, 90 frames
    /// toward one target, 90 toward another, wave on at phase π) and
    /// compare every joint against LuaJIT's f64 output. Tolerance covers
    /// f32-vs-f64 drift across 180 FABRIK solves.
    #[test]
    fn spine_trajectory_matches_lua() {
        const LUA_JOINTS: [(f32, f32); JOINTS] = [
            (630.33351, 428.72438),
            (635.46098, 424.89436),
            (640.49679, 420.94461),
            (645.46061, 416.90474),
            (650.37374, 412.80339),
            (655.25854, 408.66833),
            (660.13771, 404.52662),
            (665.03365, 400.40475),
            (669.96781, 396.32872),
            (674.96000, 392.32397),
            (680.02774, 388.41527),
            (685.18566, 384.62635),
        ];
        let mut rng = rand::rng();
        let mut fish = Fish::new(Vec2::new(640.0, 400.0), 0.1, 200.0, &mut rng);
        fish.wave_phase = PI; // the harness stubs love.math.random() = 0.5
        fish.update(0.0);
        for _ in 0..90 {
            fish.set_target_at_speed(Vec2::new(900.0, 300.0), 1.0 / 60.0);
            fish.update(1.0 / 60.0);
        }
        for _ in 0..90 {
            fish.set_target_at_speed(Vec2::new(200.0, 600.0), 1.0 / 60.0);
            fish.update(1.0 / 60.0);
        }
        for (i, (expected, got)) in LUA_JOINTS.iter().zip(fish.spine.joints).enumerate() {
            let d = Vec2::new(expected.0, expected.1).distance(got);
            assert!(d < 0.05, "joint {i}: {got} vs Lua {expected:?} (off {d})");
        }
    }

    /// The settle pass from all-zero joints: head snaps to the target, the
    /// rest trail toward (0,0) at link spacing — the original's reset look.
    #[test]
    fn settle_spreads_joints() {
        let mut spine = Spine::new(Vec2::new(640.0, 400.0));
        spine.link = 6.4; // scale 0.1
        spine.update();
        assert_eq!(spine.joints[0], Vec2::new(640.0, 400.0));
        for i in 0..JOINTS - 1 {
            let d = spine.joints[i].distance(spine.joints[i + 1]);
            assert!((d - 6.4).abs() < 1e-3, "link {i} = {d}");
        }
        // Trailing toward (0,0): joint 1 sits between head and origin.
        assert!(spine.joints[1].x < 640.0 && spine.joints[1].y < 400.0);
    }

    /// Within 2px of the target the whole solve is skipped (the chain
    /// freezes) — the reason the orbit keeps the target moving.
    #[test]
    fn fabrik_early_out_freezes_chain() {
        let mut spine = Spine::new(Vec2::new(100.0, 100.0));
        spine.link = 6.4;
        spine.update();
        let before = spine.joints;
        spine.target = Vec2::new(101.0, 100.0); // < 2px away
        spine.update();
        assert_eq!(before, spine.joints);
    }

    /// Bend clamping: a hairpin target can't fold the spine past π/8 per
    /// joint.
    #[test]
    fn angle_constraint_limits_bend() {
        let mut spine = Spine::new(Vec2::new(0.0, 0.0));
        spine.link = 10.0;
        // Straight line along +x: head at 0, tail toward +x.
        for (i, joint) in spine.joints.iter_mut().enumerate() {
            *joint = Vec2::new(10.0 * i as f32, 0.0);
        }
        // Yank the head backwards over the body.
        spine.target = Vec2::new(25.0, 0.5);
        spine.update();
        for i in 1..JOINTS - 1 {
            let a = (spine.joints[i - 1] - spine.joints[i]).to_angle();
            let b = (spine.joints[i] - spine.joints[i + 1]).to_angle();
            let diff = relative_angle_diff(b, a).abs();
            assert!(diff <= ANGLE_CONSTRAINT + 1e-3, "joint {i} bent {diff}");
        }
    }

    // -----------------------------------------------------------------
    // School parity and rules.

    /// Per-frame ground truth from running the REAL lib/school.lua under
    /// LuaJIT (/tmp/school_truth.lua): 6 boids with injected state, mouse
    /// parked at (701, 455), max_speed 200, dt = 1/60, 60 frames of
    /// `x y vx vy` (velocities in the original's per-frame units).
    const SCHOOL_TRUTH: &str = include_str!("school_truth.txt");

    /// The harness's injected state, velocities converted to px/s.
    fn lua_school_init() -> School {
        let px = [100.0, 180.0, -40.0, 130.0, 701.0, 640.0];
        let py = [100.0, 140.0, 120.0, 90.0, 500.0, 420.0];
        let vx = [2.0f32, -1.5, 1.0, 0.5, -2.0, 3.0];
        let vy = [1.0f32, 2.5, -2.0, 0.5, 1.0, -1.0];
        School {
            pos: (0..6).map(|i| Vec2::new(px[i], py[i])).collect(),
            vel: (0..6)
                .map(|i| Vec2::new(vx[i], vy[i]) * SCHOOL_REF_FPS)
                .collect(),
            acc: vec![Vec2::ZERO; 6],
            ..Default::default()
        }
    }

    /// Truth frames as (pos, vel-in-px/s) per boid.
    fn school_truth_frames() -> Vec<Vec<(Vec2, Vec2)>> {
        let values: Vec<f32> = SCHOOL_TRUTH
            .split_whitespace()
            .map(|v| v.parse().unwrap())
            .collect();
        values
            .chunks(4 * 6)
            .map(|frame| {
                frame
                    .chunks(4)
                    .map(|b| {
                        (
                            Vec2::new(b[0], b[1]),
                            Vec2::new(b[2], b[3]) * SCHOOL_REF_FPS,
                        )
                    })
                    .collect()
            })
            .collect()
    }

    const SCHOOL_MOUSE: Option<Vec2> = Some(Vec2::new(701.0, 455.0));

    /// The LuaJIT harness's tunables: max_speed 200, the lib's default
    /// steering weights. `calm = 0` in every parity step — the calm
    /// regime is our deliberate deviation and the zero path is bit-exact.
    fn lua_settings() -> FishSettings {
        FishSettings {
            speed: 200.0,
            ..Default::default()
        }
    }

    /// Free-run the first frames against LuaJIT. Boids are chaotic, so a
    /// long free run can't hold a meaningful tolerance — these first
    /// frames prove the kernel (steer sums, mouse attract/repel, unit
    /// conversion, integration) before divergence compounds; the per-
    /// frame test below covers the long horizon.
    #[test]
    fn school_step_matches_lua_free_run() {
        let truth = school_truth_frames();
        let mut school = lua_school_init();
        for frame in &truth[..5] {
            school.step(SCHOOL_MOUSE, None, &lua_settings(), 0.0, false, 1.0 / 60.0);
            for (i, (pos, _)) in frame.iter().enumerate() {
                let d = school.pos[i].distance(*pos);
                assert!(d < 1e-2, "boid {i}: {} vs Lua {pos} (off {d})", school.pos[i]);
            }
        }
    }

    /// Every frame, resynced: load the Lua state, step once, compare the
    /// next Lua state — the kernel tested at 60 distinct states without
    /// chaotic error compounding. Covers grid-order accumulation, the
    /// negative-coordinate cell (boid 2 starts at x = -40), separation's
    /// 1/d falloff, and both mouse regimes.
    #[test]
    fn school_step_matches_lua_per_frame() {
        let truth = school_truth_frames();
        assert_eq!(truth.len(), 60);
        for f in 0..truth.len() {
            let mut school = if f == 0 {
                lua_school_init()
            } else {
                School {
                    pos: truth[f - 1].iter().map(|(p, _)| *p).collect(),
                    vel: truth[f - 1].iter().map(|(_, v)| *v).collect(),
                    acc: vec![Vec2::ZERO; 6],
                    ..Default::default()
                }
            };
            school.step(SCHOOL_MOUSE, None, &lua_settings(), 0.0, false, 1.0 / 60.0);
            for (i, (pos, vel)) in truth[f].iter().enumerate() {
                let dp = school.pos[i].distance(*pos);
                let dv = school.vel[i].distance(*vel);
                assert!(
                    dp < 1e-3 && dv < 0.05,
                    "frame {f} boid {i}: pos {} vs {pos} (off {dp}), vel {} vs {vel} (off {dv})",
                    school.pos[i],
                    school.vel[i]
                );
            }
        }
    }

    /// Lua's `math.floor(x / size)` floors toward -infinity; `as i32`
    /// truncation would fold cells -1 and 0 together (the school isn't
    /// screen-wrapped, so negative coordinates are normal).
    #[test]
    fn school_cell_floors_negative_coordinates() {
        assert_eq!(school_cell(10.0), 0);
        assert_eq!(school_cell(-10.0), -1);
        assert_eq!(school_cell(150.0), 1);
        assert_eq!(school_cell(-150.0), -1);
        assert_eq!(school_cell(-150.1), -2);
    }

    /// Any fish can eat: the eater scores, only the eater grows, and the
    /// food respawns at least 200px away.
    #[test]
    fn school_eats_grows_eater_respawns_food() {
        let _ = bevy::tasks::ComputeTaskPool::get_or_init(Default::default);
        let mut rng = rand::rng();
        let bounds = Vec2::new(1280.0, 800.0);
        let food = Vec2::new(400.0, 300.0);
        let settings = FishSettings::default();
        let mut fishes = Vec::new();
        let mut school = School::default();
        for origin in [Vec2::new(100.0, 100.0), food, Vec2::new(900.0, 600.0)] {
            let mut fish = Fish::new(origin, settings.start_scale, settings.speed, &mut rng);
            fish.update(0.0); // head lands exactly on the origin
            fishes.push(fish);
            school.pos.push(origin);
            school.vel.push(Vec2::new(60.0, 0.0));
            school.acc.push(Vec2::ZERO);
        }
        let mut game = FishGame {
            food,
            orbit_dir: 1.0,
            ..Default::default()
        };
        drive_school(
            &mut fishes,
            &mut school,
            &mut game,
            &settings,
            bounds,
            None,
            1.0 / 60.0,
            &mut rng,
        );
        assert_eq!(game.eaten, 1);
        assert!(
            (fishes[1].scale - (settings.start_scale + settings.growth_rate)).abs() < 1e-6,
            "the eater grows"
        );
        assert_eq!(fishes[0].scale, settings.start_scale, "bystanders don't");
        assert_eq!(fishes[2].scale, settings.start_scale);
        assert!(game.food.distance(food) >= 200.0, "food respawns afar");
    }

    /// `School:setSize` live: grow keeps the existing fish (the game fish
    /// joins the school at its own centerline), new fish arrive settled,
    /// shrink pops the tail, the boid arrays stay aligned, floor at 1.
    #[test]
    fn set_size_grows_and_shrinks_aligned() {
        let mut rng = rand::rng();
        let bounds = Vec2::new(1280.0, 800.0);
        let centre = bounds / 2.0;
        let mut fishes = vec![Fish::new(centre, 0.05, 200.0, &mut rng)];
        fishes[0].update(0.0);
        let mut school = School::default();
        school.pos.push(centre);
        school.vel.push(Vec2::ZERO);
        school.acc.push(Vec2::ZERO);

        // The lone fish has swum away from its restart slot.
        let swim_spot = Vec2::new(300.0, 200.0);
        fishes[0].base_target = Some(swim_spot);
        set_size(&mut fishes, &mut school, 8, 200.0, 0.05, bounds, &mut rng);
        assert_eq!(fishes.len(), 8);
        assert_eq!(school.pos.len(), 8);
        assert_eq!(school.vel.len(), 8);
        assert_eq!(school.acc.len(), 8);
        assert_eq!(school.pos[0], swim_spot, "game fish joins where it swims");
        assert_eq!(school.vel[0], Vec2::ZERO, "and from rest");
        // New fish are settled (head snapped to its spawn, not (0,0)).
        assert_ne!(fishes[7].head(), Vec2::ZERO);
        assert_eq!(fishes[7].head(), school.pos[7]);

        fishes[0].set_scale(0.2);
        set_size(&mut fishes, &mut school, 1, 200.0, 0.05, bounds, &mut rng);
        assert_eq!(fishes.len(), 1);
        assert_eq!(school.pos.len(), 1);
        assert_eq!(fishes[0].scale, 0.2, "the survivor keeps its growth");

        set_size(&mut fishes, &mut school, 0, 200.0, 0.05, bounds, &mut rng);
        assert_eq!(fishes.len(), 1, "floor at 1");
    }

    /// The user contract: a school parked on the food by the cursor must
    /// actually eat it — the mouse repels within 50px but fish cross the
    /// ring with momentum (plus the wiggle), and any head within 20px
    /// scores. 10 simulated seconds is the budget; the schools eat far
    /// faster in practice.
    #[test]
    fn school_reaches_the_food() {
        let _ = bevy::tasks::ComputeTaskPool::get_or_init(Default::default);
        let mut rng = rand::rng();
        let bounds = Vec2::new(1280.0, 800.0);
        let food = Vec2::new(900.0, 300.0);
        let settings = FishSettings::default();
        for count in [10usize, 30] {
            let mut fishes = Vec::new();
            let mut school = School::default();
            for i in 0..count {
                // Deterministic spread; random wave phases (the shipped
                // config keeps the wiggle on).
                let origin = Vec2::new(
                    80.0 + (i % 8) as f32 * 150.0,
                    80.0 + (i / 8) as f32 * 180.0,
                );
                let mut fish = Fish::new(origin, settings.start_scale, settings.speed, &mut rng);
                fish.update(0.0);
                fishes.push(fish);
                school.pos.push(origin);
                school.vel.push(Vec2::from_angle(i as f32) * 100.0);
                school.acc.push(Vec2::ZERO);
            }
            let mut game = FishGame {
                food,
                orbit_dir: 1.0,
                ..Default::default()
            };
            let mut ate_at = None;
            for frame in 0..600 {
                drive_school(
                    &mut fishes,
                    &mut school,
                    &mut game,
                    &settings,
                    bounds,
                    Some(food), // cursor parked exactly on the food
                    1.0 / 60.0,
                    &mut rng,
                );
                if game.eaten > 0 {
                    ate_at = Some(frame);
                    break;
                }
            }
            assert!(
                ate_at.is_some(),
                "a school of {count} never reached the food in 10 simulated seconds"
            );
        }
    }

    /// The flip side of the food homing: the player keeps control. A
    /// school sitting on the food must still follow the cursor away —
    /// captured fish eat (no repulsion around food, so heads converge),
    /// the food respawns 200px+ off, and everyone rejoins the school.
    #[test]
    fn school_stays_steerable_away_from_food() {
        let _ = bevy::tasks::ComputeTaskPool::get_or_init(Default::default);
        let mut rng = rand::rng();
        let bounds = Vec2::new(1280.0, 800.0);
        let food = Vec2::new(900.0, 300.0);
        let mouse = Vec2::new(200.0, 600.0);
        let settings = FishSettings::default();
        let mut fishes = Vec::new();
        let mut school = School::default();
        for i in 0..10usize {
            // Clustered around the food — the worst capture case.
            let origin = food + Vec2::from_angle(i as f32 * 0.63) * 60.0;
            let mut fish = Fish::new(origin, settings.start_scale, settings.speed, &mut rng);
            fish.update(0.0);
            fishes.push(fish);
            school.pos.push(origin);
            school.vel.push(Vec2::ZERO);
            school.acc.push(Vec2::ZERO);
        }
        let mut game = FishGame {
            food,
            orbit_dir: 1.0,
            ..Default::default()
        };
        for _ in 0..600 {
            drive_school(
                &mut fishes,
                &mut school,
                &mut game,
                &settings,
                bounds,
                Some(mouse),
                1.0 / 60.0,
                &mut rng,
            );
        }
        let centroid = school.pos.iter().sum::<Vec2>() / school.pos.len() as f32;
        assert!(
            centroid.distance(mouse) < 300.0,
            "school stuck near the food: centroid {centroid} is {} px from the cursor",
            centroid.distance(mouse)
        );
    }

    /// The school's steering weights are live tunables (the original
    /// school minigame's separation/alignment/cohesion specs): zeroing
    /// all of them leaves pure integration, and each weight individually
    /// changes the step.
    #[test]
    fn school_weights_are_tunable() {
        let _ = bevy::tasks::ComputeTaskPool::get_or_init(Default::default);
        // Two boids inside both radii (60 < 75) so all three rules fire.
        let init = || School {
            pos: vec![Vec2::new(100.0, 100.0), Vec2::new(160.0, 100.0)],
            vel: vec![Vec2::new(60.0, 30.0), Vec2::new(-30.0, 60.0)],
            acc: vec![Vec2::ZERO; 2],
            ..Default::default()
        };
        let step_with = |settings: &FishSettings| {
            let mut school = init();
            // Lua-exact path (no slew): this test pins the force wiring.
            school.step(None, None, settings, 0.0, false, 1.0 / 60.0);
            (school.pos[0], school.pos[1])
        };

        let defaults = step_with(&lua_settings());
        let zeroed = step_with(&FishSettings {
            speed: 200.0,
            separation: 0.0,
            alignment: 0.0,
            cohesion: 0.0,
            ..Default::default()
        });
        // All weights zero, no mouse, no food: velocities just integrate
        // (same `v * dt` op as the sim, for bitwise equality).
        let school = init();
        assert_eq!(zeroed.0, school.pos[0] + school.vel[0] * (1.0 / 60.0));
        assert_eq!(zeroed.1, school.pos[1] + school.vel[1] * (1.0 / 60.0));
        assert_ne!(defaults, zeroed);

        for (separation, alignment, cohesion) in
            [(8.0, 1.0, 1.0), (2.8, 0.0, 1.0), (2.8, 1.0, 6.0)]
        {
            let varied = step_with(&FishSettings {
                speed: 200.0,
                separation,
                alignment,
                cohesion,
                ..Default::default()
            });
            assert_ne!(
                varied, defaults,
                "({separation}, {alignment}, {cohesion}) didn't change the step"
            );
        }
    }

    /// The calm regime: a school piled on a resting pointer settles into
    /// a smooth mill — per-frame heading changes stay small (the raw
    /// bang-bang mouse force flips headings by up to π per frame) while
    /// the fish keep swimming rather than freezing.
    #[test]
    fn school_calms_into_a_mill_at_idle_pointer() {
        let _ = bevy::tasks::ComputeTaskPool::get_or_init(Default::default);
        let mut rng = rand::rng();
        let bounds = Vec2::new(1280.0, 800.0);
        let mouse = Vec2::new(400.0, 400.0);
        let settings = FishSettings::default();
        let mut fishes = Vec::new();
        let mut school = School::default();
        for i in 0..24usize {
            // Piled around the pointer, straddling the 50px repel ring.
            let origin = mouse + Vec2::from_angle(i as f32 * 0.7) * (20.0 + (i % 5) as f32 * 15.0);
            let mut fish = Fish::new(origin, settings.start_scale, settings.speed, &mut rng);
            fish.update(0.0);
            fishes.push(fish);
            school.pos.push(origin);
            school.vel.push(Vec2::from_angle(i as f32) * 80.0);
            school.acc.push(Vec2::ZERO);
        }
        let mut game = FishGame {
            food: Vec2::new(1200.0, 80.0), // far away: no eats, no dives
            orbit_dir: 1.0,
            ..Default::default()
        };
        let mut drive = |school: &mut School, fishes: &mut [Fish], game: &mut FishGame| {
            drive_school(
                fishes, school, game, &settings, bounds, Some(mouse), 1.0 / 60.0, &mut rng,
            );
        };
        // Let the calm ramp in (delay 0.25s + ramp 0.75s) and settle.
        for _ in 0..180 {
            drive(&mut school, &mut fishes, &mut game);
        }
        let mut max_turn: f32 = 0.0;
        let mut speeds = Vec::new();
        for _ in 0..120 {
            let before = school.vel.clone();
            drive(&mut school, &mut fishes, &mut game);
            for (i, (prev, cur)) in before.iter().zip(&school.vel).enumerate() {
                if school.pos[i].distance(mouse) > SCHOOL_CALM_FAR {
                    continue; // raw zone — only the calm zone is smoothed
                }
                speeds.push(cur.length());
                if prev.length() > 1.0 && cur.length() > 1.0 {
                    let turn = prev.perp_dot(*cur).atan2(prev.dot(*cur)).abs();
                    max_turn = max_turn.max(turn);
                }
            }
        }
        assert!(
            max_turn < 0.5,
            "calmed fish still snap their heading: {max_turn} rad in one frame"
        );
        let mean_speed = speeds.iter().sum::<f32>() / speeds.len() as f32;
        assert!(
            mean_speed > 30.0,
            "calmed fish froze: mean speed {mean_speed} px/s"
        );
        assert_eq!(game.eaten, 0, "the far-away food got eaten somehow");
    }

    /// THE no-flicker invariant, everywhere: whatever the school is doing
    /// — packed into a dense blob, charging a moving cursor, crossing the
    /// attract/repel ring, parked on a resting pointer, diving for food —
    /// no fish ever turns its heading faster than the slew limit or jumps
    /// its speed. This is the property the spline bodies need; the raw
    /// Lua rules violate it by up to π per frame.
    #[test]
    fn school_never_snaps_headings() {
        let _ = bevy::tasks::ComputeTaskPool::get_or_init(Default::default);
        let mut rng = rand::rng();
        let bounds = Vec2::new(1280.0, 800.0);
        let settings = FishSettings::default();
        let dt = 1.0 / 60.0;
        let mut fishes = Vec::new();
        let mut school = School::default();
        for i in 0..48usize {
            // A tight blob: every fish inside everyone's separation radius
            // AND straddling the mouse ring — the worst conflict pile.
            let origin = Vec2::new(640.0, 400.0) + Vec2::from_angle(i as f32 * 2.39) * 55.0;
            let mut fish = Fish::new(origin, settings.start_scale, settings.speed, &mut rng);
            fish.update(0.0);
            fishes.push(fish);
            school.pos.push(origin);
            school.vel.push(Vec2::from_angle(i as f32) * 150.0);
            school.acc.push(Vec2::ZERO);
        }
        let mut game = FishGame {
            food: Vec2::new(700.0, 430.0), // close: dives happen mid-test
            orbit_dir: 1.0,
            ..Default::default()
        };
        let max_turn = SCHOOL_TURN_RATE * dt + 1e-3;
        for frame in 0..900 {
            // Phase 1: cursor sweeps through the blob (ring crossings,
            // mass approach). Phase 2: cursor rests inside it (calm ramps
            // in). Phase 3: it bolts away (calm releases, full chase).
            let mouse = match frame {
                0..300 => Vec2::new(640.0, 400.0) + Vec2::from_angle(frame as f32 * 0.05) * 120.0,
                300..600 => Vec2::new(640.0, 400.0),
                _ => Vec2::new(150.0, 650.0),
            };
            let before = school.vel.clone();
            drive_school(
                &mut fishes,
                &mut school,
                &mut game,
                &settings,
                bounds,
                Some(mouse),
                dt,
                &mut rng,
            );
            for (i, (prev, cur)) in before.iter().zip(&school.vel).enumerate() {
                if prev.length() < SCHOOL_SLEW_MIN_SPEED {
                    continue; // heading undefined below the slew floor
                }
                let turn = prev.perp_dot(*cur).atan2(prev.dot(*cur)).abs();
                assert!(
                    turn <= max_turn,
                    "frame {frame} fish {i}: turned {turn} rad in one frame (cap {max_turn})"
                );
                let dv = (cur.length() - prev.length()).abs();
                assert!(
                    dv <= settings.speed * 0.35,
                    "frame {frame} fish {i}: speed jumped {dv} px/s in one frame"
                );
            }
        }
        assert!(game.eaten > 0, "the blob sat on food territory yet never ate");
    }

    /// A moving pointer never engages the calm regime — the chase stays
    /// the raw Lua behaviour.
    #[test]
    fn moving_pointer_keeps_the_school_raw() {
        let _ = bevy::tasks::ComputeTaskPool::get_or_init(Default::default);
        let mut school = School::default();
        let mut mouse = Vec2::new(400.0, 400.0);
        for frame in 0..120 {
            // Plenty above the 0.5px/frame rest threshold.
            mouse += Vec2::new(2.0, (frame as f32 * 0.3).sin());
            let calm = school.idle_calm(Some(mouse), 1.0 / 60.0);
            assert_eq!(calm, 0.0, "frame {frame}: calm engaged on a moving pointer");
        }
        // And the moment it rests long enough, calm ramps in…
        for _ in 0..120 {
            school.idle_calm(Some(mouse), 1.0 / 60.0);
        }
        assert!((school.idle_calm(Some(mouse), 1.0 / 60.0) - SCHOOL_CALM_MAX).abs() < 1e-6);
        // …and one twitch resets it.
        assert_eq!(school.idle_calm(Some(mouse + Vec2::X), 1.0 / 60.0), 0.0);
    }

    /// `set_target_at_speed` overshoots like the Lua (no clamp to the
    /// goal), and the wave offsets the spine target perpendicular to the
    /// travel direction.
    #[test]
    fn target_at_speed_and_wave() {
        let mut rng = rand::rng();
        let mut fish = Fish::new(Vec2::new(0.0, 0.0), 1.0, 200.0, &mut rng);
        fish.update(0.0);
        fish.wave = false;
        // 1s at speed 200 toward a point 100 away: overshoots to 200.
        fish.set_target_at_speed(Vec2::new(100.0, 0.0), 1.0);
        assert!((fish.base_target.unwrap().x - 200.0).abs() < 1e-3);

        fish.wave = true;
        fish.wave_amp = 15.0;
        fish.wave_freq = 4.5;
        fish.wave_phase = 0.0;
        fish.base_target = Some(Vec2::new(50.0, 0.0));
        fish.travel_dir = Some(Vec2::new(1.0, 0.0));
        // phase advances to dt·freq; offset = amp·sin(phase) along (-dy, dx)/m = (0, 1).
        fish.update(0.1);
        let expected = 15.0 * (0.1f32 * 4.5).sin();
        assert!((fish.spine.target.y - expected).abs() < 1e-4);
        assert!((fish.spine.target.x - 50.0).abs() < 1e-4);
    }
}
