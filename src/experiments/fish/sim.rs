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
//! The state is a plain `Vec<Fish>`: the game drives one fish from the
//! mouse, the perf harness (`<count> fish`) drives many from wander
//! targets, and a future school can drive each from a boid via
//! [`Fish::set_target`] — the original `school.lua`'s contract.

use std::f32::consts::{PI, TAU};

use bevy::prelude::*;
use bevy::window::PrimaryWindow;
use rand::Rng;

use super::settings::FishSettings;
use crate::app::{
    AppState, PinnedAttractor, RestartRequested, SimBounds, sim_active, update_sim_bounds,
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
    /// travel direction override — how a school drives its fish (the
    /// head-to-target vector is too short/noisy there). Unused until the
    /// school experiment lands; it is the original `school.lua`'s contract.
    #[allow(dead_code)]
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

/// Per-fish wander destinations for the perf harness.
#[derive(Resource, Default)]
struct WanderTargets(Vec<Vec2>);

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

/// Number of fish this run simulates: 1 in the game, `<count>` in perf
/// runs that select the fish (`boids <count> fish ...`).
fn perf_fish_count() -> usize {
    if !std::env::args().skip(1).any(|arg| arg == "fish") {
        return 1;
    }
    std::env::args()
        .nth(1)
        .and_then(|arg| arg.parse().ok())
        .unwrap_or(1)
}

/// Label for the fish sim systems; the renderer rebuilds its mesh after.
#[derive(SystemSet, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct FishSimSet;

pub fn plugin(app: &mut App) {
    app.init_resource::<Fishes>()
        .init_resource::<FishGame>()
        .init_resource::<WanderTargets>()
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
/// activation — the original's `reset()`: fresh fish at screen centre at
/// the start scale, joints settled with one dt=0 pass, fresh food.
#[allow(clippy::too_many_arguments)]
fn handle_restart(
    keys: Res<ButtonInput<KeyCode>>,
    state: Res<State<AppState>>,
    settings: Res<FishSettings>,
    bounds: Res<SimBounds>,
    mut request: ResMut<RestartRequested>,
    mut fishes: ResMut<Fishes>,
    mut game: ResMut<FishGame>,
    mut wander: ResMut<WanderTargets>,
) {
    let key_restart = *state.get() == AppState::Playing && keys.just_pressed(KeyCode::KeyR);
    if !(request.0 || key_restart || fishes.0.is_empty()) {
        return;
    }
    request.0 = false;

    let mut rng = rand::rng();
    let count = perf_fish_count();
    let centre = bounds.0 / 2.0;
    fishes.0.clear();
    wander.0.clear();
    for _ in 0..count {
        // The game fish spawns at screen centre; the perf school spreads.
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
        // otherwise they all sit at (0,0) and the body collapses.
        fish.update(0.0);
        fishes.0.push(fish);
        wander.0.push(origin);
    }
    *game = FishGame {
        food: random_food(bounds.0, &mut rng),
        orbit_dir: 1.0,
        ..default()
    };
}

/// The per-frame minigame update, in the original's exact order: sync live
/// tunables, eat, track the pointer, orbit it when reached and stationary,
/// then move and solve. Perf runs (many fish) swap the pointer for
/// per-fish wander targets.
#[allow(clippy::too_many_arguments)]
fn drive_fish(
    time: Res<Time>,
    settings: Res<FishSettings>,
    bounds: Res<SimBounds>,
    pinned: Res<PinnedAttractor>,
    window: Query<&Window, With<PrimaryWindow>>,
    mut fishes: ResMut<Fishes>,
    mut game: ResMut<FishGame>,
    mut wander: ResMut<WanderTargets>,
) {
    let dt = time.delta_secs();
    let centre = bounds.0 / 2.0;

    if fishes.0.len() > 1 {
        // Perf harness: every fish chases its own wander point (or the
        // centre with `pin` — the sustained pile-up worst case). Chunked
        // across the compute pool: each fish's solve is independent.
        let pool = bevy::tasks::ComputeTaskPool::get();
        let count = fishes.0.len();
        let chunk_size = count.div_ceil((pool.thread_num().max(1) * 3).min(count));
        let pinned = pinned.0;
        let bounds = bounds.0;
        let settings = settings.clone();
        pool.scope(|scope| {
            for (fish_chunk, target_chunk) in fishes
                .0
                .chunks_mut(chunk_size)
                .zip(wander.0.chunks_mut(chunk_size))
            {
                let settings = &settings;
                scope.spawn(async move {
                    let mut rng = rand::rng();
                    for (fish, target) in fish_chunk.iter_mut().zip(target_chunk) {
                        fish.speed = settings.speed;
                        fish.wave = settings.wave;
                        fish.wave_freq = settings.wave_freq;
                        fish.wave_amp = settings.wave_amp;
                        if pinned {
                            *target = bounds / 2.0;
                        } else if fish.head().distance(*target) < 50.0 {
                            *target = Vec2::new(
                                rng.random_range(0.0..bounds.x),
                                rng.random_range(0.0..bounds.y),
                            );
                        }
                        fish.set_target_at_speed(*target, dt);
                        fish.update(dt);
                    }
                });
            }
        });
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
