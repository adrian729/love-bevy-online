//! The flow-field simulation — a Bevy/Rust port of `lib/flow.lua`: a grid
//! of flow angles from fractal Perlin noise (octaves, domain warp, swirl
//! bias), streamlines traced through it, and particles advected along it.
//!
//! Deliberate differences from the Lua (the user asked for *better, not
//! 1:1*; see README):
//! - **Bilinear sampling**: the Lua reads the nearest cell's angle, so
//!   trajectories kink at every cell edge. This port lerps the direction
//!   *vectors* of the four surrounding cells (vector lerp is wrap-safe
//!   where angle lerp isn't) — smooth everywhere.
//! - **RK2 (midpoint) advection** for streamlines and particles instead of
//!   Euler steps.
//! - **Evolve**: the field can drift through a third noise dimension over
//!   time (the Lua field is static). Ticks at a fixed 30 Hz while playing.
//! - **Continuous seeds**: the Lua re-seeds an RNG per seed value, so
//!   adjacent seeds reshuffle completely; here the seed pans the noise-space
//!   offset linearly — nearby seeds are related fields, and small seed
//!   steps morph. (The slider spans the full [`SEED_PERIOD`] so a fast
//!   drag still hops fields — type into the value label for fine steps.)
//! - **Normalized grid math**: the Lua feeds its noise x-axis from the
//!   *row* (a transpose) and its pixel→cell clamp is off by one cell
//!   (1-based table quirks); this port uses the conventional col→x/row→y
//!   mapping with clean edge-clamped sampling at cell centers. The arrows
//!   view anchors each arrow at its cell's *center* for the same reason
//!   (the Lua anchored at 1-based grid corners, one cell down-right).
//! - **Frame-rate-independent trails**: the Lua records one trail sample
//!   per rendered frame (trails shrink as fps rises); this port records at
//!   a fixed 60 Hz of simulated time, so a trail spans the same ~0.33 s of
//!   motion at any frame rate.

use std::f32::consts::TAU;
use std::sync::OnceLock;

use bevy::prelude::*;
use bevy::tasks::ComputeTaskPool;

use super::settings::{FlowMode, FlowSettings};
use crate::app::{AppState, RestartRequested, SimBounds, sim_active, update_sim_bounds};
use crate::experiments::{ExperimentId, experiment_active};

/// Trail ring-buffer size per particle (the original's TRAIL_CAP).
pub const TRAIL_CAP: usize = 20;
/// Stop drawing trail segments dimmer than this (the original's cutoff).
pub const TRAIL_CUTOFF: f32 = 0.05;
/// Trail samples are recorded at this fixed rate of *simulated* time, so a
/// trail covers the same wall-clock span at any frame rate.
const TRAIL_RECORD_HZ: f32 = 60.0;
/// The evolving field re-generates at most this often — fps-independent
/// cost, and the field morphs slowly enough that 30 Hz reads as continuous.
const EVOLVE_TICK: f32 = 1.0 / 30.0;
/// Noise-space z drift per second at evolve = 1.
const EVOLVE_RATE: f32 = 0.4;
/// Cap on how often a held slider triggers a live rebuild (the original's
/// REBUILD_THROTTLE, ~15/sec), with an exact rebuild on release.
const REBUILD_THROTTLE: f32 = 1.0 / 15.0;
/// Noise-space pan per seed unit. Adjacent seeds pan by ~a quarter cell —
/// neighbouring fields are related, scrubbing morphs. Asymmetric so the
/// pan is diagonal rather than along one axis. f64: the pan is multiplied
/// out then wrapped in [`seed_base`], and the multiply must not lose
/// precision before the wrap.
const SEED_STEP_X: f64 = 0.11;
const SEED_STEP_Y: f64 = 0.067;
/// How many integer seeds give distinct fields. The Perlin lattice repeats
/// every 256 units, so a pan of `step` per seed returns to its starting
/// phase every `256/step` seeds — x: 0.11 → every 25,600 integer seeds
/// (25,600·0.11 = 2,816 = 11·256), y: 0.067 → every 256,000 (256,000·0.067
/// = 17,152 = 67·256). Jointly: lcm = 256,000. The Seed slider spans
/// exactly this; past it the fields would repeat verbatim.
pub const SEED_PERIOD: f32 = 256_000.0;

/// The noise-space offset for a seed, wrapped into the Perlin lattice's
/// 256-unit period. The wrap is invisible — `noise3(x + 256, …)` equals
/// `noise3(x, …)` (the permutation table indexes `& 255`) to within an f32
/// ulp of 256 — and it keeps the offset small, so every seed in
/// [0, [`SEED_PERIOD`]] gets full f32 noise precision (unwrapped, a
/// six-digit seed's offset would quantize the finest noise-scale × scale
/// steps). The multiply runs in f64 so the product itself doesn't lose
/// the low bits before wrapping.
pub fn seed_base(seed: f32) -> Vec2 {
    Vec2::new(
        ((seed as f64) * SEED_STEP_X).rem_euclid(256.0) as f32,
        ((seed as f64) * SEED_STEP_Y).rem_euclid(256.0) as f32,
    )
}
/// Particle lifetime range in seconds (the original's 2 + random * 3).
const LIFE_MIN: f32 = 2.0;
const LIFE_MAX: f32 = 5.0;

// ---------------------------------------------------------------------------
// Perlin noise — deterministic, seedable, 3D (the third axis is `evolve`
// time). The permutation table is fixed (like love.math.noise's global
// table); the *seed* tunable pans the sample offsets instead, which is what
// makes seed scrubbing continuous.

pub struct Perlin {
    perm: [u8; 512],
}

impl Perlin {
    fn new(seed: u32) -> Self {
        // Fisher-Yates over 0..256 driven by xorshift32.
        let mut state = seed | 1;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            state
        };
        let mut table: [u8; 256] = std::array::from_fn(|i| i as u8);
        for i in (1..256).rev() {
            let j = next() as usize % (i + 1);
            table.swap(i, j);
        }
        let perm = std::array::from_fn(|i| table[i & 255]);
        Self { perm }
    }

    /// The shared instance every field uses.
    pub fn shared() -> &'static Perlin {
        static PERLIN: OnceLock<Perlin> = OnceLock::new();
        PERLIN.get_or_init(|| Perlin::new(0x9E37_79B9))
    }

    fn fade(t: f32) -> f32 {
        t * t * t * (t * (t * 6.0 - 15.0) + 10.0)
    }

    fn grad(hash: u8, x: f32, y: f32, z: f32) -> f32 {
        // Ken Perlin's 12 edge-direction gradients (+4 repeats).
        match hash & 15 {
            0 => x + y,
            1 => -x + y,
            2 => x - y,
            3 => -x - y,
            4 => x + z,
            5 => -x + z,
            6 => x - z,
            7 => -x - z,
            8 => y + z,
            9 => -y + z,
            10 => y - z,
            11 => -y - z,
            12 => x + y,
            13 => -y + z,
            14 => -x + y,
            _ => -y - z,
        }
    }

    /// Classic 3D Perlin, roughly in [-1, 1].
    pub fn noise3(&self, x: f32, y: f32, z: f32) -> f32 {
        let (fx, fy, fz) = (x.floor(), y.floor(), z.floor());
        let xi = (fx as i32 & 255) as usize;
        let yi = (fy as i32 & 255) as usize;
        let zi = (fz as i32 & 255) as usize;
        let (x, y, z) = (x - fx, y - fy, z - fz);
        let (u, v, w) = (Self::fade(x), Self::fade(y), Self::fade(z));
        let p = &self.perm;

        let a = p[xi] as usize + yi;
        let aa = p[a] as usize + zi;
        let ab = p[a + 1] as usize + zi;
        let b = p[xi + 1] as usize + yi;
        let ba = p[b] as usize + zi;
        let bb = p[b + 1] as usize + zi;

        let lerp = |t: f32, a: f32, b: f32| a + t * (b - a);
        lerp(
            w,
            lerp(
                v,
                lerp(
                    u,
                    Self::grad(p[aa], x, y, z),
                    Self::grad(p[ba], x - 1.0, y, z),
                ),
                lerp(
                    u,
                    Self::grad(p[ab], x, y - 1.0, z),
                    Self::grad(p[bb], x - 1.0, y - 1.0, z),
                ),
            ),
            lerp(
                v,
                lerp(
                    u,
                    Self::grad(p[aa + 1], x, y, z - 1.0),
                    Self::grad(p[ba + 1], x - 1.0, y, z - 1.0),
                ),
                lerp(
                    u,
                    Self::grad(p[ab + 1], x, y - 1.0, z - 1.0),
                    Self::grad(p[bb + 1], x - 1.0, y - 1.0, z - 1.0),
                ),
            ),
        )
    }

    /// Perlin mapped to [0, 1] like `love.math.noise`. Over-scaled before
    /// the clamp so typical values span the full range — raw 3D Perlin
    /// rarely leaves ±0.6, and without the boost the angle range (and with
    /// it the palette coverage) would visibly compress vs the original's
    /// Simplex noise.
    pub fn noise01(&self, x: f32, y: f32, z: f32) -> f32 {
        (self.noise3(x, y, z) * 0.75 + 0.5).clamp(0.0, 1.0)
    }
}

/// Fractal Brownian noise in [0, 1]: `octaves` layers at doubling frequency
/// and `persistence`-decaying amplitude (lib/flow.lua's `fbm`). The evolve
/// axis `z` is *not* frequency-scaled — all octaves drift together, which
/// keeps the evolution coherent rather than shimmery.
pub fn fbm(noise: &Perlin, x: f32, y: f32, z: f32, octaves: u32, persistence: f32) -> f32 {
    let (mut sum, mut amp, mut freq, mut norm) = (0.0, 1.0, 1.0, 0.0);
    for _ in 0..octaves.max(1) {
        sum += amp * noise.noise01(x * freq, y * freq, z);
        norm += amp;
        amp *= persistence;
        freq *= 2.0;
    }
    sum / norm
}

// ---------------------------------------------------------------------------
// The field

/// Everything `cellAngle` needs, frozen for one (possibly parallel) build.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct FieldParams {
    pub base: Vec2,
    pub noise_scale: f32,
    pub octaves: u32,
    pub persistence: f32,
    pub warp: f32,
    /// Radians (the Lua converts its degree tunable with `math.rad`).
    pub swirl: f32,
    /// Evolve-axis position.
    pub z: f32,
}

impl FieldParams {
    pub fn of(settings: &FlowSettings, z: f32) -> Self {
        Self {
            base: seed_base(settings.seed),
            noise_scale: settings.noise_scale,
            octaves: settings.octaves.round().max(1.0) as u32,
            persistence: settings.roughness,
            warp: settings.warp,
            swirl: settings.swirl.to_radians(),
            z,
        }
    }

    /// The flow angle (radians) at grid cell (row, col) — lib/flow.lua's
    /// `cellAngle`, with the conventional col→x / row→y mapping. The
    /// optional domain warp nudges the sample point by the field itself;
    /// the second offset reads the *already-warped* x (the Lua's sequential
    /// dependency, kept — it is part of the look).
    pub fn cell_angle(&self, noise: &Perlin, col: f32, row: f32) -> f32 {
        let mut x = (col + self.base.x) * self.noise_scale;
        let mut y = (row + self.base.y) * self.noise_scale;
        if self.warp > 0.0 {
            x += self.warp * (fbm(noise, x + 3.7, y + 1.2, self.z, self.octaves, self.persistence) - 0.5);
            y += self.warp * (fbm(noise, x + 8.3, y + 6.9, self.z, self.octaves, self.persistence) - 0.5);
        }
        fbm(noise, x, y, self.z, self.octaves, self.persistence) * TAU + self.swirl
    }
}

/// The flow field: one angle (and its unit direction) per grid cell, cell
/// size `scale` px, covering the sim bounds. Row-major.
#[derive(Resource, Default)]
pub struct FlowField {
    pub cols: usize,
    pub rows: usize,
    pub scale: f32,
    pub angles: Vec<f32>,
    pub dirs: Vec<Vec2>,
}

impl FlowField {
    /// Re-generate every cell, chunk-parallel over rows on the compute
    /// pool (each cell is independent; the worst grid — a large window at
    /// scale 4 with warp and deep octaves — is hundreds of thousands of
    /// fbm evaluations).
    pub fn rebuild(&mut self, params: &FieldParams, bounds: Vec2, scale: f32) {
        let noise = Perlin::shared();
        self.scale = scale;
        self.cols = ((bounds.x / scale).floor() as usize).max(1);
        self.rows = ((bounds.y / scale).floor() as usize).max(1);
        let cells = self.cols * self.rows;
        self.angles.resize(cells, 0.0);
        self.dirs.resize(cells, Vec2::X);

        let cols = self.cols;
        let pool = ComputeTaskPool::get_or_init(Default::default);
        let chunk_rows = self
            .rows
            .div_ceil((pool.thread_num().max(1) * 3).min(self.rows));
        let chunk = chunk_rows * cols;
        pool.scope(|scope| {
            for (chunk_index, (angles, dirs)) in self
                .angles
                .chunks_mut(chunk)
                .zip(self.dirs.chunks_mut(chunk))
                .enumerate()
            {
                let row0 = chunk_index * chunk_rows;
                scope.spawn(async move {
                    for (i, (angle, dir)) in angles.iter_mut().zip(dirs.iter_mut()).enumerate() {
                        let row = row0 + i / cols;
                        let col = i % cols;
                        let a = params.cell_angle(noise, col as f32, row as f32);
                        *angle = a;
                        *dir = Vec2::new(a.cos(), a.sin());
                    }
                });
            }
        });
    }

    /// The field direction at pixel (x, y): edge-clamped bilinear lerp of
    /// the four surrounding cell centers' direction vectors, normalized.
    /// Vector lerp is wrap-safe where angle lerp isn't (ε and 2π−ε average
    /// to ~+x, not to a backwards π).
    pub fn sample_dir(&self, p: Vec2) -> Vec2 {
        if self.cols == 0 || self.rows == 0 {
            return Vec2::X;
        }
        let gx = (p.x / self.scale - 0.5).clamp(0.0, (self.cols - 1) as f32);
        let gy = (p.y / self.scale - 0.5).clamp(0.0, (self.rows - 1) as f32);
        let c0 = gx as usize;
        let r0 = gy as usize;
        let c1 = (c0 + 1).min(self.cols - 1);
        let r1 = (r0 + 1).min(self.rows - 1);
        let fx = gx - c0 as f32;
        let fy = gy - r0 as f32;
        let top = self.dirs[r0 * self.cols + c0].lerp(self.dirs[r0 * self.cols + c1], fx);
        let bottom = self.dirs[r1 * self.cols + c0].lerp(self.dirs[r1 * self.cols + c1], fx);
        let dir = top.lerp(bottom, fy);
        let len = dir.length();
        if len > 1e-4 {
            dir / len
        } else {
            // Opposing neighbours cancelled out; fall back to the nearest cell.
            self.dirs[gy.round() as usize * self.cols + gx.round() as usize]
        }
    }

    /// The field angle at pixel (x, y), from the smooth sampled direction
    /// (used for palette colours, so colours match the motion).
    pub fn angle_at(&self, p: Vec2) -> f32 {
        let dir = self.sample_dir(p);
        dir.y.atan2(dir.x)
    }
}

// ---------------------------------------------------------------------------
// Streamlines ('illustrate'): `detail` lines traced through the field from
// seeded random starts. RK2 with a half-cell step (the Lua stepped a full
// cell with the nearest-cell angle — visibly polygonal).

#[derive(Resource, Default)]
pub struct FlowStreamlines(pub Vec<Vec<Vec2>>);

/// Advance one RK2 (midpoint) step of `step` px through the field.
pub fn rk2_step(field: &FlowField, p: Vec2, step: f32) -> Vec2 {
    let d1 = field.sample_dir(p);
    let d2 = field.sample_dir(p + d1 * (step * 0.5));
    p + d2 * step
}

fn trace_streamline(field: &FlowField, start: Vec2, max_steps: usize, bounds: Vec2, out: &mut Vec<Vec2>) {
    out.clear();
    let step = field.scale * 0.5;
    let mut p = start;
    out.push(p);
    for _ in 0..max_steps {
        p = rk2_step(field, p, step);
        out.push(p);
        if p.x <= 0.0 || p.x >= bounds.x || p.y <= 0.0 || p.y >= bounds.y {
            break;
        }
    }
}

/// Retrace every streamline: starts come from a seed-derived RNG (the same
/// seed gives the same picture; an evolve tick keeps the same starts so the
/// lines morph coherently instead of reshuffling). Lines are independent —
/// traced chunk-parallel.
fn retrace_streamlines(
    lines: &mut FlowStreamlines,
    field: &FlowField,
    seed: f32,
    detail: usize,
    length: usize,
    bounds: Vec2,
) {
    let mut rng = pcg_seed(seed.to_bits() as u64 ^ 0xF10F);
    lines.0.resize_with(detail, Vec::new);
    let starts: Vec<Vec2> = (0..detail)
        .map(|_| {
            Vec2::new(
                pcg_next_f32(&mut rng) * bounds.x,
                pcg_next_f32(&mut rng) * bounds.y,
            )
        })
        .collect();
    // The Lua steps a full cell `length` times; this port steps half-cells,
    // so doubling the step count keeps the same pixel reach.
    let max_steps = length * 2;

    let pool = ComputeTaskPool::get_or_init(Default::default);
    let chunk = detail.div_ceil((pool.thread_num().max(1) * 3).min(detail.max(1)));
    pool.scope(|scope| {
        for (line_chunk, start_chunk) in lines.0.chunks_mut(chunk).zip(starts.chunks(chunk)) {
            scope.spawn(async move {
                for (line, start) in line_chunk.iter_mut().zip(start_chunk) {
                    trace_streamline(field, *start, max_steps, bounds, line);
                }
            });
        }
    });
}

// ---------------------------------------------------------------------------
// Particles: advected along the field, respawning when they leave the
// screen or age out (the Lua's stepParticles). Positions live in window
// coordinates [0,w]×[0,h], like everything else here.

#[derive(Resource)]
pub struct FlowParticles {
    pub pos: Vec<Vec2>,
    pub life: Vec<f32>,
    /// Ring buffer of recent positions, `TRAIL_CAP` per particle.
    pub trail: Vec<Vec2>,
    /// Ring head (index of the newest sample) per particle.
    pub head: Vec<u8>,
    /// How many samples the ring holds, ≤ TRAIL_CAP.
    pub len: Vec<u8>,
    rng: u64,
    record_acc: f32,
}

impl Default for FlowParticles {
    fn default() -> Self {
        Self {
            pos: Vec::new(),
            life: Vec::new(),
            trail: Vec::new(),
            head: Vec::new(),
            len: Vec::new(),
            rng: pcg_seed(0xF1E1D),
            record_acc: 0.0,
        }
    }
}

impl FlowParticles {
    pub fn count(&self) -> usize {
        self.pos.len()
    }

    /// Iterate one particle's trail newest → oldest — the ring-buffer
    /// read the trail shader mirrors (tests pin the semantics here).
    #[cfg(test)]
    pub fn trail_iter(&self, i: usize) -> impl Iterator<Item = Vec2> + '_ {
        let head = self.head[i] as usize;
        let len = self.len[i] as usize;
        (0..len).map(move |age| self.trail[i * TRAIL_CAP + (head + TRAIL_CAP - age) % TRAIL_CAP])
    }

    /// Drop every particle (they respawn lazily on the next step).
    pub fn clear(&mut self) {
        self.pos.clear();
        self.life.clear();
        self.trail.clear();
        self.head.clear();
        self.len.clear();
        self.record_acc = 0.0;
    }

    fn spawn_one(&mut self, bounds: Vec2) {
        let pos = Vec2::new(
            pcg_next_f32(&mut self.rng) * bounds.x,
            pcg_next_f32(&mut self.rng) * bounds.y,
        );
        // The original seeds initial lifetimes shorter (random * 4) so the
        // first wave of respawns doesn't happen all at once.
        self.pos.push(pos);
        self.life.push(pcg_next_f32(&mut self.rng) * 4.0);
        let index = self.pos.len() - 1;
        self.trail.resize((index + 1) * TRAIL_CAP, Vec2::ZERO);
        self.trail[index * TRAIL_CAP] = pos;
        self.head.push(0);
        self.len.push(1);
    }

    fn respawn(&mut self, i: usize, bounds: Vec2) {
        let pos = Vec2::new(
            pcg_next_f32(&mut self.rng) * bounds.x,
            pcg_next_f32(&mut self.rng) * bounds.y,
        );
        self.pos[i] = pos;
        self.life[i] = LIFE_MIN + pcg_next_f32(&mut self.rng) * (LIFE_MAX - LIFE_MIN);
        // Reset the trail to the single new point so no streak is drawn
        // across the screen from the old position (the Lua's comment).
        self.trail[i * TRAIL_CAP] = pos;
        self.head[i] = 0;
        self.len[i] = 1;
    }

    /// Match the live count tunable: spawn or truncate.
    pub fn resize(&mut self, n: usize, bounds: Vec2) {
        while self.pos.len() < n {
            self.spawn_one(bounds);
        }
        if self.pos.len() > n {
            self.pos.truncate(n);
            self.life.truncate(n);
            self.trail.truncate(n * TRAIL_CAP);
            self.head.truncate(n);
            self.len.truncate(n);
        }
    }

    /// Advance every particle one frame: RK2 along the field at `speed`
    /// px/s, trail samples recorded at a fixed 60 Hz of simulated time,
    /// respawn on ageing out or leaving the screen. The advance runs
    /// chunk-parallel (particles are independent); respawns — a handful per
    /// frame — run serially after, sharing the one RNG.
    pub fn step(&mut self, field: &FlowField, speed: f32, dt: f32, bounds: Vec2) {
        let n = self.pos.len();
        if n == 0 {
            return;
        }

        // How many trail samples this frame records (usually 0 or 1).
        self.record_acc += dt;
        let interval = 1.0 / TRAIL_RECORD_HZ;
        let mut records = 0usize;
        while self.record_acc >= interval {
            self.record_acc -= interval;
            records += 1;
        }
        // A long hitch shouldn't burn the whole ring on duplicates.
        let records = records.min(TRAIL_CAP);

        let pool = ComputeTaskPool::get_or_init(Default::default);
        let chunk = n.div_ceil((pool.thread_num().max(1) * 3).min(n));
        pool.scope(|scope| {
            for ((((pos, life), trail), head), len) in self
                .pos
                .chunks_mut(chunk)
                .zip(self.life.chunks_mut(chunk))
                .zip(self.trail.chunks_mut(chunk * TRAIL_CAP))
                .zip(self.head.chunks_mut(chunk))
                .zip(self.len.chunks_mut(chunk))
            {
                scope.spawn(async move {
                    for i in 0..pos.len() {
                        let old = pos[i];
                        let p = rk2_step_speed(field, old, speed * dt);
                        pos[i] = p;
                        life[i] -= dt;
                        let out = p.x < 0.0 || p.x > bounds.x || p.y < 0.0 || p.y > bounds.y;
                        if life[i] <= 0.0 || out {
                            // Mark for the serial respawn pass; don't record.
                            life[i] = f32::MIN;
                            continue;
                        }
                        // A slow frame records several samples; spacing them
                        // along this frame's movement keeps the trail's
                        // wall-clock span intact instead of stacking
                        // duplicates at the head.
                        for r in 1..=records {
                            let sample = old.lerp(p, r as f32 / records as f32);
                            head[i] = (head[i] + 1) % TRAIL_CAP as u8;
                            trail[i * TRAIL_CAP + head[i] as usize] = sample;
                            len[i] = (len[i] + 1).min(TRAIL_CAP as u8);
                        }
                    }
                });
            }
        });

        for i in 0..n {
            if self.life[i] == f32::MIN {
                self.respawn(i, bounds);
            }
        }
    }
}

/// One RK2 step of `dist` px (particles' per-frame move).
fn rk2_step_speed(field: &FlowField, p: Vec2, dist: f32) -> Vec2 {
    let d1 = field.sample_dir(p);
    let d2 = field.sample_dir(p + d1 * (dist * 0.5));
    p + d2 * dist
}

// ---------------------------------------------------------------------------
// PCG-ish RNG (xorshift64*): tiny, deterministic, no `rand` dependence —
// seeds reproduce fields and tests exactly.

fn pcg_seed(seed: u64) -> u64 {
    seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1
}

fn pcg_next(state: &mut u64) -> u32 {
    let mut x = *state;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    *state = x;
    (x.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 32) as u32
}

fn pcg_next_f32(state: &mut u64) -> f32 {
    (pcg_next(state) >> 8) as f32 / (1u32 << 24) as f32
}

// ---------------------------------------------------------------------------
// Build state: when to (re)generate the field, the streamlines, and the
// static geometry — the Lua's `signature()` + throttle + canvas rebuild.

/// Every value whose change requires a rebuild (the Lua's `signature`).
/// The seed is the raw slider value, not floored — dragging it pans the
/// field continuously. The particle tunables are deliberately absent: they
/// animate live over the static field.
#[derive(Clone, PartialEq, Debug)]
pub struct BuildSig {
    pub seed: f32,
    pub scale: f32,
    pub noise_scale: f32,
    pub octaves: u32,
    pub roughness: f32,
    pub warp: f32,
    pub swirl: f32,
    pub mode: FlowMode,
    pub palette: super::settings::FlowPalette,
    pub detail: usize,
    pub length: usize,
    pub line_width: f32,
    pub opacity: f32,
    pub arrowheads: bool,
    pub background: bool,
    pub bounds: Vec2,
    pub z: f32,
}

impl BuildSig {
    pub fn of(settings: &FlowSettings, bounds: Vec2, z: f32) -> Self {
        Self {
            seed: settings.seed,
            scale: settings.scale.round().max(1.0),
            noise_scale: settings.noise_scale,
            octaves: settings.octaves.round().max(1.0) as u32,
            roughness: settings.roughness,
            warp: settings.warp,
            swirl: settings.swirl,
            mode: settings.mode,
            palette: settings.palette,
            detail: settings.detail.round() as usize,
            length: settings.length.round() as usize,
            line_width: settings.line_width,
            opacity: settings.opacity,
            arrowheads: settings.arrowheads,
            background: settings.background,
            bounds,
            z,
        }
    }
}

/// Rebuild bookkeeping: what's applied, the evolve clock, and the version
/// the renderer keys its static-geometry re-emit on.
#[derive(Resource)]
pub struct FlowState {
    pub applied: Option<BuildSig>,
    /// Evolve-axis position the *applied* field was built at.
    pub z: f32,
    /// Continuously accumulating evolve time, promoted into `z` at most
    /// every `EVOLVE_TICK` seconds.
    z_accum: f32,
    last_build: f32,
    /// Bumped on every rebuild; the renderer re-emits static geometry when
    /// it sees a version it hasn't drawn.
    pub version: u64,
}

impl Default for FlowState {
    fn default() -> Self {
        Self {
            applied: None,
            z: 0.0,
            z_accum: 0.0,
            last_build: f32::MIN,
            version: 0,
        }
    }
}

/// Label for the flow sim systems; the renderer emits geometry after.
#[derive(SystemSet, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct FlowSimSet;

pub fn plugin(app: &mut App) {
    app.init_resource::<FlowField>()
        .init_resource::<FlowStreamlines>()
        .init_resource::<FlowParticles>()
        .init_resource::<FlowState>()
        .add_systems(
            Update,
            (
                // Restart and particles step while playing AND behind the
                // menu (the live backdrop) — they pause in Options, like
                // the fish. The field, by contrast, rebuilds in Options
                // too: the popup sliders are flow's main controls and the
                // original redrew live as they dragged. (Evolve time only
                // advances while Playing — a menu backdrop stays static.)
                handle_restart.run_if(sim_active),
                update_field,
                step_particles.run_if(sim_active),
            )
                .chain()
                .in_set(FlowSimSet)
                .after(update_sim_bounds)
                .run_if(experiment_active(ExperimentId::Flow)),
        );
}

/// [R] / the popup's Restart / the menu's entry: a brand-new field (the
/// original `reset()` re-seeds) and fresh particles. The "New field" button
/// re-seeds without touching the particles (the original `regenerate()`) —
/// that one lives in the UI and just writes `settings.seed`.
fn handle_restart(
    keys: Res<ButtonInput<KeyCode>>,
    state: Res<State<AppState>>,
    text_edits: Query<(), With<crate::ui::ValueEdit>>,
    mut request: ResMut<RestartRequested>,
    mut settings: ResMut<FlowSettings>,
    mut flow_state: ResMut<FlowState>,
    mut particles: ResMut<FlowParticles>,
) {
    // An "r" typed into a value edit is not a restart.
    let key_restart = *state.get() == AppState::Playing
        && keys.just_pressed(KeyCode::KeyR)
        && text_edits.is_empty();
    if !(request.0 || key_restart) {
        return;
    }
    request.0 = false;
    settings.seed = rand::Rng::random_range(&mut rand::rng(), 0..SEED_PERIOD as u32) as f32;
    flow_state.z = 0.0;
    flow_state.z_accum = 0.0;
    particles.clear();
}

/// Rebuild the field (and retrace the streamlines) when a build-affecting
/// tunable, the window size, or the evolve clock changed — throttled to
/// ~15/s while the mouse button is held (a drag), with an exact rebuild on
/// release, like the original.
#[allow(clippy::too_many_arguments)]
fn update_field(
    settings: Res<FlowSettings>,
    bounds: Res<SimBounds>,
    time: Res<Time>,
    state: Res<State<AppState>>,
    mouse: Res<ButtonInput<MouseButton>>,
    mut flow: ResMut<FlowState>,
    mut field: ResMut<FlowField>,
    mut lines: ResMut<FlowStreamlines>,
) {
    let now = time.elapsed_secs();

    // Advance the evolve clock (Playing only — the audit's menu-backdrop
    // decision), promoting it into the build at most every EVOLVE_TICK.
    if settings.evolve > 0.0 && *state.get() == AppState::Playing {
        flow.z_accum += settings.evolve * time.delta_secs() * EVOLVE_RATE;
        if now - flow.last_build >= EVOLVE_TICK {
            flow.z = flow.z_accum;
        }
    }

    let sig = BuildSig::of(&settings, bounds.0, flow.z);
    if flow.applied.as_ref() == Some(&sig) {
        return;
    }
    // Live preview while a slider is held: rebuild as it changes, but
    // throttled so a heavy field isn't regenerated every single frame.
    // (Evolve ticks pass: they're already capped at 30 Hz above.)
    if mouse.pressed(MouseButton::Left) && now - flow.last_build < REBUILD_THROTTLE {
        return;
    }

    let params = FieldParams::of(&settings, flow.z);
    field.rebuild(&params, bounds.0, sig.scale);
    if sig.mode == FlowMode::Streamlines {
        retrace_streamlines(&mut lines, &field, sig.seed, sig.detail, sig.length, bounds.0);
    } else {
        lines.0.clear();
    }
    flow.applied = Some(sig);
    flow.last_build = now;
    flow.version = flow.version.wrapping_add(1);
}

/// Advance the particle layer (the Particles view, or the overlay checkbox
/// on the other views). When neither is on the particles idle untouched —
/// the renderer clears their geometry.
fn step_particles(
    settings: Res<FlowSettings>,
    bounds: Res<SimBounds>,
    time: Res<Time>,
    field: Res<FlowField>,
    mut particles: ResMut<FlowParticles>,
) {
    if !(settings.animate || settings.mode == FlowMode::Particles) || field.cols == 0 {
        return;
    }
    particles.resize(settings.particle_count.round().max(1.0) as usize, bounds.0);
    particles.step(&field, settings.particle_speed, time.delta_secs(), bounds.0);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::experiments::flow::settings::FlowPalette;

    const BOUNDS: Vec2 = Vec2::new(1280.0, 800.0);

    fn lua_params() -> FieldParams {
        FieldParams::of(&FlowSettings::default(), 0.0)
    }

    #[test]
    fn noise_is_deterministic_and_bounded() {
        let noise = Perlin::shared();
        for i in 0..1000 {
            let (x, y, z) = (i as f32 * 0.137, i as f32 * 0.071, i as f32 * 0.029);
            let a = noise.noise01(x, y, z);
            let b = noise.noise01(x, y, z);
            assert_eq!(a, b);
            assert!((0.0..=1.0).contains(&a), "noise01({x},{y},{z}) = {a}");
        }
        // Not a constant: samples spread over a decent share of [0,1].
        let samples: Vec<f32> = (0..1000)
            .map(|i| noise.noise01(i as f32 * 0.137, i as f32 * 0.071, 0.0))
            .collect();
        let min = samples.iter().cloned().fold(1.0f32, f32::min);
        let max = samples.iter().cloned().fold(0.0f32, f32::max);
        assert!(max - min > 0.5, "noise range collapsed: [{min}, {max}]");
    }

    #[test]
    fn noise_is_continuous() {
        let noise = Perlin::shared();
        for i in 0..500 {
            let (x, y) = (i as f32 * 0.193, i as f32 * 0.117);
            let a = noise.noise01(x, y, 0.3);
            let b = noise.noise01(x + 1e-3, y, 0.3);
            assert!((a - b).abs() < 0.01, "jump at ({x},{y}): {a} vs {b}");
        }
    }

    /// The wrap in `seed_base` is only sound if the noise really repeats
    /// every 256 lattice units (the permutation table indexes `& 255`).
    /// Equality is approximate: the +256 costs the f32 fraction a few low
    /// bits (≤ ulp(256) ≈ 3e-5 — far below any per-cell noise delta).
    #[test]
    fn noise_lattice_period_is_256() {
        let noise = Perlin::shared();
        for i in 0..200 {
            let (x, y, z) = (i as f32 * 0.73, i as f32 * 0.41, i as f32 * 0.13);
            let a = noise.noise3(x, y, z);
            assert!((a - noise.noise3(x + 256.0, y, z)).abs() < 1e-4);
            assert!((a - noise.noise3(x, y + 256.0, z)).abs() < 1e-4);
        }
    }

    /// `seed_base`: offsets stay inside one lattice period (precision at
    /// any seed), the fields repeat exactly at SEED_PERIOD, and seeds
    /// inside the period are distinct.
    #[test]
    fn seed_base_wraps_into_the_lattice_period() {
        for seed in [0.0, 1.0, 9999.0, 123_456.0, SEED_PERIOD - 1.0, SEED_PERIOD] {
            let base = seed_base(seed);
            assert!(
                (0.0..256.0).contains(&base.x) && (0.0..256.0).contains(&base.y),
                "seed {seed}: base {base} outside the lattice period"
            );
        }
        // One full period later the pan lands on the same lattice phase.
        let a = seed_base(1234.0);
        let b = seed_base(1234.0 + SEED_PERIOD);
        assert!((a - b).length() < 1e-3, "period off: {a} vs {b}");
        // But seeds inside the period are distinct fields.
        assert!((seed_base(1234.0) - seed_base(1235.0)).length() > 0.05);
        // And a six-digit seed keeps sub-milli precision: its neighbour a
        // quarter-step away stays resolvable (the unwrapped offset would
        // have an f32 ulp comparable to the step itself).
        let big = seed_base(200_000.0);
        let near = seed_base(200_000.25);
        assert!((near - big).length() > 0.01, "big-seed precision collapsed");
    }

    #[test]
    fn fbm_one_octave_is_plain_noise() {
        let noise = Perlin::shared();
        for i in 0..100 {
            let (x, y) = (i as f32 * 0.31, i as f32 * 0.17);
            assert_eq!(fbm(noise, x, y, 0.0, 1, 0.5), noise.noise01(x, y, 0.0));
        }
    }

    #[test]
    fn evolve_axis_changes_the_field() {
        let mut a = FlowField::default();
        let mut b = FlowField::default();
        a.rebuild(&FieldParams { z: 0.0, ..lua_params() }, BOUNDS, 20.0);
        b.rebuild(&FieldParams { z: 0.5, ..lua_params() }, BOUNDS, 20.0);
        let differing = a
            .angles
            .iter()
            .zip(&b.angles)
            .filter(|(x, y)| (**x - **y).abs() > 1e-3)
            .count();
        assert!(differing > a.angles.len() / 2, "evolve barely moved the field");
    }

    #[test]
    fn field_dims_match_bounds() {
        let mut field = FlowField::default();
        field.rebuild(&lua_params(), BOUNDS, 20.0);
        assert_eq!(field.cols, 64);
        assert_eq!(field.rows, 40);
        assert_eq!(field.angles.len(), 64 * 40);
        assert_eq!(field.dirs.len(), 64 * 40);
    }

    #[test]
    fn field_is_deterministic_per_seed() {
        let mut a = FlowField::default();
        let mut b = FlowField::default();
        a.rebuild(&lua_params(), BOUNDS, 20.0);
        b.rebuild(&lua_params(), BOUNDS, 20.0);
        assert_eq!(a.angles, b.angles);

        let other = FieldParams {
            base: seed_base(99.0),
            ..lua_params()
        };
        b.rebuild(&other, BOUNDS, 20.0);
        assert_ne!(a.angles, b.angles);
    }

    /// Swirl is a constant rotation added in radians (the Lua's
    /// `math.rad(t.swirl)` after `mapValue`), not in degrees.
    #[test]
    fn swirl_adds_radians() {
        let mut plain = FlowField::default();
        let mut swirled = FlowField::default();
        plain.rebuild(&FieldParams { swirl: 0.0, ..lua_params() }, BOUNDS, 20.0);
        swirled.rebuild(
            &FieldParams {
                swirl: 90.0f32.to_radians(),
                ..lua_params()
            },
            BOUNDS,
            20.0,
        );
        for (a, b) in plain.angles.iter().zip(&swirled.angles) {
            let delta = (b - a - 90.0f32.to_radians()).abs();
            assert!(delta < 1e-4, "swirl delta off: {delta}");
        }
    }

    /// The settings → params path converts the degree tunable to radians
    /// (the Lua's `math.rad(t.swirl)`).
    #[test]
    fn field_params_convert_swirl_degrees() {
        let mut settings = FlowSettings::default();
        settings.swirl = 90.0;
        let params = FieldParams::of(&settings, 0.0);
        assert!((params.swirl - std::f32::consts::FRAC_PI_2).abs() < 1e-6);
        settings.swirl = -100.0;
        let params = FieldParams::of(&settings, 0.0);
        assert!((params.swirl - (-100.0f32).to_radians()).abs() < 1e-6);
    }

    #[test]
    fn warp_changes_the_field() {
        let mut plain = FlowField::default();
        let mut warped = FlowField::default();
        plain.rebuild(&lua_params(), BOUNDS, 20.0);
        warped.rebuild(&FieldParams { warp: 2.0, ..lua_params() }, BOUNDS, 20.0);
        assert_ne!(plain.angles, warped.angles);
    }

    /// Hand-built 2×2 field: centers exact, midpoints the normalized mean,
    /// outside points clamped to the edge cells.
    #[test]
    fn bilinear_sample_interpolates_cell_centers() {
        let field = FlowField {
            cols: 2,
            rows: 2,
            scale: 10.0,
            angles: vec![0.0; 4],
            dirs: vec![Vec2::X, Vec2::Y, Vec2::X, Vec2::Y],
        };
        // Cell centers: (5,5)=X, (15,5)=Y, (5,15)=X, (15,15)=Y.
        assert!((field.sample_dir(Vec2::new(5.0, 5.0)) - Vec2::X).length() < 1e-5);
        assert!((field.sample_dir(Vec2::new(15.0, 5.0)) - Vec2::Y).length() < 1e-5);
        // Midway between X and Y dirs: the normalized 45° mean.
        let mid = field.sample_dir(Vec2::new(10.0, 5.0));
        let expect = Vec2::new(1.0, 1.0).normalize();
        assert!((mid - expect).length() < 1e-5, "midpoint {mid}");
        // Outside the grid clamps to the edge.
        assert!((field.sample_dir(Vec2::new(-50.0, -50.0)) - Vec2::X).length() < 1e-5);
        assert!((field.sample_dir(Vec2::new(999.0, 5.0)) - Vec2::Y).length() < 1e-5);
    }

    /// ε and 2π−ε must average to ~+x (vector lerp), not to a backwards π
    /// (naive angle lerp).
    #[test]
    fn bilinear_sample_is_wrap_safe() {
        let e = 0.2f32;
        let field = FlowField {
            cols: 2,
            rows: 1,
            scale: 10.0,
            angles: vec![e, TAU - e],
            dirs: vec![Vec2::new(e.cos(), e.sin()), Vec2::new(e.cos(), -e.sin())],
        };
        let mid = field.sample_dir(Vec2::new(10.0, 5.0));
        assert!((mid - Vec2::X).length() < 1e-5, "wrap midpoint {mid}");
    }

    #[test]
    fn streamlines_terminate_and_stay_near_bounds() {
        let mut field = FlowField::default();
        field.rebuild(&lua_params(), BOUNDS, 20.0);
        let mut lines = FlowStreamlines::default();
        retrace_streamlines(&mut lines, &field, 1234.0, 300, 50, BOUNDS);
        assert_eq!(lines.0.len(), 300);
        let margin = field.scale; // one step past the border at most
        for line in &lines.0 {
            assert!(line.len() <= 50 * 2 + 1, "line too long: {}", line.len());
            assert!(!line.is_empty());
            for p in line {
                assert!(
                    p.x >= -margin
                        && p.x <= BOUNDS.x + margin
                        && p.y >= -margin
                        && p.y <= BOUNDS.y + margin,
                    "point far out of bounds: {p}"
                );
            }
        }
    }

    #[test]
    fn streamlines_are_deterministic_per_seed() {
        let mut field = FlowField::default();
        field.rebuild(&lua_params(), BOUNDS, 20.0);
        let mut a = FlowStreamlines::default();
        let mut b = FlowStreamlines::default();
        retrace_streamlines(&mut a, &field, 1234.0, 100, 30, BOUNDS);
        retrace_streamlines(&mut b, &field, 1234.0, 100, 30, BOUNDS);
        assert_eq!(a.0, b.0);
        retrace_streamlines(&mut b, &field, 1235.0, 100, 30, BOUNDS);
        assert_ne!(a.0, b.0);
    }

    #[test]
    fn particles_respawn_resets_the_trail() {
        let mut field = FlowField::default();
        // Constant rightward field: particles march off the right edge.
        field.rebuild(
            &FieldParams {
                swirl: 0.0,
                ..lua_params()
            },
            BOUNDS,
            20.0,
        );
        for dir in &mut field.dirs {
            *dir = Vec2::X;
        }
        let mut particles = FlowParticles::default();
        particles.resize(32, BOUNDS);
        // March long enough that everyone leaves the screen or ages out at
        // least once (screen width / speed < total time).
        for _ in 0..600 {
            particles.step(&field, 400.0, 1.0 / 60.0, BOUNDS);
        }
        for i in 0..particles.count() {
            let p = particles.pos[i];
            assert!(
                p.x >= 0.0 && p.x <= BOUNDS.x && p.y >= 0.0 && p.y <= BOUNDS.y,
                "particle {i} not respawned in bounds: {p}"
            );
            assert!(particles.life[i] > f32::MIN);
            // Trails never bridge a respawn: every stored sample is close
            // to its neighbour (a cross-screen streak would be ~1000px).
            let pts: Vec<Vec2> = particles.trail_iter(i).collect();
            for pair in pts.windows(2) {
                assert!(
                    (pair[0] - pair[1]).length() < 60.0,
                    "trail streak on particle {i}: {} -> {}",
                    pair[0],
                    pair[1]
                );
            }
        }
    }

    /// Trail recording is keyed to simulated time, not frames: the same
    /// elapsed time records the same number of samples whatever the dt
    /// slicing, and the head ends at the same place on a uniform field.
    #[test]
    fn trail_recording_is_frame_rate_independent() {
        let mut field = FlowField::default();
        field.rebuild(&lua_params(), BOUNDS, 20.0);
        for dir in &mut field.dirs {
            *dir = Vec2::X;
        }
        let run = |dt: f32, steps: usize| {
            let mut particles = FlowParticles::default();
            particles.resize(4, BOUNDS);
            // Pin starting state: identical spawns come from the seeded RNG.
            for _ in 0..steps {
                particles.step(&field, 60.0, dt, BOUNDS);
            }
            let lens: Vec<u8> = particles.len.clone();
            (particles.pos.clone(), lens)
        };
        let (pos_60, len_60) = run(1.0 / 60.0, 12);
        let (pos_30, len_30) = run(1.0 / 30.0, 6);
        assert_eq!(len_60, len_30, "sample counts diverge with dt");
        for (a, b) in pos_60.iter().zip(&pos_30) {
            assert!((a - b).length() < 1e-3, "heads diverge: {a} vs {b}");
        }
    }

    #[test]
    fn particle_resize_up_and_down() {
        let mut particles = FlowParticles::default();
        particles.resize(100, BOUNDS);
        assert_eq!(particles.count(), 100);
        assert_eq!(particles.trail.len(), 100 * TRAIL_CAP);
        particles.resize(10, BOUNDS);
        assert_eq!(particles.count(), 10);
        assert_eq!(particles.trail.len(), 10 * TRAIL_CAP);
        for i in 0..10 {
            let p = particles.pos[i];
            assert!(p.x >= 0.0 && p.x <= BOUNDS.x && p.y >= 0.0 && p.y <= BOUNDS.y);
        }
    }

    /// The rebuild signature: build-affecting fields trigger, particle
    /// fields don't, and the raw (un-floored) seed morphs mid-drag.
    #[test]
    fn build_signature_tracks_the_right_fields() {
        let settings = FlowSettings::default();
        let base = BuildSig::of(&settings, BOUNDS, 0.0);

        let mut s = settings.clone();
        s.seed += 0.25; // a fraction of one seed step — must still trigger
        assert_ne!(BuildSig::of(&s, BOUNDS, 0.0), base);

        let mut s = settings.clone();
        s.warp = 1.0;
        assert_ne!(BuildSig::of(&s, BOUNDS, 0.0), base);

        let mut s = settings.clone();
        s.palette = FlowPalette::Fire;
        assert_ne!(BuildSig::of(&s, BOUNDS, 0.0), base);

        // Particle tunables never rebuild the field.
        let mut s = settings.clone();
        s.particle_count = 9999.0;
        s.particle_speed = 999.0;
        s.trail_fade = 0.4;
        s.animate = true;
        assert_eq!(BuildSig::of(&s, BOUNDS, 0.0), base);

        // The evolve z and the window size do.
        assert_ne!(BuildSig::of(&settings, BOUNDS, 0.1), base);
        assert_ne!(BuildSig::of(&settings, Vec2::new(640.0, 480.0), 0.0), base);
    }
}
