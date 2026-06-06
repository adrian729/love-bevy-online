//! The boids simulation: Reynolds separation / alignment / cohesion plus
//! mouse attraction/repulsion, integrated on a toroidal screen. A direct
//! behavioural port of `lib/flock.lua`.
//!
//! The flock is plain data, not entities: [`Flock`] holds the state, the
//! steering kernel runs over cell-sorted SoA arrays on the compute task
//! pool, and each frame publishes `[x, y, angle]` records that
//! [`crate::render`] uploads as one instanced draw. (One entity per boid
//! worked to ~40k; past that, per-entity engine bookkeeping — transform
//! propagation, visibility, extraction — dominated the frame.)

use std::f32::consts::TAU;

use bevy::asset::RenderAssetUsages;
use bevy::camera::RenderTarget;
use bevy::prelude::*;
use bevy::render::render_resource::{Extent3d, TextureDimension, TextureFormat, TextureUsages};
use bevy::render::view::screenshot::{Screenshot, save_to_disk};
use bevy::tasks::ComputeTaskPool;
use bevy::window::PrimaryWindow;
use rand::Rng;

use crate::AppState;
use crate::render::{BoidInstance, FlockRenderData};
use crate::settings::{
    MAX_FORCE, MOUSE_ATTRACT_K, MOUSE_NEAR, MOUSE_REPEL_K, NEIGHBOUR_DIST, REF_FPS, SEPARATE_DIST,
    SimSettings,
};

/// True when this run simulates on the CPU (the `cpu` or `nosim` perf
/// flags); the GPU compute sim is the default otherwise.
pub fn cpu_sim_selected() -> bool {
    std::env::args().any(|arg| arg == "cpu" || arg == "nosim")
}

pub fn plugin(app: &mut App) {
    // Perf-test only (`boids <count> nosim`): spawn and render the flock but
    // skip steering, to measure the render floor in isolation.
    let nosim = std::env::args().any(|arg| arg == "nosim");
    let cpu = cpu_sim_selected();
    app.init_resource::<PointerOverUi>()
        .init_resource::<RestartRequested>()
        .init_resource::<SimBounds>()
        .init_resource::<Flock>()
        .add_systems(Startup, setup)
        .add_systems(Update, (update_sim_bounds, headless_snapshots))
        .add_systems(
            Update,
            (
                handle_restart,
                sync_flock_size,
                flocking.run_if(move || !nosim),
            )
                .chain()
                .after(update_sim_bounds)
                .run_if(move || cpu)
                .run_if(in_state(AppState::Playing)),
        );
}

/// One boid: position and velocity in world px / px-per-second.
#[derive(Clone, Copy, Default)]
pub struct BoidState {
    pub pos: Vec2,
    pub vel: Vec2,
}

/// The whole flock. Stored in spatial (cell-sorted) order as a side effect
/// of the per-frame counting sort; boids carry no identity.
#[derive(Resource, Default)]
pub struct Flock(pub Vec<BoidState>);

/// True while the cursor is busy on the UI — the flock ignores the mouse
/// then, like `ignore_mouse` in the original.
#[derive(Resource, Default)]
pub struct PointerOverUi(pub bool);

/// Set by the UI (or the R key) to respawn the flock.
#[derive(Resource, Default)]
pub struct RestartRequested(pub bool);

/// Perf-test only (`boids <count> pin`): pretend the mouse sits at screen
/// centre. A spawn blob disperses in under a second, so the only way to
/// measure the sustained worst case — the whole flock held in a dense ring —
/// is a permanent attractor.
#[derive(Resource, Default)]
pub struct PinnedAttractor(pub bool);

/// Perf-test only (`boids <count> headless`): there is no window; the camera
/// renders to an offscreen texture instead of a swapchain.
#[derive(Resource, Default)]
pub struct HeadlessRender(pub bool);

/// The offscreen texture headless mode renders into. Its presence also
/// enables [`headless_snapshots`].
#[derive(Resource)]
struct HeadlessTarget(Handle<Image>);

/// The simulation area. Mirrors the primary window's size while one exists;
/// in headless perf runs it stays at the default window size so the flock
/// density (and therefore the workload) matches the windowed game.
#[derive(Resource)]
pub struct SimBounds(pub Vec2);

impl Default for SimBounds {
    fn default() -> Self {
        Self(Vec2::new(1280.0, 800.0))
    }
}

/// Keep [`SimBounds`] in sync with the window (live resizing included).
pub fn update_sim_bounds(
    window: Query<&Window, With<PrimaryWindow>>,
    mut bounds: ResMut<SimBounds>,
) {
    if let Ok(window) = window.single() {
        bounds.0 = Vec2::new(window.width(), window.height()).max(Vec2::ONE);
    }
}

/// In headless perf runs, save the offscreen target to
/// `/tmp/boids_headless_{0,1,2}.png` every few seconds (cycling), so the
/// flock's behaviour stays visually verifiable even while the machine's
/// display is asleep — macOS throttles presentation then, but offscreen
/// rendering is unaffected.
fn headless_snapshots(
    mut commands: Commands,
    time: Res<Time>,
    target: Option<Res<HeadlessTarget>>,
    mut next: Local<f32>,
    mut index: Local<u32>,
) {
    let Some(target) = target else { return };
    if *next == 0.0 {
        // Skip the first seconds: the flock is still dispersing from spawn.
        *next = 4.0;
        return;
    }
    if time.elapsed_secs() < *next {
        return;
    }
    *next = time.elapsed_secs() + 5.0;
    let path = format!("/tmp/boids_headless_{}.png", *index % 3);
    *index += 1;
    commands
        .spawn(Screenshot::image(target.0.clone()))
        .observe(save_to_disk(path));
}

fn setup(
    mut commands: Commands,
    mut images: ResMut<Assets<Image>>,
    headless: Res<HeadlessRender>,
    bounds: Res<SimBounds>,
) {
    if headless.0 {
        // No window to present to: render into an offscreen texture of the
        // same size, so perf runs exercise the real render pipeline.
        let mut target = Image::new_fill(
            Extent3d {
                width: bounds.0.x as u32,
                height: bounds.0.y as u32,
                depth_or_array_layers: 1,
            },
            TextureDimension::D2,
            &[0, 0, 0, 255],
            TextureFormat::Bgra8UnormSrgb,
            RenderAssetUsages::default(),
        );
        target.texture_descriptor.usage =
            TextureUsages::TEXTURE_BINDING | TextureUsages::RENDER_ATTACHMENT;
        let handle = images.add(target);
        commands.insert_resource(HeadlessTarget(handle.clone()));
        // Msaa off: the LÖVE original drew without antialiasing, and at high
        // counts the flock piles up — 4x the blending samples is pure
        // fill-rate cost on exactly those frames.
        commands.spawn((Camera2d, Msaa::Off, RenderTarget::Image(handle.into())));
    } else {
        commands.spawn((Camera2d, Msaa::Off));
    }
}

/// Random position on screen, random heading, random speed up to max — the
/// original's `initPositions` / `initVelocities`.
fn random_boid(half: Vec2, max_speed: f32, rng: &mut impl Rng) -> BoidState {
    BoidState {
        pos: Vec2::new(
            rng.random_range(-half.x..=half.x),
            rng.random_range(-half.y..=half.y),
        ),
        vel: Vec2::from_angle(rng.random_range(0.0..TAU)) * rng.random_range(0.0..=max_speed),
    }
}

/// Clear the whole flock on [R] or when the UI requests a restart;
/// `sync_flock_size` rebuilds it at random positions, which is exactly what
/// the original's `reset()` does.
fn handle_restart(
    keys: Res<ButtonInput<KeyCode>>,
    state: Res<State<AppState>>,
    mut request: ResMut<RestartRequested>,
    mut flock: ResMut<Flock>,
    mut instances: ResMut<FlockRenderData>,
) {
    let key_restart = *state.get() == AppState::Playing && keys.just_pressed(KeyCode::KeyR);
    if request.0 || key_restart {
        request.0 = false;
        flock.0.clear();
        instances.0.clear();
    }
}

/// Grow or shrink the flock to the tuned count live, like `Flock:setSize`.
/// The instance records stay index-aligned with the state vector.
fn sync_flock_size(
    settings: Res<SimSettings>,
    bounds: Res<SimBounds>,
    mut flock: ResMut<Flock>,
    mut instances: ResMut<FlockRenderData>,
) {
    let half = bounds.0 / 2.0;
    let target = settings.count.round().max(1.0) as usize;
    let current = flock.0.len();
    if current < target {
        let mut rng = rand::rng();
        flock.0.reserve(target - current);
        instances.0.reserve(target - current);
        for _ in current..target {
            let boid = random_boid(half, settings.speed, &mut rng);
            flock.0.push(boid);
            instances.0.push(BoidInstance {
                pos: boid.pos,
                rot: boid.vel.normalize_or(Vec2::X),
            });
        }
    } else {
        // Remove pseudo-randomly: the vector is in spatial order, so popping
        // the tail would visibly eat the flock from one screen corner.
        let mut h = 0x9E37_79B9u32 ^ current as u32;
        for _ in target..current {
            h = h.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let j = (h as usize) % flock.0.len();
            flock.0.swap_remove(j);
            instances.0.swap_remove(j);
        }
    }
}

/// Element-wise square root (IEEE-exact, one NEON instruction).
#[cfg(target_arch = "aarch64")]
#[inline(always)]
fn sqrt4(v: Vec4) -> Vec4 {
    use core::arch::aarch64::float32x4_t;
    // glam's `Vec4` is `repr(transparent)` over `float32x4_t`.
    unsafe {
        core::mem::transmute::<float32x4_t, Vec4>(core::arch::aarch64::vsqrtq_f32(
            core::mem::transmute::<Vec4, float32x4_t>(v),
        ))
    }
}

#[cfg(not(target_arch = "aarch64"))]
#[inline(always)]
fn sqrt4(v: Vec4) -> Vec4 {
    Vec4::new(v.x.sqrt(), v.y.sqrt(), v.z.sqrt(), v.w.sqrt())
}

/// The original's `target_force` — `k * limit(normalize(dir) * max_speed -
/// velocity, MAX_FORCE)` in per-frame units at the 60 fps reference,
/// converted to px/s² — evaluated for all four forces at once (separation,
/// alignment, cohesion, mouse), one per SIMD lane. A lane with `k = 0`
/// contributes exactly zero, which is how the caller expresses the scalar
/// version's "skip this force" branches.
#[inline(always)]
fn steering_forces(dir_x: Vec4, dir_y: Vec4, ks: Vec4, vel: Vec2, max_speed: f32) -> Vec2 {
    let tiny = Vec4::splat(1e-30);
    // normalize_or_zero, lane-wise.
    let d2 = dir_x * dir_x + dir_y * dir_y;
    let inv_len = Vec4::select(
        d2.cmpgt(Vec4::ZERO),
        Vec4::ONE / sqrt4(d2.max(tiny)),
        Vec4::ZERO,
    );
    let scaled = inv_len * Vec4::splat(max_speed);
    let steer_x = dir_x * scaled - Vec4::splat(vel.x / REF_FPS);
    let steer_y = dir_y * scaled - Vec4::splat(vel.y / REF_FPS);
    // clamp_length_max(MAX_FORCE): scale by min(1, MAX_FORCE / |steer|).
    let len = sqrt4(steer_x * steer_x + steer_y * steer_y);
    let clamp = (Vec4::splat(MAX_FORCE) / len.max(tiny)).min(Vec4::ONE);
    let w = ks * clamp * Vec4::splat(REF_FPS * REF_FPS);
    Vec2::new((w * steer_x).element_sum(), (w * steer_y).element_sum())
}

/// Per-boid cap on neighbour candidates examined per frame. When the whole
/// flock piles onto the cursor, every boid has thousands of in-radius
/// neighbours and the per-neighbour rules degenerate to O(n²). Above this
/// cap we sample the 3x3 candidate cells instead: all three steering forces
/// only use the *direction* of the neighbour aggregate (`target_force`
/// normalizes it), so a few hundred uniform samples give a statistically
/// identical answer. With at most this many candidates the scan is
/// exhaustive and exact, which covers the LÖVE original's entire 10–300
/// range in ordinary play (a 300-boid flock piled into a single cell by the
/// mouse can momentarily exceed the budget and sample every 2nd or 3rd
/// candidate — statistically indistinguishable for a dense isotropic cloud).
const MAX_NEIGHBOUR_SAMPLES: usize = 128;

/// Frame-reused scratch for [`flocking`]: a flat counting-sort spatial grid
/// (no `HashMap`, no per-frame allocations once warm). The flock snapshot is
/// sorted by grid cell into per-component arrays, so each boid's 3x3
/// neighbourhood is three *contiguous* spans of SIMD-friendly memory, and
/// consecutive boids share a home cell — they reread the same spans straight
/// out of L1.
#[derive(Default)]
struct FlockGrid {
    cell_of: Vec<u32>,
    /// Per-task histograms: task `t` owns row `histo[t * ncells..][..ncells]`.
    histo: Vec<u32>,
    /// Exclusive prefix sums: cell `c` holds `s??[starts[c]..starts[c+1]]`.
    starts: Vec<u32>,
    spx: Vec<f32>,
    spy: Vec<f32>,
    svx: Vec<f32>,
    svy: Vec<f32>,
}

/// A raw output pointer for the parallel counting-sort scatter. The sort's
/// per-task cursor ranges partition the output exactly — every index is
/// written once, by one task — but that invariant lives in the cursor
/// arithmetic where the borrow checker can't see it.
#[derive(Clone, Copy)]
struct ScatterPtr<T>(*mut T);

unsafe impl<T: Send> Send for ScatterPtr<T> {}
unsafe impl<T: Send> Sync for ScatterPtr<T> {}

impl<T> ScatterPtr<T> {
    /// Safety: `i` must be in bounds, and no other task may write index `i`
    /// during the scatter.
    #[inline(always)]
    unsafe fn write(self, i: usize, value: T) {
        unsafe { self.0.add(i).write(value) }
    }
}

const NEIGHBOUR_D2: f32 = NEIGHBOUR_DIST * NEIGHBOUR_DIST;
const SEPARATE_D2: f32 = SEPARATE_DIST * SEPARATE_DIST;

/// Chunk sizes `(sort, steer)` for the parallel passes, derived from the
/// compute pool size so the tuning travels to other core counts: ~a dozen
/// steering chunks per worker (local density makes some chunks far heavier
/// than others — the work-stealing pool needs slack to balance), and 4x
/// bigger sort chunks (sort work is uniform per boid, and fewer tasks keep
/// the serial histogram combine short). Rounded to the nearest power of two
/// and floored at 1024 boids so the ~µs task-spawn cost stays invisible at
/// small counts. On the 6-thread pool this was tuned on, the derivation
/// reproduces the hand-found 4096/16384 at the certified 320k exactly.
fn chunk_sizes(n: usize, threads: usize) -> (usize, usize) {
    let ideal = n / (threads.max(1) * 12);
    let up = ideal.next_power_of_two();
    let steer = if up - ideal <= ideal.saturating_sub(up / 2) {
        up
    } else {
        up / 2
    }
    .clamp(1024, 65_536);
    (steer * 4, steer)
}

/// Borrowed view of the sorted SoA snapshot.
#[derive(Clone, Copy)]
struct FlockSoa<'a> {
    spx: &'a [f32],
    spy: &'a [f32],
    svx: &'a [f32],
    svy: &'a [f32],
}

/// Run the three neighbour rules over the candidate index runs for the boid
/// at `p`, returning `(sum_align, sum_cohere, sum_separate, n, n_avoid)`.
///
/// Branchless and four-wide (which is why the snapshot is SoA): the
/// in-radius tests become 0/1 lane weights — the acceptance branches they
/// replace flip randomly per candidate, and mispredictions cost more than
/// always accumulating. `d² > 0` excludes the boid itself, exactly like the
/// original's `d > 0` check. All accumulators are plain locals in one flat
/// loop so they stay in NEON registers.
#[inline(always)]
fn neighbour_rules(p: Vec2, soa: FlockSoa, runs: &[(usize, usize)]) -> (Vec2, Vec2, Vec2, f32, f32) {
    let px4 = Vec4::splat(p.x);
    let py4 = Vec4::splat(p.y);
    // Hoisted constants: written inline in the loop, each `splat` compiles
    // to a `memset_pattern16` *call* per iteration (LLVM models NEON's
    // load-dup as a 16-byte pattern fill), and the calls force every live
    // accumulator to spill around them — a ~4x kernel slowdown.
    let neighbour_d2 = Vec4::splat(NEIGHBOUR_D2);
    let separate_d2 = Vec4::splat(SEPARATE_D2);
    let eps = Vec4::splat(1e-12);
    let one = Vec4::ONE;
    let zero = Vec4::ZERO;
    let mut align_x = Vec4::ZERO;
    let mut align_y = Vec4::ZERO;
    let mut coh_x = Vec4::ZERO;
    let mut coh_y = Vec4::ZERO;
    let mut sep_x = Vec4::ZERO;
    let mut sep_y = Vec4::ZERO;
    let mut cnt_n = Vec4::ZERO;
    let mut cnt_s = Vec4::ZERO;
    let mut s_align = Vec2::ZERO;
    let mut s_coh = Vec2::ZERO;
    let mut s_sep = Vec2::ZERO;
    let mut s_cnt_n = 0.0f32;
    let mut s_cnt_s = 0.0f32;

    for &(s, e) in runs {
        let (qxs, qxr) = soa.spx[s..e].as_chunks::<4>();
        let (qys, qyr) = soa.spy[s..e].as_chunks::<4>();
        let (vxs, vxr) = soa.svx[s..e].as_chunks::<4>();
        let (vys, vyr) = soa.svy[s..e].as_chunks::<4>();
        for (((qx, qy), vx), vy) in qxs.iter().zip(qys).zip(vxs).zip(vys) {
            let qx = Vec4::from_array(*qx);
            let qy = Vec4::from_array(*qy);
            let dx = px4 - qx;
            let dy = py4 - qy;
            let d2 = dx * dx + dy * dy;
            let nonself = d2.cmpgt(zero);
            let w_n = Vec4::select(nonself & d2.cmplt(neighbour_d2), one, zero);
            let w_s = Vec4::select(nonself & d2.cmplt(separate_d2), one, zero);
            align_x += Vec4::from_array(*vx) * w_n;
            align_y += Vec4::from_array(*vy) * w_n;
            coh_x += qx * w_n;
            coh_y += qy * w_n;
            cnt_n += w_n;
            // Away-vector with 1/d falloff: normalize(p−q)/d is exactly
            // (p−q)/d². The `max` keeps the division finite where w_s
            // already zeroes the term out. (A NEON reciprocal-estimate +
            // Newton sequence was tried here and lost to plain `fdiv` —
            // Apple's divider is fast, and the extra temporaries spill.)
            let inv = w_s / d2.max(eps);
            sep_x += dx * inv;
            sep_y += dy * inv;
            cnt_s += w_s;
        }
        for (((qx, qy), vx), vy) in qxr.iter().zip(qyr).zip(vxr).zip(vyr) {
            let dx = p.x - qx;
            let dy = p.y - qy;
            let d2 = dx * dx + dy * dy;
            let w_n = (d2 > 0.0 && d2 < NEIGHBOUR_D2) as u32 as f32;
            let w_s = (d2 > 0.0 && d2 < SEPARATE_D2) as u32 as f32;
            s_align += Vec2::new(*vx, *vy) * w_n;
            s_coh += Vec2::new(*qx, *qy) * w_n;
            s_cnt_n += w_n;
            s_sep += Vec2::new(dx, dy) * (w_s / d2.max(1e-12));
            s_cnt_s += w_s;
        }
    }

    (
        s_align + Vec2::new(align_x.element_sum(), align_y.element_sum()),
        s_coh + Vec2::new(coh_x.element_sum(), coh_y.element_sum()),
        s_sep + Vec2::new(sep_x.element_sum(), sep_y.element_sum()),
        s_cnt_n + cnt_n.element_sum(),
        s_cnt_s + cnt_s.element_sum(),
    )
}

// Bevy systems take their dependencies as parameters; eight is fine.
#[allow(clippy::too_many_arguments)]
fn flocking(
    time: Res<Time>,
    settings: Res<SimSettings>,
    pointer_over_ui: Res<PointerOverUi>,
    pinned: Res<PinnedAttractor>,
    bounds: Res<SimBounds>,
    window: Query<&Window, With<PrimaryWindow>>,
    camera: Query<(&Camera, &GlobalTransform), With<Camera2d>>,
    mut flock: ResMut<Flock>,
    mut instances: ResMut<FlockRenderData>,
    mut grid: Local<FlockGrid>,
    mut timing: Local<(f64, f64, u32)>, // sort/steer diagnostics (stderr)
) {
    let _t0 = std::time::Instant::now();
    let dt = time.delta_secs();
    let size = bounds.0.max(Vec2::ONE);
    let half = size / 2.0;
    let max_speed = settings.speed;
    let (separation, alignment, cohesion) =
        (settings.separation, settings.alignment, settings.cohesion);

    // Cursor in world space; `None` while outside the window or busy on UI.
    let mouse = if pinned.0 {
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

    // Grid dimensions, cell size = neighbour radius: every neighbour within
    // NEIGHBOUR_DIST lives in the 3x3 cells around a boid (~O(n), not O(n²)).
    // Positions are wrapped into [-half, half), so cells cover the window;
    // the `min` clamp absorbs float edge cases and mid-resize stragglers.
    let cols = (size.x / NEIGHBOUR_DIST).ceil().max(1.0) as usize;
    let rows = (size.y / NEIGHBOUR_DIST).ceil().max(1.0) as usize;
    let ncells = cols * rows;
    let cell_x = move |x: f32| (((x + half.x) / NEIGHBOUR_DIST) as usize).min(cols - 1);
    let cell_y = move |y: f32| (((y + half.y) / NEIGHBOUR_DIST) as usize).min(rows - 1);

    // Parallel counting sort: per-task histograms, one combine pass, then a
    // scatter into cell-major SoA order where each task's cursors cover
    // disjoint output ranges (same trick GPU sorts use). Stable, so the
    // flock's spatial order — and the render order — stays steady frame to
    // frame. Everything downstream reads this sorted snapshot, so every boid
    // steers against the same frame, as the original.
    let n = flock.0.len();
    debug_assert_eq!(n, instances.0.len());
    let (sort_chunk, steer_chunk) = chunk_sizes(n, ComputeTaskPool::get().thread_num());
    let ntasks = n.div_ceil(sort_chunk).max(1);
    let g = &mut *grid;
    g.cell_of.resize(n, 0);
    g.histo.clear();
    g.histo.resize(ntasks * ncells, 0);
    let flock_ref = &flock.0;
    ComputeTaskPool::get().scope(|scope| {
        for ((boids, cells), histo) in flock_ref
            .chunks(sort_chunk)
            .zip(g.cell_of.chunks_mut(sort_chunk))
            .zip(g.histo.chunks_mut(ncells))
        {
            scope.spawn(async move {
                for (boid, cell) in boids.iter().zip(cells.iter_mut()) {
                    let c = cell_y(boid.pos.y) * cols + cell_x(boid.pos.x);
                    *cell = c as u32;
                    histo[c] += 1;
                }
            });
        }
    });
    // Combine: starts[c] = total before cell c; histo rows become each
    // task's write cursors.
    g.starts.clear();
    g.starts.resize(ncells + 1, 0);
    let mut running = 0u32;
    for c in 0..ncells {
        g.starts[c] = running;
        for t in 0..ntasks {
            let count = g.histo[t * ncells + c];
            g.histo[t * ncells + c] = running;
            running += count;
        }
    }
    g.starts[ncells] = running;
    g.spx.resize(n, 0.0);
    g.spy.resize(n, 0.0);
    g.svx.resize(n, 0.0);
    g.svy.resize(n, 0.0);
    let (spx, spy) = (ScatterPtr(g.spx.as_mut_ptr()), ScatterPtr(g.spy.as_mut_ptr()));
    let (svx, svy) = (ScatterPtr(g.svx.as_mut_ptr()), ScatterPtr(g.svy.as_mut_ptr()));
    let cell_of = &g.cell_of;
    ComputeTaskPool::get().scope(|scope| {
        for (boids, (cells, cursors)) in flock_ref
            .chunks(sort_chunk)
            .zip(cell_of.chunks(sort_chunk).zip(g.histo.chunks_mut(ncells)))
        {
            scope.spawn(async move {
                for (boid, &c) in boids.iter().zip(cells) {
                    let dst = cursors[c as usize] as usize;
                    cursors[c as usize] += 1;
                    // Safety: cursor ranges partition 0..n across tasks.
                    unsafe {
                        spx.write(dst, boid.pos.x);
                        spy.write(dst, boid.pos.y);
                        svx.write(dst, boid.vel.x);
                        svy.write(dst, boid.vel.y);
                    }
                }
            });
        }
    });
    let starts = &g.starts;
    let soa = FlockSoa {
        spx: &g.spx,
        spy: &g.spy,
        svx: &g.svx,
        svy: &g.svy,
    };
    let _t1 = std::time::Instant::now();

    // Salt the sampling start per frame so any residual sampling bias
    // dithers away over time instead of pushing steadily in one direction.
    let frame_salt = time.elapsed().as_millis() as usize;

    // Steer + integrate every boid in parallel on the compute task pool, in
    // sorted order; the new state overwrites `flock` (input comes from the
    // sorted snapshot, so the flock vector is free to be reused as output —
    // the flock simply *stays* in cell order, no identity, no write-back
    // permutation). Each task also publishes its boids' instance records.
    // Chunks sized (see `chunk_sizes`) so task-spawn overhead stays
    // negligible while the work-stealing pool gets enough pieces to balance.
    ComputeTaskPool::get().scope(|scope| {
        for (chunk_index, (states, insts)) in flock
            .0
            .chunks_mut(steer_chunk)
            .zip(instances.0.chunks_mut(steer_chunk))
            .enumerate()
        {
            scope.spawn(async move {
                for (k, (state, inst)) in states.iter_mut().zip(insts.iter_mut()).enumerate() {
                    let dst = chunk_index * steer_chunk + k;
                    let p = Vec2::new(soa.spx[dst], soa.spy[dst]);
                    let v = Vec2::new(soa.svx[dst], soa.svy[dst]);

                    // The 3x3 candidate cells around the boid. Cells are
                    // row-major, so each row of up-to-3 cells is one
                    // contiguous span of the sorted snapshot; edge rows and
                    // columns simply shrink (no toroidal lookup, matching
                    // the original's grid).
                    let (cx, cy) = (cell_x(p.x), cell_y(p.y));
                    let (x0, x1) = (cx.saturating_sub(1), (cx + 2).min(cols));
                    let (y0, y1) = (cy.saturating_sub(1), (cy + 2).min(rows));
                    let mut spans = [(0usize, 0usize); 3];
                    let mut nspans = 0usize;
                    let mut candidates = 0usize;
                    for y in y0..y1 {
                        let s = starts[y * cols + x0] as usize;
                        let e = starts[y * cols + x1] as usize;
                        spans[nspans] = (s, e);
                        nspans += 1;
                        candidates += e - s;
                    }

                    // Visit every candidate while they fit the sample
                    // budget; above it, take each span's proportional share
                    // as a contiguous block starting at a per-boid,
                    // per-frame pseudo-random offset, wrapping around the
                    // span (see MAX_NEIGHBOUR_SAMPLES). A circular window
                    // keeps every candidate equally likely while the kernel
                    // stays on contiguous, SIMD-friendly memory.
                    let stride = candidates.div_ceil(MAX_NEIGHBOUR_SAMPLES).max(1);
                    let mut runs = [(0usize, 0usize); 6];
                    let mut nruns = 0usize;
                    if stride == 1 {
                        runs[..nspans].copy_from_slice(&spans[..nspans]);
                        nruns = nspans;
                    } else {
                        let salt = dst.wrapping_mul(0x9E37_79B9).wrapping_add(frame_salt);
                        for &(s, e) in &spans[..nspans] {
                            let len = e - s;
                            if len == 0 {
                                continue;
                            }
                            let take = len.div_ceil(stride);
                            let start = salt % len;
                            let first = (start + take).min(len);
                            runs[nruns] = (s + start, s + first);
                            nruns += 1;
                            let wrapped = take - (first - start);
                            if wrapped > 0 {
                                runs[nruns] = (s, s + wrapped);
                                nruns += 1;
                            }
                        }
                    }
                    let (sum_align, sum_cohere, sum_separate, cnt_align, cnt_avoid) =
                        neighbour_rules(p, soa, &runs[..nruns]);

                    // The four steering forces, one SIMD lane each; a zeroed
                    // `k` lane is the scalar version's "skip this force".
                    let cohere = sum_cohere / cnt_align.max(1.0) - p;
                    // The mouse attracts from afar and repels up close.
                    let mouse_diff = mouse.map_or(Vec2::ZERO, |m| m - p);
                    let mouse_k = match mouse {
                        None => 0.0,
                        Some(_) if mouse_diff.length_squared() < MOUSE_NEAR * MOUSE_NEAR => {
                            MOUSE_REPEL_K
                        }
                        Some(_) => MOUSE_ATTRACT_K,
                    };
                    let (has_avoid, has_align) =
                        ((cnt_avoid > 0.0) as u32 as f32, (cnt_align > 0.0) as u32 as f32);
                    let acc = steering_forces(
                        Vec4::new(sum_separate.x, sum_align.x, cohere.x, mouse_diff.x),
                        Vec4::new(sum_separate.y, sum_align.y, cohere.y, mouse_diff.y),
                        Vec4::new(
                            separation * has_avoid,
                            alignment * has_align,
                            cohesion * has_align,
                            mouse_k,
                        ),
                        v,
                        max_speed,
                    );

                    // Integrate, wrap around the screen edges (toroidal
                    // world), publish. One conditional wrap equals
                    // `rem_euclid` here — a boid moves a few px per frame,
                    // never a full screen — without the per-axis division.
                    let new_v = (v + acc * dt).clamp_length_max(max_speed);
                    let np = p + new_v * dt;
                    let mut pos = np;
                    if pos.x < -half.x {
                        pos.x += size.x;
                    } else if pos.x >= half.x {
                        pos.x -= size.x;
                    }
                    if pos.y < -half.y {
                        pos.y += size.y;
                    } else if pos.y >= half.y {
                        pos.y -= size.y;
                    }
                    *state = BoidState { pos, vel: new_v };
                    *inst = BoidInstance {
                        pos,
                        // Normalized velocity = (cos, sin) of the heading;
                        // keep the previous heading at a standstill.
                        rot: if new_v != Vec2::ZERO {
                            new_v.normalize_or(Vec2::X)
                        } else {
                            inst.rot
                        },
                    };
                }
            });
        }
    });

    // Diagnostics for the CPU path: sort vs steer split, once a second.
    let t2 = std::time::Instant::now();
    timing.0 += (_t1 - _t0).as_secs_f64() * 1000.0;
    timing.1 += (t2 - _t1).as_secs_f64() * 1000.0;
    timing.2 += 1;
    if timing.2 >= 120 {
        eprintln!(
            "build: {:.3}ms  steer: {:.3}ms",
            timing.0 / timing.2 as f64,
            timing.1 / timing.2 as f64
        );
        *timing = (0.0, 0.0, 0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The derived chunk sizes exist for portability, not to retune this
    /// machine: on the 6-thread pool the kernel was certified on, they must
    /// reproduce the hand-found 16384/4096 at the certified 320k.
    #[test]
    fn chunk_sizes_reproduce_certified_tuning() {
        assert_eq!(chunk_sizes(320_000, 6), (16_384, 4_096));
    }

    /// The floor keeps tiny flocks from drowning in task-spawn overhead, and
    /// many-core machines must get *more* chunks per worker count, not the
    /// 6-thread machine's absolute sizes.
    #[test]
    fn chunk_sizes_scale_with_pool() {
        let (_, steer_small) = chunk_sizes(50, 6);
        assert_eq!(steer_small, 1024, "small flocks hit the floor");
        let (_, steer_many) = chunk_sizes(320_000, 48);
        assert!(
            steer_many < 4096,
            "a 48-thread pool should get smaller chunks than the 6-thread \
             pool, got {steer_many}"
        );
        // tasks-per-worker stays in the same band as the certified machine.
        let tasks_per_worker = 320_000usize.div_ceil(steer_many) / 48;
        assert!((6..=26).contains(&tasks_per_worker), "{tasks_per_worker}");
    }
}
