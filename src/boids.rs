//! The boids simulation: Reynolds separation / alignment / cohesion plus
//! mouse attraction/repulsion, integrated on a toroidal screen. A direct
//! behavioural port of `lib/flock.lua`, restructured as ECS systems.

use std::collections::HashMap;
use std::f32::consts::TAU;

use bevy::asset::RenderAssetUsages;
use bevy::mesh::{Indices, PrimitiveTopology};
use bevy::prelude::*;
use bevy::window::PrimaryWindow;
use rand::Rng;

use crate::AppState;
use crate::settings::{
    MAX_FORCE, MOUSE_ATTRACT_K, MOUSE_NEAR, MOUSE_REPEL_K, NEIGHBOUR_DIST, REF_FPS, SEPARATE_DIST,
    SimSettings,
};

pub fn plugin(app: &mut App) {
    app.init_resource::<PointerOverUi>()
        .init_resource::<RestartRequested>()
        .add_systems(Startup, setup)
        .add_systems(Update, handle_restart)
        .add_systems(
            Update,
            (sync_flock_size, flocking)
                .chain()
                .after(handle_restart)
                .run_if(in_state(AppState::Playing)),
        );
}

/// A single boid.
#[derive(Component)]
pub struct Boid;

/// Boid velocity in px/s (world units per second).
#[derive(Component, Default)]
pub struct Velocity(pub Vec2);

/// True while the cursor is busy on the UI — the flock ignores the mouse
/// then, like `ignore_mouse` in the original.
#[derive(Resource, Default)]
pub struct PointerOverUi(pub bool);

/// Set by the UI (or the R key) to respawn the flock.
#[derive(Resource, Default)]
pub struct RestartRequested(pub bool);

/// Shared mesh/material for every boid: a red dot with a white heading
/// triangle, built once as a single vertex-colored 2D mesh.
#[derive(Resource)]
struct BoidAssets {
    mesh: Handle<Mesh>,
    material: Handle<ColorMaterial>,
}

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<ColorMaterial>>,
) {
    commands.spawn(Camera2d);
    commands.insert_resource(BoidAssets {
        mesh: meshes.add(boid_mesh()),
        material: materials.add(ColorMaterial::from(Color::WHITE)),
    });
}

/// Red body dot (radius 3) + white heading triangle (base half-width 5 at the
/// dot, tip 14 px forward) — matching the original's
/// `circle('fill', x, y, 3)` and `p:perpendicular(v, 5)` / `setMag(14)` draw.
fn boid_mesh() -> Mesh {
    let mut positions: Vec<[f32; 3]> = Vec::new();
    let mut colors: Vec<[f32; 4]> = Vec::new();
    let mut indices: Vec<u32> = Vec::new();

    let red = [1.0, 0.0, 0.0, 1.0];
    let segments = 20u32;
    positions.push([0.0, 0.0, 0.0]);
    colors.push(red);
    for i in 0..=segments {
        let a = i as f32 / segments as f32 * TAU;
        positions.push([3.0 * a.cos(), 3.0 * a.sin(), 0.0]);
        colors.push(red);
    }
    for i in 1..=segments {
        indices.extend([0, i, i + 1]);
    }

    // Slight z offset so the triangle always draws over the dot.
    let white = [1.0, 1.0, 1.0, 1.0];
    let base = positions.len() as u32;
    positions.extend([[0.0, -5.0, 0.1], [14.0, 0.0, 0.1], [0.0, 5.0, 0.1]]);
    colors.extend([white; 3]);
    indices.extend([base, base + 1, base + 2]);

    Mesh::new(PrimitiveTopology::TriangleList, RenderAssetUsages::default())
        .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, positions)
        .with_inserted_attribute(Mesh::ATTRIBUTE_COLOR, colors)
        .with_inserted_indices(Indices::U32(indices))
}

/// Random position on screen, random heading, random speed up to max — the
/// original's `initPositions` / `initVelocities`.
fn spawn_boid(
    commands: &mut Commands,
    assets: &BoidAssets,
    half: Vec2,
    max_speed: f32,
    rng: &mut impl Rng,
) {
    let pos = Vec2::new(
        rng.random_range(-half.x..=half.x),
        rng.random_range(-half.y..=half.y),
    );
    let vel = Vec2::from_angle(rng.random_range(0.0..TAU)) * rng.random_range(0.0..=max_speed);
    commands.spawn((
        Boid,
        Velocity(vel),
        Mesh2d(assets.mesh.clone()),
        MeshMaterial2d(assets.material.clone()),
        Transform::from_translation(pos.extend(0.0))
            .with_rotation(Quat::from_rotation_z(vel.to_angle())),
    ));
}

/// Despawn the whole flock on [R] or when the UI requests a restart;
/// `sync_flock_size` rebuilds it at random positions, which is exactly what
/// the original's `reset()` does.
fn handle_restart(
    mut commands: Commands,
    keys: Res<ButtonInput<KeyCode>>,
    state: Res<State<AppState>>,
    mut request: ResMut<RestartRequested>,
    boids: Query<Entity, With<Boid>>,
) {
    let key_restart = *state.get() == AppState::Playing && keys.just_pressed(KeyCode::KeyR);
    if request.0 || key_restart {
        request.0 = false;
        for entity in &boids {
            commands.entity(entity).despawn();
        }
    }
}

/// Grow or shrink the flock to the tuned count live, like `Flock:setSize`.
fn sync_flock_size(
    mut commands: Commands,
    settings: Res<SimSettings>,
    assets: Res<BoidAssets>,
    window: Query<&Window, With<PrimaryWindow>>,
    boids: Query<Entity, With<Boid>>,
) {
    let Ok(window) = window.single() else { return };
    let half = Vec2::new(window.width(), window.height()) / 2.0;
    let target = settings.count.round().max(1.0) as usize;
    let current = boids.iter().len();
    let mut rng = rand::rng();
    for _ in current..target {
        spawn_boid(&mut commands, &assets, half, settings.speed, &mut rng);
    }
    for entity in boids.iter().skip(target) {
        commands.entity(entity).despawn();
    }
}

/// Reynolds steering, translated from the original's `target_force`:
/// `k * limit(normalize(dir) * max_speed - velocity, MAX_FORCE)` in per-frame
/// units at the 60 fps reference, converted here to px/s².
fn target_force(k: f32, dir: Vec2, vel: Vec2, max_speed: f32) -> Vec2 {
    let steer = dir.normalize_or_zero() * max_speed - vel / REF_FPS;
    k * steer.clamp_length_max(MAX_FORCE) * REF_FPS * REF_FPS
}

fn flocking(
    time: Res<Time>,
    settings: Res<SimSettings>,
    pointer_over_ui: Res<PointerOverUi>,
    window: Query<&Window, With<PrimaryWindow>>,
    camera: Query<(&Camera, &GlobalTransform), With<Camera2d>>,
    mut boids: Query<(Entity, &mut Transform, &mut Velocity), With<Boid>>,
) {
    let dt = time.delta_secs();
    let Ok(window) = window.single() else { return };
    let size = Vec2::new(window.width(), window.height()).max(Vec2::ONE);
    let half = size / 2.0;
    let max_speed = settings.speed;
    let (separation, alignment, cohesion) =
        (settings.separation, settings.alignment, settings.cohesion);

    // Cursor in world space; `None` while outside the window or busy on UI.
    let mouse = if pointer_over_ui.0 {
        None
    } else {
        window.cursor_position().and_then(|screen| {
            let (cam, cam_tf) = camera.single().ok()?;
            cam.viewport_to_world_2d(cam_tf, screen).ok()
        })
    };

    // Snapshot so every boid steers against the same frame, as the original.
    let flock: Vec<(Entity, Vec2, Vec2)> = boids
        .iter()
        .map(|(entity, tf, vel)| (entity, tf.translation.truncate(), vel.0))
        .collect();

    // Spatial hash with cell size = neighbour radius: every neighbour within
    // NEIGHBOUR_DIST lives in the 3x3 cells around a boid (~O(n), not O(n²)).
    let cell = |p: Vec2| (p / NEIGHBOUR_DIST).floor().as_ivec2();
    let mut grid: HashMap<IVec2, Vec<usize>> = HashMap::new();
    for (i, (_, p, _)) in flock.iter().enumerate() {
        grid.entry(cell(*p)).or_default().push(i);
    }

    // Steer + integrate every boid in parallel on the compute task pool; the
    // snapshot and grid are read-only, each boid only writes its own state.
    boids.par_iter_mut().for_each(|(entity, mut tf, mut vel)| {
        let p = tf.translation.truncate();
        let v = vel.0;

        let mut sum_separate = Vec2::ZERO;
        let mut sum_align = Vec2::ZERO;
        let mut sum_cohere = Vec2::ZERO;
        let mut n_align = 0u32;
        let mut n_avoid = 0u32;

        let home = cell(p);
        for dy in -1..=1 {
            for dx in -1..=1 {
                let Some(bucket) = grid.get(&(home + IVec2::new(dx, dy))) else {
                    continue;
                };
                for &j in bucket {
                    let (other, pj, vj) = flock[j];
                    if other == entity {
                        continue;
                    }
                    let d = p.distance(pj);
                    if d > 0.0 && d < NEIGHBOUR_DIST {
                        sum_align += vj;
                        sum_cohere += pj;
                        n_align += 1;
                    }
                    if d > 0.0 && d < SEPARATE_DIST {
                        // Weighted away-vector, falling off with distance.
                        sum_separate += (p - pj).normalize_or_zero() / d;
                        n_avoid += 1;
                    }
                }
            }
        }

        let mut acc = Vec2::ZERO;
        if n_avoid > 0 {
            acc += target_force(separation, sum_separate, v, max_speed);
        }
        if n_align > 0 {
            acc += target_force(alignment, sum_align, v, max_speed);
            let cohere = sum_cohere / n_align as f32 - p;
            acc += target_force(cohesion, cohere, v, max_speed);
        }
        // The mouse attracts from afar and repels up close.
        if let Some(m) = mouse {
            let diff = m - p;
            let k = if diff.length() < MOUSE_NEAR {
                MOUSE_REPEL_K
            } else {
                MOUSE_ATTRACT_K
            };
            acc += target_force(k, diff, v, max_speed);
        }

        // Integrate and wrap around the screen edges (toroidal world).
        let new_v = (v + acc * dt).clamp_length_max(max_speed);
        vel.0 = new_v;
        let np = p + new_v * dt;
        tf.translation.x = (np.x + half.x).rem_euclid(size.x) - half.x;
        tf.translation.y = (np.y + half.y).rem_euclid(size.y) - half.y;
        if new_v != Vec2::ZERO {
            tf.rotation = Quat::from_rotation_z(new_v.to_angle());
        }
    });
}
