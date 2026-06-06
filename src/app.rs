//! Cross-experiment application shell: the top-level state machine, the
//! camera, the simulation bounds, and the perf-harness plumbing every
//! experiment shares (headless render target, pinned attractor).

use bevy::asset::RenderAssetUsages;
use bevy::camera::RenderTarget;
use bevy::prelude::*;
use bevy::render::render_resource::{Extent3d, TextureDimension, TextureFormat, TextureUsages};
use bevy::render::view::screenshot::{Screenshot, save_to_disk};
use bevy::window::{PresentMode, PrimaryWindow};

/// Top-level game state, mirroring the original's `menu` / `playing` /
/// `options` screens. Perf runs (any CLI args) boot straight into
/// [`Playing`](Self::Playing), so the harness never sees the menu.
#[derive(States, Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum AppState {
    #[default]
    Menu,
    Playing,
    Options,
}

impl AppState {
    /// States in which the simulation steps: while playing, and behind the
    /// menu — the live backdrop, like the original's `menuBg`. Options
    /// pauses.
    pub fn sim_runs(self) -> bool {
        matches!(self, Self::Menu | Self::Playing)
    }
}

/// Run condition for [`AppState::sim_runs`].
pub fn sim_active(state: Res<State<AppState>>) -> bool {
    state.get().sim_runs()
}

/// Whether presentation waits for the display (the options popup's VSync
/// checkbox). Off, the renderer presents as fast as it can — fps readings
/// then show real headroom past the display's refresh rate, exactly like
/// the perf harness (which starts with this off). Main inserts the initial
/// value; [`apply_vsync`] forwards changes to the live window.
#[derive(Resource)]
pub struct VsyncEnabled(pub bool);

/// True while the cursor is busy on the UI — the experiments ignore the
/// mouse then, like `ignore_mouse` in the original.
#[derive(Resource, Default)]
pub struct PointerOverUi(pub bool);

/// Set by the UI (or the R key) to respawn the active experiment.
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

pub fn plugin(app: &mut App) {
    app.init_resource::<PointerOverUi>()
        .init_resource::<RestartRequested>()
        .init_resource::<SimBounds>()
        .add_systems(Startup, setup_camera)
        .add_systems(Update, (update_sim_bounds, headless_snapshots, apply_vsync));
}

/// Forward [`VsyncEnabled`] changes to the window; the surface reconfigures
/// live (the `window_settings` Bevy example's pattern). No-op headless.
fn apply_vsync(vsync: Res<VsyncEnabled>, mut windows: Query<&mut Window, With<PrimaryWindow>>) {
    if !vsync.is_changed() {
        return;
    }
    for mut window in &mut windows {
        window.present_mode = if vsync.0 {
            PresentMode::AutoVsync
        } else {
            PresentMode::AutoNoVsync
        };
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

fn setup_camera(
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
