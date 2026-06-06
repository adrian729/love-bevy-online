//! UI pieces shared by the menu and every experiment — the Bevy port of
//! what `main.lua` owned in the original: the SUIT theme, the themed
//! widgets (button, slider, checkbox), the HUD (score / fps / hint), the
//! options-popup shell with its nav buttons, the [Esc]/[O] pause toggle,
//! and the pointer-over-UI tracking the simulations read.
//!
//! Experiments contribute only their own content: a score string, popup
//! instructions and tunable widgets, and (if they declare one, like flock)
//! an on-screen control panel.

use bevy::diagnostic::{DiagnosticsStore, FrameTimeDiagnosticsPlugin};
use bevy::ecs::relationship::RelatedSpawnerCommands;
use bevy::prelude::*;
use bevy::ui::{FocusPolicy, UiGlobalTransform};
use bevy::window::PrimaryWindow;

use crate::app::{AppState, PointerOverUi, RestartRequested, VsyncEnabled};

pub type ChildSpawner<'w> = RelatedSpawnerCommands<'w, ChildOf>;

// SUIT theme from the original's main.lua: translucent backgrounds so the
// scene — including the menu's live backdrop — shows through.
pub const COLOR_NORMAL: Color = Color::srgba(0.25, 0.25, 0.25, 0.55);
pub const COLOR_HOVERED: Color = Color::srgba(0.19, 0.60, 0.73, 0.70);
pub const COLOR_ACTIVE: Color = Color::srgba(1.0, 0.60, 0.0, 0.85);
pub const COLOR_FILL: Color = Color::srgba(0.19, 0.60, 0.73, 0.90);
pub const COLOR_PANEL: Color = Color::srgba(0.0, 0.0, 0.0, 0.45);
pub const COLOR_TEXT_DIM: Color = Color::srgba(1.0, 1.0, 1.0, 0.75);

pub fn plugin(app: &mut App) {
    app.add_systems(Startup, spawn_hud)
        .add_systems(OnExit(AppState::Options), despawn_options_popup)
        .add_systems(
            Update,
            (
                button_colors,
                track_pointer_over_ui,
                toggle_vsync_checkbox,
                sync_vsync_checkbox,
                keyboard_options_toggle,
                nav_button_actions,
                slider_feedback,
                sync_hud_visibility,
                update_hud_fps,
            ),
        );
}

/// Spawn a themed button. `marker` is whatever component(s) the caller's
/// click handler looks for.
pub fn spawn_button(
    parent: &mut ChildSpawner,
    marker: impl Bundle,
    label: &str,
    size: Vec2,
    font_size: f32,
) {
    parent
        .spawn((
            Button,
            marker,
            Node {
                width: Val::Px(size.x),
                height: Val::Px(size.y),
                justify_content: JustifyContent::Center,
                align_items: AlignItems::Center,
                border_radius: BorderRadius::all(Val::Px(4.0)),
                ..default()
            },
            BackgroundColor(COLOR_NORMAL),
        ))
        .with_children(|button| {
            button.spawn((
                Text::new(label),
                TextFont::from_font_size(font_size),
                TextColor(Color::WHITE),
            ));
        });
}

// ---------------------------------------------------------------------------
// Checkboxes

/// A SUIT-style checkbox row: a clickable 18px box (hover/active colors come
/// from [`button_colors`] via `Button`) with a label beside it. `box_marker`
/// identifies the clickable box for the caller's toggle system, `mark_marker`
/// the inner check mark for its sync system.
pub fn spawn_checkbox(
    parent: &mut ChildSpawner,
    box_marker: impl Bundle,
    mark_marker: impl Bundle,
    label: &str,
    checked: bool,
    font_size: f32,
) {
    parent
        .spawn(Node {
            flex_direction: FlexDirection::Row,
            align_items: AlignItems::Center,
            column_gap: Val::Px(8.0),
            ..default()
        })
        .with_children(|row| {
            row.spawn((
                box_marker,
                Button,
                Node {
                    width: Val::Px(18.0),
                    height: Val::Px(18.0),
                    justify_content: JustifyContent::Center,
                    align_items: AlignItems::Center,
                    border_radius: BorderRadius::all(Val::Px(4.0)),
                    ..default()
                },
                BackgroundColor(COLOR_NORMAL),
            ))
            .with_children(|checkbox| {
                checkbox.spawn((
                    mark_marker,
                    Node {
                        width: Val::Px(10.0),
                        height: Val::Px(10.0),
                        border_radius: BorderRadius::all(Val::Px(2.0)),
                        ..default()
                    },
                    BackgroundColor(COLOR_FILL),
                    if checked {
                        Visibility::Inherited
                    } else {
                        Visibility::Hidden
                    },
                ));
            });
            row.spawn((
                Text::new(label),
                TextFont::from_font_size(font_size),
                TextColor(Color::WHITE),
            ));
        });
}

/// The clickable box of the VSync checkbox.
#[derive(Component)]
struct VsyncCheckbox;

/// Its inner check mark, shown while vsync is on.
#[derive(Component)]
struct VsyncCheckMark;

/// The VSync checkbox row, bound to [`VsyncEnabled`]. Lives here rather than
/// in an experiment: vsync is a display option every experiment's options
/// popup includes.
pub fn spawn_vsync_checkbox(parent: &mut ChildSpawner, vsync: &VsyncEnabled, font_size: f32) {
    spawn_checkbox(
        parent,
        VsyncCheckbox,
        VsyncCheckMark,
        "VSync (off = uncapped fps)",
        vsync.0,
        font_size,
    );
}

/// A click on the checkbox flips [`VsyncEnabled`]; `apply_vsync` (app.rs)
/// pushes it to the window.
fn toggle_vsync_checkbox(
    boxes: Query<&Interaction, (Changed<Interaction>, With<VsyncCheckbox>)>,
    mut vsync: ResMut<VsyncEnabled>,
) {
    for interaction in &boxes {
        if *interaction == Interaction::Pressed {
            vsync.0 = !vsync.0;
        }
    }
}

/// Keep the check mark in sync with the resource, wherever the change came
/// from.
fn sync_vsync_checkbox(
    vsync: Res<VsyncEnabled>,
    mut marks: Query<&mut Visibility, With<VsyncCheckMark>>,
) {
    if !vsync.is_changed() {
        return;
    }
    for mut visibility in &mut marks {
        *visibility = if vsync.0 {
            Visibility::Inherited
        } else {
            Visibility::Hidden
        };
    }
}

// ---------------------------------------------------------------------------
// Sliders
//
// Hand-rolled: `Interaction::Pressed` persists while the mouse is held, so
// dragging is just "while pressed, map cursor x onto the track". The widget
// is generic over a binding component, so each experiment's tunables (the
// original's `tunableSpecs`) plug into the same mechanics.

/// Binds a slider to one field of an experiment's settings resource.
pub trait SliderBinding: Component + Copy {
    type Settings: Resource;

    fn label(self) -> &'static str;
    fn range(self) -> (f32, f32);
    fn get(self, settings: &Self::Settings) -> f32;
    fn set(self, settings: &mut Self::Settings, value: f32);
    /// Display format, matching the original's `%d` / `%.2f` specs.
    fn format(self, value: f32) -> String;

    /// Normalized 0..1 slider position for the current value. Linear by
    /// default; override for log scales (flock's Count).
    fn t(self, settings: &Self::Settings) -> f32 {
        let (min, max) = self.range();
        (self.get(settings).clamp(min, max) - min) / (max - min)
    }

    /// Inverse of [`Self::t`]: the value for a 0..1 slider position.
    fn value_from_t(self, t: f32) -> f32 {
        let (min, max) = self.range();
        min + t.clamp(0.0, 1.0) * (max - min)
    }
}

/// Register the drag/sync systems for one binding type.
pub fn slider_plugin<B: SliderBinding>(app: &mut App) {
    app.add_systems(Update, (drag_sliders::<B>, sync_slider_visuals::<B>));
}

/// Draggable slider bar; carries its binding component alongside.
#[derive(Component)]
pub struct SliderTrack;

/// Filled portion of a slider.
#[derive(Component)]
pub struct SliderFill;

/// Text showing a parameter's current value.
#[derive(Component)]
pub struct ValueLabel;

/// Label + value readout above a draggable bar, one per tunable. The whole
/// row carries `row_marker` (so callers can show/hide it — the original's
/// `visibleIf`).
pub fn spawn_slider<B: SliderBinding>(
    parent: &mut ChildSpawner,
    row_marker: impl Bundle,
    binding: B,
    settings: &B::Settings,
    font_size: f32,
) {
    parent
        .spawn((
            row_marker,
            Node {
                flex_direction: FlexDirection::Column,
                width: Val::Percent(100.0),
                row_gap: Val::Px(2.0),
                ..default()
            },
        ))
        .with_children(|slider| {
            slider
                .spawn(Node {
                    flex_direction: FlexDirection::Row,
                    justify_content: JustifyContent::SpaceBetween,
                    width: Val::Percent(100.0),
                    ..default()
                })
                .with_children(|labels| {
                    labels.spawn((
                        Text::new(binding.label()),
                        TextFont::from_font_size(font_size),
                        TextColor(Color::WHITE),
                    ));
                    labels.spawn((
                        ValueLabel,
                        binding,
                        Text::new(binding.format(binding.get(settings))),
                        TextFont::from_font_size(font_size),
                        TextColor(COLOR_TEXT_DIM),
                    ));
                });
            slider
                .spawn((
                    SliderTrack,
                    binding,
                    Interaction::default(),
                    FocusPolicy::Block,
                    Node {
                        width: Val::Percent(100.0),
                        height: Val::Px(14.0),
                        border_radius: BorderRadius::all(Val::Px(4.0)),
                        ..default()
                    },
                    BackgroundColor(COLOR_NORMAL),
                ))
                .with_children(|track| {
                    track.spawn((
                        SliderFill,
                        binding,
                        Node {
                            width: Val::Percent(binding.t(settings) * 100.0),
                            height: Val::Percent(100.0),
                            border_radius: BorderRadius::all(Val::Px(4.0)),
                            ..default()
                        },
                        BackgroundColor(COLOR_FILL),
                    ));
                });
        });
}

/// While a track is held (`Pressed` persists during a drag, even off-node),
/// map the cursor's x onto the track to set the bound parameter.
fn drag_sliders<B: SliderBinding>(
    window: Query<&Window, With<PrimaryWindow>>,
    tracks: Query<(&Interaction, &B, &ComputedNode, &UiGlobalTransform), With<SliderTrack>>,
    mut settings: ResMut<B::Settings>,
) {
    let Ok(window) = window.single() else { return };
    let Some(cursor) = window.cursor_position() else {
        return;
    };
    for (interaction, binding, node, transform) in &tracks {
        if *interaction != Interaction::Pressed {
            continue;
        }
        // ComputedNode is in physical pixels; the cursor is logical.
        let scale = node.inverse_scale_factor();
        let center_x = transform.translation.x * scale;
        let width = node.size().x * scale;
        if width <= 0.0 {
            continue;
        }
        let t = ((cursor.x - (center_x - width / 2.0)) / width).clamp(0.0, 1.0);
        binding.set(&mut settings, binding.value_from_t(t));
    }
}

/// Highlight a slider's fill while it is being dragged. Binding-agnostic.
fn slider_feedback(
    tracks: Query<(&Interaction, &Children), (Changed<Interaction>, With<SliderTrack>)>,
    mut fills: Query<&mut BackgroundColor, With<SliderFill>>,
) {
    for (interaction, children) in &tracks {
        for child in children.iter() {
            if let Ok(mut background) = fills.get_mut(child) {
                background.0 = match interaction {
                    Interaction::Pressed => COLOR_ACTIVE,
                    _ => COLOR_FILL,
                };
            }
        }
    }
}

/// Keep every bound widget (fill width, value text) in sync with the
/// settings, wherever the change came from (panel, popup, or reset).
fn sync_slider_visuals<B: SliderBinding>(
    settings: Res<B::Settings>,
    mut fills: Query<(&B, &mut Node), With<SliderFill>>,
    mut labels: Query<(&B, &mut Text), With<ValueLabel>>,
) {
    if !settings.is_changed() {
        return;
    }
    for (binding, mut node) in &mut fills {
        node.width = Val::Percent(binding.t(&settings) * 100.0);
    }
    for (binding, mut text) in &mut labels {
        text.0 = binding.format(binding.get(&settings));
    }
}

// ---------------------------------------------------------------------------
// HUD — the original's `drawHUD`: score top-left, hint bottom-left, plus the
// port's fps readout. Experiments write their own score string (the
// original's per-game `scoreLabel`); everything hides on the menu, where the
// current experiment is only a backdrop.

/// The score line, top-left. The current experiment keeps its text updated.
#[derive(Component)]
pub struct HudScore;

#[derive(Component)]
struct HudFps;

/// Everything the HUD shows during play.
#[derive(Component)]
struct HudItem;

fn spawn_hud(mut commands: Commands) {
    commands.spawn((
        HudItem,
        HudScore,
        Text::new(""),
        TextFont::from_font_size(18.0),
        TextColor(Color::WHITE),
        Node {
            position_type: PositionType::Absolute,
            top: Val::Px(8.0),
            left: Val::Px(12.0),
            ..default()
        },
    ));
    commands.spawn((
        HudItem,
        HudFps,
        Text::new("-- fps"),
        TextFont::from_font_size(12.0),
        TextColor(COLOR_TEXT_DIM),
        Node {
            position_type: PositionType::Absolute,
            top: Val::Px(34.0),
            left: Val::Px(12.0),
            ..default()
        },
    ));
    commands.spawn((
        HudItem,
        Text::new("[R] Restart   [Esc] Options"),
        TextFont::from_font_size(14.0),
        TextColor(COLOR_TEXT_DIM),
        Node {
            position_type: PositionType::Absolute,
            bottom: Val::Px(8.0),
            left: Val::Px(12.0),
            ..default()
        },
    ));
}

/// Hide the HUD on the menu, like the original (no `drawHUD` there).
fn sync_hud_visibility(
    state: Res<State<AppState>>,
    mut hud_items: Query<&mut Visibility, With<HudItem>>,
) {
    if !state.is_changed() {
        return;
    }
    for mut visibility in &mut hud_items {
        *visibility = if *state.get() == AppState::Menu {
            Visibility::Hidden
        } else {
            Visibility::Inherited
        };
    }
}

fn update_hud_fps(diagnostics: Res<DiagnosticsStore>, mut fps: Query<&mut Text, With<HudFps>>) {
    if let Some(value) = diagnostics
        .get(&FrameTimeDiagnosticsPlugin::FPS)
        .and_then(|fps| fps.smoothed())
    {
        let label = format!("{value:.0} fps");
        for mut text in &mut fps {
            if text.0 != label {
                text.0.clone_from(&label);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Options popup — the shell (overlay, title, instructions, vsync checkbox,
// nav buttons) is shared; each experiment spawns it with its own
// instructions and tunable widgets when [Esc] pauses the game.

/// Root of the popup, for symmetric despawn on leaving `Options`.
#[derive(Component)]
pub struct OptionsPopup;

/// The popup's shared nav buttons. `ResetSettings` is handled per
/// experiment (each resets its own settings resource, gated on being
/// current); the rest by [`nav_button_actions`].
#[derive(Component, Clone, Copy)]
pub enum NavAction {
    ResetSettings,
    Resume,
    Restart,
    MainMenu,
}

/// The paused options popup: title, instruction lines, the experiment's
/// own controls (`content`), then the vsync checkbox and the four nav
/// buttons — the original's data-driven `updateOptions`.
pub fn spawn_options_popup(
    commands: &mut Commands,
    vsync: &VsyncEnabled,
    instructions: &[&str],
    content: impl FnOnce(&mut ChildSpawner),
) {
    commands
        .spawn((
            OptionsPopup,
            Interaction::default(),
            FocusPolicy::Block,
            GlobalZIndex(10),
            Node {
                position_type: PositionType::Absolute,
                width: Val::Percent(100.0),
                height: Val::Percent(100.0),
                justify_content: JustifyContent::Center,
                align_items: AlignItems::Center,
                ..default()
            },
            // Dim the scene behind, like the original popup.
            BackgroundColor(Color::srgba(0.0, 0.0, 0.0, 0.45)),
        ))
        .with_children(|overlay| {
            overlay
                .spawn((
                    Node {
                        flex_direction: FlexDirection::Column,
                        align_items: AlignItems::Center,
                        row_gap: Val::Px(10.0),
                        padding: UiRect::all(Val::Px(20.0)),
                        width: Val::Px(480.0),
                        border_radius: BorderRadius::all(Val::Px(6.0)),
                        ..default()
                    },
                    BackgroundColor(Color::srgba(0.10, 0.10, 0.12, 0.95)),
                ))
                .with_children(|popup| {
                    popup.spawn((
                        Text::new("Options"),
                        TextFont::from_font_size(24.0),
                        TextColor(Color::WHITE),
                    ));
                    for line in instructions {
                        popup.spawn((
                            Text::new(*line),
                            TextFont::from_font_size(13.0),
                            TextColor(COLOR_TEXT_DIM),
                        ));
                    }
                    popup
                        .spawn(Node {
                            flex_direction: FlexDirection::Column,
                            row_gap: Val::Px(6.0),
                            width: Val::Percent(100.0),
                            margin: UiRect::vertical(Val::Px(6.0)),
                            ..default()
                        })
                        .with_children(|body| {
                            content(body);
                            spawn_vsync_checkbox(body, vsync, 14.0);
                        });
                    popup
                        .spawn(Node {
                            flex_direction: FlexDirection::Row,
                            column_gap: Val::Px(8.0),
                            ..default()
                        })
                        .with_children(|nav| {
                            // 28 px high, like the original's nav row.
                            let button = |nav: &mut ChildSpawner,
                                          action: NavAction,
                                          label: &str,
                                          width: f32| {
                                spawn_button(nav, action, label, Vec2::new(width, 28.0), 14.0);
                            };
                            button(nav, NavAction::ResetSettings, "Reset settings", 120.0);
                            button(nav, NavAction::Resume, "Resume", 80.0);
                            button(nav, NavAction::Restart, "Restart", 80.0);
                            button(nav, NavAction::MainMenu, "Main Menu", 100.0);
                        });
                });
        });
}

pub fn despawn_options_popup(mut commands: Commands, popups: Query<Entity, With<OptionsPopup>>) {
    for entity in &popups {
        commands.entity(entity).despawn();
    }
}

/// [Esc] / [O] toggles the options popup, pausing the simulation — the
/// original's `love.keypressed`. (The menu handles its own keys: Esc quits
/// there.)
fn keyboard_options_toggle(
    keys: Res<ButtonInput<KeyCode>>,
    state: Res<State<AppState>>,
    mut next: ResMut<NextState<AppState>>,
) {
    if keys.just_pressed(KeyCode::Escape) || keys.just_pressed(KeyCode::KeyO) {
        match state.get() {
            AppState::Playing => next.set(AppState::Options),
            AppState::Options => next.set(AppState::Playing),
            AppState::Menu => {}
        }
    }
}

/// The popup's shared nav actions (`ResetSettings` is per experiment).
fn nav_button_actions(
    buttons: Query<(&Interaction, &NavAction), Changed<Interaction>>,
    mut restart: ResMut<RestartRequested>,
    mut next: ResMut<NextState<AppState>>,
) {
    for (interaction, action) in &buttons {
        if *interaction != Interaction::Pressed {
            continue;
        }
        match action {
            NavAction::Resume => next.set(AppState::Playing),
            NavAction::Restart => {
                restart.0 = true;
                next.set(AppState::Playing);
            }
            NavAction::MainMenu => next.set(AppState::Menu),
            NavAction::ResetSettings => {}
        }
    }
}

/// SUIT-style button feedback: gray / cyan hover / orange active.
fn button_colors(
    mut buttons: Query<(&Interaction, &mut BackgroundColor), (Changed<Interaction>, With<Button>)>,
) {
    for (interaction, mut background) in &mut buttons {
        background.0 = match interaction {
            Interaction::Pressed => COLOR_ACTIVE,
            Interaction::Hovered => COLOR_HOVERED,
            Interaction::None => COLOR_NORMAL,
        };
    }
}

/// The experiments ignore the mouse while it hovers or drags any UI — every
/// interactive widget carries `Interaction`, so "any of them is not None" is
/// exactly the original's `pointerOverUI`. (The fish doesn't read it: the
/// original's fish update ignores the flag and chases the live cursor.)
fn track_pointer_over_ui(
    interactions: Query<&Interaction>,
    mut pointer_over_ui: ResMut<PointerOverUi>,
) {
    let over = interactions
        .iter()
        .any(|interaction| *interaction != Interaction::None);
    if pointer_over_ui.0 != over {
        pointer_over_ui.0 = over;
    }
}
