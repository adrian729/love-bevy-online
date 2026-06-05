//! Native Bevy UI: HUD, the top-right live-tuning panel, and the [Esc]
//! options popup. Recreates the SUIT-based UI of the original — same
//! tunables, same theme colors — with retained-mode widgets. Sliders are
//! hand-rolled: `Interaction::Pressed` persists while the mouse is held, so
//! dragging is just "while pressed, map cursor x onto the track".

use bevy::diagnostic::{DiagnosticsStore, FrameTimeDiagnosticsPlugin};
use bevy::ecs::relationship::RelatedSpawnerCommands;
use bevy::prelude::*;
use bevy::ui::{FocusPolicy, UiGlobalTransform};
use bevy::window::PrimaryWindow;

use crate::AppState;
use crate::boids::{Boid, PointerOverUi, RestartRequested};
use crate::settings::{Param, SimSettings};

type ChildSpawner<'w> = RelatedSpawnerCommands<'w, ChildOf>;

// SUIT theme from the original's main.lua.
const COLOR_NORMAL: Color = Color::srgba(0.25, 0.25, 0.25, 0.55);
const COLOR_HOVERED: Color = Color::srgba(0.19, 0.60, 0.73, 0.70);
const COLOR_ACTIVE: Color = Color::srgba(1.0, 0.60, 0.0, 0.85);
const COLOR_FILL: Color = Color::srgba(0.19, 0.60, 0.73, 0.90);
const COLOR_PANEL: Color = Color::srgba(0.0, 0.0, 0.0, 0.45);
const COLOR_TEXT_DIM: Color = Color::srgba(1.0, 1.0, 1.0, 0.75);

pub fn plugin(app: &mut App) {
    app.add_plugins(FrameTimeDiagnosticsPlugin::default())
        .insert_resource(PanelOpen(true))
        .add_systems(Startup, spawn_hud)
        .add_systems(OnEnter(AppState::Options), spawn_options_popup)
        .add_systems(OnExit(AppState::Options), despawn_options_popup)
        .add_systems(
            Update,
            (
                keyboard_input,
                button_actions,
                button_colors,
                drag_sliders,
                slider_feedback,
                sync_slider_visuals,
                sync_panel_visibility,
                update_hud,
                track_pointer_over_ui,
            ),
        );
}

#[derive(Component)]
struct HudScore;

#[derive(Component)]
struct HudFps;

/// The top-right on-screen controls panel.
#[derive(Component)]
struct PanelRoot;

/// The collapsible part of the panel (the slider rows).
#[derive(Component)]
struct PanelBody;

/// Label of the Hide/Show toggle button.
#[derive(Component)]
struct PanelToggleLabel;

#[derive(Component)]
struct OptionsPopup;

/// Draggable slider bar; carries a [`Param`] binding alongside.
#[derive(Component)]
struct SliderTrack;

/// Filled portion of a slider.
#[derive(Component)]
struct SliderFill;

/// Text showing a parameter's current value.
#[derive(Component)]
struct ValueLabel;

#[derive(Component, Clone, Copy)]
enum ButtonAction {
    TogglePanel,
    ResetSettings,
    Resume,
    Restart,
}

/// Whether the on-screen panel is expanded ("Hide"/"Show" in the original).
#[derive(Resource)]
struct PanelOpen(bool);

// ---------------------------------------------------------------------------
// Spawning

fn spawn_hud(mut commands: Commands, settings: Res<SimSettings>) {
    // Score, top-left — like the original's HUD.
    commands.spawn((
        HudScore,
        Text::new("Boids: 0"),
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

    // On-screen controls, top-right: Hide/Show toggle + the five sliders.
    commands
        .spawn((
            PanelRoot,
            Interaction::default(),
            Node {
                position_type: PositionType::Absolute,
                top: Val::Px(12.0),
                right: Val::Px(12.0),
                flex_direction: FlexDirection::Column,
                align_items: AlignItems::FlexEnd,
                row_gap: Val::Px(6.0),
                padding: UiRect::all(Val::Px(8.0)),
                border_radius: BorderRadius::all(Val::Px(6.0)),
                ..default()
            },
            BackgroundColor(COLOR_PANEL),
        ))
        .with_children(|panel| {
            // 52x20 toggle, like the original's TOGGLE_W/TOGGLE_H.
            panel
                .spawn((
                    Button,
                    ButtonAction::TogglePanel,
                    Node {
                        width: Val::Px(52.0),
                        height: Val::Px(20.0),
                        justify_content: JustifyContent::Center,
                        align_items: AlignItems::Center,
                        border_radius: BorderRadius::all(Val::Px(4.0)),
                        ..default()
                    },
                    BackgroundColor(COLOR_NORMAL),
                ))
                .with_children(|button| {
                    button.spawn((
                        PanelToggleLabel,
                        Text::new("Hide"),
                        TextFont::from_font_size(12.0),
                        TextColor(Color::WHITE),
                    ));
                });

            panel
                .spawn((
                    PanelBody,
                    Node {
                        flex_direction: FlexDirection::Column,
                        row_gap: Val::Px(4.0),
                        width: Val::Px(200.0),
                        ..default()
                    },
                ))
                .with_children(|body| {
                    for param in Param::ALL {
                        spawn_slider(body, param, &settings, 12.0);
                    }
                });
        });
}

/// Label + value readout above a draggable bar, one per tunable.
fn spawn_slider(parent: &mut ChildSpawner, param: Param, settings: &SimSettings, font_size: f32) {
    parent
        .spawn(Node {
            flex_direction: FlexDirection::Column,
            width: Val::Percent(100.0),
            row_gap: Val::Px(2.0),
            ..default()
        })
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
                        Text::new(param.label()),
                        TextFont::from_font_size(font_size),
                        TextColor(Color::WHITE),
                    ));
                    labels.spawn((
                        ValueLabel,
                        param,
                        Text::new(param.format(param.get(settings))),
                        TextFont::from_font_size(font_size),
                        TextColor(COLOR_TEXT_DIM),
                    ));
                });
            slider
                .spawn((
                    SliderTrack,
                    param,
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
                        param,
                        Node {
                            width: Val::Percent(param.t(settings) * 100.0),
                            height: Val::Percent(100.0),
                            border_radius: BorderRadius::all(Val::Px(4.0)),
                            ..default()
                        },
                        BackgroundColor(COLOR_FILL),
                    ));
                });
        });
}

fn spawn_button(parent: &mut ChildSpawner, action: ButtonAction, label: &str, width: f32) {
    parent
        .spawn((
            Button,
            action,
            Node {
                width: Val::Px(width),
                height: Val::Px(28.0),
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
                TextFont::from_font_size(14.0),
                TextColor(Color::WHITE),
            ));
        });
}

/// The paused options popup: instructions, the same five sliders, nav buttons.
fn spawn_options_popup(mut commands: Commands, settings: Res<SimSettings>) {
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
                        width: Val::Px(420.0),
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
                    for line in [
                        // Plain hyphen: Bevy's default font has no em-dash glyph.
                        "Move your mouse - the flock follows from afar and scatters up close.",
                        "Tune separation / alignment / cohesion below.",
                    ] {
                        popup.spawn((
                            Text::new(line),
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
                            for param in Param::ALL {
                                spawn_slider(body, param, &settings, 14.0);
                            }
                        });
                    popup
                        .spawn(Node {
                            flex_direction: FlexDirection::Row,
                            column_gap: Val::Px(8.0),
                            ..default()
                        })
                        .with_children(|nav| {
                            spawn_button(nav, ButtonAction::ResetSettings, "Reset settings", 130.0);
                            spawn_button(nav, ButtonAction::Resume, "Resume", 90.0);
                            spawn_button(nav, ButtonAction::Restart, "Restart", 90.0);
                        });
                });
        });
}

fn despawn_options_popup(mut commands: Commands, popups: Query<Entity, With<OptionsPopup>>) {
    for entity in &popups {
        commands.entity(entity).despawn();
    }
}

// ---------------------------------------------------------------------------
// Interaction

/// [Esc] / [O] toggles the options popup, pausing the simulation.
fn keyboard_input(
    keys: Res<ButtonInput<KeyCode>>,
    state: Res<State<AppState>>,
    mut next: ResMut<NextState<AppState>>,
) {
    if keys.just_pressed(KeyCode::Escape) || keys.just_pressed(KeyCode::KeyO) {
        match state.get() {
            AppState::Playing => next.set(AppState::Options),
            AppState::Options => next.set(AppState::Playing),
        }
    }
}

fn button_actions(
    buttons: Query<(&Interaction, &ButtonAction), Changed<Interaction>>,
    mut settings: ResMut<SimSettings>,
    mut panel_open: ResMut<PanelOpen>,
    mut restart: ResMut<RestartRequested>,
    mut next: ResMut<NextState<AppState>>,
) {
    for (interaction, action) in &buttons {
        if *interaction != Interaction::Pressed {
            continue;
        }
        match action {
            ButtonAction::TogglePanel => panel_open.0 = !panel_open.0,
            ButtonAction::ResetSettings => *settings = SimSettings::default(),
            ButtonAction::Resume => next.set(AppState::Playing),
            ButtonAction::Restart => {
                restart.0 = true;
                next.set(AppState::Playing);
            }
        }
    }
}

/// SUIT-style button feedback: gray / cyan hover / orange active.
fn button_colors(
    mut buttons: Query<
        (&Interaction, &mut BackgroundColor),
        (Changed<Interaction>, With<ButtonAction>),
    >,
) {
    for (interaction, mut background) in &mut buttons {
        background.0 = match interaction {
            Interaction::Pressed => COLOR_ACTIVE,
            Interaction::Hovered => COLOR_HOVERED,
            Interaction::None => COLOR_NORMAL,
        };
    }
}

/// While a track is held (`Pressed` persists during a drag, even off-node),
/// map the cursor's x onto the track to set the bound parameter.
fn drag_sliders(
    window: Query<&Window, With<PrimaryWindow>>,
    tracks: Query<(&Interaction, &Param, &ComputedNode, &UiGlobalTransform), With<SliderTrack>>,
    mut settings: ResMut<SimSettings>,
) {
    let Ok(window) = window.single() else { return };
    let Some(cursor) = window.cursor_position() else {
        return;
    };
    for (interaction, param, node, transform) in &tracks {
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
        param.set(&mut settings, param.value_from_t(t));
    }
}

/// Highlight a slider's fill while it is being dragged.
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
fn sync_slider_visuals(
    settings: Res<SimSettings>,
    mut fills: Query<(&Param, &mut Node), With<SliderFill>>,
    mut labels: Query<(&Param, &mut Text), With<ValueLabel>>,
) {
    if !settings.is_changed() {
        return;
    }
    for (param, mut node) in &mut fills {
        node.width = Val::Percent(param.t(&settings) * 100.0);
    }
    for (param, mut text) in &mut labels {
        text.0 = param.format(param.get(&settings));
    }
}

/// Collapse/expand the panel body and hide the whole panel while paused
/// (the original only draws the on-screen controls during play).
fn sync_panel_visibility(
    panel_open: Res<PanelOpen>,
    state: Res<State<AppState>>,
    mut bodies: Query<&mut Node, With<PanelBody>>,
    mut toggle_labels: Query<&mut Text, With<PanelToggleLabel>>,
    mut roots: Query<&mut Visibility, With<PanelRoot>>,
) {
    if !panel_open.is_changed() && !state.is_changed() {
        return;
    }
    for mut node in &mut bodies {
        node.display = if panel_open.0 {
            Display::Flex
        } else {
            Display::None
        };
    }
    for mut text in &mut toggle_labels {
        text.0 = if panel_open.0 { "Hide" } else { "Show" }.into();
    }
    for mut visibility in &mut roots {
        *visibility = if *state.get() == AppState::Playing {
            Visibility::Inherited
        } else {
            Visibility::Hidden
        };
    }
}

fn update_hud(
    boids: Query<(), With<Boid>>,
    diagnostics: Res<DiagnosticsStore>,
    mut score: Query<&mut Text, (With<HudScore>, Without<HudFps>)>,
    mut fps: Query<&mut Text, With<HudFps>>,
) {
    // `len()` is O(1) for archetype-filtered queries, unlike `count()`.
    let label = format!("Boids: {}", boids.iter().len());
    for mut text in &mut score {
        if text.0 != label {
            text.0.clone_from(&label);
        }
    }
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

/// The flock ignores the mouse while it hovers or drags any UI — the panel
/// root and every interactive widget carry `Interaction`, so "any of them is
/// not None" is exactly the original's `pointerOverUI`.
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
