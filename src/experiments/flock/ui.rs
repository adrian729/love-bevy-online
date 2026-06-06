//! The flock's own UI content: the top-right live-tuning panel (the
//! original flock declares `onscreenControls`), its options-popup content
//! (the same five sliders), and its score line. The chrome itself — HUD,
//! popup shell, nav buttons, slider mechanics — is shared (src/ui.rs).

use bevy::prelude::*;

use super::settings::{Param, SimSettings};
use super::sim::Flock;
use crate::app::{AppState, VsyncEnabled};
use crate::experiments::{CurrentExperiment, ExperimentId, experiment_active};
use crate::ui::{
    COLOR_NORMAL, COLOR_PANEL, ChildSpawner, HudScore, NavAction, slider_plugin,
    spawn_options_popup, spawn_slider,
};

pub fn plugin(app: &mut App) {
    slider_plugin::<Param>(app);
    app.insert_resource(PanelOpen(true))
        .add_systems(Startup, spawn_panel)
        .add_systems(
            OnEnter(AppState::Options),
            spawn_popup.run_if(experiment_active(ExperimentId::Flock)),
        )
        .add_systems(
            Update,
            (
                sync_panel_visibility,
                (button_actions, update_score).run_if(experiment_active(ExperimentId::Flock)),
            ),
        );
}

/// The top-right on-screen controls panel.
#[derive(Component)]
struct PanelRoot;

/// The collapsible part of the panel (the slider rows).
#[derive(Component)]
struct PanelBody;

/// Label of the Hide/Show toggle button.
#[derive(Component)]
struct PanelToggleLabel;

/// The panel's Hide/Show toggle button.
#[derive(Component, Clone, Copy)]
struct TogglePanel;

/// Whether the on-screen panel is expanded ("Hide"/"Show" in the original).
#[derive(Resource)]
struct PanelOpen(bool);

/// On-screen controls, top-right: Hide/Show toggle + the five sliders.
fn spawn_panel(mut commands: Commands, settings: Res<SimSettings>) {
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
                    TogglePanel,
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
                        spawn_slider(body, (), param, &settings, 12.0);
                    }
                });
        });
}

/// The flock's options-popup content: instructions + the five sliders.
fn spawn_popup(mut commands: Commands, settings: Res<SimSettings>, vsync: Res<VsyncEnabled>) {
    spawn_options_popup(
        &mut commands,
        &vsync,
        &[
            // Plain hyphen: Bevy's default font has no em-dash glyph.
            "Move your mouse - the flock follows from afar and scatters up close.",
            "Tune separation / alignment / cohesion below.",
        ],
        |body: &mut ChildSpawner| {
            for param in Param::ALL {
                spawn_slider(body, (), param, &settings, 14.0);
            }
        },
    );
}

fn button_actions(
    toggles: Query<&Interaction, (Changed<Interaction>, With<TogglePanel>)>,
    nav: Query<(&Interaction, &NavAction), Changed<Interaction>>,
    mut settings: ResMut<SimSettings>,
    mut panel_open: ResMut<PanelOpen>,
) {
    for interaction in &toggles {
        if *interaction == Interaction::Pressed {
            panel_open.0 = !panel_open.0;
        }
    }
    // The shared popup's "Reset settings": each experiment restores its own
    // defaults while it is the current one (this system is gated on that).
    for (interaction, action) in &nav {
        if *interaction == Interaction::Pressed && matches!(action, NavAction::ResetSettings) {
            *settings = SimSettings::default();
        }
    }
}

/// Collapse/expand the panel body, and show the whole panel only while the
/// flock is actually being played (the original only draws the on-screen
/// controls during play; other experiments have no panel at all).
fn sync_panel_visibility(
    panel_open: Res<PanelOpen>,
    state: Res<State<AppState>>,
    current: Res<CurrentExperiment>,
    mut bodies: Query<&mut Node, With<PanelBody>>,
    mut toggle_labels: Query<&mut Text, With<PanelToggleLabel>>,
    mut roots: Query<&mut Visibility, With<PanelRoot>>,
) {
    if !panel_open.is_changed() && !state.is_changed() && !current.is_changed() {
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
    let shown = *state.get() == AppState::Playing && current.0 == ExperimentId::Flock;
    for mut visibility in &mut roots {
        *visibility = if shown {
            Visibility::Inherited
        } else {
            Visibility::Hidden
        };
    }
}

fn update_score(
    flock: Res<Flock>,
    gpu_count: Option<Res<super::gpu_sim::GpuFlockCount>>,
    mut score: Query<&mut Text, With<HudScore>>,
) {
    // GPU sim mode keeps a CPU-side count mirror; the CPU sim owns `Flock`.
    let count = gpu_count.map_or(flock.0.len(), |gpu| gpu.0);
    let label = format!("Boids: {}", count);
    for mut text in &mut score {
        if text.0 != label {
            text.0.clone_from(&label);
        }
    }
}
