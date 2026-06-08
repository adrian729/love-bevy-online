//! The forest's own UI: its score line ("Trees"), the top-right on-screen panel
//! (the original declares `onscreenControls = 'all'`), and the options-popup
//! controls. Like flow, the forest has many tunables, so the popup lays them out
//! in two columns; the leaf rows gate on the Leaves checkbox (the `visibleIf`
//! pattern). The "New forest" button reseeds (the original's `randomize`).

use bevy::prelude::*;

use super::settings::{ForestParam, ForestSettings};
use super::sim::Forest;
use crate::app::{AppState, RestartRequested, VsyncEnabled};
use crate::experiments::{CurrentExperiment, ExperimentId, experiment_active};
use crate::ui::{
    COLOR_NORMAL, COLOR_PANEL, ChildSpawner, HudScore, NameLabel, NavAction, Tooltip, ValueEdit,
    ValueLabel, slider_plugin, spawn_button, spawn_checkbox, spawn_options_popup, spawn_slider,
    value_entry_plugin,
};

pub fn plugin(app: &mut App) {
    slider_plugin::<ForestParam>(app);
    // Typed value entry (exact values are shareable, and the affordance carries
    // the hover tooltip) — opt-in, like flow; the other experiments stay
    // drag-only.
    value_entry_plugin::<ForestParam>(app);
    app.insert_resource(ForestPanelOpen(true))
        .add_systems(Startup, spawn_panel)
        .add_systems(
            OnEnter(AppState::Options),
            spawn_popup.run_if(experiment_active(ExperimentId::Forest)),
        )
        .add_systems(
            Update,
            (
                sync_panel_visibility,
                toggle_leaves,
                sync_leaf_check,
                sync_row_visibility,
                make_value_labels_editable,
                cancel_edits_on_screen_change,
                attach_tooltips,
                (new_forest_clicks, reset_settings, update_score)
                    .run_if(experiment_active(ExperimentId::Forest)),
            ),
        );
}

/// The leaf sliders — shown only while Leaves is on (the `visibleIf` pattern).
#[derive(Component, Clone, Copy)]
struct ForestRow;

/// The Leaves checkbox's clickable box.
#[derive(Component)]
struct LeavesCheckbox;

/// Its inner check mark.
#[derive(Component)]
struct LeavesCheckMark;

/// The "New forest" button (the original's `randomize`: a fresh seed + regrow).
#[derive(Component)]
struct NewForestButton;

/// The top-right on-screen controls panel.
#[derive(Component)]
struct ForestPanelRoot;

/// Its collapsible body.
#[derive(Component)]
struct ForestPanelBody;

/// Label of the panel's Hide/Show toggle.
#[derive(Component)]
struct ForestPanelToggleLabel;

/// The panel's Hide/Show toggle button.
#[derive(Component, Clone, Copy)]
struct ForestTogglePanel;

/// Whether the on-screen panel is expanded.
#[derive(Resource)]
struct ForestPanelOpen(bool);

/// One widget cell at `width`; gated rows carry the `ForestRow` marker so a
/// closed Leaves toggle releases their layout space.
fn cell(body: &mut ChildSpawner, width: Val, leaf: bool, spawn: impl FnOnce(&mut ChildSpawner)) {
    let node = Node {
        width,
        flex_direction: FlexDirection::Column,
        ..default()
    };
    if leaf {
        body.spawn((ForestRow, node)).with_children(spawn);
    } else {
        body.spawn(node).with_children(spawn);
    }
}

/// Spawn every forest control into `body` — the original's `tunableSpecs` order
/// (structure, shape, colour), the Leaves checkbox, its gated leaf sliders, then
/// the "New forest" button. `width` sizes each cell (full in the panel, ~half in
/// the two-column popup).
fn spawn_controls(body: &mut ChildSpawner, settings: &ForestSettings, width: Val, font_size: f32) {
    for param in ForestParam::ALWAYS {
        cell(body, width, false, |c| {
            spawn_slider(c, (), param, settings, font_size);
        });
    }
    cell(body, width, false, |c| {
        spawn_checkbox(
            c,
            LeavesCheckbox,
            LeavesCheckMark,
            "Leaves",
            settings.leaves,
            font_size,
        );
    });
    for param in ForestParam::LEAF {
        cell(body, width, true, |c| {
            spawn_slider(c, (), param, settings, font_size);
        });
    }
    cell(body, width, false, |c| {
        spawn_button(c, NewForestButton, "New forest", Vec2::new(110.0, 24.0), font_size);
    });
}

/// On-screen controls, top-right: a Hide/Show toggle + every tunable (the
/// original's `onscreenControls = 'all'`), gated leaf rows included.
fn spawn_panel(mut commands: Commands, settings: Res<ForestSettings>) {
    commands
        .spawn((
            ForestPanelRoot,
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
            panel
                .spawn((
                    Button,
                    ForestTogglePanel,
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
                        ForestPanelToggleLabel,
                        Text::new("Hide"),
                        TextFont::from_font_size(12.0),
                        TextColor(Color::WHITE),
                    ));
                });

            panel
                .spawn((
                    ForestPanelBody,
                    Node {
                        flex_direction: FlexDirection::Column,
                        row_gap: Val::Px(4.0),
                        width: Val::Px(200.0),
                        ..default()
                    },
                ))
                .with_children(|body| {
                    spawn_controls(body, &settings, Val::Percent(100.0), 12.0);
                });
        });
}

/// The forest's options-popup content: instructions + every control in a
/// two-column wrap (the shared popup shell is untouched — the layout lives here).
fn spawn_popup(mut commands: Commands, settings: Res<ForestSettings>, vsync: Res<VsyncEnabled>) {
    spawn_options_popup(
        &mut commands,
        &vsync,
        &[
            "A forest of procedural L-system trees, grown by randomised branching.",
            "Tune the branching, shape and colour; toggle Leaves and Wind.",
            "Press [New forest] or [R] to grow a brand-new one.",
        ],
        |body: &mut ChildSpawner| {
            body.spawn(Node {
                flex_direction: FlexDirection::Row,
                flex_wrap: FlexWrap::Wrap,
                justify_content: JustifyContent::SpaceBetween,
                row_gap: Val::Px(6.0),
                width: Val::Percent(100.0),
                ..default()
            })
            .with_children(|grid| {
                spawn_controls(grid, &settings, Val::Percent(47.0), 13.0);
            });
        },
    );
}

/// Clicks on the Leaves checkbox flip the setting.
fn toggle_leaves(
    boxes: Query<&Interaction, (Changed<Interaction>, With<LeavesCheckbox>)>,
    mut settings: ResMut<ForestSettings>,
) {
    for interaction in &boxes {
        if *interaction == Interaction::Pressed {
            settings.leaves = !settings.leaves;
        }
    }
}

/// Keep the Leaves check mark in sync (and right after a panel/popup spawns).
fn sync_leaf_check(
    settings: Res<ForestSettings>,
    added: Query<(), Added<LeavesCheckMark>>,
    mut marks: Query<&mut Visibility, With<LeavesCheckMark>>,
) {
    if !settings.is_changed() && added.is_empty() {
        return;
    }
    for mut visibility in &mut marks {
        *visibility = if settings.leaves {
            Visibility::Inherited
        } else {
            Visibility::Hidden
        };
    }
}

/// Show/hide the leaf rows as Leaves toggles — `Display` toggling so a hidden
/// row releases its layout space (the fish/flow pattern).
fn sync_row_visibility(
    settings: Res<ForestSettings>,
    added: Query<(), Added<ForestRow>>,
    mut rows: Query<&mut Node, With<ForestRow>>,
) {
    if !settings.is_changed() && added.is_empty() {
        return;
    }
    for mut node in &mut rows {
        node.display = if settings.leaves {
            Display::Flex
        } else {
            Display::None
        };
    }
}

/// "New forest": request a reseed (the original's `randomize`). Routed through
/// `RestartRequested` so the sim's `handle_restart` reseeds — it runs in every
/// screen, so this works from the panel (Playing) and the popup (Options) alike.
fn new_forest_clicks(
    buttons: Query<&Interaction, (Changed<Interaction>, With<NewForestButton>)>,
    mut request: ResMut<RestartRequested>,
) {
    for interaction in &buttons {
        if *interaction == Interaction::Pressed {
            request.0 = true;
        }
    }
}

/// The shared popup's "Reset settings", restoring the authored defaults while
/// the forest is current.
fn reset_settings(
    nav: Query<(&Interaction, &NavAction), Changed<Interaction>>,
    mut settings: ResMut<ForestSettings>,
) {
    for (interaction, action) in &nav {
        if *interaction == Interaction::Pressed && matches!(action, NavAction::ResetSettings) {
            *settings = ForestSettings::default();
        }
    }
}

/// Hide/Show toggle clicks; the panel shows only while the forest is being
/// played (the original draws on-screen controls during play only).
fn sync_panel_visibility(
    toggles: Query<&Interaction, (Changed<Interaction>, With<ForestTogglePanel>)>,
    mut panel_open: ResMut<ForestPanelOpen>,
    state: Res<State<AppState>>,
    current: Res<CurrentExperiment>,
    mut bodies: Query<&mut Node, (With<ForestPanelBody>, Without<ForestRow>)>,
    mut toggle_labels: Query<&mut Text, With<ForestPanelToggleLabel>>,
    mut roots: Query<&mut Visibility, With<ForestPanelRoot>>,
) {
    for interaction in &toggles {
        if *interaction == Interaction::Pressed {
            panel_open.0 = !panel_open.0;
        }
    }
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
    let shown = *state.get() == AppState::Playing && current.0 == ExperimentId::Forest;
    for mut visibility in &mut roots {
        *visibility = if shown {
            Visibility::Inherited
        } else {
            Visibility::Hidden
        };
    }
}

/// Make the forest's slider value labels clickable (the shared widget spawns
/// them inert): an `Interaction` opts a label into the shared typed-entry
/// systems and the hover affordance.
fn make_value_labels_editable(
    mut commands: Commands,
    labels: Query<Entity, (With<ValueLabel>, With<ForestParam>, Added<ValueLabel>)>,
) {
    for label in &labels {
        commands.entity(label).insert(Interaction::default());
    }
}

/// Attach hover help to every forest control (the shared tooltip layer is
/// opt-in). Name labels get an `Interaction` to become hoverable; the checkbox
/// and button already have one.
fn attach_tooltips(
    mut commands: Commands,
    param_names: Query<(Entity, &ForestParam), Added<NameLabel>>,
    leaves: Query<Entity, Added<LeavesCheckbox>>,
    new_forest: Query<Entity, Added<NewForestButton>>,
    values: Query<Entity, (With<ForestParam>, Added<ValueLabel>)>,
) {
    for (label, param) in &param_names {
        commands
            .entity(label)
            .insert((Tooltip(param.tip()), Interaction::default()));
    }
    for checkbox in &leaves {
        commands
            .entity(checkbox)
            .insert(Tooltip("Sprout soft leaves at the twig tips."));
    }
    for button in &new_forest {
        commands
            .entity(button)
            .insert(Tooltip("Grow a brand-new forest from a fresh seed. ([R] does the same.)"));
    }
    for label in &values {
        commands.entity(label).insert(Tooltip(
            "Click to type an exact value - [Enter] applies, [Esc] cancels.",
        ));
    }
}

/// Cancel any open typed edit the moment the screen changes — an edit must never
/// outlive the forest being front and center, so the global [R]/[Esc]/[O]
/// bindings only see forest's edits while forest owns the keyboard (the flow
/// pattern).
fn cancel_edits_on_screen_change(
    mut commands: Commands,
    state: Res<State<AppState>>,
    current: Res<CurrentExperiment>,
    edits: Query<Entity, (With<ValueEdit>, With<ForestParam>)>,
) {
    if !state.is_changed() && !current.is_changed() {
        return;
    }
    for edit in &edits {
        commands.entity(edit).remove::<ValueEdit>();
    }
}

/// The score line — the original's `scoreLabel = 'Trees'`, plus the total built
/// segment count (perf-useful: it's what the draw cost scales with).
fn update_score(forest: Res<Forest>, mut score: Query<&mut Text, With<HudScore>>) {
    let label = format!(
        "Trees: {}   {} segments",
        forest.tree_count(),
        forest.total_segments
    );
    for mut text in &mut score {
        if text.0 != label {
            text.0.clone_from(&label);
        }
    }
}
