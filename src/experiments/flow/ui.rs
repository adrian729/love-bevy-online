//! The flow's own UI content: its score line ("Cells"), the top-right
//! on-screen panel (the original declares `onscreenControls = 'all'`), and
//! the options-popup controls. Flow has by far the most tunables of any
//! experiment, so the popup lays its widgets out in two columns; rows
//! gate on the current view, the original's `visibleIf`.

use bevy::prelude::*;

use super::settings::{FlowCycler, FlowMode, FlowParam, FlowSettings};
use super::sim::{FlowField, SEED_PERIOD};
use crate::app::{AppState, VsyncEnabled};
use crate::experiments::{CurrentExperiment, ExperimentId, experiment_active};
use crate::ui::{
    COLOR_NORMAL, COLOR_PANEL, ChildSpawner, HudScore, NameLabel, NavAction, Tooltip, ValueEdit,
    ValueLabel, cycler_plugin, slider_plugin, spawn_button, spawn_checkbox, spawn_cycler,
    spawn_options_popup, spawn_slider, value_entry_plugin,
};

pub fn plugin(app: &mut App) {
    slider_plugin::<FlowParam>(app);
    // Typed value entry is a flow feature (exact seeds are shareable);
    // the other experiments' sliders stay drag-only.
    value_entry_plugin::<FlowParam>(app);
    cycler_plugin::<FlowCycler>(app);
    app.insert_resource(FlowPanelOpen(true))
        .add_systems(Startup, spawn_panel)
        .add_systems(
            OnEnter(AppState::Options),
            spawn_popup.run_if(experiment_active(ExperimentId::Flow)),
        )
        .add_systems(
            Update,
            (
                sync_panel_visibility,
                toggle_checkboxes,
                sync_check_marks,
                sync_row_visibility,
                make_value_labels_editable,
                cancel_edits_on_screen_change,
                attach_tooltips,
                (new_field_clicks, reset_settings, update_score)
                    .run_if(experiment_active(ExperimentId::Flow)),
            ),
        );
}

/// Which views show which rows — the original's `visibleIf` specs.
#[derive(Component, Clone, Copy)]
enum FlowRow {
    /// Streamline count/length: Streamlines only.
    Stream,
    /// Stroke width/opacity: Streamlines and Arrows.
    Stroke,
    /// The arrowheads checkbox: Arrows only.
    Arrow,
    /// The background checkbox: everywhere except Gradient (which IS the
    /// background).
    Background,
    /// The particle-overlay checkbox: hidden in Particles view (redundant
    /// there).
    Overlay,
    /// Particle count/speed/fade: the overlay or the Particles view.
    Particle,
}

impl FlowRow {
    fn visible(self, settings: &FlowSettings) -> bool {
        match self {
            Self::Stream => settings.mode == FlowMode::Streamlines,
            Self::Stroke => matches!(settings.mode, FlowMode::Streamlines | FlowMode::Arrows),
            Self::Arrow => settings.mode == FlowMode::Arrows,
            Self::Background => settings.mode != FlowMode::Gradient,
            Self::Overlay => settings.mode != FlowMode::Particles,
            Self::Particle => settings.animate || settings.mode == FlowMode::Particles,
        }
    }
}

/// The flow's boolean tunables; the clickable checkbox box carries this.
#[derive(Component, Clone, Copy)]
enum FlowToggle {
    Arrowheads,
    Background,
    Animate,
}

impl FlowToggle {
    fn get(self, settings: &FlowSettings) -> bool {
        match self {
            Self::Arrowheads => settings.arrowheads,
            Self::Background => settings.background,
            Self::Animate => settings.animate,
        }
    }

    fn flip(self, settings: &mut FlowSettings) {
        match self {
            Self::Arrowheads => settings.arrowheads = !settings.arrowheads,
            Self::Background => settings.background = !settings.background,
            Self::Animate => settings.animate = !settings.animate,
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::Arrowheads => "Arrowheads",
            Self::Background => "Gradient bg",
            Self::Animate => "Particle overlay",
        }
    }

    /// Hover help, like `FlowParam::tip`.
    fn tip(self) -> &'static str {
        match self {
            Self::Arrowheads => "Draw a head on each arrow.",
            Self::Background => "A dimmed colour gradient behind the current view.",
            Self::Animate => "Ride animated particles on top of this view.",
        }
    }
}

/// A checkbox's inner mark, tagged with which toggle it shows.
#[derive(Component)]
struct FlowCheckMark(FlowToggle);

/// The "New field" button — the original's `regenerate`: a fresh random
/// seed, same particles.
#[derive(Component)]
struct NewFieldButton;

/// The top-right on-screen controls panel.
#[derive(Component)]
struct FlowPanelRoot;

/// The collapsible part of the panel.
#[derive(Component)]
struct FlowPanelBody;

/// Label of the panel's Hide/Show toggle.
#[derive(Component)]
struct FlowPanelToggleLabel;

/// The panel's Hide/Show toggle button.
#[derive(Component, Clone, Copy)]
struct FlowTogglePanel;

/// Whether the on-screen panel is expanded.
#[derive(Resource)]
struct FlowPanelOpen(bool);

/// One widget cell: every control sits in a wrapper node that carries its
/// `visibleIf` marker (or none) — the popup sizes cells to half rows, the
/// panel to full rows.
fn cell(
    body: &mut ChildSpawner,
    width: Val,
    row: Option<FlowRow>,
    spawn: impl FnOnce(&mut ChildSpawner),
) {
    let node = Node {
        width,
        flex_direction: FlexDirection::Column,
        ..default()
    };
    match row {
        Some(marker) => body.spawn((marker, node)).with_children(spawn),
        None => body.spawn(node).with_children(spawn),
    };
}

/// Spawn every flow control into `body` — the original's full
/// `tunableSpecs` order. `width` sizes each cell (full-width in the
/// panel, ~half in the two-column popup).
fn spawn_controls(body: &mut ChildSpawner, settings: &FlowSettings, width: Val, font_size: f32) {
    for param in FlowParam::FIELD {
        cell(body, width, None, |c| {
            spawn_slider(c, (), param, settings, font_size);
        });
    }
    cell(body, width, None, |c| {
        spawn_cycler(c, (), FlowCycler::Mode, settings, font_size);
    });
    cell(body, width, None, |c| {
        spawn_cycler(c, (), FlowCycler::Palette, settings, font_size);
    });
    for param in FlowParam::STREAM {
        cell(body, width, Some(FlowRow::Stream), |c| {
            spawn_slider(c, (), param, settings, font_size);
        });
    }
    for param in FlowParam::STROKE {
        cell(body, width, Some(FlowRow::Stroke), |c| {
            spawn_slider(c, (), param, settings, font_size);
        });
    }
    let checkbox = |c: &mut ChildSpawner, toggle: FlowToggle| {
        spawn_checkbox(
            c,
            toggle,
            FlowCheckMark(toggle),
            toggle.label(),
            toggle.get(settings),
            font_size,
        );
    };
    cell(body, width, Some(FlowRow::Arrow), |c| {
        checkbox(c, FlowToggle::Arrowheads);
    });
    cell(body, width, Some(FlowRow::Background), |c| {
        checkbox(c, FlowToggle::Background);
    });
    cell(body, width, Some(FlowRow::Overlay), |c| {
        checkbox(c, FlowToggle::Animate);
    });
    for param in FlowParam::PARTICLE {
        cell(body, width, Some(FlowRow::Particle), |c| {
            spawn_slider(c, (), param, settings, font_size);
        });
    }
    cell(body, width, None, |c| {
        spawn_button(c, NewFieldButton, "New field", Vec2::new(110.0, 24.0), font_size);
    });
}

/// On-screen controls, top-right: Hide/Show toggle + every tunable (the
/// original's `onscreenControls = 'all'`), gated rows included.
fn spawn_panel(mut commands: Commands, settings: Res<FlowSettings>) {
    commands
        .spawn((
            FlowPanelRoot,
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
                    FlowTogglePanel,
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
                        FlowPanelToggleLabel,
                        Text::new("Hide"),
                        TextFont::from_font_size(12.0),
                        TextColor(Color::WHITE),
                    ));
                });

            panel
                .spawn((
                    FlowPanelBody,
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

/// The flow's options-popup content: instructions + every control in a
/// two-column wrap (flow has ~20 widgets; one column would overflow the
/// window). The two-column layout lives entirely in this closure — the
/// shared popup shell is untouched.
fn spawn_popup(mut commands: Commands, settings: Res<FlowSettings>, vsync: Res<VsyncEnabled>) {
    spawn_options_popup(
        &mut commands,
        &vsync,
        &[
            "A noise-driven flow field: streamlines, arrows, a colour",
            "gradient, or animated particles.",
            "Tune it live; press [New field] or [R] for a fresh one.",
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

/// Clicks on the three checkboxes flip their settings.
fn toggle_checkboxes(
    boxes: Query<(&Interaction, &FlowToggle), Changed<Interaction>>,
    mut settings: ResMut<FlowSettings>,
) {
    for (interaction, toggle) in &boxes {
        if *interaction == Interaction::Pressed {
            toggle.flip(&mut settings);
        }
    }
}

/// Keep every check mark in sync with its setting — also right after a
/// panel/popup spawns (`Added`).
fn sync_check_marks(
    settings: Res<FlowSettings>,
    added: Query<(), Added<FlowCheckMark>>,
    mut marks: Query<(&FlowCheckMark, &mut Visibility)>,
) {
    if !settings.is_changed() && added.is_empty() {
        return;
    }
    for (mark, mut visibility) in &mut marks {
        *visibility = if mark.0.get(&settings) {
            Visibility::Inherited
        } else {
            Visibility::Hidden
        };
    }
}

/// Show/hide the `visibleIf`-gated rows as the view changes — `Display`
/// toggling so hidden rows release their layout space (the fish pattern).
fn sync_row_visibility(
    settings: Res<FlowSettings>,
    added: Query<(), Added<FlowRow>>,
    mut rows: Query<(&FlowRow, &mut Node)>,
) {
    if !settings.is_changed() && added.is_empty() {
        return;
    }
    for (row, mut node) in &mut rows {
        node.display = if row.visible(&settings) {
            Display::Flex
        } else {
            Display::None
        };
    }
}

/// "New field": a fresh random seed (the original's `regenerate`). The
/// rebuild follows from the settings change; particles keep flowing.
fn new_field_clicks(
    buttons: Query<&Interaction, (Changed<Interaction>, With<NewFieldButton>)>,
    mut settings: ResMut<FlowSettings>,
) {
    for interaction in &buttons {
        if *interaction == Interaction::Pressed {
            settings.seed = rand::Rng::random_range(&mut rand::rng(), 0..SEED_PERIOD as u32) as f32;
        }
    }
}

/// The shared popup's "Reset settings", restoring the authored defaults
/// while flow is current.
fn reset_settings(
    nav: Query<(&Interaction, &NavAction), Changed<Interaction>>,
    mut settings: ResMut<FlowSettings>,
) {
    for (interaction, action) in &nav {
        if *interaction == Interaction::Pressed && matches!(action, NavAction::ResetSettings) {
            *settings = FlowSettings::default();
        }
    }
}

/// Hide/Show toggle clicks, and the panel only shows while flow is being
/// played (the original draws on-screen controls during play only).
fn sync_panel_visibility(
    toggles: Query<&Interaction, (Changed<Interaction>, With<FlowTogglePanel>)>,
    mut panel_open: ResMut<FlowPanelOpen>,
    state: Res<State<AppState>>,
    current: Res<CurrentExperiment>,
    mut bodies: Query<&mut Node, (With<FlowPanelBody>, Without<FlowRow>)>,
    mut toggle_labels: Query<&mut Text, With<FlowPanelToggleLabel>>,
    mut roots: Query<&mut Visibility, With<FlowPanelRoot>>,
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
    let shown = *state.get() == AppState::Playing && current.0 == ExperimentId::Flow;
    for mut visibility in &mut roots {
        *visibility = if shown {
            Visibility::Inherited
        } else {
            Visibility::Hidden
        };
    }
}

/// Make flow's slider value labels clickable (the shared widget spawns
/// them inert): an `Interaction` is what opts a label into the shared
/// typed-entry systems — and into the hover affordance.
fn make_value_labels_editable(
    mut commands: Commands,
    labels: Query<Entity, (With<ValueLabel>, With<FlowParam>, Added<ValueLabel>)>,
) {
    for label in &labels {
        commands.entity(label).insert(Interaction::default());
    }
}

/// Attach hover help to every flow control (the shared tooltip layer is
/// opt-in; flock/fish attach none). Name labels get an `Interaction` to
/// become hoverable; checkboxes and buttons already have one.
fn attach_tooltips(
    mut commands: Commands,
    param_names: Query<(Entity, &FlowParam), Added<NameLabel>>,
    cycler_names: Query<(Entity, &FlowCycler), Added<NameLabel>>,
    toggles: Query<(Entity, &FlowToggle), Added<FlowToggle>>,
    new_field: Query<Entity, Added<NewFieldButton>>,
    values: Query<Entity, (With<FlowParam>, Added<ValueLabel>)>,
) {
    for (label, param) in &param_names {
        commands
            .entity(label)
            .insert((Tooltip(param.tip()), Interaction::default()));
    }
    for (label, cycler) in &cycler_names {
        commands
            .entity(label)
            .insert((Tooltip(cycler.tip()), Interaction::default()));
    }
    for (checkbox, toggle) in &toggles {
        commands.entity(checkbox).insert(Tooltip(toggle.tip()));
    }
    for button in &new_field {
        commands.entity(button).insert(Tooltip(
            "A fresh random seed; the particles keep flowing. \
             ([R] re-seeds and respawns them too.)",
        ));
    }
    for label in &values {
        commands.entity(label).insert(Tooltip(
            "Click to type an exact value — [Enter] applies, [Esc] \
             cancels, Cmd/Ctrl+V pastes.",
        ));
    }
}

/// Cancel any open typed edit the moment the screen changes (pause,
/// resume, menu, another experiment) — an edit must never outlive flow
/// being front and center, so the global key bindings (the flock/fish [R]
/// included) only ever see flow's edits while flow owns the keyboard.
fn cancel_edits_on_screen_change(
    mut commands: Commands,
    state: Res<State<AppState>>,
    current: Res<CurrentExperiment>,
    edits: Query<Entity, (With<ValueEdit>, With<FlowParam>)>,
) {
    if !state.is_changed() && !current.is_changed() {
        return;
    }
    for edit in &edits {
        commands.entity(edit).remove::<ValueEdit>();
    }
}

/// The score line — the original's `scoreLabel = 'Cells'`, rows × cols.
fn update_score(field: Res<FlowField>, mut score: Query<&mut Text, With<HudScore>>) {
    let label = format!("Cells: {}", field.cols * field.rows);
    for mut text in &mut score {
        if text.0 != label {
            text.0.clone_from(&label);
        }
    }
}
