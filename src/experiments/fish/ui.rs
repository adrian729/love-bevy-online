//! The fish's own UI content: its score line and its options-popup
//! controls. Unlike the flock, the fish has NO on-screen top-right panel —
//! the original doesn't declare `onscreenControls` for it; its tunables
//! live only in the paused popup.

use bevy::prelude::*;

use super::settings::{FishParam, FishSettings, WaterStyle};
use super::sim::FishGame;
use crate::app::{AppState, VsyncEnabled};
use crate::experiments::{ExperimentId, experiment_active};
use crate::ui::{
    COLOR_NORMAL, ChildSpawner, HudScore, NavAction, slider_plugin, spawn_checkbox,
    spawn_options_popup, spawn_slider,
};

pub fn plugin(app: &mut App) {
    slider_plugin::<FishParam>(app);
    app.add_systems(
        OnEnter(AppState::Options),
        spawn_popup.run_if(experiment_active(ExperimentId::Fish)),
    )
    .add_systems(
        Update,
        (
            toggle_wave_checkbox,
            sync_wave_widgets,
            sync_school_rows,
            toggle_water_checkboxes,
            sync_water_widgets,
            cycle_water_style,
            sync_water_style_label,
            (update_score, reset_settings).run_if(experiment_active(ExperimentId::Fish)),
        ),
    );
}

/// The clickable box of the "Wiggle (sine path)" checkbox.
#[derive(Component)]
struct WaveCheckbox;

/// Its inner check mark.
#[derive(Component)]
struct WaveCheckMark;

/// Rows only visible while the wiggle is on — the original's
/// `visibleIf = 'wave'` on the wave freq/amp sliders.
#[derive(Component)]
struct WaveRow;

/// Rows only visible while more than one fish swims — the school
/// minigame's steering weights, which do nothing to a lone fish.
#[derive(Component)]
struct SchoolRow;

/// The water layer's checkboxes (water.rs) — the component doubles as the
/// click target marker, like flow's FlowToggle.
#[derive(Component, Clone, Copy, PartialEq, Eq)]
enum WaterToggle {
    Water,
    Ripples,
    Bubbles,
}

/// The clickable "Style" button cycling the pond through its looks
/// (Natural / Sketch / Glossy), and the value label inside it.
#[derive(Component)]
struct WaterStyleButton;

#[derive(Component)]
struct WaterStyleLabel;

/// A water checkbox's inner check mark, tagged with whose state it shows.
#[derive(Component, Clone, Copy)]
struct WaterCheckMark(WaterToggle);

/// `visibleIf` for the water cells: the dials and sub-checkboxes show
/// while Water is on; each sub-slider additionally needs its checkbox.
#[derive(Component, Clone, Copy, PartialEq, Eq)]
enum WaterRow {
    Water,
    Ripple,
    Bubble,
}

/// One popup cell: a half-row wrapper carrying the control's `visibleIf`
/// marker — the flow popup's pattern (hiding the wrapper removes the cell
/// from the wrap flow, so rows close up).
fn cell(grid: &mut ChildSpawner, marker: impl Bundle, spawn: impl FnOnce(&mut ChildSpawner)) {
    grid.spawn((
        marker,
        Node {
            width: Val::Percent(47.0),
            flex_direction: FlexDirection::Column,
            ..default()
        },
    ))
    .with_children(spawn);
}

/// The fish's options-popup content: instructions, the game and school
/// sliders, the wiggle checkbox and its sliders, and the water layer's
/// toggles and dials — in a two-column wrap (single-column overflowed the
/// window once the water controls landed).
fn spawn_popup(mut commands: Commands, settings: Res<FishSettings>, vsync: Res<VsyncEnabled>) {
    spawn_options_popup(
        &mut commands,
        &vsync,
        &[
            // Plain hyphen: Bevy's default font has no em-dash glyph.
            "Move your mouse - the fish swims toward it.",
            "Feed it the gold dots and it grows.",
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
                for param in FishParam::MAIN {
                    cell(grid, (), |c| spawn_slider(c, (), param, &settings, 14.0));
                }
                for param in FishParam::SCHOOL {
                    cell(grid, SchoolRow, |c| spawn_slider(c, (), param, &settings, 14.0));
                }
                cell(grid, (), |c| {
                    spawn_checkbox(
                        c,
                        WaveCheckbox,
                        WaveCheckMark,
                        "Wiggle (sine path)",
                        settings.wave,
                        14.0,
                    );
                });
                for param in FishParam::WAVE {
                    cell(grid, WaveRow, |c| spawn_slider(c, (), param, &settings, 14.0));
                }
                // The pond (water.rs): master checkbox, the style switch
                // and dials it reveals, and the ripple/bubble sub-toggles.
                cell(grid, (), |c| {
                    spawn_checkbox(
                        c,
                        WaterToggle::Water,
                        WaterCheckMark(WaterToggle::Water),
                        "Water",
                        settings.water,
                        14.0,
                    );
                });
                cell(grid, WaterRow::Water, |c| {
                    spawn_style_row(c, settings.style);
                });
                for param in FishParam::WATER {
                    cell(grid, WaterRow::Water, |c| {
                        spawn_slider(c, (), param, &settings, 14.0);
                    });
                }
                cell(grid, WaterRow::Water, |c| {
                    spawn_checkbox(
                        c,
                        WaterToggle::Ripples,
                        WaterCheckMark(WaterToggle::Ripples),
                        "Ripples",
                        settings.ripples,
                        14.0,
                    );
                });
                for param in FishParam::RIPPLE {
                    cell(grid, WaterRow::Ripple, |c| {
                        spawn_slider(c, (), param, &settings, 14.0);
                    });
                }
                cell(grid, WaterRow::Water, |c| {
                    spawn_checkbox(
                        c,
                        WaterToggle::Bubbles,
                        WaterCheckMark(WaterToggle::Bubbles),
                        "Bubbles",
                        settings.bubbles,
                        14.0,
                    );
                });
                for param in FishParam::BUBBLE {
                    cell(grid, WaterRow::Bubble, |c| {
                        spawn_slider(c, (), param, &settings, 14.0);
                    });
                }
            });
        },
    );
}

/// The "Style" row: a label and a cycle button showing the current look —
/// clicking advances Natural -> Sketch -> Glossy -> Natural. A button
/// rather than a checkbox: three states, one obvious control.
fn spawn_style_row(parent: &mut ChildSpawner, style: WaterStyle) {
    parent
        .spawn(Node {
            flex_direction: FlexDirection::Row,
            align_items: AlignItems::Center,
            column_gap: Val::Px(8.0),
            ..default()
        })
        .with_children(|row| {
            row.spawn((
                Text::new("Style"),
                TextFont::from_font_size(14.0),
                TextColor(Color::WHITE),
            ));
            row.spawn((
                WaterStyleButton,
                Button,
                Node {
                    width: Val::Px(86.0),
                    height: Val::Px(22.0),
                    justify_content: JustifyContent::Center,
                    align_items: AlignItems::Center,
                    border_radius: BorderRadius::all(Val::Px(4.0)),
                    ..default()
                },
                BackgroundColor(COLOR_NORMAL),
            ))
            .with_children(|button| {
                button.spawn((
                    WaterStyleLabel,
                    Text::new(style.label()),
                    TextFont::from_font_size(13.0),
                    TextColor(Color::WHITE),
                ));
            });
        });
}

/// Clicks on the style button advance the pond to its next look.
fn cycle_water_style(
    buttons: Query<&Interaction, (Changed<Interaction>, With<WaterStyleButton>)>,
    mut settings: ResMut<FishSettings>,
) {
    for interaction in &buttons {
        if *interaction == Interaction::Pressed {
            settings.style = settings.style.next();
        }
    }
}

/// Keep the style button's label naming the current look — also right
/// after the popup spawns, and after Reset settings.
fn sync_water_style_label(
    settings: Res<FishSettings>,
    added: Query<(), Added<WaterStyleLabel>>,
    mut labels: Query<&mut Text, With<WaterStyleLabel>>,
) {
    if !settings.is_changed() && added.is_empty() {
        return;
    }
    for mut text in &mut labels {
        if text.0 != settings.style.label() {
            text.0 = settings.style.label().to_string();
        }
    }
}

/// Clicks on the water checkboxes flip their settings.
fn toggle_water_checkboxes(
    boxes: Query<(&Interaction, &WaterToggle), Changed<Interaction>>,
    mut settings: ResMut<FishSettings>,
) {
    for (interaction, toggle) in &boxes {
        if *interaction == Interaction::Pressed {
            match toggle {
                WaterToggle::Water => settings.water = !settings.water,
                WaterToggle::Ripples => settings.ripples = !settings.ripples,
                WaterToggle::Bubbles => settings.bubbles = !settings.bubbles,
            }
        }
    }
}

/// Keep the water check marks and cell visibility in sync with the
/// settings — also right after the popup spawns (`Added`).
fn sync_water_widgets(
    settings: Res<FishSettings>,
    added: Query<(), Or<(Added<WaterCheckMark>, Added<WaterRow>)>>,
    mut marks: Query<(&mut Visibility, &WaterCheckMark)>,
    mut rows: Query<(&mut Node, &WaterRow)>,
) {
    if !settings.is_changed() && added.is_empty() {
        return;
    }
    for (mut visibility, mark) in &mut marks {
        let on = match mark.0 {
            WaterToggle::Water => settings.water,
            WaterToggle::Ripples => settings.ripples,
            WaterToggle::Bubbles => settings.bubbles,
        };
        *visibility = if on {
            Visibility::Inherited
        } else {
            Visibility::Hidden
        };
    }
    for (mut node, row) in &mut rows {
        let on = match row {
            WaterRow::Water => settings.water,
            WaterRow::Ripple => settings.water && settings.ripples,
            WaterRow::Bubble => settings.water && settings.bubbles,
        };
        node.display = if on { Display::Flex } else { Display::None };
    }
}

fn toggle_wave_checkbox(
    boxes: Query<&Interaction, (Changed<Interaction>, With<WaveCheckbox>)>,
    mut settings: ResMut<FishSettings>,
) {
    for interaction in &boxes {
        if *interaction == Interaction::Pressed {
            settings.wave = !settings.wave;
        }
    }
}

/// Keep the check mark and the wave rows' visibility in sync with the
/// setting — also right after the popup spawns (`Added`).
fn sync_wave_widgets(
    settings: Res<FishSettings>,
    added: Query<(), Or<(Added<WaveCheckMark>, Added<WaveRow>)>>,
    mut marks: Query<&mut Visibility, With<WaveCheckMark>>,
    mut rows: Query<&mut Node, With<WaveRow>>,
) {
    if !settings.is_changed() && added.is_empty() {
        return;
    }
    for mut visibility in &mut marks {
        *visibility = if settings.wave {
            Visibility::Inherited
        } else {
            Visibility::Hidden
        };
    }
    for mut node in &mut rows {
        node.display = if settings.wave {
            Display::Flex
        } else {
            Display::None
        };
    }
}

/// Show the school's steering-weight rows only while the Fish count is
/// above 1 — dragging the count slider past 1 reveals them live, the
/// same `visibleIf` idea as the wave rows.
fn sync_school_rows(
    settings: Res<FishSettings>,
    added: Query<(), Added<SchoolRow>>,
    mut rows: Query<&mut Node, With<SchoolRow>>,
) {
    if !settings.is_changed() && added.is_empty() {
        return;
    }
    let school = settings.count.round() as usize > 1;
    for mut node in &mut rows {
        node.display = if school { Display::Flex } else { Display::None };
    }
}

/// The shared popup's "Reset settings", restoring the authored defaults
/// while the fish is current. Live tunables apply at once, reset-time ones
/// (start size) on the next restart — like the original.
fn reset_settings(
    nav: Query<(&Interaction, &NavAction), Changed<Interaction>>,
    mut settings: ResMut<FishSettings>,
) {
    for (interaction, action) in &nav {
        if *interaction == Interaction::Pressed && matches!(action, NavAction::ResetSettings) {
            *settings = FishSettings::default();
        }
    }
}

fn update_score(game: Res<FishGame>, mut score: Query<&mut Text, With<HudScore>>) {
    let label = format!("Score: {}", game.eaten);
    for mut text in &mut score {
        if text.0 != label {
            text.0.clone_from(&label);
        }
    }
}
