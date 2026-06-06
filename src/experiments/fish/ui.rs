//! The fish's own UI content: its score line and its options-popup
//! controls. Unlike the flock, the fish has NO on-screen top-right panel —
//! the original doesn't declare `onscreenControls` for it; its tunables
//! live only in the paused popup.

use bevy::prelude::*;

use super::settings::{FishParam, FishSettings};
use super::sim::FishGame;
use crate::app::{AppState, VsyncEnabled};
use crate::experiments::{ExperimentId, experiment_active};
use crate::ui::{
    ChildSpawner, HudScore, NavAction, slider_plugin, spawn_checkbox, spawn_options_popup,
    spawn_slider,
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

/// The fish's options-popup content: instructions, three sliders, the
/// wiggle checkbox, and the wave sliders it reveals.
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
            for param in FishParam::MAIN {
                spawn_slider(body, (), param, &settings, 14.0);
            }
            spawn_checkbox(
                body,
                WaveCheckbox,
                WaveCheckMark,
                "Wiggle (sine path)",
                settings.wave,
                14.0,
            );
            for param in FishParam::WAVE {
                spawn_slider(body, WaveRow, param, &settings, 14.0);
            }
        },
    );
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
