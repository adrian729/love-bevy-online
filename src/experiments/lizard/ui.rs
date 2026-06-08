//! The lizard's own UI content: its score line and its options-popup
//! controls. Like the fish, the lizard has NO on-screen top-right panel —
//! its tunables live only in the paused popup.

use bevy::prelude::*;

use super::settings::{LizardParam, LizardSettings};
use super::sim::{LizardEntity, LizardGame};
use crate::app::{AppState, VsyncEnabled};
use crate::experiments::{ExperimentId, experiment_active};
use crate::ui::{
    ChildSpawner, HudScore, NavAction, slider_plugin, spawn_checkbox, spawn_options_popup,
    spawn_slider,
};

pub fn plugin(app: &mut App) {
    slider_plugin::<LizardParam>(app);
    app.add_systems(
        OnEnter(AppState::Options),
        spawn_popup.run_if(experiment_active(ExperimentId::Lizard)),
    )
    .add_systems(
        Update,
        (
            toggle_skeleton_checkbox,
            sync_skeleton_checkmark,
            (update_score, reset_settings).run_if(experiment_active(ExperimentId::Lizard)),
        ),
    );
}

/// The clickable box of the "Skeleton view" checkbox — the guide's rig of
/// lines and circles instead of the skinned lizard.
#[derive(Component)]
struct SkeletonCheckbox;

/// Its inner check mark.
#[derive(Component)]
struct SkeletonCheckMark;

/// One popup cell at `width` percent — the fish popup's column wrap; the
/// game sliders and the skeleton toggle sit two abreast.
fn cell(grid: &mut ChildSpawner, width: f32, spawn: impl FnOnce(&mut ChildSpawner)) {
    grid.spawn(Node {
        width: Val::Percent(width),
        flex_direction: FlexDirection::Column,
        ..default()
    })
    .with_children(spawn);
}

/// The lizard's options-popup content: instructions, the three behavioural
/// sliders and the skeleton-view checkbox in a two-column wrap.
fn spawn_popup(mut commands: Commands, settings: Res<LizardSettings>, vsync: Res<VsyncEnabled>) {
    spawn_options_popup(
        &mut commands,
        &vsync,
        &[
            // Plain hyphen: Bevy's default font has no em-dash glyph.
            "Move your mouse - the lizard walks toward it.",
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
                for param in LizardParam::ALL {
                    cell(grid, 47.0, |c| spawn_slider(c, (), param, &settings, 14.0));
                }
                cell(grid, 47.0, |c| {
                    spawn_checkbox(
                        c,
                        SkeletonCheckbox,
                        SkeletonCheckMark,
                        "Skeleton view",
                        settings.skeleton,
                        14.0,
                    );
                });
            });
        },
    );
}

fn toggle_skeleton_checkbox(
    boxes: Query<&Interaction, (Changed<Interaction>, With<SkeletonCheckbox>)>,
    mut settings: ResMut<LizardSettings>,
) {
    for interaction in &boxes {
        if *interaction == Interaction::Pressed {
            settings.skeleton = !settings.skeleton;
        }
    }
}

fn sync_skeleton_checkmark(
    settings: Res<LizardSettings>,
    added: Query<(), Added<SkeletonCheckMark>>,
    mut marks: Query<&mut Visibility, With<SkeletonCheckMark>>,
) {
    if !settings.is_changed() && added.is_empty() {
        return;
    }
    for mut visibility in &mut marks {
        *visibility = if settings.skeleton {
            Visibility::Inherited
        } else {
            Visibility::Hidden
        };
    }
}

/// The shared popup's "Reset settings", restoring the authored defaults
/// while the lizard is current.
fn reset_settings(
    nav: Query<(&Interaction, &NavAction), Changed<Interaction>>,
    mut settings: ResMut<LizardSettings>,
) {
    for (interaction, action) in &nav {
        if *interaction == Interaction::Pressed && matches!(action, NavAction::ResetSettings) {
            *settings = LizardSettings::default();
        }
    }
}

fn update_score(
    game: Res<LizardGame>,
    lizard: Res<LizardEntity>,
    mut score: Query<&mut Text, With<HudScore>>,
) {
    let size = lizard.0.as_ref().map_or(0.0, |lizard| lizard.scale);
    let label = format!("Score: {}   Size {size:.2}", game.eaten);
    for mut text in &mut score {
        if text.0 != label {
            text.0.clone_from(&label);
        }
    }
}
