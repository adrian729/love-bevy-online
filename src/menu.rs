//! The main menu (state `Menu`), mirroring the original's menu screen: a
//! random experiment animates as a live backdrop — picked fresh on every
//! menu visit, the original's `pickMenuBg` — dimmed under a translucent
//! overlay, with one button per experiment in the registry.

use bevy::prelude::*;
use rand::Rng;

use crate::app::{AppState, RestartRequested};
use crate::experiments::{CurrentExperiment, EXPERIMENTS, ExperimentId};
use crate::ui::spawn_button;

pub fn plugin(app: &mut App) {
    app.add_systems(OnEnter(AppState::Menu), spawn_menu)
        .add_systems(OnExit(AppState::Menu), despawn_menu)
        .add_systems(Update, start_experiment.run_if(in_state(AppState::Menu)));
    // [Esc] on the menu quits, like the original. Native only: on the web it
    // would just freeze the canvas.
    #[cfg(not(target_arch = "wasm32"))]
    app.add_systems(Update, menu_keyboard.run_if(in_state(AppState::Menu)));
}

#[derive(Component)]
struct MenuRoot;

/// Carried by each menu button: which experiment it starts.
#[derive(Component, Clone, Copy)]
struct StartExperiment(ExperimentId);

/// The menu column over the dimmed backdrop. Entering the menu also picks a
/// random backdrop experiment and respawns it — the original picks and
/// resets a fresh one from `menuBgPool` on every visit (which also keeps
/// the fish from growing unbounded if someone lingers on the menu).
fn spawn_menu(
    mut commands: Commands,
    mut current: ResMut<CurrentExperiment>,
    mut restart: ResMut<RestartRequested>,
) {
    current.0 = EXPERIMENTS[rand::rng().random_range(0..EXPERIMENTS.len())].id;
    restart.0 = true;
    commands
        .spawn((
            MenuRoot,
            Node {
                position_type: PositionType::Absolute,
                width: Val::Percent(100.0),
                height: Val::Percent(100.0),
                flex_direction: FlexDirection::Column,
                justify_content: JustifyContent::Center,
                align_items: AlignItems::Center,
                row_gap: Val::Px(12.0),
                ..default()
            },
            // Dim the live backdrop so the buttons stay legible — the
            // original's half-alpha black over the whole screen.
            BackgroundColor(Color::srgba(0.0, 0.0, 0.0, 0.5)),
        ))
        .with_children(|menu| {
            menu.spawn((
                Text::new("love-bevy-online"),
                TextFont::from_font_size(24.0),
                TextColor(Color::WHITE),
            ));
            for experiment in EXPERIMENTS {
                // 240x40, the original's menu button size.
                spawn_button(
                    menu,
                    StartExperiment(experiment.id),
                    experiment.title,
                    Vec2::new(240.0, 40.0),
                    18.0,
                );
            }
        });
}

fn despawn_menu(mut commands: Commands, menus: Query<Entity, With<MenuRoot>>) {
    for entity in &menus {
        commands.entity(entity).despawn();
    }
}

/// A click on an experiment's button makes it current and starts it fresh,
/// like the original's `current:reset()` on selection.
fn start_experiment(
    buttons: Query<(&Interaction, &StartExperiment), Changed<Interaction>>,
    mut current: ResMut<CurrentExperiment>,
    mut restart: ResMut<RestartRequested>,
    mut next: ResMut<NextState<AppState>>,
) {
    for (interaction, start) in &buttons {
        if *interaction != Interaction::Pressed {
            continue;
        }
        current.0 = start.0;
        restart.0 = true;
        next.set(AppState::Playing);
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn menu_keyboard(keys: Res<ButtonInput<KeyCode>>, mut exit: MessageWriter<AppExit>) {
    if keys.just_pressed(KeyCode::Escape) {
        exit.write(AppExit::Success);
    }
}
