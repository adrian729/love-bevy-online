//! UI pieces shared by the menu and every experiment: the SUIT theme from
//! the original's `main.lua`, the themed button widget, and the
//! pointer-over-UI tracking the simulations read.

use bevy::ecs::relationship::RelatedSpawnerCommands;
use bevy::prelude::*;

use crate::app::{PointerOverUi, VsyncEnabled};

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
    app.add_systems(
        Update,
        (
            button_colors,
            track_pointer_over_ui,
            toggle_vsync_checkbox,
            sync_vsync_checkbox,
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

/// The clickable box of the VSync checkbox.
#[derive(Component)]
struct VsyncCheckbox;

/// Its inner check mark, shown while vsync is on.
#[derive(Component)]
struct VsyncCheckMark;

/// A SUIT-style checkbox row bound to [`VsyncEnabled`]: a clickable box
/// (hover/active colors come from [`button_colors`] via `Button`) with a
/// label beside it. Lives here rather than in an experiment: vsync is a
/// display option every experiment's options popup can include.
pub fn spawn_vsync_checkbox(parent: &mut ChildSpawner, vsync: &VsyncEnabled, font_size: f32) {
    parent
        .spawn(Node {
            flex_direction: FlexDirection::Row,
            align_items: AlignItems::Center,
            column_gap: Val::Px(8.0),
            ..default()
        })
        .with_children(|row| {
            row.spawn((
                VsyncCheckbox,
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
                    VsyncCheckMark,
                    Node {
                        width: Val::Px(10.0),
                        height: Val::Px(10.0),
                        border_radius: BorderRadius::all(Val::Px(2.0)),
                        ..default()
                    },
                    BackgroundColor(COLOR_FILL),
                    if vsync.0 {
                        Visibility::Inherited
                    } else {
                        Visibility::Hidden
                    },
                ));
            });
            row.spawn((
                Text::new("VSync (off = uncapped fps)"),
                TextFont::from_font_size(font_size),
                TextColor(Color::WHITE),
            ));
        });
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
/// exactly the original's `pointerOverUI`.
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
