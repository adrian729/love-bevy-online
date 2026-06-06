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
use bevy::input::keyboard::{Key, KeyboardInput};
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
    app.init_resource::<TooltipState>()
        .add_systems(Startup, spawn_hud)
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
                show_value_edits,
                value_label_hover,
                show_tooltips,
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

/// Register the typed-value-entry systems for one binding type — opt-in
/// per experiment (flow uses it for exact seeds); the other experiments'
/// sliders stay drag-only, untouched. The experiment must also give its
/// value labels an `Interaction` to make them clickable (flow tags its
/// own; see `make_value_labels_editable`).
pub fn value_entry_plugin<B: SliderBinding>(app: &mut App) {
    app.add_systems(Update, (begin_value_edits::<B>, edit_value_input::<B>));
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
                        NameLabel,
                        binding,
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
/// settings, wherever the change came from (panel, popup, or reset) — and
/// after a typed edit ends, restoring the label a cancelled edit borrowed
/// (a label mid-edit shows its buffer instead and is skipped here).
fn sync_slider_visuals<B: SliderBinding>(
    settings: Res<B::Settings>,
    mut ended_edits: RemovedComponents<ValueEdit>,
    mut fills: Query<(&B, &mut Node), With<SliderFill>>,
    mut labels: Query<(&B, &mut Text, &mut TextColor), (With<ValueLabel>, Without<ValueEdit>)>,
) {
    let edit_ended = !ended_edits.is_empty();
    ended_edits.clear();
    if !settings.is_changed() && !edit_ended {
        return;
    }
    for (binding, mut node) in &mut fills {
        node.width = Val::Percent(binding.t(&settings) * 100.0);
    }
    for (binding, mut text, mut color) in &mut labels {
        text.0 = binding.format(binding.get(&settings));
        if edit_ended {
            color.0 = COLOR_TEXT_DIM;
        }
    }
}

// ---------------------------------------------------------------------------
// Typed value entry — opt-in per experiment (only flow registers it; the
// flock/fish sliders are exactly as they were).
//
// Click a slider's value label to type the value instead of dragging:
// digits land in a buffer shown in place of the value, [Enter] commits
// (through the binding's `set`, so its range still clamps), [Esc] cancels,
// and Cmd/Ctrl+V / Cmd/Ctrl+C paste and copy the buffer — the way to share
// exact values (a flow seed) between sessions or people. Clicking another
// value label commits the open edit first. One edit can be open at a time,
// and an edit only exists while its experiment is front and center (flow
// cancels its edits on any screen change).
//
// While an edit is open the global key bindings stand down: [Esc] cancels
// the edit instead of toggling the options popup, and [R] types nothing
// rather than restarting the experiment — the handlers that could fire
// check [`ValueEdit`] (deferred removal keeps the marker visible to all
// systems for the rest of the frame, whatever their order).

/// An in-progress typed edit on a slider's value label. Public (and its
/// query type below) so global key handlers can stand down while typing.
#[derive(Component)]
pub struct ValueEdit {
    buffer: String,
}

/// Longest accepted entry — past any real value, short of nonsense.
const VALUE_EDIT_MAX_LEN: usize = 12;

/// Keep only characters that can appear in a number (paste filtering).
fn filter_value_chars(text: &str) -> String {
    text.chars()
        .filter(|c| c.is_ascii_digit() || *c == '.' || *c == '-')
        .take(VALUE_EDIT_MAX_LEN)
        .collect()
}

/// A finished entry's value, if it parses to something usable.
fn parse_typed_value(buffer: &str) -> Option<f32> {
    buffer.trim().parse::<f32>().ok().filter(|v| v.is_finite())
}

#[cfg(not(target_arch = "wasm32"))]
fn clipboard_text() -> Option<String> {
    arboard::Clipboard::new().ok()?.get_text().ok()
}

#[cfg(not(target_arch = "wasm32"))]
fn set_clipboard_text(text: &str) {
    if let Ok(mut clipboard) = arboard::Clipboard::new() {
        let _ = clipboard.set_text(text.to_owned());
    }
}

#[cfg(target_arch = "wasm32")]
fn clipboard_text() -> Option<String> {
    None
}

#[cfg(target_arch = "wasm32")]
fn set_clipboard_text(_text: &str) {}

/// A click on a value label opens a typed edit seeded with the current
/// value (so Cmd/Ctrl+C right away copies it). An edit already open on
/// another label commits first; one from another experiment's binding
/// type (only possible across a hot experiment switch) cancels.
fn begin_value_edits<B: SliderBinding>(
    mut commands: Commands,
    mut settings: ResMut<B::Settings>,
    clicked: Query<
        (Entity, &Interaction, &B),
        (Changed<Interaction>, With<ValueLabel>, Without<ValueEdit>),
    >,
    open: Query<(Entity, &B, &ValueEdit), With<ValueLabel>>,
    strays: Query<Entity, (With<ValueEdit>, Without<B>)>,
) {
    for (entity, interaction, binding) in &clicked {
        if *interaction != Interaction::Pressed {
            continue;
        }
        for (other, other_binding, edit) in &open {
            if let Some(value) = parse_typed_value(&edit.buffer) {
                other_binding.set(&mut settings, value);
            }
            commands.entity(other).remove::<ValueEdit>();
        }
        for stray in &strays {
            commands.entity(stray).remove::<ValueEdit>();
        }
        commands.entity(entity).insert(ValueEdit {
            buffer: binding.format(binding.get(&settings)),
        });
    }
}

/// Route the keyboard into the open edit: characters into the buffer,
/// [Enter] commits through the binding (its `set` clamps to the range),
/// [Esc] cancels, Cmd/Ctrl+V replaces the buffer with the clipboard,
/// Cmd/Ctrl+C copies it.
fn edit_value_input<B: SliderBinding>(
    mut commands: Commands,
    mut events: MessageReader<KeyboardInput>,
    keys: Res<ButtonInput<KeyCode>>,
    mut settings: ResMut<B::Settings>,
    mut edits: Query<(Entity, &B, &mut ValueEdit), With<ValueLabel>>,
) {
    let Ok((entity, binding, mut edit)) = edits.single_mut() else {
        // Stay drained so a freshly opened edit doesn't replay the
        // previous frames' keys.
        events.clear();
        return;
    };
    let modifier = keys.pressed(KeyCode::SuperLeft)
        || keys.pressed(KeyCode::SuperRight)
        || keys.pressed(KeyCode::ControlLeft)
        || keys.pressed(KeyCode::ControlRight);
    if modifier && keys.just_pressed(KeyCode::KeyC) {
        set_clipboard_text(&edit.buffer);
    }
    if modifier && keys.just_pressed(KeyCode::KeyV) {
        let pasted = clipboard_text().as_deref().map(filter_value_chars);
        if let Some(pasted) = pasted.filter(|p| !p.is_empty()) {
            edit.buffer = pasted;
        }
    }
    for event in events.read() {
        if !event.state.is_pressed() {
            continue;
        }
        match &event.logical_key {
            Key::Character(typed) if !modifier => {
                for c in typed.chars() {
                    if (c.is_ascii_digit() || c == '.' || c == '-')
                        && edit.buffer.len() < VALUE_EDIT_MAX_LEN
                    {
                        edit.buffer.push(c);
                    }
                }
            }
            Key::Backspace => {
                edit.buffer.pop();
            }
            Key::Enter => {
                if let Some(value) = parse_typed_value(&edit.buffer) {
                    binding.set(&mut settings, value);
                }
                commands.entity(entity).remove::<ValueEdit>();
                break;
            }
            Key::Escape => {
                commands.entity(entity).remove::<ValueEdit>();
                break;
            }
            _ => {}
        }
    }
}

/// Show the open edit's buffer (with a caret) in the value label, in the
/// active color. The label's normal text comes back via
/// `sync_slider_visuals` when the edit ends.
fn show_value_edits(
    mut edits: Query<(&ValueEdit, &mut Text, &mut TextColor), Changed<ValueEdit>>,
) {
    for (edit, mut text, mut color) in &mut edits {
        text.0 = format!("{}_", edit.buffer);
        color.0 = COLOR_ACTIVE;
    }
}

/// Make the value labels read as clickable: brighten on hover.
fn value_label_hover(
    mut labels: Query<
        (&Interaction, &mut TextColor),
        (Changed<Interaction>, With<ValueLabel>, Without<ValueEdit>),
    >,
) {
    for (interaction, mut color) in &mut labels {
        color.0 = match interaction {
            Interaction::None => COLOR_TEXT_DIM,
            _ => Color::WHITE,
        };
    }
}

// ---------------------------------------------------------------------------
// Option cyclers
//
// The original's `type = 'options'` tunables (the flow field's View and
// Palette pickers): a row of `Label  ‹ Value ›` where the arrow buttons step
// through a fixed option list. Same shape as the sliders: a binding trait an
// experiment implements per enum tunable, generic systems registered once
// per binding type.

/// Binds a cycler to one enum field of an experiment's settings resource,
/// as an index into its option list.
pub trait CyclerBinding: Component + Copy {
    type Settings: Resource;

    fn label(self) -> &'static str;
    /// How many options the list has.
    fn count(self) -> usize;
    /// Index of the current option.
    fn get(self, settings: &Self::Settings) -> usize;
    fn set(self, settings: &mut Self::Settings, index: usize);
    /// Display name of one option.
    fn option_label(self, index: usize) -> &'static str;
}

/// Register the click/sync systems for one cycler binding type.
pub fn cycler_plugin<B: CyclerBinding>(app: &mut App) {
    app.add_systems(Update, (cycle_clicks::<B>, sync_cycler_labels::<B>));
}

/// One of a cycler's step buttons; `0` holds -1 (‹) or +1 (›).
#[derive(Component)]
pub struct CyclerArrow(pub i32);

/// Text showing a cycler's current option.
#[derive(Component)]
pub struct CyclerValueLabel;

/// Label on the left, `‹ Value ›` on the right. The whole row carries
/// `row_marker`, like the sliders, so callers can show/hide it.
pub fn spawn_cycler<B: CyclerBinding>(
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
                flex_direction: FlexDirection::Row,
                justify_content: JustifyContent::SpaceBetween,
                align_items: AlignItems::Center,
                width: Val::Percent(100.0),
                ..default()
            },
        ))
        .with_children(|row| {
            row.spawn((
                NameLabel,
                binding,
                Text::new(binding.label()),
                TextFont::from_font_size(font_size),
                TextColor(Color::WHITE),
            ));
            row.spawn(Node {
                flex_direction: FlexDirection::Row,
                align_items: AlignItems::Center,
                column_gap: Val::Px(6.0),
                ..default()
            })
            .with_children(|picker| {
                let arrow = |picker: &mut ChildSpawner, delta: i32, glyph: &str| {
                    picker
                        .spawn((
                            Button,
                            binding,
                            CyclerArrow(delta),
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
                        .with_children(|button| {
                            button.spawn((
                                Text::new(glyph),
                                TextFont::from_font_size(font_size),
                                TextColor(Color::WHITE),
                            ));
                        });
                };
                arrow(picker, -1, "<");
                row_value(picker, binding, settings, font_size);
                arrow(picker, 1, ">");
            });
        });

    fn row_value<B: CyclerBinding>(
        picker: &mut ChildSpawner,
        binding: B,
        settings: &B::Settings,
        font_size: f32,
    ) {
        picker.spawn((
            CyclerValueLabel,
            binding,
            Text::new(binding.option_label(binding.get(settings))),
            TextFont::from_font_size(font_size),
            TextColor(COLOR_TEXT_DIM),
            Node {
                width: Val::Px(86.0),
                justify_content: JustifyContent::Center,
                ..default()
            },
            TextLayout::new_with_justify(Justify::Center),
        ));
    }
}

/// A click on either arrow steps the bound option, wrapping around.
fn cycle_clicks<B: CyclerBinding>(
    arrows: Query<(&Interaction, &B, &CyclerArrow), Changed<Interaction>>,
    mut settings: ResMut<B::Settings>,
) {
    for (interaction, binding, arrow) in &arrows {
        if *interaction != Interaction::Pressed {
            continue;
        }
        let count = binding.count() as i32;
        let next = (binding.get(&settings) as i32 + arrow.0).rem_euclid(count);
        binding.set(&mut settings, next as usize);
    }
}

/// Keep every bound value label in sync with the settings, wherever the
/// change came from — also right after spawning (`Added`).
fn sync_cycler_labels<B: CyclerBinding>(
    settings: Res<B::Settings>,
    added: Query<(), Added<CyclerValueLabel>>,
    mut labels: Query<(&B, &mut Text), With<CyclerValueLabel>>,
) {
    if !settings.is_changed() && added.is_empty() {
        return;
    }
    for (binding, mut text) in &mut labels {
        text.0 = binding.option_label(binding.get(&settings)).into();
    }
}

// ---------------------------------------------------------------------------
// Tooltips — opt-in per experiment, like the typed entry: the shared
// system is inert unless an experiment attaches [`Tooltip`] components to
// interactive widgets (flow tags its controls; flock/fish attach none and
// behave exactly as before). Hover a tagged widget for a moment and one
// floating bubble appears near the cursor; it hides on unhover or press.

/// Hover help for a UI widget. Attach alongside an `Interaction` (the
/// hover detection); the shared [`show_tooltips`] system does the rest.
#[derive(Component)]
pub struct Tooltip(pub &'static str);

/// A slider's or cycler's name text, tagged with its binding — a passive
/// hook (nothing queries it by default) so experiments can attach
/// tooltips or other affordances to the label of a specific tunable.
#[derive(Component)]
pub struct NameLabel;

/// Hover time before the bubble appears — long enough not to flicker
/// while the cursor crosses the panel, short enough to feel responsive.
const TOOLTIP_DELAY: f32 = 0.45;
/// The bubble's text wrap width (its position clamp allows for it).
const TOOLTIP_WIDTH: f32 = 260.0;

/// What is hovered, since when, and the bubble entity once shown.
#[derive(Resource, Default)]
struct TooltipState {
    target: Option<Entity>,
    since: f32,
    bubble: Option<Entity>,
}

/// The floating bubble (a single Text node, spawned on show, despawned on
/// hide).
#[derive(Component)]
struct TooltipBubble;

/// Show the hovered widget's tooltip after [`TOOLTIP_DELAY`], anchored
/// where the cursor was when it appeared. A widget mid-typed-edit shows
/// no tooltip (its label is busy displaying the buffer).
fn show_tooltips(
    mut commands: Commands,
    time: Res<Time>,
    window: Query<&Window, With<PrimaryWindow>>,
    hoverables: Query<(Entity, &Interaction, &Tooltip), Without<ValueEdit>>,
    mut state: ResMut<TooltipState>,
) {
    let hovered = hoverables
        .iter()
        .find(|(_, interaction, _)| **interaction == Interaction::Hovered);

    let Some((entity, _, tooltip)) = hovered else {
        // Nothing hovered (or it got pressed): drop the bubble.
        if let Some(bubble) = state.bubble.take() {
            commands.entity(bubble).despawn();
        }
        state.target = None;
        return;
    };

    if state.target != Some(entity) {
        // A new hover starts the delay (and hides any previous bubble).
        if let Some(bubble) = state.bubble.take() {
            commands.entity(bubble).despawn();
        }
        state.target = Some(entity);
        state.since = time.elapsed_secs();
        return;
    }
    if state.bubble.is_some() || time.elapsed_secs() - state.since < TOOLTIP_DELAY {
        return;
    }

    // Anchor beside the cursor, clamped so the bubble stays on screen.
    let cursor = window
        .iter()
        .next()
        .and_then(|window| window.cursor_position())
        .unwrap_or(Vec2::new(8.0, 8.0));
    let mut pos = cursor + Vec2::new(14.0, 18.0);
    if let Ok(window) = window.single() {
        pos.x = pos.x.min(window.width() - TOOLTIP_WIDTH - 16.0).max(4.0);
        pos.y = pos.y.min(window.height() - 80.0).max(4.0);
    }
    state.bubble = Some(
        commands
            .spawn((
                TooltipBubble,
                Text::new(tooltip.0),
                TextFont::from_font_size(12.0),
                TextColor(Color::srgba(1.0, 1.0, 1.0, 0.92)),
                Node {
                    position_type: PositionType::Absolute,
                    left: Val::Px(pos.x),
                    top: Val::Px(pos.y),
                    max_width: Val::Px(TOOLTIP_WIDTH),
                    padding: UiRect::axes(Val::Px(8.0), Val::Px(5.0)),
                    border_radius: BorderRadius::all(Val::Px(4.0)),
                    ..default()
                },
                BackgroundColor(Color::srgba(0.05, 0.05, 0.07, 0.92)),
                // Above the options popup (GlobalZIndex 10).
                GlobalZIndex(20),
            ))
            .id(),
    );
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
/// there.) Stands down while a value is being typed: [Esc] then cancels
/// the edit, and an "o" keystroke is just a rejected character.
fn keyboard_options_toggle(
    keys: Res<ButtonInput<KeyCode>>,
    state: Res<State<AppState>>,
    mut next: ResMut<NextState<AppState>>,
    edits: Query<(), With<ValueEdit>>,
) {
    if !edits.is_empty() {
        return;
    }
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

#[cfg(test)]
mod tests {
    use super::*;

    /// Pasted text keeps only numeric characters, capped in length.
    #[test]
    fn filter_value_chars_strips_noise() {
        assert_eq!(filter_value_chars("seed: 123456"), "123456");
        assert_eq!(filter_value_chars("-0.25\n"), "-0.25");
        assert_eq!(filter_value_chars("hello"), "");
        assert_eq!(filter_value_chars("123456789012345678").len(), 12);
    }

    /// Typed entries parse leniently but reject the unusable.
    #[test]
    fn parse_typed_value_accepts_numbers_only() {
        assert_eq!(parse_typed_value("123456"), Some(123456.0));
        assert_eq!(parse_typed_value(" 0.25 "), Some(0.25));
        assert_eq!(parse_typed_value("-100"), Some(-100.0));
        assert_eq!(parse_typed_value(""), None);
        assert_eq!(parse_typed_value("-"), None);
        assert_eq!(parse_typed_value("1.2.3"), None);
        assert_eq!(parse_typed_value("inf"), None);
    }

    // -----------------------------------------------------------------
    // The full click → type → commit/cancel flow, driven headlessly with
    // synthetic Interaction and KeyboardInput.

    #[derive(Resource)]
    struct TestSettings {
        value: f32,
    }

    #[derive(Component, Clone, Copy)]
    struct TestParam;

    impl SliderBinding for TestParam {
        type Settings = TestSettings;

        fn label(self) -> &'static str {
            "Test"
        }
        fn range(self) -> (f32, f32) {
            (0.0, 256_000.0)
        }
        fn get(self, settings: &TestSettings) -> f32 {
            settings.value
        }
        fn set(self, settings: &mut TestSettings, value: f32) {
            let (min, max) = self.range();
            settings.value = value.clamp(min, max);
        }
        fn format(self, value: f32) -> String {
            format!("{}", value.round() as i32)
        }
    }

    fn edit_app() -> (App, Entity) {
        let mut app = App::new();
        app.insert_resource(TestSettings { value: 1234.0 })
            .init_resource::<ButtonInput<KeyCode>>()
            .add_message::<KeyboardInput>()
            .add_systems(Update, show_value_edits);
        slider_plugin::<TestParam>(&mut app);
        value_entry_plugin::<TestParam>(&mut app);
        let label = app
            .world_mut()
            .spawn((
                ValueLabel,
                TestParam,
                Interaction::None,
                Text::new("1234"),
                TextColor(COLOR_TEXT_DIM),
            ))
            .id();
        app.update();
        (app, label)
    }

    fn press(app: &mut App, logical_key: Key) {
        app.world_mut().write_message(KeyboardInput {
            key_code: KeyCode::F35, // unused by anything under test
            logical_key,
            state: bevy::input::ButtonState::Pressed,
            text: None,
            repeat: false,
            window: Entity::PLACEHOLDER,
        });
        app.update();
    }

    fn click(app: &mut App, label: Entity) {
        *app.world_mut().get_mut::<Interaction>(label).unwrap() = Interaction::Pressed;
        app.update();
        // Commands flush between updates; let the insert land everywhere.
        app.update();
    }

    /// Click opens an edit seeded with the current value; typed digits
    /// append; Enter commits through the binding's clamp.
    #[test]
    fn typed_edit_commits_on_enter() {
        let (mut app, label) = edit_app();
        click(&mut app, label);
        let edit = app.world().get::<ValueEdit>(label).expect("edit open");
        assert_eq!(edit.buffer, "1234");
        // Type two more digits: 1234 -> 123456.
        press(&mut app, Key::Character("5".into()));
        press(&mut app, Key::Character("6".into()));
        assert_eq!(app.world().get::<ValueEdit>(label).unwrap().buffer, "123456");
        // The label shows the buffer with a caret while editing (the
        // display may trail the keystroke by one frame).
        app.update();
        assert_eq!(app.world().get::<Text>(label).unwrap().0, "123456_");
        press(&mut app, Key::Enter);
        app.update();
        assert_eq!(app.world().resource::<TestSettings>().value, 123456.0);
        assert!(app.world().get::<ValueEdit>(label).is_none(), "edit closed");
        // The label is back to showing the (formatted) committed value.
        assert_eq!(app.world().get::<Text>(label).unwrap().0, "123456");
    }

    /// Escape cancels: the settings keep their old value and the label
    /// text is restored.
    #[test]
    fn typed_edit_cancels_on_escape() {
        let (mut app, label) = edit_app();
        click(&mut app, label);
        press(&mut app, Key::Character("9".into()));
        press(&mut app, Key::Escape);
        app.update();
        assert_eq!(app.world().resource::<TestSettings>().value, 1234.0);
        assert!(app.world().get::<ValueEdit>(label).is_none());
        assert_eq!(app.world().get::<Text>(label).unwrap().0, "1234");
    }

    /// Tooltips appear only after the hover delay and vanish on unhover.
    #[test]
    fn tooltips_show_after_delay_and_hide_on_unhover() {
        use std::time::Duration;

        let mut app = App::new();
        app.init_resource::<TooltipState>()
            .insert_resource(Time::<()>::default())
            .add_systems(Update, show_tooltips);
        let widget = app
            .world_mut()
            .spawn((Interaction::None, Tooltip("explains the thing")))
            .id();
        app.update();

        let bubbles = |app: &mut App| {
            app.world_mut()
                .query_filtered::<&Text, With<TooltipBubble>>()
                .iter(app.world())
                .map(|text| text.0.clone())
                .collect::<Vec<_>>()
        };

        // Hovering does not show it instantly...
        *app.world_mut().get_mut::<Interaction>(widget).unwrap() = Interaction::Hovered;
        app.update();
        app.update();
        assert!(bubbles(&mut app).is_empty(), "tooltip showed instantly");

        // ...only after the delay.
        app.world_mut()
            .resource_mut::<Time>()
            .advance_by(Duration::from_secs_f32(TOOLTIP_DELAY + 0.1));
        app.update();
        app.update();
        assert_eq!(bubbles(&mut app), vec!["explains the thing".to_string()]);

        // Unhover removes it.
        *app.world_mut().get_mut::<Interaction>(widget).unwrap() = Interaction::None;
        app.update();
        app.update();
        assert!(bubbles(&mut app).is_empty(), "tooltip survived unhover");
    }

    /// Commits clamp to the binding's range, and non-numeric characters
    /// (an "r" — the restart key) never reach the buffer.
    #[test]
    fn typed_edit_clamps_and_filters() {
        let (mut app, label) = edit_app();
        click(&mut app, label);
        // Backspace the seeded value away, then type past the range max.
        for _ in 0..4 {
            press(&mut app, Key::Backspace);
        }
        press(&mut app, Key::Character("r".into()));
        for _ in 0..7 {
            press(&mut app, Key::Character("9".into()));
        }
        assert_eq!(app.world().get::<ValueEdit>(label).unwrap().buffer, "9999999");
        press(&mut app, Key::Enter);
        app.update();
        assert_eq!(app.world().resource::<TestSettings>().value, 256_000.0);
    }
}
