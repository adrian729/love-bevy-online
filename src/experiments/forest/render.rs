//! Forest rendering: one baked, static triangle buffer (the Lua's
//! render-once-to-canvas), uploaded only when the forest's `version` bumps and
//! redrawn every frame — the same shape as flow's **static layer**.
//!
//! Two deliberate departures from flow's static layer, both load-bearing (see
//! the plan audit):
//! - **Colour and wind live in a uniform, not the vertices.** The vertex carries
//!   only a packed `{level, leaf, coreness, sway}` (12 bytes); the **vertex
//!   shader** computes the branch colour from `lib/color.lua`'s `hsl2rgb`
//!   (`base_hue + hue_spread*level`, brightness x1.2^level) and applies a
//!   height-weighted horizontal sway. So the hue/spread/brightness/leaf-hue/wind
//!   sliders are free live updates — no regrow, no re-upload.
//! - **The fragment shader stays trivial** (it just returns the interpolated
//!   premultiplied colour). A dense feathered canopy is fill/overdraw-bound, so
//!   the CPU pre-expands the 1px feather edges (alpha 0) — moving that work into
//!   the fragment would only make the bottleneck worse.
//!
//! Colours are converted sRGB→linear in the shader before the premultiply (the
//! flow convention — the 2D target is sRGB, so a linear value is what the GPU
//! re-encodes correctly); LÖVE clamps channels at draw time, so the shader
//! clamps after `hsl2rgb` rather than capping brightness.

use bevy::core_pipeline::core_2d::{CORE_2D_DEPTH_FORMAT, Transparent2d};
use bevy::ecs::query::ROQueryItem;
use bevy::ecs::system::SystemParamItem;
use bevy::ecs::system::lifetimeless::SRes;
use bevy::math::FloatOrd;
use bevy::mesh::VertexBufferLayout;
use bevy::prelude::*;
use bevy::render::render_phase::{
    AddRenderCommand, DrawFunctions, PhaseItem, PhaseItemExtraIndex, RenderCommand,
    RenderCommandResult, SetItemPipeline, TrackedRenderPass, ViewSortedRenderPhases,
};
use bevy::render::render_resource::binding_types::uniform_buffer;
use bevy::render::render_resource::{
    BindGroup, BindGroupEntries, BindGroupLayoutDescriptor, BindGroupLayoutEntries, BlendState,
    BufferUsages, ColorTargetState, ColorWrites, CompareFunction, DepthBiasState,
    DepthStencilState, DynamicUniformBuffer, FragmentState, IndexFormat, MultisampleState,
    PipelineCache, PrimitiveState, PrimitiveTopology, RawBufferVec, RenderPipelineDescriptor,
    ShaderStages, ShaderType, SpecializedRenderPipeline, SpecializedRenderPipelines,
    StencilFaceState, StencilState, TextureFormat, VertexAttribute, VertexFormat, VertexState,
    VertexStepMode,
};
use bevy::render::renderer::{RenderDevice, RenderQueue};
use bevy::render::sync_world::MainEntity;
use bevy::render::view::{ExtractedView, ViewTarget};
use bevy::render::{Extract, Render, RenderApp, RenderStartup, RenderSystems};
use bevy::sprite_render::{
    Mesh2dPipeline, Mesh2dPipelineKey, SetMesh2dViewBindGroup, init_mesh_2d_pipeline,
};

use super::settings::ForestSettings;
use super::sim::{Forest, ForestVertex};

pub fn plugin(app: &mut App) {
    let mut shaders = app.world_mut().resource_mut::<Assets<Shader>>();
    let shader = shaders.add(Shader::from_wgsl(FOREST_SHADER, file!()));

    app.sub_app_mut(RenderApp)
        .insert_resource(ForestShader(shader))
        .add_render_command::<Transparent2d, DrawForest>()
        .init_resource::<SpecializedRenderPipelines<ForestPipeline>>()
        .add_systems(
            RenderStartup,
            (
                init_forest_pipeline.after(init_mesh_2d_pipeline),
                |mut commands: Commands| commands.init_resource::<ForestBuffers>(),
            ),
        )
        .add_systems(ExtractSchedule, extract_forest)
        .add_systems(
            Render,
            (
                prepare_forest.in_set(RenderSystems::PrepareResources),
                queue_forest.in_set(RenderSystems::Queue),
            ),
        );
}

/// The vertex shader's uniforms (mirrors the WGSL `ForestParams`): the shared
/// wind clock + the per-forest colour controls, all moved off the vertices so
/// they update live. One of these per forest, addressed by a dynamic offset.
#[derive(Clone, Copy, Default, ShaderType)]
struct ForestParams {
    time: f32,
    wind: f32,
    base_hue: f32,
    hue_spread: f32,
    brightness: f32,
    leaf_hue: f32,
}

/// One forest's draw: a contiguous slice of the merged index buffer plus the
/// dynamic offset of its colour uniform.
#[derive(Clone, Copy)]
struct ForestDraw {
    start: u32,
    count: u32,
    offset: u32,
}

/// Shader handle handed from the main world into the render app.
#[derive(Resource)]
struct ForestShader(Handle<Shader>);

/// The render world's GPU state: the baked geometry (uploaded only on a version
/// bump) and the per-frame per-forest colour/wind uniforms (one dynamic-offset
/// uniform buffer holding one `ForestParams` per forest, plus the draw list).
#[derive(Resource)]
struct ForestBuffers {
    vertices: RawBufferVec<ForestVertex>,
    indices: RawBufferVec<u32>,
    index_count: u32,
    seen: Option<u64>,
    dirty: bool,
    params: DynamicUniformBuffer<ForestParams>,
    params_bind_group: Option<BindGroup>,
    draws: Vec<ForestDraw>,
}

impl Default for ForestBuffers {
    fn default() -> Self {
        Self {
            vertices: RawBufferVec::new(BufferUsages::VERTEX),
            indices: RawBufferVec::new(BufferUsages::INDEX),
            index_count: 0,
            seen: None,
            dirty: false,
            params: DynamicUniformBuffer::default(),
            params_bind_group: None,
            draws: Vec::new(),
        }
    }
}

/// Copy this frame's state into the render world: the colour/wind uniform every
/// frame (cheap), the baked geometry only when its version moved. (`Option`:
/// the buffers resource is created in `RenderStartup`, which may not have run.)
fn extract_forest(
    forest: Extract<Res<Forest>>,
    settings: Extract<Res<ForestSettings>>,
    buffers: Option<ResMut<ForestBuffers>>,
) {
    let Some(mut buffers) = buffers else { return };

    // Rebuild the per-forest colour/wind uniforms + draw list every frame (a
    // handful of forests — cheap). Colour comes from the live settings (free
    // update, no rebuild); the index ranges come from the baked geometry. Wind +
    // the clock are scene-global, shared by every forest.
    let time = forest.wind_time;
    let wind = settings.wind;
    buffers.params.clear();
    buffers.draws.clear();
    let mut pending = Vec::with_capacity(forest.ranges.len());
    for (i, range) in forest.ranges.iter().enumerate() {
        if range.count == 0 {
            continue;
        }
        // Geometry and settings are index-aligned; if a transient mismatch ever
        // leaves a range with no matching forest, skip it (nothing to colour).
        let Some(p) = settings.forests.get(i) else {
            continue;
        };
        let offset = buffers.params.push(&ForestParams {
            time,
            wind,
            base_hue: p.hue,
            hue_spread: p.hue_spread,
            brightness: p.brightness,
            leaf_hue: p.leaf_hue,
        });
        pending.push(ForestDraw {
            start: range.start,
            count: range.count,
            offset,
        });
    }
    buffers.draws = pending;

    if buffers.seen != Some(forest.version) {
        buffers.vertices.values_mut().clone_from(&forest.vertices);
        buffers.indices.values_mut().clone_from(&forest.indices);
        buffers.index_count = forest.indices.len() as u32;
        buffers.seen = Some(forest.version);
        buffers.dirty = true;
    }
}

/// Upload: the geometry only when freshly extracted, the uniform every frame.
/// The params bind group is built once the uniform buffer exists (a fixed-size
/// uniform never reallocates, so it stays valid).
fn prepare_forest(
    mut buffers: ResMut<ForestBuffers>,
    pipeline: Option<Res<ForestPipeline>>,
    pipeline_cache: Res<PipelineCache>,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
) {
    if buffers.dirty {
        if buffers.index_count > 0 {
            buffers
                .vertices
                .write_buffer(&render_device, &render_queue);
            buffers.indices.write_buffer(&render_device, &render_queue);
        }
        buffers.dirty = false;
    }
    buffers.params.write_buffer(&render_device, &render_queue);

    let Some(pipeline) = pipeline else { return };
    // The dynamic uniform buffer can reallocate when a forest is added, so
    // rebuild the bind group each frame from its current binding (one bind group,
    // a handful of forests — negligible). `binding()` is `None` until the first
    // entry exists.
    buffers.params_bind_group = buffers.params.binding().map(|resource| {
        render_device.create_bind_group(
            "forest_params",
            &pipeline_cache.get_bind_group_layout(&pipeline.params_layout),
            &BindGroupEntries::single(resource),
        )
    });
}

/// The forest pipeline: one baked vertex buffer over the standard 2D view
/// uniform (group 0), the colour/wind uniform in group 1.
#[derive(Resource)]
struct ForestPipeline {
    mesh2d_pipeline: Mesh2dPipeline,
    shader: Handle<Shader>,
    params_layout: BindGroupLayoutDescriptor,
}

fn init_forest_pipeline(
    mut commands: Commands,
    mesh2d_pipeline: Res<Mesh2dPipeline>,
    shader: Res<ForestShader>,
) {
    let params_layout = BindGroupLayoutDescriptor::new(
        "forest_params_layout",
        &BindGroupLayoutEntries::single(ShaderStages::VERTEX, uniform_buffer::<ForestParams>(true)),
    );
    commands.insert_resource(ForestPipeline {
        mesh2d_pipeline: mesh2d_pipeline.clone(),
        shader: shader.0.clone(),
        params_layout,
    });
}

impl SpecializedRenderPipeline for ForestPipeline {
    type Key = Mesh2dPipelineKey;

    fn specialize(&self, key: Self::Key) -> RenderPipelineDescriptor {
        let format = match key.contains(Mesh2dPipelineKey::HDR) {
            true => ViewTarget::TEXTURE_FORMAT_HDR,
            false => TextureFormat::bevy_default(),
        };

        RenderPipelineDescriptor {
            label: Some("forest_pipeline".into()),
            vertex: VertexState {
                shader: self.shader.clone(),
                buffers: vec![VertexBufferLayout {
                    array_stride: size_of::<ForestVertex>() as u64,
                    step_mode: VertexStepMode::Vertex,
                    attributes: vec![
                        VertexAttribute {
                            format: VertexFormat::Float32x2,
                            offset: 0,
                            shader_location: 0,
                        },
                        VertexAttribute {
                            format: VertexFormat::Uint32,
                            offset: 8,
                            shader_location: 1,
                        },
                    ],
                }],
                ..default()
            },
            fragment: Some(FragmentState {
                shader: self.shader.clone(),
                targets: vec![Some(ColorTargetState {
                    format,
                    // Premultiplied: the feather edges emit alpha 0 (transparent),
                    // the solid cores alpha 1 — one blend serves both.
                    blend: Some(BlendState::PREMULTIPLIED_ALPHA_BLENDING),
                    write_mask: ColorWrites::ALL,
                })],
                ..default()
            }),
            layout: vec![
                self.mesh2d_pipeline.view_layout.clone(),
                self.params_layout.clone(),
            ],
            primitive: PrimitiveState {
                topology: PrimitiveTopology::TriangleList,
                ..default()
            },
            depth_stencil: Some(DepthStencilState {
                format: CORE_2D_DEPTH_FORMAT,
                // Blended vector art: no depth interaction, pure paint order.
                depth_write_enabled: false,
                depth_compare: CompareFunction::Always,
                stencil: StencilState {
                    front: StencilFaceState::IGNORE,
                    back: StencilFaceState::IGNORE,
                    read_mask: 0,
                    write_mask: 0,
                },
                bias: DepthBiasState {
                    constant: 0,
                    slope_scale: 0.0,
                    clamp: 0.0,
                },
            }),
            multisample: MultisampleState {
                count: key.msaa_samples(),
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            ..default()
        }
    }
}

/// Draws the baked forest: bind the colour/wind uniform, then one indexed draw.
struct DrawForestGeometry;

impl<P: PhaseItem> RenderCommand<P> for DrawForestGeometry {
    type Param = SRes<ForestBuffers>;
    type ViewQuery = ();
    type ItemQuery = ();

    fn render<'w>(
        _: &P,
        _: ROQueryItem<'w, '_, Self::ViewQuery>,
        _: Option<ROQueryItem<'w, '_, Self::ItemQuery>>,
        buffers: SystemParamItem<'w, '_, Self::Param>,
        pass: &mut TrackedRenderPass<'w>,
    ) -> RenderCommandResult {
        let buffers = buffers.into_inner();
        if buffers.index_count == 0 || buffers.draws.is_empty() {
            return RenderCommandResult::Success;
        }
        let Some(bind_group) = &buffers.params_bind_group else {
            return RenderCommandResult::Failure("forest params bind group not prepared");
        };
        let (Some(vertices), Some(indices)) =
            (buffers.vertices.buffer(), buffers.indices.buffer())
        else {
            return RenderCommandResult::Failure("forest buffers not uploaded");
        };
        pass.set_vertex_buffer(0, vertices.slice(..));
        pass.set_index_buffer(indices.slice(..), IndexFormat::Uint32);
        // One draw per forest, each binding its own colour uniform via a dynamic
        // offset into the shared buffer (the shared wind/clock ride along in it).
        for draw in &buffers.draws {
            pass.set_bind_group(1, bind_group, &[draw.offset]);
            pass.draw_indexed(draw.start..draw.start + draw.count, 0, 0..1);
        }
        RenderCommandResult::Success
    }
}

type DrawForest = (
    SetItemPipeline,
    SetMesh2dViewBindGroup<0>,
    DrawForestGeometry,
);

/// Queue the forest draw into every 2D view (sort key 0.0; no other experiment
/// queues while forest owns the screen, and `clear_when_inactive` empties the
/// buffer the frame it stops being current).
fn queue_forest(
    transparent_draw_functions: Res<DrawFunctions<Transparent2d>>,
    pipeline: Option<Res<ForestPipeline>>,
    mut pipelines: ResMut<SpecializedRenderPipelines<ForestPipeline>>,
    pipeline_cache: Res<PipelineCache>,
    buffers: Option<Res<ForestBuffers>>,
    mut transparent_render_phases: ResMut<ViewSortedRenderPhases<Transparent2d>>,
    views: Query<(&ExtractedView, &Msaa)>,
) {
    let (Some(pipeline), Some(buffers)) = (pipeline, buffers) else {
        return;
    };
    if buffers.index_count == 0 {
        return;
    }
    let draw_function = transparent_draw_functions.read().id::<DrawForest>();
    for (view, msaa) in &views {
        let Some(transparent_phase) = transparent_render_phases.get_mut(&view.retained_view_entity)
        else {
            continue;
        };
        let key = Mesh2dPipelineKey::from_msaa_samples(msaa.samples())
            | Mesh2dPipelineKey::from_hdr(view.hdr)
            | Mesh2dPipelineKey::from_primitive_topology(PrimitiveTopology::TriangleList);
        let pipeline_id = pipelines.specialize(&pipeline_cache, &pipeline, key);
        transparent_phase.add(Transparent2d {
            entity: (Entity::PLACEHOLDER, MainEntity::from(Entity::PLACEHOLDER)),
            draw_function,
            pipeline: pipeline_id,
            sort_key: FloatOrd(0.0),
            batch_range: 0..1,
            extra_index: PhaseItemExtraIndex::None,
            extracted_index: usize::MAX,
            indexed: true,
        });
    }
}

const FOREST_SHADER: &str = r"
#import bevy_sprite::mesh2d_view_bindings::view

struct ForestParams {
    time: f32,
    wind: f32,
    base_hue: f32,
    hue_spread: f32,
    brightness: f32,
    leaf_hue: f32,
};
@group(1) @binding(0) var<uniform> params: ForestParams;

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec4<f32>,
};

// lib/color.lua's hue2rgb / hsl2rgb (h in degrees, s/l in percent).
fn hue2rgb(p: f32, q: f32, t_in: f32) -> f32 {
    var t = t_in;
    if (t < 0.0) { t = t + 1.0; }
    if (t > 1.0) { t = t - 1.0; }
    if (t < 1.0 / 6.0) { return p + (q - p) * 6.0 * t; }
    if (t < 0.5) { return q; }
    if (t < 2.0 / 3.0) { return p + (q - p) * (2.0 / 3.0 - t) * 6.0; }
    return p;
}

fn hsl2rgb(h_deg: f32, s_pct: f32, l_pct: f32) -> vec3<f32> {
    let h = fract(h_deg / 360.0);
    let s = s_pct / 100.0;
    let l = l_pct / 100.0;
    if (s == 0.0) { return vec3<f32>(l, l, l); }
    var q: f32;
    if (l < 0.5) { q = l * (1.0 + s); } else { q = l + s - l * s; }
    let p = 2.0 * l - q;
    return vec3<f32>(hue2rgb(p, q, h + 1.0 / 3.0), hue2rgb(p, q, h), hue2rgb(p, q, h - 1.0 / 3.0));
}

// sRGB -> linear (the 2D target is sRGB; the GPU re-encodes a linear value).
fn srgb_to_linear(c: vec3<f32>) -> vec3<f32> {
    let lo = c / 12.92;
    let hi = pow((c + 0.055) / 1.055, vec3<f32>(2.4));
    return select(hi, lo, c <= vec3<f32>(0.04045));
}

@vertex
fn vertex(@location(0) pos: vec2<f32>, @location(1) packed: u32) -> VertexOutput {
    var out: VertexOutput;
    let byte0 = packed & 0xffu;
    let is_leaf = (byte0 & 0x80u) != 0u;
    let level = f32(byte0 & 0x7fu);
    let alpha = f32((packed >> 8u) & 0xffu) / 255.0;
    let sway = f32((packed >> 16u) & 0xffu) / 255.0;
    let tree_phase = f32((packed >> 24u) & 0xffu) / 255.0 * 6.2831853;

    var rgb: vec3<f32>;
    if (is_leaf) {
        rgb = hsl2rgb(params.leaf_hue, 55.0, 45.0);
    } else {
        // Brightness brightens x1.2 per branch level; LÖVE clamps the result at
        // draw time, so clamp after hsl2rgb (NOT by capping brightness, which
        // would shift the hue mix).
        rgb = hsl2rgb(params.base_hue + params.hue_spread * level, 100.0, params.brightness * pow(1.2, level));
    }
    rgb = clamp(rgb, vec3<f32>(0.0), vec3<f32>(1.0));
    let lin = srgb_to_linear(rgb);

    // Wind: a height-weighted horizontal shear. `sway` is 0 at the ground and 1
    // at the window top (concentrated toward the tips by pow), and the gust is a
    // single oscillation PER TREE (keyed off tree_phase, NOT pos.x) — so the
    // whole tree leans back and forth as one body, horizontal limbs keep their
    // length, and branch junctions (coincident verts, same height+phase) never
    // crack. Two detuned sines give it a living, non-metronomic wobble.
    let gust = sin(params.time * 1.1 + tree_phase) * 0.75 + sin(params.time * 2.3 + tree_phase * 1.7 + 1.3) * 0.25;
    let dx = pow(sway, 1.5) * params.wind * 28.0 * gust;

    out.clip_position = view.clip_from_world * vec4<f32>(pos.x + dx, pos.y, 0.0, 1.0);
    out.color = vec4<f32>(lin * alpha, alpha); // premultiplied
    return out;
}

@fragment
fn fragment(in: VertexOutput) -> @location(0) vec4<f32> {
    return in.color;
}
";

#[cfg(test)]
mod tests {
    use super::*;

    /// The shader string only reaches naga at runtime, so a typo passes every
    /// other test then kills the layer live. Parse and validate, with the one
    /// bevy `#import` stubbed (the convention caught flow's reserved-word trap).
    #[test]
    fn shaders_compile() {
        let stub = "struct View { clip_from_world: mat4x4<f32> }\n\
                    @group(0) @binding(0) var<uniform> view: View;";
        let src = FOREST_SHADER.replace("#import bevy_sprite::mesh2d_view_bindings::view", stub);
        let module = naga::front::wgsl::parse_str(&src)
            .unwrap_or_else(|e| panic!("forest shader: parse: {e}"));
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .unwrap_or_else(|e| panic!("forest shader: validate: {e}"));
    }
}
