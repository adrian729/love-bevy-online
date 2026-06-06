//! Instanced flock rendering: the whole flock is a single instanced draw
//! call.
//!
//! Up to ~40k boids, one `Mesh2d` entity per boid was fine. Past that the
//! per-entity engine bookkeeping (transform propagation, visibility,
//! extraction, batching) dominated the frame, growing linearly with the
//! flock. Here the simulation publishes [`FlockRenderData`] once per frame,
//! and the render world uploads it as a raw instance buffer: per-boid render
//! cost is 12 bytes of memcpy.
//!
//! Two interchangeable pipelines (`RenderStyle`):
//!
//! - **Quads** (default): the boid shape is baked once, on the CPU, into a
//!   20x12 coverage texture, and each boid is an alpha-tested quad — 6
//!   shader-generated vertices, no vertex fetch beyond the instance record.
//!   Past ~640k boids raw vertex rate is the frame's wall, and this is 3x
//!   fewer vertex invocations than the geometry path.
//! - **Geometry** (`geo` flag): the original 12-vertex octagon + triangle
//!   mesh in vertex slot 0. Kept as the visual reference the baked texture
//!   is compared against.
//!
//! Follows the `mesh2d_manual` / `custom_phase_item` Bevy examples: a custom
//! [`RenderCommand`] queued into the [`Transparent2d`] phase, reusing the
//! standard `Mesh2dPipeline` view bind group (group 0) so the shader gets the
//! ordinary 2D camera.

use std::f32::consts::TAU;

use bevy::asset::RenderAssetUsages;
use bevy::core_pipeline::core_2d::{CORE_2D_DEPTH_FORMAT, Transparent2d};
use bevy::ecs::query::ROQueryItem;
use bevy::ecs::system::SystemParamItem;
use bevy::ecs::system::lifetimeless::SRes;
use bevy::image::ImageSampler;
use bevy::math::FloatOrd;
use bevy::mesh::VertexBufferLayout;
use bevy::prelude::*;
use bevy::render::extract_resource::{ExtractResource, ExtractResourcePlugin};
use bevy::render::render_asset::RenderAssets;
use bevy::render::render_phase::{
    AddRenderCommand, DrawFunctions, PhaseItem, PhaseItemExtraIndex, RenderCommand,
    RenderCommandResult, SetItemPipeline, TrackedRenderPass, ViewSortedRenderPhases,
};
use bevy::render::render_resource::binding_types::{sampler, texture_2d};
use bevy::render::render_resource::{
    BindGroup, BindGroupEntries, BindGroupLayoutDescriptor, BindGroupLayoutEntries, BufferUsages,
    ColorTargetState, ColorWrites, CompareFunction, DepthBiasState, DepthStencilState, Extent3d,
    FragmentState, IndexFormat, MultisampleState, PipelineCache, PrimitiveState, PrimitiveTopology,
    RawBufferVec, RenderPipelineDescriptor, SamplerBindingType, ShaderStages,
    SpecializedRenderPipeline, SpecializedRenderPipelines, StencilFaceState, StencilState,
    TextureDimension, TextureFormat, TextureSampleType, VertexAttribute, VertexFormat, VertexState,
    VertexStepMode,
};
use bevy::render::renderer::{RenderDevice, RenderQueue};
use bevy::render::sync_world::MainEntity;
use bevy::render::texture::GpuImage;
use bevy::render::view::{ExtractedView, ViewTarget};
use bevy::render::{Extract, Render, RenderApp, RenderStartup, RenderSystems};
use bevy::sprite_render::{
    Mesh2dPipeline, Mesh2dPipelineKey, SetMesh2dViewBindGroup, init_mesh_2d_pipeline,
};
use bytemuck::{Pod, Zeroable};

use super::gpu_sim::{FlockGpuBuffers, GpuFlockParams};

/// Which simulation feeds the instanced draw. With `Gpu`, vertex slot 1 is
/// the compute sim's instance buffer and no per-boid data touches the CPU;
/// with `Cpu`, it's the `FlockRenderData` upload.
#[derive(Resource, Clone, Copy, PartialEq, Eq, ExtractResource)]
pub enum SimMode {
    Gpu,
    Cpu,
}

/// How a boid reaches the screen: an alpha-tested textured quad (default) or
/// the original triangle geometry (`geo` flag). Static for the process —
/// inserted straight into the render world at plugin build, no extraction.
#[derive(Resource, Clone, Copy, PartialEq, Eq)]
enum RenderStyle {
    Quads,
    Geometry,
}

/// One boid on screen. The simulation fills a `Vec` of these every frame
/// (and `sync_flock_size` appends/removes alongside the sim state).
/// The heading is stored as `(cos, sin)` — the normalized velocity comes for
/// free in the simulation (no `atan2`), and the vertex shader applies the
/// rotation as a 2x2 matrix without transcendentals (millions of vertices).
#[derive(Clone, Copy, Pod, Zeroable)]
#[repr(C)]
pub struct BoidInstance {
    pub pos: Vec2,
    pub rot: Vec2,
}

/// Main-world handoff: the instance records the render world will upload.
/// Index-aligned with the `Flock` state vector.
#[derive(Resource, Default)]
pub struct FlockRenderData(pub Vec<BoidInstance>);

/// One vertex of the shared boid mesh.
#[derive(Clone, Copy, Pod, Zeroable)]
#[repr(C)]
struct BoidVertex {
    position: [f32; 3],
    color: u32,
}

/// Red body dot (radius 3) + white heading triangle (base half-width 5 at
/// the dot, tip 14 px forward) — the same geometry the per-entity mesh used,
/// matching the LÖVE original's draw calls.
fn boid_mesh_data() -> (Vec<BoidVertex>, Vec<u32>) {
    let mut vertices = Vec::new();
    let mut indices = Vec::new();

    let red = LinearRgba::rgb(1.0, 0.0, 0.0).as_u32();
    // 8 segments: at a 3 px radius an octagon rasterizes identically to the
    // original's 20-gon, and at hundreds of thousands of boids the vertex
    // count is the render bottleneck.
    let segments = 8u32;
    vertices.push(BoidVertex {
        position: [0.0, 0.0, 0.0],
        color: red,
    });
    for i in 0..=segments {
        let a = i as f32 / segments as f32 * TAU;
        vertices.push(BoidVertex {
            position: [3.0 * a.cos(), 3.0 * a.sin(), 0.0],
            color: red,
        });
    }
    for i in 1..=segments {
        indices.extend([0, i, i + 1]);
    }

    // The triangle's indices come after the dot's, so it draws over the dot.
    let white = LinearRgba::WHITE.as_u32();
    let base = vertices.len() as u32;
    for position in [[0.0, -5.0, 0.1], [14.0, 0.0, 0.1], [0.0, 5.0, 0.1]] {
        vertices.push(BoidVertex {
            position,
            color: white,
        });
    }
    indices.extend([base, base + 1, base + 2]);

    (vertices, indices)
}

/// The quad each boid is drawn as, in boid-local space (heading = +x). The
/// shape's bounding box is x in [-3, 14], y in [-5, 5]; the extra margin
/// keeps bilinear sampling at the quad's edge inside fully transparent
/// texels. One texel = one pixel on screen, so sampling is ~1:1 and needs no
/// mipmaps.
const QUAD_MIN: Vec2 = Vec2::new(-5.0, -6.0);
const QUAD_SIZE: Vec2 = Vec2::new(20.0, 12.0);

/// Bake [`boid_mesh_data`]'s shape into a `QUAD_SIZE` coverage texture: red
/// dot, white triangle on top, 8x8 supersampled coverage per texel.
///
/// RGB is stored *premultiplied* by coverage so bilinear filtering at shape
/// edges interpolates toward transparent-black correctly; the fragment
/// shader divides alpha back out after its `a < 0.5` test. The test itself
/// reproduces the no-MSAA rasterizer's hard edge: the bilinear 0.5
/// iso-contour of supersampled coverage tracks the analytic outline the
/// geometry pipeline rasterizes.
fn boid_texture_image() -> Image {
    // Closed CCW outlines (first vertex repeated) of the two shapes.
    let octagon: Vec<Vec2> = (0..=8)
        .map(|i| {
            let a = i as f32 / 8.0 * TAU;
            3.0 * Vec2::new(a.cos(), a.sin())
        })
        .collect();
    let triangle = [
        Vec2::new(0.0, -5.0),
        Vec2::new(14.0, 0.0),
        Vec2::new(0.0, 5.0),
        Vec2::new(0.0, -5.0),
    ];
    let inside = |poly: &[Vec2], p: Vec2| {
        poly.windows(2)
            .all(|e| (e[1] - e[0]).perp_dot(p - e[0]) >= 0.0)
    };

    let (width, height) = (QUAD_SIZE.x as usize, QUAD_SIZE.y as usize);
    let mut data = Vec::with_capacity(width * height * 4);
    for row in 0..height {
        for col in 0..width {
            let (mut covered, mut white) = (0u32, 0u32);
            for sub_y in 0..8 {
                for sub_x in 0..8 {
                    let p = QUAD_MIN
                        + Vec2::new(
                            col as f32 + (sub_x as f32 + 0.5) / 8.0,
                            // Texture rows run top-down; local y runs up.
                            QUAD_SIZE.y - row as f32 - (sub_y as f32 + 0.5) / 8.0,
                        );
                    if inside(&triangle, p) {
                        covered += 1;
                        white += 1;
                    } else if inside(&octagon, p) {
                        covered += 1;
                    }
                }
            }
            // Premultiplied linear color: red (1,0,0) and white (1,1,1) both
            // have r = 1, so r = coverage; g = b = the white share.
            let alpha = covered as f32 / 64.0;
            let srgba: Srgba =
                LinearRgba::rgb(alpha, white as f32 / 64.0, white as f32 / 64.0).into();
            data.extend([
                (srgba.red * 255.0).round() as u8,
                (srgba.green * 255.0).round() as u8,
                (srgba.blue * 255.0).round() as u8,
                (alpha * 255.0).round() as u8,
            ]);
        }
    }

    let mut image = Image::new(
        Extent3d {
            width: width as u32,
            height: height as u32,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        data,
        TextureFormat::Rgba8UnormSrgb,
        RenderAssetUsages::RENDER_WORLD,
    );
    image.sampler = ImageSampler::linear();
    image
}

/// Strong handle keeping the baked boid texture alive (also handed to the
/// render world, where the bind group is built from its `GpuImage`).
#[derive(Resource)]
struct BoidTexture(Handle<Image>);

/// The texture + sampler bind group (group 1) for the quad pipeline. Created
/// once, as soon as the baked texture has been uploaded.
#[derive(Resource)]
struct FlockQuadBindGroup(BindGroup);

/// GPU buffers: the static boid mesh (uploaded once) and the per-frame
/// instance buffer.
#[derive(Resource)]
struct FlockBuffers {
    vertices: RawBufferVec<BoidVertex>,
    indices: RawBufferVec<u32>,
    index_count: u32,
    instances: RawBufferVec<BoidInstance>,
}

impl FromWorld for FlockBuffers {
    fn from_world(world: &mut World) -> Self {
        let render_device = world.resource::<RenderDevice>();
        let render_queue = world.resource::<RenderQueue>();

        let (mesh_vertices, mesh_indices) = boid_mesh_data();
        let mut vertices = RawBufferVec::new(BufferUsages::VERTEX);
        *vertices.values_mut() = mesh_vertices;
        let mut indices = RawBufferVec::new(BufferUsages::INDEX);
        let index_count = mesh_indices.len() as u32;
        *indices.values_mut() = mesh_indices;
        vertices.write_buffer(render_device, render_queue);
        indices.write_buffer(render_device, render_queue);

        Self {
            vertices,
            indices,
            index_count,
            instances: RawBufferVec::new(BufferUsages::VERTEX),
        }
    }
}

/// Copy this frame's instances into the render world — in REVERSE: both
/// shaders walk the buffer front-to-back (z decreasing with the instance
/// index) so early-z can reject the quad pipeline's occluded alpha-tested
/// fragments; reversed buffer x reversed z keeps the original
/// later-boids-draw-on-top layering. The GPU sim writes its instance buffer
/// pre-reversed (see `gpu_sim.rs`); this is the CPU path's equivalent.
/// (`Option`: the buffers resource is created in `RenderStartup`, which may
/// not have run yet.)
fn extract_flock(data: Extract<Res<FlockRenderData>>, buffers: Option<ResMut<FlockBuffers>>) {
    let Some(mut buffers) = buffers else { return };
    let instances = buffers.instances.values_mut();
    instances.clear();
    instances.extend(data.0.iter().rev().copied());
}

/// Upload the instances.
fn prepare_flock(
    mut buffers: ResMut<FlockBuffers>,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
) {
    buffers
        .instances
        .write_buffer(&render_device, &render_queue);
}

#[derive(Resource)]
struct FlockPipeline {
    mesh2d_pipeline: Mesh2dPipeline,
    geo_shader: Handle<Shader>,
    quad_shader: Handle<Shader>,
    texture_layout: BindGroupLayoutDescriptor,
    quads: bool,
}

fn init_flock_pipeline(
    mut commands: Commands,
    mesh2d_pipeline: Res<Mesh2dPipeline>,
    shader: Res<FlockShader>,
    style: Res<RenderStyle>,
) {
    let texture_layout = BindGroupLayoutDescriptor::new(
        "flock_quad_texture_layout",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::FRAGMENT,
            (
                texture_2d(TextureSampleType::Float { filterable: true }),
                sampler(SamplerBindingType::Filtering),
            ),
        ),
    );
    commands.insert_resource(FlockPipeline {
        mesh2d_pipeline: mesh2d_pipeline.clone(),
        geo_shader: shader.geo.clone(),
        quad_shader: shader.quad.clone(),
        texture_layout,
        quads: *style == RenderStyle::Quads,
    });
}

/// Build the quad pipeline's texture bind group once the baked image reaches
/// the GPU (all `Option`s cover the first frames before upload/startup).
fn prepare_quad_bind_group(
    mut commands: Commands,
    existing: Option<Res<FlockQuadBindGroup>>,
    pipeline: Option<Res<FlockPipeline>>,
    pipeline_cache: Res<PipelineCache>,
    texture: Res<BoidTexture>,
    gpu_images: Res<RenderAssets<GpuImage>>,
    render_device: Res<RenderDevice>,
) {
    if existing.is_some() {
        return;
    }
    let Some(pipeline) = pipeline else { return };
    let Some(image) = gpu_images.get(&texture.0) else {
        return;
    };
    commands.insert_resource(FlockQuadBindGroup(render_device.create_bind_group(
        "flock_quad_texture",
        &pipeline_cache.get_bind_group_layout(&pipeline.texture_layout),
        &BindGroupEntries::sequential((&image.texture_view, &image.sampler)),
    )));
}

impl SpecializedRenderPipeline for FlockPipeline {
    type Key = Mesh2dPipelineKey;

    fn specialize(&self, key: Self::Key) -> RenderPipelineDescriptor {
        let format = match key.contains(Mesh2dPipelineKey::HDR) {
            true => ViewTarget::TEXTURE_FORMAT_HDR,
            false => TextureFormat::bevy_default(),
        };

        // The per-boid instance record; slot 0 for quads (their corners come
        // from the vertex index alone), slot 1 after the mesh for geometry.
        let instance_layout = |first_location: u32| VertexBufferLayout {
            array_stride: size_of::<BoidInstance>() as u64,
            step_mode: VertexStepMode::Instance,
            attributes: vec![
                VertexAttribute {
                    format: VertexFormat::Float32x2,
                    offset: 0,
                    shader_location: first_location,
                },
                VertexAttribute {
                    format: VertexFormat::Float32x2,
                    offset: 8,
                    shader_location: first_location + 1,
                },
            ],
        };
        let (shader, buffers, layout) = if self.quads {
            (
                self.quad_shader.clone(),
                // Slot 0: one record per boid; the quad corners come from
                // the vertex index alone.
                vec![instance_layout(0)],
                // Group 0: the standard 2D view uniform; group 1: the baked
                // boid texture.
                vec![
                    self.mesh2d_pipeline.view_layout.clone(),
                    self.texture_layout.clone(),
                ],
            )
        } else {
            (
                self.geo_shader.clone(),
                vec![
                    // Slot 0: the shared boid mesh.
                    VertexBufferLayout {
                        array_stride: size_of::<BoidVertex>() as u64,
                        step_mode: VertexStepMode::Vertex,
                        attributes: vec![
                            VertexAttribute {
                                format: VertexFormat::Float32x3,
                                offset: 0,
                                shader_location: 0,
                            },
                            VertexAttribute {
                                format: VertexFormat::Uint32,
                                offset: 12,
                                shader_location: 1,
                            },
                        ],
                    },
                    // Slot 1: one record per boid.
                    instance_layout(2),
                ],
                // Group 0 only: the standard 2D view uniform.
                vec![self.mesh2d_pipeline.view_layout.clone()],
            )
        };

        RenderPipelineDescriptor {
            label: Some("flock_instanced_pipeline".into()),
            vertex: VertexState {
                shader: shader.clone(),
                buffers,
                ..default()
            },
            fragment: Some(FragmentState {
                shader,
                targets: vec![Some(ColorTargetState {
                    format,
                    // Boids are opaque; skipping the blend read-modify-write
                    // matters when a dense pile overdraws the same pixels
                    // thousands of times.
                    blend: None,
                    write_mask: ColorWrites::ALL,
                })],
                ..default()
            }),
            layout,
            primitive: PrimitiveState {
                topology: PrimitiveTopology::TriangleList,
                ..default()
            },
            depth_stencil: Some(DepthStencilState {
                format: CORE_2D_DEPTH_FORMAT,
                // Boids are opaque and get a tiny per-instance z in the
                // shader; writing depth lets the GPU's hidden-surface
                // removal kill the overdraw of dense piles (on Apple's
                // tile-based GPUs that's *all* of it).
                depth_write_enabled: true,
                depth_compare: CompareFunction::GreaterEqual,
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

/// Binds the instance buffer (+ mesh or texture) and issues the single
/// instanced draw.
struct DrawFlockInstanced;

impl<P: PhaseItem> RenderCommand<P> for DrawFlockInstanced {
    type Param = (
        SRes<FlockBuffers>,
        SRes<SimMode>,
        SRes<RenderStyle>,
        Option<SRes<FlockGpuBuffers>>,
        Option<SRes<GpuFlockParams>>,
        Option<SRes<FlockQuadBindGroup>>,
    );
    type ViewQuery = ();
    type ItemQuery = ();

    fn render<'w>(
        _: &P,
        _: ROQueryItem<'w, '_, Self::ViewQuery>,
        _: Option<ROQueryItem<'w, '_, Self::ItemQuery>>,
        (buffers, mode, style, gpu, gpu_params, quad_bind_group): SystemParamItem<
            'w,
            '_,
            Self::Param,
        >,
        pass: &mut TrackedRenderPass<'w>,
    ) -> RenderCommandResult {
        let buffers = buffers.into_inner();

        // Pick the instance source: the GPU sim's buffer (same layout), or
        // the CPU sim's upload.
        let (instances, count) = match *mode.into_inner() {
            SimMode::Gpu => match (gpu, gpu_params) {
                (Some(gpu), Some(params)) => {
                    (gpu.into_inner().instances_rev.slice(..), params.count())
                }
                _ => return RenderCommandResult::Success,
            },
            SimMode::Cpu => {
                let Some(instances) = buffers.instances.buffer() else {
                    return RenderCommandResult::Failure("flock instances not uploaded");
                };
                (instances.slice(..), buffers.instances.len() as u32)
            }
        };
        if count == 0 {
            return RenderCommandResult::Success;
        }

        match *style.into_inner() {
            RenderStyle::Quads => {
                // The baked texture may still be in flight on the very first
                // frames; skip the draw until its bind group exists.
                let Some(bind_group) = quad_bind_group else {
                    return RenderCommandResult::Success;
                };
                pass.set_vertex_buffer(0, instances);
                pass.set_bind_group(1, &bind_group.into_inner().0, &[]);
                pass.draw(0..6, 0..count);
            }
            RenderStyle::Geometry => {
                let (Some(vertices), Some(indices)) =
                    (buffers.vertices.buffer(), buffers.indices.buffer())
                else {
                    return RenderCommandResult::Failure("flock mesh not uploaded");
                };
                pass.set_vertex_buffer(0, vertices.slice(..));
                pass.set_vertex_buffer(1, instances);
                pass.set_index_buffer(indices.slice(..), IndexFormat::Uint32);
                pass.draw_indexed(0..buffers.index_count, 0, 0..count);
            }
        }
        RenderCommandResult::Success
    }
}

type DrawFlock = (
    SetItemPipeline,
    SetMesh2dViewBindGroup<0>,
    DrawFlockInstanced,
);

/// Queue the one flock draw into every 2D view.
#[allow(clippy::too_many_arguments)]
fn queue_flock(
    transparent_draw_functions: Res<DrawFunctions<Transparent2d>>,
    flock_pipeline: Res<FlockPipeline>,
    mut pipelines: ResMut<SpecializedRenderPipelines<FlockPipeline>>,
    pipeline_cache: Res<PipelineCache>,
    buffers: Res<FlockBuffers>,
    mode: Option<Res<SimMode>>,
    gpu_params: Option<Res<GpuFlockParams>>,
    mut transparent_render_phases: ResMut<ViewSortedRenderPhases<Transparent2d>>,
    views: Query<(&ExtractedView, &Msaa)>,
) {
    // `mode` arrives with the first extract.
    let any_boids = match mode.as_deref() {
        Some(SimMode::Gpu) => gpu_params.is_some_and(|params| params.count() > 0),
        Some(SimMode::Cpu) => !buffers.instances.is_empty(),
        None => false,
    };
    if !any_boids {
        return;
    }
    let draw_flock = transparent_draw_functions.read().id::<DrawFlock>();

    for (view, msaa) in &views {
        let Some(transparent_phase) = transparent_render_phases.get_mut(&view.retained_view_entity)
        else {
            continue;
        };

        let key = Mesh2dPipelineKey::from_msaa_samples(msaa.samples())
            | Mesh2dPipelineKey::from_hdr(view.hdr)
            | Mesh2dPipelineKey::from_primitive_topology(PrimitiveTopology::TriangleList);
        let pipeline_id = pipelines.specialize(&pipeline_cache, &flock_pipeline, key);

        transparent_phase.add(Transparent2d {
            // The draw is fully described by resources; no entity is involved.
            entity: (Entity::PLACEHOLDER, MainEntity::from(Entity::PLACEHOLDER)),
            draw_function: draw_flock,
            pipeline: pipeline_id,
            sort_key: FloatOrd(0.0),
            batch_range: 0..1,
            extra_index: PhaseItemExtraIndex::None,
            extracted_index: usize::MAX,
            indexed: true,
        });
    }
}

const FLOCK_SHADER: &str = r"
#import bevy_sprite::mesh2d_view_bindings::view

struct Vertex {
    @builtin(instance_index) instance: u32,
    @location(0) position: vec3<f32>,
    @location(1) color: u32,
    @location(2) i_pos: vec2<f32>,
    @location(3) i_rot: vec2<f32>, // (cos, sin) of the heading
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec4<f32>,
};

@vertex
fn vertex(vertex: Vertex) -> VertexOutput {
    var out: VertexOutput;
    let world = vec2<f32>(
        vertex.i_rot.x * vertex.position.x - vertex.i_rot.y * vertex.position.y,
        vertex.i_rot.y * vertex.position.x + vertex.i_rot.x * vertex.position.y,
    ) + vertex.i_pos;
    // Tiny per-instance z (plus the mesh's own triangle-over-dot offset) so
    // depth testing reproduces the later-draws-on-top layering. The
    // instance buffer arrives in reverse boid order (see extract_flock /
    // gpu_sim.rs), so z DECREASES with the instance index — front-to-back,
    // matching the quad pipeline. The mesh offset is folded in below 1.0 so
    // clip z never exceeds 1 (the triangle has mesh z 0.1, the dot 0.0).
    let z = 1.0 - f32(vertex.instance) * 2e-7 - (0.1 - vertex.position.z) * 1e-7;
    out.clip_position = view.clip_from_world * vec4<f32>(world, z, 1.0);
    out.color = vec4<f32>((vec4<u32>(vertex.color) >> vec4<u32>(0u, 8u, 16u, 24u)) & vec4<u32>(255u)) / 255.0;
    return out;
}

@fragment
fn fragment(in: VertexOutput) -> @location(0) vec4<f32> {
    return in.color;
}
";

/// The quad pipeline: corners are generated from the vertex index (no mesh
/// buffer at all), the boid shape comes from the baked coverage texture.
/// `discard` below the 0.5 coverage iso-line reproduces the geometry
/// pipeline's hard no-MSAA edge while keeping the pipeline opaque and
/// depth-written.
///
/// Alpha-tested fragments forfeit the tile GPU's hidden-surface removal
/// (depth can only be committed after the shader decides not to discard), so
/// a dense pile would be fragment-shaded back-to-front, thousands deep. The
/// fix: fetch instance records *in reverse*, so the draw runs front-to-back
/// and early-z kills occluded fragments against already-committed depth —
/// while each boid keeps the same z it had in the geometry pipeline
/// (later-in-the-flock draws on top, like the LÖVE original). Measured at
/// 640k pinned: ~78 fps back-to-front, ~140 front-to-back.
const QUAD_SHADER: &str = r"
#import bevy_sprite::mesh2d_view_bindings::view

@group(1) @binding(0) var boid_texture: texture_2d<f32>;
@group(1) @binding(1) var boid_sampler: sampler;

// Keep in sync with QUAD_MIN / QUAD_SIZE in render.rs.
const QUAD_MIN = vec2<f32>(-5.0, -6.0);
const QUAD_SIZE = vec2<f32>(20.0, 12.0);

struct Vertex {
    @builtin(vertex_index) index: u32,
    @builtin(instance_index) instance: u32,
    @location(0) i_pos: vec2<f32>,
    @location(1) i_rot: vec2<f32>, // (cos, sin) of the heading
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vertex(vertex: Vertex) -> VertexOutput {
    var out: VertexOutput;
    // Two CCW triangles over the unit square: (0,0)(1,0)(1,1), (0,0)(1,1)(0,1).
    let corner = vec2<f32>(
        f32(vertex.index == 1u || vertex.index == 2u || vertex.index == 4u),
        f32(vertex.index == 2u || vertex.index == 4u || vertex.index == 5u),
    );
    let local = QUAD_MIN + corner * QUAD_SIZE;
    let world = vec2<f32>(
        vertex.i_rot.x * local.x - vertex.i_rot.y * local.y,
        vertex.i_rot.y * local.x + vertex.i_rot.x * local.y,
    ) + vertex.i_pos;
    // The instance buffer is in reverse boid order and z DECREASES with the
    // instance index: the draw runs front-to-back (early-z rejects occluded
    // alpha-tested fragments, which HSR cannot) while later boids still draw
    // on top, like the LOVE original. The in-boid dot/triangle layering is
    // baked into the texture.
    let z = 1.0 - f32(vertex.instance) * 2e-7;
    out.clip_position = view.clip_from_world * vec4<f32>(world, z, 1.0);
    // Texture rows run top-down; local y runs up.
    out.uv = vec2<f32>(corner.x, 1.0 - corner.y);
    return out;
}

@fragment
fn fragment(in: VertexOutput) -> @location(0) vec4<f32> {
    let color = textureSample(boid_texture, boid_sampler, in.uv);
    if color.a < 0.5 {
        discard;
    }
    // RGB is premultiplied by coverage in the texture; divide it back out.
    return vec4<f32>(color.rgb / color.a, 1.0);
}
";

/// Resource holding the shader handles for the pipeline to take.
#[derive(Resource)]
struct FlockShader {
    geo: Handle<Shader>,
    quad: Handle<Shader>,
}

pub struct FlockRenderPlugin {
    /// `false` = the original triangle-geometry path (the `geo` perf flag).
    pub quads: bool,
}

impl Plugin for FlockRenderPlugin {
    fn build(&self, app: &mut App) {
        let geo = app
            .world_mut()
            .resource_mut::<Assets<Shader>>()
            .add(Shader::from_wgsl(FLOCK_SHADER, file!()));
        let quad = app
            .world_mut()
            .resource_mut::<Assets<Shader>>()
            .add(Shader::from_wgsl(QUAD_SHADER, file!()));
        let texture = app
            .world_mut()
            .resource_mut::<Assets<Image>>()
            .add(boid_texture_image());

        app.init_resource::<FlockRenderData>()
            // A strong handle in the main world keeps the asset alive.
            .insert_resource(BoidTexture(texture.clone()))
            .add_plugins(ExtractResourcePlugin::<SimMode>::default());

        app.sub_app_mut(RenderApp)
            .insert_resource(FlockShader { geo, quad })
            .insert_resource(BoidTexture(texture))
            .insert_resource(if self.quads {
                RenderStyle::Quads
            } else {
                RenderStyle::Geometry
            })
            .add_render_command::<Transparent2d, DrawFlock>()
            .init_resource::<SpecializedRenderPipelines<FlockPipeline>>()
            .add_systems(
                RenderStartup,
                (
                    init_flock_pipeline.after(init_mesh_2d_pipeline),
                    |mut commands: Commands| commands.init_resource::<FlockBuffers>(),
                ),
            )
            .add_systems(ExtractSchedule, extract_flock)
            .add_systems(
                Render,
                (
                    // Chained: the bind group binds the buffer the first
                    // system may (re)allocate.
                    (prepare_flock, prepare_quad_bind_group)
                        .chain()
                        .in_set(RenderSystems::PrepareResources),
                    queue_flock.in_set(RenderSystems::Queue),
                ),
            );
    }
}
