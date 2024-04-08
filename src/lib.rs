use std::{
    collections::HashMap,
    f32::consts::PI,
    ops::Deref,
    path::Path,
    sync::{Arc, Mutex},
    thread,
    time::Instant,
};

use block::{BlockType, BLOCK_SIZE};
use camera::{Camera, CameraController, Projection};
use cgmath::prelude::*;
use chunk::CHUNK_SIZE;
use egui_renderer::EguiRenderer;
use egui_wgpu::ScreenDescriptor;
use fps::Fps;
use light::LightUniform;
use log::{debug, info};
use model::{DrawLight, DrawModel, ModelVertex};
use noise::{Fbm, Simplex};
use notify::{event::ModifyKind, RecommendedWatcher, RecursiveMode, Watcher};
use wgpu::util::DeviceExt;

use winit::{
    event::WindowEvent,
    event::*,
    event_loop::EventLoop,
    keyboard::{Key, KeyCode, PhysicalKey},
    window::Window,
    window::WindowBuilder,
};

use crate::{
    block::Render, chunk::Chunk16,  model::Vertex, resources::load_texture,
};

mod aabb;
mod block;
mod camera;
mod chunk;
mod egui_renderer;
mod light;
mod model;
mod resources;
mod texture;
mod voxel;
mod physics;
mod fps;

const CHUNK_RENDER_DISTANCE: i32 = 1;
pub const GRAVITY: f32 = 9.8;

pub struct Instance {
    position: cgmath::Point3<f32>,
    rotation: cgmath::Quaternion<f32>,
    block_type: BlockType,
    is_selected: bool,
}

impl Default for Instance {
    fn default() -> Self {
        Self {
            position: cgmath::Point3::new(0.0, 0.0, 0.0),
            rotation: cgmath::Quaternion::from_axis_angle(
                cgmath::Vector3::unit_z(),
                cgmath::Deg(0.0),
            ),
            block_type: BlockType::Dirt,
            is_selected: false,
        }
    }
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct InstanceRaw {
    model: [[f32; 4]; 4],
    normal: [[f32; 3]; 3],
    block_data_0: u32,
}

impl Default for InstanceRaw {
    fn default() -> Self {
        Self {
            model: cgmath::Matrix4::identity().into(),
            normal: cgmath::Matrix3::identity().into(),
            block_data_0: 0,
        }
    }
}

impl Instance {
    fn to_raw(&self) -> InstanceRaw {
        let block_data_0 = 0u32;
        let block_type: u16 = self.block_type.into();
        let block_data_0 =
            block_data_0 | block_type as u32 | if self.is_selected { 1 << 16 } else { 0 };

        InstanceRaw {
            model: (cgmath::Matrix4::from_translation(self.position.to_vec())
                * cgmath::Matrix4::from(self.rotation))
            .into(),
            normal: cgmath::Matrix3::from(self.rotation).into(),
            block_data_0,
        }
    }
}

impl InstanceRaw {
    fn desc() -> wgpu::VertexBufferLayout<'static> {
        use std::mem;
        wgpu::VertexBufferLayout {
            array_stride: mem::size_of::<InstanceRaw>() as wgpu::BufferAddress,
            // We need to switch from using a step mode of Vertex to Instance
            // This means that our shaders will only change to use the next
            // instance when the shader starts processing a new instance
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 5,
                    format: wgpu::VertexFormat::Float32x4,
                },
                wgpu::VertexAttribute {
                    offset: mem::size_of::<[f32; 4]>() as wgpu::BufferAddress,
                    shader_location: 6,
                    format: wgpu::VertexFormat::Float32x4,
                },
                wgpu::VertexAttribute {
                    offset: mem::size_of::<[f32; 8]>() as wgpu::BufferAddress,
                    shader_location: 7,
                    format: wgpu::VertexFormat::Float32x4,
                },
                wgpu::VertexAttribute {
                    offset: mem::size_of::<[f32; 12]>() as wgpu::BufferAddress,
                    shader_location: 8,
                    format: wgpu::VertexFormat::Float32x4,
                },
                wgpu::VertexAttribute {
                    offset: mem::size_of::<[f32; 16]>() as wgpu::BufferAddress,
                    shader_location: 9,
                    format: wgpu::VertexFormat::Float32x3,
                },
                wgpu::VertexAttribute {
                    offset: mem::size_of::<[f32; 19]>() as wgpu::BufferAddress,
                    shader_location: 10,
                    format: wgpu::VertexFormat::Float32x3,
                },
                wgpu::VertexAttribute {
                    offset: mem::size_of::<[f32; 22]>() as wgpu::BufferAddress,
                    shader_location: 11,
                    format: wgpu::VertexFormat::Float32x3,
                },
                wgpu::VertexAttribute {
                    offset: mem::size_of::<[f32; 25]>() as wgpu::BufferAddress,
                    shader_location: 12,
                    format: wgpu::VertexFormat::Uint32,
                },
            ],
        }
    }
}
pub const ROTATE_MATRIX: cgmath::Matrix4<f32> = cgmath::Matrix4::new(
    0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0., 0.0, 0.0, 0.0, 0.0, 1.0,
);

// We need this for Rust to store our data correctly for the shaders
#[repr(C)]
// This is so we can store this in a buffer
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct CameraUniform {
    // We can't use cgmath with bytemuck directly, so we'll have
    // to convert the Matrix4 into a 4x4 f32 array
    view_proj: [[f32; 4]; 4],
    view_position: [f32; 4],
}

impl CameraUniform {
    fn new() -> Self {
        Self {
            view_position: [0.0; 4],
            view_proj: cgmath::Matrix4::identity().into(),
        }
    }

    fn update_view_proj(&mut self, camera: &Camera, projection: &Projection) {
        self.view_position = camera.position.to_homogeneous().into();
        self.view_proj = (projection.calc_matrix_opengl() * camera.calc_matrix()).into();
    }
}

struct State {
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    size: winit::dpi::PhysicalSize<u32>,
    render_pipeline: wgpu::RenderPipeline,
    render_pipeline_layout: wgpu::PipelineLayout,
    bg_color: wgpu::Color,
    camera: Camera,
    camera_uniform: CameraUniform,
    camera_buffer: wgpu::Buffer,
    camera_bind_group: wgpu::BindGroup,
    projection: camera::Projection,
    camera_controller: CameraController,
    light_uniform: LightUniform,
    light_render_pipeline: wgpu::RenderPipeline,
    light_bind_group: wgpu::BindGroup,
    light_buffer: wgpu::Buffer,
    depth_texture: texture::Texture,
    diffuse_bind_group: wgpu::BindGroup,
    instance_buffer: wgpu::Buffer,
    instances: Vec<Instance>,
    // The window must be declared after the surface so
    // it gets dropped after it as the surface contains
    // unsafe references to the window's resources.
    obj_model: model::Model,
    window: Window,
    chunks: Vec<Chunk16>,
    file_watcher: FileWatcher,
    mouse_pressed: bool,
    noise: Fbm<Simplex>,
    wireframe: Wireframe,
    render_pipeline_dirty: bool,
    egui_renderer: EguiRenderer,
    fps_tracker: Fps
}

#[derive(Clone)]
struct FileWatcher {
    file_changes: Arc<Mutex<Vec<String>>>,
}

impl Default for FileWatcher {
    fn default() -> Self {
        Self {
            file_changes: Arc::new(Mutex::new(Vec::new())),
        }
    }
}

impl FileWatcher {
    fn write(&self, path: &str) {
        self.file_changes.lock().unwrap().push(path.to_owned());
    }
}

impl Iterator for FileWatcher {
    type Item = String;

    fn next(&mut self) -> Option<Self::Item> {
        let mut changes = self.file_changes.lock().unwrap();
        if changes.is_empty() {
            return None;
        }
        changes.pop()
    }
}

impl State {
    // Creating some of the wgpu types requires async code
    async fn new(window: Window, file_watcher: FileWatcher) -> Self {
        let size = window.inner_size();

        // The instance is a handle to our GPU
        // Backends::all => Vulkan + Metal + DX12 + Browser WebGPU
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        // # Safety
        //
        // The surface needs to live as long as the window that created it.
        // State owns the window, so this should be safe.
        let surface = unsafe {
            instance.create_surface_unsafe(
                wgpu::SurfaceTargetUnsafe::from_window(&window)
                    .expect("Unable to crate surface target from window"),
            )
        }
        .unwrap();

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::default(),
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .unwrap();

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    required_features: wgpu::Features::POLYGON_MODE_LINE,
                    // WebGL doesn't support all of wgpu's features, so if
                    // we're building for the web, we'll have to disable some.
                    required_limits: if cfg!(target_arch = "wasm32") {
                        wgpu::Limits::downlevel_webgl2_defaults()
                    } else {
                        wgpu::Limits::default()
                    },
                    label: None,
                },
                None, // Trace path
            )
            .await
            .unwrap();

        let surface_caps = surface.get_capabilities(&adapter);
        // Shader code in this tutorial assumes an sRGB surface texture. Using a different
        // one will result in all 1he uolors coming out daruer. If you want to support non
        // sRGB surfaces, you'll need to account for that when drawing to the frame.
        let surface_format = surface_caps
            .formats
            .iter()
            .copied()
            .find(|f| f.is_srgb())
            .unwrap_or(surface_caps.formats[0]);
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            desired_maximum_frame_latency: 2, // old default?
            format: surface_format,
            width: size.width,
            height: size.height,
            present_mode: wgpu::PresentMode::AutoNoVsync,
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
        };
        surface.configure(&device, &config);

        let (texture_bind_group_layout, diffuse_bind_group) = texture::setup(&device, &queue).await;

        let camera = camera::Camera::new((9.5, 10.0, -11.27), cgmath::Deg(-90.), cgmath::Rad(-0.0));
        let projection =
            camera::Projection::new(size.width, size.height, cgmath::Deg(67.0), 0.1, 100.);
        let camera_controller = CameraController::new(10.0, 1.0, 20.0, 20.0);
        let mut camera_uniform = CameraUniform::new();
        camera_uniform.update_view_proj(&camera, &projection);

        let camera_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Camera Buffer"),
            contents: bytemuck::cast_slice(&[camera_uniform]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let camera_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
                label: Some("camera_bind_group_layout"),
            });

        let camera_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &camera_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: camera_buffer.as_entire_binding(),
            }],
            label: Some("camera_bind_group"),
        });

        let depth_texture =
            texture::Texture::create_depth_texture(&device, &config, "depth_texture");

        let (light_bind_group_layout, light_bind_group, light_buffer, light_uniform) =
            light::setup(&device).await;
        // lib.rs
        let light_render_pipeline = {
            let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Light Pipeline Layout"),
                bind_group_layouts: &[&camera_bind_group_layout, &light_bind_group_layout],
                push_constant_ranges: &[],
            });
            let shader = wgpu::ShaderModuleDescriptor {
                label: Some("Light Shader"),
                source: wgpu::ShaderSource::Wgsl(include_str!("light.wgsl").into()),
            };
            create_render_pipeline(
                &device,
                &layout,
                config.format,
                Some(texture::Texture::DEPTH_FORMAT),
                &[model::ModelVertex::desc()],
                shader,
                &Wireframe::Off,
            )
        };

        let render_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Render Pipeline Layout"),
                bind_group_layouts: &[
                    &camera_bind_group_layout,
                    &light_bind_group_layout,
                    &texture_bind_group_layout,
                ],
                push_constant_ranges: &[],
            });

        let wireframe = Wireframe::Off;
        let render_pipeline = {
            let shader = wgpu::ShaderModuleDescriptor {
                label: Some("Normal Shader"),
                source: wgpu::ShaderSource::Wgsl(include_str!("../res/shader.wgsl").into()),
            };
            create_render_pipeline(
                &device,
                &render_pipeline_layout,
                config.format,
                Some(texture::Texture::DEPTH_FORMAT),
                &[model::ModelVertex::desc(), InstanceRaw::desc()],
                shader,
                &wireframe,
            )
        };

        let chunks = vec![];

        //window.set_cursor_visible(false);
        const TOTAL_CHUNKS: i32 = (CHUNK_RENDER_DISTANCE * 2 + 1) * (CHUNK_RENDER_DISTANCE * 2 + 1);
        const EXPECTED_INSTANCE_COUNT: i32 = 16 * 16 * 16 * TOTAL_CHUNKS;
        let instance_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Instance Buffer"),
            contents: bytemuck::cast_slice(
                &[InstanceRaw::default(); EXPECTED_INSTANCE_COUNT as usize],
            ),
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        });

        let obj_model = voxel::load_block(&device, &queue).unwrap();
        let noise = Fbm::<Simplex>::new(0);

        let egui_renderer = EguiRenderer::new(
            &device,       // wgpu Device
            config.format, // wgpu TextureFormat
            None,          // this can be None
            1,             // samples
            &window,       // winit Window
        );

        Self {
            window,
            egui_renderer,
            surface,
            device,
            queue,
            config,
            chunks,
            size,
            bg_color: wgpu::Color::BLACK,
            render_pipeline,
            render_pipeline_layout,
            camera_uniform,
            camera_buffer,
            projection,
            camera_bind_group,
            light_buffer,
            light_uniform,
            instances: vec![],
            instance_buffer,
            light_bind_group,
            obj_model,
            light_render_pipeline,
            camera,
            camera_controller,
            depth_texture,
            diffuse_bind_group,
            file_watcher,
            mouse_pressed: false,
            noise,
            wireframe,
            render_pipeline_dirty: false,
            fps_tracker: Fps::new(120),
        }
    }

    fn window(&self) -> &Window {
        &self.window
    }

    // impl State
    fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            self.size = new_size;
            self.config.width = new_size.width;
            self.config.height = new_size.height;
            self.projection
                .resize(self.config.width, self.config.height);
            self.surface.configure(&self.device, &self.config);
            self.depth_texture =
                texture::Texture::create_depth_texture(&self.device, &self.config, "depth_texture");
        }
    }

    fn input(&mut self, event: &WindowEvent) -> bool {
        let window = &self.window;
        let egui_renderer = &mut self.egui_renderer;
        egui_renderer.handle_input(window, event);
        match event {
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        physical_key: key,
                        state,
                        repeat,
                        ..
                    },
                ..
            } => {
                if !(self.camera_controller.process_keyboard(*key, *state)) {
                    if *state == ElementState::Pressed
                        && *key == PhysicalKey::Code(KeyCode::KeyP)
                        && repeat == &false
                    {
                        self.wireframe.toggle();
                        self.render_pipeline_dirty = true;
                        return true;
                    }
                    return false;
                }
                false
            }
            WindowEvent::MouseWheel { delta, .. } => {
                self.camera_controller.process_scroll(delta);
                true
            }
            WindowEvent::MouseInput {
                button: MouseButton::Left,
                state,
                ..
            } => {
                self.mouse_pressed = *state == ElementState::Pressed;
                true
            }
            _ => false,
        }
    }

    fn update(&mut self, dt: std::time::Duration) {
        self.camera_controller.update_camera(&mut self.camera, dt);
        physics::update_body(&mut self.camera, &self.chunks, dt);


        self.camera_uniform
            .update_view_proj(&self.camera, &self.projection);
        self.queue.write_buffer(
            &self.camera_buffer,
            0,
            bytemuck::cast_slice(&[self.camera_uniform]),
        );

        // update loaded chunks based on camera position
        let camera_pos = self.camera.position;
        let camera_chunk = (
            (camera_pos.x / (CHUNK_SIZE as f32 * BLOCK_SIZE)).floor() as i32,
            (camera_pos.z / (CHUNK_SIZE as f32 * BLOCK_SIZE)).floor() as i32,
        );

        let mut loaded = HashMap::<(i32, i32), bool>::new();
        let mut dirty = false;
        // unload oob chunks
        self.chunks.retain(|c| {
            let chunk_pos = c.origin;
            let distance = (
                (chunk_pos.x - camera_chunk.0).abs(),
                (chunk_pos.z - camera_chunk.1).abs(),
            );
            if distance.0 > CHUNK_RENDER_DISTANCE || distance.1 > CHUNK_RENDER_DISTANCE {
                false
            } else {
                loaded.insert((chunk_pos.x, chunk_pos.z), true);
                dirty = true;
                true
            }
        });

        for x in -CHUNK_RENDER_DISTANCE..=CHUNK_RENDER_DISTANCE {
            for z in -CHUNK_RENDER_DISTANCE..=CHUNK_RENDER_DISTANCE {
                let chunk_pos = (camera_chunk.0 + x, camera_chunk.1 + z);
                if loaded.contains_key(&chunk_pos) {
                    continue;
                }
                let chunk = Chunk16::new(chunk_pos.0, -1, chunk_pos.1).generate(&self.noise);
                self.chunks.push(chunk);
            }
        }

        let frustrum = self.camera.frustrum(&self.projection);
        // if a chunk has updated, update the instance data buffer
        if dirty {
            self.instances = self
                .chunks
                .iter()
                .flat_map(|c| c.culled_render(&frustrum))
                //.flat_map(|c| c.render())
                .collect::<Vec<_>>();

            log::info!("Instances: {}", self.instances.len());

            let instance_data = self
                .instances
                .iter()
                .map(Instance::to_raw)
                .collect::<Vec<_>>();

            self.queue.write_buffer(
                &self.instance_buffer,
                0,
                bytemuck::cast_slice(&instance_data),
            );
        }

        // Update the light
        let old_position: cgmath::Vector3<_> = self.light_uniform.position.into();
        self.light_uniform.position =
            (cgmath::Quaternion::from_axis_angle((0.0, 1.0, 0.0).into(), cgmath::Deg(0.0))
                * old_position)
                .into();
        self.queue.write_buffer(
            &self.light_buffer,
            0,
            bytemuck::cast_slice(&[self.light_uniform]),
        );
        // Force cursor back to middle
        if self.window.has_focus() {
            self.window
                .set_cursor_position(winit::dpi::PhysicalPosition::new(
                    self.size.width as f64 / 2.0,
                    self.size.height as f64 / 2.0,
                ))
                .unwrap();
        }
    }

    fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        let output = self.surface.get_current_texture()?;

        let updated = self.file_watcher.clone().next();

        // if shader has updated, recreate render Pipeline
        if let Some(path) = updated {
            if path.ends_with("shader.wgsl") {
                // reload the shader
                let shader_source = std::fs::read_to_string(path).unwrap();
                let shader = wgpu::ShaderModuleDescriptor {
                    label: Some("Normal Shader"),
                    source: wgpu::ShaderSource::Wgsl(shader_source.into()),
                };
                self.render_pipeline = create_render_pipeline(
                    &self.device,
                    &self.render_pipeline_layout,
                    self.config.format,
                    Some(texture::Texture::DEPTH_FORMAT),
                    &[model::ModelVertex::desc(), InstanceRaw::desc()],
                    shader,
                    &self.wireframe,
                );
            }
        }

        if self.render_pipeline_dirty {
            self.render_pipeline = create_render_pipeline(
                &self.device,
                &self.render_pipeline_layout,
                self.config.format,
                Some(texture::Texture::DEPTH_FORMAT),
                &[model::ModelVertex::desc(), InstanceRaw::desc()],
                wgpu::ShaderModuleDescriptor {
                    label: Some("Normal Shader"),
                    source: wgpu::ShaderSource::Wgsl(include_str!("../res/shader.wgsl").into()),
                },
                &self.wireframe,
            );
        }
        self.render_pipeline_dirty = false;

        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Render Encoder"),
            });

        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(self.bg_color),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth_texture.view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),

                occlusion_query_set: None,
                timestamp_writes: None,
            });

            render_pass.set_pipeline(&self.render_pipeline);

            render_pass.set_vertex_buffer(1, self.instance_buffer.slice(..));
            render_pass.set_pipeline(&self.light_render_pipeline);
            render_pass.draw_light_model(
                &self.obj_model,
                &self.camera_bind_group,
                &self.light_bind_group,
            );
            render_pass.set_pipeline(&self.render_pipeline);
            render_pass.set_bind_group(2, &self.diffuse_bind_group, &[]);
            render_pass.draw_model_instanced(
                &self.obj_model,
                0..self.instances.len() as u32,
                &self.camera_bind_group,
                &self.light_bind_group,
            );
        }

        self.fps_tracker.update(Instant::now());

        self.egui_renderer.draw(
            &self.device,
            &self.queue,
            &mut encoder,
            &self.window,
            &view,
            ScreenDescriptor {
                size_in_pixels: [self.size.width, self.size.height],
                pixels_per_point: self.window.scale_factor() as f32,
            },
            |ui| {
                egui::Window::new("Debug").show(ui, |ui| {
                    ui.label(format!("Camera Position: {:?}", self.camera.position));
                    ui.label(format!("Chunks: {}", self.chunks.len()));
                    ui.label(format!("Instances: {}", self.instances.len()));
                    ui.label(format!("FPS: {:.2}", self.fps_tracker.get_fps()));
                    ui.label(format!("Velocity: {:?}", self.camera.velocity));
                });
            },
        );


        self.queue.submit(std::iter::once(encoder.finish()));
        output.present();

        Ok(())
    }
}

enum Wireframe {
    On,
    Off,
}

impl Wireframe {
    fn toggle(&mut self) {
        match self {
            Wireframe::On => *self = Wireframe::Off,
            Wireframe::Off => *self = Wireframe::On,
        }
    }
}

fn create_render_pipeline(
    device: &wgpu::Device,
    layout: &wgpu::PipelineLayout,
    color_format: wgpu::TextureFormat,
    depth_format: Option<wgpu::TextureFormat>,
    vertex_layouts: &[wgpu::VertexBufferLayout],
    shader: wgpu::ShaderModuleDescriptor,
    wireframes: &Wireframe,
) -> wgpu::RenderPipeline {
    let shader = device.create_shader_module(shader);

    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("Render Pipeline"),
        layout: Some(layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: "vs_main",
            buffers: vertex_layouts,
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: "fs_main",
            targets: &[Some(wgpu::ColorTargetState {
                format: color_format,
                blend: Some(wgpu::BlendState {
                    alpha: wgpu::BlendComponent::REPLACE,
                    color: wgpu::BlendComponent::REPLACE,
                }),
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            strip_index_format: None,
            front_face: wgpu::FrontFace::Ccw,
            cull_mode: Some(wgpu::Face::Back),
            // Setting this to anything other than Fill requires Features::NON_FILL_POLYGON_MODE
            polygon_mode: match wireframes {
                Wireframe::On => wgpu::PolygonMode::Line,
                Wireframe::Off => wgpu::PolygonMode::Fill,
            },
            unclipped_depth: false,
            // Requires Features::CONSERVATIVE_RASTERIZATION
            conservative: false,
        },
        depth_stencil: depth_format.map(|format| wgpu::DepthStencilState {
            format,
            depth_write_enabled: true,
            depth_compare: wgpu::CompareFunction::Less,
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState::default(),
        }),
        multisample: wgpu::MultisampleState {
            count: 1,
            mask: !0,
            alpha_to_coverage_enabled: false,
        },
        multiview: None,
    })
}

pub async fn run() {
    let event_loop = EventLoop::new().unwrap();
    let window = WindowBuilder::new()
        .build(&event_loop).unwrap();

    let file_watcher = FileWatcher::default();

    // setup file watching
    let f = file_watcher.clone();
    let mut watcher = RecommendedWatcher::new(
        move |result: Result<notify::Event, notify::Error>| {
            let event = result.unwrap();
            match event.kind {
                notify::EventKind::Modify(ModifyKind::Data(_)) => {
                    for path in event.paths {
                        f.write(path.to_str().unwrap());
                    }
                }
                _ => {}
            }
        },
        notify::Config::default(),
    )
    .unwrap();

    watcher
        .watch(Path::new("res/"), RecursiveMode::Recursive)
        .unwrap();

    let mut state = State::new(window, file_watcher).await;
    let mut last_render_time = Instant::now();

    event_loop
        .run(move |event, elwt| match event {
            Event::DeviceEvent {
                event: DeviceEvent::MouseMotion{ delta, },
                .. // We're not using device_id currently
            } => {
                state.camera_controller.process_mouse(delta.0, delta.1)
            }
            Event::AboutToWait => {
                // RedrawRequested will only trigger once unless we manually
                // request it.
                state.window().request_redraw();
            }
            Event::WindowEvent {
                ref event,
                window_id,
            } if window_id == state.window.id() && !state.input(event) => match event {
                WindowEvent::RedrawRequested if window_id == state.window().id() => {
                    let now = Instant::now();
                    let dt = now - last_render_time;
                    last_render_time = now;
                    state.update(dt);
                    match state.render() {
                        Ok(_) => {}
                        // Reconfigure the surface if lost
                        Err(wgpu::SurfaceError::Lost) => state.resize(state.size),
                        // The system is out of memory, we should probably quit
                        Err(wgpu::SurfaceError::OutOfMemory) => elwt.exit(),
                        // All other errors (Outdated, Timeout) should be resolved by the next frame
                        Err(e) => eprintln!("{:?}", e),
                    }
                }
                WindowEvent::CloseRequested => elwt.exit(),
                WindowEvent::Resized(physical_size) => state.resize(*physical_size),
                _ => {}
            },
            _ => {}
        })
        .unwrap();
}
