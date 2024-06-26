use std::{
    collections::HashMap,
    f32::consts::PI,
    ops::Deref,
    path::Path,
    sync::{Arc, Mutex},
    thread,
    time::Instant,
};

use aabb::Aabb;
use block::{Block, BlockType, BLOCK_SIZE};
use camera::{Camera, CameraController, Projection};
use cgmath::{prelude::*, Vector3};
use chunk::{Chunk, ChunkWithMesh, CHUNK_SIZE};
use chunk_manager::ChunkManager;
use drops::Drop;
use egui_renderer::EguiRenderer;
use egui_wgpu::ScreenDescriptor;
use fps::Fps;
use inventory::Inventory;
use light::LightUniform;
use log::{debug, info};
use model::{DrawLight, DrawModel, MeshHandle, Model, ModelVertex};
use noise::{Fbm, Simplex};
use notify::{event::ModifyKind, RecommendedWatcher, RecursiveMode, Watcher};
use physics::{block_collision_side, KinematicBody, KinematicBodyState};
use rand::Rng;
use voxel::load_block;
use wgpu::util::DeviceExt;

use winit::{
    event::WindowEvent,
    event::*,
    event_loop::EventLoop,
    keyboard::{Key, KeyCode, PhysicalKey},
    window::Window,
    window::WindowBuilder,
};

use crate::{chunk::Chunk16, model::Vertex, resources::load_texture};

mod aabb;
mod block;
mod camera;
mod chunk;
mod chunk_manager;
mod drops;
mod egui_renderer;
mod fps;
mod inventory;
mod light;
mod model;
mod physics;
mod resources;
mod texture;
mod voxel;

const CHUNK_RENDER_DISTANCE: i32 = 12;
pub const GRAVITY: f32 = 9.8;
pub const ROTATE_MATRIX: cgmath::Matrix4<f32> = cgmath::Matrix4::new(
    0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0., 0.0, 0.0, 0.0, 0.0, 1.0,
);

const ROTATION_SPEED: f32 = 2.0 * std::f32::consts::PI / 10.0;

// We need this forRust to store our data correctly for the shaders
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
        self.view_position = camera.physics_state.position.to_homogeneous().into();
        self.view_proj = (projection.calc_matrix_opengl() * camera.calc_matrix()).into();
    }
}

struct Instance {
    position: cgmath::Point3<f32>,
    texture: u32,
}

impl Instance {
    fn to_raw(&self) -> InstanceRaw {
        InstanceRaw {
            model: cgmath::Matrix4::from_translation(self.position.to_vec()).into(),
            texture: self.texture,
        }
    }
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct InstanceRaw {
    model: [[f32; 4]; 4],
    texture: u32,
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
                // A mat4 takes up 4 vertex slots as it is technically 4 vec4s. We need to define a slot
                // for each vec4. We'll have to reassemble the mat4 in the shader.
                wgpu::VertexAttribute {
                    offset: 0,
                    // While our vertex shader only uses locations 0, and 1 now, in later tutorials, we'll
                    // be using 2, 3, and 4, for Vertex. We'll start at slot 5, not conflict with them later
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
                    format: wgpu::VertexFormat::Uint32,
                },
            ],
        }
    }
}

struct State {
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    size: winit::dpi::PhysicalSize<u32>,
    render_pipeline: wgpu::RenderPipeline,
    entity_render_pipeline: wgpu::RenderPipeline,
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
    cube_model: Model,
    // The window must be declared after the surface so
    // it gets dropped after it as the surface contains
    // unsafe references to the window's resources.
    window: Window,
    chunk_manager: ChunkManager,
    file_watcher: FileWatcher,
    mouse_pressed: bool,
    mouse_press_latched: bool,
    mouse_right_pressed: bool,
    mouse_right_press_latched: bool,
    noise: Fbm<Simplex>,
    wireframe: Wireframe,
    render_pipeline_dirty: bool,
    egui_renderer: EguiRenderer,
    fps_tracker: Fps,
    drops: Vec<Drop>,
    inventory: Inventory,
    ready: bool,
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

        let camera = camera::Camera::new((9.5, 50.0, -11.27), cgmath::Deg(-90.), cgmath::Rad(-0.0));
        let projection =
            camera::Projection::new(size.width, size.height, cgmath::Deg(67.0), 0.1, 1000.);
        let camera_controller = CameraController::new(7., 1.0, GRAVITY * 1.2, 100.0);
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
                &[model::ModelVertex::desc()],
                shader,
                &wireframe,
            )
        };

        let entity_render_pipeline = {
            let shader = wgpu::ShaderModuleDescriptor {
                label: Some("Normal Shader2"),
                source: wgpu::ShaderSource::Wgsl(include_str!("../res/entity_shader.wgsl").into()),
            };
            create_render_pipeline(
                &device,
                &render_pipeline_layout,
                config.format,
                Some(texture::Texture::DEPTH_FORMAT),
                &[model::ModelVertex::desc(), InstanceRaw::desc()],
                shader,
                &Wireframe::Off,
            )
        };

        let noise = Fbm::<Simplex>::new(0);

        let egui_renderer = EguiRenderer::new(
            &device,       // wgpu Device
            config.format, // wgpu TextureFormat
            None,          // this can be None
            1,             // samples
            &window,       // winit Window
        );

        let chunk_manager = ChunkManager::new();

        // size the instance buffer to 1000 for now
        let instance_data: Vec<InstanceRaw> = vec![
            InstanceRaw {
                model: cgmath::Matrix4::identity().into(),
                texture: 0,
            };
            1000
        ];
        let instance_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Instance Buffer"),
            contents: bytemuck::cast_slice(&instance_data),
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        });

        let cube = load_block(&device, &queue).unwrap();

        let drops = vec![];

        Self {
            window,
            egui_renderer,
            surface,
            device,
            queue,
            config,
            chunk_manager,
            size,
            bg_color: wgpu::Color::BLACK,
            render_pipeline,
            render_pipeline_layout,
            entity_render_pipeline,
            camera_uniform,
            camera_buffer,
            projection,
            cube_model: cube,
            camera_bind_group,
            light_buffer,
            light_uniform,
            light_bind_group,
            light_render_pipeline,
            camera,
            camera_controller,
            depth_texture,
            drops,
            diffuse_bind_group,
            file_watcher,
            mouse_pressed: false,
            mouse_press_latched: false,
            mouse_right_pressed: false,
            mouse_right_press_latched: false,
            noise,
            wireframe,
            render_pipeline_dirty: false,
            fps_tracker: Fps::new(120),
            ready: false,
            instance_buffer,
            inventory: Inventory::new(),
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
                    match (*state, *key, *repeat) {
                        (ElementState::Pressed, PhysicalKey::Code(KeyCode::KeyP), false) => {
                            self.wireframe.toggle();
                            self.render_pipeline_dirty = true;
                            return true;
                        }
                        (ElementState::Pressed, PhysicalKey::Code(code), false) => {
                            return match code {
                                KeyCode::Digit1 => {
                                    self.inventory.selected_index = 0;
                                    true
                                }
                                KeyCode::Digit2 => {
                                    self.inventory.selected_index = 1;
                                    true
                                }
                                KeyCode::Digit3 => {
                                    self.inventory.selected_index = 2;
                                    true
                                }
                                KeyCode::Digit4 => {
                                    self.inventory.selected_index = 3;
                                    true
                                }
                                KeyCode::Digit5 => {
                                    self.inventory.selected_index = 4;
                                    true
                                }
                                KeyCode::Digit6 => {
                                    self.inventory.selected_index = 5;
                                    true
                                }
                                KeyCode::Digit7 => {
                                    self.inventory.selected_index = 6;
                                    true
                                }
                                KeyCode::Digit8 => {
                                    self.inventory.selected_index = 7;
                                    true
                                }
                                KeyCode::Digit9 => {
                                    self.inventory.selected_index = 8;
                                    true
                                }
                                _ => {
                                    return false;
                                }
                            }
                        }
                        _ => return false,
                    }
                }
                false
            }
            WindowEvent::MouseInput {
                button: MouseButton::Left,
                state,
                ..
            } => {
                self.mouse_pressed = *state == ElementState::Pressed;
                if *state == ElementState::Released {
                    self.mouse_press_latched = false;
                }
                true
            }
            WindowEvent::MouseInput {
                button: MouseButton::Right,
                state,
                ..
            } => {
                self.mouse_right_pressed = *state == ElementState::Pressed;
                if *state == ElementState::Released {
                    self.mouse_right_press_latched = false;
                }
                true
            }
            _ => false,
        }
    }

    fn update(&mut self, dt: std::time::Duration) {
        self.camera_controller.update_camera(&mut self.camera, dt);

        self.camera_uniform
            .update_view_proj(&self.camera, &self.projection);

        self.queue.write_buffer(
            &self.camera_buffer,
            0,
            bytemuck::cast_slice(&[self.camera_uniform]),
        );

        // update loaded chunks based on camera position
        let camera_pos = self.camera.physics_state.position;
        let camera_chunk = (
            (camera_pos.x / (CHUNK_SIZE as f32 * BLOCK_SIZE)).floor() as i32,
            -1,
            (camera_pos.z / (CHUNK_SIZE as f32 * BLOCK_SIZE)).floor() as i32,
        );

        self.chunk_manager.update_loaded_chunks(
            &self.noise,
            &self.device,
            &self.queue,
            camera_chunk,
            CHUNK_RENDER_DISTANCE,
        );

        physics::update_body(
            &mut self.camera,
            self.chunk_manager
                .loaded_chunks
                .values()
                .map(|chunk| &chunk.chunk),
            dt,
        );

        for drop in self.drops.iter_mut() {
            // apply some visual rotation, we want blocks to rotate fully around the y axis every
            // 10 seconds
            let amount =
                cgmath::Quaternion::from_angle_y(cgmath::Rad(ROTATION_SPEED * dt.as_secs_f32()));
            let current = drop.rotation;
            drop.rotation = amount * current;

            physics::update_body(
                drop,
                self.chunk_manager
                    .loaded_chunks
                    .values()
                    .map(|chunk| &chunk.chunk),
                dt,
            );
        }

        self.drops.retain(|drop| {
            if drop.aabb().intersects(&self.camera.collider()) {
                self.inventory.add(drop.block_type);
                return false;
            }
            true
        });

        // reset _all_ blocks to inactive (this feels very inefficient...)
        for chunk in self.chunk_manager.loaded_chunks.values_mut() {
            for block in chunk.chunk.blocks.iter_mut() {
                block.is_selected = false;
            }
        }

        // check if player collides with any of the drops
        let mut block_updates = vec![];

        // update active block based on camera position
        if let Some(mut raycast_result) = physics::cast_ray_chunks_mut(
            &self.camera.physics_state.position,
            &self.camera.look_direction(),
            self.chunk_manager
                .loaded_chunks
                .values_mut()
                .map(|chunk| &mut chunk.chunk),
        ) {
            if !raycast_result.block().is_selected {
                raycast_result.block_mut().is_selected = true;
            }
            if self.mouse_pressed && !self.mouse_press_latched {
                let block = raycast_result.block_mut();
                block.is_active = false;
                self.mouse_press_latched = true;
                let mut drop = block.drop();
                // randomize initial velocity a bit
                drop.physics_state.velocity = cgmath::Vector3::new(
                    rand::thread_rng().gen_range(-2.0..2.0),
                    rand::thread_rng().gen_range(1.0..3.0),
                    rand::thread_rng().gen_range(-2.0..2.0),
                );

                self.drops.push(drop);
            } else if self.mouse_right_pressed && !self.mouse_right_press_latched {
                // figure out what side is selected?
                let side = block_collision_side(
                    &self.camera.physics_state.position,
                    &self.camera.look_direction(),
                    raycast_result.block(),
                );

                self.mouse_right_press_latched = true;
                let mut new_block = Block::new(raycast_result.block().coords + Vector3::from((side.x as i32, side.y as i32, side.z as i32)));
                new_block.is_active = true;
                new_block.t = self.inventory.items[self.inventory.selected_index].block_type;
                self.inventory.remove(new_block.t);
                block_updates.push(new_block);
            }
        }

        self.chunk_manager.update_blocks(block_updates);

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

        self.queue.write_buffer(
            &self.instance_buffer,
            0,
            bytemuck::cast_slice(
                &self
                    .drops
                    .iter()
                    .map(|drop| InstanceRaw {
                        model: drop.quaternion().into(),
                        texture: drop.block_type.into(),
                    })
                    .collect::<Vec<_>>(),
            ),
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

        for chunk in self
            .chunk_manager
            .loaded_chunks
            .values_mut()
            .filter(|chunk| chunk.chunk.dirty)
        {
            let new_mesh = chunk.chunk.generate_mesh();
            chunk
                .mesh_handle
                .update_mesh(&new_mesh, &self.device, &self.queue);
            chunk.chunk.dirty = false;
        }

        // if shader has updated, recreate render Pipeline
        if let Some(path) = updated {
            if path.ends_with("/shader.wgsl") {
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
                    &[model::ModelVertex::desc()],
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
                &[model::ModelVertex::desc()],
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
            render_pass.set_bind_group(2, &self.diffuse_bind_group, &[]);

            for chunk in self.chunk_manager.loaded_chunks.values() {
                render_pass.draw_mesh_instanced(
                    &chunk.mesh_handle,
                    0..1,
                    &self.camera_bind_group,
                    &self.light_bind_group,
                );
            }

            render_pass.set_bind_group(2, &self.diffuse_bind_group, &[]);
            render_pass.set_vertex_buffer(1, self.instance_buffer.slice(..));
            render_pass.set_pipeline(&self.entity_render_pipeline);
            render_pass.draw_model_instanced(
                &self.cube_model,
                0..self.drops.len() as u32,
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
                    ui.label(format!(
                        "Camera Position: {:.2}, {:.2}, {:.2}",
                        self.camera.physics_state.position.x,
                        self.camera.physics_state.position.y,
                        self.camera.physics_state.position.z
                    ));
                    ui.label(format!("FPS: {:.2}", self.fps_tracker.get_fps()));
                    ui.label(format!(
                        "Velocity: {:.2}, {:.2}, {:.2}",
                        self.camera.physics_state.velocity.x,
                        self.camera.physics_state.velocity.y,
                        self.camera.physics_state.velocity.z
                    ));
                    ui.label(format!(
                        "Look Direction: {:.2}, {:.2}, {:.2}",
                        self.camera.look_direction().x,
                        self.camera.look_direction().y,
                        self.camera.look_direction().z
                    ));
                    ui.label(format!(
                        "Grounded: {:?}",
                        self.camera.physics_state.grounded
                    ));
                    // Subsection for inventory
                    ui.separator();
                    ui.label("Inventory");
                    for (index, item) in self.inventory.items.iter().enumerate() {
                        let active_text = if index == self.inventory.selected_index {
                            " (active)"
                        } else {
                            ""
                        };
                        ui.label(format!(
                            "{:?}: {} {}",
                            item.block_type, item.count, active_text
                        ));
                    }
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
    let window = WindowBuilder::new().build(&event_loop).unwrap();

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
    let mut last_render_time = None;

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
                    let dt = now - last_render_time.unwrap_or(now);
                    last_render_time = Some(now);
                    state.update(dt);
                    match state.render() {
                        Ok(_) => {}
                        // Reconfigure the surface if lost
                        Err(wgpu::SurfaceError::Lost) => state.resize(state.size),
                        // The system is out of memory, we shuld probably quit
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
