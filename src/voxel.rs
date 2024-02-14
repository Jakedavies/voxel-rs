use crate::model::{Model, ModelVertex};

use std::io::{BufReader, Cursor};
use wgpu::util::DeviceExt;

use crate::{model, texture};

pub fn load_block(device: &wgpu::Device, queue: &wgpu::Queue) -> anyhow::Result<Model> {
    let vertices = [
        // Top
        ModelVertex::new([1.0, 1.0, 1.0], [0.0, 1.0, 0.0]),
        ModelVertex::new([1.0, 1.0, -1.0], [0.0, 1.0, 0.0]),
        ModelVertex::new([-1.0, 1.0, -1.0], [0.0, 1.0, 0.0]),
        ModelVertex::new([-1.0, 1.0, 1.0], [0.0, 1.0, 0.0]),

        // Bottom
        ModelVertex::new([-1.0, -1.0, 1.0], [0.0, -1.0, 0.0]),
        ModelVertex::new([-1.0, -1.0, -1.0], [0.0, -1.0, 0.0]),
        ModelVertex::new([1.0, -1.0, -1.0], [0.0, -1.0, 0.0]),
        ModelVertex::new([1.0, -1.0, 1.0], [0.0, -1.0, 0.0]),
        
        // Front
        ModelVertex::new([1.0, 1.0, 1.0], [0.0, 0.0, 1.0]),
        ModelVertex::new([-1.0, 1.0, 1.0], [0.0, 0.0, 1.0]),
        ModelVertex::new([-1.0, -1.0, 1.0], [0.0, 0.0, 1.0]),
        ModelVertex::new([1.0, -1.0, 1.0], [0.0, 0.0, 1.0]),

        // Back
        ModelVertex::new([1.0, -1.0, -1.0], [0.0, 0.0, -1.0]),
        ModelVertex::new([-1.0, -1.0, -1.0], [0.0, 0.0, -1.0]),
        ModelVertex::new([-1.0, 1.0, -1.0], [0.0, 0.0, -1.0]),
        ModelVertex::new([1.0, 1.0, -1.0], [0.0, 0.0, -1.0]),

        // Left
        ModelVertex::new([-1.0, 1.0, 1.0], [-1.0, 0.0, 0.0]),
        ModelVertex::new([-1.0, 1.0, -1.0], [-1.0, 0.0, 0.0]),
        ModelVertex::new([-1.0, -1.0, -1.0], [-1.0, 0.0, 0.0]),
        ModelVertex::new([-1.0, -1.0, 1.0], [-1.0, 0.0, 0.0]),

        // Right
        ModelVertex::new([1.0, 1.0, -1.0], [1.0, 0.0, 0.0]),
        ModelVertex::new([1.0, 1.0, 1.0], [1.0, 0.0, 0.0]),
        ModelVertex::new([1.0, -1.0, 1.0], [1.0, 0.0, 0.0]),
        ModelVertex::new([1.0, -1.0, -1.0], [1.0, 0.0, 0.0]),
    ];

    let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(&format!("Vertex Buffer")),
        contents: bytemuck::cast_slice(&vertices),
        usage: wgpu::BufferUsages::VERTEX,
    });

    let indices = [
        // Top
        0, 1, 2, 2, 3, 0, 
        // Bottom
        4, 5, 6, 6, 7, 4,
        // Front
        8, 9, 10, 10, 11, 8,
        // Back
        12, 13, 14, 14, 15, 12,
        // Left
        16, 17, 18, 18, 19, 16,
        // Right
        20, 21, 22, 22, 23, 20,

    ];

    let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(&format!("Index Buffer")),
        contents: bytemuck::cast_slice(&indices),
        usage: wgpu::BufferUsages::INDEX,
    });

    let obj = model::Mesh {
        name: "block".to_string(),
        vertex_buffer,
        index_buffer,
        num_elements: indices.len() as u32,
    };

    Ok(Model {
        meshes: vec![obj],
        materials: vec![],
    })
}
