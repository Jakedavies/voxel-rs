use std::collections::HashMap;

use log::info;
use noise::NoiseFn;

use crate::chunk::{Chunk16, ChunkWithMesh};

pub struct ChunkManager {
    pub loaded_chunks: HashMap<(i32, i32, i32), ChunkWithMesh>,
}

impl ChunkManager {
    pub fn new() -> Self {
        Self {
            loaded_chunks: HashMap::new(),
        }
    }

    pub fn update_loaded_chunks(
        &mut self,
        noise: &impl NoiseFn<f64, 2>,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        position: (i32, i32, i32),
        radius: i32,
    ) {
        for x in (position.0 - radius)..=(position.0 + radius) {
            for z in (position.2 - radius)..=(position.2 + radius) {
                self.loaded_chunks
                    .entry((x, position.1, z))
                    .or_insert_with(|| {
                        info!("Loading chunk at {:?}", (x, position.1, z));
                        Chunk16::new(x, position.1, z)
                            .generate(&noise)
                            .generate_mesh(device, queue)
                            .expect("Failed to generate chunk mesh")
                    });
            }
        }

        // unload any chunks out of radius
        let to_remove = self
            .loaded_chunks
            .iter()
            .filter(|(pos, _)| {
                (pos.0 - position.0).abs() > radius || (pos.2 - position.2).abs() > radius
            })
            .map(|(pos, _)| *pos)
            .collect::<Vec<_>>();

        for pos in to_remove {
            info!("Unloading chunk at {:?}, too far from {:?}", pos, position);
            self.loaded_chunks.remove(&pos);
        }
    }

    pub fn tick_chunks(&self) {
        for (_, chunk) in self.loaded_chunks.iter() {
            // TODO
            //chunk.tick();
        }
    }
}
