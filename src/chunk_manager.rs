use std::collections::HashMap;

use log::info;
use noise::NoiseFn;

use crate::{block::Block, chunk::{Chunk16, ChunkWithMesh, CHUNK_SIZE}, model::{Mesh, MeshHandle}};

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
                        let chunk = Chunk16::new(x, position.1, z)
                            .generate(&noise);

                        let mesh = chunk.generate_mesh();
                        let mesh_handle = MeshHandle::from_mesh(device, &mesh);
                        ChunkWithMesh {
                            chunk,
                            mesh,
                            mesh_handle,
                        }
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

    pub fn update_blocks(&mut self, blocks: Vec<Block>) {
        for block in blocks {
            // determine the chunk
            let chunk_coords = (
                (block.coords.x).div_euclid(16),
                (block.coords.y).div_euclid(16),
                (block.coords.z).div_euclid(16),
            );

            if let Some(chunk) = self.loaded_chunks.get_mut(&chunk_coords) {
                let x = block.coords.x.rem_euclid(CHUNK_SIZE as i32);
                let y = block.coords.y.rem_euclid(CHUNK_SIZE as i32);
                let z = block.coords.z.rem_euclid(CHUNK_SIZE as i32);
                chunk.chunk.set_block(x as u8, y as u8, z as u8, block);
            }
        }
    }

    pub fn tick_chunks(&self) {
        for (_, chunk) in self.loaded_chunks.iter() {
            // TODO
            //chunk.tick();
        }
    }
}
