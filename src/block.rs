use cgmath::{ElementWise, Vector3};
use log::info;

use crate::Instance;

const CHUNK_SIZE: usize = 1;

#[derive(Copy, Clone, Debug, Default)]
pub enum BlockType {
    #[default]
    Dirt,
    Grass,
    Stone,
    Wood,
    Water,
    Ore,
}

#[derive(Copy, Clone)]
pub struct Block {
    is_active: bool,
    is_selected: bool,
    t: BlockType,
    chunk_location: cgmath::Vector3<u8>,
}

impl Block {
    fn new() -> Self {
        Self {
            is_active: false,
            is_selected: false,
            t: BlockType::Stone,
            chunk_location: cgmath::Vector3::new(0, 0, 0),
        }
    }

    fn block_to_aabb(&self, chunk: Vector3<i32>) -> aabb {
        let x = (chunk.x as f32 * 16.0 + (self.chunk_location.x as f32)) * BLOCK_SIZE - BLOCK_SIZE / 2.0;
        let y = (chunk.y as f32 * 16.0 + (self.chunk_location.y as f32)) * BLOCK_SIZE - BLOCK_SIZE / 2.0;
        let z = (chunk.z as f32 * 16.0 + (self.chunk_location.z as f32)) * BLOCK_SIZE - BLOCK_SIZE / 2.0;
        let min = cgmath::Point3::new(x, y, z);
        let max = cgmath::Point3::new(x + BLOCK_SIZE, y + BLOCK_SIZE, z + BLOCK_SIZE);
        aabb { min, max }
    }
}

impl BlockType {
    pub fn to_chunk_data(self) -> u32 {
        match self {
            BlockType::Stone => 0,
            BlockType::Dirt => 1,
            BlockType::Grass => 2,
            BlockType::Ore => 16,
            _ => 255,
        }
    }
}

pub trait Render {
    fn render(&self) -> Vec<Instance>;
}

pub trait Chunk {
    fn get_block(&self, x: u8, y: u8, z: u8) -> &Block;
    fn get_block_mut(&mut self, x: u8, y: u8, z: u8) -> &mut Block;
    fn set_block(&mut self, x: u8, y: u8, z: u8, block: Block);
}

pub struct Chunk16 {
    location: cgmath::Vector3<i32>,
    blocks: [Block; CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE],
}

#[derive(Debug)]
struct aabb {
    min: cgmath::Point3<f32>,
    max: cgmath::Point3<f32>,
}

impl aabb {
    fn intersect_ray(
        &self,
        d0: cgmath::Point3<f32>,
        dir: cgmath::Vector3<f32>,
    ) -> Option<[f32; 2]> {
        let t1 = (self.min - d0).div_element_wise(dir);
        let t2 = (self.max - d0).div_element_wise(dir);
        let t_min = t1.zip(t2, f32::min);
        let t_max = t1.zip(t2, f32::max);

        let mut hit_near = t_min.x;
        let mut hit_far = t_max.x;

        if hit_near > t_max.y || t_min.y > hit_far {
            return None;
        }

        if t_min.y > hit_near {
            hit_near = t_min.y;
        }
        if t_max.y < hit_far {
            hit_far = t_max.y;
        }

        if (hit_near > t_max.z) || (t_min.z > hit_far) {
            return None;
        }

        if t_min.z > hit_near {
            hit_near = t_min.z;
        }
        if t_max.z < hit_far {
            hit_far = t_max.z;
        }
        Some([hit_near, hit_far])
    }
}

impl Chunk16 {
    // get all blocks that intersect with a ray
    pub fn raycast(&mut self, d0: cgmath::Point3<f32>, dir: cgmath::Vector3<f32>) {
        let mut candidates = Vec::new();
        let location = self.location;
        for block in self.blocks.iter_mut() {
            if block.is_active {
                let aabb = block.block_to_aabb(location);
                if let Some(hit) = aabb.intersect_ray(d0, dir) {
                    info!("hit: {:?}", hit);
                    candidates.push((block, hit));
                }
            }
        }
        candidates.sort_by(|a, b| a.1[0].partial_cmp(&b.1[0]).unwrap());
        if let Some(c) = candidates.first_mut() {
            c.0.is_selected = true;
        };
    }
}

impl Default for Chunk16 {
    fn default() -> Self {
        let mut s = Self {
            blocks: [Block::new(); CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE],
            location: cgmath::Vector3::new(0, 0, 0),
        };
        // set all the positions for blocks
        for i in 0..CHUNK_SIZE as u8 {
            for j in 0..CHUNK_SIZE as u8 {
                for k in 0..CHUNK_SIZE as u8 {
                    s.set_block(
                        i,
                        j,
                        k,
                        Block {
                            is_active: true,
                            is_selected: false,
                            t: BlockType::Stone,
                            chunk_location: cgmath::Vector3::new(i, j, k),
                        },
                    );
                }
            }
        }
        s
    }
}
const BLOCK_SIZE: f32 = 2.0;

impl Render for Chunk16 {
    fn render(&self) -> Vec<Instance> {
        self.blocks
            .iter()
            .filter(|block| block.is_active)
            .map(|block| {
                Instance {
                    position: block.chunk_location.cast::<f32>().unwrap() * BLOCK_SIZE,
                    block_type: block.t,
                    ..Default::default()
                }
            })
            .collect()
    }
}

impl Chunk16 {
    fn xyz_to_index(x: u8, y: u8, z: u8) -> usize {
        let (x, y, z) = (x as usize, y as usize, z as usize);
        x + y * CHUNK_SIZE + z * CHUNK_SIZE * CHUNK_SIZE
    }
}

impl Chunk for Chunk16 {
    fn get_block(&self, x: u8, y: u8, z: u8) -> &Block {
        let index = Self::xyz_to_index(x, y, z);
        &self.blocks[index]
    }

    fn get_block_mut(&mut self, x: u8, y: u8, z: u8) -> &mut Block {
        let index = Self::xyz_to_index(x, y, z);
        &mut self.blocks[index]
    }

    fn set_block(&mut self, x: u8, y: u8, z: u8, block: Block) {
        let index = Self::xyz_to_index(x, y, z);
        self.blocks[index] = block;
    }
}
