use crate::{
    aabb::{Aabb, AabbBounds},
    block::{Block, BlockType, Render, BLOCK_SIZE},
    camera::Frustrum,
    Instance,
};
use cgmath::{prelude::*, Point3, Vector3};
use log::info;
use noise::NoiseFn;

pub const CHUNK_SIZE: usize = 16;
const NOISE_SCALE: f64 = 0.01;

pub trait Chunk {
    fn get_block(&self, x: u8, y: u8, z: u8) -> &Block;
    fn get_block_mut(&mut self, x: u8, y: u8, z: u8) -> &mut Block;
    fn set_block(&mut self, x: u8, y: u8, z: u8, block: Block);
}

pub struct Chunk16 {
    pub origin: cgmath::Point3<i32>,
    pub blocks: [Block; CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE],
}

impl Chunk16 {
    pub fn new(x: i32, y: i32, z: i32) -> Self {
        Self {
            origin: cgmath::Point3::new(x, y, z),
            blocks: {
                let mut blocks = Vec::with_capacity(CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE);
                let chunk_origin = cgmath::Point3::new(
                    x as f32 * CHUNK_SIZE as f32 * BLOCK_SIZE,
                    y as f32 * CHUNK_SIZE as f32 * BLOCK_SIZE,
                    z as f32 * CHUNK_SIZE as f32 * BLOCK_SIZE,
                );

                for index in 0..CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE {
                    blocks.push(Block::new(chunk_origin));
                    let (x_, y_, z_) = Self::index_to_xyz(index);
                    let block_origin = chunk_origin
                        + cgmath::Vector3::new(
                            x_ as f32 * BLOCK_SIZE + BLOCK_SIZE / 2.0,
                            y_ as f32 * BLOCK_SIZE + BLOCK_SIZE / 2.0,
                            z_ as f32 * BLOCK_SIZE + BLOCK_SIZE / 2.0,
                        );
                    blocks[index] = Block::new(block_origin);
                }
                blocks[..CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE]
                    .try_into()
                    .unwrap()
            },
        }
    }

    pub fn generate(mut self, noise: &impl NoiseFn<f64, 2>) -> Chunk16 {
        // for min/max on this chunk, generate a 2d noise map
        for x in self.origin.x * CHUNK_SIZE as i32..(self.origin.x + 1) * CHUNK_SIZE as i32 {
            for z in self.origin.z * CHUNK_SIZE as i32..(self.origin.z + 1) * CHUNK_SIZE as i32 {
                // we are rendering chunk on the -1 y so we shift down by 1
                let height =
                    noise.get([x as f64 * NOISE_SCALE, z as f64 * NOISE_SCALE]) * 16. + 8.0 - 16.0;
                info!("height for {} {}: {}", x, z, height);

                for y in self.origin.y * CHUNK_SIZE as i32..(self.origin.y + 1) * CHUNK_SIZE as i32
                {
                    let index = Self::xyz_to_index(
                        (x - self.origin.x * CHUNK_SIZE as i32) as u8,
                        (y - self.origin.y * CHUNK_SIZE as i32) as u8,
                        (z - self.origin.z * CHUNK_SIZE as i32) as u8,
                    );
                    info!("y: {}", y);

                    self.blocks[index].is_active = false;

                    // if y is at the height, make it grass
                    if (y as f64 - height) < 1.0 && (y as f64 - height) > 0.0 {
                        self.blocks[index].t = BlockType::Grass;
                        self.blocks[index].is_active = true;
                    } else if (y as f64) < height && (y as f64) > height - 2. {
                        self.blocks[index].t = BlockType::Dirt;
                        self.blocks[index].is_active = true;
                    } else if (y as f64) < height {
                        self.blocks[index].t = BlockType::Stone;
                        self.blocks[index].is_active = true;
                    }
                }
            }
        }
        self
    }

    // get all blocks that intersect with a ray
    pub fn collision_check(&mut self, d0: cgmath::Point3<f32>, dir: cgmath::Vector3<f32>) {
        let mut candidates = Vec::new();
        let location = self.origin;
        for block in self.blocks.iter_mut() {
            if block.is_active {
                if let Some(hit) = block.intersect_ray(d0, dir) {
                    let position = d0 + dir * hit[0];
                    candidates.push((block, hit));
                } else {
                    block.is_selected = false;
                }
            }
        }
        candidates.sort_by(|a, b| a.1[0].partial_cmp(&b.1[0]).unwrap());
        if let Some(c) = candidates.first_mut() {
            c.0.is_selected = true;
        };
        for c in candidates.iter_mut().skip(1) {
            c.0.is_selected = false;
        }
    }
}

impl Render for Chunk16 {
    fn render(&self) -> Vec<Instance> {
        let chunk_offset = self.origin.cast::<f32>().unwrap() * BLOCK_SIZE * CHUNK_SIZE as f32;
        self.blocks
            .iter()
            .enumerate()
            .filter(|(index, block)| block.is_active)
            .map(|(index, block)| Instance {
                position: block.origin,
                block_type: block.t,
                is_selected: block.is_selected,
                ..Default::default()
            })
            .collect()
    }
}

impl Chunk16 {
    fn xyz_to_index(x: u8, y: u8, z: u8) -> usize {
        let (x, y, z) = (x as usize, y as usize, z as usize);
        x + y * CHUNK_SIZE + z * CHUNK_SIZE * CHUNK_SIZE
    }

    fn index_to_xyz(index: usize) -> (u8, u8, u8) {
        let x = index % CHUNK_SIZE;
        let y = (index / CHUNK_SIZE) % CHUNK_SIZE;
        let z = index / CHUNK_SIZE / CHUNK_SIZE;
        (x as u8, y as u8, z as u8)
    }

    pub fn culled_render(&self, frustum: &Frustrum) -> Vec<Instance> {
        self.blocks
            .iter()
            .enumerate()
            .map(|(index, block)| (Self::index_to_xyz(index), block))
            .filter(|(_, block)| block.is_active)
            .filter(|(_, block)| frustum.contains(&block.aabb()))
            .filter(|(pos, _block)| {
                // check if block is occluded by another block
                let (x, y, z) = (pos.0, pos.1, pos.2);
                // check bounds
                if x == 0
                    || y == 0
                    || z == 0
                    || x == CHUNK_SIZE as u8 - 1
                    || y == CHUNK_SIZE as u8 - 1
                    || z == CHUNK_SIZE as u8 - 1
                {
                    return true;
                }
                // inner block
                let neighbors = [
                    self.get_block(x + 1, y, z),
                    self.get_block(x - 1, y, z),
                    self.get_block(x, y + 1, z),
                    self.get_block(x, y - 1, z),
                    self.get_block(x, y, z + 1),
                    self.get_block(x, y, z - 1),
                ];
                !neighbors.iter().all(|n| n.is_active) // all neighbors are active, so this block is occluded, return false
            })
            .map(|(_, block)| Instance {
                position: block.origin,
                block_type: block.t,
                is_selected: block.is_selected,
                ..Default::default()
            })
            .collect()
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

impl Aabb for Chunk16 {
    fn min(&self) -> Point3<f32> {
        self.origin.cast::<f32>().unwrap() * BLOCK_SIZE * CHUNK_SIZE as f32
    }

    fn max(&self) -> Point3<f32> {
        self.origin.cast::<f32>().unwrap() * BLOCK_SIZE * CHUNK_SIZE as f32
            + Vector3::new(
                BLOCK_SIZE * CHUNK_SIZE as f32,
                BLOCK_SIZE * CHUNK_SIZE as f32,
                BLOCK_SIZE * CHUNK_SIZE as f32,
            )
    }
}
