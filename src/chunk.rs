use crate::{
    block::{Block, Render, BLOCK_SIZE},
    Instance, aabb::Aabb,
};
use cgmath::{prelude::*, Vector3, Point3};

const CHUNK_SIZE: usize = 16;

pub trait Chunk {
    fn get_block(&self, x: u8, y: u8, z: u8) -> &Block;
    fn get_block_mut(&mut self, x: u8, y: u8, z: u8) -> &mut Block;
    fn set_block(&mut self, x: u8, y: u8, z: u8, block: Block);
}

pub struct Chunk16 {
    pub origin: cgmath::Point3<i32>,
    blocks: [Block; CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE],
}

impl Chunk16 {
    pub fn new(x: i32, y: i32, z: i32) -> Self {
        Self {
            origin: cgmath::Point3::new(x, y, z),
            ..Default::default()
        }
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

impl Default for Chunk16 {
    fn default() -> Self {
        let mut s = Self {
            blocks: [Block::new(Point3::new(0, 0, 0)); CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE],
            origin: cgmath::Point3::new(0, 0, 0),
        };
        // set all the positions for blocks
        for i in 0..CHUNK_SIZE as u8 {
            for j in 0..CHUNK_SIZE as u8 {
                for k in 0..CHUNK_SIZE as u8 {
                    s.set_block(i, j, k, Block::new(Point3::new(i, j, k)));
                }
            }
        }
        s
    }
}

impl Render for Chunk16 {
    fn render(&self) -> Vec<Instance> {
        let chunk_offset = self.origin.cast::<f32>().unwrap() * BLOCK_SIZE * CHUNK_SIZE as f32;
        self.blocks
            .iter()
            .filter(|block| block.is_active)
            .map(|block| {
                Instance {
                    position: block.origin() + chunk_offset.to_vec(),
                    block_type: block.t,
                    is_selected: block.is_selected,
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
