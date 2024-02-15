use cgmath::{ElementWise, Vector3, Point3};
use log::info;

use crate::{aabb::Aabb, Instance};

pub const BLOCK_SIZE: f32 = 2.0;

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

#[derive(Copy, Clone)]
pub struct Block {
    pub is_active: bool,
    pub is_selected: bool,
    pub t: BlockType,
    chunk_space_origin: cgmath::Point3<u8>,
}

impl Block {
    pub fn new(position: Point3<u8>) -> Self {
        Self {
            is_active: true,
            is_selected: false,
            t: BlockType::Stone,
            chunk_space_origin: position,
        }
    }

    pub fn origin(&self) -> Point3<f32> {
        self.chunk_space_origin.cast::<f32>().unwrap() * BLOCK_SIZE
    }
}

impl Aabb for Block {
    fn min(&self) -> Point3<f32> {
        self.origin() - Vector3::new(BLOCK_SIZE / 2.0, BLOCK_SIZE / 2.0, BLOCK_SIZE / 2.0)
    }
    fn max(&self) -> Point3<f32> {
        self.origin() + Vector3::new(BLOCK_SIZE / 2.0, BLOCK_SIZE / 2.0, BLOCK_SIZE / 2.0)
    }
}

pub trait Render {
    fn render(&self) -> Vec<Instance>;
}


