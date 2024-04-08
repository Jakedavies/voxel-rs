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

impl Into<u16> for BlockType {
    fn into(self) -> u16 {
        match self {
            BlockType::Stone => 0,
            BlockType::Dirt => 1 << 8 | 1,
            BlockType::Grass => 3 << 8 | 2, // grass uses 3 for the top and 2 for the other sides
            BlockType::Ore => 16 << 8 | 16,
            _ => 255,
        }
    }
}

#[derive(Copy, Clone, Debug)]
pub struct Block {
    pub is_active: bool,
    pub is_selected: bool,
    pub t: BlockType,
    pub origin: cgmath::Point3<f32>,
}

impl Block {
    pub fn new(origin: Point3<f32>) -> Self {
        Self {
            is_active: false,
            is_selected: false,
            t: BlockType::Stone,
            origin
        }
    }
}

impl Aabb for Block {
    fn min(&self) -> Point3<f32> {
        self.origin - Vector3::new(BLOCK_SIZE / 2.0, BLOCK_SIZE / 2.0, BLOCK_SIZE / 2.0)
    }
    fn max(&self) -> Point3<f32> {
        self.origin + Vector3::new(BLOCK_SIZE / 2.0, BLOCK_SIZE / 2.0, BLOCK_SIZE / 2.0)
    }
}

pub trait Render {
    fn render(&self) -> Vec<Instance>;
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_block_min_max() {
        let block = Block::new(Point3::new(0.0, 0.0, 0.0));
        assert_eq!(block.min(), Point3::new(-1.0, -1.0, -1.0));
        assert_eq!(block.max(), Point3::new(1.0, 1.0, 1.0));
    }

    #[test]
    fn test_block_min_max_2() {
        let block = Block::new(Point3::new(16.0, 16.0, 16.0));
        assert_eq!(block.min(), Point3::new(15.0, 15.0, 15.0));
        assert_eq!(block.max(), Point3::new(17.0, 17.0, 17.0));
    }
}
