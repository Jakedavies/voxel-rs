use cgmath::{ElementWise, Vector3, Point3};
use log::info;

use crate::{aabb::Aabb, drops::Drop, physics::KinematicBodyState};

pub const BLOCK_SIZE: f32 = 2.0;

#[derive(Copy, Clone, Debug, Default, Eq, Hash, PartialEq)]
pub enum BlockType {
    #[default]
    Dirt,
    Grass,
    Stone,
    Wood,
    Water,
    Ore,
}

impl Into<u32> for BlockType {
    fn into(self) -> u32 {
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
    pub coords: cgmath::Point3<i32>,
}

const HALF_BLOCK_SIZE: f32 = BLOCK_SIZE / 2.0;
impl Block {
    pub fn new(coords: Point3<i32>) -> Self {
        Self {
            is_active: false,
            is_selected: false,
            t: BlockType::Stone,
            coords,
        }
    }

    pub fn drop(&self) -> Drop {
        Drop::from_block(self.origin(), self.t)
    }

    pub fn origin(&self) -> Point3<f32> {
        Point3::new(
            self.coords.x as f32 * BLOCK_SIZE + HALF_BLOCK_SIZE,
            self.coords.y as f32 * BLOCK_SIZE + HALF_BLOCK_SIZE,
            self.coords.z as f32 * BLOCK_SIZE + HALF_BLOCK_SIZE
        )
    }
}

impl Aabb for Block {
    fn min(&self) -> Point3<f32> {
        self.origin() - Vector3::new(HALF_BLOCK_SIZE, HALF_BLOCK_SIZE, HALF_BLOCK_SIZE)
    }
    fn max(&self) -> Point3<f32> {
        self.origin() + Vector3::new(HALF_BLOCK_SIZE, HALF_BLOCK_SIZE, HALF_BLOCK_SIZE)
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_block_min_max() {
        let block = Block::new(Point3::new(0, 0, 0));
        assert_eq!(block.min(), Point3::new(-1.0, -1.0, -1.0));
        assert_eq!(block.max(), Point3::new(1.0, 1.0, 1.0));
    }

    #[test]
    fn test_block_min_max_2() {
        let block = Block::new(Point3::new(16, 16, 16));
        assert_eq!(block.min(), Point3::new(15.0, 15.0, 15.0));
        assert_eq!(block.max(), Point3::new(17.0, 17.0, 17.0));
    }

    #[test]
    fn test_block_coords() {
        let block = Block::new(Point3::new(1, 1, 1));
        assert_eq!(block.coords, (1, 1, 1).into());
    }

    #[test]
    fn test_block_coords_2() {
        let block = Block::new(Point3::new(3, 1, 1));
        assert_eq!(block.coords, (1, 0, 0).into());
    }

    #[test]
    fn test_block_coords_3() {
        let block = Block::new(Point3::new(3, -1, 1));
        assert_eq!(block.coords, (1, -1, 0).into());
    }

}
