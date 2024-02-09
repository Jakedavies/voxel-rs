use crate::Instance;

#[derive(Copy, Clone, Debug)]
pub enum BlockType {
    Dirt,
    Grass,
    Stone,
    Wood,
    Water,
    Ore,
}

#[derive(Copy, Clone, Debug)]
pub struct Block {
    is_active: bool,
    t: BlockType,
}

impl Default for Block {
    fn default() -> Self {
        Self {
            is_active: true,
            t: BlockType::Stone,
        }
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
    fn set_block(&mut self, x: u8, y: u8, z: u8, block: Block);
}

pub struct Chunk16 {
    chunk_location: cgmath::Vector3<i32>,
    blocks: [Block; 16 * 16 * 16],
}

impl Chunk16 {
    // get all blocks that intersect with a ray
    pub fn query(&self, d0: cgmath::Point3<f32>, dir: cgmath::Vector3<f32>) -> Option<Block> {
        let mut t = 0.0;
        let mut current = d0;
        let step = 0.1;
        None
    }
}

impl Default for Chunk16 {
    fn default() -> Self {
        let mut s = Self {
            blocks: [Block::default(); 16 * 16 * 16],
            chunk_location: cgmath::Vector3::new(0, 0, 0),
        };

        for i in 0..16 {
            for j in 0..16 {
                s.set_block(
                    i,
                    15,
                    j,
                    Block {
                        is_active: true,
                        t: BlockType::Grass,
                    },
                );
                s.set_block(
                    i,
                    14,
                    j,
                    Block {
                        is_active: true,
                        t: BlockType::Dirt,
                    },
                );
                s.set_block(
                    i,
                    13,
                    j,
                    Block {
                        is_active: true,
                        t: BlockType::Dirt,
                    },
                );
            }
        }

        // randomly select some blocks in lower layers to turn to ore
        for _ in 0..100 {
            let x = rand::random::<u8>() % 16;
            let y = rand::random::<u8>() % 13;
            let z = rand::random::<u8>() % 16;
            s.set_block(
                x,
                y,
                z,
                Block {
                    is_active: true,
                    t: BlockType::Ore,
                },
            );
        }
        s
    }
}
const BLOCK_SIZE: f32 = 2.0;

impl Render for Chunk16 {
    fn render(&self) -> Vec<Instance> {
        let mut instances = Vec::new();
        for x in 0..16 {
            for y in 0..16 {
                for z in 0..16 {
                    let block = self.get_block(x, y, z);
                    if block.is_active {
                        let x = BLOCK_SIZE * (x as f32);
                        let y = BLOCK_SIZE * (y as f32);
                        let z = BLOCK_SIZE * (z as f32);
                        instances.push(Instance {
                            position: cgmath::Vector3 {
                                x,
                                y,
                                z,
                            },
                            block_type: block.t,
                            ..Default::default()
                        });
                    }
                }
            }
        }
        instances
    }
}

impl Chunk16 {
    fn xyz_to_index(x: u8, y: u8, z: u8) -> usize {
        let (x, y, z) = (x as usize, y as usize, z as usize);
        x + y * 16 + z * 16 * 16
    }
}

impl Chunk for Chunk16 {
    fn get_block(&self, x: u8, y: u8, z: u8) -> &Block {
        let index = Self::xyz_to_index(x, y, z);
        &self.blocks[index]
    }

    fn set_block(&mut self, x: u8, y: u8, z: u8, block: Block) {
        let index = Self::xyz_to_index(x, y, z);
        self.blocks[index] = block;
    }
}
