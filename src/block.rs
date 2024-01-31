use crate::Instance;

#[derive(Copy, Clone, Debug)]
pub enum BlockType {
    Dirt,
    Grass,
    Stone,
    Wood,
    Water,
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

pub trait Render {
    fn render(&self) -> Vec<Instance>;
}

pub trait Chunk {
    fn get_block(&self, x: u8, y: u8, z: u8) -> &Block;
    fn set_block(&mut self, x: u8, y: u8, z: u8, block: Block);
}

pub struct Chunk16 {
    blocks: [Block; 16 * 16 * 16],
}

impl Default for Chunk16 {
    fn default() -> Self {
        let mut s = Self {
            blocks: [Block::default(); 16 * 16 * 16],
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
        s
    }
}
const SPACE_BETWEEN: f32 = 2.0;
const INSTANCE_DISPLACEMENT: cgmath::Vector3<f32> = cgmath::Vector3 {
    x: 1.0,
    y: 1.0,
    z: 1.0,
};

impl Render for Chunk16 {
    fn render(&self) -> Vec<Instance> {
        let mut instances = Vec::new();
        for x in 0..16 {
            for y in 0..16 {
                for z in 0..16 {
                    let block = self.get_block(x, y, z);
                    if block.is_active {
                        let x = SPACE_BETWEEN * (x as f32 - 16 as f32 / 2.0);
                        let y = SPACE_BETWEEN * (y as f32 - 16 as f32 / 2.0);
                        let z = SPACE_BETWEEN * (z as f32 - 16 as f32 / 2.0);
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
