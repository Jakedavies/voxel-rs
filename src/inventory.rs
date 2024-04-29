use std::collections::HashMap;

use crate::block::BlockType;


pub struct Inventory {
    pub items: HashMap<BlockType, u32>,
}

impl Inventory {
    pub fn new() -> Self {
        let mut items = HashMap::new();
        Self { items }
    }

    pub fn add(&mut self, block_type: BlockType) {
        let count = self.items.entry(block_type).or_insert(0);
        *count += 1;
    }

    pub fn remove(&mut self, block_type: BlockType) {
        let count = self.items.entry(block_type).or_insert(0);
        if *count > 0 {
            *count -= 1;
        }
    }

    pub fn count(&self, block_type: BlockType) -> u32 {
        *self.items.get(&block_type).unwrap_or(&0)
    }
}
