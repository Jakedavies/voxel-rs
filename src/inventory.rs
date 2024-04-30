use std::collections::HashMap;

use crate::block::BlockType;

pub struct InventoryItem {
    pub block_type: BlockType,
    pub count: u32,
}

pub struct Inventory {
    pub items: Vec<InventoryItem>,
    pub selected_index: usize,
}

impl Inventory {
    pub fn new() -> Self {
        Inventory {
            items: vec![],
            selected_index: 0,
        }
    }

    pub fn add(&mut self, block_type: BlockType) {
        let mut found = false;
        for item in self.items.iter_mut() {
            if item.block_type == block_type {
                item.count += 1;
                found = true;
                break;
            }
        }

        if !found {
            self.items.push(InventoryItem {
                block_type,
                count: 1,
            });
        }
    }

    pub fn remove(&mut self, block_type: BlockType) {
        for i in 0..self.items.len() {
            if self.items[i].block_type == block_type {
                if self.items[i].count > 1 {
                    self.items[i].count -= 1;
                } else {
                    self.items.remove(i);
                }
                break;
            }
        }
    }

    pub fn count(&self, block_type: BlockType) -> u32 {
        for item in self.items.iter() {
            if item.block_type == block_type {
                return item.count;
            }
        }
        0
    }
}
