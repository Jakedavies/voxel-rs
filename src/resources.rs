use std::io::{BufReader, Cursor};
use wgpu::util::DeviceExt;

use crate::{model, texture};

pub async fn load_texture_from_bytes(
    file_name: &str,
    bytes: &[u8],
    device: &wgpu::Device,
    queue: &wgpu::Queue,
) -> anyhow::Result<texture::Texture> {
    texture::Texture::from_bytes(device, queue, &bytes, file_name)
}
