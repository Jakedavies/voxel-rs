use std::time::Duration;

use cgmath::num_traits::Float;
use wgpu::naga::Block;

use crate::{aabb::Aabb, chunk::Chunk16};

const GRAVITY: f32 = 9.8;

pub trait KinematicBody {
    fn velocity(&mut self) -> &mut cgmath::Vector3<f32>;
    fn position(&mut self) -> &mut cgmath::Point3<f32>;
}

#[derive(Debug, PartialEq)]
pub struct CollisionInfo {
    pub normal: cgmath::Vector3<f32>,
    pub penetration: f32,
}

impl CollisionInfo {
    pub fn new(normal: cgmath::Vector3<f32>, penetration: f32) -> Self {
        Self {
            normal,
            penetration,
        }
    }

    pub fn vector(&self) -> cgmath::Vector3<f32> {
        self.normal * self.penetration
    }
}

#[derive(Clone)]
struct CubeCollider {
    height: f32,
    width: f32,
    origin: cgmath::Point3<f32>,
}

impl Aabb for CubeCollider {
    fn min(&self) -> cgmath::Point3<f32> {
        self.origin - cgmath::Vector3::new(self.width / 2.0, self.height / 2.0, self.width / 2.0)
    }

    fn max(&self) -> cgmath::Point3<f32> {
        self.origin + cgmath::Vector3::new(self.width / 2.0, self.height / 2.0, self.width / 2.0)
    }
}

// this function will take a list of chunks and a cube collider and return the reverse direction of the collision
fn collide_chunks(
    chunks: &Vec<Chunk16>,
    cube_collider: &CubeCollider,
    velocity: cgmath::Vector3<f32>,
) -> Option<cgmath::Vector3<f32>> {
    // build an aabb from the current position and future position to prune chunks
    let old_aabb = cube_collider.aabb();
    let mut new_position = (*cube_collider).clone();
    new_position.origin += velocity;

    // test this larger aabb against all the chunks to see if any are relevant
    let blocks = chunks
        .iter()
        .filter(|chunk| (*chunk).aabb().intersects(&new_position))
        .flat_map(|chunk| chunk.blocks.iter());

    let mut collision_reverse = cgmath::Vector3::new(0.0, 0.0, 0.0);
    if blocks.clone().count() == 0 {
        return None;
    }
    println!("new position: {:?}", new_position.aabb());
    // for each intersection axis, take the max reverse direction
    for block in blocks {
        if let Some(collision_info) = new_position.aabb().intersection(&block.aabb()) {
            println!("collision info: {:?}, {:?}", block.aabb(), collision_info);
            // abs value of penetration in each axis
            if collision_info.vector().x.abs() > collision_reverse.x.abs() {
                collision_reverse.x = collision_info.vector().x;
            }
            if collision_info.vector().y.abs() > collision_reverse.y.abs() {
                collision_reverse.y = collision_info.vector().y;
            }
            if collision_info.vector().z.abs() > collision_reverse.z.abs() {
                collision_reverse.z = collision_info.vector().z;
            }
        }
    }

    Some(collision_reverse)
}

pub fn update_body(
    body: &mut impl KinematicBody,
    chunks: &Vec<Chunk16>,
    dt: Duration,
) {
    // apply gravity to velocity
    body.velocity().y -= GRAVITY * dt.as_secs_f32();

    // apply velocity to position
    body.position().x += body.velocity().x * dt.as_secs_f32();
    body.position().y += body.velocity().y * dt.as_secs_f32();

    // collide with chunks
    if let Some(collision_vector) = collide_chunks(
        chunks,
        &CubeCollider {
            height: 1.0,
            width: 1.0,
            origin: *body.position(),
        },
        *body.velocity(),
    ) {
        // zero out the velocity in the direction of the collision
        // update the position with the inverse of the collision
        if collision_vector.x != 0.0 {
            body.velocity().x = 0.0;
            body.position().x -= collision_vector.x;
        }

        if collision_vector.y != 0.0 {
            body.velocity().y = 0.0;
            body.position().y -= collision_vector.y;
        }

        if collision_vector.z != 0.0 {
            body.velocity().z = 0.0;
            body.position().z -= collision_vector.z;
        }
    }
}


#[cfg(test)]
mod tests {
    use cgmath::AbsDiffEq;

    use super::*;

    #[test]
    fn test_collide_chunks_no_expected_collision() {
        let chunk = Chunk16::new(0, 0, 0);
        let chunks = vec![chunk];
        let cube_collider = CubeCollider {
            height: 1.0,
            width: 1.0,
            origin: cgmath::Point3::new(-1.0, 0.0, 0.0),
        };
        let velocity = cgmath::Vector3::new(-1.0, 0.0, 0.0);
        let result = collide_chunks(&chunks, &cube_collider, velocity);
        assert_eq!(result, None)
    }

    #[test]
    fn test_collide_chunks() {
        let chunk = Chunk16::new(0, 0, 0);
        println!("Chunk: {:?}", chunk.blocks.first().unwrap().aabb());
        let chunks = vec![chunk];
        let cube_collider = CubeCollider {
            height: 1.0,
            width: 1.0,
            origin: cgmath::Point3::new(-0.5, 2.0, 2.0),
        };
        let velocity = cgmath::Vector3::new(0.1, 0.0, 0.0);
        let result = collide_chunks(&chunks, &cube_collider, velocity);

        // we expect a collision in the x axis, accounting for floating point error
        assert!(AbsDiffEq::abs_diff_eq(&result.unwrap(), &cgmath::Vector3::new(0.1, 0.0, 0.0), 1e-6));
    }
}
