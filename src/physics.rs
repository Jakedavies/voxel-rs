use std::time::Duration;

use cgmath::num_traits::{Float, Signed};
use log::info;

use crate::{
    aabb::{Aabb, AabbBounds},
    block::Block,
    chunk::{Chunk16, ChunkWithMesh},
};

const GRAVITY: f32 = 9.8 * 2.0; // our blocks are 2.0 wide, gravity feels funky unless scaled

#[derive(Debug)]
pub struct KinematicBodyState {
    pub velocity: cgmath::Vector3<f32>,
    pub position: cgmath::Point3<f32>,
    pub grounded: bool,
}

impl KinematicBodyState {
    pub fn new() -> Self {
        Self {
            velocity: cgmath::Vector3::new(0.0, 0.0, 0.0),
            position: cgmath::Point3::new(0.0, 0.0, 0.0),
            grounded: false,
        }
    }
}

pub trait KinematicBody {
    fn state(&mut self) -> &mut KinematicBodyState;
    fn collider(&self) -> AabbBounds;
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
pub struct CubeCollider {
    pub height: f32,
    pub width: f32,
    pub origin: cgmath::Point3<f32>,
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
fn collide_chunks<'a>(
    chunks: impl Iterator<Item = &'a Chunk16> + Clone,
    collision_body: &impl Aabb,
) -> Option<cgmath::Vector3<f32>> {
    // build an aabb from the current position and future position to prune chunks
    // test this larger aabb against all the chunks to see if any are relevant
    let blocks = chunks
        .filter(|chunk| chunk.aabb().intersects(collision_body))
        .flat_map(|chunk| chunk.blocks.iter())
        .filter(|block| block.aabb().intersects(collision_body))
        .filter(|block| block.is_active);

    let mut collision_reverse = cgmath::Vector3::new(0.0, 0.0, 0.0);
    if blocks.clone().count() == 0 {
        return None;
    }
    // for each intersection axis, take the max reverse direction
    for block in blocks {
        if let Some(collision_info) = collision_body.aabb().intersection(&block.aabb()) {
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

const FRICTION_ACCEL: f32 = 30.;

pub fn update_body<'a>(
    body: &mut impl KinematicBody,
    chunks: impl Iterator<Item = &'a Chunk16> + Clone,
    dt: Duration,
) {
    {
        let physics_state = body.state();
        // apply gravity to velocity
        physics_state.velocity.y -= GRAVITY * dt.as_secs_f32();

        // apply velocity to position
        physics_state.position.x += physics_state.velocity.x * dt.as_secs_f32();
        physics_state.position.y += physics_state.velocity.y * dt.as_secs_f32();
        physics_state.position.z += physics_state.velocity.z * dt.as_secs_f32();
        // f = ma
        // a = f / m
        // apply a friction force in the opposite directiuon of the current velocity
        // to slow down the object
        if physics_state.grounded {
            if (physics_state.velocity.x).abs() > 0.1 {
                physics_state.velocity.x -= (physics_state.velocity.x
                    / physics_state.velocity.x.abs())
                    * FRICTION_ACCEL
                    * dt.as_secs_f32();
            } else {
                physics_state.velocity.x = 0.0;
            }
            if (physics_state.velocity.z).abs() > 0.1 {
                physics_state.velocity.z -= (physics_state.velocity.z
                    / physics_state.velocity.z.abs())
                    * FRICTION_ACCEL
                    * dt.as_secs_f32();
            } else {
                physics_state.velocity.z = 0.0;
            }
        }
    }

    let collider = body.collider();
    let physics_state = body.state();
    // collide with chunks
    if let Some(collision_vector) = collide_chunks(chunks.clone(), &collider) {
        // zero out the velocity in the direction of the collision
        // update the position with the inverse of the collision
        if collision_vector.x != 0.0 {
            physics_state.velocity.x = 0.0;
            physics_state.position.x -= collision_vector.x;
        }

        if collision_vector.y != 0.0 {
            physics_state.velocity.y = 0.0;
            physics_state.position.y -= collision_vector.y;
        }

        if collision_vector.z != 0.0 {
            physics_state.velocity.z = 0.0;
            physics_state.position.z -= collision_vector.z;
        }
    }

    // test if we are grounded, extend the collider by a small amount in the y direction
    let mut grounded_collider = collider;
    grounded_collider.min.y -= 0.05;
    if let Some(collision_vector) = collide_chunks(chunks, &grounded_collider) {
        if collision_vector.y != 0.0 {
            physics_state.grounded = true;
        } else {
            physics_state.grounded = false;
        }
    } else {
        physics_state.grounded = false;
    }
}

pub struct ChunkRaycastResult<'a> {
    pub chunk: &'a mut Chunk16,
    pub intersection: [f32; 2],
    block_index: usize,
}

impl ChunkRaycastResult<'_> {
    pub fn block(&self) -> &Block {
        &self.chunk.blocks[self.block_index]
    }
    pub fn block_mut(&mut self) -> &mut Block {
        self.chunk.dirty = true;
        &mut self.chunk.blocks[self.block_index]
    }
}

// TODO: _mega_spaghetti
pub fn cast_ray_chunks_mut<'a>(
    origin: &cgmath::Point3<f32>,
    direction: &cgmath::Vector3<f32>,
    chunks: impl Iterator<Item = &'a mut Chunk16>,
) -> Option<ChunkRaycastResult<'a>> {
    let valid_chunks =
        chunks.filter(|chunk| chunk.aabb().intersect_ray(origin, direction).is_some());

    valid_chunks
        .filter_map(|chunk| {
            let block_intersections = chunk
                .blocks
                .iter_mut()
                .enumerate()
                .filter(|(_index, block)| block.is_active)
                .filter_map(|(index, block)| {
                    block
                        .aabb()
                        .intersect_ray(origin, direction)
                        .map(|intersection| (index, intersection))
                })
                .min_by(|a, b| {
                    a.1[0]
                        .partial_cmp(&b.1[0])
                        .unwrap_or(std::cmp::Ordering::Equal)
                });

            block_intersections.map(|(block_index, intersection)| ChunkRaycastResult {
                chunk,
                block_index,
                intersection,
            })
        })
        .min_by(|a, b| {
            a.intersection[0]
                .partial_cmp(&b.intersection[0])
                .unwrap_or(std::cmp::Ordering::Equal)
        })
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
        let result = collide_chunks(chunks.iter(), &cube_collider);
        assert_eq!(result, None)
    }

    #[test]
    fn test_collide_chunks() {
        let mut chunk = Chunk16::new(0, 0, 0);
        chunk.blocks[0].is_active = true;
        let chunks = vec![chunk];
        let cube_collider = CubeCollider {
            height: 1.0,
            width: 1.0,
            origin: cgmath::Point3::new(-0.4, 0.0, 0.0),
        };
        let result = collide_chunks(chunks.iter(), &cube_collider);
        println!("{:?}", result);

        // we expect a collision in the x axis, accounting for floating point error
        assert!(AbsDiffEq::abs_diff_eq(
            &result.unwrap(),
            &cgmath::Vector3::new(0.1, 0.0, 0.0),
            1e-6
        ));
    }

    #[test]
    fn test_raycasting() {
        let mut chunk = Chunk16::new(0, 0, 0);
        chunk.blocks[0].is_active = true;
        let mut chunks = vec![chunk];
        let origin = cgmath::Point3::new(-0.1, 0.5, 0.5);
        let direction = cgmath::Vector3::new(1.0, 0.0, 0.0);
        let result = cast_ray_chunks_mut(&origin, &direction, chunks.iter_mut());
        assert_eq!(
            result.unwrap().block().origin,
            cgmath::Point3::new(1.0, 1.0, 1.0)
        );
    }

    #[test]
    fn test_chunk_collision_looking_up() {
        let mut chunk = Chunk16::new(0, -1, -1);
        for block in chunk.blocks.iter_mut() {
            block.is_active = true;
        }

        let look_direction = cgmath::Vector3::new(0.0, 1.0, 0.0);
        let origin = cgmath::Point3::new(9.0, 3.0, -16.0);
        let mut chunks = vec![chunk];
        let result = cast_ray_chunks_mut(&origin, &look_direction, chunks.iter_mut());
        // looking straight up, no expected collision
        assert!(result.is_none());
    }

    #[test]
    fn test_chunk_collision_looking_down() {
        let mut chunk = Chunk16::new(0, -1, -1);
        for block in chunk.blocks.iter_mut() {
            block.is_active = true;
        }

        println!("{:?}", chunk.aabb());

        let look_direction = cgmath::Vector3::new(0.0, -1.0, 0.0);
        let origin = cgmath::Point3::new(9.0, 2.0, -17.0);
        let mut chunks = vec![chunk];
        let result = cast_ray_chunks_mut(&origin, &look_direction, chunks.iter_mut());
        // looking straight up, no expected collision
        assert_eq!(
            result.unwrap().block().origin,
            cgmath::Point3::new(9.0, -1.0, -17.0)
        );
    }
}
