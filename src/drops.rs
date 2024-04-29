use cgmath::{EuclideanSpace, Matrix4, Point3, Rotation3, Vector3, Zero};
use rand::Rng;

use crate::{
    aabb::AabbBounds, block::BlockType, physics::{KinematicBody, KinematicBodyState}
};

pub struct Drop {
    pub physics_state: KinematicBodyState,
    pub rotation: cgmath::Quaternion<f32>,
    pub block_type: BlockType,
}

const COLLIDER_SIZE: f32 = 0.35;

impl KinematicBody for Drop {
    fn state(&mut self) -> &mut KinematicBodyState {
        &mut self.physics_state
    }

    fn collider(&self) -> AabbBounds {
        AabbBounds::new(
            self.physics_state.position - Vector3::new(COLLIDER_SIZE, COLLIDER_SIZE, COLLIDER_SIZE),
            self.physics_state.position + Vector3::new(COLLIDER_SIZE, COLLIDER_SIZE, COLLIDER_SIZE),
        )
    }
}

impl Drop {
    pub fn from_block(origin: Point3<f32>, block_type: BlockType) -> Self {
        let physics_state = KinematicBodyState {
            velocity: Vector3::zero(),
            position: origin,
            grounded: false,
        };
        Self {
            physics_state,
            rotation: cgmath::Quaternion::from_angle_y(cgmath::Deg(0.01)),
            block_type,
        }
    }

    pub fn quaternion(&self) -> cgmath::Matrix4<f32> {
        // we scale the normal instance to 0.25, apply rotation and position
        Matrix4::from_translation(self.physics_state.position.to_vec())
            * Matrix4::from(self.rotation)
            * Matrix4::from_scale(0.25)
    }
}
