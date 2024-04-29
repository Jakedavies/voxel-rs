use cgmath::prelude::*;
use cgmath::*;
use cgmath::*;
use log::info;
use std::{f32::consts::FRAC_PI_2, time::Duration};
use winit::dpi::PhysicalPosition;
use winit::event::*;
use winit::{
    dpi::PhysicalSize,
    event::WindowEvent,
    event::*,
    keyboard::{Key, KeyCode, NamedKey, PhysicalKey},
};

use crate::aabb::{Aabb, AabbBounds};
use crate::physics::{CubeCollider, KinematicBody, KinematicBodyState};
use crate::GRAVITY;

#[rustfmt::skip]
pub const OPENGL_TO_WGPU_MATRIX: cgmath::Matrix4<f32> = cgmath::Matrix4::new(
    1.0, 0.0, 0.0, 0.0,
    0.0, 1.0, 0.0, 0.0,
    0.0, 0.0, 0.5, 0.5,
    0.0, 0.0, 0.0, 1.0,
);

const SAFE_FRAC_PI_2: f32 = FRAC_PI_2 - 0.0001;

#[derive(Debug)]
pub struct Camera {
    yaw: Rad<f32>,
    pitch: Rad<f32>,
    collider: Vector3<f32>,
    pub physics_state: KinematicBodyState,
}

pub struct Plane {
    pub normal: Vector3<f32>,
    pub distance: f32,
}

pub struct Frustrum {
    top: Plane,
    bottom: Plane,
    left: Plane,
    right: Plane,
    near: Plane,
    far: Plane,
}

impl Frustrum {
    fn contains_point(&self, point: &Point3<f32>) -> bool {
        let planes = [
            &self.top,
            &self.bottom,
            &self.left,
            &self.right,
            &self.near,
            &self.far,
        ];

        for plane in &planes {
            if plane.normal.dot(point.to_vec()) + plane.distance <= 0.0 {
                return false;
            }
        }

        true
    }

    pub fn contains(&self, other: &dyn Aabb) -> bool {
        let points = [
            other.min(),
            Point3::new(other.min().x, other.min().y, other.max().z),
            Point3::new(other.min().x, other.max().y, other.min().z),
            Point3::new(other.min().x, other.max().y, other.max().z),
            Point3::new(other.max().x, other.min().y, other.min().z),
            Point3::new(other.max().x, other.min().y, other.max().z),
            Point3::new(other.max().x, other.max().y, other.min().z),
            other.max(),
        ];

        for point in &points {
            if self.contains_point(point) {
                return true;
            }
        }

        false
    }
}

impl Camera {
    pub fn new<V: Into<Point3<f32>>, Y: Into<Rad<f32>>, P: Into<Rad<f32>>>(
        position: V,
        yaw: Y,
        pitch: P,
    ) -> Self {
        Self {
            yaw: yaw.into(),
            pitch: pitch.into(),
            collider: Vector3::new(0.8, 2.0, 0.8),
            physics_state: {
                let mut state = KinematicBodyState::new();
                state.position = position.into();
                state
            },
        }
    }

    pub fn calc_matrix(&self) -> Matrix4<f32> {
        let (sin_pitch, cos_pitch) = self.pitch.0.sin_cos();
        let (sin_yaw, cos_yaw) = self.yaw.0.sin_cos();

        Matrix4::look_to_rh(
            self.physics_state.position,
            Vector3::new(cos_pitch * cos_yaw, sin_pitch, cos_pitch * sin_yaw).normalize(),
            Vector3::unit_y(),
        )
    }

    pub fn look_direction(&self) -> Vector3<f32> {
        let (yaw_sin, yaw_cos) = self.yaw.0.sin_cos();
        let (pitch_sin, pitch_cos) = self.pitch.0.sin_cos();

        Vector3::new(pitch_cos * yaw_cos, pitch_sin, pitch_cos * yaw_sin)
    }

    pub fn frustrum(&self, projection: &Projection) -> Frustrum {
        let matrix = projection.calc_matrix_opengl() * self.calc_matrix();

        let plane_right = Plane {
            normal: Vector3::new(
                matrix[0][3] - matrix[0][0],
                matrix[1][3] - matrix[1][0],
                matrix[2][3] - matrix[2][0],
            ),
            distance: matrix[3][3] - matrix[3][0],
        };

        let plane_left = Plane {
            normal: Vector3::new(
                matrix[0][3] + matrix[0][0],
                matrix[1][3] + matrix[1][0],
                matrix[2][3] + matrix[2][0],
            ),
            distance: matrix[3][3] + matrix[3][0],
        };

        let plane_bottom = Plane {
            normal: Vector3::new(
                matrix[0][3] + matrix[0][1],
                matrix[1][3] + matrix[1][1],
                matrix[2][3] + matrix[2][1],
            ),
            distance: matrix[3][3] + matrix[3][1],
        };

        let plane_top = Plane {
            normal: Vector3::new(
                matrix[0][3] - matrix[0][1],
                matrix[1][3] - matrix[1][1],
                matrix[2][3] - matrix[2][1],
            ),
            distance: matrix[3][3] - matrix[3][1],
        };

        let plane_near = Plane {
            normal: Vector3::new(
                matrix[0][3] + matrix[0][2],
                matrix[1][3] + matrix[1][2],
                matrix[2][3] + matrix[2][2],
            ),
            distance: matrix[3][3] + matrix[3][2],
        };

        let plane_far = Plane {
            normal: Vector3::new(
                matrix[0][3] - matrix[0][2],
                matrix[1][3] - matrix[1][2],
                matrix[2][3] - matrix[2][2],
            ),
            distance: matrix[3][3] - matrix[3][2],
        };

        Frustrum {
            top: plane_top,
            bottom: plane_bottom,
            left: plane_left,
            right: plane_right,
            near: plane_near,
            far: plane_far,
        }
    }
}

// check frustrum intersection for culling
impl Aabb for Camera {
    fn min(&self) -> Point3<f32> {
        self.physics_state.position - (self.collider / 2.0)
    }

    fn max(&self) -> Point3<f32> {
        self.physics_state.position + (self.collider / 2.0)
    }
}

pub struct Projection {
    aspect: f32,
    fovy: Rad<f32>,
    znear: f32,
    zfar: f32,
}

impl Projection {
    pub fn new<F: Into<Rad<f32>>>(width: u32, height: u32, fovy: F, znear: f32, zfar: f32) -> Self {
        Self {
            aspect: width as f32 / height as f32,
            fovy: fovy.into(),
            znear,
            zfar,
        }
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        self.aspect = width as f32 / height as f32;
    }

    pub fn calc_matrix_opengl(&self) -> Matrix4<f32> {
        OPENGL_TO_WGPU_MATRIX * perspective(self.fovy, self.aspect, self.znear, self.zfar)
    }
}

#[derive(Debug)]
pub struct CameraController {
    amount_left: f32,
    amount_right: f32,
    amount_forward: f32,
    amount_backward: f32,
    amount_up: f32,
    amount_down: f32,
    rotate_horizontal: f32,
    rotate_vertical: f32,
    speed: f32,
    sensitivity: f32,
    acceleration: f32,
    jump_velocity: f32,
}

impl CameraController {
    pub fn new(speed: f32, sensitivity: f32, jump_velocity: f32, acceleration: f32) -> Self {
        Self {
            amount_left: 0.0,
            amount_right: 0.0,
            amount_forward: 0.0,
            amount_backward: 0.0,
            amount_up: 0.0,
            amount_down: 0.0,
            rotate_horizontal: 0.0,
            rotate_vertical: 0.0,
            speed,
            sensitivity,
            acceleration,
            jump_velocity,
        }
    }

    pub fn process_keyboard(&mut self, key: PhysicalKey, state: ElementState) -> bool {
        let amount = match state {
            ElementState::Pressed => 1.0,
            ElementState::Released => 0.0,
        };

        match key {
            PhysicalKey::Code(KeyCode::KeyW) | PhysicalKey::Code(KeyCode::ArrowUp) => {
                self.amount_forward = amount;
                true
            }
            //VirtualKeyCode::S | VirtualKeyCode::Down => {
            PhysicalKey::Code(KeyCode::KeyS) | PhysicalKey::Code(KeyCode::ArrowDown) => {
                self.amount_backward = amount;
                true
            }
            //VirtualKeyCode::A | VirtualKeyCode::Left => {
            PhysicalKey::Code(KeyCode::KeyA) | PhysicalKey::Code(KeyCode::ArrowLeft) => {
                self.amount_left = amount;
                true
            }
            // VirtualKeyCode::D | VirtualKeyCode::Right => {
            PhysicalKey::Code(KeyCode::KeyD) | PhysicalKey::Code(KeyCode::ArrowRight) => {
                self.amount_right = amount;
                true
            }
            //VirtualKeyCode::Space => {
            PhysicalKey::Code(KeyCode::Space) => {
                self.amount_up = amount;
                true
            }
            //VirtualKeyCode::LShift => {
            PhysicalKey::Code(KeyCode::ShiftLeft) => {
                self.amount_down = amount;
                true
            }
            _ => false,
        }
    }

    pub fn process_mouse(&mut self, mouse_dx: f64, mouse_dy: f64) {
        self.rotate_horizontal = mouse_dx as f32;
        self.rotate_vertical = mouse_dy as f32;
    }

    pub fn update_camera(&mut self, camera: &mut Camera, dt: Duration) {
        let dt = dt.as_secs_f32();

        // Move forward/backward and left/right
        let (yaw_sin, yaw_cos) = camera.yaw.0.sin_cos();
        let forward = Vector3::new(yaw_cos, 0.0, yaw_sin).normalize();
        let right = Vector3::new(-yaw_sin, 0.0, yaw_cos).normalize();

        // this is a bit wonky because we reach into velocity directly, ideally we would apply
        // forces
        let velocity = &mut camera.physics_state.velocity;

        let jump_scaling = if !camera.physics_state.grounded {
            0.2
        } else {
            1.0
        };

        *velocity += forward
            * (self.amount_forward - self.amount_backward)
            * self.acceleration
            * dt
            * jump_scaling;
        *velocity +=
            right * (self.amount_right - self.amount_left) * self.acceleration * dt * jump_scaling;
        // clamp velocity to max speed
        velocity.x = velocity.x.clamp(-self.speed, self.speed);
        velocity.z = velocity.z.clamp(-self.speed, self.speed);

        // Jumping
        if self.amount_up > 0.0 && camera.physics_state.grounded && velocity.y < 0.1 {
            velocity.y += self.amount_up * self.jump_velocity;
        }

        // Rotate
        camera.yaw += Rad(self.rotate_horizontal) * self.sensitivity * dt;
        camera.pitch += Rad(-self.rotate_vertical) * self.sensitivity * dt;

        // If process_mouse isn't called every frame, these values
        // will not get set to zero, and the camera will rotate
        // when moving in a non-cardinal direction.
        self.rotate_horizontal = 0.0;
        self.rotate_vertical = 0.0;

        // Keep the camera's angle from going too high/low.
        if camera.pitch < -Rad(SAFE_FRAC_PI_2) {
            camera.pitch = -Rad(SAFE_FRAC_PI_2);
        } else if camera.pitch > Rad(SAFE_FRAC_PI_2) {
            camera.pitch = Rad(SAFE_FRAC_PI_2);
        }
    }
}

impl KinematicBody for Camera {
    fn state(&mut self) -> &mut KinematicBodyState {
        &mut self.physics_state
    }

    fn collider(&self) -> AabbBounds {
        // collider cube where camera origin is offset towards top of collider
        AabbBounds::new(
            self.physics_state.position - Vector3::new(0.4, 3., 0.4),
            self.physics_state.position + Vector3::new(0.4, 0., 0.4),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_camera_frustrum_calculation() {
        // pointing in -z direction
        let camera = Camera::new(
            Point3::new(0.0, 0.0, 0.0),
            cgmath::Deg(-90.),
            cgmath::Deg(0.0),
        );
        let projection = Projection::new(800, 600, cgmath::Deg(70.), 0.1, 100.0);
        let frustrum = camera.frustrum(&projection);

        assert!(frustrum.contains_point(&Point3::new(0.0, 0.0, -90.0)));
        assert!(frustrum.contains_point(&Point3::new(0.0, 0.0, -1.0)));
        assert!(frustrum.contains_point(&Point3::new(50.0, 50.0, -90.0)));

        // at 1 unit from the camera, the frustrum should not contain the point
        assert!(!frustrum.contains_point(&Point3::new(0.0, 2.0, -1.0)));
        // behind the frustrum
        assert!(!frustrum.contains_point(&Point3::new(0.0, 2.0, 1.0)));

        assert!(!frustrum.contains_point(&Point3::new(0.0, 100.0, -10.0)));
        assert!(!frustrum.contains_point(&Point3::new(100.0, 0.0, -10.0)));
    }

    #[test]
    fn test_camera_frustrum_calculation_backwards() {
        let camera = Camera::new(
            Point3::new(0.0, 0.0, 0.0),
            cgmath::Deg(-90.),
            cgmath::Deg(180.0),
        );
        let projection = Projection::new(800, 600, cgmath::Deg(70.), 0.1, 100.0);
        let frustrum = camera.frustrum(&projection);

        assert!(frustrum.contains_point(&Point3::new(0.0, 0.0, 90.0)));
        assert!(frustrum.contains_point(&Point3::new(0.0, 0.0, 1.0)));
        assert!(frustrum.contains_point(&Point3::new(50.0, 50.0, 90.0)));

        // at 1 unit from the camera, the frustrum should not contain the point
        assert!(!frustrum.contains_point(&Point3::new(0.0, 2.0, 1.0)));
        // behind the frustrum
        assert!(!frustrum.contains_point(&Point3::new(0.0, 2.0, -1.0)));

        assert!(!frustrum.contains_point(&Point3::new(0.0, 100.0, 10.0)));
        assert!(!frustrum.contains_point(&Point3::new(100.0, 0.0, 10.0)));
    }

    #[test]
    fn test_frustrum_contains_point() {
        let frustrum = Frustrum {
            top: Plane {
                normal: Vector3::new(0.0, -1.0, 0.0),
                distance: 1.0,
            },
            bottom: Plane {
                normal: Vector3::new(0.0, 1.0, 0.0),
                distance: 1.0,
            },
            left: Plane {
                normal: Vector3::new(1.0, 0.0, 0.0),
                distance: 1.0,
            },
            right: Plane {
                normal: Vector3::new(-1.0, 0.0, 0.0),
                distance: 1.0,
            },
            near: Plane {
                normal: Vector3::new(0.0, 0.0, -1.0),
                distance: 1.0,
            },
            far: Plane {
                normal: Vector3::new(0.0, 0.0, 1.0),
                distance: 1.0,
            },
        };
    }
}
