use cgmath::{ElementWise, Point3};

use crate::physics::CollisionInfo;

pub trait Aabb {
    fn min(&self) -> Point3<f32>;
    fn max(&self) -> Point3<f32>;
    fn aabb(&self) -> AabbBounds {
        AabbBounds::new(self.min(), self.max())
    }

    fn intersect_ray(
        &self,
        d0: cgmath::Point3<f32>,
        dir: cgmath::Vector3<f32>,
    ) -> Option<[f32; 2]> {
        let t1 = (self.min() - d0).div_element_wise(dir);
        let t2 = (self.max() - d0).div_element_wise(dir);
        let t_min = t1.zip(t2, f32::min);
        let t_max = t1.zip(t2, f32::max);

        let mut hit_near = t_min.x;
        let mut hit_far = t_max.x;

        if hit_near > t_max.y || t_min.y > hit_far {
            return None;
        }

        if t_min.y > hit_near {
            hit_near = t_min.y;
        }
        if t_max.y < hit_far {
            hit_far = t_max.y;
        }

        if (hit_near > t_max.z) || (t_min.z > hit_far) {
            return None;
        }

        if t_min.z > hit_near {
            hit_near = t_min.z;
        }
        if t_max.z < hit_far {
            hit_far = t_max.z;
        }
        Some([hit_near, hit_far])
    }
}

#[derive(Debug)]
pub struct AabbBounds {
    min: Point3<f32>,
    max: Point3<f32>,
}

impl AabbBounds {
    pub fn new(min: Point3<f32>, max: Point3<f32>) -> Self {
        Self { min, max }
    }

    pub fn merge (&self, other: &Self) -> Self {
        let min = Point3::new(
            f32::min(self.min.x, other.min.x),
            f32::min(self.min.y, other.min.y),
            f32::min(self.min.z, other.min.z),
        );
        let max = Point3::new(
            f32::max(self.max.x, other.max.x),
            f32::max(self.max.y, other.max.y),
            f32::max(self.max.z, other.max.z),
        );
        Self::new(min, max)
    }

    pub fn intersects(&self, other: &impl Aabb) -> bool {
        self.min.x <= other.max().x
            && self.max.x >= other.min().x
            && self.min.y <= other.max().y
            && self.max.y >= other.min().y
            && self.min.z <= other.max().z
            && self.max.z >= other.min().z
    }

    pub fn intersection(&self, other: &impl Aabb) -> Option<CollisionInfo> {
        if !self.intersects(other) {
            return None;
        }
        // gets the intersection point between two aabb based on the least overlapping axis
        let min_a = self.min();
        let max_a = self.max();
        let min_b = other.min();
        let max_b = other.max();

        let distances = [
            max_a.x - min_b.x,
            max_b.x - min_a.x,
            max_a.y - min_b.y,
            max_b.y - min_a.y,
            max_a.z - min_b.z,
            max_b.z - min_a.z,
        ];

        println!("distances: {:?}", distances);

        let min_distance = distances.iter().cloned().fold(f32::INFINITY, f32::min);
        println!("min_distance: {:?}", min_distance);
        let axis = distances.iter().position(|&d| d == min_distance).unwrap();
        println!("axis: {:?}", axis);

        let normal = match axis {
            0 => cgmath::Vector3::new(1.0, 0.0, 0.0),
            1 => cgmath::Vector3::new(-1.0, 0.0, 0.0),
            2 => cgmath::Vector3::new(0.0, 1.0, 0.0),
            3 => cgmath::Vector3::new(0.0, -1.0, 0.0),
            4 => cgmath::Vector3::new(0.0, 0.0, 1.0),
            5 => cgmath::Vector3::new(0.0, 0.0, -1.0),
            _ => panic!("Invalid axis"),
        };

        Some(CollisionInfo::new(normal, min_distance))
    }
}

impl Aabb for AabbBounds {
    fn min(&self) -> Point3<f32> {
        self.min
    }
    fn max(&self) -> Point3<f32> {
        self.max
    }
}

#[cfg(test)]
mod tests { 
    use super::*;

    #[test]
    fn test_intersection() {
        let a = AabbBounds::new(Point3::new(0.0, 0.0, 0.0), Point3::new(1.0, 1.0, 1.0));
        let b = AabbBounds::new(Point3::new(0.0, 0.0, 0.5), Point3::new(1.5, 1.5, 1.5));

        assert_eq!(a.intersection(&b), Some(CollisionInfo::new(cgmath::Vector3::new(0.0, 0.0, 1.0), 0.5)));
    }

    #[test]
    fn test_intersection_2() {
        let a = AabbBounds::new(Point3::new(-0.3, 1.5, 1.5), Point3::new(0.7, 2.5, 2.5));
        let b = AabbBounds::new(Point3::new(0.0, 0.0, 0.0), Point3::new(2.0, 2.0, 2.0));

        assert_eq!(a.intersection(&b), Some(CollisionInfo::new(cgmath::Vector3::new(0.0, -1.0, 0.0), 0.5)));
    }
}
