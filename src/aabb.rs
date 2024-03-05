use cgmath::{ElementWise, Point3};

pub trait Aabb {
    fn min(&self) -> Point3<f32>;
    fn max(&self) -> Point3<f32>;

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
