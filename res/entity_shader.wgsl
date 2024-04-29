// Vertex shader
struct CameraUniform {
    view_proj: mat4x4<f32>,
};
@group(0) @binding(0) // 1.
var<uniform> camera: CameraUniform;

struct Light {
    position: vec3<f32>,
    color: vec3<f32>,
}
@group(1) @binding(0)
var<uniform> light: Light;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) texture_index: u32,
}

struct InstanceInput {
    @location(5) model_matrix_0: vec4<f32>,
    @location(6) model_matrix_1: vec4<f32>,
    @location(7) model_matrix_2: vec4<f32>,
    @location(8) model_matrix_3: vec4<f32>,
    @location(9) texture: u32,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) texture_index: u32,
    @location(2) face_normal: vec3<f32>,
}

@vertex
fn vs_main(
    model: VertexInput,
    instance: InstanceInput,
) -> VertexOutput {
    let model_matrix = mat4x4<f32>(
        instance.model_matrix_0,
        instance.model_matrix_1,
        instance.model_matrix_2,
        instance.model_matrix_3,
    );

    var out: VertexOutput;
    let world_position = model_matrix * vec4<f32>(model.position, 1.0);
    out.clip_position = camera.view_proj * world_position;

    if (abs(model.normal.y) > .9) {
        out.uv = vec2<f32>(model.position.x, model.position.z);
    } else if (abs(model.normal.z) > 0.9) {
        out.uv = vec2<f32>(model.position.x, model.position.y);
    } else if (abs(model.normal.x) > 0.9) {
        out.uv = vec2<f32>(model.position.z, model.position.y);
    }
    out.uv = (out.uv + 1.0) / 2.0;
    out.uv.y = 1.0 - out.uv.y;
    out.texture_index = instance.texture;
    out.face_normal = model.normal;
    return out;
}

@group(2) @binding(0)
var t_diffuse: texture_2d<f32>;
@group(2) @binding(1)
var s_diffuse: sampler;
 
@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // bitshift sides 
    let texture_index_sides = in.texture_index & 0x000000FFu;
    let texture_index_top = (in.texture_index & 0x0000FF00u) >> 8u;
    let uv = in.uv;
    
    // if we are facing up, use the top texture, else use sides
    var texture_position: vec2<f32>;
    if in.face_normal.y > 0.9 {
        texture_position = vec2<f32>(f32(texture_index_top % 16u), f32(texture_index_top / 16u));
    } else {
        texture_position = vec2<f32>(f32(texture_index_sides % 16u), f32(texture_index_sides / 16u));
    }

    let texture_offset = 1.0 / 16.0;
    let total_offset = texture_position * texture_offset;
    return textureSample(t_diffuse, s_diffuse, vec2<f32>(uv.x / 16.0 + total_offset.x, uv.y / 16.0 + total_offset.y));
}

