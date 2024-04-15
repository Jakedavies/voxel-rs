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
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(1) world_normal: vec3<f32>,
    @location(2) world_position: vec3<f32>,
    @location(3) block_data_0: u32,
    @location(4) position: vec3<f32>,
    @location(5) normal: vec3<f32>,
}

@vertex
fn vs_main(
    model: VertexInput,
) -> VertexOutput {

    var out: VertexOutput;
    out.world_normal = model.normal;

    var world_position: vec4<f32> = vec4<f32>(model.position, 1.0);
    out.world_position = world_position.xyz;
    out.clip_position = camera.view_proj * world_position;
    out.block_data_0 = 0x00000000u;
    out.position = model.position;
    out.normal = model.normal;

    return out;
}

@group(2) @binding(0)
var t_diffuse: texture_2d<f32>;
@group(2) @binding(1)
var s_diffuse: sampler;
 
@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    var uv: vec2<f32>;
    let faceNormal: vec3<f32> = abs(in.normal);

    if faceNormal.x > faceNormal.y && faceNormal.x > faceNormal.z {
        // (1, 0, 0) is the abs normal of the face
        uv = in.position.zy;
    } else if faceNormal.y > faceNormal.x && faceNormal.y > faceNormal.z {
        // (0, 1, 0) is the abs normal of the face
        uv = in.position.xz;
    } else {
        // (0, 0, 1) is the abs normal of the face
        uv = in.position.xy;
    }

    let selected = in.block_data_0 & 0x00FF0000u;

    // dot normal and position to get a uv coordinate
    //let uv = vec2<f32>(dot(in.normal.zxy, in.position), dot(in.normal.yzx, in.position));
    // convert position to a color for debugging
    let corrected_position = 1.0 - (uv + vec2<f32>(1.0, 1.0)) / 2.0;
    //let object_color: vec4<f32> = vec4<f32>(corrected_position.x, corrected_position.y,  0.0, 1.0);

    // bitshift sides 
    let texture_index_top = in.block_data_0 & 0x000000FFu;
    let texture_index_sides = (in.block_data_0 & 0x0000FF00u) >> 8u;
    
    // if we are facing up, use the top texture, else use sides
    var texture_position: vec2<f32>;
    if faceNormal.y > 0.0 {
        texture_position = vec2<f32>(f32(texture_index_sides % 16u), f32(texture_index_sides / 16u));
    } else {
        texture_position = vec2<f32>(f32(texture_index_top % 16u), f32(texture_index_top / 16u));
    }

    let texture_offset = 1.0 / 16.0;
    let total_offset = texture_position * texture_offset;
    //let object_color: vec4<f32> = textureSample(t_diffuse, s_diffuse, vec2<f32>(corrected_position.x / 16.0 + total_offset.x, corrected_position.y / 16.0 + total_offset.y));
    
    // color from uv only 
    let object_color: vec4<f32> = vec4<f32>((uv.x) % 2.0, 0.0, (uv.y) % 2.0, 1.0); 
    
    // We don't need (or want) much ambient light, so 0.1 is fine
    var ambient_strength = 0.2;
    if selected != 0u {
        ambient_strength = 0.5;
    }
    let ambient_color = light.color * ambient_strength;

    let light_dir = normalize(light.position - in.world_position);

    let diffuse_strength = max(dot(in.world_normal, light_dir), 0.0);
    let diffuse_color = light.color * diffuse_strength;

    let result = (ambient_color + diffuse_color) * object_color.xyz;

    return vec4<f32>(result, object_color.a);
}

