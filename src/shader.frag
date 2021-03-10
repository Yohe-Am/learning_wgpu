#version 450

layout(location=0) in vec2 v_tex_coords;
layout(location=0) out vec4 f_color;

layout(set = 0, binding = 0) uniform texture2D t_diffuse;
layout(set = 0, binding = 1) uniform sampler s_diffuse;

// layout(set = 2, binding = 0) uniform texture2D t_depth;
// layout(set = 2, binding = 1) uniform samplerShadow s_depth;

void main() {
    f_color = texture(sampler2D(t_diffuse, s_diffuse), v_tex_coords);

    // float level = texture(sampler2DShadow(t_depth, s_depth), vec3(v_tex_coords, 0.50));
    // f_color = vec4(level, level, level, 1.0);
}   
