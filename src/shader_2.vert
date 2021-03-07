#version 450

const vec2 positions[3] = vec2[3](
    vec2(0.0, 0.5),
    vec2(-0.5, -0.5),
    vec2(0.5, -0.5)
);

layout(location=1) out vec4 v_color;

void main() {
    gl_Position = vec4(positions[gl_VertexIndex], 0.0, 1.0);
    v_color[gl_VertexIndex] = positions[gl_VertexIndex][gl_VertexIndex % 2];
}