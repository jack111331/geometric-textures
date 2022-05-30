#version 410

//uniform mat4 mvpmatrix;

layout (location = 0) in vec3 position;
layout (location = 1) in vec3 normal;

layout (location = 0) out vec3 fPosition;
layout (location = 1) out vec3 fNormal;

uniform mat4 mvp_matrix;
uniform mat4 model_matrix;
uniform mat3 inv_model_matrix;

void main() {
    fNormal = normalize(inv_model_matrix * normal);
    fPosition = vec3(model_matrix * vec4(position, 1.0));
    gl_Position = mvp_matrix * vec4(position, 1.0);
}