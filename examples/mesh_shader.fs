#version 410

layout (location = 0) in vec3 fPosition;
layout (location = 1) in vec3 fNormal;
out vec4 out_color;

uniform vec3 eye_position;
uniform vec3 light_position;
uniform vec3 ambient_color;
uniform vec3 diffuse_color;

void main()
{
    out_color = vec4(ambient_color, 1.0);
    float diffuse = dot(fNormal, normalize(light_position - fPosition));
    if (diffuse <= 0) {
        diffuse = -diffuse;
    }
    out_color += vec4(diffuse_color * diffuse, 1.0);
}