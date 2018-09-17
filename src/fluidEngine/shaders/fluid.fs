#version 330 core

in VS_OUT{
	vec2 pos;
	vec2 vel;
	float pressure;
} fs_in;


out vec4 fragColor;

void main(){
	fragColor = vec4(vec3(fs_in.pressure), 1.0);
}