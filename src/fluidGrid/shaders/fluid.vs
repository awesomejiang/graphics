#version 330 core

layout(location = 0) in vec2 aPos;
layout(location = 1) in vec2 aVel;
layout(location = 2) in float aPressure;
layout(location = 3) in vec3 aColor;

out VS_OUT {
	vec2 pos;
	vec2 vel;
	float pressure;
	vec3 color;
} vs_out;

void main(){
	gl_Position = vec4(aPos, 0.0, 1.0);
	vs_out.pos = aPos;
	vs_out.vel = aVel;
	vs_out.pressure = aPressure;
	vs_out.color = aColor;
}