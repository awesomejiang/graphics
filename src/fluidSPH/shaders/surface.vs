#version 330 core

layout(location = 0) in vec3 aPos;

out VS_OUT {
	vec3 camCoord;
	float radius;
} vs_out;

//uniform mat4 model;
//uniform mat4 projection;
//uniform mat4 view;
uniform mat4 VP;
uniform vec3 camPos;
uniform float pSize;

void main(){
	//gl_Position = projection * view * model * vec4(aPos, 1.0);
	gl_Position = VP * vec4(aPos, 1.0);
	gl_PointSize = pSize/length(camPos-aPos);
	vs_out.camCoord = gl_Position.xyz;
	vs_out.radius = gl_PointSize/2.0/800.0;
}