#version 330 core

layout(location = 0) in vec3 aPos;

out VS_OUT {
	vec3 pos;
	vec4 color;
} vs_out;

//uniform mat4 model;
uniform mat4 projection;
uniform mat4 view;
uniform vec4 color;

void main(){
	//gl_Position = projection * view * model * vec4(aPos, 1.0);
	gl_Position = projection * view * vec4(aPos, 1.0);
	vs_out.pos = aPos;
	vs_out.color = color;
}