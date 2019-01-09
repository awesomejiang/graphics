#version 330 core

layout(location = 0) in vec3 aPos;

out VS_OUT {
	vec3 pos;
} vs_out;

//uniform mat4 model;
//uniform mat4 projection;
//uniform mat4 view;
uniform mat4 VP;
uniform float pSize;

void main(){
	//gl_Position = projection * view * model * vec4(aPos, 1.0);
	gl_Position = VP * vec4(aPos, 1.0);
	gl_PointSize = pSize/gl_Position.z;
	vs_out.pos = aPos;
}