#version 330 core

layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aNorm;
layout(location = 2) in vec2 aTex;

out vec2 texCoord;

uniform mat4 model;

void main(){
	gl_Position = model * vec4(aPos, 1.0);
	texCoord = aTex;
	texCoord.x = 1.0 - texCoord.x; //do a horizontal flip cuz the origin pattern is at back.
}