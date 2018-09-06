#version 330 core

in vec2 texCoord;

out vec4 fragColor;

uniform sampler2D front;
uniform sampler2D back;
uniform int isFront;

void main(){
	vec4 texColor;
	if(isFront == 1)
		texColor = texture(front, texCoord);
	else
		texColor = texture(back, texCoord);

	if(texColor.a == 0.0)
		discard;

	fragColor = texColor;
}