#version 330 core

in vec2 texCoord;

out vec4 fragColor;

uniform sampler2D back;
uniform vec4 color;

void main(){	
	vec4 texColor = texture(back, texCoord);

	if(texColor.a < 0.1)
		discard;

	fragColor = color;
	fragColor.a *= 0.5 - abs(texCoord.x-0.5)+0.5 - abs(texCoord.y-0.5);
}