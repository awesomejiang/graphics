/*
#version 330 core

in vec2 texCoord;

layout(location = 0) out vec4 fragColor;
layout(location = 1) out vec4 brightColor;

uniform sampler2D front;
uniform sampler2D back;
uniform int isFront;

void main(){
	vec4 texColor;
	if(isFront == 1)
		texColor = texture(front, texCoord);
	else
		texColor = texture(back, texCoord);

	if(texColor.a < 0.1)
		discard;

	fragColor = texColor;
	brightColor = vec4(1.0, 0.0, 0.0, 1.0);
}
*/

#version 330 core

in vec2 texCoord;

layout(location = 0) out vec4 fragColor;
layout(location = 1) out vec4 brightColor;


uniform sampler2D front;
uniform sampler2D back;
uniform int isFront;

uniform vec4 selectedBoundary;

uniform vec4 rotatingBoundary;
uniform float rotatingWidth;


//local vars
vec4 texColor;
vec2 texOffset;

vec4 cardPattern(){
	if(isFront == 1){
		texColor = texture(front, texCoord);
		texOffset = 1.0/textureSize(front, 0);
	}
	else{
		texColor = texture(back, texCoord);
		texOffset = 1.0/textureSize(back, 0);
	}

	return texColor;
}


vec4 selected(float range, vec4 color){
	if(texColor.a < 0.05){
		vec4 ret = vec4(0.0, 0.0, 0.0, 0.0);

		float glowAlpha = 0.0;
		float count = 0.0;
		for(float i=-range; i<=range; i+=range/10)
			for(float j=-range; j<=range; j+=range/10){
				vec2 samplerTexCoord = texCoord+vec2(texOffset.x*i, texOffset.y*j);
				if(samplerTexCoord.x<0.0 || samplerTexCoord.x>1.0
					|| samplerTexCoord.y<0.0 || samplerTexCoord.y>1.0)
					glowAlpha += 0.0;
				else if(isFront == 1)
					glowAlpha += texture(front, samplerTexCoord).a;
				else
					glowAlpha += texture(back, samplerTexCoord).a;
				count += 1.0;
			}
			
		glowAlpha /= count;
		glowAlpha *= glowAlpha;

		ret = color;
		ret.a *= glowAlpha;

		return ret;
	}
	else
		return vec4(0.0, 0.0, 0.0, 0.0);
}

/*
vec4 rotating(float range, vec4 color){
	if(texColor.a < 0.05)
		for(float i=-range; i<=range; i+=range/10)
			for(float j=-range; j<=range; j+=range/10){
				vec2 samplerTexCoord = texCoord+vec2(texOffset.x*i, texOffset.y*j);
				if( //in sampler coordinate
					(samplerTexCoord.x>0.0 && samplerTexCoord.x<1.0
					 || samplerTexCoord.y>0.0 || samplerTexCoord.y<1.0)
					&&//and is near boundary
					(isFront == 1 
					 && texture(front, samplerTexCoord).a>0.9
					 ||
					 isFront == 0
					 && texture(back, samplerTexCoord).a>0.9))
					return color;
			}

	return vec4(0.0, 0.0, 0.0, 0.0);
}
*/
vec4 rotating(float range, vec4 color){
	if(texColor.a < 0.05)
		for(float i=-range; i<=range; i+=range*2.0)
			for(float j=-range; j<=range; j+=range*2.0){
				vec2 samplerTexCoord = texCoord+vec2(texOffset.x*i, texOffset.y*j);
				if( //in sampler coordinate
					samplerTexCoord.x>0.0 && samplerTexCoord.x<1.0
					&& samplerTexCoord.y>0.0 && samplerTexCoord.y<1.0
					//and is near boundary
					&&
					(isFront == 1 
					 && texture(front, samplerTexCoord).a>0.9
					 ||
					 isFront == 0
					 && texture(back, samplerTexCoord).a>0.9))
					return color;
			}

	return vec4(0.0, 0.0, 0.0, 0.0);
}

void main(){
	//card pattern
	fragColor = cardPattern();

	//light when selected
	fragColor += selected(80.0, selectedBoundary);

	//highlights when rotating
	if(rotatingWidth > 0.0)
		fragColor += rotating(rotatingWidth, rotatingBoundary);

	if(fragColor.a < 0.01)
		discard;
}
