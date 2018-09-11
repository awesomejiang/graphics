#version 330 core

in vec2 texCoord;

out vec4 fragColor;

uniform sampler2D image;

uniform bool horizontal;
uniform float weight[5] = float[](0.227027, 0.1945946, 0.1216216, 0.054054, 0.016216);

void main(){
	vec2 texOffset = 1.0/textureSize(image, 0);
	vec3 result = texture(image, texCoord).rgb * weight[0];

	if(horizontal){
		for(int i=0; i<5; ++i){
			result += texture(image, texCoord+vec2(texOffset.x*i, 0.0)).rgb * weight[i];
			result += texture(image, texCoord-vec2(texOffset.x*i, 0.0)).rgb * weight[i];
		}
	} 
	else{
		for(int i=0; i<5; ++i){
			result += texture(image, texCoord+vec2(0.0, texOffset.y*i)).rgb * weight[i];
			result += texture(image, texCoord-vec2(0.0, texOffset.y*i)).rgb * weight[i];
		}
	}

	fragColor = vec4(result, 1.0);
}