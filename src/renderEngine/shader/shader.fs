#version 330 core

struct Material{
   sampler2D textureDiffuse1;
   sampler2D textureSpecular1;
   float shininess;
};

in VS_OUT{
   vec3 norm;
   vec3 fragPos;
   vec2 texCoords;
} fs_in;

out vec4 fragColor;

uniform vec3 viewPos;

//for material
uniform Material material;

void main(){
	fragColor = texture(material.textureDiffuse1, fs_in.texCoords);
}
