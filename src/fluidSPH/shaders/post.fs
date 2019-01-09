#version 330 core

out vec4 fragColor;

in vec2 texCoords;

uniform sampler2D surface;
uniform sampler2D thickness;

uniform mat4 VP;
uniform vec3 camPos;

vec3 lightDir = vec3(-1.0, -1.0, -1.0);
vec4 light_spec = vec4(0.5);
vec4 mat_spec = vec4(1);

void main(){
	//discard empty pixel
	float blend = texture(thickness, texCoords).x;
	if(blend < 0.01)
		discard;

	//calculate normal
	float offset = 1.0/800.0;
	vec2 dt = vec2(offset, 0.0);
	vec2 ds = vec2(0.0, offset);

	vec3 p = vec3(texture(surface, texCoords));
	vec3 s1 = vec3(texture(surface, texCoords + ds)) - p;
	vec3 s2 = vec3(texture(surface, texCoords - ds)) - p;
	vec3 t1 = vec3(texture(surface, texCoords + dt)) - p;
	vec3 t2 = vec3(texture(surface, texCoords - dt)) - p;

	vec3 s = length(s1) > length(s2)? s2: s1;
	vec3 t = length(t1) > length(t2)? t2: t1;

	vec3 normal = normalize(cross(s, t));
	if(normal.z < 0.0)
		normal = -normal;
	normal.x = -normal.x;

	//diffuse
	float diffuse = max(0.0, dot(lightDir, normal));

	//specular
	vec3 halfVec = normalize(camPos + lightDir);
	float spec = max(0.0, pow(dot(normal, halfVec), 1000));
	vec4 specColor = light_spec * mat_spec * spec;

	//final color
	if(p.z > 0.001){
		fragColor = vec4(1.0, 0.0, 0.0, 1.0);
		fragColor.w = 1.0;
	}

/*
	//transform lightDir to camera coord
	lightDir = normalize(vec3(VP * vec4(lightDir, 1.0)));

	//reflection dir
	vec3 iDir = reflect(lightDir, normal);

	//refraction dir
	vec rDir = refract(lightDir, normal, 1.33);
*/
}