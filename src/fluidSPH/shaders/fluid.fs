#version 330 core

in VS_OUT{
	vec3 pos;
	vec4 color;
} fs_in;


out vec4 fragColor;


uniform mat4 projection;
uniform mat4 view;
uniform vec3 camPos;

vec3 lightDir = vec3(-1.0, -1.0, -1.0);
vec4 light_spec = vec4(0.5);
vec4 mat_spec = vec4(1);

void main(){
	//calculate normal
	vec3 normal;
	normal.xy = gl_PointCoord * 2.0 - vec2(1.0);
	float mag = dot(normal.xy, normal.xy);
	if(mag > 1.0) discard;

	normal.z = sqrt(1.0 - mag);

	//diffuse
	float diffuse = max(0.0, dot(lightDir, normal));

	//specular
	vec3 halfVec = normalize(camPos + lightDir);
	float spec = max(0.0, pow(dot(normal, halfVec), 1000));
	vec4 s = light_spec * mat_spec * spec;

	//final color
	fragColor = fs_in.color * (0.5 + diffuse) + s;
	fragColor.w = 1.0;
}