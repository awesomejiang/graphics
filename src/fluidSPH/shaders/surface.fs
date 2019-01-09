#version 330 core

in VS_OUT {
	vec3 camCoord;
	float radius;
} fs_in;

out vec3 pos;

void main(){
	//depth should be from camera coord
	vec2 xy = gl_PointCoord * 2.0 - vec2(1.0);
	float mag = dot(xy, xy);
	if(mag > 1.0) discard;
	float z = -sqrt(1.0 - mag);
	pos = fs_in.camCoord + vec3(xy, z)*fs_in.radius;
	gl_FragDepth = pos.z * gl_FragCoord.w;
}