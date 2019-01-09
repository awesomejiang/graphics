#version 330 core

in VS_OUT{
	vec3 pos;
} fs_in;

out float thickness;

void main(){
	vec2 xy = gl_PointCoord * 2.0 - vec2(1.0);
	float mag = dot(xy, xy);
	if(mag > 1.0) discard;

	thickness = sqrt(1.0 - mag);
}