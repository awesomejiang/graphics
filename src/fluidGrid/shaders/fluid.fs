#version 330 core

in VS_OUT{
	vec2 pos;
	vec2 vel;
	float pressure;
	vec3 color;
} fs_in;


out vec4 fragColor;

void main(){
	fragColor = vec4(0.0, 0.0, 0.0, 1.0);

	float x = fs_in.vel.x , y = fs_in.vel.y;
//	fragColor += vec4(0.0, x, y, 1.0);
//	fragColor += vec4(fs_in.pressure, 0.0, 0.0, 1.0);
//	if(x<0.0)
//		fragColor.x = 1.0;
//	else
//		fragColor.x = 0.5;
//	if(y<0.0)
//		fragColor.y = 0.5;
//	else if(y > 0.0)
//		fragColor.y = 1.0;

	float r = fs_in.color.x;
	fragColor += vec4(fs_in.color, 1.0);

//	fragColor = vec4(fs_in.pressure, 0.0, 0.0, 1.0);
}