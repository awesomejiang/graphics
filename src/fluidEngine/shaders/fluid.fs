#version 330 core

in VS_OUT{
	vec2 pos;
	vec2 vel;
	float pressure;
	vec3 color;
} fs_in;


out vec4 fragColor;

void main(){
	float x = abs(fs_in.vel.x) , y = abs(fs_in.vel.y);
	//fragColor = vec4(x, y, 0.0, 1.0);
	fragColor = vec4(0.0, 0.0, 0.0, 1.0);

	float r = fs_in.color.x;
	if(r < -100 || r > 100)
		fragColor += vec4(1.0);
	else
		fragColor += vec4(fs_in.color, 1.0);

//	fragColor = vec4(fs_in.pressure, 0.0, 0.0, 1.0);
}