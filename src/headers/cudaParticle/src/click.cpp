#include "click.h"

//kernels
__GLOBAL__ void initKernel(ClickParticle* p, int n, Mouse const &mouse){
	auto idx = getIdx();
	if(idx > n)
		return ;

	ClickParticle &pr = p[idx];

	curand_init(clock64(), idx, 0, &pr.rand);
	pr.lifetime = 9;//idx % 1000;
	pr.color = vec4(0.0f);
}


__GLOBAL__ void updateKernel(ClickParticle* p, int n, Mouse const &mouse){
	auto idx = getIdx();
	if(idx > n)
		return ;

	ClickParticle &pr = p[idx];

	//particle dies, create a new one to subsititute
	if(mouse.pressed && pr.lifetime == 0){
		//lifetime
		pr.lifetime = 1000;
		//position
		pr.pos = mouse.pos;
		//velocity
		float r = 0.001f*curand_normal(&pr.rand), theta = curand_uniform(&pr.rand)*2*M_PI;
		pr.vel = vec2(r*cos(theta), r*sin(theta));
		//color
		pr.color = vec4(0.2f, 0.3f, 0.7f, 1.0f);
	}
	else if(pr.lifetime != 0){
		//lifetime
		--pr.lifetime;
		//position
		pr.pos += pr.vel;
		//color
		pr.color /= 1.002f;
		if(pr.color[3] < 0.01f)
			pr.color = vec4(0.0f);
	}
}
