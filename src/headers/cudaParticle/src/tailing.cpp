#include "tailing.h"

//kernels
__GLOBAL__ void initKernel(TailingParticle* p, int n, Mouse const &mouse){
	auto idx = getIdx();
	if(idx > n)
		return ;

	TailingParticle &pr = p[idx];

	curand_init(clock64(), idx, 0, &pr.rand);
	pr.lifetime = idx % 1000;
	pr.color = vec4(0.0f, 0.0f, 0.0f, 0.0f);
}


__GLOBAL__ void updateKernel(TailingParticle* p, int n, Mouse const &mouse){
	auto idx = getIdx();
	if(idx > n)
		return ;

	TailingParticle &pr = p[idx];

	//particle dies, create a new one to subsititute
	if(pr.lifetime == 0){
		//lifetime
		pr.lifetime = 1000;
		//position
		pr.pos = mouse.pos;
		float r = 0.001f*curand_normal(&pr.rand), theta = curand_uniform(&pr.rand)*2*M_PI;
		pr.pos += vec2(r*cos(theta), r*sin(theta));
		//color
		pr.color = vec4(0.2f, 0.3f, 0.7f, 1.0f);
	}
	else{
		//lifetime
		--pr.lifetime;
		//position
		float r = 0.001f*curand_normal(&pr.rand), theta = curand_uniform(&pr.rand)*2*M_PI;
		pr.pos += vec2(r*cos(theta), r*sin(theta));
		//color
		pr.color /= 1.001f;
	}
}
