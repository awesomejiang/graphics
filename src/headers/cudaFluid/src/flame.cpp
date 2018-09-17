#include "flame.h"


__DEVICE__ __HOST__ vec3 Potential::samplePotential(vec3 const &pos) const{
	return vec3{0.0f, pos[1], 0.0f};
}

__DEVICE__ __HOST__ Potential::Potential()
: d{1e-4f}, dx{d, 0.0f, 0.0f}, dy{0.0f, d, 0.0f}, dz{0.0f, 0.0f, d} {}


//3d case:
__DEVICE__ __HOST__ vec3 Potential::computeCurl(vec3 const &pos) const{
	float x =
		samplePotential(pos + dy)[2] - samplePotential(pos - dy)[2]
		- samplePotential(pos + dz)[1] + samplePotential(pos - dz)[1];
	float y =
		samplePotential(pos + dz)[0] - samplePotential(pos - dz)[0]
		- samplePotential(pos + dx)[2] + samplePotential(pos - dx)[2];
	float z =
		samplePotential(pos + dx)[1] - samplePotential(pos - dx)[1]
		- samplePotential(pos + dy)[0] + samplePotential(pos - dy)[0];

	printf("%f, %f, %f\n", x, y, z);
	return vec3(x, y, z)/(2*d);
}

__DEVICE__ __HOST__ vec3 Potential::computeGradient(vec3 const &pos) const{
/*
	float f = samplePotential(pos);
	float fdx = samplePotential(pos + dx) - f;
	float fdy = samplePotential(pos + dy) - f;
	float fdz = samplePotential(pos + dz) - f;

	return -normalize(vec3{fdx, fdy, fdz});
*/
	return vec3{0.0f};
}

//kernels
__GLOBAL__ void initKernel(FlameParticle* p, int n, Mouse const &mouse){
    auto idx = getIdx();
    if(idx > n)
    	return ;

    FlameParticle &pr = p[idx];
	curand_init(clock64(), idx, 0, &pr.rand);
	pr.pos = {curand_uniform(&pr.rand) * 2.0f - 1.0f, 0.0f};
	pr.vel = {0.0f, 0.0f};
	pr.color = {1.0f, 1.0f, 1.0f, 1.0f};
}

__GLOBAL__ void updateKernel(FlameParticle* p, int n, Mouse const &mouse){
    auto idx = getIdx();
    if(idx > n)
    	return ;

    FlameParticle &pr = p[idx];
	//FlamePotential U;
	//vec3 v = U.computeCurl({pr.pos[0], pr.pos[1], 0.0f});
	Potential U;
	vec3 v = U.computeCurl({pr.pos[0], pr.pos[1], 0.0f});
	pr.pos += {v[0], v[1]};
}

template class ParticleSystem<FlameParticle>; 