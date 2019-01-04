#ifndef FLUID_H
#define FLUID_H

#include <curand.h>
#include <curand_kernel.h>

#include "vec_float.h"
#include "particlesystem.hpp"

struct FluidParticle{
	vec3 pos;
	vec3 vel;
	float density;
	float pressure;
	vec4 color;
	curandState rand;
};

class Fluid: public ParticleSystem<FluidParticle>{
public:
	Fluid(int n)
	: ParticleSystem<FluidParticle>(n, {"shaders/fluid.vs", "shaders/fluid.fs"}) {}

	virtual void setVAO() const;
};

__GLOBAL__ void initKernel(FluidParticle* p, int n, Mouse const &mouse);
__GLOBAL__ void updateKernel(FluidParticle* p, int n, Mouse const &mouse);

//helper functions for PCISPH algorithm
__DEVICE__ float weight(vec3 const &src, vec3 const &dst, float const &h);
__DEVICE__ vec3 divWeight(vec3 const &src, vec3 const &dst, float const &h);
#endif