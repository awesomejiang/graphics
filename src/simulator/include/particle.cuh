#ifndef PARTICLE_H
#define PARTICLE_H

#include <cuda.h>
#include <cstdio>
#include "vec.cuh"

enum class UpdateKernel{
	none,
	gravity
};

class Particle{
public:
	__device__ __host__ Particle(
		UpdateKernel const &update = UpdateKernel::none,
		vec2 const &pos = {0.0f, 0.0f},
		vec2 const &vel = {0.0f, 0.0f},
		vec4 const &color = {1.0f, 1.0f, 1.0f, 1.0f}
	);

	__device__ __host__ void update(vec2 const &forceCenter, bool pressed);

	vec2 pos, vel;
	vec4 color;

private:
	UpdateKernel updateK;
	__device__ __host__ void gravityKernel(Particle &p, vec2 const &forceCenter, bool pressed);
};


#endif