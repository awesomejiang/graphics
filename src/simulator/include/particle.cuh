#ifndef PARTICLE_CUH
#define PARTICLE_CUH

#include <cuda.h>
#include <cstdio>

#include "macros.cuh"
#include "utility.cuh"
#include "vec.cuh"

enum class UpdateKernel{
	none,
	gravity,
	shinning
};

class Particle{
public:
	__DEVICE__ __HOST__ Particle(
		UpdateKernel const &update = UpdateKernel::none,
		vec2 const &pos = {0.0f, 0.0f},
		vec2 const &vel = {0.0f, 0.0f},
		vec4 const &color = {1.0f, 1.0f, 1.0f, 1.0f}
	);

	__DEVICE__ __HOST__ void update(Mouse const &mouse);

	vec2 pos, vel;
	vec4 color;

private:
	UpdateKernel updateK;

	template<UpdateKernel K>
	__DEVICE__ __HOST__ void kernel(Mouse const &mouse);
};


#endif