#ifndef PARTICLE_CUH
#define PARTICLE_CUH

#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>

#include "macros.cuh"
#include "utility.cuh"
#include "vec.cuh"

enum class InitKernel{
	bottom,
	square,
	none
};

enum class UpdateKernel{
	gravity,
	shinning,
	none
};

class Particle{
public:
	__DEVICE__ Particle(
		vec2 const &pos = {0.0f, 0.0f},
		vec2 const &vel = {0.0f, 0.0f},
		vec4 const &color = {1.0f, 1.0f, 1.0f, 1.0f}
	);

	template<InitKernel UK>
	__DEVICE__ void init(curandState* state);

	template<UpdateKernel K>
	__DEVICE__ void update(curandState* state, Mouse const &mouse);

	vec2 pos, vel;
	vec4 color;
};

#endif