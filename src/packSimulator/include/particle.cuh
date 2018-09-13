#ifndef PARTICLE_CUH
#define PARTICLE_CUH

#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>

#include "macros.cuh"
#include "utility.cuh"
#include "vec.cuh"

enum class InitKernelEnum{
	bottom,
	square,
	none
};

enum class UpdateKernelEnum{
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

	__DEVICE__ void init(InitKernelEnum const &ik, curandState* state);

	__DEVICE__ void update(UpdateKernelEnum const &uk, Mouse const &mouse);

	vec2 pos, vel;
	vec4 color;

private:
	template<InitKernelEnum IK>
	__DEVICE__ void initKernel();

	template<UpdateKernelEnum UK>
	__DEVICE__ void updateKernel(Mouse const &mouse);

	curandState *randState;
};

#endif