#ifndef SPIRIT_CUH
#define SPIRIT_CUH

#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <cuda.h>
#include <cuda_gl_interop.h>
#include <curand.h>
#include <curand_kernel.h>

#include <vector>
#include <stdexcept>

#include "macros.cuh"
#include "vec.cuh"
#include "utility.cuh"
#include "shader.h"
#include "particle.cuh"

class Spirit{
public:
	Spirit(
		int const &n = 102400,
		InitKernelEnum const &ik = InitKernelEnum::bottom
	);
	~Spirit();
	void initCuda(InitKernelEnum const &ik);
	void render(UpdateKernelEnum const &uk, Mouse const &mouse);

private:
	void createVBO();
	void setCallBacks() const;

	void deployGrid();

	int nParticle;
	Particle* deviceParticles;
	curandState* deviceRandStates;
	unsigned int width, height, VBO, VAO;
	Shader pShader;
	cudaGraphicsResource_t resource;
	dim3 block, grid;
};

__GLOBAL__ void initKernel(InitKernelEnum const &ik, Particle* dp, curandState* dr, int n);

__GLOBAL__ void renderKernel(
	UpdateKernelEnum const &uk, Particle* dp, int n, Mouse const &mouse);


#endif