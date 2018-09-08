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
	Spirit(int const &n);
	~Spirit();
	void initCuda();
	void render(Mouse const &mouse);

private:
	void createVBO();
	void setCallBacks() const;

	void deployGrid();

	Particle* deviceParticles;
	curandState* deviceRandStates;
	unsigned int width, height, VBO, VAO;
	Shader pShader;
	cudaGraphicsResource_t resource;
	int nParticle;
	dim3 block, grid;
};


__GLOBAL__ void initKernel(Particle* dp, curandState* dr, int n);
__GLOBAL__ void renderKernel(Particle* dp, curandState* dr, int n, Mouse const &mouse);
__DEVICE__ int getIdx();


#endif