#ifndef PARTICLESYSTEM_H
#define PARTICLESYSTEM_H

#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <cuda.h>
#include <cuda_gl_interop.h>
#include <curand.h>
#include <curand_kernel.h>

#include <vector>
#include <stdexcept>

#include "macros.h"
#include "vec_float.h"
#include "utility.h"
#include "shader.h"

// for every concrete particle class, those function have to be overrided/specialized:
// void Particle::setVAO (optional)
// void initKernel(Particle*, int n, Mouse const &mouse);
// void initKernel(Particle*, int n, Mouse const &mouse);

template <typename ParticleType>
class ParticleSystem{
public:
	ParticleSystem(int const &n, Shader const &shader);
	~ParticleSystem();
	void initCuda();
	void render(Mouse const &mouse);

private:
	void createVBO();
	virtual void setVAO() const;

	void deployGrid();

	bool init = false;
	int nParticle;
	ParticleType* deviceParticles = nullptr;
	unsigned int width, height, VBO, VAO;
	Shader shader;
	cudaGraphicsResource_t resource = 0;
	dim3 block, grid;
};

__DEVICE__ inline int getIdx(){
	int grid = gridDim.x*gridDim.y*blockIdx.z + gridDim.x*blockIdx.y + blockIdx.x;
	return blockDim.x*grid + threadIdx.x;
}

#endif