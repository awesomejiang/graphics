#ifndef SPIRIT_CUH
#define SPIRIT_CUH

#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <cuda.h>
#include <cuda_gl_interop.h>

#include <vector>
#include <stdexcept>

#include "macros.cuh"
#include "vec.cuh"
#include "particle.cuh"
#include "utility.cuh"
#include "shader.h"

class Spirit{
public:
	Spirit(std::vector<Particle> particles);
	~Spirit();
	void initCuda();
	void render(Mouse const &mouse);

private:
	void createVBO();
	void setCallBacks() const;

	void deployGrid();

	unsigned int width, height, VBO, VAO;
	std::vector<Particle> particles;
	Shader pShader;
	cudaGraphicsResource_t resource;
	int nParticle;
	dim3 block, grid;
};


__GLOBAL__ void initKernel(Particle* vbo, int n, Particle *p);
__GLOBAL__ void renderKernel(Particle* dptr, int n, Mouse* mouse);
__DEVICE__ int getIdx();


#endif