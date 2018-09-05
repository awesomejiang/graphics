#ifndef SPIRIT_H
#define SPIRIT_H

#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <cuda.h>
#include <cuda_gl_interop.h>

#include <vector>
#include <stdexcept>

#include "vec.cuh"
#include "particle.cuh"
#include "utility.cuh"
#include "scene.h"
#include "shader.h"

class Spirit{
public:
	Spirit(std::vector<Particle> particles);
	~Spirit();
	void initCuda();
	void render(Scene const &scene);

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


__global__ void initKernel(Particle* vbo, int n, Particle *p);
__global__ void renderKernel(Particle* dptr, int n, vec2 *pos, int state);
__device__ int getIdx();


#endif