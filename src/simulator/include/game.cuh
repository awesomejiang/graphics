#ifndef GAME_H
#define GAME_H

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


class Game{
public:
	Game(unsigned int const &width, unsigned int const &height, std::vector<Particle> particles);
	~Game();
	void init(long long int const &num);
	void processInput() const;
	void render();

	bool shouldClose() const;

private:
	void createVBO();
	void setCallBacks() const;

	void deployGrid(long long int const &n);

	Scene const scene;
	unsigned int width, height, VBO, VAO;
	std::vector<Particle> particles;
	cudaGraphicsResource_t resource;
	int nParticle;
	dim3 block, grid;
};


__global__ void initKernel(Particle* vbo, int n, Particle *p);
__global__ void renderKernel(Particle* dptr, int n, vec2 *pos, int state);
__device__ int getIdx();


#endif