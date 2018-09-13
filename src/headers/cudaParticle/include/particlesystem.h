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

template <typename ParticleType>
class ParticleSystem{
public:
	ParticleSystem(int const &n, Shader const &shader);
	~ParticleSystem();
	void initCuda();
	void render(Mouse const &mouse);

private:
	void createVBO();
	//virtual void setVAO() const;
	void setCallBacks() const;

	void deployGrid();

	int nParticle;
	ParticleType* deviceParticles = nullptr;
	unsigned int width, height, VBO, VAO;
	Shader shader;
	cudaGraphicsResource_t resource = 0;
	dim3 block, grid;
};

#endif
/*
    curandState *state = &dr[index];
	//init curand states
	curand_init(clock64(), index, 0, state);
*/