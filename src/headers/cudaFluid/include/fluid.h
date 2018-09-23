#ifndef FLUID_H
#define FLUID_H

#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <cuda.h>
#include <cuda_gl_interop.h>

#include "macros.h"
#include "vec_float.h"
#include "utility.h"
#include "shader.h"
#include "fluidcommon.h"
#include "mathsolver.h"

#include <stdexcept>
#include <utility>

#define VBO_NUM 4
#define t 0.1f
#define niu 1.0f
#define halfIteration 10 //"std::swap" involved, so iteration number must be even


class Fluid{
public:
	Fluid(int const &width = 800, int const &height = 600);
	~Fluid();

	void render();

private:
	void initGL();
	void initCuda();

	void colorSpread();
	void solveMomentum();
	void correctPressure();

	vec2 *pos = nullptr, *oldV = nullptr, *currV = nullptr, *tempV = nullptr;
	float *oldP = nullptr, *currP = nullptr, *div = nullptr;
	vec3 *oldC = nullptr, *currC = nullptr, *tempC = nullptr;

	Indexing *indexing = nullptr;
	int width, height, size;
	bool firstIteration = true;
	dim3 block, grid;

	Shader shader{"shaders/fluid.vs", "shaders/fluid.fs"};
	unsigned int VBO[VBO_NUM], VAO;
	cudaGraphicsResource_t resource[VBO_NUM];// = {0, 0, 0, 0};
};

__GLOBAL__ void initIndexing(int w, int h, Indexing *indexing);
__GLOBAL__ void initFluid(Indexing *indexing, vec2* pos, vec2* v, float* p, vec3* c);

__GLOBAL__ void force(Indexing *indexing, vec2 *vel, float dt);
__GLOBAL__ void divBC(Indexing *indexing, float *div);
__GLOBAL__ void velBC(Indexing *indexing, vec2 *vel);
__GLOBAL__ void pressureBC(Indexing *indexing, float* p);

#endif