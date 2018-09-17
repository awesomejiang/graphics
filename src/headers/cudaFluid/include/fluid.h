#ifndef FLUID_H
#define FLUID_H

#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <cuda.h>
#include <cuda_gl_interop.h>
#include <curand.h>
#include <curand_kernel.h>

#include "macros.h"
#include "vec_float.h"
#include "utility.h"
#include "shader.h"

#include <stdexcept>



#define dt 0.0001f
#define rou 0.00001f

struct FluidState{
	vec2 pos;
	vec2 vel;
	float p;
	float color;
};


//indexing helping class
class Indexing{
public:
	__DEVICE__ Indexing(int const &w, int const &h);
	__DEVICE__ int getIdx();
	__DEVICE__ int getLeft();
	__DEVICE__ int getRight();
	__DEVICE__ int getTop();
	__DEVICE__ int getBottom();

	int w, h;
};


class Fluid{
public:
	Fluid(int const &width = 800, int const &height = 600);
	~Fluid();

	void render();

private:
	void deployGrid();

	FluidState *currState = nullptr, *starState = nullptr;
	Indexing *indexing = nullptr;
	int width, height, size;
	bool firstIteration = true;
	dim3 block, grid;
	Shader shader{"shaders/fluid.vs", "shaders/fluid.fs"};
	unsigned int VBO, VAO;
	cudaGraphicsResource_t resource = 0;
};

__GLOBAL__ void initIndexing(int w, int h, Indexing *indexing);
__GLOBAL__ void initFluid(Indexing *indexing, FluidState *currState);
__GLOBAL__ void advect(Indexing *indexing, FluidState *currState, FluidState *starState);
__GLOBAL__ void diffusion(Indexing *indexing, FluidState *currState, FluidState *starState);
__GLOBAL__ void div(Indexing *indexing, FluidState *currState, FluidState *starState);
__GLOBAL__ void pressure(Indexing *indexing, FluidState *currState, FluidState *starState);
__GLOBAL__ void swapPressure(Indexing *indexing, FluidState *currState, FluidState *starState);
__GLOBAL__ void correction(Indexing *indexing, FluidState *currState, FluidState *starState);

#endif