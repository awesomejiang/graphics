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
#include "indexing.h"

#include <stdexcept>
#include <utility>

#define VBO_NUM 4
#define NIU 0.01f
#define HALFITERATION 10 //"std::swap" involved, so iteration number must be even


class Fluid{
public:
	Fluid(int const &width = 800, int const &height = 600);
	~Fluid();

	void render(Mouse const &mouse, float const &t);

private:
	void initGL();
	void initCuda();

	void solveMomentum();
	void correctPressure();
	void colorSpread();

	vec2 *pos = nullptr, *oldV = nullptr, *currV = nullptr, *tempV = nullptr;
	float *oldP = nullptr, *currP = nullptr, *div = nullptr;
	vec3 *oldC = nullptr, *currC = nullptr, *tempC = nullptr;

	Indexing *indexing = nullptr;
	int width, height, size;
	bool firstIteration = true;
	dim3 block, grid;

	Mouse* deviceMouse;
	float dt;

	Shader shader{"shaders/fluid.vs", "shaders/fluid.fs"};
	unsigned int VBO[VBO_NUM], VAO;
	cudaGraphicsResource_t resource[VBO_NUM] = {0, 0, 0, 0};
};

__GLOBAL__ void initIndexing(int w, int h, Indexing *indexing);
__GLOBAL__ void initFluid(Indexing *indexing, vec2* pos, vec2* v, float* p, vec3* c);

__GLOBAL__ void addForce(Indexing *indexing, Mouse *mouse, vec2 *vel, float dt);
__GLOBAL__ void addDye(Indexing *indexing, Mouse *mouse, vec3 *color, float dt);

#endif