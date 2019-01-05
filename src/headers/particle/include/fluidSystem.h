#ifndef FLUIDSYSTEM_H
#define FLUIDSYSTEM_H

#include <curand.h>
#include <curand_kernel.h>

#include "camera.h"
#include "shader.h"

#include "vec_float.h"
#include "particle.h"
#include "gridcells.h"

#include <vector>


class FluidSystem{
public:
	FluidSystem(float h): h{h}, gc{h} {}

	void addCube(Cube range, float k = 0.01f, float gamma = 1.1f, float dt = 0.01, vec4 color = {1.0f, 1.0f, 1.0f, 1.0f});
	void render(Camera const &camera);
private:
	//graphics
	Shader shader={"shaders/fluid.vs", "shaders/fluid.fs"};
	float h;

	//particles
	std::vector<Particle> particles;

	//uniform grid
	GridCells gc;

};

#endif