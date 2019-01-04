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

/*
struct FluidParticle{
	vec3 pos;
	vec3 vel;
	float density;
	float pressure;
	vec4 color;
	curandState rand;
};

class Fluid: public ParticleSystem<FluidParticle>{
public:
	Fluid(int n)
	: ParticleSystem<FluidParticle>(n, {"shaders/fluid.vs", "shaders/fluid.fs"}) {}

	virtual void setVAO() const;
};

__GLOBAL__ void initKernel(FluidParticle* p, int n, Mouse const &mouse);
__GLOBAL__ void updateKernel(FluidParticle* p, int n, Mouse const &mouse);

//helper functions for PCISPH algorithm
__DEVICE__ float weight(vec3 const &src, vec3 const &dst, float const &h);
__DEVICE__ vec3 divWeight(vec3 const &src, vec3 const &dst, float const &h);
*/


class FluidSystem{
public:
	FluidSystem(int length, int width, int height, float h): length{length}, width{width}, height{height}, gc{h} {}

	void addParticle(ParticleParams const &params);
	void render(Camera const &camera);
private:
	//graphics
	int length, width, height;
	Shader shader={"shaders/fluid.vs", "shaders/fluid.fs"};

	//particles
	std::vector<Particle> particles;

	//uniform grid
	GridCells gc;

};

#endif