#ifndef FLAME_H
#define FLAME_H

#include "vec_float.h"
#include "utility.h"
//#include "potential.h"

#include "particlesystem.hpp"


/*
class FlamePotential: public Potential{
public:
	__DEVICE__ __HOST__ FlamePotential(): Potential() {}
	__DEVICE__ __HOST__ vec3 samplePotential(vec3 const &pos) const;
};
*/
class Potential{
public:
	__DEVICE__ __HOST__ Potential();

	__DEVICE__ __HOST__ vec3 samplePotential(vec3 const &pos) const;
	__DEVICE__ __HOST__ vec3 computeCurl(vec3 const &pos) const;
	__DEVICE__ __HOST__ vec3 computeGradient(vec3 const &pos) const;

private:

	float d;
	vec3 dx, dy, dz;
};

struct FlameParticle{
	vec2 pos;
	vec2 vel;
	vec4 color;
	curandState rand;
};

class Flame: public ParticleSystem<FlameParticle>{
public:
	Flame(int n): ParticleSystem<FlameParticle>(n, {"shaders/particle.vs", "shaders/particle.fs"}) {}
};

__GLOBAL__ void initKernel(FlameParticle* dp, int n, Mouse const &mouse);
__GLOBAL__ void updateKernel(FlameParticle* dp, int n, Mouse const &mouse);


#endif