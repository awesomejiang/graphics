#ifndef GRAVITY_H
#define GRAVITY_H

#include <curand.h>
#include <curand_kernel.h>

#include "vec_float.h"
#include "particlesystem.hpp"

struct GravityParticle{
	vec2 pos;
	vec2 vel;
	vec4 color;
	bool live = false;
	curandState rand;
};

class Gravity: public ParticleSystem<GravityParticle>{
public:
	Gravity(int n)
	: ParticleSystem<GravityParticle>(n, {"shaders/particle.vs", "shaders/particle.fs"}) {}

	virtual void setVAO() const;
};

__GLOBAL__ void initKernel(GravityParticle* p, int n, Mouse const &mouse);
__GLOBAL__ void updateKernel(GravityParticle* p, Mouse const &mouse);

#endif