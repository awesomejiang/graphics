#ifndef CLICK_H
#define CLICK_H


#include "vec_float.h"
#include "utility.h"
#include "particlesystem.hpp"

struct ClickParticle{
	vec2 pos;
	vec2 vel;
	vec4 color;
	int lifetime;

	curandState rand;
};


class Click: public ParticleSystem<ClickParticle>{
public:
	Click(int n)
	: ParticleSystem<ClickParticle>(n, {"shaders/particle.vs", "shaders/particle.fs"}) {}
};


__GLOBAL__ void initKernel(ClickParticle* p, int n, Mouse const &mouse);
__GLOBAL__ void updateKernel(ClickParticle* p, int n, Mouse const &mouse);

#endif