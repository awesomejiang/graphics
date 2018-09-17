#ifndef TAILING_H
#define TAILING_H


#include "vec_float.h"
#include "utility.h"
#include "particlesystem.hpp"

struct TailingParticle{
	vec2 pos;
	vec2 vel;
	vec4 color;
	int lifetime;

	curandState rand;
};


class Tailing: public ParticleSystem<TailingParticle>{
public:
	Tailing(int n)
	: ParticleSystem<TailingParticle>(n, {"shaders/particle.vs", "shaders/particle.fs"}) {}
};


__GLOBAL__ void initKernel(TailingParticle* p, int n, Mouse const &mouse);
__GLOBAL__ void updateKernel(TailingParticle* p, int n, Mouse const &mouse);

#endif