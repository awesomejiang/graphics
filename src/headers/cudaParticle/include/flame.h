#ifndef FLAME_H
#define FLAME_H

#include "vec_float.h"
#include "utility.h"
#include "particlesystem.hpp"


struct FlameParticle{
	vec2 pos = {0.0f, 0.0f};
	vec2 vel = {0.0f, 0.0f};
	vec4 color = {1.0f, 1.0f, 1.0f, 1.0f};
	bool live = false;
};

class Flame: public ParticleSystem<FlameParticle>{
public:
	Flame(): ParticleSystem<FlameParticle>(1024, {"shaders/particle.vs", "shaders/particle.fs"}) {}
};

__GLOBAL__ void updateKernel(FlameParticle* dp, int n, Mouse const &mouse);


#endif