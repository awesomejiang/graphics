#ifndef SQUARE_H
#define SQUARE_H

#include <curand.h>
#include <curand_kernel.h>

#include "vec_float.h"
#include "particlesystem.hpp"

struct SquareParticle{
	vec2 pos;
	vec2 vel;
	vec4 color;
	bool live = false;
	curandState rand;
};

class Square: public ParticleSystem<SquareParticle>{
public:
	Square(int n)
	: ParticleSystem<SquareParticle>(n, {"shaders/particle.vs", "shaders/particle.fs"}) {}

	virtual void setVAO() const;
};


__GLOBAL__ void initKernel(SquareParticle* p, int n, Mouse const &mouse);
__GLOBAL__ void updateKernel(SquareParticle* p, Mouse const &mouse);

#endif