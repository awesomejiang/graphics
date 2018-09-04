#include "particle.cuh"

__device__ __host__ Particle::Particle(
	UpdateKernel const &update,
	vec2 const &posIn,
	vec2 const &velIn,
	vec4 const &colorIn)
: pos(posIn), vel(velIn), color(colorIn), updateK(update) {}


__device__ __host__ void Particle::update(vec2 const &forceCenter, bool pressed){
	if(updateK == UpdateKernel::gravity)
		gravityKernel(*this, forceCenter, pressed);
	//else if(k == Kernel::none)
	// do nothing

	//printf("%f, %f\n", pos[0], pos[1]);
}

/* ----- Start implementing kernel functions here ----- */
#define SOFTEN 0.00000001f
#define THRESHOLD 0.01f


__device__ __host__ void Particle::gravityKernel(Particle &p, vec2 const &forceCenter, bool pressed){
	if(pressed){
		// gravity
		float G = 0.000005f;
		auto dist = length(forceCenter-p.pos) + SOFTEN;
		p.vel += G/(dist*dist*dist) * (forceCenter-p.pos);

		auto speed = length(p.vel);
		p.vel = speed>THRESHOLD? norm(p.vel)*THRESHOLD: p.vel;
	}
	//pos move
	p.pos += p.vel;
	//add a downside force field
	float g = 0.000005f;
	p.vel += vec2(0.0f, -g);

	//when meet boundary
	//bump back and get a vel decay
	if(p.pos[0] < -1.0f){
		p.pos[0] = -1.0f;
		p.vel[0] = abs(p.vel[0]) * 0.5;
	}
	else if(p.pos[0] > 1.0f){
		p.pos[0] = 1.0f;
		p.vel[0] = -abs(p.vel[0]) * 0.5;
	}
	if(p.pos[1] > 1.0f){
		p.pos[1] = 1.0f;
		p.vel[1] = -abs(p.vel[1]) * 0.5;
	}
	else if(p.pos[1] < -1.0f){
		p.pos[1] = -1.0f;
		p.vel[1] = abs(p.vel[1]) * 0.5;
	}
}