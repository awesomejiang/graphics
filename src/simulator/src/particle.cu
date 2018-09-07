#include "particle.cuh"

__DEVICE__ __HOST__ Particle::Particle(
	UpdateKernel const &update,
	vec2 const &posIn,
	vec2 const &velIn,
	vec4 const &colorIn)
: pos(posIn), vel(velIn), color(colorIn), updateK(update) {}


/* ----- Start implementing kernel functions here ----- */
#define SOFTEN 0.00000001f
#define THRESHOLD 0.01f


//particle effects by a downside gravity. Mouse press implement a attraction force at click point.
template <>
__DEVICE__ __HOST__ void Particle::kernel<UpdateKernel::gravity>(Mouse const &mouse){
	if(mouse.pressed){
		// gravity
		float G = 0.000005f;
		auto dist = length(mouse.pos-pos) + SOFTEN;
		vel += G/(dist*dist*dist) * (mouse.pos-pos);

		auto speed = length(vel);
		vel = speed>THRESHOLD? norm(vel)*THRESHOLD: vel;
	}
	//pos move
	pos += vel;
	//add a downside force field
	float g = 0.000005f;
	vel += vec2(0.0f, -g);

	//when meet boundary
	//bump back and get a vel decay
	if(pos[0] < -1.0f){
		pos[0] = -1.0f;
		vel[0] = abs(vel[0]) * 0.5;
	}
	else if(pos[0] > 1.0f){
		pos[0] = 1.0f;
		vel[0] = -abs(vel[0]) * 0.5;
	}
	if(pos[1] > 1.0f){
		pos[1] = 1.0f;
		vel[1] = -abs(vel[1]) * 0.5;
	}
	else if(pos[1] < -1.0f){
		pos[1] = -1.0f;
		vel[1] = abs(vel[1]) * 0.5;
	}
}

template <>
__DEVICE__ __HOST__ void Particle::kernel<UpdateKernel::shinning>(Mouse const &mouse){
	pos += vel;
	
	if(length(vel) > 0.00000001f)
		vel /= 1.2f;
	else{
		vel = vec2(0.0f, 0.0f);
		pos += norm(mouse.pos) * 0.1f;
		printf("%f\n", norm(mouse.pos)[0]);
	}
}


__DEVICE__ __HOST__ void Particle::update(Mouse const &mouse){
	UpdateKernel const arg = UpdateKernel::shinning;
	kernel<arg>(mouse);
}
