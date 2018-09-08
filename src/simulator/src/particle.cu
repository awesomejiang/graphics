#include "particle.cuh"

__DEVICE__ Particle::Particle(
	vec2 const &posIn,
	vec2 const &velIn,
	vec4 const &colorIn)
: pos(posIn), vel(velIn), color(colorIn) {}


/* ----- Start implementing kernel functions here ----- */


/* Initialization kernels */
// particle randomly generated at screen bottom
template <>
__DEVICE__ void Particle::init<InitKernel::square>(curandState* state){
	float rand = curand_uniform(state) - 0.5f;
	pos = vec2(-0.5f, rand);
	float r = 0.1f*curand_normal(state), theta = curand_uniform(state)*2*M_PI;
	pos += vec2(r*cos(theta), r*sin(theta));
	
	color = vec4(0.1f, 0.2f, 0.3f, 1.0f);
}

template <>
__DEVICE__ void Particle::init<InitKernel::bottom>(curandState* state){
	float rand = curand_uniform(state) * 2.0f - 1.0f;
	pos = vec2(rand, -1.0f);
	vel = vec2(0.0f, 0.0f);
	//color = vec4(1.0f, 1.0f, 1.0f, 1.0f);
	color = vec4(0.01f, 0.02f, 0.03f, 1.0f);
}

/* Update kernels */
// particle effects by a downside gravity. Mouse press implement a attraction force at click point.
template <>
__DEVICE__ void Particle::update<UpdateKernel::gravity>(curandState* state, Mouse const &mouse){
	float SOFTEN = 0.00000001f;
	float THRESHOLD = 0.01f;

	if(mouse.pressed){
		// gravity
		float G = 0.000005f;
		auto dist = length(mouse.pos-pos) + SOFTEN;
		vel += G/(dist*dist*dist) * (mouse.pos-pos);
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
__DEVICE__ void Particle::update<UpdateKernel::shinning>(curandState* state, Mouse const &mouse){
	//pos += mouse.pos;
}