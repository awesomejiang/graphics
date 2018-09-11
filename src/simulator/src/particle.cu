#include "particle.cuh"

__DEVICE__ Particle::Particle(
	vec2 const &posIn,
	vec2 const &velIn,
	vec4 const &colorIn)
: pos(posIn), vel(velIn), color(colorIn) {}


/* ----- Start implementing kernel functions here ----- */


/* --- Initialization kernels --- */
// square: particle randomly generated at screen bottom

template <>
__DEVICE__ void Particle::initKernel<InitKernelEnum::bottom>(){
	float rand = curand_uniform(randState) * 2.0f - 1.0f;
	pos = vec2(rand, -1.0f);
	vel = vec2(0.0f, 0.0f);
	//color = vec4(1.0f, 1.0f, 1.0f, 1.0f);
	color = vec4(0.1f, 0.2f, 0.3f, 1.0f);
}
// bottom: 
template <>
__DEVICE__ void Particle::initKernel<InitKernelEnum::square>(){
	float randX = (curand_uniform(randState)*2.0f - 1.0f) * HALFWIDTH;
	float randY = (curand_uniform(randState)*2.0f - 1.0f) * HALFHEIGHT;
	int index = getIdx();
	switch(index%4){
		case 0: pos = vec2(-HALFWIDTH, randY); break;
		case 1: pos = vec2(HALFWIDTH, randY); break;
		case 2: pos = vec2(randX, -HALFHEIGHT); break;
		case 3: pos = vec2(randX, HALFHEIGHT); break;
	}
	
	float r = 0.1f*curand_normal(randState), theta = curand_uniform(randState)*2*M_PI;
	pos += vec2(r*cos(theta), r*sin(theta));
	
	color = vec4(0.1f, 0.2f, 0.3f, 1.0f);
}

/* --- Update kernels --- */

// particle effects by a downside gravity. Mouse press implement a attraction force at click point.
template <>
__DEVICE__ void Particle::updateKernel<UpdateKernelEnum::gravity>(Mouse const &mouse){
	float SOFTEN = 0.00000001f;
	float THRESH = 0.01f;

	if(mouse.pressed){
		// gravity
		float G = 0.000005f;
		auto dist = length(mouse.pos-pos) + SOFTEN;
		vel += G/(dist*dist*dist) * (mouse.pos-pos);

		auto spd = length(vel);
		if(spd > THRESH)
			vel /= (spd/THRESH);
	}
	//printf("%f, %f\n", pos[0], pos[1]);

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
__DEVICE__ void Particle::updateKernel<UpdateKernelEnum::shinning>(Mouse const &mouse){
	float r = 0.001f*curand_normal(randState), theta = curand_uniform(randState)*2*M_PI;
	pos += vec2(r*cos(theta), r*sin(theta));
	
	if(mouse.firstClicked)
		color -= vec4(0.0001f, 0.0002f, 0.0003f, 0.0f);
}

/* ------ End of kernel implementations ------ */



__DEVICE__ void Particle::init(InitKernelEnum const &ik, curandState *state){
	randState = state;

	//find correct template init kernel
	switch(ik){
		case InitKernelEnum::bottom : initKernel<InitKernelEnum::bottom>(); break;
		case InitKernelEnum::square : initKernel<InitKernelEnum::square>(); break;
	}
}

__DEVICE__ void Particle::update(UpdateKernelEnum const &uk, Mouse const &mouse){
	//find correct template update kernel

	switch(uk){
		case UpdateKernelEnum::gravity : updateKernel<UpdateKernelEnum::gravity>(mouse);
		case UpdateKernelEnum::shinning : updateKernel<UpdateKernelEnum::shinning>(mouse);
	}
}
