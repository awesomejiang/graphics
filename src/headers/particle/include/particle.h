#ifndef PARTICLE_H
#define PARTICLE_H

#include "vec_float.h"
#include "utility.h"
#include "deviceStructs.h"

#include <cuda.h>
#include <cuda_gl_interop.h>

struct ParticleParams{
	int num;
	float mass, k, gamma, h, dt;
	vec4 color;
};

class Particle{
public:
	Particle(ParticleParams const &pp);
	~Particle();

	void render(DeviceGridCell const *cell);

	//get interfaces
	int getNum() const{return params.num;}
	vec4 getColor() const {return params.color;}
	dim3 getBlock() const {return block;}
	dim3 getGrid() const {return grid;}
	DeviceParticle *getParticle() const{return particle;}

private:
	//cuda params
	ParticleParams params;
	DeviceParticle *particle = nullptr;
	dim3 block, grid;
	void deployGrid();

	//opengl params
	unsigned int width, height, VBO, VAO;
	cudaGraphicsResource_t resource = 0;
	void createGLBuffer();

};

//cuda kernels
__GLOBAL__ void initParticle(ParticleParams params, DeviceParticle *p);
__GLOBAL__ void updateParticle(DeviceGridCell const *cell, ParticleParams params, DeviceParticle *p);

//helper functions for PCISPH algorithm
__DEVICE__ float weight(vec3 const &src, vec3 const &dst, float const &h);
__DEVICE__ vec3 divWeight(vec3 const &src, vec3 const &dst, float const &h);

#endif