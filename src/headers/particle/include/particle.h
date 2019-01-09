#ifndef PARTICLE_H
#define PARTICLE_H

#include "vec_float.h"
#include "utility.h"
#include "structs.h"

#include <cuda.h>
#include <cuda_gl_interop.h>

#include <vector>

class Particle{
public:
	Particle(ParticleParams const &pp);
	~Particle();

	void setDeviceParticle(std::vector<vec3> const &p, std::vector<float> const &d);

	void update(DeviceGridCell const *cells);
	void render();

	//get interfaces
	ParticleParams getParams() const{return params;}
	dim3 getBlock() const {return block;}
	dim3 getGrid() const {return grid;}
	DeviceParticleArray getParticle() const{return particle;}

private:
	//cuda params
	ParticleParams params;
	DeviceParticleArray particle;
	dim3 block, grid;
	void deployGrid();

	//opengl params
	unsigned int width, height, VBO, VAO;
	cudaGraphicsResource_t resource = 0;
	void createGLBuffer();

};

//cuda kernels
__GLOBAL__ void updateDensityAndPressure(DeviceGridCell const *cells, ParticleParams params, DeviceParticleArray dpa);
__GLOBAL__ void updateForce(DeviceGridCell const *cells, ParticleParams params, DeviceParticleArray dpa);
__GLOBAL__ void updatePositionAndVelocity(DeviceGridCell const *cells, ParticleParams params, DeviceParticleArray dpa);

//helper functions for PCISPH algorithm
__DEVICE__ float weight(vec3 const &src, vec3 const &dst, float const &h);
__DEVICE__ vec3 divWeight(vec3 const &src, vec3 const &dst, float const &h);

#endif