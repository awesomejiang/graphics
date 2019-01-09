#include "particle.h"

Particle::Particle(ParticleParams const &pp): params{pp} {
	//init buffers

	createGLBuffer();

	//register buffer to cuda
	CUDA_SAFE_CALL( cudaGraphicsGLRegisterBuffer(&resource, VBO, cudaGraphicsRegisterFlagsNone) );
	CUDA_SAFE_CALL( cudaGraphicsMapResources(1, &resource) );

	//map dptr to VBO
	size_t retSz;
	CUDA_SAFE_CALL( cudaGraphicsResourceGetMappedPointer((void**)&(particle.pos), &retSz, resource) );

	deployGrid();
}

Particle::~Particle(){
	//unmap resource
	CUDA_SAFE_CALL( cudaGraphicsUnmapResources(1, &resource) );
	CUDA_SAFE_CALL( cudaGraphicsUnregisterResource(resource) );

	CUDA_SAFE_CALL( cudaFree(particle.vel) );
	CUDA_SAFE_CALL( cudaFree(particle.force) );
	CUDA_SAFE_CALL( cudaFree(particle.pressure) );
}

void Particle::setDeviceParticle(std::vector<vec3> const &p, std::vector<float> const &d){
	int sz = p.size();
	//alloc fields(except for pos field)
	CUDA_SAFE_CALL( cudaMalloc((void**)&(particle.vel), sz*sizeof(vec3)) );
	CUDA_SAFE_CALL( cudaMalloc((void**)&(particle.force), sz*sizeof(vec3)) );
	CUDA_SAFE_CALL( cudaMalloc((void**)&(particle.pressure), sz*sizeof(float)) );
	CUDA_SAFE_CALL( cudaMalloc((void**)&(particle.density), sz*sizeof(float)) );

	//copy data from input
	CUDA_SAFE_CALL( cudaMemcpy(particle.pos, p.data(), sz*sizeof(vec3), cudaMemcpyHostToDevice) );
	CUDA_SAFE_CALL( cudaMemcpy(particle.density, d.data(), sz*sizeof(float), cudaMemcpyHostToDevice) );
}


void Particle::update(DeviceGridCell const *cells){
	//update particles
	updateDensityAndPressure<<<grid, block>>>(cells, params, particle);
	CUDA_ERROR_CHECKER;
	updateForce<<<grid, block>>>(cells, params, particle);
	CUDA_ERROR_CHECKER;
	updatePositionAndVelocity<<<grid, block>>>(cells, params, particle);
	CUDA_ERROR_CHECKER;
}

void Particle::render(){
	//draw particles
	glEnable(GL_PROGRAM_POINT_SIZE);
	glBindVertexArray(VAO);
	glDrawArrays(GL_POINTS, 0, params.num);
	glBindVertexArray(0);
	glDisable(GL_PROGRAM_POINT_SIZE);
}

void Particle::createGLBuffer(){
	glGenVertexArrays(1, &VAO);
	glGenBuffers(1, &VBO);

	glBindVertexArray(VAO);

	//set VBO
	glBindBuffer(GL_ARRAY_BUFFER, VBO);
	glBufferData(GL_ARRAY_BUFFER, params.num*sizeof(vec3), particle.pos, GL_STATIC_DRAW);

	//set VAO
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(vec3), (void*)(0));

	//unbind
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindVertexArray(0);
}

void Particle::deployGrid(){
	//set block and grid
	//no communications even between threads, so configuration method is not important.
	unsigned int blockX = params.num>MAX_THREAD? MAX_THREAD: static_cast<unsigned int>(params.num);
	block = {blockX, 1, 1};

	float nGrid = static_cast<float>(params.num)/blockX;
	if(nGrid > MAX_BLOCK_X*MAX_BLOCK_Y*MAX_BLOCK_Z)
		throw std::runtime_error("Number of particles out of gpu limits.");
	else if(nGrid > MAX_BLOCK_X*MAX_BLOCK_Y){
		unsigned int z = std::ceil(nGrid/MAX_BLOCK_X/MAX_BLOCK_Y);
		grid = {MAX_BLOCK_X, MAX_BLOCK_Y, z};
	}
	else if(nGrid > MAX_BLOCK_X){
		unsigned int y = std::ceil(nGrid/MAX_BLOCK_X);
		grid = {MAX_BLOCK_X, y, 1};
	}
	else if(nGrid > 0){
		unsigned int x = std::ceil(nGrid);
		grid = {x, 1, 1};
	}
	else
		throw std::runtime_error("No particles in screen.");
}

/* update kernels */
__GLOBAL__ void updateDensityAndPressure(DeviceGridCell const *cells, ParticleParams params, DeviceParticleArray dpa){
	auto idx = getIdx();
	if(idx >= params.num)
		return ;
	auto pos = dpa.pos[idx];

	int cellDim = 2.0f/params.h;
	int x = (pos[0]+1.0f)/params.h;
	int y = (pos[1]+1.0f)/params.h;
	int z = (pos[2]+1.0f)/params.h;

//if(idx==35929) printf("den before: %f\n", dp[idx].density);
	//sum all neighbors for density
	float d = 0.0f;
	for(auto i=-1; i<2; ++i){
		for(auto j=-1; j<2; ++j){
			for(auto k=-1; k<2; ++k){
				if(x+i >=0 && x+i<cellDim && y+j >=0 && y+j<cellDim && z+k >=0 && z+k<cellDim){
					auto dc = cells[(x+i)*cellDim*cellDim + (y+j)*cellDim + (z+k)];
//if(idx==35929 && !i && !j && !k) printf("neighbor: %d\n", dc.num);
					for(int v=0; v<dc.num; ++v)
						d += weight(pos, dpa.pos[dc.index[v]], params.h);
				}
			}
		}
	}
	dpa.density[idx] = d;
//if(idx==35929) printf("den: %f\n", d);

	//update pressure based on new density
	dpa.pressure[idx] = params.k*d/params.gamma*(powf(d/1.0f, params.gamma) - 1);

//if(idx==35929) printf("pressure: %f\n", dpa.pressure[idx]);
}


__GLOBAL__ void updateForce(DeviceGridCell const *cells, ParticleParams params, DeviceParticleArray dpa){
	auto idx = getIdx();
	if(idx >= params.num)
		return ;
	auto pos = dpa.pos[idx];

	int cellDim = 2.0f/params.h;
	int x = (pos[0]+1.0f)/params.h;
	int y = (pos[1]+1.0f)/params.h;
	int z = (pos[2]+1.0f)/params.h;

	//compute external force
	vec3 f;
	float commonCoef = dpa.pressure[idx]/powf(dpa.density[idx], 2);
	for(auto i=-1; i<2; ++i){
		for(auto j=-1; j<2; ++j){
			for(auto k=-1; k<2; ++k){
				if(x+i >=0 && x+i<cellDim && y+j >=0 && y+j<cellDim && z+k >=0 && z+k<cellDim){
					auto dc = cells[(x+i)*cellDim*cellDim + (y+j)*cellDim + (z+k)]; 
					for(int v=0; v<dc.num; ++v){
						auto nbIdx = dc.index[v];
						f += -(commonCoef + dpa.pressure[nbIdx]/powf(dpa.density[nbIdx], 2))
							 * divWeight(pos, dpa.pos[nbIdx], params.h);
					}
				}
			}
		}
	}

	//add gravity here
	dpa.force[idx] = f + vec3{0.0f, -30.0f, 0.0f};
//if(idx==35929) printf("f: %f %f %f\n", f[0], f[1], f[2]);
}


__GLOBAL__ void updatePositionAndVelocity(DeviceGridCell const *cells, ParticleParams params, DeviceParticleArray dpa){
	auto idx = getIdx();
	if(idx >= params.num)
		return ;
	auto &posRef = dpa.pos[idx];
	auto &velRef = dpa.vel[idx];

	//update pos and vel
	velRef += dpa.force[idx]*params.dt;
	posRef += velRef*params.dt;

	//when meet boundary
	//bump back and get a vel decay
	for(int i=0; i<3; ++i){
		if(posRef[i] <= -1.0f){
			posRef[i] = -1.0f + params.h;
			velRef[i] = abs(velRef[i])*0.5f;
		}
		else if(posRef[i] >= 1.0f){
			posRef[i] = 1.0f - params.h;
			velRef[i] = -abs(velRef[i])*0.5f;
		}
	}
}


__DEVICE__ float weight(vec3 const &src, vec3 const &dst, float const &h){
	auto q = length(src-dst)/h;
	return 0.25 * (q<2? powf(2-q, 3): 0.0f) - (q<1? powf(1-q, 3): 0.0);
}


__DEVICE__ vec3 divWeight(vec3 const &src, vec3 const &dst, float const &h){
	auto diff = (dst-src)/h;
	auto q = length(diff);
	
	vec3 ret;
	if(q > 0 && q < 2){
		ret += 0.25f * (-3.0f*powf(q-2, 2)/q) * diff;
	}
	if(q > 0 && q < 1){
		ret -= (-3.0f*powf(q-1, 2)/q) * diff;
	}

	return 1/h * ret;
}