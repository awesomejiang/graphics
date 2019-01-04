#include "particle.h"

Particle::Particle(ParticleParams const &pp): params{pp} {
	//init buffers

	createGLBuffer();

	//register buffer to cuda
	CUDA_SAFE_CALL( cudaGraphicsGLRegisterBuffer(&resource, VBO, cudaGraphicsRegisterFlagsNone) );
	CUDA_SAFE_CALL( cudaGraphicsMapResources(1, &resource) );

	//map dptr to VBO
	size_t retSz;
	CUDA_SAFE_CALL( cudaGraphicsResourceGetMappedPointer((void**)&particle, &retSz, resource) );

	deployGrid();

	initParticle<<<grid, block>>>(params, particle);
	CUDA_ERROR_CHECKER;
}

Particle::~Particle(){
	//unmap resource
	CUDA_SAFE_CALL( cudaGraphicsUnmapResources(1, &resource) );
	CUDA_SAFE_CALL( cudaGraphicsUnregisterResource(resource) );
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

void Particle::createGLBuffer(){
	glGenVertexArrays(1, &VAO);
	glGenBuffers(1, &VBO);

	glBindVertexArray(VAO);

	//set VBO
	glBindBuffer(GL_ARRAY_BUFFER, VBO);
	glBufferData(GL_ARRAY_BUFFER, params.num*sizeof(DeviceParticle), particle, GL_STATIC_DRAW);

	//set VAO
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(DeviceParticle), (void*)(0));

	//unbind
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindVertexArray(0);
}

void Particle::render(DeviceGridCell const *cell){
	//update particles
	updateParticle<<<grid, block>>>(cell, params, particle);
	CUDA_ERROR_CHECKER;

	glBindVertexArray(VAO);
	glDrawArrays(GL_POINTS, 0, params.num);
	glBindVertexArray(0);
}

__GLOBAL__ void initParticle(ParticleParams params, DeviceParticle *p){
	auto idx = getIdx();
	if(idx >= params.num)
		return ;

	auto &pr = p[idx];
	pr.pos[0] = -1.0f + 0.02f * (idx/1000);
	pr.pos[1] = 0.02f * (idx%1000)/100;
	pr.pos[2] = -1.0f + 0.02f * (idx%100);

	pr.vel = {0.0f, 0.0f, 0.0f};
	pr.density = 1.0f;
	pr.pressure = 1.0f;
}

__GLOBAL__ void updateParticle(DeviceGridCell const *cell, ParticleParams params, DeviceParticle *p){
	auto idx = getIdx();
	if(idx >= params.num)
		return ;

	auto &pr = p[idx];

	auto cellDim = static_cast<int>(1/params.h);

	float d = 0.0f;
	//sum all neighbors for density
	int x = (pr.pos[0]+1.0f)/2*cellDim;
	int y = (pr.pos[1]+1.0f)/2*cellDim;
	int z = (pr.pos[2]+1.0f)/2*cellDim;
//if(idx==35929) printf("dim: %d %d %d\n", x, y, z);
//if(idx==35929) printf("num: %d\n", cell[x*100+y*10+z].num);
	for(auto i=-1; i<2; ++i){
		for(auto j=-1; j<2; ++j){
			for(auto k=-1; k<2; ++k){
				if(x+i >=0 && x+i<cellDim && y+j >=0 && y+j<cellDim && z+k >=0 && z+k<cellDim){
					auto dc = cell[(x+i)*cellDim*cellDim + (y+j)*cellDim + (z+k)];
					for(int v=0; v<dc.num; ++v)
						d += weight(pr.pos, dc.pos[v], params.h);
				}
			}
		}
	}
	pr.density = params.mass * d;
//if(idx==35929) printf("den: %f\n", d);

	//compute pressure based on density
	pr.pressure = params.k*pr.density/params.gamma*(powf(pr.density/1.0f, params.gamma) - 1);
//if(idx==35929) printf("p: %f\n", pr.pressure);
//if(idx==35929) printf("pos: %f %f %f\n", pr.pos[0], pr.pos[1], pr.pos[2]);

	//compute external force
	vec3 f;
	for(auto i=-1; i<2; ++i){
		for(auto j=-1; j<2; ++j){
			for(auto k=-1; k<2; ++k){
				if(x+i >=0 && x+i<cellDim && y+j >=0 && y+j<cellDim && z+k >=0 && z+k<cellDim){
					auto dc = cell[(x+i)*cellDim*cellDim + (y+j)*cellDim + (z+k)]; 
					for(int v=0; v<dc.num; ++v)
						f += (pr.pressure/(pr.density*pr.density))*divWeight(pr.pos, dc.pos[v], params.h);
				}
			}
		}
	}
	f *= -2*params.mass*params.mass;
	f += {0.0f, -100.0f, 0.0f};
//if(idx==35929) printf("f: %f %f %f\n", f[0], f[1], f[2]);

	//update pos and vel
	pr.vel += f*params.dt/params.mass;
	pr.pos += pr.vel*params.dt;

	//when meet boundary
	//bump back and get a vel decay
	for(int i=0; i<3; ++i){
		if(pr.pos[i] < -1.0f){
			pr.pos[i] = -0.99f;
			pr.vel[i] = abs(pr.vel[i])*0.5f;
		}
		else if(pr.pos[i] > 1.0f){
			pr.pos[i] = 0.99f;
			pr.vel[i] = -abs(pr.vel[i])*0.5f;
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