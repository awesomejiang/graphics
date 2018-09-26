#include "fluid.h"
#include "mathsolver.hpp"

#define MAX_THREAD_X 16
#define MAX_THREAD_Y 16
#define MAX_BLOCK_X 65535ll
#define MAX_BLOCK_Y 65535ll
#define IDX 240400

Fluid::Fluid(int const &width, int const &height)
: width{width}, height{height}, size{width*height} {}

Fluid::~Fluid(){
	if(!firstIteration){
		//unmap resource
		CUDA_SAFE_CALL( cudaGraphicsUnmapResources(4, resource) );
		for(auto i=0; i<VBO_NUM; ++i)
			CUDA_SAFE_CALL( cudaGraphicsUnregisterResource(resource[i]) );
		//free memory
		CUDA_SAFE_CALL( cudaFree(indexing) );
		CUDA_SAFE_CALL( cudaFree(div) );
		CUDA_SAFE_CALL( cudaFree(oldV) );
		CUDA_SAFE_CALL( cudaFree(tempV) );
		CUDA_SAFE_CALL( cudaFree(oldP) );
		CUDA_SAFE_CALL( cudaFree(oldC) );
		CUDA_SAFE_CALL( cudaFree(tempC) );

		CUDA_SAFE_CALL( cudaFree(deviceMouse) );
	}
}

void Fluid::initGL(){
	//create VAO
	glGenVertexArrays(1, &VAO);
	glGenBuffers(VBO_NUM, VBO);
	//set VBOs
	size_t sz[VBO_NUM] = {sizeof(vec2), sizeof(vec2), sizeof(float), sizeof(vec3)};
	glBindVertexArray(VAO);
	for(auto i=0; i<VBO_NUM; ++i){
		//create VBO
		glBindBuffer(GL_ARRAY_BUFFER, VBO[i]);
		glBufferData(GL_ARRAY_BUFFER, size*sz[i], nullptr, GL_STATIC_DRAW);
		//set VAO
		glEnableVertexAttribArray(i);
		glVertexAttribPointer(i, sz[i]/sizeof(GL_FLOAT), GL_FLOAT, GL_FALSE,
			sz[i], (void*)0);
		//unbind VBO
		glBindBuffer(GL_ARRAY_BUFFER, 0);
	}
	//unbind VAO
	glBindVertexArray(0);
}

void Fluid::initCuda(){
	//deploy grid
		//To reduce communication between blocks, set:
		//1. 16*16 threads per block
		//2. blocks as 2d array(easy for indexing)
	block = {MAX_THREAD_X, MAX_THREAD_Y, 1};

	unsigned int gridX = std::ceil(static_cast<float>(width)/MAX_THREAD_X);
	unsigned int gridY = std::ceil(static_cast<float>(height)/MAX_THREAD_Y);

	//assume grid x and y is under block limits(which is large enough for pixel drawing)
	if(gridX > MAX_BLOCK_X || gridY > MAX_BLOCK_Y || gridX == 0 || gridY == 0)
		throw std::runtime_error("Number of particles out of gpu limits.");
	grid = {gridX, gridY};

	//TODO: performance improve: shared memory + use ghost cell method
	CUDA_SAFE_CALL( cudaMalloc((void**)&indexing, sizeof(Indexing)) );
	CUDA_SAFE_CALL( cudaMalloc((void**)&div, size*sizeof(float)) );
	CUDA_SAFE_CALL( cudaMalloc((void**)&oldV, size*sizeof(vec2)) );
	CUDA_SAFE_CALL( cudaMalloc((void**)&tempV, size*sizeof(vec2)) );
	CUDA_SAFE_CALL( cudaMalloc((void**)&oldP, size*sizeof(float)) );
	CUDA_SAFE_CALL( cudaMalloc((void**)&oldC, size*sizeof(vec3)) );
	CUDA_SAFE_CALL( cudaMalloc((void**)&tempC, size*sizeof(vec3)) );
	//register buffer to cuda
	for(auto i=0; i<VBO_NUM; ++i)
		CUDA_SAFE_CALL( cudaGraphicsGLRegisterBuffer(resource+i, VBO[i], cudaGraphicsRegisterFlagsNone) );
	CUDA_SAFE_CALL( cudaGraphicsMapResources(4, resource) );
	//get mapped pointer
	size_t retSz;
	CUDA_SAFE_CALL( cudaGraphicsResourceGetMappedPointer((void**)&pos, &retSz, resource[0]) );
	CUDA_SAFE_CALL( cudaGraphicsResourceGetMappedPointer((void**)&currV, &retSz, resource[1]) );
	CUDA_SAFE_CALL( cudaGraphicsResourceGetMappedPointer((void**)&currP, &retSz, resource[2]) );
	CUDA_SAFE_CALL( cudaGraphicsResourceGetMappedPointer((void**)&currC, &retSz, resource[3]) );

	initIndexing<<<grid, block>>>(width, height, indexing);
	initFluid<<<grid, block>>>(indexing, pos, currV, currP, currC);
}

void Fluid::render(Mouse const &mouse, float const &t){
	//init
	if(firstIteration == true){
		initGL();
		initCuda();
		firstIteration = false;
	}

	//set mouse position to device
	deviceMouse = nullptr;
	auto sz = sizeof(Mouse);
	CUDA_SAFE_CALL( cudaMalloc((void**)&deviceMouse, sz) );
	CUDA_SAFE_CALL( cudaMemcpy(deviceMouse, &mouse, sz, cudaMemcpyHostToDevice) );

	//update dt
	dt = t;

	//do math here
	solveMomentum();

	correctPressure();

	colorSpread();

	//draw
	shader.use();

    //glEnable(GL_BLEND);
    //glBlendFunc(GL_SRC_ALPHA, GL_ONE);

	glBindVertexArray(VAO);
	glDrawArrays(GL_POINTS, 0, size);
	glBindVertexArray(0);

	//glDisable(GL_BLEND);
	//usleep(100000);
}

void Fluid::solveMomentum(){
	//add force term
	addForce<<<grid, block>>>(indexing, deviceMouse, currV, dt);

	//swap data from last step to oldV
	std::swap(oldV, currV);
	//call backtrace method for advection term
	advect<<<grid, block>>>(indexing, currV, oldV, oldV, dt);
	//correct for BC
	dirichletBC<<<grid, block>>>(indexing, currV);

	//swap intermediate result into oldV
	std::swap(oldV, currV);
	//implement gauss siedel solver for diffusion term
	CUDA_SAFE_CALL( cudaMemcpy(tempV, oldV, size*sizeof(vec2), cudaMemcpyDeviceToDevice) );
	for(auto i=0; i<HALFITERATION*2; ++i){
		diffusionStep<<<grid, block>>>(indexing, currV, tempV, oldV, dt*NIU);
		dirichletBC<<<grid, block>>>(indexing, currV);
		std::swap(tempV, currV);
	}
}

void Fluid::correctPressure(){
	//copy data from currP to oldP
	CUDA_SAFE_CALL( cudaMemcpy(oldP, currP, size*sizeof(float), cudaMemcpyDeviceToDevice) );
	//call Jacobi solver for pressure poisson equation
	divergence<<<grid, block>>>(indexing, div, currV);
	dirichletBC<<<grid, block>>>(indexing, div);
	for(auto i=0; i<HALFITERATION*2; ++i){
		poissonStep<<<grid, block>>>(indexing, currP, oldP, div);
		neumannBC<<<grid, block>>>(indexing, currP);
		//CUDA_SAFE_CALL( cudaMemcpy(oldP, currP, size*sizeof(float), cudaMemcpyDeviceToDevice) );
		std::swap(oldP, currP);
	}

	//substract grad(P) from former intermiedate velocity
	subGrad<<<grid, block>>>(indexing, currV, currP);
	dirichletBC<<<grid, block>>>(indexing, currV);
}

void Fluid::colorSpread(){
	//add source term
	addDye<<<grid, block>>>(indexing, deviceMouse, currC, dt);

	//swap data from last step to oldV
	std::swap(oldC, currC);
	//call backtrace method for advection term
	advect<<<grid, block>>>(indexing, currC, oldC, currV, dt);

	//swap intermediate result into oldC
	std::swap(oldC, currC);
	//implement gauss siedel solver for diffusion term
	CUDA_SAFE_CALL( cudaMemcpy(tempC, oldC, size*sizeof(vec3), cudaMemcpyDeviceToDevice) );
	for(auto i=0; i<HALFITERATION*2; ++i){
		diffusionStep<<<grid, block>>>(indexing, currC, tempC, oldC, dt*NIU);
		std::swap(tempC, currC);
	}
}


__GLOBAL__ void initIndexing(int w, int h, Indexing *indexing){
	//in "first" thread, init the indexing object
	if(threadIdx.x == 0 && threadIdx.y == 0 && blockIdx.x == 0 && blockIdx.y == 0)
		*indexing = Indexing{w, h};
}

__GLOBAL__ void initFluid(Indexing *indexing, vec2* pos, vec2* v, float* p, vec3* c){
	auto idx = indexing->getIdx();
	if(idx == -1)
		return ;

	pos[idx] = indexing->getPos();
	v[idx] = {0.0f, 0.0f};
	p[idx] = 0.0f;
	c[idx] = {0.0f, 0.0f, 0.0f}; 
}


// out force source
__GLOBAL__ void addForce(Indexing *indexing, Mouse *mouse, vec2 *vel, float dt){
	auto idx = indexing->getIdx();
	if(idx == -1)
		return ;

	auto dist = indexing->getPos() - mouse->pos;
	if(mouse->pressed && length(dist) < 0.3f)
		vel[idx] += norm(dist)*1000.0f*dt;
}

__GLOBAL__ void addDye(Indexing *indexing, Mouse *mouse, vec3 *color, float dt){
	auto idx = indexing->getIdx();
	if(idx == -1)
		return ;

	//add dye, color based on move direction
	//vec3 dye{mouse->dir[0], mouse->dir[1], 0.0f};
	vec3 dye{1.0f, 0.5f, 0.0f};
	auto dist = indexing->getPos() - mouse->pos;
	if(mouse->pressed && length(dist) < 0.1f)
		color[idx] += dye*2.0f*dt;
}