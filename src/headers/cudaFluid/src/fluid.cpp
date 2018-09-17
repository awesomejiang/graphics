#include "fluid.h"

#define MAX_THREAD_X 16
#define MAX_THREAD_Y 16
#define MAX_BLOCK_X 65535ll
#define MAX_BLOCK_Y 65535ll

Fluid::Fluid(int const &width, int const &height)
: width{width}, height{height}, size{width*height} {
	//create VBO
	glGenVertexArrays(1, &VAO);
	glGenBuffers(1, &VBO);
	glBindVertexArray(VAO);
	//set VBO
	glBindBuffer(GL_ARRAY_BUFFER, VBO);
	glBufferData(GL_ARRAY_BUFFER, size*sizeof(FluidState), nullptr, GL_STATIC_DRAW);
	//set VAO
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, sizeof(vec2)/sizeof(GL_FLOAT), GL_FLOAT, GL_FALSE,
		sizeof(FluidState), OFFSETOF(FluidState, pos));
	glEnableVertexAttribArray(1);
	glVertexAttribPointer(1, sizeof(vec2)/sizeof(GL_FLOAT), GL_FLOAT, GL_FALSE,
		sizeof(FluidState), OFFSETOF(FluidState, vel));
	glEnableVertexAttribArray(2);
	glVertexAttribPointer(2, sizeof(float)/sizeof(GL_FLOAT), GL_FLOAT, GL_FALSE,
		sizeof(FluidState), OFFSETOF(FluidState, p));
	//unbind
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindVertexArray(0);

	//init cuda
	deployGrid();
	//TODO: performance improve: shared memory + use ghost cell method 
	CUDA_SAFE_CALL( cudaMalloc((void**)&starState, size*sizeof(FluidState)) );
	CUDA_SAFE_CALL( cudaMalloc((void**)&indexing, sizeof(Indexing)) );
	//register buffer to cuda
	CUDA_SAFE_CALL( cudaGraphicsGLRegisterBuffer(&resource, VBO, cudaGraphicsRegisterFlagsNone) );
	CUDA_SAFE_CALL( cudaGraphicsMapResources(1, &resource) );
	//map dptr to VBO
	size_t retSz;
	CUDA_SAFE_CALL( cudaGraphicsResourceGetMappedPointer((void**)&currState, &retSz, resource) );
}

Fluid::~Fluid(){
	//unmap resource
	CUDA_SAFE_CALL( cudaGraphicsUnmapResources(1, &resource) );
	CUDA_SAFE_CALL( cudaGraphicsUnregisterResource(resource) );
	//free memory
	CUDA_SAFE_CALL( cudaFree(starState) );
	CUDA_SAFE_CALL( cudaFree(indexing) );
}


void Fluid::deployGrid(){
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
}

void Fluid::render(){
	if(firstIteration == true){
		initIndexing<<<grid, block>>>(width, height, indexing);
		initFluid<<<grid, block>>>(indexing, currState);
		firstIteration = false;
	}

	//solve v*: advection term
	advect<<<grid, block>>>(indexing, currState, starState);
	//solve v*: diffusion term
	diffusion<<<grid, block>>>(indexing, currState, starState);
	//solve p: posiion equation by Jacobi iterative solver
		//hack: calculate div(v*) and put into curr.vel[0]
	div<<<grid, block>>>(indexing, currState, starState);
	for(auto i=0; i<10; ++i){
		pressure<<<grid, block>>>(indexing, currState, starState);
		swapPressure<<<grid, block>>>(indexing, currState, starState);
	}
	//correct v* to final v
	correction<<<grid, block>>>(indexing, currState, starState);

	//draw
	shader.use();

	glBindVertexArray(VAO);
	glDrawArrays(GL_POINTS, 0, size);
	glBindVertexArray(0);
}


__GLOBAL__ void initIndexing(int w, int h, Indexing *indexing){
	//in "first" thread, init the indexing object
	if(threadIdx.x == 0 && threadIdx.y == 0 && blockIdx.x == 0 && blockIdx.y == 0)
		*indexing = Indexing{w, h};
}

__GLOBAL__ void initFluid(Indexing *indexing, FluidState *currState){
	auto idx = indexing->getIdx();
	if(idx == -1)
		return ;
	FluidState& cr = currState[idx];

	auto w = indexing->w, h = indexing->h;
	auto x = static_cast<float>(idx%w)/w * 2.0f - 1.0f;
	auto y = -static_cast<float>(idx/w)/h * 2.0f + 1.0f;
	cr.pos = {x, y};
	cr.vel = {0.0f, 0.0f};
	cr.p = static_cast<float>(idx%w)/w;
}

//v* = v - (v*div(v))
__GLOBAL__ void advect(Indexing *indexing, FluidState *currState, FluidState *starState){
	auto idx = indexing->getIdx();
	if(idx == -1)
		return ;
	vec2& currv = currState[idx].vel;
	vec2& starv = starState[idx].vel;
	vec2& rightv = currState[indexing->getRight()].vel;
	vec2& topv = currState[indexing->getTop()].vel;

	auto vx = 
		currv[0]*(rightv[0] - currv[0])
		+ currv[1]*(topv[0] - currv[0]);
	auto vy = 
		currv[0]*(rightv[1] - currv[1])
		+ currv[1]*(topv[1] - currv[1]);
	starv = currv - dt * vec2{vx, vy};
}

//v* += laplace(v)
__GLOBAL__ void diffusion(Indexing *indexing, FluidState *currState, FluidState *starState){
	auto idx = indexing->getIdx();
	if(idx == -1)
		return ;
	vec2& currv = currState[idx].vel;
	vec2& starv = starState[idx].vel;
	vec2& leftv = currState[indexing->getLeft()].vel;
	vec2& rightv = currState[indexing->getRight()].vel;
	vec2& topv = currState[indexing->getTop()].vel;
	vec2& bottomv = currState[indexing->getBottom()].vel;

	auto vx = leftv[0] + rightv[0] + topv[0] + bottomv[0] - 4*currv[0];
	auto vy = leftv[1] + rightv[1] + topv[1] + bottomv[1] - 4*currv[1];

	starv += vec2{vx, vy};
}

//jacobi iterative solver: div(v*)
__GLOBAL__ void div(Indexing *indexing, FluidState *currState, FluidState *starState){
	auto idx = indexing->getIdx();
	if(idx == -1)
		return ;
	vec2& currv = currState[idx].vel;
	vec2& starv = starState[idx].vel;
	vec2& rightv = starState[indexing->getRight()].vel;
	vec2& topv = starState[indexing->getTop()].vel;

	currv[0] = (rightv[0] - starv[0] + topv[1] - starv[1]);
}

__GLOBAL__ void pressure(Indexing *indexing, FluidState *currState, FluidState *starState){
	auto idx = indexing->getIdx();
	if(idx == -1)
		return ;
	float div = currState[idx].vel[0];
	float& starp = starState[idx].p;
	float& leftp = currState[indexing->getLeft()].p;
	float& rightp = currState[indexing->getRight()].p;
	float& topp = currState[indexing->getTop()].p;
	float& bottomp = currState[indexing->getBottom()].p;

	starp = (div + leftp + rightp + topp + bottomp)/4;
}

__GLOBAL__ void swapPressure(Indexing *indexing, FluidState *currState, FluidState *starState){
	auto idx = indexing->getIdx();
	if(idx == -1)
		return ;
	float& currp = currState[idx].p;
	float& starp = starState[idx].p;

	float temp = currp;
	currp = starp;
	starp = temp;
}

__GLOBAL__ void correction(Indexing *indexing, FluidState *currState, FluidState *starState){
	auto idx = indexing->getIdx();
	if(idx == -1)
		return ;
	vec2& currv = currState[idx].vel;
	vec2& starv = starState[idx].vel;
	float& currp = currState[idx].p;
	float& rightp = currState[indexing->getRight()].p;
	float& topp = currState[indexing->getTop()].p;

	currv = starv - vec2{rightp-currp, topp-currp};
}


/***** Indexing class implementation *****/
__DEVICE__ Indexing::Indexing(int const &w, int const &h): w{w}, h{h} {}

__DEVICE__ int Indexing::getIdx(){
	auto x = blockDim.x*blockIdx.x + threadIdx.x;
	auto y = blockDim.y*blockIdx.y + threadIdx.y;
	if(x < w && y < h)
		return y*w + x;
	else
		return -1;
}

__DEVICE__ int Indexing::getLeft(){
	auto x = blockDim.x*blockIdx.x + threadIdx.x;
	auto y = blockDim.y*blockIdx.y + threadIdx.y;
	if(x < w && y < h){
		x = (x==0)? w-1: x;
		return y*w + x-1;
	}
	else
		return -1;
}

__DEVICE__ int Indexing::getRight(){
	auto x = blockDim.x*blockIdx.x + threadIdx.x;
	auto y = blockDim.y*blockIdx.y + threadIdx.y;
	if(x < w-1 && y < h){
		x = (x==w-1)? 0: x;
		return y*w + x+1;
	}
	else
		return -1;
}

__DEVICE__ int Indexing::getTop(){
	auto x = blockDim.x*blockIdx.x + threadIdx.x;
	auto y = blockDim.y*blockIdx.y + threadIdx.y;
	if(x < w && y < h){
		y = (y==0)? h-1: y;
		return (y-1)*w + x;
	}
	else
		return -1;
}

__DEVICE__ int Indexing::getBottom(){
	auto x = blockDim.x*blockIdx.x + threadIdx.x;
	auto y = blockDim.y*blockIdx.y + threadIdx.y;
	if(x < w && y < h-1){
		y = (y==h-1)? 0: y;
		return (y+1)*w + x;
	}
	else
		return -1;
}