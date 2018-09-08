#include "spirit.cuh"

#define MAX_THREAD 1024
#define MAX_BLOCK_X 65535ll
#define MAX_BLOCK_Y 65535ll
#define MAX_BLOCK_Z 65535ll

Spirit::Spirit(int const &n)
: resource(0),
  nParticle(n),
  deviceParticles(nullptr),
  deviceRandStates(nullptr),
  pShader("shaders/particle.vs", "shaders/particle.fs"){
	createVBO();
	setCallBacks();
	initCuda();
}

Spirit::~Spirit(){
	//unmap resource
	CUDA_SAFE_CALL( cudaGraphicsUnmapResources(1, &resource) );
	CUDA_SAFE_CALL( cudaGraphicsUnregisterResource(resource) );

	//free
	CUDA_SAFE_CALL( cudaFree(deviceParticles) );
	CUDA_SAFE_CALL( cudaFree(deviceRandStates) );
}

void Spirit::createVBO(){
	glGenVertexArrays(1, &VAO);
	glGenBuffers(1, &VBO);

	glBindVertexArray(VAO);

	//set VBO
	glBindBuffer(GL_ARRAY_BUFFER, VBO);
	glBufferData(GL_ARRAY_BUFFER, nParticle*sizeof(Particle), deviceParticles, GL_STATIC_DRAW);

	//set VAO
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(Particle), (void*)(0));
	glEnableVertexAttribArray(1);
	glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, sizeof(Particle), (void*)(sizeof(vec2)*1));
	glEnableVertexAttribArray(2);
	glVertexAttribPointer(2, 4, GL_FLOAT, GL_FALSE, sizeof(Particle), (void*)(sizeof(vec2)*2));

	//unbind
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindVertexArray(0);

}

void Spirit::setCallBacks() const{
	//glfwSetCursorPosCallback(scene.window, [](GLFWWindow *window, float x, float y){});
}


void Spirit::initCuda(){
	deployGrid();

	// cuda allocations
	CUDA_SAFE_CALL( cudaMalloc((void**)&deviceParticles, nParticle*sizeof(Particle)) );
	CUDA_SAFE_CALL( cudaMalloc((void**)&deviceRandStates, nParticle*sizeof(curandState)) );

	//register buffer to cuda
	CUDA_SAFE_CALL( cudaGraphicsGLRegisterBuffer(&resource, VBO, cudaGraphicsRegisterFlagsNone) );

	//map dptr to VBO
	size_t retSz;
	Particle* dp;
	CUDA_SAFE_CALL( cudaGraphicsMapResources(1, &resource) );
	CUDA_SAFE_CALL( cudaGraphicsResourceGetMappedPointer((void**)&dp, &retSz, resource) );

	//run cuda kernel
	initKernel<<<grid, block>>>(dp, deviceRandStates, nParticle);
	CUDA_ERROR_CHECKER;
	CUDA_SAFE_CALL( cudaDeviceSynchronize() );
}

__GLOBAL__ void initKernel(Particle* dp, curandState* dr, int n){
    int index = getIdx();
    if(index > n)
    	return ;

    curandState *state = &dr[index];
	//init curand states
	curand_init(clock64(), index, 0, state);

	dp[index].init<InitKernel::bottom>(state);
}


void Spirit::render(Mouse const &mouse){
	//get mouse position
	Mouse* deviceMouse = nullptr;
	auto sz = sizeof(Mouse);
	CUDA_SAFE_CALL( cudaMalloc((void**)&deviceMouse, sz) );
	CUDA_SAFE_CALL( cudaMemcpy(deviceMouse, &mouse, sz, cudaMemcpyHostToDevice) );

	//map dptr to VBO
	size_t retSz;
	Particle *dptr = nullptr;
	CUDA_SAFE_CALL( cudaGraphicsResourceGetMappedPointer((void**)&dptr, &retSz, resource) );
	//run cuda kernel
	renderKernel<<<block, grid>>>(dptr, deviceRandStates, nParticle, *deviceMouse);
	CUDA_ERROR_CHECKER;
	CUDA_SAFE_CALL( cudaDeviceSynchronize() );

	//draw
	pShader.use();

    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE);

	glBindVertexArray(VAO);
	glDrawArrays(GL_POINTS, 0, nParticle);
	glBindVertexArray(0);

	glDisable(GL_BLEND);

	//free
	CUDA_SAFE_CALL( cudaFree(deviceMouse) );
}

__GLOBAL__ void renderKernel(Particle* dp, curandState* dr, int n, Mouse const &mouse){
    int index = getIdx();
    if(index > n)
    	return ;

    dp[index].update<UpdateKernel::gravity>(&dr[index], mouse);
}


void Spirit::deployGrid(){
	unsigned int blockX = nParticle>MAX_THREAD? MAX_THREAD: static_cast<unsigned int>(nParticle);
	block = {blockX, 1, 1};

	float nGrid = static_cast<float>(nParticle)/blockX;
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


__DEVICE__ int getIdx(){
	int grid = gridDim.x*gridDim.y*blockIdx.z + gridDim.x*blockIdx.y + blockIdx.x;
	return blockDim.x*grid + threadIdx.x;
}
