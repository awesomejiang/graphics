#include "game.cuh"

#define MAX_THREAD 1024
#define MAX_BLOCK_X 65535ll
#define MAX_BLOCK_Y 65535ll
#define MAX_BLOCK_Z 65535ll

Game::Game(unsigned int const &width, unsigned int const &height, std::vector<Particle> particles)
: scene({width, height, "particle"}),
  width(width), height(height),
  particles(particles),
  resource(0){
	createVBO();
	setCallBacks();
}

Game::~Game(){
	//unmap resource
	CUDA_SAFE_CALL( cudaGraphicsUnmapResources(1, &resource) );
	CUDA_SAFE_CALL( cudaGraphicsUnregisterResource(resource) );
}

void Game::createVBO(){
	glGenVertexArrays(1, &VAO);
	glGenBuffers(1, &VBO);

	glBindVertexArray(VAO);

	//set VBO
	glBindBuffer(GL_ARRAY_BUFFER, VBO);
	glBufferData(GL_ARRAY_BUFFER, particles.size()*sizeof(Particle), particles.data(), GL_STATIC_DRAW);

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

void Game::setCallBacks() const{
	//glfwSetCursorPosCallback(scene.window, [](GLFWWindow *window, float x, float y){});
}


void Game::init(long long int const &num){
	nParticle = num;
	deployGrid(nParticle);

	//cuda allocations
	auto sz = particles.size()*sizeof(Particle);
	Particle* deviceParticles = nullptr;
	CUDA_SAFE_CALL( cudaMalloc((void**)&deviceParticles, sz) );
	CUDA_SAFE_CALL( cudaMemcpy(deviceParticles, particles.data(), sz, cudaMemcpyHostToDevice) );

	//register to cuda
	CUDA_SAFE_CALL( cudaGraphicsGLRegisterBuffer(&resource, VBO, cudaGraphicsRegisterFlagsNone) );

	//map dptr to VBO
	size_t retSz;
	Particle *dptr = nullptr;
	CUDA_SAFE_CALL( cudaGraphicsMapResources(1, &resource) );
	CUDA_SAFE_CALL( cudaGraphicsResourceGetMappedPointer((void**)&dptr, &retSz, resource) );

	//run cuda kernel
	initKernel<<<grid, block>>>(dptr, nParticle, deviceParticles);
	CUDA_ERROR_CHECKER;

	//free
	CUDA_SAFE_CALL( cudaFree(deviceParticles) );
}


void Game::render(){
	//update args
	processInput();

	double mouseX, mouseY;
	glfwGetCursorPos(scene.window, &mouseX, &mouseY);
	mouseX = mouseX/width * 2 - 1.0;
	mouseY = -mouseY/height * 2 + 1.0; //mouseY is bottom down
	vec2 mousePos = {static_cast<float>(mouseX), static_cast<float>(mouseY)};

	int state = glfwGetMouseButton(scene.window, GLFW_MOUSE_BUTTON_LEFT);

	//get mouse position
	vec2* deviceMousePos = nullptr;
	auto sz = sizeof(vec2);
	CUDA_SAFE_CALL( cudaMalloc((void**)&deviceMousePos, sz) );
	CUDA_SAFE_CALL( cudaMemcpy(deviceMousePos, &mousePos, sz, cudaMemcpyHostToDevice) );

	//map dptr to VBO
	size_t retSz;
	Particle *dptr = nullptr;
	CUDA_SAFE_CALL( cudaGraphicsResourceGetMappedPointer((void**)&dptr, &retSz, resource) );
	//run cuda kernel
	renderKernel<<<block, grid>>>(dptr, nParticle, deviceMousePos, state);
	CUDA_ERROR_CHECKER;

	//draw screen
	glViewport(0, 0, width, height);
    glClearColor(0.05f, 0.05f, 0.05f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE);

	glBindVertexArray(VAO);
	glDrawArrays(GL_POINTS, 0, particles.size());
	glBindVertexArray(0);

	glDisable(GL_BLEND);

    //check status
    glfwSwapBuffers(scene.window);
    glfwPollEvents();

	//free
	CUDA_SAFE_CALL( cudaFree(deviceMousePos) );
}

__global__ void initKernel(Particle* dptr, int n, Particle *p){
    int index = getIdx();
    if(index > n)
    	return ;

    dptr[index] = p[index];
}

__global__ void renderKernel(Particle* dptr, int n, vec2 *pos, int state){
    int index = getIdx();
    if(index > n)
    	return ;

    bool pressed = state==GLFW_PRESS? true: false;
    dptr[index].update(*pos, pressed);
}

__device__ int getIdx(){
	int grid = gridDim.x*gridDim.y*blockIdx.z + gridDim.x*blockIdx.y + blockIdx.x;
	return blockDim.x*grid + threadIdx.x;
}


void Game::deployGrid(long long int const &n){
	unsigned int blockX = n>MAX_THREAD? MAX_THREAD: static_cast<unsigned int>(n);
	block = {blockX, 1, 1};

	float nGrid = static_cast<float>(n)/blockX;
	if(nGrid > MAX_BLOCK_X*MAX_BLOCK_Y*MAX_BLOCK_Z)
		std::runtime_error("Number of particles out of gpu limits.");
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
		std::runtime_error("No particles in screen.");
}

void Game::processInput() const{
    if(glfwGetKey(scene.window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(scene.window, true);
}


bool Game::shouldClose() const{
	return glfwWindowShouldClose(scene.window);
}