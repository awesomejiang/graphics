#include "utility.cuh"

void __cudaSafeCall(cudaError error, const char *file, const int line){
	if(cudaSuccess != error)
		printf("Error in %s, line %i: %s\n", file, line, cudaGetErrorString(error));
}

void __cudaErrorChecker(const char *file, const int line){
	cudaError error = cudaGetLastError();
	if(cudaSuccess != error)
		printf("Error in %s, line %i: %s\n", file, line, cudaGetErrorString(error));
}


__DEVICE__ int getIdx(){
	int grid = gridDim.x*gridDim.y*blockIdx.z + gridDim.x*blockIdx.y + blockIdx.x;
	return blockDim.x*grid + threadIdx.x;
}


Mouse getMouse(Scene const &scene){
	//if click on card, flip it
	double mouseX, mouseY;
	glfwGetCursorPos(scene.window, &mouseX, &mouseY);
	mouseX = mouseX/scene.width * 2 - 1.0;
	mouseY = -mouseY/scene.height * 2 + 1.0; //mouseY is bottom down
	vec2 mousePos = {static_cast<float>(mouseX), static_cast<float>(mouseY)};

	bool pressed = glfwGetMouseButton(scene.window, GLFW_MOUSE_BUTTON_LEFT);

	return {mousePos, pressed};
}