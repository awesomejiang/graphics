#include "utility.h"

void __cudaSafeCall(cudaError error, const char *file, const int line){
	if(cudaSuccess != error)
		printf("Error in %s, line %i: %s\n", file, line, cudaGetErrorString(error));
}

void __cudaErrorChecker(const char *file, const int line){
	cudaError error = cudaGetLastError();
	if(cudaSuccess != error)
		printf("Error in %s, line %i: %s\n", file, line, cudaGetErrorString(error));
}

void getMouse(Mouse &mouse, Scene const &scene){
	double mouseX, mouseY;
	glfwGetCursorPos(scene.window, &mouseX, &mouseY);
	mouseX = mouseX/scene.width * 2 - 1.0;
	mouseY = -mouseY/scene.height * 2 + 1.0; //mouseY is bottom down	
	vec2 newPos = {static_cast<float>(mouseX), static_cast<float>(mouseY)};
	mouse.dir = norm(newPos - mouse.pos);
	mouse.pos = newPos;

	mouse.pressed = glfwGetMouseButton(scene.window, GLFW_MOUSE_BUTTON_LEFT);

	if(!mouse.firstClicked && mouse.pressed)
		mouse.firstClicked = true;
}

