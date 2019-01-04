#include "utility.h"

void __cudaSafeCall(cudaError error, const char *file, const int line){
	if(cudaSuccess != error)
		printf("Error in %s, line %i: %s\n", file, line, cudaGetErrorString(error));
}

void __cudaErrorChecker(const char *file, const int line){
	cudaError error = cudaGetLastError();
	if(cudaSuccess != error){
		printf("Error in %s, line %i: %s\n", file, line, cudaGetErrorString(error));
		exit(1);
	}
}

void getMouse(Mouse &mouse, Window const &window){
	double mouseX, mouseY;
	glfwGetCursorPos(window.window, &mouseX, &mouseY);
	mouseX = mouseX/window.width * 2 - 1.0;
	mouseY = -mouseY/window.height * 2 + 1.0; //mouseY is bottom down	
	vec2 newPos = {static_cast<float>(mouseX), static_cast<float>(mouseY)};
	mouse.dir = norm(newPos - mouse.pos);
	mouse.pos = newPos;

	mouse.pressed = glfwGetMouseButton(window.window, GLFW_MOUSE_BUTTON_LEFT);

	if(!mouse.firstClicked && mouse.pressed)
		mouse.firstClicked = true;
}

