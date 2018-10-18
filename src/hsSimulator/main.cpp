#include <vector>
#include <random>
#include <cstdlib>
#include <cmath>
#include <iostream>

#include "spirit.cuh"
#include "utility.cuh"
#include "scene.h"
#include "pack.h"

int main(int argc, char** argv){
	//long long int n = atoll(argv[1]);

	Scene scene(800, 600, "pack simulator");

	Mouse mouse;

	//Spirit spirit(n, InitKernelEnum::square);
	//auto uk = UpdateKernelEnum::shinning;

	Pack pack("Classic");

	float prevTime = glfwGetTime();
	float currTime;
	int ctr = 0;

	while(!glfwWindowShouldClose(scene.window)){
		ctr++;
		if(ctr == 1000){
			currTime = glfwGetTime();
			printf("fps: %f\n", ctr/(currTime-prevTime));
			ctr = 0;
			prevTime = currTime;
		}

		scene.processInput();
		getMouse(mouse, scene);

		//clear output
		glViewport(0, 0, scene.width, scene.height);
    	glClearColor(0.05f, 0.05f, 0.05f, 1.0f);
    	glClear(GL_COLOR_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);

    	//draw
		//spirit.render(uk, mouse);
		pack.render(mouse);

    	//check status
    	glfwSwapBuffers(scene.window);
    	glfwPollEvents();
	}
}