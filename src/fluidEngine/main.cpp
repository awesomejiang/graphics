#include <vector>
#include <random>
#include <cstdlib>
#include <cmath>
#include <iostream>


#include "utility.h"
#include "scene.h"
#include "fluid.h"

int main(int argc, char** argv){
	int w = atoi(argv[1]);
	int h = atoi(argv[2]);

	Scene scene(800, 600);

	Mouse mouse;

	Fluid fluid(w, h);

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
		fluid.render();

    	//check status
    	glfwSwapBuffers(scene.window);
    	glfwPollEvents();
	}

}