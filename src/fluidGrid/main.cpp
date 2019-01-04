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

	Scene scene(w, h);

	Mouse mouse;

	Fluid fluid(w, h);

	float prevTime = glfwGetTime();
	float currTime;
	float sumTime = 0;
	int ctr = 0;

	while(!glfwWindowShouldClose(scene.window)){
		currTime = glfwGetTime();
		float dt = currTime - prevTime;
		prevTime = currTime;

		sumTime += dt;
		ctr++;
		if(ctr == 10){
			printf("fps: %f\n", ctr/(sumTime));
			ctr = 0;
			sumTime = 0.0f;
		}

		scene.processInput();
		getMouse(mouse, scene);
		
		//clear output
		glViewport(0, 0, scene.width, scene.height);
    	glClearColor(0.05f, 0.05f, 0.05f, 1.0f);
    	glClear(GL_COLOR_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);

    	//draw
		fluid.render(mouse, dt);

    	//check status
    	glfwSwapBuffers(scene.window);
    	glfwPollEvents();
	}

}