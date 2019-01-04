#include <vector>
#include <random>
#include <cstdlib>
#include <cmath>
#include <iostream>


#include "utility.h"
#include "window.h"
#include "gravity.h"
#include "square.h"
#include "tailing.h"
#include "click.h"
#include "fluid.h"

int main(int argc, char** argv){
	long long int n = atoll(argv[1]);

	Window window(800, 800);

	Mouse mouse;

	Fluid ps(n);

	float prevTime = glfwGetTime();
	float currTime;
	int ctr = 0;

	while(!glfwWindowShouldClose(window.window)){
		ctr++;
		if(ctr == 1000){
			currTime = glfwGetTime();
			printf("fps: %f\n", ctr/(currTime-prevTime));
			ctr = 0;
			prevTime = currTime;
		}

		window.processInput();
		getMouse(mouse, window);
		
		//clear output
		glViewport(0, 0, window.width, window.height);
    	glClearColor(0.05f, 0.05f, 0.05f, 1.0f);
    	glClear(GL_COLOR_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);

    	//draw
		ps.render(mouse);

    	//check status
    	glfwSwapBuffers(window.window);
    	glfwPollEvents();
	}
}