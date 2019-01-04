#include <vector>
#include <random>
#include <cstdlib>
#include <cmath>
#include <iostream>
#include <unistd.h>

#include "utility.h"
#include "window.h"
#include "fluidSystem.h"
#include "mesh.h"


Mesh createBox(){
  //create lamp by hand
  std::vector<Vertex> vertices = {
    // positions          // normals           // texture coords
    {{-1.0f, -1.0f, -1.0f},  {0.0f,  0.0f, -1.0f},  {0.0f, 0.0f}},
    {{ 1.0f, -1.0f, -1.0f},  {0.0f,  0.0f, -1.0f},  {1.0f, 0.0f}},
    {{ 1.0f,  1.0f, -1.0f},  {0.0f,  0.0f, -1.0f},  {1.0f, 1.0f}},
    {{-1.0f,  1.0f, -1.0f},  {0.0f,  0.0f, -1.0f},  {0.0f, 1.0f}},

    {{-1.0f, -1.0f,  1.0f},  {0.0f,  0.0f, 1.0f},   {0.0f, 0.0f}},
    {{ 1.0f, -1.0f,  1.0f},  {0.0f,  0.0f, 1.0f},   {1.0f, 0.0f}},
    {{ 1.0f,  1.0f,  1.0f},  {0.0f,  0.0f, 1.0f},   {1.0f, 1.0f}},
    {{-1.0f,  1.0f,  1.0f},  {0.0f,  0.0f, 1.0f},   {0.0f, 1.0f}},
  };
  std::vector<unsigned int> indices = {
  	0, 1, 1, 2, 2, 3, 3, 0, 4, 5, 5, 6, 6, 7, 7, 4, 0, 4, 1, 5, 2, 6, 3, 7
  };
  return  Mesh(vertices, indices);
}


int main(int argc, char** argv){
	int n = atoll(argv[1]);

	Window window(800, 600);
	Camera camera(window.window);

	FluidSystem fs(800, 800, 800, 0.1f);
	fs.addParticle({n, 1, 1, 1, 0.1f, 0.001f, {1.0f, 1.0f, 1.0f, 1.0f}});

    auto box = createBox();
    Shader boxShader = {"shaders/box.vs", "shaders/box.fs"};

	float prevTime = glfwGetTime();
	float currTime;
	int ctr = 0;

	while(!glfwWindowShouldClose(window.window)){
		ctr++;
		if(ctr == 100){
			currTime = glfwGetTime();
			printf("fps: %f\n", ctr/(currTime-prevTime));
			ctr = 0;
			prevTime = currTime;
		}

		window.processInput();
		camera.update();
		
		//clear output
		glViewport(0, 0, window.width, window.height);
    	glClearColor(0.05f, 0.05f, 0.05f, 1.0f);
    	glClear(GL_COLOR_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);

    	//draw particle
		fs.render(camera);

		//box
		boxShader.use();
		boxShader.setUniform("view", camera.getView());
		boxShader.setUniform("projection", camera.getProjection());
		box.draw(boxShader, GL_LINES);

    	//check status
    	glfwSwapBuffers(window.window);
    	glfwPollEvents();
//    	if(ctr == 3) break;
//    	printf("%d\n", ctr);
	}
}