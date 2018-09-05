#include <vector>
#include <random>
#include <cstdlib>
#include <cmath>
#include <iostream>

#include "spirit.cuh"
#include "card.h"
#include "shader.h"
#include "scene.h"
#include "pack.h"

std::vector<Particle> generateRandomParticles(int n){
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_real_distribution<float> disUnit(0.0f, 1.0f);
	std::uniform_real_distribution<float> disScreen(-1.0f, 1.0f);

	std::vector<Particle> ret;
	for(auto i=0; i<n; ++i){
		float x = disScreen(gen);
		auto k = UpdateKernel::gravity;
		ret.emplace_back(
			k, vec2(x, -1.0f), vec2(0.0f, 0.0f), vec4(0.1f, 0.2f, 0.3f, 1.0f)
		);
	}

	return ret;
}



int main(int argc, char** argv){
	long long int n = atoll(argv[1]);

	Scene scene(800, 600, "pack simulator");
	//Spirit spirit(generateRandomParticles(n));

	Card card("resources/cardinfo.json");

	auto image = card.getRandomCard("Classic");
	std::cout << image << std::endl;
	Pack pack(image);

	while(!glfwWindowShouldClose(scene.window)){
		scene.processInput();

		//clear output
		glViewport(0, 0, scene.width, scene.height);
    	glClearColor(0.05f, 0.05f, 0.05f, 1.0f);
    	glClear(GL_COLOR_BUFFER_BIT);

    	//draw
		//spirit.render(scene);
    	pack.render();
    	
    	//check status
    	glfwSwapBuffers(scene.window);
    	glfwPollEvents();
	}
}