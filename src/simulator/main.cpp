#include <vector>
#include <random>
#include <cstdlib>
#include <cmath>
#include <iostream>

#include "spirit.cuh"
#include "utility.cuh"
#include "pickcard.h"
#include "shader.h"
#include "scene.h"
#include "card.h"

/*
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
*/

std::vector<Particle> generateRandomParticles(int n){
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_real_distribution<float> uniformDis(0.0f, 1.0f);
	std::normal_distribution<float> normalDis(0.0f, 1.0f);

	std::vector<Particle> ret;
	for(auto i=0; i<n; ++i){
		float rand = uniformDis(gen) - 0.5f;
		vec2 pos = vec2(-0.5f, rand);
		float vr = 0.01f*normalDis(gen), vtheta = uniformDis(gen)*2*M_PI;
		auto k = UpdateKernel::shinning;
		ret.emplace_back(
			k, pos, vec2(vr*cos(vtheta), vr*sin(vtheta)), vec4(0.1f, 0.2f, 0.3f, 1.0f)
		);
	}

	return ret;
}


int main(int argc, char** argv){
	long long int n = atoll(argv[1]);

	Scene scene(800, 600, "pack simulator");
	Spirit spirit(generateRandomParticles(n));

	PickCard cardEngine("resources/cardinfo.json");

	auto randomCard = cardEngine.getRandomCard("Classic");
	Card card(randomCard, {0.2f, 0.2f});

	while(!glfwWindowShouldClose(scene.window)){
		scene.processInput();
		auto mouse = getMouse(scene);

		//clear output
		glViewport(0, 0, scene.width, scene.height);
    	glClearColor(0.05f, 0.05f, 0.05f, 1.0f);
    	glClear(GL_COLOR_BUFFER_BIT);

    	//draw
		spirit.render(mouse);
    	//card.render(mouse);
    	
    	//check status
    	glfwSwapBuffers(scene.window);
    	glfwPollEvents();
	}
}