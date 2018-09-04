#include <vector>
#include <random>
#include <cstdlib>
#include <cmath>
#include <iostream>

#include "game.cuh"
#include "card.h"
#include "shader.h"

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
			k, vec2(x, -1.0f), vec2(0.0f, 0.0f), vec4(0.1f, 0.5f, 0.1f, 1.0f)
		);
	}

	return ret;
}



int main(int argc, char** argv){
	Card card("resources/cardinfo.json");
	std::cout << card.getRandomCard("Classic") << std::endl;
	/*
	long long int n = atoll(argv[1]);
	std::vector<Particle> particles = generateRandomParticles(n);
	Game game(800, 600, particles);
	Shader shader("shaders/particle.vs", "shaders/particle.fs");

	game.init(n);

	//int ctr = 0;
	//while(ctr++ < 10){
	while(!game.shouldClose()){
		shader.use();
		game.render();
	}
	*/


}