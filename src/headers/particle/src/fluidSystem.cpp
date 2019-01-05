#include "fluidSystem.h"


void FluidSystem::addCube(Cube range, float k, float gamma, float dt, vec4 color){
	int sizeX = (range.upperX-range.lowerX)/h;
	int sizeY = (range.upperY-range.lowerY)/h;
	int sizeZ = (range.upperZ-range.lowerZ)/h;
	int pNum = sizeX*sizeY*sizeZ;
	particles.emplace_back(ParticleParams{pNum, k, gamma, h, dt, color});

	auto density = std::vector<float>(pNum, 2.141506f);	//cube neighbors: 6 1s, 12 rt(2)s, 8 rt(3)s
	auto pos = std::vector<vec3>(pNum);
	for(int i=0; i<sizeX; ++i){
		for(int j=0; j<sizeY; ++j){
			for(int k=0; k<sizeZ; ++k){
				pos[i*sizeY*sizeZ+j*sizeZ+k] = vec3{range.lowerX+i*h, range.lowerY+j*h, range.lowerZ+k*h};
			}
		}
	}

	particles.back().setDeviceParticle(pos, density);
	printf("Add a cube consisted of %zu particles into system\n", pos.size());
}


void FluidSystem::render(Camera const &camera){
	//
	//update cell
	gc.clear();
	for(auto &p: particles){
		gc.insertParticle(p.getParticle(), p.getParams(), p.getGrid(), p.getBlock());
	}
	auto cells = gc.getCells(); 

	//update particles and draw
	shader.use();
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE);

	shader.setUniform("projection", camera.getProjection());
	shader.setUniform("view", camera.getView());

	for(auto &p: particles){
		p.render(cells);
		auto color = p.getParams().color;
		shader.setUniform("color", color[0], color[1], color[2], color[3]);
	}

	glDisable(GL_BLEND);
}