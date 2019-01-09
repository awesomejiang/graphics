#include "fluidSystem.h"


FluidSystem::FluidSystem(float h): h{h}, gc{h} {
  //create lamp by hand
  std::vector<Vertex> vertices = {
    // positions          // normals           // texture coords
    {{-1.0f,  1.0f,  0.0f}, {0.0f,  0.0f, 1.0f}, {0.0f, 1.0f}},
    {{-1.0f, -1.0f,  0.0f}, {0.0f,  0.0f, 1.0f}, {0.0f, 0.0f}},
    {{ 1.0f, -1.0f,  0.0f}, {0.0f,  0.0f, 1.0f}, {1.0f, 0.0f}},
    {{-1.0f,  1.0f,  0.0f}, {0.0f,  0.0f, 1.0f}, {0.0f, 1.0f}},
    {{ 1.0f, -1.0f,  0.0f}, {0.0f,  0.0f, 1.0f}, {1.0f, 0.0f}},
    {{ 1.0f,  1.0f,  0.0f}, {0.0f,  0.0f, 1.0f}, {1.0f, 1.0f}}
  };
  std::vector<unsigned int> indices(6);
  std::iota(indices.begin(), indices.end(), 0);

  quad = Mesh(vertices,
  			  indices,
  			  {Texture(sFB.getTex(), "texture2D", "surface"),
  			   Texture(tFB.getTex(), "texture2D", "thickness")
  			  }
  			 );
}

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

void FluidSystem::update(){
	//update cell
	gc.clear();
	for(auto &p: particles){
		gc.insertParticle(p.getParticle(), p.getParams(), p.getGrid(), p.getBlock());
	}
	auto cells = gc.getCells(); 

	//update particles
	for(auto &p: particles){
		p.update(cells);
	}
}

void FluidSystem::draw(Camera const &camera){
	//surface(position in camera coord)
	sShader.use();
	sShader.setUniform("VP", camera.getVP());
	sShader.setUniform("pSize", h*2000.0f);
	sShader.setUniform("camPos", camera.getPos());

	sFB.bind();
	sFB.clearBuffers();
	glEnable(GL_DEPTH_TEST);
	for(auto &p: particles){
		p.render();
	}
	glDisable(GL_DEPTH_TEST);
	sFB.unbind();

	//thickness
	tShader.use();
	sShader.setUniform("VP", camera.getVP());
	tShader.setUniform("pSize", h*2000.0f);

	tFB.bind();
	tFB.clearBuffers();
    glEnable(GL_BLEND);
    glBlendFunc(GL_ONE, GL_ONE);
	for(auto &p: particles){
		p.render();
	}
	glDisable(GL_BLEND);
	tFB.unbind();

	//construct quad
	glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	postShader.use();
	postShader.setUniform("VP", camera.getVP());
	postShader.setUniform("camPos", camera.getPos());

	quad.draw(postShader);
}

void FluidSystem::render(Camera const &camera){
	update();
	draw(camera);
}