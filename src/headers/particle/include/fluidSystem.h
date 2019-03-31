#ifndef FLUIDSYSTEM_H
#define FLUIDSYSTEM_H

#include <curand.h>
#include <curand_kernel.h>

#include "camera.h"
#include "shader.h"
#include "mesh.h"
#include "framebuffer.h"

#include "vec_float.h"
#include "particle.h"
#include "gridcells.h"

#include <vector>


class FluidSystem{
public:
	FluidSystem(float h);

	void addCube(Cube range, float k = 0.01f, float gamma = 7.0f, float miu = 1.0f,
		float dt = 0.01, vec4 color = {1.0f, 1.0f, 1.0f, 1.0f});
	void render(Camera const &camera);
private:
	//graphics
	float h;
	Shader sShader={"shaders/surface.vs", "shaders/surface.fs"};
	Shader tShader={"shaders/thickness.vs", "shaders/thickness.fs"};
	Shader postShader={"shaders/post.vs", "shaders/post.fs"};
	Framebuffer sFB = {800, 600, {GL_RGB32F}}, tFB = {800, 600, {GL_R32F}};
	Mesh quad;
	
	void draw(Camera const &camera);
	
	//particles
	std::vector<Particle> particles;
	void update();

	//uniform grid
	GridCells gc;

};

#endif