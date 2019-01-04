#ifndef SCENE_H
#define SCENE_H

#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>
#include <string>
#include <stdexcept>

#include "camera.h"
#include "window.h"
#include "shader.h"
#include "mesh.h"
#include "model.h"
#include "gbuffer.h"

class Scene{
public:
	Scene(int width, int height);
	void loadModel(std::string const &dir);
	void loadMesh(std::string const &dir);
	void loadSkybox(std::string const &dir);
	void loadLight(std::string const &dir);

private:
	Shader gbufferShader, skyboxShader;
	Gbuffer gbuffer, gbufferBlend;

	std::vector<Model> models;
	std::vector<Mesh> meshs;
	Mesh skybox;

	std::vector<glm::vec3> directionals, points, spots;

	std::vector<Vertex> readVertices(std::string const &file);
	std::vector<unsigned int> readIndices(std::string const &file);
	std::vector<Texture> readTextures(std::string const &file);
};

#endif