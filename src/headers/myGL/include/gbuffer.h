#ifndef GBUFFER_H
#define GBUFFER_H

#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include "shader.h"
#include "texture.h"
#include "mesh.h"

#include <stdexcept>
#include <vector>
#include <string>
#include <utility>
#include <numeric>

class Gbuffer{
public:
	template<typename ...T>
	Gbuffer(int width, int height, T ...t): width(width), height(height){
		glGenFramebuffers(1, &buffer);
		initTextures({t...});
	}
	void bind() const;
	void unbind() const;
	void setTextures(Mesh &mesh) const;
	void blitData(unsigned int target, unsigned int dataBit) const;
	void blitData(Gbuffer const &gbuffer, unsigned int dataBit) const;

	unsigned int getTexture(int index) const{ return textures[index].ID; }
	unsigned int getFramebuffer() const{ return buffer; }
private:
	void initTextures(std::vector<std::pair<std::string, unsigned int>> const &pairs);

	unsigned int buffer, rbo;
	int const width, height;
	std::vector<Texture> textures;
};

#endif