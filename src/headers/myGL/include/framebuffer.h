#ifndef FRAMEBUFFER_H
#define FRAMEBUFFER_H

#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <stdexcept>
#include <vector>
#include <numeric>

class Framebuffer{
public:
	Framebuffer(int width, int height, std::vector<GLint> texFormats = {GL_RGB});

	void bind() const;
	void unbind() const;
	void clearBuffers() const;
	unsigned int getTex(unsigned int const &idx = 0) const;
	
private:
	int width, height;
	unsigned int framebuffer, rbo;
	std::vector<unsigned int> colorBuffers;
};

#endif