#ifndef FRAMEBUFFER_H
#define FRAMEBUFFER_H

#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <stdexcept>
#include <vector>
#include <numeric>

class Framebuffer{
public:
	Framebuffer(int width, int height, int colors = 1);

	void bind() const;
	void unbind() const;
	
	std::vector<unsigned int> colorBuffers;
	
private:
	int width, height;
	unsigned int framebuffer, rbo;
};

#endif