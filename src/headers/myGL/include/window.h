#ifndef WINDOW_H
#define WINDOW_H

#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <stdexcept>
#include <iostream>
#include <vector>
#include <string>
#include <cmath>

class Window{
public:
	Window(unsigned int = 800, unsigned int = 600, std::string = "window");

	void processInput() const;

	unsigned int width, height;
	GLFWwindow *window;
};


#endif
