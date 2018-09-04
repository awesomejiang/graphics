#ifndef SCENE_H
#define SCENE_H

#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <stdexcept>
#include <iostream>
#include <vector>
#include <string>
#include <cmath>

class Scene{
public:
	Scene(unsigned int = 800, unsigned int = 600, std::string = "scene");

	void processInput() const;

	GLFWwindow *window;
};


#endif
