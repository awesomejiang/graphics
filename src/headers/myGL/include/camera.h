#ifndef CAMERA_H
#define CAMERA_H

#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <cmath>
#include <algorithm>

class helperWindow{
public:
	void mouseCallback(double xpos, double ypos);
	void scrollCallback(double yoff);

	float yaw = -90.0f, pitch = 0.0f, fov = 45.0f;
private:
	bool firstMouse = true;
	float lastX = 400.0f, lastY = 300.0f;
};

class Camera{
public:
	Camera(GLFWwindow *window,
		glm::vec3 const &pos = {0.0f, 1.0f, 3.0f},
		glm::vec3 const &front = {0.0f, 0.0f, -1.0f},
		glm::vec3 const &up = {0.0f, 1.0f, 0.0f});

	void update();
	glm::mat4 getView() const;
	glm::mat4 getProjection() const;
	glm::vec3 getFront() const;
	glm::vec3 getPos() const;

private:
	GLFWwindow *window;
	helperWindow *helper;
	
	glm::vec3 pos, front, worldUp, right, up;
	glm::mat4 view, projection;

	float currentTime = 0.0f, deltaTime = 0.0f, lastTime = 0.0f;
	float yaw = 0.0f, pitch = 0.0f;
	float fov= 45.0f;
};

#endif 