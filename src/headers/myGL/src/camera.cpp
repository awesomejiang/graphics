#include "camera.h"

#include <iostream>
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/string_cast.hpp>

void helperWindow::mouseCallback(double xpos, double ypos){
	if(firstMouse){
		lastX = xpos;
		lastY = ypos;
		firstMouse = false;
	}
	float sensitivity = 0.05f;
	float xoff = (xpos - lastX) * sensitivity;
	float yoff = (lastY - ypos) * sensitivity;
	lastX = xpos;
	lastY = ypos;
	yaw += xoff;
	pitch += yoff;
	//limit pith range
	pitch = std::min(89.0f, std::max(pitch, -89.0f));	
}
		
void helperWindow::scrollCallback(double yoff){
	fov -= yoff;
	fov = std::max(std::min(fov, 45.0f), 1.0f);
}

Camera::Camera(GLFWwindow *window, glm::vec3 const &pos, glm::vec3 const &front, glm::vec3 const &up)
: pos(pos), front(front), worldUp(up), right(glm::normalize(glm::cross(front, up))),
  view(glm::lookAt(pos, pos+front, up)),
  projection(glm::perspective(glm::radians(fov), 800.0f/600.0f, 0.1f, 100.0f)),
  window(window){
	helper = new helperWindow();
	//mouse
	glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
	glfwSetWindowUserPointer(window, helper);
	glfwSetCursorPosCallback(window, [](GLFWwindow *window, double xpos, double ypos){
		static_cast<helperWindow*>(glfwGetWindowUserPointer(window))->mouseCallback(xpos, ypos);
	});
	//scroll
	glfwSetScrollCallback(window, [](GLFWwindow *window, double xoff, double yoff){
		static_cast<helperWindow*>(glfwGetWindowUserPointer(window))->scrollCallback(yoff);
	});
}


void Camera::update(){
	//calculate delta time
	currentTime = glfwGetTime();
	deltaTime = currentTime - lastTime;
	lastTime = currentTime;

	//check keys pressed
	auto speed = 10.0f * deltaTime;
	if(glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
		pos += front * speed;
	if(glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
		pos -= front * speed;
	if(glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
		pos -= right * speed;
	if(glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
		pos += right * speed;

	//update vecs
	pitch = helper->pitch;
	yaw = helper->yaw;
	fov = helper->fov;
	front.x = std::cos(glm::radians(pitch)) * std::cos(glm::radians(yaw));
	front.y = std::sin(glm::radians(pitch));
	front.z = std::cos(glm::radians(pitch)) * std::sin(glm::radians(yaw));
	right = glm::normalize(glm::cross(front, worldUp));
	up = glm::normalize(glm::cross(right, front));

	view = glm::lookAt(pos, pos+front, up);
	projection = glm::perspective(glm::radians(fov), 800.0f/600.0f, 0.1f, 100.0f);
}


glm::mat4 Camera::getView() const{
	return view;
}

glm::mat4 Camera::getProjection() const{
	return projection;
}

glm::mat4 Camera::getVP() const{
	return projection * view;
}
glm::mat4 Camera::getVPInverse() const{
	return glm::inverse(projection * view);
}

glm::vec3 Camera::getFront() const{
	return front;
}

glm::vec3 Camera::getPos() const{
	return pos;
}