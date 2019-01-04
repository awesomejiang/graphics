#include "window.h"

Window::Window(unsigned int width, unsigned int height, std::string name)
:width(width), height(height){
	//init glfw
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    //create a window object
    window = glfwCreateWindow(width, height, name.c_str(), nullptr, nullptr);
    if(window == nullptr){
    	glfwTerminate();
    	throw std::runtime_error("Fail to create window");
    }

    glfwMakeContextCurrent(window);
    
    //call glad
    if(!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
    	throw std::runtime_error("Fail to initialize GLAD");
    

    //set size callback
    glfwSetFramebufferSizeCallback(window, [](GLFWwindow* window, int w, int h){
    	glViewport(0, 0, w, h);
    });
}


void Window::processInput() const{
    if(glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(window, true);
}