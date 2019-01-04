#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include "scene.h"

int main(){
    //init scene
    int screenWidth = 800, screenHeight = 600;
    Window window(screenWidth, screenHeight, "learningOpenGL");

    //create a camera
    Camera camera(scene.window);

    //load objects
    Scene scene;
    scene.loadMesh("resources/floor");
    scene.loadModel("resources/theresa");

    //tmp: create shader
    Shader shader("resources/shader.vs", "resources/shader.fs");
    shader.use();

    //rendering loop
    while(!glfwWindowShouldClose(scene.window)){
    	//input
    	scene.processInput();
    	camera.update();

    	//draw objects
    	glViewport(0, 0, screenWidth, screenHeight);
    	glClearColor(0.05f, 0.05f, 0.05f, 1.0f);
    	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    	glEnable(GL_DEPTH_TEST);
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

		shader.use();
		shader.setUniform("model", glm::mat4(1.0f));
		shader.setUniform("view", camera.getView());
		shader.setUniform("projection", camera.getProjection());
		shader.setUniform("viewPos", camera.getPos());


        loader.models[0].draw(shader);
		loader.meshs[0].draw(shader);

    	//check status
    	glfwSwapBuffers(scene.window);
    	glfwPollEvents();
    }

    glfwTerminate();
}