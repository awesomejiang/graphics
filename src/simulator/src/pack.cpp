#include "pack.h"

Pack::Pack(std::string const &texFront, std::string const &texBack)
: quad({
        // positions          // normals           // texture coords
	    {{-0.3575f, -0.6583f,  0.0f},  {0.0f,  0.0f, 1.0f},   {0.0f, 0.0f}},
	    {{ 0.3575f, -0.6583f,  0.0f},  {0.0f,  0.0f, 1.0f},   {1.0f, 0.0f}},
	    {{ 0.3575f,  0.6583f,  0.0f},  {0.0f,  0.0f, 1.0f},   {1.0f, 1.0f}},
	    {{ 0.3575f,  0.6583f,  0.0f},  {0.0f,  0.0f, 1.0f},   {1.0f, 1.0f}},
	    {{-0.3575f,  0.6583f,  0.0f},  {0.0f,  0.0f, 1.0f},   {0.0f, 1.0f}},
	    {{-0.3575f, -0.6583f,  0.0f},  {0.0f,  0.0f, 1.0f},   {0.0f, 0.0f}}
	   },
	   {},
	   {{texFront, "texture2D", "front"},
		{texBack, "texture2D", "back"}}),
	quadShader("shaders/pack.vs", "shaders/pack.fs"),
	rotate(0.0f),
	startRotate(false) {}


void Pack::render(Mouse const &mouse){
	if(mouse.pressed)
		startRotate = true;

	if(startRotate && rotate < 180.0f)
		rotate += 0.01f;

	glm::mat4 model = glm::mat4(1.0f);
	model = glm::translate(model, glm::vec3(0.0f, 0.0f, 0.0f));
	model = glm::rotate(model, glm::radians(rotate), glm::vec3(0.0f, 1.0f, 0.0f));
	model = glm::scale(model, glm::vec3(1.0f));

	quadShader.use();
	quadShader.setUniform("model", model);

	glEnable(GL_CULL_FACE);
	glFrontFace(GL_CCW);
	if(rotate < 90.0f){
		glCullFace(GL_BACK);
		quadShader.setUniform("isFront", 0);
	}
	else{
		glCullFace(GL_FRONT);
		quadShader.setUniform("isFront", 1);
	}
	quad.draw(quadShader);

	glDisable(GL_CULL_FACE);
}