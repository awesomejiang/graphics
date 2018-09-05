#include "pack.h"

Pack::Pack(std::string const & texPath)
: quad({
        // positions          // normals           // texture coords
	    {{-0.36f, -0.5f,  0.0f},  {0.0f,  0.0f, 1.0f},   {0.0f, 0.0f}},
	    {{ 0.36f, -0.5f,  0.0f},  {0.0f,  0.0f, 1.0f},   {1.0f, 0.0f}},
	    {{ 0.36f,  0.5f,  0.0f},  {0.0f,  0.0f, 1.0f},   {1.0f, 1.0f}},
	    {{ 0.36f,  0.5f,  0.0f},  {0.0f,  0.0f, 1.0f},   {1.0f, 1.0f}},
	    {{-0.36f,  0.5f,  0.0f},  {0.0f,  0.0f, 1.0f},   {0.0f, 1.0f}},
	    {{-0.36f, -0.5f,  0.0f},  {0.0f,  0.0f, 1.0f},   {0.0f, 0.0f}}
	   },
	   {},
	   {{texPath, "texture2D", "pattern"}}),
	quadShader("shaders/pack.vs", "shaders/pack.fs") {}


void Pack::render() const{
	glm::mat4 model = glm::mat4(1.0f);
	model = glm::translate(model, glm::vec3(0.2f, 0.2f, 0.0f));
	model = glm::scale(model, glm::vec3(1.0f));

	quadShader.use();
	quadShader.setUniform("model", model);

	quad.draw(quadShader);
}