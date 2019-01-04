#ifndef SHADOW_H
#define SHADOW_H

#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include "shader.h"
#include "mesh.h"
#include "model.h"

enum ShadowType{orthographic, perspective};

template<int type>
class Shadow{
public:
	Shadow(glm::vec3 const &dir, float const &farPlane);

	void bind();
	void unbind() const;
	void setDepth(Mesh const &m, glm::mat4 const &model) const;
	void setDepth(Model const &m, glm::mat4 const &model) const;
	glm::mat4 lightSpaceMatrix;

	//only used when debug
	unsigned int getDepthMap() const {return depthMap;};

private:
	Shader shader;
	glm::vec3 const dir;
	float const farPlane;
	unsigned int depthMapFBO, depthMap;
};

template class Shadow<orthographic>;

template class Shadow<perspective>;

#endif