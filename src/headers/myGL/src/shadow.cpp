#include "shadow.h"

//directional shadow
template<>
Shadow<orthographic>::Shadow(glm::vec3 const &dir, float const &farPlane)
: shader("shaders/shadow/ortho.vs", "shaders/shadow/ortho.fs"),
  dir(dir), farPlane(farPlane){
	//create an FBO
	glGenFramebuffers(1, &depthMapFBO);

	//store depthMap in a texture
	glGenTextures(1, &depthMap);
	glBindTexture(GL_TEXTURE_2D, depthMap);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT,
		1024, 1024, 0, GL_DEPTH_COMPONENT, GL_FLOAT, nullptr);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
	float borderColor[] = {1.0f, 1.0f, 1.0f, 1.0f};
	glTexParameterfv(GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR, borderColor);

	//bind texture to framebuffer
	glBindFramebuffer(GL_FRAMEBUFFER, depthMapFBO);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, depthMap, 0);
	glDrawBuffer(GL_NONE);
	glReadBuffer(GL_NONE);
	glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

template<>
void Shadow<orthographic>::bind(){
	glViewport(0, 0, 1024, 1024);
	glBindFramebuffer(GL_FRAMEBUFFER, depthMapFBO);
	glClear(GL_DEPTH_BUFFER_BIT);
	//config mat
	glm::mat4 lightProjection = glm::ortho(-10.0f , 10.0f, -10.0f, 10.0f, 1.0f, farPlane);
	glm::mat4 lightView = glm::lookAt(
		dir,
		glm::vec3(0.0f, 0.0f, 0.0f),
		glm::vec3(0.0f, 1.0f, 0.0f));
	lightSpaceMatrix = lightProjection * lightView;
	shader.use();
	shader.setUniform("farPlane", farPlane);
	shader.setUniform("lightSpaceMatrix", lightSpaceMatrix);
}


//point(omnidirectional) shadow
template<>
Shadow<perspective>::Shadow(glm::vec3 const &dir, float const &farPlane)
: shader("shaders/shadow/perspect.vs", "shaders/shadow/perspect.fs", "shaders/shadow/perspect.gs"),
  dir(dir), farPlane(farPlane){
	//create an FBO
	glGenFramebuffers(1, &depthMapFBO);

	//store depthMap in a texture
	glGenTextures(1, &depthMap);
	glBindTexture(GL_TEXTURE_CUBE_MAP, depthMap);
	for(auto i=0; i<6; ++i)
		glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X+i, 0, GL_DEPTH_COMPONENT,
			1024, 1024, 0, GL_DEPTH_COMPONENT, GL_FLOAT, nullptr);
	glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);

	//bind texture to framebuffer
	glBindFramebuffer(GL_FRAMEBUFFER, depthMapFBO);
	glFramebufferTexture(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, depthMap, 0);
	glDrawBuffer(GL_NONE);
	glReadBuffer(GL_NONE);
	glBindFramebuffer(GL_FRAMEBUFFER, 0);
}


template<>
void Shadow<perspective>::bind(){
	glViewport(0, 0, 1024, 1024);
	glBindFramebuffer(GL_FRAMEBUFFER, depthMapFBO);
	glClear(GL_DEPTH_BUFFER_BIT);
	//config mat
	glm::mat4 shadowProjection = glm::perspective(glm::radians(90.0f), 1024.0f/1024.0f, 1.0f, farPlane);
	std::vector<glm::mat4> shadowTransforms = {
		shadowProjection * glm::lookAt(dir, dir+glm::vec3(1.0f, 0.0f, 0.0f), glm::vec3(0.0f, -1.0f, 0.0f)),
		shadowProjection * glm::lookAt(dir, dir+glm::vec3(-1.0f, 0.0f, 0.0f), glm::vec3(0.0f, -1.0f, 0.0f)),
		shadowProjection * glm::lookAt(dir, dir+glm::vec3(0.0f, 1.0f, 0.0f), glm::vec3(0.0f, 0.0f, 1.0f)),
		shadowProjection * glm::lookAt(dir, dir+glm::vec3(0.0f, -1.0f, 0.0f), glm::vec3(0.0f, 0.0f, -1.0f)),
		shadowProjection * glm::lookAt(dir, dir+glm::vec3(0.0f, 0.0f, 1.0f), glm::vec3(0.0f, -1.0f, 0.0f)),
		shadowProjection * glm::lookAt(dir, dir+glm::vec3(0.0f, 0.0f, -1.0f), glm::vec3(0.0f, -1.0f, 0.0f))
	};
	shader.use();
	shader.setUniform("farPlane", farPlane);
	shader.setUniform("lightPos", dir);
	for(auto i=0; i<6; ++i)
		shader.setUniform("shadowMatrices["+std::to_string(i)+"]", shadowTransforms[i]);
}


//common helper functions
template<int type>
void Shadow<type>::setDepth(Mesh const &m, glm::mat4 const &model) const{
	shader.setUniform("model", model);
	m.draw(shader);
}

template<int type>
void Shadow<type>::setDepth(Model const &m, glm::mat4 const &model) const{
	shader.setUniform("model", model);
	m.draw(shader);
}

template<int type>
void Shadow<type>::unbind() const{
	glBindFramebuffer(GL_FRAMEBUFFER, 0);
}