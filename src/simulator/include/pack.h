#ifndef PACK_H
#define PACK_H

#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <string>

#include "mesh.h"
#include "texture.h"
#include "shader.h"

class Pack{
public:
	Pack(std::string const &texPath);
	void render() const;
private:
	Mesh quad;
	Shader quadShader;
};

#endif