#ifndef SHADER_H
#define SHADER_H

#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <string>
#include <vector>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

class Shader{
public:
	Shader(std::string const &vshader, std::string const &fshader, std::string const &gshader = "");

	void use() const;

	template<typename ...Ts>
	void setUniform(std::string const &name, Ts const &...ts) const;
	void deleteShader() const;
	void deleteProgram() const;

private:
	unsigned int ID, vertex, geometry, fragment;
	std::string vcode, fcode;
};

#endif
