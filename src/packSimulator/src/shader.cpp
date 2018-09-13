#include "shader.h"

#include <fstream>
#include <sstream>
#include <iostream>
#include <stdexcept>
#include <vector>
#include <type_traits>

Shader::Shader(std::string const &vshader, std::string const &fshader, std::string const &gshader){
	size_t sz = gshader.empty()? 2: 3;
	std::vector<std::string> shader = {vshader, fshader, gshader};
	std::vector<std::string> code(sz, "");

	//read files
	for(auto i=0; i<sz; ++i){
		std::ifstream fs(shader[i]);
		if(fs.is_open()){
			std::stringstream ss;
			ss << fs.rdbuf();
			code[i] = ss.str();
		}
		else
			throw std::runtime_error("Cannot open shader file: " + shader[i]);
	}

	//compile shaders
	ID = glCreateProgram();
	int success;
	char infoLog[512];

	std::vector<unsigned int*> shaderPtr = {&vertex, &fragment, &geometry};
	std::vector<GLenum> shaderType = {GL_VERTEX_SHADER, GL_FRAGMENT_SHADER, GL_GEOMETRY_SHADER};
	for(auto i=0; i<sz; ++i){
		char const *ptr = code[i].c_str();
		*shaderPtr[i] = glCreateShader(shaderType[i]);
		auto shaderID = *shaderPtr[i];
		glShaderSource(shaderID, 1, &ptr, nullptr);
		glCompileShader(shaderID);

		glGetShaderiv(shaderID, GL_COMPILE_STATUS, &success);
		if(!success){
			glGetShaderInfoLog(shaderID, 512, nullptr, infoLog);
			throw std::runtime_error(infoLog);
		}
		//link it to program
		glAttachShader(ID, shaderID);
	}
	glLinkProgram(ID);

	glGetProgramiv(ID, GL_LINK_STATUS, &success);
	if(!success){
		glGetProgramInfoLog(ID, 512, nullptr, infoLog);
		throw std::runtime_error(infoLog);
	}

	//TODO: potential memory link if thrown exception?
	deleteShader();
}


void Shader::use() const{
	glUseProgram(ID);
}


void Shader::deleteShader() const{
	glDeleteShader(vertex);
	glDeleteShader(fragment);
}


void Shader::deleteProgram() const{
	glDeleteProgram(ID);
}


//setUniform overrides
template<>
void Shader::setUniform(std::string const &name,
float const &v1) const{
	glUniform1f(glGetUniformLocation(ID, name.c_str()), v1);
}

template<>
void Shader::setUniform(std::string const &name,
float const &v1, float const &v2) const{
	glUniform2f(glGetUniformLocation(ID, name.c_str()), v1, v2);
}

template<>
void Shader::setUniform(std::string const &name,
float const &v1, float const &v2, float const &v3) const{
	glUniform3f(glGetUniformLocation(ID, name.c_str()), v1, v2, v3);
}

template<>
void Shader::setUniform(std::string const &name,
float const &v1, float const &v2, float const &v3, float const &v4) const{
	glUniform4f(glGetUniformLocation(ID, name.c_str()), v1, v2, v3, v4);
}

template<>
void Shader::setUniform(std::string const &name,
int const &v1) const{
	glUniform1i(glGetUniformLocation(ID, name.c_str()), v1);
}

template<>
void Shader::setUniform(std::string const &name,
int const &v1, int const &v2) const{
	glUniform2i(glGetUniformLocation(ID, name.c_str()), v1, v2);
}

template<>
void Shader::setUniform(std::string const &name,
int const &v1, int const &v2, int const &v3) const{
	glUniform3i(glGetUniformLocation(ID, name.c_str()), v1, v2, v3);
}

template<>
void Shader::setUniform(std::string const &name,
int const &v1, int const &v2, int const &v3, int const &v4) const{
	glUniform4i(glGetUniformLocation(ID, name.c_str()), v1, v2, v3, v4);
}

template<>
void Shader::setUniform(std::string const &name,
glm::vec3 const &vec) const{
	glUniform3fv(glGetUniformLocation(ID, name.c_str()), 1, glm::value_ptr(vec));
}

template<>
void Shader::setUniform(std::string const &name,
glm::vec4 const &vec) const{
	glUniform4fv(glGetUniformLocation(ID, name.c_str()), 1, glm::value_ptr(vec));
}

template<>
void Shader::setUniform(std::string const &name,
glm::mat4 const &mat) const{
	glUniformMatrix4fv(glGetUniformLocation(ID, name.c_str()), 1, GL_FALSE, glm::value_ptr(mat));
}
