#ifndef TEXTURE_H
#define TEXTURE_H

#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <string>
#include <stdexcept>
#include <vector>

class Texture{
public:
	Texture(unsigned int const &ID, std::string const &type, std::string const &name = "");
	Texture(std::string const &path, std::string const &type, std::string const &name = "");
	Texture(std::vector<std::string> const &paths, std::string const &type, std::string const &name = "");
	
	//copy/assign constructors
	Texture() = default;
	Texture(Texture const &text) = default;
	Texture &operator=(Texture const &text) = default;
	Texture(Texture &&text) = default;
	Texture &operator=(Texture &&text) = default;

	unsigned int ID;
	std::string type, name;
};

#endif