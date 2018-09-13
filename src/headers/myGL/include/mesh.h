#ifndef MASH_H
#define MASH_H

#include <glad/glad.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include "shader.h"
#include "texture.h"
#include <string>
#include <vector>
#include <numeric>

struct Vertex{
	glm::vec3 position;
	glm::vec3 normal;
	glm::vec2 texCoords;
};
class Mesh{
public:
	Mesh() = default;
	Mesh(std::vector<Vertex> const &vertices,
		 std::vector<unsigned int> const &indices = {},
		 std::vector<Texture> const &textures = {});

	void setTexture(Texture const &texture);

	void draw(Shader const &shader) const;

	unsigned int getTexID(int index) const;
	std::string getTexName(int index) const;
private:
	unsigned int pts;
	std::vector<Texture> textures;

	unsigned int VAO, VBO, EBO;
};

#endif