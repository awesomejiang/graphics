#ifndef CARD_H
#define CARD_H

#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <string>
#include <algorithm>

#include "mesh.h"
#include "texture.h"
#include "shader.h"
#include "utility.cuh"
#include "vec.cuh"
#include "pickcard.h"

#define HALFWIDTH 0.3575f * 0.4f
#define HALFHEIGHT 0.6583f * 0.4f

class Card{
public:
	Card(CardAttrib const &cardAttrib, vec2 const &offset);
	void render(Mouse const &mouse);
private:
	bool isSelected(Mouse const &mouse) const;

	Mesh quad;
	Shader quadShader;
	std::string rarity;
	glm::vec3 offset;
	float rotate, enlarge;
	bool startRotate, startEnlarge;
};

#endif