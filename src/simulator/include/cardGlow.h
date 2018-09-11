#ifndef CARD_H
#define CARD_H

#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <string>

#include "mesh.h"
#include "texture.h"
#include "shader.h"
#include "framebuffer.h"
#include "utility.cuh"
#include "vec.cuh"
#include "pickcard.h"

class Card{
public:
	Card(CardAttrib const &cardAttrib, vec2 const &offset = {0.0f, 0.0f});
	void render(Mouse const &mouse);

private:
	bool isSelected(Mouse const &mouse) const;
	void drawCard(Mouse const &mouse);
	void drawGlow();

	std::string rarity;

	Framebuffer colorFB;
	Mesh card, glowQuad;
	Shader cardShader, glowShader, finalShader;

	//transform args
	glm::vec3 offset;
	float rotate, enlarge;
	bool startRotate, startEnlarge;
	float prevTime, currTime;
};

#endif