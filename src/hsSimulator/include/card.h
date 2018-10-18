#ifndef CARD_H
#define CARD_H

#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <string>
#include <random>

#include "mesh.h"
#include "texture.h"
#include "shader.h"
#include "framebuffer.h"
#include "utility.cuh"
#include "vec.cuh"
#include "pickcard.h"

#define OPEN_TIME 1.0f

class Card{
public:
	Card(CardAttrib const &cardAttrib, vec2 const &offset = {0.0f, 0.0f});
	void render(Mouse const &m);

private:
	bool isSelected(Mouse const &m) const;
	void update(Mouse const &m);
	void drawCard() const;
	//void drawBoundary();

	std::string rarity;

	Mesh card;
	Shader cardShader{"shaders/card.vs", "shaders/card.fs"};
	//Shader boundaryShader("shaders/card.vs", "shaders/boundary.fs");

	//transform args
	glm::mat4 model;
	glm::vec3 offset;
	float rotate = 0.0f, enlarge = 0.0f, whiteWidth = 0.0f;
	bool startRotate = false, selected = false;
	float prevTime = 0.0f, currTime = 0.0f;

};

#endif