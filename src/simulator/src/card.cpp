#include "card.h"

Card::Card(CardAttrib const &cardAttrib, vec2 const &offset)
: quad({
        // positions          // normals           // texture coords
	    {{-HALFWIDTH, -HALFHEIGHT,  0.0f},  {0.0f,  0.0f, 1.0f},   {0.0f, 0.0f}},
	    {{ HALFWIDTH, -HALFHEIGHT,  0.0f},  {0.0f,  0.0f, 1.0f},   {1.0f, 0.0f}},
	    {{ HALFWIDTH,  HALFHEIGHT,  0.0f},  {0.0f,  0.0f, 1.0f},   {1.0f, 1.0f}},
	    {{ HALFWIDTH,  HALFHEIGHT,  0.0f},  {0.0f,  0.0f, 1.0f},   {1.0f, 1.0f}},
	    {{-HALFWIDTH,  HALFHEIGHT,  0.0f},  {0.0f,  0.0f, 1.0f},   {0.0f, 1.0f}},
	    {{-HALFWIDTH, -HALFHEIGHT,  0.0f},  {0.0f,  0.0f, 1.0f},   {0.0f, 0.0f}}
	   },
	   {},
	   {{cardAttrib.path, "texture2D", "front"},
		{"resources/cardback.png", "texture2D", "back"}}), //TODO: changable cardback
	quadShader("shaders/pack.vs", "shaders/pack.fs"),
	rarity(cardAttrib.rarity),
	offset({offset[0], offset[1], 0.0f}),
	rotate(0.0f), enlarge(0.0f),
	startRotate(false), startEnlarge(false) {}


void Card::render(Mouse const &mouse){
	//transforms
	bool selected = isSelected(mouse);

	if(selected)
		startEnlarge = true;

	if(mouse.pressed && selected)
		startRotate = true;

	if(startRotate){
		enlarge = std::min(enlarge+0.04f, 180.0f);
		rotate = std::min(rotate+0.04f, 180.0f);
	}
	else if(startEnlarge){
		if(selected)
			enlarge = std::min(enlarge+0.04f, 180.0f);
		else
			enlarge = std::max(enlarge-0.04f, 0.0f);
	}

	glm::mat4 model = glm::mat4(1.0f);
	model = glm::translate(model, offset);
	model = glm::rotate(model, 0.3f*static_cast<float>(sin(glm::radians(enlarge))), glm::vec3(1.0f, 1.0f, 0.0f));
	model = glm::scale(model, glm::vec3(1.0f+0.2f*enlarge/180.0f));
	model = glm::rotate(model, glm::radians(rotate), glm::vec3(0.0f, 1.0f, 0.0f));

	quadShader.use();
	quadShader.setUniform("model", model);

	glEnable(GL_CULL_FACE);
	glFrontFace(GL_CCW);
	if(rotate < 90.0f){
		glCullFace(GL_BACK);
		quadShader.setUniform("isFront", 0);
	}
	else{
		glCullFace(GL_FRONT);
		quadShader.setUniform("isFront", 1);
	}
	quad.draw(quadShader);

	glDisable(GL_CULL_FACE);
}


bool Card::isSelected(Mouse const &mouse) const{
	return mouse.pos[0] > -HALFWIDTH + offset.x
		&& mouse.pos[0] < HALFWIDTH + offset.x
		&& mouse.pos[1] > -HALFHEIGHT + offset.y
		&& mouse.pos[1] < HALFHEIGHT + offset.y;
}