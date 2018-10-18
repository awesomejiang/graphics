#include "card.h"

Card::Card(CardAttrib const &cardAttrib, vec2 const &offset)
: card({
        // positions          // normals           // texture coords
	    {{-HALFWIDTH, -HALFHEIGHT,  0.0f},  {0.0f,  0.0f, 1.0f},   {-0.2f, -0.2f}},
	    {{ HALFWIDTH, -HALFHEIGHT,  0.0f},  {0.0f,  0.0f, 1.0f},   {1.2f, -0.2f}},
	    {{ HALFWIDTH,  HALFHEIGHT,  0.0f},  {0.0f,  0.0f, 1.0f},   {1.2f, 1.2f}},
	    {{ HALFWIDTH,  HALFHEIGHT,  0.0f},  {0.0f,  0.0f, 1.0f},   {1.2f, 1.2f}},
	    {{-HALFWIDTH,  HALFHEIGHT,  0.0f},  {0.0f,  0.0f, 1.0f},   {-0.2f, 1.2f}},
	    {{-HALFWIDTH, -HALFHEIGHT,  0.0f},  {0.0f,  0.0f, 1.0f},   {-0.2f, -0.2f}}
	   },
	   {},
	   {{cardAttrib.path, "texture2D", "front"},
		{"resources/cardback.png", "texture2D", "back"}}), //TODO: changable cardback
  rarity(cardAttrib.rarity),
  offset({offset[0], offset[1], 0.0f}) {}


void Card::render(Mouse const &m){
	update(m);

	if(prevTime == 0.0f)
		prevTime = glfwGetTime();
	else
		prevTime = currTime;

	currTime = glfwGetTime();

	drawCard();

/*
    glEnable(GL_STENCIL_TEST);
    glStencilOp(GL_KEEP, GL_KEEP, GL_REPLACE);

    //set stencil
    glStencilFunc(GL_ALWAYS, 1, 0xFF);
    glStencilMask(0xFF);
	drawCard();

    glStencilFunc(GL_NOTEQUAL, 1, 0xFF);
    glStencilMask(0x00);

    //drawBoundary();

    glStencilMask(0xFF);
    glDisable(GL_STENCIL_TEST);
*/

}


void Card::drawCard() const{
	cardShader.use();
	cardShader.setUniform("model", model);

	// card pattern
	glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	glEnable(GL_CULL_FACE);
	glFrontFace(GL_CCW);
	if(rotate < 90.0f){
		glCullFace(GL_BACK);
		cardShader.setUniform("isFront", 0);
	}
	else{
		glCullFace(GL_FRONT);
		cardShader.setUniform("isFront", 1);
	}

	// selected boundary
	if(rotate == 180.f || !selected || rarity == "Common")
		cardShader.setUniform("selectedBoundary", 0.0f, 0.0f, 0.0f, 0.0f);
	else if(rarity == "Rare")
		cardShader.setUniform("selectedBoundary", 0.5f, 0.7f, 1.0f, 1.0f);
	else if(rarity == "Epic")
		cardShader.setUniform("selectedBoundary", 0.4f, 0.0f, 0.8f, 1.0f);
	else if(rarity == "Legendary")
		cardShader.setUniform("selectedBoundary", 1.0f, 0.6f, 0.0f, 1.0f);

	// hightlight boundary
	if(startRotate){
		//seems wired, but yes.
		//Starting showing up once starting rotating. the width determined by 'enlarge'.
		auto w = (180.0f - whiteWidth)/18.0f;
		cardShader.setUniform("rotatingWidth", w);
		cardShader.setUniform("rotatingBoundary", 1.0f, 1.0f, 1.0f, 1.0f);
	}

	// draw!
	card.draw(cardShader);

	//reset
	glDisable(GL_CULL_FACE);
	glDisable(GL_BLEND);
}

bool Card::isSelected(Mouse const &m) const{
	return m.pos[0] > -HALFWIDTH + offset.x
		&& m.pos[0] < HALFWIDTH + offset.x
		&& m.pos[1] > -HALFHEIGHT + offset.y
		&& m.pos[1] < HALFHEIGHT + offset.y;
}


void Card::update(Mouse const &m){
	//transforms
	auto rate = (currTime - prevTime)/OPEN_TIME * 180.0f;

	selected = isSelected(m);

	if(m.pressed && selected)
		startRotate = true;

	if(startRotate){
		rotate = std::min(rotate+rate, 180.0f);
		enlarge = std::min(enlarge+4.0f*rate, 180.0f);
		whiteWidth = std::min(whiteWidth+0.5f*rate, 180.0f);
	}
	else if(selected){
		enlarge = std::min(enlarge+4.0f*rate, 180.0f);
		whiteWidth = std::min(whiteWidth+0.5f*rate, 180.0f);
	}
	else{
		enlarge = std::max(enlarge-4.0f*rate, 0.0f);
		whiteWidth = std::max(whiteWidth-0.5f*rate, 0.0f);
	}

	//transforms: down to top
	//flip car -> enlarge card -> change stir axis to left up corner
	//-> stir card -> translate to final pos
	auto stirGain = 0.3f*static_cast<float>(sin(glm::radians(enlarge)));
	auto offsetGain = 0.1f * enlarge/180.0f;
	auto stirAxisShift = glm::vec3(HALFWIDTH, -HALFHEIGHT, 0.0f);
	model = glm::mat4(1.0f);
	model = glm::translate(model, offset*(1.0f+offsetGain)-stirAxisShift);
	model = glm::rotate(model, stirGain, glm::vec3(1.0f, 1.0f, 0.0f));
	model = glm::translate(model, stirAxisShift);
	model = glm::scale(model, glm::vec3(1.0f+2.0f*offsetGain));
	model = glm::rotate(model, glm::radians(rotate), glm::vec3(0.0f, 1.0f, 0.0f));
}
/*
void Card::drawBoundary(){
	glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

	//random args
	std::random_device rd;
	std::uniform_real_distribution<float> dis(1.1f, 1.2f);
	//scaled card a little
	auto scaleSize = 1.1f;//dis(rd);
	auto boundModel = model;
	boundModel = glm::scale(boundModel, glm::vec3(scaleSize));

	boundaryShader.use();
	boundaryShader.setUniform("model", boundModel);
	if(rarity == "Common")
		boundaryShader.setUniform("color", 0.0f, 0.0f, 0.0f, 0.0f);
	else if(rarity == "Rare")
		boundaryShader.setUniform("color", 0.5f, 0.7f, 1.0f, 1.0f);
	else if(rarity == "Epic")
		boundaryShader.setUniform("color", 0.4f, 0.0f, 0.8f, 1.0f);
	else if(rarity == "Legendary")
		boundaryShader.setUniform("color", 1.0f, 0.6f, 0.0f, 1.0f);

	card.draw(boundaryShader);

	glDisable(GL_BLEND);
}
*/
