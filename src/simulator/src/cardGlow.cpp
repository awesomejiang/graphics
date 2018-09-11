#include "card.h"

Card::Card(CardAttrib const &cardAttrib, vec2 const &offset)
: colorFB(800, 600, 2),
  card({
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
  glowQuad({
        // positions          // normals           // texture coords
	    {{-1.0, -1.0,  0.0f},  {0.0f,  0.0f, 1.0f},   {0.0f, 0.0f}},
	    {{ 1.0, -1.0,  0.0f},  {0.0f,  0.0f, 1.0f},   {1.0f, 0.0f}},
	    {{ 1.0,  1.0,  0.0f},  {0.0f,  0.0f, 1.0f},   {1.0f, 1.0f}},
	    {{ 1.0,  1.0,  0.0f},  {0.0f,  0.0f, 1.0f},   {1.0f, 1.0f}},
	    {{-1.0,  1.0,  0.0f},  {0.0f,  0.0f, 1.0f},   {0.0f, 1.0f}},
	    {{-1.0, -1.0,  0.0f},  {0.0f,  0.0f, 1.0f},   {0.0f, 0.0f}}
	   },
	   {},
	   {}),
  cardShader("shaders/card.vs", "shaders/card.fs"),
  glowShader("shaders/glow.vs", "shaders/glow.fs"),
  finalShader("shaders/glow.vs", "shaders/final.fs"),
  rarity(cardAttrib.rarity),
  offset({offset[0], offset[1], 0.0f}),
  rotate(0.0f), enlarge(0.0f),
  startRotate(false), startEnlarge(false),
  prevTime(0.0f), currTime(0.0f) {}


bool Card::isSelected(Mouse const &mouse) const{
	return mouse.pos[0] > -HALFWIDTH + offset.x
		&& mouse.pos[0] < HALFWIDTH + offset.x
		&& mouse.pos[1] > -HALFHEIGHT + offset.y
		&& mouse.pos[1] < HALFHEIGHT + offset.y;
}

void Card::render(Mouse const &mouse){
	if(prevTime == 0.0f)
		prevTime = glfwGetTime();
	currTime = glfwGetTime();

	colorFB.bind();
	drawCard(mouse);
	colorFB.unbind();

	drawGlow();
	drawCard(mouse);

	prevTime = currTime;
}

void Card::drawCard(Mouse const &mouse){
	//transforms
	bool selected = isSelected(mouse);

	if(selected)
		startEnlarge = true;

	if(mouse.pressed && selected)
		startRotate = true;

	auto rate = (currTime - prevTime)/2.0f * 180.0f;
	if(startRotate){
		enlarge = std::min(enlarge+rate, 180.0f);
		rotate = std::min(rotate+rate, 180.0f);
	}
	else if(startEnlarge){
		if(selected)
			enlarge = std::min(enlarge+rate, 180.0f);
		else
			enlarge = std::max(enlarge-rate, 0.0f);
	}

	glm::mat4 model = glm::mat4(1.0f);
	model = glm::translate(model, offset);
	model = glm::rotate(model, 0.3f*static_cast<float>(sin(glm::radians(enlarge))), glm::vec3(1.0f, 1.0f, 0.0f));
	model = glm::scale(model, glm::vec3(1.0f+0.2f*enlarge/180.0f));
	model = glm::rotate(model, glm::radians(rotate), glm::vec3(0.0f, 1.0f, 0.0f));

	cardShader.use();
	cardShader.setUniform("model", model);

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

	//tell opengl to draw into colorFB
	card.draw(cardShader);

	glDisable(GL_CULL_FACE);
}


void Card::drawGlow(){
  	Framebuffer pingpongFB[2] = {{800, 600, 1}, {800, 600, 1}};

	bool horizontal = true, firstIter = true;
	unsigned int amount = 10;
	glowShader.use();
	glowShader.setUniform("image", 0);
	for(auto i=0; i<amount; ++i){
		pingpongFB[horizontal].bind();
		glowShader.setUniform("horizontal", static_cast<int>(horizontal));
		glBindTexture(GL_TEXTURE_2D, 
			firstIter? colorFB.colorBuffers[1]: pingpongFB[!horizontal].colorBuffers[0]);
		glowQuad.draw(glowShader);

		horizontal = !horizontal;
		if(firstIter)
			firstIter = false;
	}
	pingpongFB[0].unbind(); //call arbitray FB unbind function

    Mesh quad = {
       {
        // positions          // normals           // texture coords
	    {{-1.0, -1.0,  0.0f},  {0.0f,  0.0f, 1.0f},   {0.0f, 0.0f}},
	    {{ 1.0, -1.0,  0.0f},  {0.0f,  0.0f, 1.0f},   {1.0f, 0.0f}},
	    {{ 1.0,  1.0,  0.0f},  {0.0f,  0.0f, 1.0f},   {1.0f, 1.0f}},
	    {{ 1.0,  1.0,  0.0f},  {0.0f,  0.0f, 1.0f},   {1.0f, 1.0f}},
	    {{-1.0,  1.0,  0.0f},  {0.0f,  0.0f, 1.0f},   {0.0f, 1.0f}},
	    {{-1.0, -1.0,  0.0f},  {0.0f,  0.0f, 1.0f},   {0.0f, 0.0f}}
	   },
	   {},
	   {
	   	{pingpongFB[!horizontal].colorBuffers[0], "texture2D", "card"},
	   	{pingpongFB[!horizontal].colorBuffers[0], "texture2D", "glow"}
	   }
	};
	finalShader.use();
	quad.draw(finalShader);
}