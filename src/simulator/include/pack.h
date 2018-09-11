#ifndef PACK_H
#define PACK_H

#include <string>
#include <vector>
#include <algorithm>
#include <random>

#include "card.h"
#include "utility.cuh"
#include "framebuffer.h"
#include "shader.h"

class Pack{
public:
	Pack(std::string const &set = "Classic");
	void render(Mouse const &mouse);

private:
	PickCard cardEngine;
	std::vector<Card> cards;

};

#endif