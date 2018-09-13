#include "pack.h"

Pack::Pack(std::string const &set)
: cardEngine("resources/cardinfo.json"){
	//pick 5 cards randomly
	std::vector<CardAttrib> cardAttribs;
	for(auto i=0; i<4; ++i)
		cardAttribs.push_back(cardEngine.getRandomCard(set));

	//
	bool allCommons = std::all_of(cardAttribs.begin(), cardAttribs.end(),
		[](auto &attr){return attr.rarity == "Common";});
	if(allCommons){
		auto lastCard = cardEngine.getRandomCard(set);
		while(lastCard.rarity == "Common"){
			lastCard = cardEngine.getRandomCard(set);
		}
		cardAttribs.push_back(lastCard);
		std::random_device rd;
    	std::mt19937 g(rd());
		std::shuffle(cardAttribs.begin(), cardAttribs.end(), g);
	}
	else
		cardAttribs.push_back(cardEngine.getRandomCard(set));

	//create Card objects
	float r = 0.5f;
	float theta[5] = {0.17f * M_PI, 0.5 * M_PI, 0.83 * M_PI, 1.33 * M_PI, 1.67 * M_PI};
	for(auto i=0; i<5; ++i)
		cards.emplace_back(cardAttribs[i], vec2{r*cos(theta[i]), r*sin(theta[i])});
}

void Pack::render(Mouse const &mouse){
	for_each(cards.begin(), cards.end(), [&mouse](auto &card){card.render(mouse);});
	//cards[0].render(mouse);
}