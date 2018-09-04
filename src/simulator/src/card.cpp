#include "card.h"
#include <iostream>
Card::Card(std::string const &path)
:gen(std::random_device()()), dis(0.0f, 100.0f){
	//open json file
	std::ifstream ifs(path);
	if(!ifs.is_open())
		throw std::runtime_error("Error: Cannot open " + path + " for json parsing.");

	//read json line by line
	while(!ifs.eof()){
		std::string line;
		std::getline(ifs, line);
		rapidjson::Document doc;
		if(doc.Parse(line.c_str()).HasParseError() || !doc.IsObject())
			continue;

		//only save collectiable cards in proper cardsets
		auto set = doc.FindMember("Set");
		auto tip = doc.FindMember("Tip");
		auto images = doc.FindMember("images");
		CardSet setEnum;
		bool collectiable;
		if(set != doc.MemberEnd() && set->value.IsArray()
		   && tip != doc.MemberEnd() && tip->value.IsArray()
		   && images != doc.MemberEnd() && images->value.IsArray()
		   ){
			std::string setName = set->value[0].GetString();
			if(isInPack(setName) 
			   && std::string(tip->value[0].GetString())=="Collectible"
			   ){
				//get rarity
				auto rarityIter = doc.FindMember("Rarity");
				auto rarity = std::string(rarityIter->value[0].GetString());
				commons[setName][rarity].push_back(images->value[0]["path"].GetString());
				goldens[setName][rarity].push_back(images->value[1]["path"].GetString());
			}
		}
		else
			continue;
	}

	for()
}

std::string Card::getRandomCard(std::string const &set){
	bool golden;
	std::string rarity;
	float dice = dis(gen);
	if(dice < 0.111f){
		golden = true; rarity = "Legendary";
	}
	else if(dice < 1.191f){
		golden = false; rarity = "Legendary";
	}
	else if(dice < 1.499f){
		golden = true; rarity = "Epic";
	}
	else if(dice < 5.779f){
		golden = false; rarity = "Epic";
	}
	else if(dice < 7.149f){
		golden = true; rarity = "Rare";
	}
	else if(dice < 28.549f){
		golden = false; rarity = "Rare";
	}
	else if(dice < 30.019f){
		golden = true; rarity = "Common";
	}
	else{
		golden = false; rarity = "Common";
	}

	int rand = dis(gen)/100.0f * commons.size();
	if(golden)
		return goldens[set][rarity][rand];
	else
		return commons[set][rarity][rand];	
}

bool Card::isInPack(std::string const &set) const{
	if(set == "Classic"
	   || set == "Goblins vs Gnomes"
	   || set == "The Grand Tournament"
	   || set == "Whispers of the Old Gods"
	   || set == "Mean Streets of Gadgetzan"
	   || set == "Journey to Un'Goro"
	   || set == "Knights of the Frozen Throne"
	   || set == "Kobolds and Catacombs"
	   || set == "The Witchwood"
	   || set == "The Boomsday Project"
	   )
		return true;
	else
		return false;
}


/*
CardSet Card::getSetEnum(std::string const &set) const{
	if(set == "Classic")
		return CardSet::classic;
	else if(set == "Goblins vs Gnomes")
		return CardSet::gvg;
	else if(set == "The Grand Tournament")
		return CardSet::tournament;
	else if(set == "Whispers of the Old Gods")
		return CardSet::oldGods;
	else if(set == "Mean Streets of Gadgetzan")
		return CardSet::gadgetzan;
	else if(set == "Journey to Un'Goro")
		return CardSet::ungoro;
	else if(set == "Knights of the Frozen Throne")
		return CardSet::frozenThrone;
	else if(set == "Kobolds and Catacombs")
		return CardSet::kobolds;
	else if(set == "The Witchwood")
		return CardSet::witchwood;
	else if(set == "The Boomsday Project")
		return CardSet::boomsday;
	else
		return CardSet::others;
}
*/