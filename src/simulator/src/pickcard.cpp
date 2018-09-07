#include "pickcard.h"
#include <iostream>
PickCard::PickCard(std::string const &path)
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
		auto set = doc.FindMember("cardSet");
		auto tip = doc.FindMember("cardTip");
		auto images = doc.FindMember("cardImagePaths");
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
				auto rarityIter = doc.FindMember("cardRarity");
				auto rarity = std::string(rarityIter->value[0].GetString());
				commons[setName][rarity].push_back(images->value[0].GetString());
				goldens[setName][rarity].push_back(images->value[1].GetString());
			}
		}
		else
			continue;
	}
}

CardAttrib PickCard::getRandomCard(std::string const &set){
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

	int rand = dis(gen)/100.0f * commons[set][rarity].size();
	std::string path = golden? goldens[set][rarity][rand]: commons[set][rarity][rand];
	path = "resources/images/" + path;
	return {rarity, path};
}

bool PickCard::isInPack(std::string const &set) const{
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