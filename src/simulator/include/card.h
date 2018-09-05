#ifndef CARD_H
#define CARD_H

#include <rapidjson/document.h>

#include <string>
#include <unordered_map>
#include <fstream>
#include <stdexcept>
#include <random>

#define MAX_SETS 10


class Card{
public:
	Card(std::string const &path);

	std::string getRandomCard(std::string const &set);

private:
	bool isInPack(std::string const &set) const;

	std::mt19937 gen;
	std::uniform_real_distribution<float> dis;
	std::unordered_map<
		std::string, 
		std::unordered_map<std::string, std::vector<std::string>>
		> commons, goldens;
};

#endif