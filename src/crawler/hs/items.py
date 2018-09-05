# -*- coding: utf-8 -*-

# Define here the models for your scraped items
#
# See documentation in:
# http://doc.scrapy.org/en/latest/topics/items.html

from scrapy import Field, Item


class CardItem(Item):
	cardName = Field()
	cardType = Field()
	cardClass = Field()
	cardRarity = Field()
	cardSet = Field()
	cardTip = Field()
	cardSkill = Field()
	cardImageUrls = Field()
	cardImagePaths = Field()
