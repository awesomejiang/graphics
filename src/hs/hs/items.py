# -*- coding: utf-8 -*-

# Define here the models for your scraped items
#
# See documentation in:
# http://doc.scrapy.org/en/latest/topics/items.html

from scrapy import Field, Item


class CardItem(Item):
	Name = Field()
	Type = Field()
	Class = Field()
	Rarity = Field()
	Set = Field()
	Tip = Field()
	Skill = Field()
	image_urls = Field()
	images = Field()
	
