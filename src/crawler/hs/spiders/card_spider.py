from scrapy.spiders import CrawlSpider, Rule
from scrapy.linkextractors import LinkExtractor
from hs.items import CardItem
import re

class card_spider(CrawlSpider):
	name = 'cards'
	allowed_domains = ['hearthpwn.com']
	start_urls = ['http://www.hearthpwn.com/cards?page=1']
	
	rules = (
		Rule( LinkExtractor(allow = '/cards/\d*\-\w*'), callback = 'parse_item' ),
		Rule( LinkExtractor(restrict_xpaths = u"//a[@rel='next']"), callback = 'parse_next' )
			)
		
	#codes for benchmark	
	"""
	rules = (
		Rule( LinkExtractor(allow = '/cards/94-molten-giant'), callback = 'parse_item' ),
		Rule( LinkExtractor(restrict_xpaths = u"//a[@rel='next']"), callback = 'parse_next' )
			)
	"""
			
	def parse_item(self, response):
		item = CardItem()
		
		sel = response.xpath('//div[@class="details card-details"]')
		
		temp = sel.xpath('//header[@class="h2 no-sub with-nav"]/h2/text()').extract()
		item['cardName'] = temp if temp else None
		
		for temp in sel.xpath('aside/ul/li'):
			text = temp.xpath('text()').extract()
			if 'Type: ' in text:
				item['cardType'] = temp.xpath('a/text()').extract()
			elif 'Class: ' in text:
				item['cardClass'] = temp.xpath('a/text()').extract()
			elif 'Rarity: ' in text:
				item['cardRarity'] = temp.xpath('a/text()').extract()
			elif 'Set: ' in text:
				item['cardSet'] = temp.xpath('a/text()').extract()

		temp = sel.xpath('aside/ul/li/span/text()').extract()
		item['cardTip'] = temp[:1] if temp else None
		
		temp = sel.xpath('div[@class="card-info u-typography-format"]//span/text()').extract()
		item['cardSkill'] = temp if temp else None
		
		#notice that some urls will be ''(quite annoying T T)
		temp = sel.xpath('section/img/@data-imageurl').extract()

		if  '' in temp:
			pass
		else:
			item['cardImageUrls'] =  temp if temp else None
			temp = sel.xpath('section/img/@data-goldurl').extract()
			item['cardImageUrls'] +=  temp if temp else None

		return item
		
	def parse_next(self, response):
		return self.make_requests_from_url(response.url)