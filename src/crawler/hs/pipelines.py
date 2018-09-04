# -*- coding: utf-8 -*-


import scrapy
import json
import codecs
from scrapy.pipelines.images import ImagesPipeline
from scrapy.exceptions import DropItem

#images output
class HsImagesPipeline(ImagesPipeline):

	def get_media_requests(self, item, info):
		for image_url in item['image_urls']:
			yield scrapy.Request(image_url)

	def item_completed(self, results, item, info):
		image_paths = [x['path'] for ok, x in results if ok]
		# do not want to dropitem in this case even if no images
		"""if not image_paths:
			raise DropItem("Item contains no images")"""
		item['image_paths'] = image_paths
		return item
#data output
class JsonWithEncodingCnblogsPipeline(object):

	def __init__(self):
        	self.file = codecs.open('cardinfo.json', 'w', encoding='utf-8')

	def process_item(self, item, spider):
		line = json.dumps(dict(item), ensure_ascii=False) + "\n"
		self.file.write(line)
		return item

	def spider_closed(self, spider):
		self.file.close()
