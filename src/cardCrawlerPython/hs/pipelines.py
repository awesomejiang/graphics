# -*- coding: utf-8 -*-

import scrapy
import json
import codecs
import hashlib
import os
from scrapy.pipelines.files import FilesPipeline
from scrapy.exceptions import DropItem

#images output
class HsFilesPipeline(FilesPipeline):
	def file_path(self, request, response=None, info=None):
		url = request.url
		return hashlib.sha1(url.encode('utf-8')).hexdigest() + os.path.splitext(url)[1]

	def get_media_requests(self, item, info):
		for url in item['cardImageUrls']:
			yield scrapy.Request(url)

	def item_completed(self, results, item, info):
		image_paths = [x['path'] for ok, x in results if ok]
		# do not want to dropitem in this case even if no images
		if not image_paths:
			raise DropItem("Item contains no images")
		item['cardImagePaths'] = image_paths

		return item

#data output
class JsonWithEncodingPipeline(object):

	def __init__(self):
        	self.file = codecs.open('cardinfo.json', 'w', encoding='utf-8')

	def process_item(self, item, spider):
		line = json.dumps(dict(item), ensure_ascii=False) + "\n"
		self.file.write(line)
		return item

	def spider_closed(self, spider):
		self.file.close()
