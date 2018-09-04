
BOT_NAME = 'hs'

SPIDER_MODULES = ['hs.spiders']
NEWSPIDER_MODULE = 'hs.spiders'


ROBOTSTXT_OBEY = True

# do not use 'hs.pipelines.HsPipelines' cuz we only refactor two functions.
ITEM_PIPELINES = {'scrapy.pipelines.images.ImagesPipeline':1, 'hs.pipelines.JsonWithEncodingCnblogsPipeline':300}

IMAGES_STORE = 'images/'

#clean former data before run
import os, shutil

try:
	shutil.rmtree(IMAGES_STORE)
except:
	pass



