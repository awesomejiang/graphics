
BOT_NAME = 'hs'

SPIDER_MODULES = ['hs.spiders']
NEWSPIDER_MODULE = 'hs.spiders'


ROBOTSTXT_OBEY = True

ITEM_PIPELINES = {'hs.pipelines.HsFilesPipeline':1, 'hs.pipelines.JsonWithEncodingPipeline':300}

FILES_STORE = 'images/'

#clean former data before run
import os, shutil

try:
	shutil.rmtree(FILES_STORE)
except:
	pass



