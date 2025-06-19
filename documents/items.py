import scrapy

class DocItem(scrapy.Item):
    file_urls  = scrapy.Field()   # required by FilesPipeline
    files      = scrapy.Field()   # auto-filled after download
    src_page   = scrapy.Field()
    title      = scrapy.Field()   # link text
    url        = scrapy.Field()   # actual URL
    page_count = scrapy.Field()
    language   = scrapy.Field()