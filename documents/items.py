import scrapy

class DocItem(scrapy.Item):
    file_urls  = scrapy.Field()   # required by FilesPipeline
    files      = scrapy.Field()   # auto-filled after download
    src_page   = scrapy.Field()
    regulation = scrapy.Field()   # set by scrapy-llm
    page_count = scrapy.Field()
    language   = scrapy.Field()