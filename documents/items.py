import scrapy

class DocItem(scrapy.Item):
    file_urls  = scrapy.Field()   # required by FilesPipeline
    files      = scrapy.Field()   # auto-filled after download
    src_page   = scrapy.Field()
    title      = scrapy.Field()   # link text
    pdf_url    = scrapy.Field()   # PDF direct URL for LLM
    page_count = scrapy.Field()
    language   = scrapy.Field()