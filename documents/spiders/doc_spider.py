import re, scrapy
from w3lib.url import urljoin
from scrapy.linkextractors import LinkExtractor
from scrapy_llm.config import LLM_RESPONSE_MODEL_KEY      # optional
from documents.items import DocItem
from documents.models import RegulationModel

PDF_RE = re.compile(r"\.pdf$", re.I)

class DocSpider(scrapy.Spider):
    name = "docs"
    custom_settings = {"FILES_STORE": "downloads"}

    def __init__(self, start_url: str, *args, **kw):
        super().__init__(*args, **kw)
        self.start_urls = [start_url]
        self.allowed_domains = [scrapy.utils.url.get_base_url(scrapy.Request(start_url)).split("//")[1].split("/")[0]]

    def parse(self, response):
        # 1. enqueue every internal link
        for href in response.css("a::attr(href)").getall():
            url = urljoin(response.url, href.strip())
            if self.allowed_domains[0] in url:
                if PDF_RE.search(url):
                    yield DocItem(
                        file_urls=[url],
                        src_page=response.url,
                        # tell scrapy-llm which schema to fill
                        **{LLM_RESPONSE_MODEL_KEY: RegulationModel},
                    )
                else:
                    yield response.follow(url, self.parse)