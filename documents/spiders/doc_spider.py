import re, scrapy
from urllib.parse import urlparse
from documents.items import DocItem

PDF_RE = re.compile(r"\.pdf$", re.I)

class DocSpider(scrapy.Spider):
    name = "docs"
    custom_settings = {"FILES_STORE": "downloads"}

    def __init__(self, start_url: str, *args, **kw):
        super().__init__(*args, **kw)
        self.start_urls = [start_url]
        parsed_url = urlparse(start_url)
        self.allowed_domains = [parsed_url.netloc]

    def parse(self, response):
        # Download all PDFs found
        for href in response.css("a::attr(href)").getall():
            url = response.urljoin(href.strip())
            if self.allowed_domains[0] in url:
                if PDF_RE.search(url):
                    yield DocItem(
                        file_urls=[url],
                        src_page=response.url,
                        title=href,  # Store original link text
                        url=url      # Store the actual URL
                    )
                else:
                    yield response.follow(url, self.parse)