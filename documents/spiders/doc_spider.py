import re, scrapy
from urllib.parse import urlparse
from documents.items import DocItem
import json
from datetime import datetime
from pathlib import Path
import hashlib

PDF_RE = re.compile(r"\.pdf$", re.I)

class DocSpider(scrapy.Spider):
    name = "docs"
    custom_settings = {
        "FILES_STORE": "downloads",
        
        # Anti-loop and duplicate prevention
        "DUPEFILTER_DEBUG": True,           # Log duplicate requests
        "DEPTH_LIMIT": 10,                  # Limit crawl depth
        "DEPTH_PRIORITY": 1,                # Prioritize shallow pages
        
        # Request limits and timeouts
        "CLOSESPIDER_ITEMCOUNT": 1000,      # Stop after 1000 items
        "CLOSESPIDER_PAGECOUNT": 5000,      # Stop after 5000 pages
        "CLOSESPIDER_TIMEOUT": 3600,        # Stop after 1 hour
        "DOWNLOAD_TIMEOUT": 30,             # 30s timeout per request
        
        # URL filtering
        "URLLENGTH_LIMIT": 2083,            # Skip very long URLs
        "HTTPERROR_ALLOWED_CODES": [404],   # Handle 404s gracefully
        
        # Content filtering
        "DOWNLOAD_MAXSIZE": 50 * 1024 * 1024,  # 50MB max file size
        "DOWNLOAD_WARNSIZE": 10 * 1024 * 1024,  # Warn at 10MB
        
        # High-performance settings
        "CONCURRENT_REQUESTS": 32,          # Total concurrent requests
        "CONCURRENT_REQUESTS_PER_DOMAIN": 16, # Per domain concurrency
        "DOWNLOAD_DELAY": 0.5,              # Reduced delay between requests
        "RANDOMIZE_DOWNLOAD_DELAY": 0.5,    # Randomize delay ±50%
        "AUTOTHROTTLE_ENABLED": True,       # Auto-adjust delays
        "AUTOTHROTTLE_START_DELAY": 0.5,    # Start with 0.5s delay
        "AUTOTHROTTLE_MAX_DELAY": 5,        # Max 5s delay
        "AUTOTHROTTLE_TARGET_CONCURRENCY": 8.0,  # Target 8 concurrent requests
        
        # Memory management
        "MEMUSAGE_ENABLED": True,           # Monitor memory usage
        "MEMUSAGE_LIMIT_MB": 500,           # Stop if using >500MB
        "MEMUSAGE_WARNING_MB": 300,         # Warn at 300MB
    }

    def __init__(self, start_url: str, *args, **kw):
        super().__init__(*args, **kw)
        self.start_urls = [start_url]
        parsed_url = urlparse(start_url)
        self.allowed_domains = [parsed_url.netloc]
        
        # Initialize metadata tracking
        self.pdf_metadata = []

    def parse(self, response):
        # Download all PDFs found
        pdf_count = 0
        link_count = 0
        
        for href in response.css("a::attr(href)").getall():
            link_count += 1
            url = response.urljoin(href.strip())
            if self.allowed_domains[0] in url:
                if PDF_RE.search(url):
                    pdf_count += 1
                    self.logger.info(f"📄 Found PDF #{pdf_count}: {url}")
                    
                    # Generate filename using same logic as FilesPipeline
                    # This matches how Scrapy's FilesPipeline generates filenames
                    url_hash = hashlib.sha1(url.encode()).hexdigest()
                    scrapy_filename = f"{url_hash}.pdf"
                    original_filename = url.split('/')[-1]
                    if not original_filename.endswith('.pdf'):
                        original_filename += '.pdf'
                    
                    metadata = {
                        "filename": scrapy_filename,           # Actual file on disk
                        "original_filename": original_filename, # Original from URL
                        "pdf_url": url,
                        "src_page": response.url,
                        "title": href,
                        "scraped_at": datetime.now().isoformat(),
                        "status": "found"
                    }
                    self.pdf_metadata.append(metadata)
                    
                    yield DocItem(
                        file_urls=[url],
                        src_page=response.url,
                        title=href,     # Store original link text
                        pdf_url=url     # Store PDF URL for LLM
                    )
                else:
                    yield response.follow(url, self.parse)
        
        self.logger.info(f"📊 Page {response.url}: Found {pdf_count} PDFs out of {link_count} links")
    
    def closed(self, reason):
        """Save PDF metadata when spider closes"""
        if self.pdf_metadata:
            downloads_dir = Path(self.settings["FILES_STORE"])
            downloads_dir.mkdir(exist_ok=True)
            metadata_file = downloads_dir / "pdf_metadata.json"
            
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(self.pdf_metadata, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"💾 Saved metadata for {len(self.pdf_metadata)} PDFs to {metadata_file}")
        else:
            self.logger.warning("⚠️ No PDF metadata to save")