import re, scrapy
from urllib.parse import urlparse
from documents.items import DocItem
import json
from datetime import datetime
from pathlib import Path
import hashlib
from scrapy_playwright.page import PageMethod

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
        
        # Store the base URL path to restrict crawling
        self.base_url = start_url.rstrip('/')
        
        # Initialize metadata tracking
        self.pdf_metadata = []

    def start_requests(self):
        """Generate initial requests with Playwright enabled for Cloudflare bypass"""
        for url in self.start_urls:
            yield scrapy.Request(
                url,
                meta={
                    "playwright": True,
                    "playwright_page_methods": [
                        # Comprehensive stealth measures
                        PageMethod("evaluate", """() => {
                            // Remove webdriver property
                            Object.defineProperty(navigator, 'webdriver', {get: () => undefined});
                            
                            // Add chrome runtime
                            window.chrome = { runtime: {} };
                            
                            // Mock plugins
                            Object.defineProperty(navigator, 'plugins', {
                                get: () => [1, 2, 3, 4, 5]
                            });
                            
                            // Mock languages
                            Object.defineProperty(navigator, 'languages', {
                                get: () => ['en-US', 'en']
                            });
                            
                            // Mock permissions
                            const originalQuery = window.navigator.permissions.query;
                            window.navigator.permissions.query = (parameters) => (
                                parameters.name === 'notifications' ?
                                Promise.resolve({ state: Notification.permission }) :
                                originalQuery(parameters)
                            );
                            
                            // Mock WebGL
                            const getParameter = WebGLRenderingContext.getParameter;
                            WebGLRenderingContext.prototype.getParameter = function(parameter) {
                                if (parameter === 37445) {
                                    return 'Intel Inc.';
                                }
                                if (parameter === 37446) {
                                    return 'Intel Iris OpenGL Engine';
                                }
                                return getParameter(parameter);
                            };
                            
                            // Mock device memory
                            Object.defineProperty(navigator, 'deviceMemory', {
                                get: () => 8
                            });
                            
                            // Mock hardware concurrency
                            Object.defineProperty(navigator, 'hardwareConcurrency', {
                                get: () => 4
                            });
                        }"""),
                        # Wait for page load
                        PageMethod("wait_for_load_state", "domcontentloaded"),
                        PageMethod("wait_for_timeout", 3000),  # Initial wait
                        
                        # Human-like interactions
                        PageMethod("evaluate", "() => { window.scrollTo(0, 100); }"),  # Small scroll
                        PageMethod("wait_for_timeout", 2000),
                        PageMethod("evaluate", "() => { window.scrollTo(0, 0); }"),    # Back to top
                        PageMethod("wait_for_timeout", 1000),
                        
                        # Wait for Cloudflare to complete
                        PageMethod("wait_for_timeout", 25000),  # Extended Cloudflare wait
                        PageMethod("wait_for_selector", "body", timeout=120000),  # 120 second timeout
                        PageMethod("wait_for_timeout", 5000),  # Final stability wait
                    ],
                    "playwright_include_page": True,
                },
                callback=self.parse,
            )

    def is_url_allowed(self, url):
        """Check if URL is within the allowed base URL path"""
        return url.startswith(self.base_url)

    async def parse(self, response):
        # Clean up Playwright page if it exists
        if "playwright_page" in response.meta:
            page = response.meta["playwright_page"]
            await page.close()  # Important to avoid memory leaks
            self.logger.info("✅ Successfully bypassed Cloudflare protection")
        
        # Extract and save page text content
        page_text = self.extract_page_content(response)
        if page_text and len(page_text.strip()) > 500:  # Only save substantial content
            self.save_page_content(response, page_text)
        
        # Download all PDFs found
        pdf_count = 0
        link_count = 0
        
        for href in response.css("a::attr(href)").getall():
            link_count += 1
            url = response.urljoin(href.strip())
            if self.allowed_domains[0] in url and self.is_url_allowed(url):
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
                    
                    # Save individual metadata file immediately
                    self.save_individual_metadata(metadata)
                    
                    yield DocItem(
                        file_urls=[url],
                        src_page=response.url,
                        title=href,     # Store original link text
                        pdf_url=url     # Store PDF URL for LLM
                    )
                else:
                    # Follow links with Playwright for potential Cloudflare-protected pages
                    yield scrapy.Request(
                        url,
                        meta={
                            "playwright": True,
                            "playwright_page_methods": [
                                # Anti-detection measures
                                PageMethod("evaluate", "() => { Object.defineProperty(navigator, 'webdriver', {get: () => undefined}) }"),
                                PageMethod("wait_for_load_state", "domcontentloaded"),
                                PageMethod("wait_for_timeout", 5000),  # Initial wait
                                PageMethod("wait_for_timeout", 15000),  # Cloudflare processing time
                                PageMethod("wait_for_selector", "body", timeout=120000),  # 120 second timeout
                            ],
                            "playwright_include_page": True,
                        },
                        callback=self.parse,
                    )
        
        self.logger.info(f"📊 Page {response.url}: Found {pdf_count} PDFs out of {link_count} links")
    
    def extract_page_content(self, response):
        """Extract meaningful text content from web page"""
        # Remove script and style elements
        text_content = response.css('body *:not(script):not(style)::text').getall()
        
        # Join and clean text
        page_text = ' '.join(text_content)
        page_text = ' '.join(page_text.split())  # Normalize whitespace
        
        return page_text
    
    def save_page_content(self, response, page_text):
        """Save page content as text file with metadata"""
        downloads_dir = Path(self.settings["FILES_STORE"])
        downloads_dir.mkdir(exist_ok=True)
        
        # Generate filename using same logic as PDFs
        url_hash = hashlib.sha1(response.url.encode()).hexdigest()
        text_filename = f"{url_hash}.txt"
        
        # Save text content
        text_dir = downloads_dir / "text"
        text_dir.mkdir(exist_ok=True)
        text_file = text_dir / text_filename
        
        with open(text_file, 'w', encoding='utf-8') as f:
            f.write(page_text)
        
        # Create metadata for text content
        metadata = {
            "filename": text_filename,
            "original_filename": f"{response.url.split('/')[-1] or 'index'}.txt",
            "content_url": response.url,
            "src_page": response.url,
            "title": response.css('title::text').get() or response.url.split('/')[-1],
            "scraped_at": datetime.now().isoformat(),
            "status": "text_extracted",
            "content_type": "text",
            "text_length": len(page_text)
        }
        
        # Save individual metadata for text content
        json_filename = text_filename.replace('.txt', '.json')
        metadata_file = downloads_dir / json_filename
        
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"📝 Saved text content: {text_filename} ({len(page_text)} chars)")
    
    def save_individual_metadata(self, metadata):
        """Save individual metadata file for each PDF"""
        downloads_dir = Path(self.settings["FILES_STORE"])
        downloads_dir.mkdir(exist_ok=True)
        
        # Create metadata filename: scrapy_hash.json (matches PDF filename)
        pdf_filename = metadata["filename"]
        json_filename = pdf_filename.replace('.pdf', '.json')
        metadata_file = downloads_dir / json_filename
        
        # Save individual metadata
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"💾 Saved metadata: {json_filename}")
    
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