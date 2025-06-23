import re, scrapy
from urllib.parse import urlparse
from documents.items import DocItem
import json
from datetime import datetime
from pathlib import Path
import hashlib
import requests

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
        # FlareSolverr endpoint
        self.flaresolverr_url = "http://localhost:8191/v1"

    def solve_cloudflare(self, url):
        """Use FlareSolverr to bypass Cloudflare protection"""
        payload = {
            "cmd": "request.get",
            "url": url,
            "maxTimeout": 60000,  # 60 seconds timeout
            "headers": {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
            }
        }
        
        try:
            response = requests.post(self.flaresolverr_url, json=payload, timeout=70)
            response.raise_for_status()
            result = response.json()
            
            if result.get("status") == "ok":
                solution = result.get("solution", {})
                return {
                    "url": solution.get("url"),
                    "html": solution.get("response"),
                    "cookies": solution.get("cookies", []),
                    "user_agent": solution.get("userAgent")
                }
            else:
                self.logger.error(f"FlareSolverr failed: {result.get('message', 'Unknown error')}")
                return None
                
        except Exception as e:
            self.logger.error(f"FlareSolverr request failed: {str(e)}")
            return None

    def start_requests(self):
        """Generate initial requests using FlareSolverr for Cloudflare bypass"""
        for url in self.start_urls:
            # Try FlareSolverr first
            solution = self.solve_cloudflare(url)
            if solution:
                # Create a fake response with the solved content
                yield scrapy.Request(
                    solution["url"],
                    meta={
                        "flaresolverr_solution": solution,
                        "dont_cache": True
                    },
                    callback=self.parse_flaresolverr_response,
                    dont_filter=True
                )
            else:
                # Fallback to regular request if FlareSolverr fails
                self.logger.warning(f"FlareSolverr failed for {url}, trying regular request")
                yield scrapy.Request(url, callback=self.parse)

    def is_url_allowed(self, url):
        """Check if URL is within the allowed base URL path"""
        return url.startswith(self.base_url)

    def parse_flaresolverr_response(self, response):
        """Parse response that was solved by FlareSolverr"""
        solution = response.meta.get("flaresolverr_solution")
        if not solution:
            return
            
        self.logger.info("✅ Successfully bypassed Cloudflare with FlareSolverr")
        
        # Create a mock response with the solved HTML
        from scrapy.http import HtmlResponse
        solved_response = HtmlResponse(
            url=solution["url"],
            body=solution["html"].encode('utf-8'),
            encoding='utf-8'
        )
        
        # Process the solved response
        yield from self.parse_content(solved_response)

    def parse(self, response):
        """Regular parse method for non-Cloudflare pages"""
        yield from self.parse_content(response)

    def parse_content(self, response):
        """Common parsing logic for both FlareSolverr and regular responses"""
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
                    # Follow links - try FlareSolverr for potential Cloudflare pages
                    solution = self.solve_cloudflare(url)
                    if solution:
                        yield scrapy.Request(
                            solution["url"],
                            meta={
                                "flaresolverr_solution": solution,
                                "dont_cache": True
                            },
                            callback=self.parse_flaresolverr_response,
                            dont_filter=True
                        )
                    else:
                        # Fallback to regular request
                        yield scrapy.Request(url, callback=self.parse)
        
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