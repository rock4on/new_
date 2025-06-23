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
        
        # Allow error responses for debugging
        "HTTPERROR_ALLOWED_CODES": [403, 404, 500],  # Handle error responses
        
        # Enhanced retry and connection settings for PDFs
        "RETRY_ENABLED": True,
        "RETRY_TIMES": 8,                   # More retries for unstable connections
        "RETRY_HTTP_CODES": [500, 502, 503, 504, 408, 429, 403, 0],  # Include connection errors
        
        # Conservative settings for government sites
        "CONCURRENT_REQUESTS": 1,           # Single request at a time
        "CONCURRENT_REQUESTS_PER_DOMAIN": 1, # One request per domain
        "DOWNLOAD_DELAY": 15,               # 15 second delay between requests
        "RANDOMIZE_DOWNLOAD_DELAY": 0.5,    # Randomize delay ±50%
        
        # Anti-loop and duplicate prevention
        "DUPEFILTER_DEBUG": True,           # Log duplicate requests
        "DEPTH_LIMIT": 10,                  # Limit crawl depth
        "DEPTH_PRIORITY": 1,                # Prioritize shallow pages
        
        # Request limits and timeouts
        "CLOSESPIDER_ITEMCOUNT": 1000,      # Stop after 1000 items
        "CLOSESPIDER_PAGECOUNT": 5000,      # Stop after 5000 pages
        "CLOSESPIDER_TIMEOUT": 3600,        # Stop after 1 hour
        "DOWNLOAD_TIMEOUT": 300,            # 5 minute timeout for large PDFs
        
        # URL filtering
        "URLLENGTH_LIMIT": 2083,            # Skip very long URLs
        
        # Content filtering
        "DOWNLOAD_MAXSIZE": 50 * 1024 * 1024,  # 50MB max file size
        "DOWNLOAD_WARNSIZE": 10 * 1024 * 1024,  # Warn at 10MB
        
        # Auto-throttling for dynamic adjustment
        "AUTOTHROTTLE_ENABLED": True,       # Auto-adjust delays
        "AUTOTHROTTLE_START_DELAY": 10,     # Start with 10s delay
        "AUTOTHROTTLE_MAX_DELAY": 30,       # Max 30s delay
        "AUTOTHROTTLE_TARGET_CONCURRENCY": 1.0,  # Target 1 concurrent request
        
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
        
        # Rotating User Agents for Philippines government sites
        self.user_agents = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:124.0) Gecko/20100101 Firefox/124.0",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.3 Safari/605.1.15"
        ]
        import random
        self.current_ua = random.choice(self.user_agents)

    def solve_cloudflare(self, url):
        """Use FlareSolverr to bypass protection"""
        import random
        import time
        
        # Add random delay to seem more human
        time.sleep(random.uniform(2, 5))
        
        # Rotate user agent
        self.current_ua = random.choice(self.user_agents)
        
        payload = {
            "cmd": "request.get",
            "url": url,
            "maxTimeout": 90000,  # 90 seconds timeout
            "headers": {
                "User-Agent": self.current_ua,
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.9,tl;q=0.8,fil;q=0.7",
                "Accept-Encoding": "gzip, deflate, br",
                "Cache-Control": "no-cache",
                "Pragma": "no-cache",
                "Connection": "keep-alive",
                "Upgrade-Insecure-Requests": "1",
                "Sec-Fetch-Dest": "document",
                "Sec-Fetch-Mode": "navigate",
                "Sec-Fetch-Site": "cross-site",
                "Sec-Fetch-User": "?1",
                "Referer": "https://www.google.com.ph/"
            },
            "session": f"sec_gov_ph_session_{random.randint(1000, 9999)}",
            "returnOnlyCookies": False
        }
        
        try:
            response = requests.post(self.flaresolverr_url, json=payload, timeout=100)
            response.raise_for_status()
            result = response.json()
            
            if result.get("status") == "ok":
                solution = result.get("solution", {})
                status_code = solution.get("status", 0)
                response_text = solution.get("response", "")
                
                # Log the actual response for debugging
                self.logger.info(f"FlareSolverr response status: {status_code}")
                if status_code == 403:
                    self.logger.error(f"403 Forbidden received. Response preview: {response_text[:500]}...")
                    # Still return the response so we can analyze it
                
                return {
                    "url": solution.get("url"),
                    "html": response_text,
                    "cookies": solution.get("cookies", []),
                    "user_agent": solution.get("userAgent"),
                    "status_code": status_code
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
        # Check if this is a 403 response
        if response.status == 403:
            self.logger.error(f"403 Forbidden for {response.url}")
            self.logger.error(f"Response headers: {dict(response.headers)}")
            self.logger.error(f"Response body preview: {response.text[:1000]}...")
            
            # Check if it's a Cloudflare 403 or something else
            if "cloudflare" in response.text.lower():
                self.logger.error("This appears to be a Cloudflare 403 block")
            elif "access denied" in response.text.lower():
                self.logger.error("This appears to be an access denied error")
            elif "forbidden" in response.text.lower():
                self.logger.error("This appears to be a generic forbidden error")
            
            return  # Don't process 403 responses further
        
        # Extract and save page text content
        page_text = self.extract_page_content(response)
        if page_text and len(page_text.strip()) > 500:  # Only save substantial content
            self.save_page_content(response, page_text)
        
        # Download all PDFs found
        pdf_count = 0
        link_count = 0
        
        # Check for PDFs in regular links
        for href in response.css("a::attr(href)").getall():
            link_count += 1
            url = response.urljoin(href.strip())
            if self.allowed_domains[0] in url and self.is_url_allowed(url):
                if PDF_RE.search(url):
                    pdf_count += 1
                    self.logger.info(f"📄 Found PDF #{pdf_count}: {url}")
                    yield from self.create_pdf_item(url, response.url, href)
        
        # Check for PDFs in iframes
        iframe_urls = response.css("iframe::attr(src)").getall()
        for iframe_src in iframe_urls:
            iframe_url = response.urljoin(iframe_src.strip())
            if PDF_RE.search(iframe_url):
                pdf_count += 1
                self.logger.info(f"📄 Found PDF in iframe #{pdf_count}: {iframe_url}")
                yield from self.create_pdf_item(iframe_url, response.url, f"iframe: {iframe_src}")
            elif self.allowed_domains[0] in iframe_url and self.is_url_allowed(iframe_url):
                # Process iframe content to look for PDFs
                self.logger.info(f"🔍 Processing iframe: {iframe_url}")
                solution = self.solve_cloudflare(iframe_url)
                if solution:
                    yield scrapy.Request(
                        solution["url"],
                        meta={
                            "flaresolverr_solution": solution,
                            "dont_cache": True,
                            "is_iframe": True
                        },
                        callback=self.parse_flaresolverr_response,
                        dont_filter=True
                    )
        
        # Check for PDFs in embed tags
        embed_urls = response.css("embed::attr(src)").getall()
        for embed_src in embed_urls:
            embed_url = response.urljoin(embed_src.strip())
            if PDF_RE.search(embed_url):
                pdf_count += 1
                self.logger.info(f"📄 Found PDF in embed #{pdf_count}: {embed_url}")
                yield from self.create_pdf_item(embed_url, response.url, f"embed: {embed_src}")
        
        # Check for PDFs in object tags
        object_urls = response.css("object::attr(data)").getall()
        for object_data in object_urls:
            object_url = response.urljoin(object_data.strip())
            if PDF_RE.search(object_url):
                pdf_count += 1
                self.logger.info(f"📄 Found PDF in object #{pdf_count}: {object_url}")
                yield from self.create_pdf_item(object_url, response.url, f"object: {object_data}")
        
        # Look for JavaScript-loaded PDFs
        script_texts = response.css("script::text").getall()
        for script in script_texts:
            # Look for PDF URLs in JavaScript
            import re
            pdf_matches = re.findall(r'["\']([^"\']*\.pdf[^"\']*)["\']', script, re.IGNORECASE)
            for pdf_match in pdf_matches:
                pdf_url = response.urljoin(pdf_match)
                if self.allowed_domains[0] in pdf_url and self.is_url_allowed(pdf_url):
                    pdf_count += 1
                    self.logger.info(f"📄 Found PDF in JavaScript #{pdf_count}: {pdf_url}")
                    yield from self.create_pdf_item(pdf_url, response.url, f"javascript: {pdf_match}")
        
        # Continue with regular link following for non-PDF links
        for href in response.css("a::attr(href)").getall():
            url = response.urljoin(href.strip())
            if self.allowed_domains[0] in url and self.is_url_allowed(url):
                if not PDF_RE.search(url):
                    # Follow non-PDF links - try FlareSolverr for potential Cloudflare pages
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
    
    def create_pdf_item(self, pdf_url, src_page, title):
        """Create a PDF item and download via FlareSolverr"""
        # Generate filename using same logic as FilesPipeline
        url_hash = hashlib.sha1(pdf_url.encode()).hexdigest()
        scrapy_filename = f"{url_hash}.pdf"
        original_filename = pdf_url.split('/')[-1]
        if not original_filename.endswith('.pdf'):
            original_filename += '.pdf'
        
        metadata = {
            "filename": scrapy_filename,           # Actual file on disk
            "original_filename": original_filename, # Original from URL
            "pdf_url": pdf_url,
            "src_page": src_page,
            "title": title,
            "scraped_at": datetime.now().isoformat(),
            "status": "downloading"
        }
        self.pdf_metadata.append(metadata)
        
        # Save individual metadata file immediately
        self.save_individual_metadata(metadata)
        
        # Download PDF using FlareSolverr
        self.download_pdf_with_flaresolverr(pdf_url, metadata)
        
        # Also yield DocItem for Scrapy pipeline as backup
        yield DocItem(
            file_urls=[pdf_url],
            src_page=src_page,
            title=title,
            pdf_url=pdf_url
        )
    
    def download_pdf_with_flaresolverr(self, pdf_url, metadata):
        """Download PDF using FlareSolverr"""
        self.logger.info(f"📥 Downloading PDF via FlareSolverr: {pdf_url}")
        
        try:
            # Use FlareSolverr to solve the PDF URL
            solution = self.solve_cloudflare(pdf_url)
            
            if solution and solution.get("status_code") == 200:
                # Get cookies and session from FlareSolverr
                cookies = solution.get("cookies", [])
                cookie_dict = {cookie["name"]: cookie["value"] for cookie in cookies}
                user_agent = solution.get("userAgent", self.current_ua)
                
                # Make direct request with solved session
                import requests
                pdf_response = requests.get(
                    pdf_url,
                    cookies=cookie_dict,
                    headers={
                        "User-Agent": user_agent,
                        "Accept": "application/pdf,application/octet-stream,*/*",
                        "Referer": metadata.get("src_page", ""),
                        "Connection": "keep-alive"
                    },
                    timeout=120,
                    stream=True
                )
                
                if pdf_response.status_code == 200:
                    # Check if it's actually a PDF
                    content_type = pdf_response.headers.get('content-type', '').lower()
                    if 'pdf' in content_type or pdf_response.content[:4] == b'%PDF':
                        
                        # Save the PDF
                        downloads_dir = Path(self.settings["FILES_STORE"])
                        downloads_dir.mkdir(exist_ok=True)
                        
                        pdf_filename = metadata["filename"]
                        pdf_path = downloads_dir / pdf_filename
                        
                        with open(pdf_path, 'wb') as f:
                            for chunk in pdf_response.iter_content(chunk_size=8192):
                                if chunk:
                                    f.write(chunk)
                        
                        file_size = pdf_path.stat().st_size
                        self.logger.info(f"✅ PDF downloaded successfully: {pdf_filename} ({file_size} bytes)")
                        
                        # Update metadata
                        metadata["status"] = "downloaded"
                        metadata["download_size"] = file_size
                        metadata["downloaded_at"] = datetime.now().isoformat()
                        metadata["content_type"] = content_type
                        
                        # Save updated metadata
                        self.save_individual_metadata(metadata)
                        return True
                    else:
                        self.logger.warning(f"❌ Response is not a PDF: {content_type}")
                        metadata["status"] = f"not_pdf_{content_type}"
                else:
                    self.logger.error(f"❌ PDF download failed: HTTP {pdf_response.status_code}")
                    metadata["status"] = f"http_error_{pdf_response.status_code}"
            else:
                self.logger.error(f"❌ FlareSolverr failed for PDF: {pdf_url}")
                metadata["status"] = "flaresolverr_failed"
                
        except Exception as e:
            self.logger.error(f"❌ PDF download error: {e}")
            metadata["status"] = f"error_{type(e).__name__}"
        
        # Save metadata with error status
        self.save_individual_metadata(metadata)
        return False
    
    
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