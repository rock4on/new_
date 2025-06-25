import re, scrapy
from urllib.parse import urlparse, urljoin
from documents.items import DocItem
import json
import requests
import time
import random
from datetime import datetime
from pathlib import Path
import hashlib

PDF_RE = re.compile(r"\.pdf$", re.I)

class RegulationSpider(scrapy.Spider):
    name = "regulation"
    custom_settings = {
        "FILES_STORE": "regulation_downloads",
        
        # Allow error responses for debugging and FlareSolverr fallback
        "HTTPERROR_ALLOWED_CODES": [403, 404, 500, 502, 503],
        
        # Enhanced retry settings
        "RETRY_ENABLED": True,
        "RETRY_TIMES": 3,  # Reduced retries since we have FlareSolverr fallback
        "RETRY_HTTP_CODES": [500, 502, 503, 504, 408, 429, 0],
        
        # Conservative settings for government sites
        "CONCURRENT_REQUESTS": 1,
        "CONCURRENT_REQUESTS_PER_DOMAIN": 1,
        "DOWNLOAD_DELAY": 10,
        "RANDOMIZE_DOWNLOAD_DELAY": 0.5,
        
        # Crawl limits
        "DEPTH_LIMIT": 3,  # User specified 2-3 depth
        "DEPTH_PRIORITY": 1,
        
        # Timeouts
        "DOWNLOAD_TIMEOUT": 180,  # 3 minutes
        "DOWNLOAD_MAXSIZE": 100 * 1024 * 1024,  # 100MB max
        
        # Auto-throttling
        "AUTOTHROTTLE_ENABLED": True,
        "AUTOTHROTTLE_START_DELAY": 5,
        "AUTOTHROTTLE_MAX_DELAY": 20,
        "AUTOTHROTTLE_TARGET_CONCURRENCY": 1.0,
        
        # Memory management
        "MEMUSAGE_ENABLED": True,
        "MEMUSAGE_LIMIT_MB": 1000,
        "MEMUSAGE_WARNING_MB": 500,
        
        # Pipeline for saving content
        "ITEM_PIPELINES": {
            "scrapy.pipelines.files.FilesPipeline": 400,
            "documents.pipelines.MetaPipeline": 500,
        }
    }

    def __init__(self, regulation_name=None, start_urls=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.regulation_name = regulation_name or "unknown_regulation"
        self.start_urls = start_urls or []
        
        # Create regulation-specific folder
        self.regulation_folder = Path("regulation_downloads") / self.safe_folder_name(self.regulation_name)
        self.regulation_folder.mkdir(parents=True, exist_ok=True)
        
        # Override FILES_STORE to regulation-specific folder
        self.custom_settings["FILES_STORE"] = str(self.regulation_folder)
        
        # Set allowed domains from start URLs
        self.allowed_domains = []
        for url in self.start_urls:
            parsed = urlparse(url)
            if parsed.netloc and parsed.netloc not in self.allowed_domains:
                self.allowed_domains.append(parsed.netloc)
        
        # FlareSolverr setup
        self.flaresolverr_url = "http://localhost:8191/v1"
        self.user_agents = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:124.0) Gecko/20100101 Firefox/124.0",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.3 Safari/605.1.15"
        ]
        
        # HTML and PDF downloaders
        from pathlib import Path
        import sys
        sys.path.append(str(Path(__file__).parent.parent.parent))
        
        try:
            from html_downloader import HTMLDownloader
            from pdf_downloader import PDFDownloader
            self.html_downloader = HTMLDownloader()
            self.pdf_downloader = PDFDownloader()
            self.flaresolverr_available = True
            print("✅ FlareSolverr downloaders initialized")
        except ImportError as e:
            print(f"⚠️  FlareSolverr downloaders not available: {e}")
            self.flaresolverr_available = False
        
        # Track scraped content
        self.scraped_content = []
        
        print(f"🚀 RegulationSpider initialized for: {self.regulation_name}")
        print(f"📁 Output folder: {self.regulation_folder}")
        print(f"🎯 Start URLs: {len(self.start_urls)}")
        print(f"🌐 Allowed domains: {self.allowed_domains}")
    
    def safe_folder_name(self, name):
        """Convert regulation name to safe folder name"""
        # Remove/replace invalid characters
        safe_name = re.sub(r'[<>:"/\\|?*]', '_', name)
        safe_name = re.sub(r'\s+', '_', safe_name)
        safe_name = safe_name.strip('._')
        return safe_name[:100]  # Limit length
    
    def start_requests(self):
        """Generate initial requests with normal Scrapy first"""
        for url in self.start_urls:
            self.logger.info(f"🌐 Starting crawl: {url}")
            yield scrapy.Request(
                url,
                callback=self.parse,
                meta={
                    'regulation_name': self.regulation_name,
                    'original_url': url
                },
                errback=self.handle_error
            )
    
    def handle_error(self, failure):
        """Handle request errors and fallback to FlareSolverr"""
        request = failure.request
        url = request.url
        
        self.logger.error(f"❌ Request failed for {url}: {failure.value}")
        
        # Check if it's a 403 error
        if hasattr(failure.value, 'response') and failure.value.response.status == 403:
            self.logger.info(f"🔄 403 error detected, trying FlareSolverr fallback for: {url}")
            return self.fallback_to_flaresolverr(url, request.meta)
        
        # For other errors, also try FlareSolverr
        if self.flaresolverr_available:
            self.logger.info(f"🔄 Trying FlareSolverr fallback for: {url}")
            return self.fallback_to_flaresolverr(url, request.meta)
        
        self.logger.error(f"❌ No fallback available for: {url}")
    
    def parse(self, response):
        """Main parsing method"""
        # Check for 403 errors
        if response.status == 403:
            self.logger.info(f"🔄 403 response received, trying FlareSolverr for: {response.url}")
            yield from self.fallback_to_flaresolverr(response.url, response.meta)
            return
        
        # Save page content
        yield from self.save_page_content(response)
        
        # Extract and download PDFs
        yield from self.extract_pdfs(response)
        
        # Follow links (within depth limit)
        if response.meta.get('depth', 0) < self.settings.getint('DEPTH_LIMIT', 3):
            yield from self.follow_links(response)
    
    def fallback_to_flaresolverr(self, url, meta=None):
        """Use FlareSolverr downloaders as fallback"""
        if not self.flaresolverr_available:
            return
        
        meta = meta or {}
        regulation_name = meta.get('regulation_name', self.regulation_name)
        
        try:
            # Determine if this looks like a PDF URL
            if PDF_RE.search(url):
                self.logger.info(f"📄 Using FlareSolverr PDF downloader for: {url}")
                success = self.download_pdf_with_flaresolverr(url, regulation_name)
            else:
                self.logger.info(f"🌐 Using FlareSolverr HTML downloader for: {url}")
                success = self.download_html_with_flaresolverr(url, regulation_name)
            
            if success:
                self.logger.info(f"✅ FlareSolverr download successful: {url}")
            else:
                self.logger.error(f"❌ FlareSolverr download failed: {url}")
                
        except Exception as e:
            self.logger.error(f"❌ FlareSolverr fallback error for {url}: {e}")
    
    def download_pdf_with_flaresolverr(self, pdf_url, regulation_name):
        """Download PDF using FlareSolverr"""
        try:
            # Generate safe filename
            url_hash = hashlib.sha1(pdf_url.encode()).hexdigest()[:8]
            filename = f"{regulation_name}_{url_hash}"
            
            # Use the PDF downloader
            success = self.pdf_downloader.download_pdf(pdf_url, filename)
            
            if success:
                # Record the download
                self.scraped_content.append({
                    'url': pdf_url,
                    'type': 'pdf',
                    'regulation': regulation_name,
                    'filename': f"{filename}.pdf",
                    'downloaded_at': datetime.now().isoformat(),
                    'method': 'flaresolverr'
                })
            
            return success
            
        except Exception as e:
            self.logger.error(f"❌ PDF download error: {e}")
            return False
    
    def download_html_with_flaresolverr(self, html_url, regulation_name):
        """Download HTML using FlareSolverr"""
        try:
            # Generate safe filename
            url_hash = hashlib.sha1(html_url.encode()).hexdigest()[:8]
            parsed_url = urlparse(html_url)
            domain = parsed_url.netloc.replace('www.', '')
            filename = f"{regulation_name}_{domain}_{url_hash}"
            
            # Use the HTML downloader
            success = self.html_downloader.download_html(html_url, filename)
            
            if success:
                # Record the download
                self.scraped_content.append({
                    'url': html_url,
                    'type': 'html',
                    'regulation': regulation_name,
                    'filename': f"{filename}.html",
                    'downloaded_at': datetime.now().isoformat(),
                    'method': 'flaresolverr'
                })
                
                # Try to extract links from downloaded HTML for further crawling
                html_file = Path("downloads") / f"{filename}.html"
                if html_file.exists():
                    yield from self.parse_downloaded_html(html_file, html_url)
            
            return success
            
        except Exception as e:
            self.logger.error(f"❌ HTML download error: {e}")
            return False
    
    def parse_downloaded_html(self, html_file, original_url):
        """Parse downloaded HTML file and extract links"""
        try:
            with open(html_file, 'r', encoding='utf-8') as f:
                html_content = f.read()
            
            # Create a mock response for link extraction
            from scrapy.http import HtmlResponse
            response = HtmlResponse(
                url=original_url,
                body=html_content.encode('utf-8'),
                encoding='utf-8'
            )
            
            # Extract PDFs and follow links
            yield from self.extract_pdfs(response)
            yield from self.follow_links(response)
            
        except Exception as e:
            self.logger.error(f"❌ Error parsing downloaded HTML: {e}")
    
    def save_page_content(self, response):
        """Save page HTML and text content"""
        try:
            # Generate filename
            url_hash = hashlib.sha1(response.url.encode()).hexdigest()
            base_filename = f"{self.safe_folder_name(self.regulation_name)}_{url_hash}"
            
            # Save HTML
            html_file = self.regulation_folder / f"{base_filename}.html"
            with open(html_file, 'w', encoding='utf-8') as f:
                f.write(response.text)
            
            # Extract and save text content
            text_content = self.extract_text_content(response)
            text_file = self.regulation_folder / f"{base_filename}.txt"
            with open(text_file, 'w', encoding='utf-8') as f:
                f.write(text_content)
            
            # Save metadata
            metadata = {
                'url': response.url,
                'regulation': self.regulation_name,
                'html_file': str(html_file),
                'text_file': str(text_file),
                'title': response.css('title::text').get() or '',
                'status_code': response.status,
                'scraped_at': datetime.now().isoformat(),
                'method': 'scrapy',
                'content_length': len(response.text),
                'text_length': len(text_content)
            }
            
            metadata_file = self.regulation_folder / f"{base_filename}_metadata.json"
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            
            # Record the content
            self.scraped_content.append({
                'url': response.url,
                'type': 'html',
                'regulation': self.regulation_name,
                'filename': str(html_file),
                'downloaded_at': datetime.now().isoformat(),
                'method': 'scrapy'
            })
            
            self.logger.info(f"💾 Saved page content: {response.url}")
            
        except Exception as e:
            self.logger.error(f"❌ Error saving page content: {e}")
    
    def extract_text_content(self, response):
        """Extract clean text content from HTML"""
        # Remove script and style elements
        text_parts = response.css('body *:not(script):not(style)::text').getall()
        text_content = ' '.join(text_parts)
        # Normalize whitespace
        text_content = ' '.join(text_content.split())
        return text_content
    
    def extract_pdfs(self, response):
        """Extract and download PDF files"""
        pdf_count = 0
        
        # Find PDF links
        for href in response.css("a::attr(href)").getall():
            url = response.urljoin(href.strip())
            if PDF_RE.search(url) and self.is_allowed_domain(url):
                pdf_count += 1
                self.logger.info(f"📄 Found PDF #{pdf_count}: {url}")
                yield DocItem(
                    file_urls=[url],
                    src_page=response.url,
                    title=href,
                    pdf_url=url
                )
        
        # Check iframes for PDFs
        for iframe_src in response.css("iframe::attr(src)").getall():
            iframe_url = response.urljoin(iframe_src.strip())
            if PDF_RE.search(iframe_url) and self.is_allowed_domain(iframe_url):
                pdf_count += 1
                self.logger.info(f"📄 Found PDF in iframe #{pdf_count}: {iframe_url}")
                yield DocItem(
                    file_urls=[iframe_url],
                    src_page=response.url,
                    title=f"iframe: {iframe_src}",
                    pdf_url=iframe_url
                )
        
        # Check embed and object tags
        for selector in ["embed::attr(src)", "object::attr(data)"]:
            for src in response.css(selector).getall():
                url = response.urljoin(src.strip())
                if PDF_RE.search(url) and self.is_allowed_domain(url):
                    pdf_count += 1
                    self.logger.info(f"📄 Found PDF in {selector.split('::')[0]} #{pdf_count}: {url}")
                    yield DocItem(
                        file_urls=[url],
                        src_page=response.url,
                        title=f"{selector.split('::')[0]}: {src}",
                        pdf_url=url
                    )
        
        if pdf_count > 0:
            self.logger.info(f"📊 Found {pdf_count} PDFs on {response.url}")
    
    def follow_links(self, response):
        """Follow links to other pages within domain and depth limit"""
        current_depth = response.meta.get('depth', 0)
        max_depth = self.settings.getint('DEPTH_LIMIT', 3)
        
        if current_depth >= max_depth:
            self.logger.info(f"🛑 Max depth {max_depth} reached for {response.url}")
            return
        
        link_count = 0
        for href in response.css("a::attr(href)").getall():
            url = response.urljoin(href.strip())
            
            if self.is_allowed_domain(url) and not PDF_RE.search(url):
                link_count += 1
                if link_count <= 20:  # Limit links per page
                    yield scrapy.Request(
                        url,
                        callback=self.parse,
                        meta={
                            'regulation_name': self.regulation_name,
                            'depth': current_depth + 1
                        },
                        errback=self.handle_error,
                        dont_filter=True  # Allow revisiting with different depth
                    )
                else:
                    break
        
        self.logger.info(f"🔗 Following {min(link_count, 20)} links from {response.url} (depth: {current_depth + 1})")
    
    def is_allowed_domain(self, url):
        """Check if URL is from allowed domain"""
        try:
            parsed = urlparse(url)
            return any(domain in parsed.netloc for domain in self.allowed_domains)
        except:
            return False
    
    def closed(self, reason):
        """Save summary when spider closes"""
        try:
            # Save scraping summary
            summary = {
                'regulation_name': self.regulation_name,
                'scraping_completed_at': datetime.now().isoformat(),
                'total_content_scraped': len(self.scraped_content),
                'output_folder': str(self.regulation_folder),
                'scraped_content': self.scraped_content,
                'close_reason': reason
            }
            
            summary_file = self.regulation_folder / 'scraping_summary.json'
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"📊 Scraping completed for {self.regulation_name}")
            self.logger.info(f"📁 Output folder: {self.regulation_folder}")
            self.logger.info(f"📄 Total content saved: {len(self.scraped_content)}")
            self.logger.info(f"💾 Summary saved: {summary_file}")
            
        except Exception as e:
            self.logger.error(f"❌ Error saving summary: {e}")