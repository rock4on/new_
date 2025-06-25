import scrapy
import json
import hashlib
import re
from pathlib import Path
from datetime import datetime
from urllib.parse import urlparse, urljoin
from documents.items import DocItem

class RegulationSpider(scrapy.Spider):
    name = "regulation"
    
    custom_settings = {
        'ROBOTSTXT_OBEY': False,
        'DOWNLOAD_DELAY': 3,
        'RANDOMIZE_DOWNLOAD_DELAY': 0.5,
        'CONCURRENT_REQUESTS': 1,
        'CONCURRENT_REQUESTS_PER_DOMAIN': 1,
        'RETRY_TIMES': 2,
        'DOWNLOAD_TIMEOUT': 30,
        'DEPTH_LIMIT': 3,
        'HTTPERROR_ALLOWED_CODES': [403, 404, 500, 502, 503],
        'USER_AGENT': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    def __init__(self, regulation_name=None, start_urls=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Set regulation name
        self.regulation_name = regulation_name or "unknown_regulation"
        self.logger.info(f"Initializing spider for regulation: {self.regulation_name}")
        
        # Parse start URLs
        if start_urls:
            if isinstance(start_urls, str):
                # Split by comma and clean
                raw_urls = [url.strip() for url in start_urls.split(',')]
                self.start_urls = []
                
                for url in raw_urls:
                    if url:
                        # Add https if missing
                        if not url.startswith(('http://', 'https://')):
                            url = f'https://{url}'
                        self.start_urls.append(url)
                        self.logger.info(f"Added start URL: {url}")
            else:
                self.start_urls = list(start_urls)
        else:
            self.start_urls = []
            
        if not self.start_urls:
            self.logger.error("No valid start URLs provided!")
            
        # Set allowed domains
        self.allowed_domains = []
        for url in self.start_urls:
            try:
                domain = urlparse(url).netloc
                if domain and domain not in self.allowed_domains:
                    self.allowed_domains.append(domain)
                    self.logger.info(f"Added allowed domain: {domain}")
            except Exception as e:
                self.logger.error(f"Error parsing domain from {url}: {e}")
        
        # Create output folder
        safe_name = self.make_safe_filename(self.regulation_name)
        self.output_folder = Path("regulation_downloads") / safe_name
        self.output_folder.mkdir(parents=True, exist_ok=True)
        self.logger.info(f"Output folder: {self.output_folder}")
        
        # Track scraped content
        self.scraped_pages = []
        self.scraped_pdfs = []
        
        # PDF pattern
        self.pdf_pattern = re.compile(r'\.pdf$', re.IGNORECASE)
        
    def make_safe_filename(self, name):
        """Convert name to safe filename"""
        safe = re.sub(r'[<>:"/\\|?*]', '_', name)
        safe = re.sub(r'\s+', '_', safe)
        return safe[:100].strip('._')
    
    def start_requests(self):
        """Generate initial requests"""
        for url in self.start_urls:
            self.logger.info(f"Starting request to: {url}")
            yield scrapy.Request(
                url=url,
                callback=self.parse,
                errback=self.handle_error,
                meta={'regulation_name': self.regulation_name}
            )
    
    def parse(self, response):
        """Main parsing method"""
        self.logger.info(f"Parsing: {response.url} (status: {response.status})")
        
        # Handle error responses
        if response.status == 403:
            self.logger.warning(f"403 Forbidden: {response.url}")
            yield from self.handle_403_error(response)
            return
        elif response.status >= 400:
            self.logger.warning(f"HTTP {response.status}: {response.url}")
            return
        
        # Save page content
        yield from self.save_page_content(response)
        
        # Find and download PDFs
        yield from self.extract_pdfs(response)
        
        # Follow links if within depth limit
        current_depth = response.meta.get('depth', 0)
        if current_depth < self.settings.getint('DEPTH_LIMIT', 3):
            yield from self.follow_links(response)
    
    def save_page_content(self, response):
        """Save HTML and text content"""
        try:
            # Generate filename
            url_hash = hashlib.md5(response.url.encode()).hexdigest()[:8]
            base_filename = f"{self.make_safe_filename(self.regulation_name)}_{url_hash}"
            
            # Save HTML
            html_file = self.output_folder / f"{base_filename}.html"
            with open(html_file, 'w', encoding='utf-8', errors='ignore') as f:
                f.write(response.text)
            
            # Extract and save text
            text_content = self.extract_text(response)
            text_file = self.output_folder / f"{base_filename}.txt"
            with open(text_file, 'w', encoding='utf-8', errors='ignore') as f:
                f.write(text_content)
            
            # Save metadata
            metadata = {
                'url': response.url,
                'regulation': self.regulation_name,
                'title': response.css('title::text').get() or '',
                'status_code': response.status,
                'scraped_at': datetime.now().isoformat(),
                'html_file': str(html_file),
                'text_file': str(text_file),
                'content_length': len(response.text),
                'text_length': len(text_content)
            }
            
            metadata_file = self.output_folder / f"{base_filename}_metadata.json"
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            
            self.scraped_pages.append(metadata)
            self.logger.info(f"Saved page: {html_file.name}")
            
        except Exception as e:
            self.logger.error(f"Error saving page content: {e}")
    
    def extract_text(self, response):
        """Extract clean text from HTML"""
        try:
            # Get text from body, excluding script and style tags
            text_parts = response.css('body *:not(script):not(style)::text').getall()
            text = ' '.join(text_parts)
            # Clean up whitespace
            text = ' '.join(text.split())
            return text
        except Exception as e:
            self.logger.error(f"Error extracting text: {e}")
            return ""
    
    def extract_pdfs(self, response):
        """Find and yield PDF downloads"""
        pdf_count = 0
        
        # Find PDF links
        for link in response.css('a::attr(href)').getall():
            if not link:
                continue
                
            # Make absolute URL
            pdf_url = urljoin(response.url, link.strip())
            
            # Check if it's a PDF and from allowed domain
            if self.pdf_pattern.search(pdf_url) and self.is_allowed_url(pdf_url):
                pdf_count += 1
                self.logger.info(f"Found PDF #{pdf_count}: {pdf_url}")
                
                # Create PDF metadata
                pdf_metadata = {
                    'url': pdf_url,
                    'source_page': response.url,
                    'regulation': self.regulation_name,
                    'found_at': datetime.now().isoformat()
                }
                self.scraped_pdfs.append(pdf_metadata)
                
                # Yield DocItem for download
                yield DocItem(
                    file_urls=[pdf_url],
                    src_page=response.url,
                    title=link,
                    pdf_url=pdf_url
                )
        
        # Check for PDFs in iframes, embeds, objects
        for selector in ['iframe::attr(src)', 'embed::attr(src)', 'object::attr(data)']:
            for src in response.css(selector).getall():
                if not src:
                    continue
                    
                pdf_url = urljoin(response.url, src.strip())
                if self.pdf_pattern.search(pdf_url) and self.is_allowed_url(pdf_url):
                    pdf_count += 1
                    self.logger.info(f"Found PDF in {selector.split('::')[0]} #{pdf_count}: {pdf_url}")
                    
                    pdf_metadata = {
                        'url': pdf_url,
                        'source_page': response.url,
                        'regulation': self.regulation_name,
                        'found_in': selector.split('::')[0],
                        'found_at': datetime.now().isoformat()
                    }
                    self.scraped_pdfs.append(pdf_metadata)
                    
                    yield DocItem(
                        file_urls=[pdf_url],
                        src_page=response.url,
                        title=f"{selector.split('::')[0]}: {src}",
                        pdf_url=pdf_url
                    )
        
        if pdf_count > 0:
            self.logger.info(f"Total PDFs found on {response.url}: {pdf_count}")
    
    def follow_links(self, response):
        """Follow links to other pages"""
        current_depth = response.meta.get('depth', 0)
        
        link_count = 0
        for link in response.css('a::attr(href)').getall():
            if not link:
                continue
                
            # Make absolute URL
            url = urljoin(response.url, link.strip())
            
            # Skip if it's a PDF or not from allowed domain
            if self.pdf_pattern.search(url) or not self.is_allowed_url(url):
                continue
            
            link_count += 1
            if link_count <= 10:  # Limit links per page
                yield scrapy.Request(
                    url=url,
                    callback=self.parse,
                    errback=self.handle_error,
                    meta={
                        'regulation_name': self.regulation_name,
                        'depth': current_depth + 1
                    }
                )
        
        self.logger.info(f"Following {min(link_count, 10)} links from {response.url}")
    
    def is_allowed_url(self, url):
        """Check if URL is from allowed domain"""
        try:
            domain = urlparse(url).netloc
            return any(allowed_domain in domain for allowed_domain in self.allowed_domains)
        except:
            return False
    
    def handle_error(self, failure):
        """Handle request failures"""
        self.logger.error(f"Request failed: {failure.request.url} - {failure.value}")
        
        # Try FlareSolverr fallback if available
        if hasattr(failure.value, 'response') and failure.value.response.status == 403:
            return self.handle_403_error(failure.request)
    
    def handle_403_error(self, request_or_response):
        """Handle 403 errors with FlareSolverr fallback"""
        if hasattr(request_or_response, 'url'):
            url = request_or_response.url
        else:
            url = request_or_response.request.url
            
        self.logger.info(f"Attempting FlareSolverr fallback for 403 error: {url}")
        
        try:
            # Try to use FlareSolverr downloaders
            if self.pdf_pattern.search(url):
                success = self.download_with_flaresolverr_pdf(url)
            else:
                success = self.download_with_flaresolverr_html(url)
                
            if success:
                self.logger.info(f"FlareSolverr fallback successful: {url}")
            else:
                self.logger.error(f"FlareSolverr fallback failed: {url}")
                
        except Exception as e:
            self.logger.error(f"FlareSolverr fallback error: {e}")
    
    def download_with_flaresolverr_pdf(self, pdf_url):
        """Download PDF using FlareSolverr"""
        try:
            import sys
            from pathlib import Path
            sys.path.insert(0, str(Path(__file__).parent.parent.parent))
            from pdf_downloader import PDFDownloader
            
            downloader = PDFDownloader()
            url_hash = hashlib.md5(pdf_url.encode()).hexdigest()[:8]
            filename = f"{self.regulation_name}_{url_hash}"
            
            success = downloader.download_pdf(pdf_url, filename)
            if success:
                self.scraped_pdfs.append({
                    'url': pdf_url,
                    'regulation': self.regulation_name,
                    'method': 'flaresolverr',
                    'downloaded_at': datetime.now().isoformat()
                })
            return success
            
        except Exception as e:
            self.logger.error(f"FlareSolverr PDF download error: {e}")
            return False
    
    def download_with_flaresolverr_html(self, html_url):
        """Download HTML using FlareSolverr"""
        try:
            import sys
            from pathlib import Path
            sys.path.insert(0, str(Path(__file__).parent.parent.parent))
            from html_downloader import HTMLDownloader
            
            downloader = HTMLDownloader()
            url_hash = hashlib.md5(html_url.encode()).hexdigest()[:8]
            filename = f"{self.regulation_name}_{url_hash}"
            
            success = downloader.download_html(html_url, filename)
            if success:
                self.scraped_pages.append({
                    'url': html_url,
                    'regulation': self.regulation_name,
                    'method': 'flaresolverr',
                    'downloaded_at': datetime.now().isoformat()
                })
            return success
            
        except Exception as e:
            self.logger.error(f"FlareSolverr HTML download error: {e}")
            return False
    
    def closed(self, reason):
        """Called when spider closes"""
        # Save summary
        summary = {
            'regulation_name': self.regulation_name,
            'spider_closed_at': datetime.now().isoformat(),
            'close_reason': reason,
            'total_pages_scraped': len(self.scraped_pages),
            'total_pdfs_found': len(self.scraped_pdfs),
            'output_folder': str(self.output_folder),
            'scraped_pages': self.scraped_pages,
            'scraped_pdfs': self.scraped_pdfs
        }
        
        summary_file = self.output_folder / 'spider_summary.json'
        try:
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Spider completed for {self.regulation_name}")
            self.logger.info(f"Pages scraped: {len(self.scraped_pages)}")
            self.logger.info(f"PDFs found: {len(self.scraped_pdfs)}")
            self.logger.info(f"Summary saved: {summary_file}")
            
        except Exception as e:
            self.logger.error(f"Error saving summary: {e}")