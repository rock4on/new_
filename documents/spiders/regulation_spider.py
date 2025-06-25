import scrapy
import json
import hashlib
import re
import requests
from pathlib import Path
from datetime import datetime
from urllib.parse import urlparse, urljoin
# from documents.items import DocItem  # Not needed for direct downloads

class RegulationSpider(scrapy.Spider):
    name = "regulation"
    
    custom_settings = {
        'ROBOTSTXT_OBEY': False,
        'DOWNLOAD_DELAY': 2,
        'RANDOMIZE_DOWNLOAD_DELAY': 0.5,
        'CONCURRENT_REQUESTS': 2,
        'CONCURRENT_REQUESTS_PER_DOMAIN': 1,
        'RETRY_TIMES': 3,
        'DOWNLOAD_TIMEOUT': 60,
        'DEPTH_LIMIT': 3,
        'CLOSESPIDER_PAGECOUNT': 50,  # Limit for testing
        'HTTPERROR_ALLOWED_CODES': [403, 404, 500, 502, 503],
        'USER_AGENT': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'LOG_LEVEL': 'INFO'
    }
    
    def __init__(self, regulation_name=None, country=None, start_urls=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Set regulation name and country
        self.regulation_name = regulation_name or "unknown_regulation"
        self.country = country or "Unknown_Country"
        self.logger.info(f"🚀 Initializing spider for regulation: [{self.country}] {self.regulation_name}")
        
        # Parse start URLs with extensive logging
        self.start_urls = self.parse_start_urls(start_urls)
        self.logger.info(f"📋 Parsed {len(self.start_urls)} start URLs")
        
        # Set up domains - be more permissive
        self.allowed_domains = self.setup_allowed_domains()
        self.logger.info(f"🌐 Allowed domains: {self.allowed_domains}")
        
        # Create output folder - settings not available in __init__, use default
        safe_name = self.make_safe_filename(self.regulation_name)
        self.output_folder = Path("regulation_downloads") / safe_name
            
        self.output_folder.mkdir(parents=True, exist_ok=True)
        self.logger.info(f"📁 Output folder: {self.output_folder}")
        
        # Tracking
        self.pages_crawled = 0
        self.pdfs_found = 0
        self.links_followed = 0
        self.scraped_items = []
        
        # PDF pattern
        self.pdf_pattern = re.compile(r'\.pdf$', re.IGNORECASE)
        
        self.logger.info("✅ Spider initialization complete")
    
    def parse_start_urls(self, start_urls):
        """Parse and validate start URLs with detailed logging"""
        if not start_urls:
            self.logger.error("❌ No start URLs provided")
            return []
        
        if isinstance(start_urls, str):
            raw_urls = [url.strip() for url in start_urls.split(',')]
        else:
            raw_urls = list(start_urls)
        
        parsed_urls = []
        for i, url in enumerate(raw_urls):
            if not url:
                continue
                
            # Add scheme if missing
            if not url.startswith(('http://', 'https://')):
                if url.startswith('www.'):
                    url = f'https://{url}'
                elif '.' in url:
                    url = f'https://{url}'
                else:
                    self.logger.warning(f"⚠️  Skipping invalid URL {i+1}: {url}")
                    continue
            
            parsed_urls.append(url)
            self.logger.info(f"✅ Start URL {i+1}: {url}")
        
        return parsed_urls
    
    def setup_allowed_domains(self):
        """Set up allowed domains with permissive matching"""
        domains = set()
        
        for url in self.start_urls:
            try:
                parsed = urlparse(url)
                domain = parsed.netloc.lower()
                
                # Add main domain
                domains.add(domain)
                
                # Add without www
                if domain.startswith('www.'):
                    domains.add(domain[4:])
                else:
                    domains.add(f'www.{domain}')
                
                self.logger.info(f"🌐 Added domain: {domain}")
                
            except Exception as e:
                self.logger.error(f"❌ Error parsing domain from {url}: {e}")
        
        return list(domains)
    
    def make_safe_filename(self, name):
        """Convert name to safe filename"""
        safe = re.sub(r'[<>:"/\\|?*]', '_', name)
        safe = re.sub(r'\s+', '_', safe)
        return safe[:100].strip('._')
    
    def start_requests(self):
        """Generate initial requests with logging"""
        # Now we can access settings, update output folder if needed
        try:
            files_store = self.crawler.settings.get('FILES_STORE')
            if files_store and files_store != 'regulation_downloads':
                self.output_folder = Path(files_store)
                self.output_folder.mkdir(parents=True, exist_ok=True)
                self.logger.info(f"📁 Updated output folder from settings: {self.output_folder}")
        except:
            # Keep default if settings not accessible
            pass
        
        self.logger.info(f"🎯 Starting {len(self.start_urls)} initial requests")
        
        for i, url in enumerate(self.start_urls):
            self.logger.info(f"📨 Request {i+1}: {url}")
            
            # Check if start URL is a PDF
            if self.pdf_pattern.search(url):
                self.logger.info(f"📄 Start URL is a PDF, downloading with FlareSolverr: {url}")
                self.pdfs_found += 1
                
                # Use FlareSolverr directly for PDF URLs
                if self.download_pdf_with_flaresolverr(url, url):
                    self.logger.info(f"✅ Start URL PDF downloaded via FlareSolverr")
                else:
                    self.logger.error(f"❌ Start URL PDF download failed with FlareSolverr")
                
                # Don't make a Scrapy request for PDF URLs
                continue
            
            # Regular HTML request
            yield scrapy.Request(
                url=url,
                callback=self.parse,
                errback=self.handle_error,
                meta={
                    'regulation_name': self.regulation_name,
                    'depth': 0
                },
                dont_filter=False  # Allow Scrapy's duplicate filtering
            )
    
    def parse(self, response):
        """Main parsing method with extensive logging"""
        self.pages_crawled += 1
        current_depth = response.meta.get('depth', 0)
        
        self.logger.info(f"🔍 PARSING #{self.pages_crawled}: {response.url}")
        self.logger.info(f"📊 Status: {response.status}, Depth: {current_depth}, Size: {len(response.body)} bytes")
        
        # Handle error responses
        if response.status == 403:
            self.logger.warning(f"🚫 403 Forbidden: {response.url}")
            if self.handle_403_error(response):
                self.logger.info(f"✅ 403 error handled successfully with FlareSolverr")
            else:
                self.logger.error(f"❌ Failed to handle 403 error for: {response.url}")
            return
        elif response.status >= 400:
            self.logger.warning(f"❌ HTTP {response.status}: {response.url}")
            return
        
        # Save page content
        self.save_page_content(response)
        
        # Find and save PDFs directly
        self.extract_and_save_pdfs(response)
        
        # Follow links if within depth limit
        depth_limit = getattr(self, 'custom_settings', {}).get('DEPTH_LIMIT', 3)
        if current_depth < depth_limit:
            self.logger.info(f"🔗 CALLING follow_links at depth {current_depth} (limit: {depth_limit})")
            yield from self.follow_links(response, current_depth)
        else:
            self.logger.info(f"🛑 Max depth {depth_limit} reached")
    
    def save_page_content(self, response):
        """Save HTML and text content directly to files"""
        try:
            # Generate unique filename
            url_hash = hashlib.md5(response.url.encode()).hexdigest()[:8]
            timestamp = datetime.now().strftime("%H%M%S")
            base_filename = f"{self.make_safe_filename(self.regulation_name)}_{self.pages_crawled:03d}_{url_hash}_{timestamp}"
            
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
                'country': self.country,
                'title': response.css('title::text').get() or '',
                'status_code': response.status,
                'depth': response.meta.get('depth', 0),
                'scraped_at': datetime.now().isoformat(),
                'html_file': html_file.name,
                'text_file': text_file.name,
                'content_length': len(response.text),
                'text_length': len(text_content),
                'page_number': self.pages_crawled
            }
            
            metadata_file = self.output_folder / f"{base_filename}_metadata.json"
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            
            self.scraped_items.append(metadata)
            self.logger.info(f"💾 SAVED PAGE #{self.pages_crawled}: {html_file.name}")
            
        except Exception as e:
            self.logger.error(f"❌ Error saving page content: {e}")
    
    def extract_text(self, response):
        """Extract clean text from HTML"""
        try:
            # Get text from body, excluding script and style tags
            text_parts = response.css('body *:not(script):not(style)::text').getall()
            if not text_parts:
                # Fallback: get all text
                text_parts = response.css('::text').getall()
            
            text = ' '.join(text_parts)
            # Clean up whitespace
            text = ' '.join(text.split())
            return text
        except Exception as e:
            self.logger.error(f"❌ Error extracting text: {e}")
            return ""
    
    def extract_and_save_pdfs(self, response):
        """Find PDFs and save them directly"""
        pdf_count = 0
        
        # Find PDF links
        all_links = response.css('a::attr(href)').getall()
        self.logger.info(f"🔗 Found {len(all_links)} total links on page")
        
        for link in all_links:
            if not link:
                continue
                
            # Make absolute URL
            pdf_url = urljoin(response.url, link.strip())
            
            # Check if it's a PDF
            if self.pdf_pattern.search(pdf_url):
                pdf_count += 1
                self.pdfs_found += 1
                
                self.logger.info(f"📄 FOUND PDF #{self.pdfs_found}: {pdf_url}")
                
                # Use FlareSolverr directly for PDF URLs
                if self.download_pdf_with_flaresolverr(pdf_url, response.url):
                    self.logger.info(f"✅ PDF downloaded via FlareSolverr")
                else:
                    self.logger.error(f"❌ PDF download failed with FlareSolverr")
        
        # Check for PDFs in iframes, embeds, objects
        for selector_name, selector in [
            ('iframe', 'iframe::attr(src)'),
            ('embed', 'embed::attr(src)'),
            ('object', 'object::attr(data)')
        ]:
            sources = response.css(selector).getall()
            for src in sources:
                if not src:
                    continue
                    
                pdf_url = urljoin(response.url, src.strip())
                if self.pdf_pattern.search(pdf_url):
                    pdf_count += 1
                    self.pdfs_found += 1
                    
                    self.logger.info(f"📄 FOUND PDF in {selector_name} #{self.pdfs_found}: {pdf_url}")
                    
                    # Use FlareSolverr directly for PDF URLs
                    if self.download_pdf_with_flaresolverr(pdf_url, response.url):
                        self.logger.info(f"✅ PDF downloaded via FlareSolverr")
                    else:
                        self.logger.error(f"❌ PDF download failed with FlareSolverr")
        
        if pdf_count > 0:
            self.logger.info(f"📄 TOTAL PDFs found on this page: {pdf_count}")
        else:
            self.logger.info(f"📄 No PDFs found on this page")
    
    def download_pdf_directly(self, pdf_url, source_page):
        """Download PDF directly using requests"""
        try:
            self.logger.info(f"⬇️  Downloading PDF: {pdf_url}")
            
            user_agent = getattr(self, 'custom_settings', {}).get('USER_AGENT', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36')
            headers = {
                'User-Agent': user_agent,
                'Referer': source_page,
                'Accept': 'application/pdf,application/octet-stream,*/*'
            }
            
            response = requests.get(pdf_url, headers=headers, timeout=30, stream=True)
            
            if response.status_code == 200:
                # Check if content is actually a PDF
                content_type = response.headers.get('content-type', '').lower()
                
                # Read first chunk to check PDF signature
                first_chunk = next(response.iter_content(chunk_size=1024), b'')
                if not first_chunk.startswith(b'%PDF'):
                    self.logger.warning(f"⚠️  URL doesn't return PDF content: {pdf_url}")
                    self.logger.warning(f"   Content-Type: {content_type}")
                    self.logger.warning(f"   First bytes: {first_chunk[:50]}")
                    return False
                
                # Generate filename
                url_hash = hashlib.md5(pdf_url.encode()).hexdigest()[:8]
                safe_name = self.make_safe_filename(self.regulation_name)
                pdf_filename = f"{safe_name}_pdf_{self.pdfs_found:03d}_{url_hash}.pdf"
                pdf_path = self.output_folder / pdf_filename
                
                # Save PDF
                with open(pdf_path, 'wb') as f:
                    f.write(first_chunk)  # Write the first chunk we already read
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                
                file_size = pdf_path.stat().st_size
                
                # Validate PDF file size
                if file_size < 1000:  # Less than 1KB is probably not a real PDF
                    self.logger.warning(f"⚠️  PDF file too small ({file_size} bytes), might be error page")
                    pdf_path.unlink()  # Delete the file
                    return False
                
                self.logger.info(f"💾 SAVED PDF: {pdf_filename} ({file_size} bytes)")
                
                # Save PDF metadata
                pdf_metadata = {
                    'url': pdf_url,
                    'source_page': source_page,
                    'regulation': self.regulation_name,
                    'country': self.country,
                    'filename': pdf_filename,
                    'file_size': file_size,
                    'content_type': content_type,
                    'downloaded_at': datetime.now().isoformat(),
                    'method': 'direct_download'
                }
                
                metadata_file = self.output_folder / f"{pdf_filename}_metadata.json"
                with open(metadata_file, 'w', encoding='utf-8') as f:
                    json.dump(pdf_metadata, f, indent=2, ensure_ascii=False)
                
                return True
            elif response.status_code == 403:
                self.logger.warning(f"🚫 403 Forbidden for PDF: {pdf_url}")
                # Could add FlareSolverr fallback here
                return False
            else:
                self.logger.warning(f"❌ PDF download failed: HTTP {response.status_code}")
                return False
                
        except requests.exceptions.RequestException as e:
            self.logger.error(f"❌ Network error downloading PDF: {e}")
            return False
        except Exception as e:
            self.logger.error(f"❌ PDF download error: {e}")
            return False
    
    def follow_links(self, response, current_depth):
        """Follow links to other pages with extensive logging"""
        all_links = response.css('a::attr(href)').getall()
        self.logger.info(f"🔗 Processing {len(all_links)} links at depth {current_depth}")
        
        valid_links = []
        filtered_out = {'no_href': 0, 'pdf': 0, 'external': 0, 'invalid': 0, 'duplicate': 0}
        debug_samples = {'external': [], 'invalid': []}
        
        for link in all_links:
            if not link:
                filtered_out['no_href'] += 1
                continue
                
            # Make absolute URL
            url = urljoin(response.url, link.strip())
            
            # Skip PDF links (already handled)
            if self.pdf_pattern.search(url):
                filtered_out['pdf'] += 1
                continue
            
            # Check domain (be more permissive)
            if not self.is_allowed_url(url):
                filtered_out['external'] += 1
                if len(debug_samples['external']) < 3:
                    debug_samples['external'].append(url)
                continue
            
            # Basic URL validation
            try:
                parsed = urlparse(url)
                if not parsed.scheme or not parsed.netloc:
                    filtered_out['invalid'] += 1
                    if len(debug_samples['invalid']) < 3:
                        debug_samples['invalid'].append(url)
                    continue
            except:
                filtered_out['invalid'] += 1
                if len(debug_samples['invalid']) < 3:
                    debug_samples['invalid'].append(url)
                continue
            
            # Skip duplicates in this batch
            if url in valid_links:
                filtered_out['duplicate'] += 1
                continue
            
            valid_links.append(url)
        
        self.logger.info(f"📊 Link filtering results:")
        self.logger.info(f"   ✅ Valid links: {len(valid_links)}")
        self.logger.info(f"   ❌ Filtered out:")
        self.logger.info(f"      No href: {filtered_out['no_href']}")
        self.logger.info(f"      PDFs: {filtered_out['pdf']}")
        self.logger.info(f"      External: {filtered_out['external']}")
        self.logger.info(f"      Invalid: {filtered_out['invalid']}")
        self.logger.info(f"      Duplicates: {filtered_out['duplicate']}")
        
        # Show samples of filtered URLs for debugging
        if debug_samples['external']:
            self.logger.info(f"   🌍 Sample external URLs: {debug_samples['external']}")
        if debug_samples['invalid']:
            self.logger.info(f"   ❌ Sample invalid URLs: {debug_samples['invalid']}")
        
        # Show first few valid links for debugging
        if valid_links:
            sample_links = valid_links[:3]
            self.logger.info(f"   ✅ Sample valid links: {sample_links}")
        
        # Limit links per page for performance
        max_links = min(len(valid_links), 15)
        requests_yielded = 0
        
        for i, url in enumerate(valid_links[:max_links]):
            self.links_followed += 1
            requests_yielded += 1
            
            self.logger.info(f"🔗 YIELDING LINK #{self.links_followed} (depth {current_depth + 1}): {url}")
            
            yield scrapy.Request(
                url=url,
                callback=self.parse,
                errback=self.handle_error,
                meta={
                    'regulation_name': self.regulation_name,
                    'depth': current_depth + 1
                },
                dont_filter=False
            )
        
        self.logger.info(f"📨 YIELDED {requests_yielded} REQUESTS from this page")
        
        if len(valid_links) > max_links:
            self.logger.info(f"🔗 Limited to {max_links} links (had {len(valid_links)} valid links)")
    
    def is_allowed_url(self, url):
        """Check if URL is from allowed domain - more permissive"""
        try:
            parsed = urlparse(url)
            domain = parsed.netloc.lower()
            
            # Log domain check for debugging
            self.logger.debug(f"🔍 Checking domain: {domain} against {self.allowed_domains}")
            
            # Check exact match
            if domain in self.allowed_domains:
                self.logger.debug(f"✅ Exact match: {domain}")
                return True
            
            # Check if any allowed domain is a subdomain or parent domain
            for allowed in self.allowed_domains:
                # Check if domain is subdomain of allowed
                if domain.endswith(f'.{allowed}'):
                    self.logger.debug(f"✅ Subdomain match: {domain} is subdomain of {allowed}")
                    return True
                
                # Check if allowed is subdomain of domain  
                if allowed.endswith(f'.{domain}'):
                    self.logger.debug(f"✅ Parent domain match: {allowed} is subdomain of {domain}")
                    return True
                    
                # Remove www and check again
                clean_domain = domain.replace('www.', '')
                clean_allowed = allowed.replace('www.', '')
                
                if clean_domain == clean_allowed:
                    self.logger.debug(f"✅ Clean match: {clean_domain}")
                    return True
                
                # Check partial matches (more permissive)
                if clean_domain in clean_allowed or clean_allowed in clean_domain:
                    self.logger.debug(f"✅ Partial match: {clean_domain} <-> {clean_allowed}")
                    return True
            
            self.logger.debug(f"❌ No match for: {domain}")
            return False
            
        except Exception as e:
            self.logger.error(f"❌ Error checking domain for {url}: {e}")
            return False
    
    def handle_error(self, failure):
        """Handle request failures"""
        self.logger.error(f"❌ REQUEST FAILED: {failure.request.url}")
        self.logger.error(f"   Error: {failure.value}")
        
        # Handle 403 errors with FlareSolverr fallback
        if hasattr(failure.value, 'response') and failure.value.response:
            if failure.value.response.status == 403:
                self.logger.info(f"🔄 Trying FlareSolverr fallback for 403 error: {failure.request.url}")
                if self.download_html_with_flaresolverr(failure.request.url):
                    self.logger.info(f"✅ FlareSolverr fallback successful for: {failure.request.url}")
                else:
                    self.logger.error(f"❌ FlareSolverr fallback failed for: {failure.request.url}")
    
    def download_pdf_with_flaresolverr(self, pdf_url, source_page):
        """Download PDF using FlareSolverr"""
        try:
            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent.parent))
            from pdf_downloader import PDFDownloader
            
            downloader = PDFDownloader()
            url_hash = hashlib.md5(pdf_url.encode()).hexdigest()[:8]
            safe_name = self.make_safe_filename(self.regulation_name)
            filename = f"{safe_name}_pdf_flare_{self.pdfs_found:03d}_{url_hash}"
            
            # Change to our output directory temporarily
            import os
            original_cwd = os.getcwd()
            os.chdir(self.output_folder.parent)
            
            try:
                success = downloader.download_pdf(pdf_url, filename)
                
                if success:
                    # Move the downloaded file to our output folder
                    downloaded_file = Path("downloads") / f"{filename}.pdf"
                    target_file = self.output_folder / f"{filename}.pdf"
                    
                    if downloaded_file.exists():
                        downloaded_file.rename(target_file)
                        self.logger.info(f"💾 SAVED PDF via FlareSolverr: {target_file.name}")
                        
                        # Create metadata
                        pdf_metadata = {
                            'url': pdf_url,
                            'source_page': source_page,
                            'regulation': self.regulation_name,
                            'country': self.country,
                            'filename': target_file.name,
                            'file_size': target_file.stat().st_size,
                            'downloaded_at': datetime.now().isoformat(),
                            'method': 'flaresolverr'
                        }
                        
                        metadata_file = self.output_folder / f"{filename}_metadata.json"
                        with open(metadata_file, 'w', encoding='utf-8') as f:
                            json.dump(pdf_metadata, f, indent=2, ensure_ascii=False)
                        
                        return True
                    
                return False
                
            finally:
                os.chdir(original_cwd)
                
        except Exception as e:
            self.logger.error(f"❌ FlareSolverr PDF download error: {e}")
            return False
    
    def handle_403_error(self, response):
        """Handle 403 errors with HTML downloader fallback"""
        self.logger.warning(f"🚫 403 error for: {response.url}")
        
        try:
            # Use HTML downloader for 403 errors
            if self.download_html_with_flaresolverr(response.url):
                self.logger.info(f"✅ HTML downloaded via FlareSolverr for 403 error")
                return True
            else:
                self.logger.error(f"❌ FlareSolverr HTML download failed for 403 error")
                return False
        except Exception as e:
            self.logger.error(f"❌ Error handling 403 with FlareSolverr: {e}")
            return False
    
    def download_html_with_flaresolverr(self, html_url):
        """Download HTML using FlareSolverr for 403 errors"""
        try:
            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent.parent))
            from html_downloader import HTMLDownloader
            
            downloader = HTMLDownloader()
            url_hash = hashlib.md5(html_url.encode()).hexdigest()[:8]
            safe_name = self.make_safe_filename(self.regulation_name)
            timestamp = datetime.now().strftime("%H%M%S")
            filename = f"{safe_name}_403_{self.pages_crawled:03d}_{url_hash}_{timestamp}"
            
            # Change to our output directory temporarily
            import os
            original_cwd = os.getcwd()
            os.chdir(self.output_folder.parent)
            
            try:
                success = downloader.download_html(html_url, filename)
                
                if success:
                    # Move the downloaded files to our output folder
                    downloaded_html = Path("downloads") / f"{filename}.html"
                    downloaded_txt = Path("downloads") / f"{filename}.txt"
                    downloaded_meta = Path("downloads") / f"{filename}_metadata.json"
                    
                    target_html = self.output_folder / f"{filename}.html"
                    target_txt = self.output_folder / f"{filename}.txt"
                    target_meta = self.output_folder / f"{filename}_metadata.json"
                    
                    # Move files if they exist
                    if downloaded_html.exists():
                        downloaded_html.rename(target_html)
                        self.logger.info(f"💾 SAVED HTML via FlareSolverr: {target_html.name}")
                    
                    if downloaded_txt.exists():
                        downloaded_txt.rename(target_txt)
                        self.logger.info(f"💾 SAVED TEXT via FlareSolverr: {target_txt.name}")
                    
                    if downloaded_meta.exists():
                        # Update metadata with our regulation info
                        try:
                            with open(downloaded_meta, 'r', encoding='utf-8') as f:
                                metadata = json.load(f)
                            
                            metadata.update({
                                'regulation': self.regulation_name,
                                'country': self.country,
                                'method': 'flaresolverr_403_fallback',
                                'original_status': '403_forbidden'
                            })
                            
                            with open(target_meta, 'w', encoding='utf-8') as f:
                                json.dump(metadata, f, indent=2, ensure_ascii=False)
                                
                            downloaded_meta.unlink()  # Remove original
                            
                        except Exception as meta_error:
                            self.logger.warning(f"⚠️  Could not update metadata: {meta_error}")
                            if downloaded_meta.exists():
                                downloaded_meta.rename(target_meta)
                    
                    # Update our tracking
                    self.pages_crawled += 1
                    self.scraped_items.append({
                        'url': html_url,
                        'regulation': self.regulation_name,
                        'country': self.country,
                        'method': 'flaresolverr_403_fallback',
                        'files': [str(target_html), str(target_txt), str(target_meta)]
                    })
                    
                    # Try to extract links from the downloaded HTML for further crawling
                    if target_html.exists():
                        self.process_downloaded_html_for_links(target_html, html_url)
                    
                    return True
                
                return False
                
            finally:
                os.chdir(original_cwd)
                
        except Exception as e:
            self.logger.error(f"❌ FlareSolverr HTML download error: {e}")
            return False
    
    def process_downloaded_html_for_links(self, html_file, original_url):
        """Process downloaded HTML file to extract links for further crawling"""
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
            
            self.logger.info(f"🔍 Processing downloaded HTML for links: {html_file.name}")
            
            # Extract PDFs from the downloaded content
            self.extract_and_save_pdfs(response)
            
            # Follow links if we haven't reached depth limit
            current_depth = 1  # Assume this is depth 1 since it's a fallback
            depth_limit = getattr(self, 'custom_settings', {}).get('DEPTH_LIMIT', 3)
            
            if current_depth < depth_limit:
                self.logger.info(f"🔗 Following links from FlareSolverr downloaded page")
                # Process links but don't yield directly, instead queue them
                all_links = response.css('a::attr(href)').getall()
                
                valid_links = []
                for link in all_links:
                    if not link:
                        continue
                    
                    from urllib.parse import urljoin
                    url = urljoin(original_url, link.strip())
                    
                    # Skip PDF links and check domain
                    if not self.pdf_pattern.search(url) and self.is_allowed_url(url):
                        valid_links.append(url)
                
                # Log the links we found
                if valid_links:
                    self.logger.info(f"🔗 Found {len(valid_links)} valid links in FlareSolverr downloaded page")
                    for i, url in enumerate(valid_links[:5]):  # Log first 5
                        self.logger.info(f"   Link {i+1}: {url}")
                    
                    # Note: We could implement a queue system here to process these links
                    # For now, just log them as potential follow-up URLs
                    
        except Exception as e:
            self.logger.error(f"❌ Error processing downloaded HTML for links: {e}")
    
    def closed(self, reason):
        """Called when spider closes - save comprehensive summary"""
        # Count actual files
        html_files = list(self.output_folder.glob('*.html'))
        pdf_files = list(self.output_folder.glob('*.pdf'))
        txt_files = list(self.output_folder.glob('*.txt'))
        json_files = list(self.output_folder.glob('*.json'))
        
        summary = {
            'regulation_name': self.regulation_name,
            'country': self.country,
            'spider_closed_at': datetime.now().isoformat(),
            'close_reason': reason,
            'crawl_stats': {
                'pages_crawled': self.pages_crawled,
                'pdfs_found': self.pdfs_found,
                'links_followed': self.links_followed,
                'items_scraped': len(self.scraped_items)
            },
            'files_created': {
                'html_files': len(html_files),
                'pdf_files': len(pdf_files),
                'txt_files': len(txt_files),
                'json_files': len(json_files),
                'total_files': len(list(self.output_folder.glob('*')))
            },
            'output_folder': str(self.output_folder),
            'start_urls': self.start_urls,
            'allowed_domains': self.allowed_domains,
            'scraped_items': self.scraped_items
        }
        
        summary_file = self.output_folder / 'spider_summary.json'
        try:
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"🎉 SPIDER COMPLETED FOR: [{self.country}] {self.regulation_name}")
            self.logger.info(f"📊 FINAL STATS:")
            self.logger.info(f"   📄 Pages crawled: {self.pages_crawled}")
            self.logger.info(f"   📎 PDFs found: {self.pdfs_found}")
            self.logger.info(f"   🔗 Links followed: {self.links_followed}")
            self.logger.info(f"   📁 HTML files: {len(html_files)}")
            self.logger.info(f"   📄 PDF files: {len(pdf_files)}")
            self.logger.info(f"   📝 Text files: {len(txt_files)}")
            self.logger.info(f"   📋 Total files: {len(list(self.output_folder.glob('*')))}")
            self.logger.info(f"   📁 Output: {self.output_folder}")
            self.logger.info(f"   📋 Summary: {summary_file}")
            
        except Exception as e:
            self.logger.error(f"❌ Error saving summary: {e}")