#!/usr/bin/env python3
"""
Standalone HTML Downloader using FlareSolverr
Usage: python html_downloader.py <url> [output_filename]
"""

import sys
import requests
import json
import time
import random
import hashlib
from pathlib import Path
from datetime import datetime
from urllib.parse import urlparse, urljoin
import html2text

class HTMLDownloader:
    def __init__(self, flaresolverr_url="http://localhost:8191/v1"):
        self.flaresolverr_url = flaresolverr_url
        self.user_agents = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:124.0) Gecko/20100101 Firefox/124.0",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.3 Safari/605.1.15"
        ]
    
    def solve_with_flaresolverr(self, url):
        """Use FlareSolverr to bypass protection and get HTML content"""
        print(f"🔄 Using FlareSolverr to access: {url}")
        
        # Add random delay
        time.sleep(random.uniform(2, 4))
        
        session_id = f"html_session_{random.randint(1000, 9999)}"
        
        payload = {
            "cmd": "request.get",
            "url": url,
            "maxTimeout": 300000,  # 5 minutes timeout
            "session": session_id,
            "returnOnlyCookies": False
        }
        
        try:
            response = requests.post(self.flaresolverr_url, json=payload, timeout=310)
            response.raise_for_status()
            result = response.json()
            
            # Always clean up the session
            try:
                self.destroy_session(session_id)
            except Exception as cleanup_error:
                print(f"⚠️  Warning: Failed to cleanup session {session_id}: {cleanup_error}")
            
            if result.get("status") == "ok":
                return result.get("solution", {})
            else:
                print(f"❌ FlareSolverr failed: {result.get('message', 'Unknown error')}")
                return None
                
        except Exception as e:
            print(f"❌ FlareSolverr request failed: {e}")
            # Try to cleanup session even on error
            try:
                self.destroy_session(session_id)
            except:
                pass
            return None
    
    def destroy_session(self, session_id):
        """Destroy FlareSolverr session to free resources"""
        destroy_payload = {
            "cmd": "sessions.destroy",
            "session": session_id
        }
        
        try:
            response = requests.post(self.flaresolverr_url, json=destroy_payload, timeout=30)
            if response.status_code == 200:
                print(f"🧹 Cleaned up FlareSolverr session: {session_id}")
            else:
                print(f"⚠️  Session cleanup warning for {session_id}: HTTP {response.status_code}")
        except Exception as e:
            print(f"⚠️  Session cleanup error for {session_id}: {e}")
    
    def extract_metadata(self, html_content, url):
        """Extract metadata from HTML content"""
        from bs4 import BeautifulSoup
        
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            metadata = {
                'url': url,
                'title': '',
                'description': '',
                'keywords': '',
                'author': '',
                'date': '',
                'content_length': len(html_content),
                'extracted_at': datetime.now().isoformat()
            }
            
            # Extract title
            title_tag = soup.find('title')
            if title_tag:
                metadata['title'] = title_tag.get_text().strip()
            
            # Extract meta tags
            meta_tags = soup.find_all('meta')
            for meta in meta_tags:
                name = meta.get('name', '').lower()
                property_attr = meta.get('property', '').lower()
                content = meta.get('content', '')
                
                if name == 'description' or property_attr == 'og:description':
                    metadata['description'] = content
                elif name == 'keywords':
                    metadata['keywords'] = content
                elif name == 'author':
                    metadata['author'] = content
                elif name in ['date', 'publish-date', 'article:published_time'] or property_attr == 'article:published_time':
                    metadata['date'] = content
            
            return metadata
            
        except Exception as e:
            print(f"⚠️ Error extracting metadata: {e}")
            return {
                'url': url,
                'title': '',
                'description': '',
                'keywords': '',
                'author': '',
                'date': '',
                'content_length': len(html_content),
                'extracted_at': datetime.now().isoformat()
            }
    
    def download_html(self, url, output_filename=None):
        """Download HTML content using FlareSolverr"""
        
        if not output_filename:
            # Generate filename from URL
            url_hash = hashlib.sha1(url.encode()).hexdigest()[:8]
            parsed_url = urlparse(url)
            domain = parsed_url.netloc.replace('www.', '')
            output_filename = f"{domain}_{url_hash}"
        
        # Ensure output directory exists
        output_dir = Path("downloads")
        output_dir.mkdir(exist_ok=True)
        
        html_path = output_dir / f"{output_filename}.html"
        text_path = output_dir / f"{output_filename}.txt"
        metadata_path = output_dir / f"{output_filename}_metadata.json"
        
        print(f"📥 Downloading HTML: {url}")
        print(f"📁 Output files:")
        print(f"  HTML: {html_path}")
        print(f"  Text: {text_path}")
        print(f"  Metadata: {metadata_path}")
        
        # Get content using FlareSolverr
        solution = self.solve_with_flaresolverr(url)
        
        if not solution:
            print("❌ Failed to get response from FlareSolverr")
            return False
        
        status_code = solution.get("status", 0)
        html_content = solution.get("response", "")
        headers = solution.get("headers", {})
        
        print(f"📊 Response info:")
        print(f"  Status: {status_code}")
        print(f"  Content-Type: {headers.get('content-type', 'N/A')}")
        print(f"  Content length: {len(html_content)} characters")
        
        if status_code != 200:
            print(f"❌ HTTP error: {status_code}")
            return False
        
        if not html_content:
            print("❌ No content received")
            return False
        
        try:
            # Save raw HTML
            with open(html_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            print(f"✅ HTML saved: {html_path}")
            
            # Convert to text
            h = html2text.HTML2Text()
            h.ignore_links = False
            h.ignore_images = False
            text_content = h.handle(html_content)
            
            with open(text_path, 'w', encoding='utf-8') as f:
                f.write(text_content)
            print(f"✅ Text saved: {text_path}")
            
            # Extract and save metadata
            metadata = self.extract_metadata(html_content, url)
            metadata.update({
                'status_code': status_code,
                'headers': dict(headers),
                'html_file': str(html_path),
                'text_file': str(text_path),
                'text_length': len(text_content)
            })
            
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            print(f"✅ Metadata saved: {metadata_path}")
            
            print("🎉 Download completed successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Error saving files: {e}")
            return False

def main():
    if len(sys.argv) < 2:
        print("Usage: python html_downloader.py <url> [output_filename]")
        print("Example: python html_downloader.py https://example.com/page.html my_page")
        sys.exit(1)
    
    url = sys.argv[1]
    output_filename = sys.argv[2] if len(sys.argv) > 2 else None
    
    print(f"🚀 HTML Downloader with FlareSolverr")
    print(f"🎯 Target URL: {url}")
    
    downloader = HTMLDownloader()
    
    # Check if FlareSolverr is running
    try:
        test_response = requests.get("http://localhost:8191/v1", timeout=5)
        if test_response.status_code == 405:  # Method not allowed is expected for GET
            print("✅ FlareSolverr is running")
        else:
            print("⚠️  FlareSolverr may not be running properly")
    except:
        print("❌ FlareSolverr is not running!")
        print("Start it with: docker run -d -p 8191:8191 ghcr.io/flaresolverr/flaresolverr:latest")
        sys.exit(1)
    
    # Install required dependency if needed
    try:
        import bs4
        import html2text
    except ImportError:
        print("❌ Missing dependencies. Install with:")
        print("pip install beautifulsoup4 html2text")
        sys.exit(1)
    
    success = downloader.download_html(url, output_filename)
    
    if not success:
        print("💥 Download failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()