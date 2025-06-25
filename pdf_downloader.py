#!/usr/bin/env python3
"""
Standalone PDF Downloader using FlareSolverr
Usage: python pdf_downloader.py <pdf_url> [output_filename]
"""

import sys
import requests
import json
import time
import random
import base64
import hashlib
from pathlib import Path
from datetime import datetime
import re
from urllib.parse import urljoin, urlparse

class PDFDownloader:
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
        """Use FlareSolverr to bypass protection"""
        print(f"🔄 Using FlareSolverr to access: {url}")
        
        # Add random delay
        time.sleep(random.uniform(2, 4))
        
        payload = {
            "cmd": "request.get",
            "url": url,
            "maxTimeout": 300000,  # 5 minutes timeout
            "session": f"pdf_session_{random.randint(1000, 9999)}",
            "returnOnlyCookies": False
        }
        
        try:
            response = requests.post(self.flaresolverr_url, json=payload, timeout=310)
            response.raise_for_status()
            result = response.json()
            
            if result.get("status") == "ok":
                return result.get("solution", {})
            else:
                print(f"❌ FlareSolverr failed: {result.get('message', 'Unknown error')}")
                return None
                
        except Exception as e:
            print(f"❌ FlareSolverr request failed: {e}")
            return None
    
    def extract_pdf_from_html(self, html_content, original_url):
        """Extract actual PDF URL from HTML viewer"""
        patterns = [
            r'data-src=["\']([^"\']*\.pdf[^"\']*)["\']',
            r'src=["\']([^"\']*\.pdf[^"\']*)["\']',
            r'file=["\']([^"\']*\.pdf[^"\']*)["\']',
            r'url=["\']([^"\']*\.pdf[^"\']*)["\']',
            r'pdfUrl\s*[:=]\s*["\']([^"\']*)["\']',
            r'pdf_url\s*[:=]\s*["\']([^"\']*)["\']',
            r'href=["\']([^"\']*\.pdf[^"\']*)["\']',
            r'["\']([^"\']*\.pdf[^"\']*)["\']',
        ]
        
        print("🔍 Searching for PDF URLs in HTML content...")
        
        for pattern in patterns:
            matches = re.findall(pattern, html_content, re.IGNORECASE)
            for match in matches:
                if match and match != original_url and '.pdf' in match.lower():
                    print(f"✅ Found potential PDF URL: {match}")
                    
                    # Convert relative URLs to absolute
                    if match.startswith('/'):
                        parsed_original = urlparse(original_url)
                        base_url = f"{parsed_original.scheme}://{parsed_original.netloc}"
                        full_url = urljoin(base_url, match)
                        return full_url
                    elif match.startswith('http'):
                        return match
        
        return None
    
    def download_pdf(self, pdf_url, output_filename=None):
        """Download PDF using FlareSolverr with multiple strategies"""
        
        if not output_filename:
            # Generate filename from URL
            url_hash = hashlib.sha1(pdf_url.encode()).hexdigest()
            output_filename = f"downloaded_{url_hash}.pdf"
        
        # Ensure output directory exists
        output_path = Path("downloads") / output_filename
        output_path.parent.mkdir(exist_ok=True)
        
        print(f"📥 Downloading PDF: {pdf_url}")
        print(f"📁 Output file: {output_path}")
        
        # Step 1: Try FlareSolverr to get the content
        solution = self.solve_with_flaresolverr(pdf_url)
        
        if not solution:
            print("❌ Failed to get response from FlareSolverr")
            return False
        
        status_code = solution.get("status", 0)
        response_content = solution.get("response", "")
        content_type = solution.get("headers", {}).get("content-type", "")
        
        print(f"📊 Response info:")
        print(f"  Status: {status_code}")
        print(f"  Content-Type: {content_type}")
        print(f"  Content length: {len(response_content)}")
        print(f"  Content preview: {response_content[:100]}...")
        
        if status_code != 200:
            print(f"❌ HTTP error: {status_code}")
            return False
        
        # Step 2: Check if this is direct PDF content
        is_direct_pdf = (
            response_content.startswith('%PDF') or
            response_content.startswith('JVBERi0') or  # base64 PDF
            (content_type and 'pdf' in content_type.lower())
        )
        
        if is_direct_pdf:
            print("📄 Processing direct PDF content")
            return self.save_direct_pdf(response_content, output_path)
        
        # Step 3: Check if this is HTML with PDF viewer
        elif '<html' in response_content.lower() or '<!doctype' in response_content.lower():
            print("📄 Found HTML content - looking for PDF viewer")
            
            # Save HTML for debugging
            html_path = output_path.with_suffix('.html')
            with open(html_path, 'w', encoding='utf-8') as f:
                f.write(response_content)
            print(f"💾 Saved HTML content to: {html_path}")
            
            # Try to extract actual PDF URL
            actual_pdf_url = self.extract_pdf_from_html(response_content, pdf_url)
            
            if actual_pdf_url and actual_pdf_url != pdf_url:
                print(f"🎯 Found actual PDF URL: {actual_pdf_url}")
                return self.download_pdf(actual_pdf_url, output_filename)
            else:
                # Try using cookies from FlareSolverr for direct download
                print("🔄 Trying direct download with FlareSolverr session...")
                return self.try_direct_download(pdf_url, solution, output_path)
        
        # Step 4: Try to save whatever content we got
        else:
            print("📄 Unknown content type - saving as-is")
            return self.save_unknown_content(response_content, output_path)
    
    def save_direct_pdf(self, content, output_path):
        """Save direct PDF content"""
        try:
            pdf_content = None
            
            if content.startswith('%PDF'):
                # Direct PDF text, encode to bytes
                try:
                    pdf_content = content.encode('latin1')
                except:
                    pdf_content = content.encode('utf-8', errors='ignore')
                    
            elif content.startswith('JVBERi0'):
                # Base64 encoded PDF
                try:
                    pdf_content = base64.b64decode(content)
                except Exception as e:
                    print(f"❌ Failed to decode base64: {e}")
                    return False
            
            if pdf_content and pdf_content.startswith(b'%PDF'):
                with open(output_path, 'wb') as f:
                    f.write(pdf_content)
                
                file_size = output_path.stat().st_size
                print(f"✅ PDF downloaded successfully: {output_path} ({file_size} bytes)")
                return True
            else:
                print("❌ Content doesn't appear to be a valid PDF")
                return False
                
        except Exception as e:
            print(f"❌ Error saving PDF: {e}")
            return False
    
    def try_direct_download(self, pdf_url, solution, output_path):
        """Try direct download using FlareSolverr cookies"""
        try:
            cookies = solution.get("cookies", [])
            cookie_dict = {cookie["name"]: cookie["value"] for cookie in cookies}
            user_agent = solution.get("userAgent", self.user_agents[0])
            
            response = requests.get(
                pdf_url,
                cookies=cookie_dict,
                headers={
                    "User-Agent": user_agent,
                    "Accept": "application/pdf,application/octet-stream,*/*",
                    "Referer": "https://www.sec.gov.ph/",
                },
                timeout=60,
                stream=True
            )
            
            if response.status_code == 200:
                # Check if it's actually a PDF
                content_start = response.content[:10]
                if content_start.startswith(b'%PDF'):
                    with open(output_path, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                    
                    file_size = output_path.stat().st_size
                    print(f"✅ PDF downloaded via direct request: {output_path} ({file_size} bytes)")
                    return True
                else:
                    print("❌ Direct download didn't return PDF content")
            else:
                print(f"❌ Direct download failed: HTTP {response.status_code}")
                
        except Exception as e:
            print(f"❌ Direct download error: {e}")
        
        return False
    
    def save_unknown_content(self, content, output_path):
        """Save unknown content as text file"""
        try:
            text_path = output_path.with_suffix('.txt')
            with open(text_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            file_size = text_path.stat().st_size
            print(f"💾 Content saved as text: {text_path} ({file_size} bytes)")
            return True
            
        except Exception as e:
            print(f"❌ Error saving content: {e}")
            return False

def main():
    if len(sys.argv) < 2:
        print("Usage: python pdf_downloader.py <pdf_url> [output_filename]")
        print("Example: python pdf_downloader.py https://example.com/document.pdf my_document.pdf")
        sys.exit(1)
    
    pdf_url = sys.argv[1]
    output_filename = sys.argv[2] if len(sys.argv) > 2 else None
    
    print(f"🚀 PDF Downloader with FlareSolverr")
    print(f"🎯 Target URL: {pdf_url}")
    
    downloader = PDFDownloader()
    
    # Check if FlareSolverr is running
    try:
        test_response = requests.get("http://localhost:8191/v1", timeout=5)
        if test_response.status_code == 405:  # Method not allowed is expected for GET
            print("✅ FlareSolverr is running")
        else:
            print("⚠️  FlareSolverr may not be running properly")
    except:
        print("❌ FlareSolverr is not running! Start it with: docker run -d -p 8191:8191 ghcr.io/flaresolverr/flaresolverr:latest")
        sys.exit(1)
    
    success = downloader.download_pdf(pdf_url, output_filename)
    
    if success:
        print("🎉 Download completed successfully!")
    else:
        print("💥 Download failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()