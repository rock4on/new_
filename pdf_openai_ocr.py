#!/usr/bin/env python3
"""
PDF OCR using OpenAI Vision Models

Extracts text from PDF files using OpenAI's GPT-4 Vision API.
Converts PDF pages to images and sends them to OpenAI for text extraction.
"""

import os
import sys
import base64
import json
import time
from pathlib import Path
from typing import Optional, List, Dict, Any
import argparse
from io import BytesIO
import tempfile
from urllib.parse import urlparse

try:
    import openai
    import pdf2image
    from PIL import Image
    import PyPDF2
    import requests
except ImportError as e:
    print(f"❌ Missing required dependency: {e}")
    print("💡 Install with: pip install openai pdf2image pillow PyPDF2 requests")
    sys.exit(1)


class PDFOpenAIOCR:
    """Extract text from PDFs using OpenAI Vision API"""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o", 
                 base_url: Optional[str] = None):
        """
        Initialize the PDF OpenAI OCR extractor.
        
        Args:
            api_key: OpenAI API key (uses OPENAI_API_KEY env var if None)
            model: OpenAI model to use (gpt-4o, gpt-4-vision-preview)
            base_url: Custom API endpoint URL (uses OpenAI default if None)
        """
        # Set API key
        if api_key:
            openai.api_key = api_key
        elif not os.getenv("OPENAI_API_KEY"):
            print("❌ OpenAI API key not found!")
            print("💡 Set OPENAI_API_KEY environment variable or pass api_key parameter")
            print("💡 Get your API key from: https://platform.openai.com/api-keys")
            sys.exit(1)
        
        # Initialize client with custom base URL if provided
        client_kwargs = {}
        if api_key:
            client_kwargs['api_key'] = api_key
        if base_url:
            client_kwargs['base_url'] = base_url
            print(f"🔗 Using custom API endpoint: {base_url}")
        
        self.client = openai.OpenAI(**client_kwargs)
        self.model = model
        self.base_url = base_url
        
        # Check if model supports vision (skip warning for custom endpoints)
        if not base_url and model not in ["gpt-4o", "gpt-4-vision-preview", "gpt-4o-mini"]:
            print(f"⚠️  Warning: Model {model} may not support vision capabilities")
    
    def download_pdf_from_url(self, url: str, output_dir: Optional[Path] = None) -> Path:
        """
        Download PDF from URL to temporary file.
        
        Args:
            url: URL to PDF file
            output_dir: Directory to save PDF (uses temp dir if None)
            
        Returns:
            Path to downloaded PDF file
        """
        try:
            print(f"📥 Downloading PDF from: {url}")
            
            # Parse URL to get filename
            parsed_url = urlparse(url)
            filename = Path(parsed_url.path).name
            if not filename or not filename.endswith('.pdf'):
                filename = 'downloaded_document.pdf'
            
            # Set output directory
            if output_dir is None:
                output_dir = Path(tempfile.gettempdir())
            
            output_path = output_dir / filename
            
            # Download with progress
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            response = requests.get(url, headers=headers, stream=True)
            response.raise_for_status()
            
            # Check if it's actually a PDF
            content_type = response.headers.get('content-type', '').lower()
            if 'pdf' not in content_type:
                print(f"⚠️  Warning: Content-Type is '{content_type}', not PDF")
            
            # Save file
            total_size = int(response.headers.get('content-length', 0))
            with open(output_path, 'wb') as f:
                if total_size > 0:
                    downloaded = 0
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)
                            progress = (downloaded / total_size) * 100
                            print(f"\r  📊 Progress: {progress:.1f}% ({downloaded:,}/{total_size:,} bytes)", end='')
                    print()  # New line after progress
                else:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                    print(f"  📊 Downloaded: {output_path.stat().st_size:,} bytes")
            
            print(f"✅ PDF downloaded to: {output_path}")
            return output_path
            
        except requests.RequestException as e:
            print(f"❌ Failed to download PDF: {e}")
            raise
        except Exception as e:
            print(f"❌ Error downloading PDF: {e}")
            raise
    
    def is_url(self, path_or_url: str) -> bool:
        """
        Check if string is a URL.
        
        Args:
            path_or_url: String to check
            
        Returns:
            True if it's a URL, False otherwise
        """
        try:
            result = urlparse(path_or_url)
            return all([result.scheme, result.netloc])
        except:
            return False
    
    def extract_text_traditional(self, pdf_path: Path) -> str:
        """
        Extract text using traditional PDF text extraction.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Extracted text or empty string if failed
        """
        try:
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text_chunks = []
                
                for page in reader.pages:
                    try:
                        page_text = page.extract_text()
                        if page_text and page_text.strip():
                            text_chunks.append(page_text.strip())
                    except Exception:
                        continue
                
                return "\n".join(text_chunks) if text_chunks else ""
        except Exception as e:
            print(f"❌ Traditional extraction failed: {e}")
            return ""
    
    def image_to_base64(self, image: Image.Image, format: str = "PNG") -> str:
        """
        Convert PIL Image to base64 string.
        
        Args:
            image: PIL Image object
            format: Image format (PNG, JPEG)
            
        Returns:
            Base64 encoded image string
        """
        buffer = BytesIO()
        image.save(buffer, format=format)
        return base64.b64encode(buffer.getvalue()).decode()
    
    def extract_text_from_image(self, image: Image.Image, page_num: int = 1, 
                               custom_prompt: Optional[str] = None) -> str:
        """
        Extract text from a single image using OpenAI Vision API.
        
        Args:
            image: PIL Image object
            page_num: Page number for context
            custom_prompt: Custom extraction prompt
            
        Returns:
            Extracted text
        """
        try:
            # Convert image to base64
            image_base64 = self.image_to_base64(image)
            
            # Default prompt for OCR
            if custom_prompt is None:
                prompt = """Extract all text from this image. Please:
1. Transcribe ALL visible text accurately
2. Maintain the original structure and formatting as much as possible
3. Include headings, paragraphs, bullet points, tables, etc.
4. If there are tables, format them clearly
5. If text is in multiple columns, read left to right, top to bottom
6. Don't add any commentary or descriptions, just the text content
7. If you can't read some text clearly, use [UNCLEAR] to mark it"""
            else:
                prompt = custom_prompt
            
            # Make API call
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{image_base64}",
                                    "detail": "high"  # Use high detail for better OCR
                                }
                            }
                        ]
                    }
                ],
                max_tokens=4000,
                temperature=0  # Use deterministic output for OCR
            )
            
            extracted_text = response.choices[0].message.content
            return extracted_text if extracted_text else ""
            
        except Exception as e:
            print(f"❌ OpenAI OCR failed for page {page_num}: {e}")
            return ""
    
    def extract_text_openai_ocr(self, pdf_path: Path, dpi: int = 300, 
                               max_pages: Optional[int] = None,
                               custom_prompt: Optional[str] = None) -> str:
        """
        Extract text using OpenAI Vision API on PDF pages converted to images.
        
        Args:
            pdf_path: Path to PDF file
            dpi: Resolution for image conversion
            max_pages: Maximum number of pages to process (None for all)
            custom_prompt: Custom extraction prompt
            
        Returns:
            Extracted text
        """
        try:
            print(f"🔍 Converting PDF to images (DPI: {dpi})...")
            
            # Convert PDF to images
            images = pdf2image.convert_from_path(
                pdf_path,
                dpi=dpi,
                fmt='PNG'
            )
            
            # Limit pages if specified
            if max_pages:
                images = images[:max_pages]
            
            print(f"📄 Processing {len(images)} pages with OpenAI Vision...")
            
            text_chunks = []
            total_cost_estimate = 0
            
            for i, image in enumerate(images):
                try:
                    print(f"  📖 Processing page {i+1}/{len(images)} with OpenAI...")
                    
                    # Extract text from image
                    page_text = self.extract_text_from_image(
                        image, 
                        page_num=i+1,
                        custom_prompt=custom_prompt
                    )
                    
                    if page_text and page_text.strip():
                        text_chunks.append(f"=== Page {i+1} ===\n{page_text.strip()}")
                    
                    # Estimate cost (rough calculation)
                    # GPT-4o Vision: ~$0.01 per image at high detail
                    total_cost_estimate += 0.01
                    
                    # Add small delay to avoid rate limiting
                    time.sleep(0.5)
                    
                except Exception as e:
                    print(f"❌ OpenAI OCR failed for page {i+1}: {e}")
                    continue
            
            print(f"💰 Estimated cost: ${total_cost_estimate:.2f}")
            
            return "\n\n".join(text_chunks) if text_chunks else ""
            
        except Exception as e:
            print(f"❌ OpenAI OCR extraction failed: {e}")
            return ""
    
    def extract_text_hybrid(self, pdf_path: Path, dpi: int = 300, 
                           min_text_threshold: int = 100,
                           max_pages: Optional[int] = None,
                           custom_prompt: Optional[str] = None) -> str:
        """
        Try traditional extraction first, fall back to OpenAI OCR if insufficient text.
        
        Args:
            pdf_path: Path to PDF file
            dpi: Resolution for OpenAI OCR conversion
            min_text_threshold: Minimum characters to consider traditional extraction successful
            max_pages: Maximum pages for OpenAI OCR
            custom_prompt: Custom extraction prompt
            
        Returns:
            Extracted text
        """
        print(f"📋 Extracting text from: {pdf_path}")
        
        # First, try traditional extraction
        print("🔤 Trying traditional text extraction...")
        traditional_text = self.extract_text_traditional(pdf_path)
        
        if len(traditional_text) >= min_text_threshold:
            print(f"✅ Traditional extraction successful ({len(traditional_text)} chars)")
            return traditional_text
        else:
            print(f"⚠️  Traditional extraction insufficient ({len(traditional_text)} chars)")
            print("🤖 Using OpenAI Vision API for OCR...")
            
            openai_text = self.extract_text_openai_ocr(
                pdf_path, 
                dpi=dpi,
                max_pages=max_pages,
                custom_prompt=custom_prompt
            )
            
            if openai_text:
                print(f"✅ OpenAI OCR successful ({len(openai_text)} chars)")
                return openai_text
            else:
                print("❌ OpenAI OCR failed")
                return traditional_text  # Return whatever we got
    
    def extract_to_json(self, pdf_path_or_url: str, output_path: Optional[Path] = None,
                       method: str = 'hybrid', cleanup_temp: bool = True, **kwargs) -> Path:
        """
        Extract text and save as JSON with metadata. Supports both file paths and URLs.
        
        Args:
            pdf_path_or_url: Path to PDF file or URL to PDF
            output_path: Output JSON file path (auto-generate if None)
            method: 'traditional', 'openai', or 'hybrid'
            cleanup_temp: Whether to delete temporary downloaded files
            **kwargs: Additional arguments for extraction methods
            
        Returns:
            Path to output JSON file
        """
        temp_file = None
        original_source = pdf_path_or_url
        
        try:
            # Check if it's a URL and download if needed
            if self.is_url(pdf_path_or_url):
                temp_file = self.download_pdf_from_url(pdf_path_or_url)
                pdf_path = temp_file
            else:
                pdf_path = Path(pdf_path_or_url)
                if not pdf_path.exists():
                    raise FileNotFoundError(f"PDF file not found: {pdf_path}")
            
            if output_path is None:
                if self.is_url(pdf_path_or_url):
                    # Generate output name from URL
                    parsed_url = urlparse(pdf_path_or_url)
                    filename = Path(parsed_url.path).name
                    if not filename or not filename.endswith('.pdf'):
                        filename = 'downloaded_document.pdf'
                    output_path = Path(filename).with_suffix('.json')
                else:
                    output_path = pdf_path.with_suffix('.json')
            
            start_time = time.time()
            
            # Extract text based on method
            if method == 'traditional':
                text = self.extract_text_traditional(pdf_path)
            elif method == 'openai':
                text = self.extract_text_openai_ocr(pdf_path, **kwargs)
            elif method == 'hybrid':
                text = self.extract_text_hybrid(pdf_path, **kwargs)
            else:
                raise ValueError(f"Unknown method: {method}")
            
            extraction_time = time.time() - start_time
            
            # Create JSON with metadata
            result = {
                'source': original_source,
                'source_type': 'url' if self.is_url(original_source) else 'file',
                'local_file': str(pdf_path),
                'extraction_method': method,
                'extraction_time_seconds': round(extraction_time, 2),
                'file_size_bytes': pdf_path.stat().st_size,
                'text_length': len(text),
                'word_count': len(text.split()),
                'openai_model': self.model if method in ['openai', 'hybrid'] else None,
                'openai_base_url': self.base_url if method in ['openai', 'hybrid'] else None,
                'extracted_text': text,
                'extracted_at': time.strftime('%Y-%m-%d %H:%M:%S')
            }
            
            # Save to JSON file
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            
            print(f"💾 JSON saved to: {output_path}")
            return output_path
            
        finally:
            # Clean up temporary file if requested
            if temp_file and cleanup_temp:
                try:
                    temp_file.unlink()
                    print(f"🗑️  Cleaned up temporary file: {temp_file}")
                except Exception as e:
                    print(f"⚠️  Could not delete temporary file {temp_file}: {e}")
    
    def extract_text_from_url(self, url: str, method: str = 'hybrid', **kwargs) -> str:
        """
        Extract text directly from PDF URL.
        
        Args:
            url: URL to PDF file
            method: 'traditional', 'openai', or 'hybrid'
            **kwargs: Additional arguments for extraction methods
            
        Returns:
            Extracted text
        """
        temp_file = None
        try:
            temp_file = self.download_pdf_from_url(url)
            
            if method == 'traditional':
                return self.extract_text_traditional(temp_file)
            elif method == 'openai':
                return self.extract_text_openai_ocr(temp_file, **kwargs)
            elif method == 'hybrid':
                return self.extract_text_hybrid(temp_file, **kwargs)
            else:
                raise ValueError(f"Unknown method: {method}")
                
        finally:
            if temp_file:
                try:
                    temp_file.unlink()
                except:
                    pass


def main():
    """Command line interface"""
    parser = argparse.ArgumentParser(description='Extract text from PDF using OpenAI Vision API (supports files and URLs)')
    parser.add_argument('pdf_source', help='Path to PDF file or URL to PDF')
    parser.add_argument('-o', '--output', help='Output file path')
    parser.add_argument('-m', '--method', choices=['traditional', 'openai', 'hybrid'], 
                       default='hybrid', help='Extraction method (default: hybrid)')
    parser.add_argument('-d', '--dpi', type=int, default=300, 
                       help='DPI for image conversion (default: 300)')
    parser.add_argument('--max-pages', type=int, 
                       help='Maximum pages to process with OpenAI (to control costs)')
    parser.add_argument('--model', default='gpt-4o',
                       help='OpenAI model to use (default: gpt-4o)')
    parser.add_argument('--api-key', help='OpenAI API key (or set OPENAI_API_KEY env var)')
    parser.add_argument('--base-url', help='Custom OpenAI API endpoint URL')
    parser.add_argument('--custom-prompt', help='Custom prompt for OpenAI OCR')
    parser.add_argument('--no-cleanup', action='store_true',
                       help='Keep temporary downloaded files (for URLs)')
    parser.add_argument('-j', '--json', action='store_true', 
                       help='Save as JSON with metadata')
    
    args = parser.parse_args()
    
    # Initialize extractor
    extractor = PDFOpenAIOCR(
        api_key=args.api_key, 
        model=args.model,
        base_url=args.base_url
    )
    
    # Check if it's a URL or file
    if extractor.is_url(args.pdf_source):
        print(f"🌐 Processing PDF from URL: {args.pdf_source}")
    else:
        # Check if local file exists
        pdf_path = Path(args.pdf_source)
        if not pdf_path.exists():
            print(f"❌ PDF file not found: {pdf_path}")
            sys.exit(1)
        print(f"📄 Processing local PDF file: {pdf_path}")
    
    # Set output path
    output_path = Path(args.output) if args.output else None
    
    try:
        # Extract text
        result_path = extractor.extract_to_json(
            pdf_path_or_url=args.pdf_source,
            output_path=output_path,
            method=args.method,
            cleanup_temp=not args.no_cleanup,
            dpi=args.dpi,
            max_pages=args.max_pages,
            custom_prompt=args.custom_prompt
        )
        
        print(f"🎉 Extraction complete: {result_path}")
        
    except Exception as e:
        print(f"❌ Extraction failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()