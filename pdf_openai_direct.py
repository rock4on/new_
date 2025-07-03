#!/usr/bin/env python3
"""
PDF Direct OpenAI Processing

Sends PDF files directly to OpenAI without image conversion.
Works with OpenAI's latest models that support direct PDF input.
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import Optional, Dict, Any
import argparse
import tempfile
from urllib.parse import urlparse

try:
    import openai
    import requests
except ImportError as e:
    print(f"❌ Missing required dependency: {e}")
    print("💡 Install with: pip install openai requests")
    sys.exit(1)


class PDFOpenAIDirect:
    """Process PDFs directly with OpenAI without image conversion"""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o", 
                 base_url: Optional[str] = None):
        """
        Initialize the PDF OpenAI direct processor.
        
        Args:
            api_key: OpenAI API key (uses OPENAI_API_KEY env var if None)
            model: OpenAI model to use (gpt-4o recommended for PDF support)
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
        
        # Check if model supports PDFs
        if not base_url and model not in ["gpt-4o", "gpt-4o-mini"]:
            print(f"⚠️  Warning: Model {model} may not support direct PDF input")
            print("💡 Recommended models: gpt-4o, gpt-4o-mini")
    
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
    
    def extract_text_with_openai(self, pdf_path: Path, 
                                custom_prompt: Optional[str] = None,
                                max_tokens: int = 4000) -> str:
        """
        Extract text from PDF using OpenAI's direct PDF processing.
        
        Args:
            pdf_path: Path to PDF file
            custom_prompt: Custom extraction prompt
            max_tokens: Maximum tokens in response
            
        Returns:
            Extracted text
        """
        try:
            print(f"🤖 Processing PDF with OpenAI: {pdf_path}")
            
            # Check file size (OpenAI has limits)
            file_size = pdf_path.stat().st_size
            max_size_mb = 100  # Typical limit for file uploads
            if file_size > max_size_mb * 1024 * 1024:
                print(f"⚠️  Warning: PDF is {file_size / (1024*1024):.1f}MB, may exceed OpenAI limits")
            
            # Default prompt for OCR/text extraction
            if custom_prompt is None:
                prompt = """Please extract all text content from this PDF document. 

Instructions:
1. Extract ALL visible text accurately and completely
2. Maintain the original structure and formatting where possible
3. Include headings, paragraphs, bullet points, tables, etc.
4. For tables, format them clearly with proper alignment
5. If text is in multiple columns, read left to right, top to bottom
6. Preserve any important formatting like bold/italic text if evident
7. If some text is unclear, mark it as [UNCLEAR]
8. Do not add any commentary or descriptions, just the text content

Please provide the complete text extraction:"""
            else:
                prompt = custom_prompt
            
            # Upload file and process
            with open(pdf_path, 'rb') as f:
                # Use the files API for document processing
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {
                                    "type": "document",
                                    "document": {
                                        "data": f.read(),
                                        "detail": "high"
                                    }
                                }
                            ]
                        }
                    ],
                    max_tokens=max_tokens,
                    temperature=0  # Use deterministic output for text extraction
                )
            
            extracted_text = response.choices[0].message.content
            
            if extracted_text:
                print(f"✅ Text extraction successful ({len(extracted_text)} characters)")
                return extracted_text
            else:
                print("❌ No text extracted")
                return ""
            
        except Exception as e:
            print(f"❌ OpenAI text extraction failed: {e}")
            
            # If direct PDF processing fails, suggest alternatives
            if "document" in str(e).lower() or "unsupported" in str(e).lower():
                print("💡 This model may not support direct PDF input.")
                print("💡 Try using the image-based OCR version (pdf_openai_ocr.py)")
            
            return ""
    
    def process_pdf(self, pdf_path_or_url: str, output_path: Optional[Path] = None,
                   custom_prompt: Optional[str] = None, cleanup_temp: bool = True,
                   max_tokens: int = 4000) -> Path:
        """
        Process PDF and save results as JSON. Supports both file paths and URLs.
        
        Args:
            pdf_path_or_url: Path to PDF file or URL to PDF
            output_path: Output JSON file path (auto-generate if None)
            custom_prompt: Custom extraction prompt
            cleanup_temp: Whether to delete temporary downloaded files
            max_tokens: Maximum tokens in OpenAI response
            
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
            
            # Extract text using OpenAI
            extracted_text = self.extract_text_with_openai(
                pdf_path, 
                custom_prompt=custom_prompt,
                max_tokens=max_tokens
            )
            
            extraction_time = time.time() - start_time
            
            # Create JSON with metadata
            result = {
                'source': original_source,
                'source_type': 'url' if self.is_url(original_source) else 'file',
                'local_file': str(pdf_path),
                'extraction_method': 'openai_direct_pdf',
                'extraction_time_seconds': round(extraction_time, 2),
                'file_size_bytes': pdf_path.stat().st_size,
                'text_length': len(extracted_text),
                'word_count': len(extracted_text.split()),
                'openai_model': self.model,
                'openai_base_url': self.base_url,
                'custom_prompt_used': custom_prompt is not None,
                'extracted_text': extracted_text,
                'extracted_at': time.strftime('%Y-%m-%d %H:%M:%S')
            }
            
            # Save to JSON file
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            
            print(f"💾 Results saved to: {output_path}")
            return output_path
            
        finally:
            # Clean up temporary file if requested
            if temp_file and cleanup_temp:
                try:
                    temp_file.unlink()
                    print(f"🗑️  Cleaned up temporary file: {temp_file}")
                except Exception as e:
                    print(f"⚠️  Could not delete temporary file {temp_file}: {e}")


def main():
    """Command line interface"""
    parser = argparse.ArgumentParser(description='Process PDF directly with OpenAI (no image conversion needed)')
    parser.add_argument('pdf_source', help='Path to PDF file or URL to PDF')
    parser.add_argument('-o', '--output', help='Output JSON file path')
    parser.add_argument('--model', default='gpt-4o',
                       help='OpenAI model to use (default: gpt-4o)')
    parser.add_argument('--api-key', help='OpenAI API key (or set OPENAI_API_KEY env var)')
    parser.add_argument('--base-url', help='Custom OpenAI API endpoint URL')
    parser.add_argument('--custom-prompt', help='Custom prompt for text extraction')
    parser.add_argument('--max-tokens', type=int, default=4000,
                       help='Maximum tokens in OpenAI response (default: 4000)')
    parser.add_argument('--no-cleanup', action='store_true',
                       help='Keep temporary downloaded files (for URLs)')
    
    args = parser.parse_args()
    
    # Initialize processor
    processor = PDFOpenAIDirect(
        api_key=args.api_key, 
        model=args.model,
        base_url=args.base_url
    )
    
    # Check if it's a URL or file
    if processor.is_url(args.pdf_source):
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
        # Process PDF
        result_path = processor.process_pdf(
            pdf_path_or_url=args.pdf_source,
            output_path=output_path,
            custom_prompt=args.custom_prompt,
            cleanup_temp=not args.no_cleanup,
            max_tokens=args.max_tokens
        )
        
        print(f"🎉 Processing complete: {result_path}")
        
    except Exception as e:
        print(f"❌ Processing failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()