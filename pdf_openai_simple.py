#!/usr/bin/env python3
"""
Simple PDF to OpenAI Processing

Sends PDF content directly to OpenAI for text extraction.
"""

import os
import sys
import base64
import json
import time
from pathlib import Path
from typing import Optional
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


class SimpleOpenAIPDF:
    """Simple PDF processing with OpenAI"""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o"):
        """Initialize with API key and model"""
        if not api_key:
            api_key = os.getenv("OPENAI_API_KEY")
        
        if not api_key:
            print("❌ OpenAI API key required!")
            print("💡 Set OPENAI_API_KEY environment variable or pass --api-key")
            sys.exit(1)
        
        self.client = openai.OpenAI(api_key=api_key)
        self.model = model
    
    def pdf_to_base64(self, pdf_path: Path) -> str:
        """Convert PDF to base64 string"""
        with open(pdf_path, 'rb') as f:
            return base64.b64encode(f.read()).decode('utf-8')
    
    def download_pdf(self, url: str) -> Path:
        """Download PDF from URL"""
        print(f"📥 Downloading: {url}")
        
        response = requests.get(url)
        response.raise_for_status()
        
        # Save to temp file
        temp_file = Path(tempfile.mktemp(suffix='.pdf'))
        with open(temp_file, 'wb') as f:
            f.write(response.content)
        
        print(f"✅ Downloaded to: {temp_file}")
        return temp_file
    
    def extract_text(self, pdf_path_or_url: str, prompt: Optional[str] = None) -> str:
        """Extract text from PDF using OpenAI"""
        temp_file = None
        
        try:
            # Handle URL or file path
            if pdf_path_or_url.startswith(('http://', 'https://')):
                temp_file = self.download_pdf(pdf_path_or_url)
                pdf_path = temp_file
            else:
                pdf_path = Path(pdf_path_or_url)
                if not pdf_path.exists():
                    raise FileNotFoundError(f"PDF not found: {pdf_path}")
            
            print(f"🤖 Processing with OpenAI: {pdf_path}")
            
            # Convert to base64
            pdf_base64 = self.pdf_to_base64(pdf_path)
            
            # Default prompt
            if not prompt:
                prompt = """Extract all text from this PDF document. 
                
Please:
1. Extract ALL visible text accurately
2. Maintain structure and formatting
3. Include headings, paragraphs, tables, etc.
4. Don't add commentary, just the text content"""
            
            # Send to OpenAI
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
                                    "url": f"data:application/pdf;base64,{pdf_base64}",
                                    "detail": "high"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=4000,
                temperature=0
            )
            
            text = response.choices[0].message.content or ""
            print(f"✅ Extracted {len(text)} characters")
            return text
            
        except Exception as e:
            print(f"❌ Error: {e}")
            return ""
            
        finally:
            if temp_file and temp_file.exists():
                temp_file.unlink()
    
    def save_result(self, pdf_source: str, extracted_text: str, output_path: Optional[str] = None) -> str:
        """Save extraction result to JSON"""
        if not output_path:
            if pdf_source.startswith(('http://', 'https://')):
                output_path = "extracted_text.json"
            else:
                output_path = Path(pdf_source).with_suffix('.json')
        
        result = {
            'source': pdf_source,
            'extracted_text': extracted_text,
            'text_length': len(extracted_text),
            'word_count': len(extracted_text.split()),
            'model': self.model,
            'extracted_at': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Saved to: {output_path}")
        return str(output_path)


def main():
    """Command line interface"""
    parser = argparse.ArgumentParser(description='Extract text from PDF using OpenAI')
    parser.add_argument('pdf_source', help='PDF file path or URL')
    parser.add_argument('-o', '--output', help='Output JSON file')
    parser.add_argument('--api-key', help='OpenAI API key')
    parser.add_argument('--model', default='gpt-4o', help='OpenAI model')
    parser.add_argument('--prompt', help='Custom extraction prompt')
    
    args = parser.parse_args()
    
    # Process PDF
    processor = SimpleOpenAIPDF(api_key=args.api_key, model=args.model)
    extracted_text = processor.extract_text(args.pdf_source, args.prompt)
    
    if extracted_text:
        processor.save_result(args.pdf_source, extracted_text, args.output)
        print("🎉 Done!")
    else:
        print("❌ No text extracted")
        sys.exit(1)


if __name__ == "__main__":
    main()