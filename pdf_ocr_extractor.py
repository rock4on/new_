#!/usr/bin/env python3
"""
PDF OCR Text Extractor

Extracts text from PDF files using OCR (Optical Character Recognition).
Handles both text-based PDFs and scanned/image-based PDFs.
"""

import os
import sys
from pathlib import Path
from typing import Optional, List
import argparse
import json
import time

try:
    import pytesseract
    from PIL import Image
    import pdf2image
    import PyPDF2
except ImportError as e:
    print(f"❌ Missing required dependency: {e}")
    print("💡 Install with: pip install pytesseract pillow pdf2image PyPDF2")
    print("💡 Also install Tesseract OCR: sudo apt-get install tesseract-ocr (Ubuntu/Debian)")
    print("💡 Or: brew install tesseract (macOS)")
    sys.exit(1)


class PDFOCRExtractor:
    """Extract text from PDFs using OCR when needed"""
    
    def __init__(self, tesseract_cmd: Optional[str] = None):
        """
        Initialize the PDF OCR extractor.
        
        Args:
            tesseract_cmd: Path to tesseract executable (auto-detect if None)
        """
        # Set tesseract command path if provided
        if tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
        
        # Test if tesseract is available
        try:
            pytesseract.get_tesseract_version()
        except Exception as e:
            print(f"❌ Tesseract not found: {e}")
            print("💡 Install Tesseract OCR:")
            print("   Ubuntu/Debian: sudo apt-get install tesseract-ocr")
            print("   macOS: brew install tesseract")
            print("   Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki")
            sys.exit(1)
    
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
    
    def extract_text_ocr(self, pdf_path: Path, dpi: int = 300, lang: str = 'eng') -> str:
        """
        Extract text using OCR on PDF pages converted to images.
        
        Args:
            pdf_path: Path to PDF file
            dpi: Resolution for image conversion (higher = better quality, slower)
            lang: Language for OCR (eng, fra, deu, etc.)
            
        Returns:
            Extracted text
        """
        try:
            print(f"🔍 Converting PDF to images (DPI: {dpi})...")
            
            # Convert PDF to images
            images = pdf2image.convert_from_path(
                pdf_path,
                dpi=dpi,
                fmt='PNG',
                thread_count=4  # Use multiple threads for faster conversion
            )
            
            print(f"📄 Processing {len(images)} pages with OCR...")
            
            text_chunks = []
            for i, image in enumerate(images):
                try:
                    print(f"  📖 Processing page {i+1}/{len(images)}...")
                    
                    # Perform OCR on the image
                    page_text = pytesseract.image_to_string(
                        image,
                        lang=lang,
                        config='--psm 1 --oem 3'  # Page segmentation mode 1, OCR engine mode 3
                    )
                    
                    if page_text and page_text.strip():
                        text_chunks.append(f"=== Page {i+1} ===\n{page_text.strip()}")
                    
                except Exception as e:
                    print(f"❌ OCR failed for page {i+1}: {e}")
                    continue
            
            return "\n\n".join(text_chunks) if text_chunks else ""
            
        except Exception as e:
            print(f"❌ OCR extraction failed: {e}")
            return ""
    
    def extract_text_hybrid(self, pdf_path: Path, dpi: int = 300, lang: str = 'eng', 
                           min_text_threshold: int = 100) -> str:
        """
        Try traditional extraction first, fall back to OCR if insufficient text.
        
        Args:
            pdf_path: Path to PDF file
            dpi: Resolution for OCR conversion
            lang: Language for OCR
            min_text_threshold: Minimum characters to consider traditional extraction successful
            
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
            print("🔍 Falling back to OCR...")
            
            ocr_text = self.extract_text_ocr(pdf_path, dpi=dpi, lang=lang)
            
            if ocr_text:
                print(f"✅ OCR extraction successful ({len(ocr_text)} chars)")
                return ocr_text
            else:
                print("❌ OCR extraction failed")
                return traditional_text  # Return whatever we got
    
    def extract_to_file(self, pdf_path: Path, output_path: Optional[Path] = None,
                       method: str = 'hybrid', **kwargs) -> Path:
        """
        Extract text and save to file.
        
        Args:
            pdf_path: Path to PDF file
            output_path: Output file path (auto-generate if None)
            method: 'traditional', 'ocr', or 'hybrid'
            **kwargs: Additional arguments for extraction methods
            
        Returns:
            Path to output file
        """
        if output_path is None:
            output_path = pdf_path.with_suffix('.txt')
        
        # Extract text based on method
        if method == 'traditional':
            text = self.extract_text_traditional(pdf_path)
        elif method == 'ocr':
            text = self.extract_text_ocr(pdf_path, **kwargs)
        elif method == 'hybrid':
            text = self.extract_text_hybrid(pdf_path, **kwargs)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Save to file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(text)
        
        print(f"💾 Text saved to: {output_path}")
        return output_path
    
    def extract_to_json(self, pdf_path: Path, output_path: Optional[Path] = None,
                       method: str = 'hybrid', **kwargs) -> Path:
        """
        Extract text and save as JSON with metadata.
        
        Args:
            pdf_path: Path to PDF file
            output_path: Output JSON file path (auto-generate if None)
            method: 'traditional', 'ocr', or 'hybrid'
            **kwargs: Additional arguments for extraction methods
            
        Returns:
            Path to output JSON file
        """
        if output_path is None:
            output_path = pdf_path.with_suffix('.json')
        
        start_time = time.time()
        
        # Extract text based on method
        if method == 'traditional':
            text = self.extract_text_traditional(pdf_path)
        elif method == 'ocr':
            text = self.extract_text_ocr(pdf_path, **kwargs)
        elif method == 'hybrid':
            text = self.extract_text_hybrid(pdf_path, **kwargs)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        extraction_time = time.time() - start_time
        
        # Create JSON with metadata
        result = {
            'source_file': str(pdf_path),
            'extraction_method': method,
            'extraction_time_seconds': round(extraction_time, 2),
            'file_size_bytes': pdf_path.stat().st_size,
            'text_length': len(text),
            'word_count': len(text.split()),
            'extracted_text': text,
            'extracted_at': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Save to JSON file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        print(f"💾 JSON saved to: {output_path}")
        return output_path


def main():
    """Command line interface"""
    parser = argparse.ArgumentParser(description='Extract text from PDF using OCR')
    parser.add_argument('pdf_file', help='Path to PDF file')
    parser.add_argument('-o', '--output', help='Output file path')
    parser.add_argument('-m', '--method', choices=['traditional', 'ocr', 'hybrid'], 
                       default='hybrid', help='Extraction method (default: hybrid)')
    parser.add_argument('-d', '--dpi', type=int, default=300, 
                       help='DPI for OCR conversion (default: 300)')
    parser.add_argument('-l', '--lang', default='eng', 
                       help='Language for OCR (default: eng)')
    parser.add_argument('-j', '--json', action='store_true', 
                       help='Save as JSON with metadata')
    parser.add_argument('-t', '--tesseract', help='Path to tesseract executable')
    
    args = parser.parse_args()
    
    # Check if PDF file exists
    pdf_path = Path(args.pdf_file)
    if not pdf_path.exists():
        print(f"❌ PDF file not found: {pdf_path}")
        sys.exit(1)
    
    # Initialize extractor
    extractor = PDFOCRExtractor(tesseract_cmd=args.tesseract)
    
    # Set output path
    output_path = Path(args.output) if args.output else None
    
    try:
        if args.json:
            # Extract to JSON
            result_path = extractor.extract_to_json(
                pdf_path=pdf_path,
                output_path=output_path,
                method=args.method,
                dpi=args.dpi,
                lang=args.lang
            )
        else:
            # Extract to text file
            result_path = extractor.extract_to_file(
                pdf_path=pdf_path,
                output_path=output_path,
                method=args.method,
                dpi=args.dpi,
                lang=args.lang
            )
        
        print(f"🎉 Extraction complete: {result_path}")
        
    except Exception as e:
        print(f"❌ Extraction failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()