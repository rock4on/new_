#!/usr/bin/env python3
"""
Lease Document Processor

Processes PDF lease documents using OCR and AI to extract structured lease information.
Builds upon the existing PDF OCR extractor to provide lease-specific data extraction.

Default behavior:
- Processes all PDF files in 'leases' folder
- Outputs results to 'lease_results.csv'
- Uses OCR for text extraction and AI for information extraction
"""

import os
import sys
import re
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import csv

# Import the existing OCR extractor
from pdf_ocr_extractor import PDFOCRExtractor


class LeaseInformation(BaseModel):
    """Pydantic model for structured lease information extraction"""
    
    description: Optional[str] = Field(
        None, 
        description="Description of supporting documentation for lease details (e.g. signed lease agreement)"
    )
    location: Optional[str] = Field(
        None, 
        description="Location per underlying support"
    )
    lease_start_date: Optional[str] = Field(
        None, 
        description="Lease start date in YYYY-MM-DD format if possible"
    )
    lease_end_date: Optional[str] = Field(
        None, 
        description="Lease end date in YYYY-MM-DD format if possible"
    )
    building_area: Optional[str] = Field(
        None, 
        description="Building area per underlying support (numeric value)"
    )
    area_unit: Optional[str] = Field(
        None, 
        description="Unit of measure of building area (e.g., sq ft, sq m, sqft)"
    )
    building_type: Optional[str] = Field(
        None, 
        description="Building Type (i.e., Office vs. Warehouse)"
    )

try:
    import openai
    from pydantic import BaseModel, Field
except ImportError as e:
    print(f"❌ Missing required dependency: {e}")
    print("💡 Install with: pip install openai pydantic")
    sys.exit(1)


class LeaseDocumentProcessor:
    """Extract structured lease information from PDF documents"""
    
    def __init__(self, openai_api_key: Optional[str] = None, tesseract_cmd: Optional[str] = None):
        """
        Initialize the lease document processor.
        
        Args:
            openai_api_key: OpenAI API key (or from environment)
            tesseract_cmd: Path to tesseract executable
        """
        # Initialize OCR extractor
        self.ocr_extractor = PDFOCRExtractor(tesseract_cmd=tesseract_cmd)
        
        # Initialize OpenAI client
        self.client = openai.OpenAI(
            api_key=openai_api_key or os.getenv('OPENAI_API_KEY')
        )
        
        # Check if API key is available
        if not (openai_api_key or os.getenv('OPENAI_API_KEY')):
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY environment variable or pass openai_api_key parameter.")
        
        # Define the lease information fields we want to extract
        self.lease_fields = {
            'description': 'Description of supporting documentation for lease details (e.g. signed lease agreement)',
            'location': 'Location per underlying support',
            'lease_start_date': 'Lease start date',
            'lease_end_date': 'Lease end date',
            'building_area': 'Building area per underlying support',
            'area_unit': 'Unit of measure of building area',
            'building_type': 'Building Type (i.e., Office vs. Warehouse)'
        }
    
    def extract_lease_info_ai(self, text: str) -> Dict[str, Any]:
        """
        Use AI with structured outputs to extract lease information from text.
        
        Args:
            text: Raw text from PDF
            
        Returns:
            Dictionary with extracted lease information
        """
        prompt = f"""
        Extract lease information from the following document text. Identify and extract all available fields.
        
        For dates, use YYYY-MM-DD format if possible.
        For building area, extract the numeric value only.
        For area unit, use standard units like "sq ft", "sq m", "sqft", etc.
        
        Document text:
        {text}
        """
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a document analyst specializing in lease agreement analysis. Extract information accurately from the provided text."},
                    {"role": "user", "content": prompt}
                ],
                response_format=LeaseInformation,
                temperature=0.1,
                max_tokens=1000
            )
            
            # Parse the structured response
            lease_info = response.choices[0].message.parsed
            
            # Convert Pydantic model to dictionary
            return lease_info.model_dump()
            
        except Exception as e:
            print(f"❌ AI extraction failed: {e}")
            return {field: None for field in self.lease_fields.keys()}
    
    
    def process_pdf(self, pdf_path: Path, ocr_method: str = 'ocr') -> Dict[str, Any]:
        """
        Process a PDF lease document and extract structured information.
        
        Args:
            pdf_path: Path to PDF file
            ocr_method: OCR method to use ('traditional' or 'ocr')
            
        Returns:
            Dictionary with extracted lease information
        """
        print(f"📄 Processing lease document: {pdf_path}")
        
        # Extract text using OCR
        if ocr_method == 'traditional':
            text = self.ocr_extractor.extract_text_traditional(pdf_path)
        elif ocr_method == 'ocr':
            text = self.ocr_extractor.extract_text_ocr(pdf_path)
        else:  # default to ocr
            text = self.ocr_extractor.extract_text_ocr(pdf_path)
        
        if not text.strip():
            print("❌ No text extracted from PDF")
            return {field: None for field in self.lease_fields.keys()}
        
        print(f"📝 Extracted {len(text)} characters of text")
        
        # Extract lease information using AI
        print("🤖 Using AI for lease information extraction...")
        lease_info = self.extract_lease_info_ai(text)
        
        # Add metadata
        lease_info['_metadata'] = {
            'source_file': str(pdf_path),
            'processing_method': 'ai',
            'ocr_method': ocr_method,
            'text_length': len(text),
            'processed_at': datetime.now().isoformat()
        }
        
        return lease_info
    
    def process_directory(self, directory_path: Path, output_path: Optional[Path] = None, 
                         ocr_method: str = 'ocr') -> List[Dict[str, Any]]:
        """
        Process all PDF files in a directory.
        
        Args:
            directory_path: Path to directory containing PDF files
            output_path: Path to save results (optional)
            ocr_method: OCR method to use ('traditional' or 'ocr')
            
        Returns:
            List of extracted lease information dictionaries
        """
        pdf_files = list(directory_path.glob('*.pdf'))
        
        if not pdf_files:
            print(f"❌ No PDF files found in {directory_path}")
            return []
        
        print(f"📁 Found {len(pdf_files)} PDF files to process")
        
        results = []
        for i, pdf_path in enumerate(pdf_files, 1):
            print(f"\n🔄 Processing file {i}/{len(pdf_files)}: {pdf_path.name}")
            
            try:
                lease_info = self.process_pdf(pdf_path, ocr_method=ocr_method)
                lease_info['_metadata']['file_index'] = i
                results.append(lease_info)
                
            except Exception as e:
                print(f"❌ Failed to process {pdf_path.name}: {e}")
                # Add error record
                error_record = {field: None for field in self.lease_fields.keys()}
                error_record['_metadata'] = {
                    'source_file': str(pdf_path),
                    'error': str(e),
                    'file_index': i,
                    'processed_at': datetime.now().isoformat()
                }
                results.append(error_record)
        
        # Save results if output path provided
        if output_path:
            self.save_results(results, output_path)
        
        return results
    
    def save_results(self, results: List[Dict[str, Any]], output_path: Path):
        """
        Save extraction results to file.
        
        Args:
            results: List of lease information dictionaries
            output_path: Path to save results
        """
        if output_path.suffix.lower() == '.json':
            # Save as JSON
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"💾 Results saved to JSON: {output_path}")
            
        elif output_path.suffix.lower() == '.csv':
            # Save as CSV
            if not results:
                print("❌ No results to save")
                return
            
            # Get all field names (excluding metadata)
            fieldnames = [field for field in self.lease_fields.keys()]
            fieldnames.extend(['source_file', 'processing_method', 'ocr_method', 'processed_at'])
            
            with open(output_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                
                for result in results:
                    # Flatten the result
                    row = {field: result.get(field) for field in self.lease_fields.keys()}
                    if '_metadata' in result:
                        row.update({
                            'source_file': result['_metadata'].get('source_file'),
                            'processing_method': result['_metadata'].get('processing_method'),
                            'ocr_method': result['_metadata'].get('ocr_method'),
                            'processed_at': result['_metadata'].get('processed_at')
                        })
                    writer.writerow(row)
            
            print(f"💾 Results saved to CSV: {output_path}")
        
        else:
            # Save as text
            with open(output_path, 'w', encoding='utf-8') as f:
                for i, result in enumerate(results, 1):
                    f.write(f"=== Document {i} ===\n")
                    f.write(f"Source: {result.get('_metadata', {}).get('source_file', 'Unknown')}\n")
                    f.write(f"Processing method: {result.get('_metadata', {}).get('processing_method', 'Unknown')}\n")
                    f.write(f"OCR method: {result.get('_metadata', {}).get('ocr_method', 'Unknown')}\n")
                    f.write(f"Processed at: {result.get('_metadata', {}).get('processed_at', 'Unknown')}\n\n")
                    
                    for field, description in self.lease_fields.items():
                        value = result.get(field, 'Not found')
                        f.write(f"{description}: {value}\n")
                    
                    f.write("\n" + "="*80 + "\n\n")
            
            print(f"💾 Results saved to text: {output_path}")


def main():
    """Command line interface"""
    parser = argparse.ArgumentParser(description='Extract structured lease information from PDF documents')
    parser.add_argument('input', nargs='?', default='leases', help='Path to PDF file or directory containing PDF files (default: leases)')
    parser.add_argument('-o', '--output', default='lease_results.csv', help='Output file path (JSON, CSV, or TXT) (default: lease_results.csv)')
    parser.add_argument('-m', '--ocr-method', choices=['traditional', 'ocr'], 
                       default='ocr', help='OCR method to use (default: ocr)')
    parser.add_argument('--openai-key', help='OpenAI API key (or set OPENAI_API_KEY env var)')
    parser.add_argument('-t', '--tesseract', help='Path to tesseract executable')
    
    args = parser.parse_args()
    
    # Check input path
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"❌ Input path not found: {input_path}")
        if args.input == 'leases':
            print("💡 Create a 'leases' folder and place PDF files there, or specify a different path.")
        sys.exit(1)
    
    # Initialize processor
    try:
        processor = LeaseDocumentProcessor(
            openai_api_key=args.openai_key,
            tesseract_cmd=args.tesseract
        )
    except Exception as e:
        print(f"❌ Failed to initialize processor: {e}")
        sys.exit(1)
    
    # Set output path
    output_path = Path(args.output)
    
    try:
        if input_path.is_file():
            # Process single file
            result = processor.process_pdf(
                input_path,
                ocr_method=args.ocr_method
            )
            
            # Save results to file
            processor.save_results([result], output_path)
            
            # Also print results to console
            print("\n🎉 Extraction Results:")
            print("=" * 50)
            for field, description in processor.lease_fields.items():
                value = result.get(field, 'Not found')
                print(f"{description}: {value}")
        
        else:
            # Process directory
            results = processor.process_directory(
                input_path,
                output_path=output_path,
                ocr_method=args.ocr_method
            )
            
            # Print summary to console
            print(f"\n🎉 Processed {len(results)} documents")
            print(f"📄 Results saved to: {output_path}")
        
        print(f"\n✅ Processing complete!")
        print(f"📄 Results saved to: {output_path}")
        
    except Exception as e:
        print(f"❌ Processing failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()