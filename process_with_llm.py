#!/usr/bin/env python3
"""
Simple LLM Processing Script

After Scrapy downloads PDFs and saves metadata, this script:
1. Reads the PDF metadata
2. Processes each PDF with LLM
3. Outputs comprehensive JSON analysis
"""

import json
import os
from pathlib import Path
from llm_filter import DocumentProcessor

# Configuration - Edit these settings
LLM_CRITERIA = "regulatory, compliance, and legal documents related to financial reporting and disclosure requirements"
DOWNLOADS_DIR = "downloads"
CONFIDENCE_THRESHOLD = 0.7
MODEL = "gpt-4o-mini"

# API Configuration 
API_KEY = os.getenv("OPENAI_API_KEY", "your-api-key-here")
BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")

def main():
    print("🚀 Starting LLM processing of downloaded PDFs...")
    
    downloads_path = Path(DOWNLOADS_DIR)
    metadata_file = downloads_path / "pdf_metadata.json"
    
    # Check if metadata exists
    if not metadata_file.exists():
        print(f"❌ No metadata file found at {metadata_file}")
        print("   Run Scrapy first to download PDFs and generate metadata")
        return
    
    # Load metadata
    with open(metadata_file, 'r', encoding='utf-8') as f:
        pdf_metadata = json.load(f)
    
    print(f"📁 Found metadata for {len(pdf_metadata)} PDFs")
    print(f"🎯 Criteria: {LLM_CRITERIA}")
    print(f"🤖 Model: {MODEL}")
    print(f"📊 Confidence threshold: {CONFIDENCE_THRESHOLD}")
    print("-" * 80)
    
    # Initialize LLM processor
    processor = DocumentProcessor(
        downloads_dir=DOWNLOADS_DIR,
        api_key=API_KEY,
        base_url=BASE_URL
    )
    
    # Process documents
    processor.process_documents(
        relevance_criteria=LLM_CRITERIA,
        model=MODEL,
        confidence_threshold=CONFIDENCE_THRESHOLD
    )
    
    print("✅ LLM processing complete!")
    print(f"📄 Results saved to: {downloads_path}/regulatory_analysis.json")
    print(f"📄 Relevant only: {downloads_path}/relevant_regulations.json")

if __name__ == "__main__":
    main()