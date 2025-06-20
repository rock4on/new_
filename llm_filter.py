#!/usr/bin/env python3
"""
LLM Document Filter Service

Scans downloaded PDFs, extracts text, and uses LLM to determine relevance
based on user-provided criteria. Keeps only relevant documents.
"""

import os
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any
import PyPDF2
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime


class DocumentAnalysis(BaseModel):
    """Pydantic model for comprehensive document analysis and extraction"""
    relevant: bool = Field(description="Whether the document is relevant")
    confidence: float = Field(description="Confidence score between 0.0 and 1.0")
    
    # Core regulation information
    unique_id: str = Field(description="Unique identifier for the regulation")
    country: str = Field(description="Country where regulation applies")
    jurisdiction: str = Field(description="Specific jurisdiction (state, province, etc.)")
    issuing_body: str = Field(description="Authority or body that issued the regulation")
    tag: str = Field(description="Disclosure area/category (e.g., ESG, Financial, Tax)")
    regulation_name: str = Field(description="Official name of the regulation")
    publication_date: str = Field(description="Date of publication (YYYY-MM-DD format)")
    regulation_status: str = Field(description="Status (Draft, Final, Effective, Superseded)")
    
    # Content and applicability
    summary: str = Field(description="Summary of regulation and disclosure requirements")
    applicability: str = Field(description="Scope summary - who/what it applies to")
    scoping_threshold: str = Field(description="Minimum thresholds for applicability")
    effective_date: str = Field(description="Earliest mandatory reporting effective date (YYYY-MM-DD)")
    timeline_details: str = Field(description="Implementation timeline and key dates")
    
    # Reporting requirements
    financial_integration: str = Field(description="Integration with financial reporting (Yes/No/Partial)")
    filing_mechanism: str = Field(description="How/where to file reports")
    reporting_frequency: str = Field(description="Required reporting frequency")
    assurance_requirement: str = Field(description="Assurance/audit requirements")
    penalties: str = Field(description="Non-compliance penalties")
    
    # Metadata
    full_text_link: str = Field(description="Link to the full regulation text")
    translated_flag: bool = Field(description="Whether document was translated to English")
    source_url: str = Field(description="URL where document was scraped from")
    last_scraped: str = Field(description="Date when document was scraped (YYYY-MM-DD)")
    change_detected: bool = Field(description="Whether changes were detected from previous version")


# Configuration - Set your preferences here
DEFAULT_API_KEY = "your-api-key-here"
DEFAULT_BASE_URL = "https://api.openai.com/v1"  # Change to your custom endpoint
DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_CONFIDENCE_THRESHOLD = 0.7
DEFAULT_DOWNLOADS_DIR = "downloads"
RELEVANCE_CRITERIA = "documents related to financial regulations, compliance, and legal requirements"  # Change this to what you're looking for


class DocumentProcessor:
    def __init__(self, downloads_dir: str = "downloads", api_key: str = None, base_url: str = None):
        self.downloads_dir = Path(downloads_dir)
        
        # Setup LangChain ChatOpenAI with fallback priority:
        # 1. Command line arguments
        # 2. Environment variables  
        # 3. Default constants above
        self.llm = ChatOpenAI(
            api_key=api_key or os.getenv("OPENAI_API_KEY") or DEFAULT_API_KEY,
            base_url=base_url or os.getenv("OPENAI_BASE_URL") or DEFAULT_BASE_URL,
            temperature=0.1
        )
        
        # Setup output parser
        self.parser = JsonOutputParser(pydantic_object=DocumentAnalysis)
        
        # Create prompt template
        self.prompt = ChatPromptTemplate.from_template("""
You are an expert regulatory document analyzer. Analyze the following document and extract comprehensive regulatory information.

ANALYSIS CRITERIA:
{criteria}

DOCUMENT TEXT:
{document_text}

SOURCE URL: {source_url}
DOCUMENT FILENAME: {filename}

INSTRUCTIONS:
1. First determine if this document is relevant to the criteria (regulatory/compliance/legal documents)
2. If relevant, extract ALL available regulatory information
3. If the document is not in English, translate all extracted information to English
4. Use "Unknown" or "Not specified" for fields that cannot be determined from the document
5. For dates, use YYYY-MM-DD format or "Unknown" if not available
6. Generate a unique ID using format: COUNTRY_AUTHORITY_YEAR_SHORTNAME
7. Set translated_flag to true if you translated any content
8. Set change_detected to false (this is for tracking document changes over time)
9. Use today's date for last_scraped field

{format_instructions}
""")
        
        # Create the chain
        self.chain = self.prompt | self.llm | self.parser
    
    def extract_pdf_text(self, pdf_path: Path, max_pages: int = 5) -> str:
        """Extract text from PDF - all pages if max_pages=None, otherwise first max_pages"""
        try:
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text = ""
                
                # Determine pages to extract
                if max_pages is None:
                    # Extract ALL pages
                    pages_to_extract = reader.pages
                else:
                    # Extract only first max_pages
                    pages_to_extract = reader.pages[:max_pages]
                
                # Extract text from selected pages
                for i, page in enumerate(pages_to_extract):
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
                
                return text.strip()
        except Exception as e:
            print(f"Error extracting text from {pdf_path}: {e}")
            return ""
    
    def extract_text_file(self, text_path: Path, max_chars: int = 8000) -> str:
        """Extract text from text file"""
        try:
            with open(text_path, 'r', encoding='utf-8') as file:
                text = file.read()
                
                # Truncate if too long
                if max_chars and len(text) > max_chars:
                    text = text[:max_chars] + "...[truncated]"
                
                return text.strip()
        except Exception as e:
            print(f"Error reading text from {text_path}: {e}")
            return ""
    
    def analyze_document(self, document_text: str, relevance_criteria: str, source_url: str, filename: str, model: str = "gpt-4o-mini") -> Dict[str, Any]:
        """Use LangChain LLM to comprehensively analyze regulatory document"""
        
        # Truncate text if too long (keep first 8000 chars for better extraction)
        if len(document_text) > 8000:
            document_text = document_text[:8000] + "...[truncated]"
        
        # Update model if different from default
        if model != "gpt-4o-mini":
            self.llm.model_name = model
        
        try:
            # Get current date for metadata
            current_date = datetime.now().strftime("%Y-%m-%d")
            
            # Invoke the chain with comprehensive parameters
            result = self.chain.invoke({
                "criteria": relevance_criteria,
                "document_text": document_text,
                "source_url": source_url,
                "filename": filename,
                "format_instructions": self.parser.get_format_instructions()
            })
            
            # Ensure metadata fields are populated
            if not result.get("last_scraped"):
                result["last_scraped"] = current_date
            if not result.get("source_url"):
                result["source_url"] = source_url
            if result.get("change_detected") is None:
                result["change_detected"] = False
            
            return result
            
        except Exception as e:
            print(f"Error analyzing document: {e}")
            current_date = datetime.now().strftime("%Y-%m-%d")
            return {
                "relevant": False,
                "confidence": 0.0,
                "unique_id": f"ERROR_{current_date}_{hash(filename) % 10000}",
                "country": "Unknown",
                "jurisdiction": "Unknown", 
                "issuing_body": "Unknown",
                "tag": "Error",
                "regulation_name": "Analysis Failed",
                "publication_date": "Unknown",
                "regulation_status": "Unknown",
                "summary": f"Error in analysis: {str(e)}",
                "applicability": "Unknown",
                "scoping_threshold": "Unknown",
                "effective_date": "Unknown",
                "timeline_details": "Unknown",
                "financial_integration": "Unknown",
                "filing_mechanism": "Unknown",
                "reporting_frequency": "Unknown",
                "assurance_requirement": "Unknown",
                "penalties": "Unknown",
                "full_text_link": source_url,
                "translated_flag": False,
                "source_url": source_url,
                "last_scraped": current_date,
                "change_detected": False
            }
    
    def load_document_metadata(self):
        """Load document metadata from individual JSON files in downloads directory."""
        metadata_mapping = {}
        
        # Look for individual metadata files (*.json) in main downloads directory
        json_files = list(self.downloads_dir.glob("*.json"))
        
        for json_file in json_files:
            # Skip the analysis output files
            if json_file.name in ["regulatory_analysis.json", "relevant_regulations.json", "pdf_metadata.json"]:
                continue
                
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                
                # Use the filename from metadata as key
                if "filename" in metadata:
                    metadata_mapping[metadata["filename"]] = metadata
                    
            except Exception as e:
                print(f"Warning: Could not load metadata from {json_file}: {e}")
        
        print(f"Loaded metadata for {len(metadata_mapping)} documents")
        return metadata_mapping

    def save_individual_json(self, analysis: Dict[str, Any], pdf_filename: str):
        """Save individual JSON file for each processed PDF"""
        # Create individual JSON files directory
        json_dir = self.downloads_dir / "individual_json"
        json_dir.mkdir(exist_ok=True)
        
        # Create filename for JSON (replace .pdf with .json)
        json_filename = pdf_filename.replace('.pdf', '.json')
        json_path = json_dir / json_filename
        
        # Save individual analysis
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(analysis, f, indent=2, ensure_ascii=False)
        
        print(f"Saved individual JSON: {json_path}")

    def process_single_document(self, doc_file: Path, document_metadata: dict, relevance_criteria: str, 
                               confidence_threshold: float, model: str) -> tuple:
        """Process a single document file (PDF or text) for parallel processing"""
        
        # Extract text based on file type
        if doc_file.suffix.lower() == '.pdf':
            text = self.extract_pdf_text(doc_file, max_pages=None)
        elif doc_file.suffix.lower() == '.txt':
            text = self.extract_text_file(doc_file)
        else:
            return None, False
        
        if not text:
            return None, False
        
        # Get metadata
        file_metadata = document_metadata.get(doc_file.name, {})
        
        # Handle both PDF and text metadata structures
        if doc_file.suffix.lower() == '.pdf':
            source_url = file_metadata.get("pdf_url", "Unknown")
        else:  # text file
            source_url = file_metadata.get("content_url", "Unknown")
        
        src_page = file_metadata.get("src_page", "Unknown")
        original_filename = file_metadata.get("original_filename", doc_file.name)
        title = file_metadata.get("title", "Unknown")
        scraped_at = file_metadata.get("scraped_at", "Unknown")
        file_size = doc_file.stat().st_size if doc_file.exists() else 0
        content_type = file_metadata.get("content_type", "pdf" if doc_file.suffix.lower() == '.pdf' else "text")
        
        # LLM analysis
        analysis = self.analyze_document(text, relevance_criteria, source_url, original_filename, model)
        
        # Add clean metadata (removed unwanted fields)
        analysis["file_size"] = file_size
        analysis["original_filename"] = original_filename
        analysis["title_from_link"] = title
        analysis["src_page"] = src_page
        analysis["scraped_at"] = scraped_at
        analysis["full_text_available"] = len(text) > 0
        analysis["text_length"] = len(text)
        analysis["filename"] = doc_file.name
        analysis["file_path"] = str(doc_file)
        analysis["content_type"] = content_type
        analysis["processed_at"] = datetime.now().isoformat()
        
        # Check if relevant
        is_relevant = (analysis["relevant"] and 
                      analysis["confidence"] >= confidence_threshold)
        
        return analysis, is_relevant

    def process_documents(self, relevance_criteria: str, model: str = "gpt-4o-mini", 
                         confidence_threshold: float = 0.7):
        """Process all PDFs and text files in parallel for speed"""
        
        # Get PDF files from downloads/full/ subdirectory
        pdf_dir = self.downloads_dir / "full"
        pdf_files = list(pdf_dir.glob("*.pdf")) if pdf_dir.exists() else []
        
        # Get text files from downloads/text/ subdirectory
        text_dir = self.downloads_dir / "text"
        text_files = list(text_dir.glob("*.txt")) if text_dir.exists() else []
        
        all_files = pdf_files + text_files
        
        if not all_files:
            print(f"No PDF or text files found in {pdf_dir} or {text_dir}")
            return
        
        # Load metadata mapping
        document_metadata = self.load_document_metadata()
        
        print(f"Processing {len(all_files)} documents ({len(pdf_files)} PDFs, {len(text_files)} text files) in parallel...")
        print(f"Analysis criteria: {relevance_criteria}")
        print(f"Confidence threshold: {confidence_threshold}")
        print("-" * 80)
        
        all_analyses = []
        relevant_analyses = []
        
        # Process all files in parallel using ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=4) as executor:
            # Submit all tasks
            future_to_file = {
                executor.submit(self.process_single_document, doc_file, document_metadata, 
                               relevance_criteria, confidence_threshold, model): doc_file 
                for doc_file in all_files
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_file):
                doc_file = future_to_file[future]
                try:
                    analysis, is_relevant = future.result()
                    if analysis is None:
                        print(f"Skipped: {doc_file.name} (no content)")
                        continue
                        
                    all_analyses.append(analysis)
                    
                    if is_relevant:
                        relevant_analyses.append(analysis)
                        content_type = "PDF" if doc_file.suffix == ".pdf" else "TEXT"
                        print(f"RELEVANT ({content_type}): {analysis.get('regulation_name', 'Unknown')} ({analysis['confidence']:.2f})")
                        # Save individual JSON for relevant files only
                        self.save_individual_json(analysis, analysis.get('original_filename', doc_file.name))
                    else:
                        content_type = "PDF" if doc_file.suffix == ".pdf" else "TEXT"
                        print(f"IRRELEVANT ({content_type}): {doc_file.name} ({analysis['confidence']:.2f})")
                        
                except Exception as e:
                    print(f"Error processing {doc_file.name}: {e}")
        
        # Save results
        relevant_count = len(relevant_analyses)
        total_count = len(all_analyses)
        
        # Save only relevant regulations (no complete analysis file)
        if relevant_analyses:
            relevant_file = self.downloads_dir / "relevant_regulations.json"
            output_data = {
                "metadata": {
                    "analysis_date": datetime.now().isoformat(),
                    "criteria": relevance_criteria,
                    "model": model,
                    "confidence_threshold": confidence_threshold,
                    "total_documents": total_count,
                    "relevant_documents": relevant_count,
                    "pdf_count": len(pdf_files),
                    "text_count": len(text_files),
                    "processing_mode": "PARALLEL_RELEVANT_ONLY"
                },
                "regulations": relevant_analyses
            }
            
            with open(relevant_file, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"Processing complete!")
        print(f"Results: {relevant_count}/{total_count} documents marked as relevant")
        if relevant_analyses:
            print(f"Relevant regulations saved to: {self.downloads_dir}/relevant_regulations.json")
            print(f"Individual JSON files: {self.downloads_dir}/individual_json/")
        else:
            print("No relevant documents found.")


def main():
    print("🔍 Starting Document Filter (PDFs and Text)...")
    print(f"📁 Looking for documents in: {DEFAULT_DOWNLOADS_DIR}")
    print(f"🎯 Search criteria: {RELEVANCE_CRITERIA}")
    print(f"🤖 Using model: {DEFAULT_MODEL}")
    print(f"📊 Confidence threshold: {DEFAULT_CONFIDENCE_THRESHOLD}")
    print("-" * 80)
    
    processor = DocumentProcessor(
        downloads_dir=DEFAULT_DOWNLOADS_DIR,
        api_key=DEFAULT_API_KEY,
        base_url=DEFAULT_BASE_URL
    )
    
    processor.process_documents(
        relevance_criteria=RELEVANCE_CRITERIA,
        model=DEFAULT_MODEL,
        confidence_threshold=DEFAULT_CONFIDENCE_THRESHOLD
    )


if __name__ == "__main__":
    main()