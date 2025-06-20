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
        self.relevant_dir = self.downloads_dir / "relevant"
        self.irrelevant_dir = self.downloads_dir / "irrelevant"
        
        # Create directories
        self.relevant_dir.mkdir(exist_ok=True)
        self.irrelevant_dir.mkdir(exist_ok=True)
        
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
        """Extract text from first few pages of PDF"""
        try:
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text = ""
                
                # Extract text from first max_pages pages
                for i, page in enumerate(reader.pages[:max_pages]):
                    text += page.extract_text() + "\n"
                
                return text.strip()
        except Exception as e:
            print(f"Error extracting text from {pdf_path}: {e}")
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
    
    def load_pdf_metadata(self):
        """Load PDF metadata from Scrapy output."""
        metadata_file = self.downloads_dir / "pdf_metadata.json"
        if metadata_file.exists():
            try:
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    metadata_list = json.load(f)
                # Create filename to metadata mapping
                return {item["filename"]: item for item in metadata_list}
            except Exception as e:
                print(f"Warning: Could not load PDF metadata: {e}")
        return {}

    def process_documents(self, relevance_criteria: str, model: str = "gpt-4o-mini", 
                         confidence_threshold: float = 0.7):
        """Process all PDFs in downloads directory with comprehensive analysis"""
        
        pdf_files = list(self.downloads_dir.glob("*.pdf"))
        if not pdf_files:
            print("No PDF files found in downloads directory")
            return
        
        # Load metadata mapping
        pdf_metadata = self.load_pdf_metadata()
        
        all_analyses = []
        relevant_count = 0
        
        print(f"Processing {len(pdf_files)} documents...")
        print(f"Analysis criteria: {relevance_criteria}")
        print(f"Confidence threshold: {confidence_threshold}")
        print("-" * 80)
        
        for pdf_file in pdf_files:
            print(f"Processing: {pdf_file.name}")
            
            # Extract text
            text = self.extract_pdf_text(pdf_file)
            if not text:
                print(f"  ❌ Could not extract text, skipping")
                continue
            
            # Get PDF URL from metadata
            file_metadata = pdf_metadata.get(pdf_file.name, {})
            pdf_url = file_metadata.get("pdf_url", "Unknown")
            src_page = file_metadata.get("src_page", "Unknown")
            
            print(f"  📎 PDF URL: {pdf_url}")
            
            # Comprehensive document analysis
            analysis = self.analyze_document(text, relevance_criteria, pdf_url, pdf_file.name, model)
            
            # Decision based on relevance and confidence
            is_relevant = (analysis["relevant"] and 
                          analysis["confidence"] >= confidence_threshold)
            
            if is_relevant:
                # Move to relevant folder
                destination = self.relevant_dir / pdf_file.name
                pdf_file.rename(destination)
                relevant_count += 1
                status = "✅ RELEVANT"
            else:
                # Move to irrelevant folder  
                destination = self.irrelevant_dir / pdf_file.name
                pdf_file.rename(destination)
                status = "❌ IRRELEVANT"
            
            print(f"  {status}")
            print(f"  Confidence: {analysis['confidence']:.2f}")
            print(f"  Regulation: {analysis.get('regulation_name', 'Unknown')}")
            print(f"  Country: {analysis.get('country', 'Unknown')}")
            print(f"  Authority: {analysis.get('issuing_body', 'Unknown')}")
            print(f"  Translated: {analysis.get('translated_flag', False)}")
            print()
            
            # Add metadata
            analysis["filename"] = pdf_file.name
            analysis["moved_to"] = str(destination)
            analysis["processing_status"] = status
            
            all_analyses.append(analysis)
        
        # Save comprehensive results in structured JSON format
        results_file = self.downloads_dir / "regulatory_analysis.json"
        output_data = {
            "metadata": {
                "analysis_date": datetime.now().isoformat(),
                "criteria": relevance_criteria,
                "model": model,
                "confidence_threshold": confidence_threshold,
                "total_documents": len(pdf_files),
                "relevant_documents": relevant_count,
                "irrelevant_documents": len(pdf_files) - relevant_count
            },
            "regulations": all_analyses
        }
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        # Also save just the relevant regulations in a separate file
        relevant_regulations = [doc for doc in all_analyses if doc["relevant"]]
        if relevant_regulations:
            relevant_file = self.downloads_dir / "relevant_regulations.json"
            with open(relevant_file, 'w', encoding='utf-8') as f:
                json.dump({
                    "metadata": output_data["metadata"],
                    "regulations": relevant_regulations
                }, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Processing complete!")
        print(f"📊 Results: {relevant_count}/{len(pdf_files)} documents marked as relevant")
        print(f"📁 Relevant documents: {self.relevant_dir}")
        print(f"📁 Irrelevant documents: {self.irrelevant_dir}")
        print(f"📄 Full analysis saved to: {results_file}")
        if relevant_regulations:
            print(f"📄 Relevant regulations only: {relevant_file}")


def main():
    print("🔍 Starting PDF Document Filter...")
    print(f"📁 Looking for PDFs in: {DEFAULT_DOWNLOADS_DIR}")
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