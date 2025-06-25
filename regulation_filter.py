#!/usr/bin/env python3
"""
Regulation Document Filter Service

Processes regulation scraping results from country-based folders,
extracts text from PDF/TXT/HTML files, and uses LLM to analyze and summarize
regulations. Organizes output by country with chunked processing.
"""

import os
import json
import argparse
import html2text
from pathlib import Path
from typing import List, Dict, Any, Optional
import PyPDF2
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from bs4 import BeautifulSoup
import re


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
DEFAULT_INPUT_DIR = "regulation_scraping_results"
DEFAULT_OUTPUT_DIR = "regulation_analysis_results"
DEFAULT_CHUNK_SIZE = 4000
DEFAULT_MAX_WORKERS = 3
RELEVANCE_CRITERIA = "financial regulations, compliance requirements, ESG regulations, tax regulations, and legal disclosure requirements"


class RegulationProcessor:
    def __init__(self, input_dir: str = DEFAULT_INPUT_DIR, output_dir: str = DEFAULT_OUTPUT_DIR, 
                 api_key: str = None, base_url: str = None, chunk_size: int = 4000):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.chunk_size = chunk_size
        
        # Create output directory
        self.output_dir.mkdir(exist_ok=True)
        
        # Setup LangChain ChatOpenAI
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

SOURCE COUNTRY: {country}
REGULATION NAME: {regulation_name}
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
        
        # Initialize HTML to text converter
        self.html_converter = html2text.HTML2Text()
        self.html_converter.ignore_links = False
        self.html_converter.ignore_images = True
    
    def extract_pdf_text(self, pdf_path: Path, max_pages: int = None) -> str:
        """Extract text from PDF - all pages if max_pages=None, otherwise first max_pages"""
        try:
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text = ""
                
                # Determine pages to extract
                if max_pages is None:
                    pages_to_extract = reader.pages
                else:
                    pages_to_extract = reader.pages[:max_pages]
                
                # Extract text from selected pages
                for page in pages_to_extract:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
                
                return text.strip()
        except Exception as e:
            print(f"Error extracting text from {pdf_path}: {e}")
            return ""
    
    def extract_html_text(self, html_path: Path) -> str:
        """Extract text from HTML file"""
        try:
            with open(html_path, 'r', encoding='utf-8') as file:
                html_content = file.read()
                
            # First try BeautifulSoup for cleaner extraction
            try:
                soup = BeautifulSoup(html_content, 'html.parser')
                # Remove script and style elements
                for script in soup(["script", "style"]):
                    script.decompose()
                text = soup.get_text()
                # Clean up whitespace
                lines = (line.strip() for line in text.splitlines())
                chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                text = '\n'.join(chunk for chunk in chunks if chunk)
            except:
                # Fallback to html2text
                text = self.html_converter.handle(html_content)
            
            return text.strip()
        except Exception as e:
            print(f"Error extracting text from {html_path}: {e}")
            return ""
    
    def extract_text_file(self, text_path: Path) -> str:
        """Extract text from text file"""
        try:
            with open(text_path, 'r', encoding='utf-8') as file:
                return file.read().strip()
        except Exception as e:
            print(f"Error reading text from {text_path}: {e}")
            return ""
    
    def chunk_text(self, text: str, chunk_size: int = None) -> List[str]:
        """Split text into manageable chunks"""
        if chunk_size is None:
            chunk_size = self.chunk_size
        
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        # Try to split on sentences first
        sentences = re.split(r'(?<=[.!?])\s+', text)
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) <= chunk_size:
                current_chunk += sentence + " "
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + " "
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks if chunks else [text[:chunk_size]]
    
    def analyze_document_chunk(self, document_text: str, relevance_criteria: str, 
                              country: str, regulation_name: str, source_url: str, 
                              filename: str, model: str = DEFAULT_MODEL) -> Dict[str, Any]:
        """Use LangChain LLM to analyze a document chunk"""
        
        # Truncate text if too long
        if len(document_text) > 8000:
            document_text = document_text[:8000] + "...[truncated]"
        
        # Update model if different from default
        if model != DEFAULT_MODEL:
            self.llm.model_name = model
        
        try:
            current_date = datetime.now().strftime("%Y-%m-%d")
            
            result = self.chain.invoke({
                "criteria": relevance_criteria,
                "document_text": document_text,
                "country": country,
                "regulation_name": regulation_name,
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
            print(f"Error analyzing document chunk: {e}")
            current_date = datetime.now().strftime("%Y-%m-%d")
            return {
                "relevant": False,
                "confidence": 0.0,
                "unique_id": f"ERROR_{country}_{current_date}_{hash(filename) % 10000}",
                "country": country,
                "jurisdiction": "Unknown", 
                "issuing_body": "Unknown",
                "tag": "Error",
                "regulation_name": regulation_name,
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
    
    def load_regulation_info(self, regulation_folder: Path) -> Dict[str, Any]:
        """Load regulation info from regulation_info.json"""
        info_file = regulation_folder / "regulation_info.json"
        if info_file.exists():
            try:
                with open(info_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Warning: Could not load regulation info from {info_file}: {e}")
        return {}
    
    def process_regulation_folder(self, country: str, regulation_folder: Path, 
                                 relevance_criteria: str, confidence_threshold: float, 
                                 model: str) -> Dict[str, Any]:
        """Process all documents in a regulation folder"""
        
        print(f"\n🔄 Processing: [{country}] {regulation_folder.name}")
        
        # Load regulation info
        reg_info = self.load_regulation_info(regulation_folder)
        regulation_name = reg_info.get('regulation_name', regulation_folder.name)
        
        # Find all document files
        pdf_files = list(regulation_folder.glob("*.pdf"))
        html_files = list(regulation_folder.glob("*.html"))
        txt_files = list(regulation_folder.glob("*.txt"))
        
        all_files = pdf_files + html_files + txt_files
        
        if not all_files:
            print(f"  No documents found in {regulation_folder}")
            return None
        
        print(f"  Found {len(all_files)} documents ({len(pdf_files)} PDFs, {len(html_files)} HTML, {len(txt_files)} TXT)")
        
        # Process each file and collect chunks
        all_chunks = []
        file_metadata = {}
        
        for doc_file in all_files:
            # Extract text based on file type
            if doc_file.suffix.lower() == '.pdf':
                text = self.extract_pdf_text(doc_file)
            elif doc_file.suffix.lower() == '.html':
                text = self.extract_html_text(doc_file)
            elif doc_file.suffix.lower() == '.txt':
                text = self.extract_text_file(doc_file)
            else:
                continue
            
            if not text:
                continue
            
            # Chunk the text
            chunks = self.chunk_text(text)
            
            for i, chunk in enumerate(chunks):
                chunk_info = {
                    'text': chunk,
                    'source_file': doc_file.name,
                    'source_path': str(doc_file),
                    'chunk_index': i,
                    'total_chunks': len(chunks),
                    'file_type': doc_file.suffix.lower(),
                    'regulation_name': regulation_name,
                    'country': country
                }
                all_chunks.append(chunk_info)
            
            file_metadata[doc_file.name] = {
                'file_size': doc_file.stat().st_size,
                'text_length': len(text),
                'chunk_count': len(chunks),
                'file_type': doc_file.suffix.lower()
            }
        
        if not all_chunks:
            print(f"  No text content extracted from documents")
            return None
        
        print(f"  Created {len(all_chunks)} chunks for analysis")
        
        # Analyze chunks in parallel
        relevant_analyses = []
        
        with ThreadPoolExecutor(max_workers=DEFAULT_MAX_WORKERS) as executor:
            future_to_chunk = {
                executor.submit(
                    self.analyze_document_chunk,
                    chunk['text'],
                    relevance_criteria,
                    country,
                    regulation_name,
                    reg_info.get('urls', ['Unknown'])[0] if reg_info.get('urls') else 'Unknown',
                    f"{chunk['source_file']}_chunk_{chunk['chunk_index']}",
                    model
                ): chunk for chunk in all_chunks
            }
            
            for future in as_completed(future_to_chunk):
                chunk = future_to_chunk[future]
                try:
                    analysis = future.result()
                    if analysis and analysis.get('relevant') and analysis.get('confidence', 0) >= confidence_threshold:
                        # Add chunk metadata
                        analysis.update({
                            'chunk_info': {
                                'source_file': chunk['source_file'],
                                'chunk_index': chunk['chunk_index'],
                                'total_chunks': chunk['total_chunks'],
                                'file_type': chunk['file_type']
                            },
                            'regulation_folder': regulation_folder.name,
                            'processed_at': datetime.now().isoformat()
                        })
                        relevant_analyses.append(analysis)
                        print(f"    ✅ Relevant chunk: {chunk['source_file']} (chunk {chunk['chunk_index']}/{chunk['total_chunks']})")
                    else:
                        print(f"    ❌ Irrelevant: {chunk['source_file']} (chunk {chunk['chunk_index']}/{chunk['total_chunks']})")
                        
                except Exception as e:
                    print(f"    Error processing chunk from {chunk['source_file']}: {e}")
        
        if not relevant_analyses:
            print(f"  No relevant content found")
            return None
        
        # Create summary result
        result = {
            'country': country,
            'regulation_name': regulation_name,
            'regulation_folder': regulation_folder.name,
            'total_documents': len(all_files),
            'total_chunks_analyzed': len(all_chunks),
            'relevant_chunks_found': len(relevant_analyses),
            'file_metadata': file_metadata,
            'regulation_info': reg_info,
            'relevant_analyses': relevant_analyses,
            'processed_at': datetime.now().isoformat()
        }
        
        print(f"  ✅ Found {len(relevant_analyses)} relevant chunks out of {len(all_chunks)} total")
        return result
    
    def save_country_results(self, country: str, country_results: List[Dict[str, Any]]):
        """Save results for a specific country"""
        if not country_results:
            return
        
        # Create country folder in output directory
        country_output_dir = self.output_dir / self.safe_folder_name(country)
        country_output_dir.mkdir(exist_ok=True)
        
        # Save summary for country
        country_summary = {
            'country': country,
            'total_regulations': len(country_results),
            'total_relevant_chunks': sum(r['relevant_chunks_found'] for r in country_results),
            'processed_at': datetime.now().isoformat(),
            'regulations': country_results
        }
        
        summary_file = country_output_dir / f"{self.safe_folder_name(country)}_summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(country_summary, f, indent=2, ensure_ascii=False)
        
        # Save individual chunk files
        for regulation_result in country_results:
            regulation_name = regulation_result['regulation_name']
            safe_reg_name = self.safe_folder_name(regulation_name)
            
            reg_folder = country_output_dir / safe_reg_name
            reg_folder.mkdir(exist_ok=True)
            
            # Save each relevant chunk as individual file
            for i, analysis in enumerate(regulation_result['relevant_analyses']):
                chunk_file = reg_folder / f"chunk_{i:03d}_{analysis['chunk_info']['source_file']}.json"
                with open(chunk_file, 'w', encoding='utf-8') as f:
                    json.dump(analysis, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Saved results for {country}: {len(country_results)} regulations, {country_summary['total_relevant_chunks']} chunks")
    
    def safe_folder_name(self, name: str) -> str:
        """Convert name to safe folder name"""
        safe_name = re.sub(r'[<>:"/\\|?*]', '_', name)
        safe_name = re.sub(r'\s+', '_', safe_name)
        safe_name = safe_name.strip('._')
        return safe_name[:100]
    
    def process_all_regulations(self, relevance_criteria: str = RELEVANCE_CRITERIA, 
                               model: str = DEFAULT_MODEL, 
                               confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD):
        """Process all regulation folders organized by country"""
        
        if not self.input_dir.exists():
            print(f"❌ Input directory not found: {self.input_dir}")
            return
        
        print(f"🚀 Starting Regulation Analysis")
        print(f"📂 Input directory: {self.input_dir}")
        print(f"📁 Output directory: {self.output_dir}")
        print(f"🎯 Analysis criteria: {relevance_criteria}")
        print(f"🤖 Using model: {model}")
        print(f"📊 Confidence threshold: {confidence_threshold}")
        print("=" * 80)
        
        # Find all country folders
        country_folders = [d for d in self.input_dir.iterdir() if d.is_dir()]
        
        if not country_folders:
            print(f"❌ No country folders found in {self.input_dir}")
            return
        
        print(f"Found {len(country_folders)} country folders")
        
        all_results = []
        
        # Process each country
        for country_folder in country_folders:
            country = country_folder.name
            print(f"\n{'='*60}")
            print(f"🌍 Processing country: {country}")
            
            # Find all regulation folders in this country
            regulation_folders = [d for d in country_folder.iterdir() if d.is_dir()]
            
            if not regulation_folders:
                print(f"  No regulation folders found in {country}")
                continue
            
            print(f"  Found {len(regulation_folders)} regulation folders")
            
            country_results = []
            
            # Process each regulation in this country
            for regulation_folder in regulation_folders:
                result = self.process_regulation_folder(
                    country, regulation_folder, relevance_criteria, 
                    confidence_threshold, model
                )
                
                if result:
                    country_results.append(result)
                    all_results.append(result)
            
            # Save country results
            if country_results:
                self.save_country_results(country, country_results)
            else:
                print(f"  No relevant content found for {country}")
        
        # Generate final summary
        self.generate_final_summary(all_results, relevance_criteria, model, confidence_threshold)
        
        print(f"\n🎉 Analysis completed!")
        print(f"📊 Total countries processed: {len(set(r['country'] for r in all_results))}")
        print(f"📋 Total regulations analyzed: {len(all_results)}")
        total_chunks = sum(r['relevant_chunks_found'] for r in all_results)
        print(f"📄 Total relevant chunks found: {total_chunks}")
        print(f"📁 Results saved to: {self.output_dir}")
    
    def generate_final_summary(self, all_results: List[Dict[str, Any]], 
                              relevance_criteria: str, model: str, confidence_threshold: float):
        """Generate final summary report"""
        
        final_summary = {
            'analysis_metadata': {
                'analysis_date': datetime.now().isoformat(),
                'criteria': relevance_criteria,
                'model': model,
                'confidence_threshold': confidence_threshold,
                'input_directory': str(self.input_dir),
                'output_directory': str(self.output_dir)
            },
            'summary_statistics': {
                'total_countries': len(set(r['country'] for r in all_results)),
                'total_regulations': len(all_results),
                'total_documents': sum(r['total_documents'] for r in all_results),
                'total_chunks_analyzed': sum(r['total_chunks_analyzed'] for r in all_results),
                'total_relevant_chunks': sum(r['relevant_chunks_found'] for r in all_results)
            },
            'country_breakdown': {}
        }
        
        # Create country breakdown
        for result in all_results:
            country = result['country']
            if country not in final_summary['country_breakdown']:
                final_summary['country_breakdown'][country] = {
                    'regulation_count': 0,
                    'document_count': 0,
                    'relevant_chunk_count': 0,
                    'regulations': []
                }
            
            country_data = final_summary['country_breakdown'][country]
            country_data['regulation_count'] += 1
            country_data['document_count'] += result['total_documents']
            country_data['relevant_chunk_count'] += result['relevant_chunks_found']
            country_data['regulations'].append({
                'regulation_name': result['regulation_name'],
                'folder': result['regulation_folder'],
                'documents': result['total_documents'],
                'relevant_chunks': result['relevant_chunks_found']
            })
        
        # Save final summary
        summary_file = self.output_dir / "final_analysis_summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(final_summary, f, indent=2, ensure_ascii=False)
        
        print(f"\n📊 Final summary saved to: {summary_file}")


def main():
    print("🔍 Starting Regulation Document Filter...")
    print(f"📂 Input directory: {DEFAULT_INPUT_DIR}")
    print(f"📁 Output directory: {DEFAULT_OUTPUT_DIR}")
    print(f"🎯 Analysis criteria: {RELEVANCE_CRITERIA}")
    print(f"🤖 Using model: {DEFAULT_MODEL}")
    print(f"📊 Confidence threshold: {DEFAULT_CONFIDENCE_THRESHOLD}")
    print(f"📄 Chunk size: {DEFAULT_CHUNK_SIZE}")
    print("-" * 80)
    
    processor = RegulationProcessor(
        input_dir=DEFAULT_INPUT_DIR,
        output_dir=DEFAULT_OUTPUT_DIR,
        api_key=DEFAULT_API_KEY,
        base_url=DEFAULT_BASE_URL,
        chunk_size=DEFAULT_CHUNK_SIZE
    )
    
    processor.process_all_regulations(
        relevance_criteria=RELEVANCE_CRITERIA,
        model=DEFAULT_MODEL,
        confidence_threshold=DEFAULT_CONFIDENCE_THRESHOLD
    )


def main_cli():
    """CLI version - kept for backward compatibility"""
    parser = argparse.ArgumentParser(description='Process regulation scraping results')
    parser.add_argument('--input-dir', default=DEFAULT_INPUT_DIR, 
                       help='Input directory with regulation scraping results')
    parser.add_argument('--output-dir', default=DEFAULT_OUTPUT_DIR,
                       help='Output directory for processed results')
    parser.add_argument('--api-key', help='OpenAI API key')
    parser.add_argument('--base-url', default=DEFAULT_BASE_URL, help='API base URL')
    parser.add_argument('--model', default=DEFAULT_MODEL, help='Model to use')
    parser.add_argument('--confidence-threshold', type=float, default=DEFAULT_CONFIDENCE_THRESHOLD,
                       help='Confidence threshold for relevance')
    parser.add_argument('--chunk-size', type=int, default=DEFAULT_CHUNK_SIZE,
                       help='Text chunk size for processing')
    parser.add_argument('--criteria', default=RELEVANCE_CRITERIA,
                       help='Relevance criteria for document analysis')
    
    args = parser.parse_args()
    
    processor = RegulationProcessor(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        api_key=args.api_key,
        base_url=args.base_url,
        chunk_size=args.chunk_size
    )
    
    processor.process_all_regulations(
        relevance_criteria=args.criteria,
        model=args.model,
        confidence_threshold=args.confidence_threshold
    )


if __name__ == "__main__":
    main()