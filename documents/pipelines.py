from itemadapter import ItemAdapter
from pypdf import PdfReader
from langdetect import detect
from pathlib import Path
import json
from datetime import datetime
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
import PyPDF2
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import List, Dict, Any
import os

class MetaPipeline:
    """Extract PDF metadata and save URL mapping for LLM processing."""

    def __init__(self):
        self.pdf_metadata = []

    def process_item(self, item, spider):
        ad = ItemAdapter(item)

        # Extract PDF stats and URL mapping
        for f in ad.get("files", []):
            pdf_path = Path(spider.settings["FILES_STORE"]) / f["path"]
            if pdf_path.suffix.lower() == ".pdf":
                try:
                    reader = PdfReader(str(pdf_path))
                    ad["page_count"] = len(reader.pages)
                    snippet = (reader.pages[0].extract_text() or "")[:700]
                    try:
                        ad["language"] = detect(snippet)
                    except Exception:
                        ad["language"] = "unknown"
                    
                    # Save comprehensive metadata with URL mapping
                    metadata = {
                        "filename": f["path"].split("/")[-1],  # Just the filename
                        "file_path": str(pdf_path),            # Full file path
                        "pdf_url": ad.get("pdf_url", "Unknown"), # Direct PDF URL
                        "src_page": ad.get("src_page", "Unknown"), # Source page URL
                        "title": ad.get("title", "Unknown"),    # Link text/title
                        "page_count": ad.get("page_count", 0),  # Number of pages
                        "language": ad.get("language", "unknown"), # Detected language
                        "file_size": pdf_path.stat().st_size if pdf_path.exists() else 0,
                        "scraped_at": datetime.now().isoformat(),
                        "status": "downloaded"
                    }
                    
                    self.pdf_metadata.append(metadata)
                    
                except Exception as e:
                    spider.logger.error(f"Error processing PDF {pdf_path}: {e}")
        
        return item

    def close_spider(self, spider):
        """Save all PDF metadata to JSON file when spider closes."""
        if self.pdf_metadata:
            downloads_dir = Path(spider.settings["FILES_STORE"])
            metadata_file = downloads_dir / "pdf_metadata.json"
            
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(self.pdf_metadata, f, indent=2, ensure_ascii=False)
            
            spider.logger.info(f"Saved PDF metadata for {len(self.pdf_metadata)} files to {metadata_file}")


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


class SmartDownloadPipeline:
    """Downloads PDFs, analyzes them, and only saves relevant ones"""
    
    def __init__(self):
        # Configuration
        self.api_key = os.getenv("OPENAI_API_KEY", "your-api-key-here")
        self.base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        self.model = "gpt-4o-mini"
        self.confidence_threshold = 0.7
        # Change this to your specific search criteria
        self.criteria = os.getenv("LLM_CRITERIA", "regulatory, compliance, and legal documents related to financial reporting and disclosure requirements")
        
        # LLM setup
        self.llm = ChatOpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            temperature=0.1,
            model=self.model
        )
        
        self.parser = JsonOutputParser(pydantic_object=DocumentAnalysis)
        
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
        
        self.chain = self.prompt | self.llm | self.parser
        
        # Async processing
        self.executor = ThreadPoolExecutor(max_workers=5)
        self.regulatory_data = []
        self.processed_count = 0
        
    def extract_pdf_text(self, file_content: bytes, filename: str) -> str:
        """Extract text from PDF content"""
        try:
            pdf_stream = BytesIO(file_content)
            reader = PyPDF2.PdfReader(pdf_stream)
            text = ""
            
            # Extract text from first 5 pages
            for i, page in enumerate(reader.pages[:5]):
                text += page.extract_text() + "\n"
            
            return text.strip()
        except Exception as e:
            print(f"Error extracting text from {filename}: {e}")
            return ""
    
    async def analyze_document_async(self, document_text: str, source_url: str, filename: str) -> Dict[str, Any]:
        """Async LLM analysis"""
        try:
            # Truncate text if too long
            if len(document_text) > 8000:
                document_text = document_text[:8000] + "...[truncated]"
            
            current_date = datetime.now().strftime("%Y-%m-%d")
            
            # Run LLM analysis in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self.executor,
                lambda: self.chain.invoke({
                    "criteria": self.criteria,
                    "document_text": document_text,
                    "source_url": source_url,
                    "filename": filename,
                    "format_instructions": self.parser.get_format_instructions()
                })
            )
            
            # Ensure metadata fields are populated
            if not result.get("last_scraped"):
                result["last_scraped"] = current_date
            if not result.get("source_url"):
                result["source_url"] = source_url
            if result.get("change_detected") is None:
                result["change_detected"] = False
            
            return result
            
        except Exception as e:
            print(f"Error analyzing document {filename}: {e}")
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
    
    def process_item(self, item, spider):
        """Download PDFs in memory, analyze, and save only JSON data (no PDF files)"""
        adapter = ItemAdapter(item)
        file_urls = adapter.get('file_urls', [])
        
        if not file_urls:
            return item
        
        # Process each PDF URL
        for pdf_url in file_urls:
            try:
                spider.logger.info(f"📥 Analyzing: {pdf_url}")
                
                # Download PDF content in memory only
                import requests
                response = requests.get(pdf_url, timeout=30)
                response.raise_for_status()
                
                pdf_content = response.content
                filename = pdf_url.split('/')[-1]
                if not filename.endswith('.pdf'):
                    filename += '.pdf'
                
                # Extract text from downloaded content (in memory)
                text = self.extract_pdf_text(pdf_content, filename)
                if not text:
                    spider.logger.warning(f"Could not extract text from {filename}")
                    continue
                
                # Run async LLM analysis
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                try:
                    analysis = loop.run_until_complete(
                        self.analyze_document_async(text, pdf_url, filename)
                    )
                finally:
                    loop.close()
                
                # Decision based on relevance and confidence
                is_relevant = (analysis["relevant"] and 
                             analysis["confidence"] >= self.confidence_threshold)
                
                if is_relevant:
                    spider.logger.info(f"✅ RELEVANT (JSON only): {filename} - {analysis.get('regulation_name', 'Unknown')}")
                    
                    # Add to regulatory data (NO PDF file saved)
                    analysis["filename"] = filename
                    analysis["pdf_url"] = pdf_url
                    self.regulatory_data.append(analysis)
                    
                    # Save JSON results immediately
                    self.save_regulatory_data(spider)
                    
                else:
                    spider.logger.info(f"❌ IRRELEVANT (skipped): {filename} - Confidence: {analysis['confidence']:.2f}")
                
                self.processed_count += 1
                
            except Exception as e:
                spider.logger.error(f"Error processing {pdf_url}: {e}")
        
        # Clear files to prevent FilesPipeline from saving anything
        adapter['file_urls'] = []
        adapter['files'] = []
        
        return item
    
    def save_regulatory_data(self, spider):
        """Save regulatory data to JSON file"""
        if self.regulatory_data:
            downloads_dir = Path(spider.settings["FILES_STORE"])
            output_file = downloads_dir / "live_regulatory_analysis.json"
            
            output_data = {
                "metadata": {
                    "last_updated": datetime.now().isoformat(),
                    "criteria": self.criteria,
                    "model": self.model,
                    "confidence_threshold": self.confidence_threshold,
                    "total_relevant_documents": len(self.regulatory_data),
                    "total_processed": self.processed_count
                },
                "regulations": self.regulatory_data
            }
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    def close_spider(self, spider):
        """Final processing when spider closes"""
        spider.logger.info(f"🎯 Processing complete: {len(self.regulatory_data)} relevant documents found out of {self.processed_count} processed")
        
        # Final save
        self.save_regulatory_data(spider)
        
        # Cleanup
        self.executor.shutdown(wait=True)