#!/usr/bin/env python3
"""
AI-Powered Metadata Results Consolidator V3 - Multithreaded
STEP 3 of the pipeline: Uses LLM to intelligently consolidate metadata analysis results
into structured regulatory summaries, using ConsolidatedRegulation format with row numbers.
OPTIMIZED: Processes only ESG-relevant documents from esg_relevant_by_country folder.
MULTITHREADED: Parallel processing of countries for maximum speed.
"""

import json
import os
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
import argparse
import pandas as pd
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from queue import Queue

# LangChain imports for LLM processing
try:
    from langchain_openai import ChatOpenAI
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import JsonOutputParser
    from langchain_core.pydantic_v1 import BaseModel, Field
    LANGCHAIN_AVAILABLE = True
except ImportError:
    print("Warning: LangChain not available. Install with: pip install langchain langchain-openai")
    LANGCHAIN_AVAILABLE = False


class ConsolidatedRegulation(BaseModel):
    """Pydantic model for the final consolidated regulation - from consolidate_regulations.py"""
    row_number: int = Field(description="Excel row number extracted from folder path")
    unique_id: str = Field(description="Unique identifier for the consolidated regulation")
    country: str = Field(description="Primary country or 'Multiple' if cross-jurisdictional")
    jurisdiction: str = Field(description="Primary jurisdiction or summary of multiple")
    issuing_body: str = Field(description="Primary authority or summary of multiple bodies")
    tag: str = Field(description="Primary disclosure area or summary of areas covered")
    regulation_name: str = Field(description="Comprehensive name covering all sub-regulations")
    publication_date: str = Field(description="Most recent or range of publication dates")
    regulation_status: str = Field(description="Overall status across all regulations")
    summary: str = Field(description="Comprehensive summary of all disclosure requirements")
    applicability: str = Field(description="Combined scope summary covering all regulations")
    scoping_threshold: str = Field(description="Combined minimum thresholds across regulations")
    effective_date: str = Field(description="Earliest or most relevant mandatory reporting date")
    timeline_details: str = Field(description="Comprehensive implementation timeline")
    financial_integration: str = Field(description="Overall integration with financial reporting")
    filing_mechanism: str = Field(description="Summary of all filing mechanisms")
    reporting_frequency: str = Field(description="Summary of required reporting frequencies")
    assurance_requirement: str = Field(description="Overall assurance requirements")
    penalties: str = Field(description="Summary of all non-compliance penalties")
    full_text_link: str = Field(description="Primary or most comprehensive regulation link")
    translated_flag: bool = Field(description="Whether any content was translated")
    source_url: str = Field(description="Primary source or summary of sources")
    last_scraped: str = Field(description="Most recent scraping date")
    change_detected: bool = Field(description="Whether any changes were detected")


class ThreadSafeProgress:
    """Thread-safe progress tracking"""
    def __init__(self):
        self.lock = threading.Lock()
        self.processed = 0
        self.successful = 0
        self.failed = 0
        self.total = 0
        self.results = {}
    
    def set_total(self, total: int):
        with self.lock:
            self.total = total
    
    def update_progress(self, country: str, success: bool, result: Optional[Dict] = None):
        with self.lock:
            self.processed += 1
            if success:
                self.successful += 1
                if result:
                    self.results[country] = result
            else:
                self.failed += 1
            
            # Print progress
            progress_pct = (self.processed / self.total * 100) if self.total > 0 else 0
            print(f"  📊 Progress: {self.processed}/{self.total} ({progress_pct:.1f}%) | ✅ {self.successful} | ❌ {self.failed}")
    
    def get_results(self) -> Dict:
        with self.lock:
            return {
                'processed': self.processed,
                'successful': self.successful,
                'failed': self.failed,
                'results': self.results.copy()
            }


class AIMetadataConsolidator:
    """Uses AI to consolidate metadata analysis results into structured regulation summaries"""
    
    def __init__(self, model_name: str = "gpt-4o-mini", temperature: float = 0.1):
        if not LANGCHAIN_AVAILABLE:
            raise ImportError("LangChain is required for AI consolidation. Install with: pip install langchain langchain-openai")
        
        # Create thread-local storage for LLM instances
        self.thread_local = threading.local()
        self.model_name = model_name
        self.temperature = temperature
        self.max_context_length = 15000
        
        # Create the prompt template (shared across threads)
        self.consolidation_prompt = ChatPromptTemplate.from_template("""
You are an expert regulatory analyst specializing in ESG and financial regulations. Analyze the following metadata analysis results from multiple ESG-relevant documents for a specific regulation/country and create a comprehensive, structured regulatory summary.

COUNTRY: {country}
REGULATION CONTEXT: {regulation_context}
EXCEL ROW NUMBER: {row_number}

ESG DOCUMENT ANALYSIS RESULTS:
{document_analyses}

INSTRUCTIONS:
1. Synthesize information from ALL ESG document analyses to create ONE comprehensive regulatory summary
2. Focus on ESG-relevant content and regulatory requirements
3. Extract key requirements, dates, thresholds, and compliance obligations
4. Identify the primary regulation name and issuing authority
5. Create a clear comprehensive summary that captures the essence of the regulatory framework
6. List specific disclosure and reporting requirements
7. Assign a confidence score based on the quality and completeness of source material
8. Create a unique ID in format: {country_code}_ESG_{year}_v1
9. Set row_number as the first field from the provided Excel row number

ANALYSIS FOCUS:
- Prioritize ESG-relevant requirements and disclosures
- Extract specific dates, thresholds, and quantitative requirements
- Identify who must comply and when
- Summarize reporting frequencies and filing mechanisms
- Note enforcement mechanisms and penalties
- Map to ConsolidatedRegulation format with comprehensive regulatory details

{format_instructions}
""")
    
    def get_llm_chain(self):
        """Get thread-local LLM chain"""
        if not hasattr(self.thread_local, 'llm'):
            self.thread_local.llm = ChatOpenAI(
                api_key=os.getenv("OPENAI_API_KEY"),
                base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
                temperature=self.temperature,
                model=self.model_name
            )
            
            self.thread_local.parser = JsonOutputParser(pydantic_object=ConsolidatedRegulation)
            self.thread_local.chain = self.consolidation_prompt | self.thread_local.llm | self.thread_local.parser
        
        return self.thread_local.chain, self.thread_local.parser
    
    def extract_row_index_from_country(self, country_folder_name: str) -> int:
        """Extract Excel row index from country folder name like 'Row5_Country_Name'"""
        try:
            match = re.match(r'Row(\d+)_', country_folder_name)
            return int(match.group(1)) if match else None
        except:
            return None
    
    def load_country_data_from_esg_folder(self, country_folder: Path) -> Dict[str, Any]:
        """Load ESG-relevant metadata analysis data for a country from esg_relevant_by_country folder"""
        country_folder_name = country_folder.name
        # Extract clean country name (remove row prefix if present)
        country_name = re.sub(r'^Row\d+_', '', country_folder_name)
        
        # Extract Excel row index from country folder name
        excel_row_index = self.extract_row_index_from_country(country_folder_name)
        
        # Collect all ESG document analyses from all regulations (all files in this folder are ESG-relevant)
        all_esg_documents = []
        regulation_contexts = []
        
        # Get all regulation folders
        regulation_folders = [d for d in country_folder.iterdir() if d.is_dir()]
        
        for reg_folder in regulation_folders:
            regulation_name = reg_folder.name
            
            # Load regulation summary if exists
            reg_summary_file = reg_folder / "regulation_summary.json"
            if reg_summary_file.exists():
                try:
                    with open(reg_summary_file, 'r', encoding='utf-8') as f:
                        reg_summary = json.load(f)
                        regulation_contexts.append({
                            'name': regulation_name,
                            'summary': reg_summary,
                            'excel_row_index': excel_row_index
                        })
                except Exception as e:
                    pass  # Skip failed regulation summaries
            
            # Load all document analyses (all are ESG-relevant in this folder structure)
            analysis_files = list(reg_folder.glob("*_analysis.json"))
            for analysis_file in analysis_files:
                try:
                    with open(analysis_file, 'r', encoding='utf-8') as f:
                        doc_analysis = json.load(f)
                        doc_analysis['regulation_folder'] = regulation_name
                        doc_analysis['excel_row_index'] = excel_row_index
                        # Mark as ESG relevant (since it's in the ESG folder)
                        doc_analysis['esg_relevant'] = True
                        all_esg_documents.append(doc_analysis)
                except Exception as e:
                    pass  # Skip failed document analyses
        
        return {
            'country': country_name,
            'country_folder_name': country_folder_name,
            'excel_row_index': excel_row_index,
            'regulation_contexts': regulation_contexts,
            'all_documents': all_esg_documents,  # All documents are ESG-relevant
            'esg_documents': all_esg_documents   # Same as all_documents since we're in ESG folder
        }
    
    def prepare_analysis_text(self, country_data: Dict[str, Any]) -> Dict[str, str]:
        """Prepare text for AI analysis with context length management"""
        esg_documents = country_data['esg_documents']
        
        # Create regulation context summary
        regulation_context = f"Country: {country_data['country']}\n"
        regulation_context += f"Total Regulations: {len(country_data['regulation_contexts'])}\n"
        regulation_context += f"ESG Relevant Documents: {len(esg_documents)}\n"
        regulation_context += f"Processing Source: ESG-relevant documents (pre-filtered)\n\n"
        
        for reg_ctx in country_data['regulation_contexts']:
            regulation_context += f"Regulation: {reg_ctx['name']}\n"
            if reg_ctx.get('summary'):
                summary_text = str(reg_ctx['summary'])[:500]
                regulation_context += f"Summary: {summary_text}...\n" if len(str(reg_ctx['summary'])) > 500 else f"Summary: {summary_text}\n"
            regulation_context += "\n"
        
        # Sort ESG documents by score (highest first) and limit count
        sorted_docs = sorted(esg_documents, key=lambda x: x.get('esg_match_score', 0), reverse=True)
        
        # Start with top documents and add until we approach context limit
        document_analyses = ""
        current_length = len(regulation_context)
        
        for i, doc in enumerate(sorted_docs, 1):
            doc_section = f"\n--- ESG DOCUMENT {i} ---\n"
            doc_section += f"File: {doc.get('file_name', 'Unknown')}\n"
            doc_section += f"Type: {doc.get('file_type', 'Unknown')}\n"
            doc_section += f"ESG Score: {doc.get('esg_match_score', 'N/A')}\n"
            doc_section += f"Regulation: {doc.get('regulation_folder', 'Unknown')}\n"
            doc_section += f"Source URL: {doc.get('source_url', 'Unknown')}\n"
            
            # Include metadata if available (but limit size)
            metadata = doc.get('metadata')
            if metadata and isinstance(metadata, dict):
                key_metadata = {
                    'summary': str(metadata.get('summary', ''))[:300] + '...' if len(str(metadata.get('summary', ''))) > 300 else metadata.get('summary', ''),
                    'key_requirements': metadata.get('key_requirements', [])[:3],
                    'effective_date': metadata.get('effective_date', ''),
                    'issuing_body': metadata.get('issuing_body', '')
                }
                doc_section += f"Key Metadata: {json.dumps(key_metadata, indent=1)}\n"
            
            # Include text sample
            extracted_text = doc.get('extracted_text', '')
            if extracted_text:
                remaining_context = self.max_context_length - current_length - len(doc_section)
                if remaining_context > 500:
                    max_text_length = min(1500, remaining_context - 200)
                    text_sample = extracted_text[:max_text_length]
                    if len(extracted_text) > max_text_length:
                        text_sample += "... [truncated]"
                    doc_section += f"Content Sample: {text_sample}\n"
            
            doc_section += "\n"
            
            # Check context limit
            if current_length + len(doc_section) > self.max_context_length:
                if i > 5:
                    break
                else:
                    # Reduce text for critical documents
                    extracted_text = doc.get('extracted_text', '')
                    if extracted_text:
                        text_sample = extracted_text[:800] + "... [truncated for context]"
                        doc_section = doc_section.split("Content Sample:")[0]
                        doc_section += f"Content Sample: {text_sample}\n\n"
            
            document_analyses += doc_section
            current_length += len(doc_section)
            
            # Safety limit
            if i >= 25:
                break
        
        return {
            'regulation_context': regulation_context,
            'document_analyses': document_analyses
        }
    
    def consolidate_country_with_ai(self, country_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Use AI to consolidate country metadata into structured regulation summary"""
        country = country_data['country']
        esg_count = len(country_data['esg_documents'])
        row_number = country_data.get('excel_row_index')
        
        if esg_count == 0:
            return None
        
        try:
            # Get thread-local LLM chain
            chain, parser = self.get_llm_chain()
            
            # Prepare analysis text
            analysis_data = self.prepare_analysis_text(country_data)
            
            # Generate country code for unique ID
            country_code = country[:3].upper()
            
            # Use AI to create consolidated summary
            result = chain.invoke({
                "country": country,
                "country_code": country_code,
                "year": datetime.now().year,
                "row_number": row_number,
                "regulation_context": analysis_data['regulation_context'],
                "document_analyses": analysis_data['document_analyses'],
                "format_instructions": parser.get_format_instructions()
            })
            
            # Ensure row_number is set
            if 'row_number' not in result:
                result['row_number'] = row_number
            
            # Add processing metadata
            result['processing_metadata'] = {
                'source_documents_total': len(country_data['all_documents']),
                'esg_documents_analyzed': len(country_data['esg_documents']),
                'regulations_covered': len(country_data['regulation_contexts']),
                'processing_date': datetime.now().isoformat(),
                'ai_model_used': self.model_name,
                'source_country_folder': country,
                'excel_row_index': row_number,
                'country_folder_name': country_data.get('country_folder_name'),
                'processing_source': 'esg_relevant_by_country',
                'optimization': 'v3_esg_pre_filtered_multithreaded',
                'thread_id': threading.current_thread().ident
            }
            
            return result
            
        except Exception as e:
            return self.create_fallback_summary(country_data)
    
    def create_fallback_summary(self, country_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create basic summary if AI fails"""
        country = country_data['country']
        esg_docs = country_data['esg_documents']
        row_number = country_data.get('excel_row_index')
        
        return {
            "row_number": row_number,
            "unique_id": f"{country[:3].upper()}_ESG_{datetime.now().year}_FALLBACK",
            "country": country,
            "jurisdiction": f"{country} jurisdiction",
            "issuing_body": "Multiple Authorities",
            "tag": "ESG/Sustainability",
            "regulation_name": f"{country} ESG Regulatory Framework",
            "publication_date": str(datetime.now().year),
            "regulation_status": "Analysis Generated",
            "summary": f"Automated analysis of {len(esg_docs)} ESG-relevant documents from {country}",
            "applicability": "To be determined through manual review",
            "scoping_threshold": "Various thresholds identified",
            "effective_date": str(datetime.now().year),
            "timeline_details": "Multiple implementation phases",
            "financial_integration": "Integration requirements vary",
            "filing_mechanism": "Multiple filing requirements",
            "reporting_frequency": "Various frequencies required",
            "assurance_requirement": "To be determined",
            "penalties": "Various enforcement mechanisms",
            "full_text_link": esg_docs[0].get('source_url', 'Unknown') if esg_docs else 'Unknown',
            "translated_flag": False,
            "source_url": "Multiple sources",
            "last_scraped": datetime.now().strftime("%Y-%m-%d"),
            "change_detected": False,
            "processing_metadata": {
                "fallback_reason": "AI consolidation failed",
                "source_documents_total": len(country_data['all_documents']),
                "esg_documents_analyzed": len(esg_docs),
                "processing_date": datetime.now().isoformat(),
                "excel_row_index": row_number,
                "country_folder_name": country_data.get('country_folder_name'),
                "processing_source": "esg_relevant_by_country",
                "optimization": "v3_esg_pre_filtered_multithreaded",
                "thread_id": threading.current_thread().ident
            }
        }


def process_single_country(consolidator: AIMetadataConsolidator, country_folder: Path, 
                          output_dir: Path, progress: ThreadSafeProgress) -> None:
    """Process a single country in a thread"""
    country_name = None
    try:
        # Load country data
        country_data = consolidator.load_country_data_from_esg_folder(country_folder)
        country_name = country_data['country']
        
        # AI consolidation
        consolidated_summary = consolidator.consolidate_country_with_ai(country_data)
        
        if consolidated_summary:
            row_number = consolidated_summary.get('row_number', 0)
            
            # Save individual country summary
            country_file = output_dir / f"{country_name}_consolidated_regulation_v3.json"
            with open(country_file, 'w', encoding='utf-8') as f:
                json.dump(consolidated_summary, f, indent=2, ensure_ascii=False)
            
            # Update progress
            result_info = {
                'file': str(country_file),
                'row_number': row_number,
                'unique_id': consolidated_summary.get('unique_id'),
                'regulation_name': consolidated_summary.get('regulation_name'),
                'country': consolidated_summary.get('country'),
                'jurisdiction': consolidated_summary.get('jurisdiction')
            }
            
            progress.update_progress(country_name, True, result_info)
            
            esg_count = consolidated_summary.get('processing_metadata', {}).get('esg_documents_analyzed', 0)
            thread_id = threading.current_thread().ident
            print(f"    🧵 Thread {thread_id}: ✅ Row {row_number} - {country_name}: {esg_count} ESG docs")
        else:
            progress.update_progress(country_name or country_folder.name, False)
            print(f"    🧵 Thread {threading.current_thread().ident}: ❌ {country_folder.name}: No ESG content")
            
    except Exception as e:
        progress.update_progress(country_name or country_folder.name, False)
        print(f"    🧵 Thread {threading.current_thread().ident}: 💥 Error processing {country_folder.name}: {e}")


def create_countries_excel_summary(results: Dict[str, Any], output_dir: Path) -> str:
    """Create Excel file with all countries as rows and key information as columns"""
    try:
        print("\n📊 Creating Excel summary of all countries...")
        
        # Prepare data for Excel
        excel_data = []
        
        for country, country_info in results['country_summaries'].items():
            # Load the detailed country data
            country_file = Path(country_info['file'])
            if country_file.exists():
                try:
                    with open(country_file, 'r', encoding='utf-8') as f:
                        country_data = json.load(f)
                    
                    # Extract key information for Excel row
                    row = {
                        'Row_Number': country_data.get('row_number', 0),
                        'Country': country,
                        'Unique_ID': country_data.get('unique_id', ''),
                        'Jurisdiction': country_data.get('jurisdiction', ''),
                        'Issuing_Body': country_data.get('issuing_body', ''),
                        'Tag': country_data.get('tag', ''),
                        'Regulation_Name': country_data.get('regulation_name', ''),
                        'Publication_Date': country_data.get('publication_date', ''),
                        'Regulation_Status': country_data.get('regulation_status', ''),
                        'Summary': str(country_data.get('summary', ''))[:500] + '...' if len(str(country_data.get('summary', ''))) > 500 else str(country_data.get('summary', '')),
                        'Applicability': str(country_data.get('applicability', ''))[:300] + '...' if len(str(country_data.get('applicability', ''))) > 300 else str(country_data.get('applicability', '')),
                        'Scoping_Threshold': country_data.get('scoping_threshold', ''),
                        'Effective_Date': country_data.get('effective_date', ''),
                        'Timeline_Details': country_data.get('timeline_details', ''),
                        'Financial_Integration': country_data.get('financial_integration', ''),
                        'Filing_Mechanism': country_data.get('filing_mechanism', ''),
                        'Reporting_Frequency': country_data.get('reporting_frequency', ''),
                        'Assurance_Requirement': country_data.get('assurance_requirement', ''),
                        'Penalties': country_data.get('penalties', ''),
                        'Full_Text_Link': country_data.get('full_text_link', ''),
                        'Translated_Flag': country_data.get('translated_flag', False),
                        'Source_URL': country_data.get('source_url', ''),
                        'Last_Scraped': country_data.get('last_scraped', ''),
                        'Change_Detected': country_data.get('change_detected', False),
                        
                        # Processing metadata
                        'Processing_Date': country_data.get('processing_metadata', {}).get('processing_date', ''),
                        'Source_Documents_Total': country_data.get('processing_metadata', {}).get('source_documents_total', 0),
                        'ESG_Documents_Analyzed': country_data.get('processing_metadata', {}).get('esg_documents_analyzed', 0),
                        'AI_Model_Used': country_data.get('processing_metadata', {}).get('ai_model_used', ''),
                        'Processing_Source': country_data.get('processing_metadata', {}).get('processing_source', ''),
                        'Optimization': country_data.get('processing_metadata', {}).get('optimization', ''),
                        'Thread_ID': country_data.get('processing_metadata', {}).get('thread_id', ''),
                    }
                    
                    excel_data.append(row)
                    
                except Exception as e:
                    print(f"  Warning: Could not load detailed data for {country}: {e}")
            else:
                print(f"  Warning: Country file not found for {country}")
        
        if not excel_data:
            print("  No data available for Excel export")
            return None
        
        # Create DataFrame and Excel file
        df = pd.DataFrame(excel_data)
        
        # Sort by row number first, then by country name
        df = df.sort_values(['Row_Number', 'Country'], ascending=[True, True])
        
        # Create Excel file
        excel_filename = output_dir / "consolidated_regulations_v3_threaded.xlsx"
        
        with pd.ExcelWriter(excel_filename, engine='openpyxl') as writer:
            # Main summary sheet
            df.to_excel(writer, sheet_name='Consolidated_Regulations', index=False)
            
            # Auto-adjust column widths
            workbook = writer.book
            worksheet = writer.sheets['Consolidated_Regulations']
            
            for column in worksheet.columns:
                max_length = 0
                column_letter = column[0].column_letter
                for cell in column:
                    try:
                        if len(str(cell.value)) > max_length:
                            max_length = len(str(cell.value))
                    except:
                        pass
                adjusted_width = min(max_length + 2, 50)
                worksheet.column_dimensions[column_letter].width = adjusted_width
            
            # Create statistics sheet
            stats_data = {
                'Metric': [
                    'Total Countries Processed',
                    'Successful AI Consolidations', 
                    'Failed Consolidations',
                    'Countries with Row Numbers',
                    'Average ESG Documents per Country',
                    'Processing Source',
                    'Optimization Version',
                    'Threading',
                    'Processing Date'
                ],
                'Value': [
                    len(excel_data),
                    results['consolidation_metadata']['successful_consolidations'],
                    results['consolidation_metadata']['failed_consolidations'],
                    len(df[df['Row_Number'] > 0]) if 'Row_Number' in df.columns else 'N/A',
                    f"{df['ESG_Documents_Analyzed'].mean():.1f}" if 'ESG_Documents_Analyzed' in df.columns else 'N/A',
                    'esg_relevant_by_country',
                    'v3_esg_pre_filtered_multithreaded',
                    'Enabled',
                    results['consolidation_metadata']['processing_date']
                ]
            }
            
            stats_df = pd.DataFrame(stats_data)
            stats_df.to_excel(writer, sheet_name='Statistics', index=False)
            
            # Auto-adjust stats sheet
            stats_worksheet = writer.sheets['Statistics']
            for column in stats_worksheet.columns:
                max_length = 0
                column_letter = column[0].column_letter
                for cell in column:
                    try:
                        if len(str(cell.value)) > max_length:
                            max_length = len(str(cell.value))
                    except:
                        pass
                adjusted_width = max_length + 2
                stats_worksheet.column_dimensions[column_letter].width = adjusted_width
        
        print(f"  ✅ Excel file created: {excel_filename}")
        print(f"  📊 Included {len(excel_data)} countries with ConsolidatedRegulation format")
        print(f"  🚀 Optimized processing: ESG documents pre-filtered + multithreaded")
        
        return str(excel_filename)
        
    except ImportError:
        print("  ❌ pandas and openpyxl required for Excel export. Install with: pip install pandas openpyxl")
        return None
    except Exception as e:
        print(f"  ❌ Error creating Excel file: {e}")
        return None


def main():
    """Main function for AI-powered metadata consolidation"""
    parser = argparse.ArgumentParser(
        description="AI-Powered Metadata Results Consolidator V3 - Multithreaded (Optimized)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This is STEP 3 of the regulation processing pipeline (V3 - Multithreaded):
1. regulation_pipeline.py - Scrapes regulations
2. file_metadata_processor.py - Analyzes documents  
3. ai_consolidate_metadata_v3_threaded.py - AI consolidates using ConsolidatedRegulation format (ESG-optimized + multithreaded)

V3 MULTITHREADED OPTIMIZATIONS:
- Processes only ESG-relevant documents from esg_relevant_by_country folder
- Parallel processing of countries using ThreadPoolExecutor
- Thread-safe progress tracking and result collection
- Significantly faster processing by combining ESG filtering + multithreading
- Maintains same ConsolidatedRegulation output format with row numbers

Examples:
  python ai_consolidate_metadata_v3_threaded.py
  python ai_consolidate_metadata_v3_threaded.py --input file_metadata_analysis_results --output ai_consolidated_summaries_v3_threaded
  python ai_consolidate_metadata_v3_threaded.py --model gpt-4 --temperature 0.2 --max-workers 8

Output:
  - Individual country JSON files with AI-consolidated regulatory summaries in ConsolidatedRegulation format
  - Excel file with all countries as rows (row number as first column) and key regulatory data as columns
  - Global consolidation summary with statistics
        """
    )
    
    parser.add_argument("--input", "-i", default="file_metadata_analysis_results",
                       help="Input directory from file_metadata_processor.py")
    parser.add_argument("--output", "-o", default="ai_consolidated_summaries_v3_threaded",
                       help="Output directory for AI-consolidated summaries")
    parser.add_argument("--model", default="gpt-4o-mini",
                       help="OpenAI model to use (default: gpt-4o-mini)")
    parser.add_argument("--temperature", type=float, default=0.1,
                       help="AI temperature setting (default: 0.1)")
    parser.add_argument("--max-context", type=int, default=15000,
                       help="Maximum context length for AI processing (default: 15000)")
    parser.add_argument("--max-workers", type=int, default=4,
                       help="Maximum number of worker threads (default: 4)")
    
    args = parser.parse_args()
    
    # Check for required environment variables
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Error: OPENAI_API_KEY environment variable not set")
        print("Set it with: export OPENAI_API_KEY='your-api-key'")
        return False
    
    input_dir = Path(args.input)
    esg_dir = input_dir / "esg_relevant_by_country"
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True)
    
    if not input_dir.exists():
        print(f"❌ Input directory not found: {input_dir}")
        print("Run file_metadata_processor.py first!")
        return False
    
    if not esg_dir.exists():
        print(f"❌ ESG directory not found: {esg_dir}")
        print("Expected esg_relevant_by_country folder in input directory")
        return False
    
    print("🚀 AI-POWERED METADATA CONSOLIDATION V3 - MULTITHREADED")
    print("=" * 70)
    print(f"📂 Input: {input_dir}")
    print(f"📂 ESG Source: {esg_dir}")
    print(f"📁 Output: {output_dir}")
    print(f"🤖 AI Model: {args.model}")
    print(f"🌡️  Temperature: {args.temperature}")
    print(f"🧵 Max Workers: {args.max_workers}")
    print(f"📊 Output Format: ConsolidatedRegulation with row numbers")
    print(f"🚀 Optimization: ESG documents pre-filtered + multithreaded")
    print()
    
    try:
        # Initialize AI consolidator
        consolidator = AIMetadataConsolidator(args.model, args.temperature)
        consolidator.max_context_length = args.max_context
        
        # Find all country folders
        country_folders = [d for d in esg_dir.iterdir() if d.is_dir()]
        
        if not country_folders:
            print(f"❌ No country folders found in {esg_dir}")
            return False
        
        print(f"Found {len(country_folders)} countries to process from ESG-relevant data")
        
        # Initialize progress tracker
        progress = ThreadSafeProgress()
        progress.set_total(len(country_folders))
        
        # Show folder mapping
        print("\n📋 Folder → Excel Row Mapping:")
        for folder in country_folders:
            row_idx = consolidator.extract_row_index_from_country(folder.name)
            if row_idx:
                clean_country = re.sub(r'^Row\d+_', '', folder.name)
                print(f"  {folder.name} → Excel row {row_idx} ({clean_country})")
            else:
                print(f"  {folder.name} → No row number detected")
        
        # Process countries in parallel
        print(f"\n🧵 Starting multithreaded processing with {args.max_workers} workers...")
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            # Submit all tasks
            futures = [
                executor.submit(process_single_country, consolidator, country_folder, output_dir, progress)
                for country_folder in country_folders
            ]
            
            # Wait for completion
            for future in as_completed(futures):
                try:
                    future.result()  # This will raise any exception that occurred
                except Exception as e:
                    print(f"  💥 Thread error: {e}")
        
        processing_time = time.time() - start_time
        final_results = progress.get_results()
        
        # Create results structure
        results = {
            'consolidation_metadata': {
                'processing_date': datetime.now().isoformat(),
                'ai_model': args.model,
                'temperature': args.temperature,
                'output_format': 'ConsolidatedRegulation',
                'optimization': 'v3_esg_pre_filtered_multithreaded',
                'processing_source': 'esg_relevant_by_country',
                'max_workers': args.max_workers,
                'processing_time_seconds': processing_time,
                'countries_processed': final_results['processed'],
                'successful_consolidations': final_results['successful'],
                'failed_consolidations': final_results['failed']
            },
            'country_summaries': final_results['results']
        }
        
        # Save consolidated results summary
        summary_file = output_dir / "consolidation_summary_v3_threaded.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        # Create Excel summary
        excel_file = create_countries_excel_summary(results, output_dir)
        
        # Print final summary
        metadata = results['consolidation_metadata']
        print(f"\n🎉 AI CONSOLIDATION V3 MULTITHREADED COMPLETED")
        print("=" * 70)
        print(f"📊 Countries processed: {metadata['countries_processed']}")
        print(f"✅ Successful consolidations: {metadata['successful_consolidations']}")
        print(f"❌ Failed consolidations: {metadata['failed_consolidations']}")
        print(f"⏱️  Processing time: {processing_time:.1f} seconds")
        print(f"🧵 Workers used: {args.max_workers}")
        print(f"⚡ Speed: {metadata['countries_processed']/processing_time:.1f} countries/second")
        print(f"🚀 Optimization: ESG documents pre-filtered + multithreaded")
        print(f"📁 Output directory: {output_dir}")
        print(f"📋 Summary file: {summary_file}")
        if excel_file:
            print(f"📊 Excel summary (with row numbers): {excel_file}")
        
        return True
        
    except Exception as e:
        print(f"💥 Critical error: {e}")
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)