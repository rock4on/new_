#!/usr/bin/env python3
"""
AI-Powered Metadata Results Consolidator
STEP 3 of the pipeline: Uses LLM to intelligently consolidate metadata analysis results
into structured regulatory summaries, similar to consolidate_regulations.py
"""

import json
import os
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any
import argparse
import pandas as pd
import re

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


class ConsolidatedRegulationSummary(BaseModel):
    """Pydantic model for AI-consolidated regulation summary"""
    unique_id: str = Field(description="Unique identifier for the consolidated regulation")
    country: str = Field(description="Country or jurisdiction")
    regulation_name: str = Field(description="Primary or consolidated regulation name")
    issuing_body: str = Field(description="Primary regulatory authority")
    regulation_type: str = Field(description="Type of regulation (ESG, Financial, Environmental, etc.)")
    publication_date: str = Field(description="Publication or effective date")
    regulation_status: str = Field(description="Current status (Active, Draft, Proposed, etc.)")
    
    # Core regulatory content
    executive_summary: str = Field(description="Comprehensive executive summary of the regulation")
    key_requirements: List[str] = Field(description="List of key regulatory requirements")
    scope_and_applicability: str = Field(description="Who and what the regulation applies to")
    compliance_thresholds: str = Field(description="Minimum thresholds for compliance")
    effective_dates: str = Field(description="Key implementation and compliance dates")
    
    # ESG-specific fields
    esg_focus_areas: List[str] = Field(description="Primary ESG focus areas (Environmental, Social, Governance)")
    disclosure_requirements: str = Field(description="Specific disclosure and reporting requirements")
    reporting_frequency: str = Field(description="Required reporting frequency and deadlines")
    assurance_requirements: str = Field(description="Third-party assurance or verification requirements")
    
    # Implementation details
    penalties_and_enforcement: str = Field(description="Non-compliance penalties and enforcement mechanisms")
    filing_mechanisms: str = Field(description="How and where to file required reports")
    financial_integration: str = Field(description="Integration with financial reporting requirements")
    
    # Analysis metadata
    document_sources: int = Field(description="Number of source documents analyzed")
    total_pages_analyzed: int = Field(description="Approximate total pages of content analyzed")
    confidence_score: float = Field(description="AI confidence score for the analysis (0-1)")
    key_gaps_identified: List[str] = Field(description="Areas where more information may be needed")
    
    # Source tracking
    primary_source_urls: List[str] = Field(description="Primary source URLs for the regulation")
    last_updated: str = Field(description="When this analysis was generated")
    change_indicators: List[str] = Field(description="Indicators of recent changes or updates")


class AIMetadataConsolidator:
    """Uses AI to consolidate metadata analysis results into structured regulation summaries"""
    
    def __init__(self, model_name: str = "gpt-4o-mini", temperature: float = 0.1):
        if not LANGCHAIN_AVAILABLE:
            raise ImportError("LangChain is required for AI consolidation. Install with: pip install langchain langchain-openai")
        
        self.llm = ChatOpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
            temperature=temperature,
            model=model_name
        )
        
        self.parser = JsonOutputParser(pydantic_object=ConsolidatedRegulationSummary)
        
        self.consolidation_prompt = ChatPromptTemplate.from_template("""
You are an expert regulatory analyst specializing in ESG and financial regulations. Analyze the following metadata analysis results from multiple documents for a specific regulation/country and create a comprehensive, structured regulatory summary.

COUNTRY: {country}
REGULATION CONTEXT: {regulation_context}

DOCUMENT ANALYSIS RESULTS:
{document_analyses}

INSTRUCTIONS:
1. Synthesize information from ALL document analyses to create ONE comprehensive regulatory summary
2. Focus on ESG-relevant content and regulatory requirements
3. Extract key requirements, dates, thresholds, and compliance obligations
4. Identify the primary regulation name and issuing authority
5. Create a clear executive summary that captures the essence of the regulatory framework
6. List specific disclosure and reporting requirements
7. Identify any gaps where information might be incomplete
8. Assign a confidence score based on the quality and completeness of source material
9. Create a unique ID in format: {country_code}_ESG_{year}_v1

ANALYSIS FOCUS:
- Prioritize ESG-relevant requirements and disclosures
- Extract specific dates, thresholds, and quantitative requirements
- Identify who must comply and when
- Summarize reporting frequencies and filing mechanisms
- Note enforcement mechanisms and penalties

{format_instructions}
""")
        
        self.chain = self.consolidation_prompt | self.llm | self.parser
    
    def extract_row_index_from_country(self, country_folder_name: str) -> int:
        """Extract Excel row index from country folder name like 'Row5_Country_Name'"""
        try:
            match = re.match(r'Row(\d+)_', country_folder_name)
            return int(match.group(1)) if match else None
        except:
            return None
    
    def load_country_data(self, country_folder: Path) -> Dict[str, Any]:
        """Load all metadata analysis data for a country"""
        country_folder_name = country_folder.name
        # Extract clean country name (remove row prefix if present)
        country_name = re.sub(r'^Row\d+_', '', country_folder_name)
        print(f"📂 Loading data for: {country_name} (folder: {country_folder_name})")
        
        # Extract Excel row index from country folder name
        excel_row_index = self.extract_row_index_from_country(country_folder_name)
        
        # Load country summary  
        country_summary_file = country_folder / f"{country_name}_summary.json"
        country_summary = {}
        if country_summary_file.exists():
            with open(country_summary_file, 'r', encoding='utf-8') as f:
                country_summary = json.load(f)
        
        # Collect all document analyses from all regulations
        all_documents = []
        regulation_contexts = []
        
        regulation_folders = [d for d in country_folder.iterdir() if d.is_dir()]
        
        for reg_folder in regulation_folders:
            regulation_name = reg_folder.name
            
            # Load regulation summary
            reg_summary_file = reg_folder / "regulation_summary.json"
            if reg_summary_file.exists():
                with open(reg_summary_file, 'r', encoding='utf-8') as f:
                    reg_summary = json.load(f)
                    regulation_contexts.append({
                        'name': regulation_name,
                        'summary': reg_summary,
                        'excel_row_index': excel_row_index  # Use country-level row index
                    })
            
            # Load all document analyses
            analysis_files = list(reg_folder.glob("*_analysis.json"))
            for analysis_file in analysis_files:
                try:
                    with open(analysis_file, 'r', encoding='utf-8') as f:
                        doc_analysis = json.load(f)
                        doc_analysis['regulation_folder'] = regulation_name
                        doc_analysis['excel_row_index'] = excel_row_index  # Use country-level row index
                        all_documents.append(doc_analysis)
                except Exception as e:
                    print(f"  Warning: Could not load {analysis_file}: {e}")
        
        return {
            'country': country_name,
            'country_folder_name': country_folder_name,
            'excel_row_index': excel_row_index,  # Single row index for the country
            'country_summary': country_summary,
            'regulation_contexts': regulation_contexts,
            'all_documents': all_documents,
            'esg_documents': [d for d in all_documents if d.get('esg_relevant', False)]
        }
    
    def prepare_analysis_text(self, country_data: Dict[str, Any], max_context_length: int = 15000) -> Dict[str, str]:
        """Prepare text for AI analysis with context length management"""
        esg_documents = country_data['esg_documents']
        
        # Create regulation context summary
        regulation_context = f"Country: {country_data['country']}\n"
        regulation_context += f"Total Regulations: {len(country_data['regulation_contexts'])}\n"
        regulation_context += f"ESG Relevant Documents: {len(esg_documents)}\n\n"
        
        for reg_ctx in country_data['regulation_contexts']:
            regulation_context += f"Regulation: {reg_ctx['name']}\n"
            if reg_ctx.get('summary'):
                summary_text = str(reg_ctx['summary'])[:500]  # Limit regulation summary length
                regulation_context += f"Summary: {summary_text}...\n" if len(str(reg_ctx['summary'])) > 500 else f"Summary: {summary_text}\n"
            regulation_context += "\n"
        
        # Sort ESG documents by score (highest first) and limit count
        sorted_docs = sorted(esg_documents, key=lambda x: x.get('esg_match_score', 0), reverse=True)
        
        # Start with top documents and add until we approach context limit
        document_analyses = ""
        current_length = len(regulation_context)
        
        for i, doc in enumerate(sorted_docs, 1):
            doc_section = f"\n--- DOCUMENT {i} ---\n"
            doc_section += f"File: {doc.get('file_name', 'Unknown')}\n"
            doc_section += f"Type: {doc.get('file_type', 'Unknown')}\n"
            doc_section += f"ESG Score: {doc.get('esg_match_score', 0)}\n"
            doc_section += f"Regulation: {doc.get('regulation_folder', 'Unknown')}\n"
            doc_section += f"Source URL: {doc.get('source_url', 'Unknown')}\n"
            
            # Include metadata if available (but limit size)
            metadata = doc.get('metadata')
            if metadata and isinstance(metadata, dict):
                # Only include key metadata fields to save space
                key_metadata = {
                    'summary': str(metadata.get('summary', ''))[:300] + '...' if len(str(metadata.get('summary', ''))) > 300 else metadata.get('summary', ''),
                    'key_requirements': metadata.get('key_requirements', [])[:3],  # Only first 3 requirements
                    'effective_date': metadata.get('effective_date', ''),
                    'issuing_body': metadata.get('issuing_body', '')
                }
                doc_section += f"Key Metadata: {json.dumps(key_metadata, indent=1)}\n"
            
            # Include text sample (adaptive length based on remaining context)
            extracted_text = doc.get('extracted_text', '')
            if extracted_text:
                # Calculate how much text we can include
                remaining_context = max_context_length - current_length - len(doc_section)
                if remaining_context > 500:  # Only include text if we have reasonable space
                    max_text_length = min(1500, remaining_context - 200)  # Reserve some space
                    text_sample = extracted_text[:max_text_length]
                    if len(extracted_text) > max_text_length:
                        text_sample += "... [truncated]"
                    doc_section += f"Content Sample: {text_sample}\n"
            
            doc_section += "\n"
            
            # Check if adding this document would exceed context limit
            if current_length + len(doc_section) > max_context_length:
                if i > 5:  # Ensure we include at least 5 documents
                    print(f"  📏 Context limit reached. Including top {i-1} documents out of {len(esg_documents)} ESG documents")
                    break
                else:
                    # If we're still in first 5 documents, reduce text sample size more aggressively
                    extracted_text = doc.get('extracted_text', '')
                    if extracted_text:
                        max_text_length = 800  # Very short sample for critical documents
                        text_sample = extracted_text[:max_text_length] + "... [truncated for context]"
                        # Rebuild doc_section with shorter text
                        doc_section = doc_section.split("Content Sample:")[0]
                        doc_section += f"Content Sample: {text_sample}\n\n"
            
            document_analyses += doc_section
            current_length += len(doc_section)
            
            # Safety limit - never include more than 25 documents
            if i >= 25:
                print(f"  📊 Document limit reached. Including top 25 documents out of {len(esg_documents)} ESG documents")
                break
        
        total_length = len(regulation_context + document_analyses)
        print(f"  📏 Prepared context: {total_length:,} characters ({len(esg_documents)} ESG docs available, using top documents)")
        
        return {
            'regulation_context': regulation_context,
            'document_analyses': document_analyses
        }
    
    def consolidate_country_with_ai(self, country_data: Dict[str, Any]) -> Dict[str, Any]:
        """Use AI to consolidate country metadata into structured regulation summary"""
        country = country_data['country']
        esg_count = len(country_data['esg_documents'])
        
        print(f"🤖 AI consolidating {country}: {esg_count} ESG documents")
        
        if esg_count == 0:
            print(f"  ⏭️  No ESG relevant documents found for {country}")
            return None
        
        try:
            # Prepare analysis text with context length management
            analysis_data = self.prepare_analysis_text(country_data, getattr(self, 'max_context_length', 15000))
            
            # Generate country code for unique ID
            country_code = country[:3].upper()
            
            # Use AI to create consolidated summary
            result = self.chain.invoke({
                "country": country,
                "country_code": country_code,
                "year": datetime.now().year,
                "regulation_context": analysis_data['regulation_context'],
                "document_analyses": analysis_data['document_analyses'],
                "format_instructions": self.parser.get_format_instructions()
            })
            
            # Add processing metadata
            result['processing_metadata'] = {
                'source_documents_total': len(country_data['all_documents']),
                'esg_documents_analyzed': len(country_data['esg_documents']),
                'regulations_covered': len(country_data['regulation_contexts']),
                'processing_date': datetime.now().isoformat(),
                'ai_model_used': self.llm.model_name,
                'source_country_folder': country,
                'excel_row_index': country_data.get('excel_row_index'),  # Single row index for country
                'country_folder_name': country_data.get('country_folder_name')
            }
            
            return result
            
        except Exception as e:
            print(f"  ❌ AI consolidation failed for {country}: {e}")
            return self.create_fallback_summary(country_data)
    
    def create_fallback_summary(self, country_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create basic summary if AI fails"""
        country = country_data['country']
        esg_docs = country_data['esg_documents']
        
        return {
            "unique_id": f"{country[:3].upper()}_ESG_{datetime.now().year}_FALLBACK",
            "country": country,
            "regulation_name": f"{country} ESG Regulatory Framework",
            "issuing_body": "Multiple Authorities",
            "regulation_type": "ESG/Sustainability",
            "publication_date": str(datetime.now().year),
            "regulation_status": "Analysis Generated",
            "executive_summary": f"Automated analysis of {len(esg_docs)} ESG-relevant documents from {country}",
            "key_requirements": ["Manual review required"],
            "scope_and_applicability": "To be determined through manual review",
            "compliance_thresholds": "Various thresholds identified",
            "effective_dates": "Multiple dates identified",
            "esg_focus_areas": ["Environmental", "Social", "Governance"],
            "disclosure_requirements": "Multiple disclosure frameworks identified",
            "reporting_frequency": "Various frequencies required",
            "assurance_requirements": "To be determined",
            "penalties_and_enforcement": "Various enforcement mechanisms",
            "filing_mechanisms": "Multiple filing requirements",
            "financial_integration": "Integration requirements vary",
            "document_sources": len(esg_docs),
            "total_pages_analyzed": sum(doc.get('word_count', 0) for doc in esg_docs) // 250,  # Rough page estimate
            "confidence_score": 0.3,  # Low confidence for fallback
            "key_gaps_identified": ["AI processing failed", "Manual review required"],
            "primary_source_urls": list(set(doc.get('source_url', '') for doc in esg_docs if doc.get('source_url'))),
            "last_updated": datetime.now().isoformat(),
            "change_indicators": ["Automated analysis"],
            "processing_metadata": {
                "fallback_reason": "AI consolidation failed",
                "source_documents_total": len(country_data['all_documents']),
                "esg_documents_analyzed": len(esg_docs),
                "processing_date": datetime.now().isoformat(),
                "excel_row_index": country_data.get('excel_row_index'),  # IMPORTANT: Include row index
                "country_folder_name": country_data.get('country_folder_name')  # IMPORTANT: Include folder name
            }
        }


def merge_with_original_excel(results: Dict[str, Any], original_excel_path: str, output_dir: Path) -> str:
    """Merge AI results with original Excel using row indices"""
    try:
        print(f"\n📊 Merging AI results with original Excel: {original_excel_path}")
        
        # Load original Excel
        df_original = pd.read_excel(original_excel_path)
        print(f"  📋 Original Excel: {len(df_original)} rows")
        
        # Create AI data mapping by row index - direct mapping from folder names
        ai_data_by_row = {}
        print(f"  🔍 Processing {len(results['country_summaries'])} countries for row mapping")
        
        for country, country_info in results['country_summaries'].items():
            country_file = Path(country_info['file'])
            print(f"  📂 Processing {country} from file: {country_file.name}")
            
            if country_file.exists():
                try:
                    with open(country_file, 'r', encoding='utf-8') as f:
                        ai_data = json.load(f)
                    
                    # Extract row number from the country folder name in metadata
                    country_folder_name = ai_data.get('processing_metadata', {}).get('country_folder_name', '')
                    row_number = extract_row_index_from_country_name(country_folder_name)
                    
                    if row_number:
                        ai_data_by_row[row_number] = ai_data
                        print(f"    ✅ Mapped row {row_number} to {country} AI data")
                    else:
                        print(f"    ❌ Could not extract row number from folder: {country_folder_name}")
                        
                except Exception as e:
                    print(f"    ❌ Error loading {country}: {e}")
            else:
                print(f"    ❌ File not found: {country_file}")
        
        print(f"  📊 AI data mapped for rows: {sorted(ai_data_by_row.keys())}")
        
        # Start with the original Excel data
        df_combined = df_original.copy()
        
        # Add AI columns to the DataFrame
        ai_columns = [
            'AI_Unique_ID', 'AI_Regulation_Type', 'AI_Issuing_Body', 'AI_Publication_Date',
            'AI_Regulation_Status', 'AI_Effective_Dates', 'AI_ESG_Focus_Areas', 'AI_Reporting_Frequency',
            'AI_Document_Sources', 'AI_Confidence_Score', 'AI_Filing_Mechanisms', 'AI_Assurance_Requirements',
            'AI_Financial_Integration', 'AI_Executive_Summary', 'AI_Top_3_Key_Requirements', 
            'AI_Scope_and_Applicability', 'AI_Processing_Status', 'AI_Processing_Date', 'AI_Model_Used',
            'AI_Total_Pages_Analyzed', 'AI_Key_Gaps_Identified'
        ]
        
        # Initialize all AI columns with default values
        for col in ai_columns:
            df_combined[col] = 'No ESG validated file found' if col == 'AI_Processing_Status' else ''
        
        # Fill in AI data for rows that have it
        for row_number, ai_data in ai_data_by_row.items():
            # Convert Excel row number to pandas index (row 2 becomes index 0, row 3 becomes index 1, etc.)
            pandas_index = row_number - 2
            
            if 0 <= pandas_index < len(df_combined):
                print(f"  📝 Adding AI data to Excel row {row_number} (pandas index {pandas_index})")
                
                # Set AI column values for this row
                df_combined.loc[pandas_index, 'AI_Unique_ID'] = ai_data.get('unique_id', '')
                df_combined.loc[pandas_index, 'AI_Regulation_Type'] = ai_data.get('regulation_type', '')
                df_combined.loc[pandas_index, 'AI_Issuing_Body'] = ai_data.get('issuing_body', '')
                df_combined.loc[pandas_index, 'AI_Publication_Date'] = ai_data.get('publication_date', '')
                df_combined.loc[pandas_index, 'AI_Regulation_Status'] = ai_data.get('regulation_status', '')
                df_combined.loc[pandas_index, 'AI_Effective_Dates'] = ai_data.get('effective_dates', '')
                
                esg_areas = ai_data.get('esg_focus_areas', [])
                df_combined.loc[pandas_index, 'AI_ESG_Focus_Areas'] = ', '.join(esg_areas) if isinstance(esg_areas, list) else str(esg_areas)
                
                df_combined.loc[pandas_index, 'AI_Reporting_Frequency'] = ai_data.get('reporting_frequency', '')
                df_combined.loc[pandas_index, 'AI_Document_Sources'] = ai_data.get('document_sources', 0)
                df_combined.loc[pandas_index, 'AI_Confidence_Score'] = ai_data.get('confidence_score', 0)
                df_combined.loc[pandas_index, 'AI_Filing_Mechanisms'] = ai_data.get('filing_mechanisms', '')
                df_combined.loc[pandas_index, 'AI_Assurance_Requirements'] = ai_data.get('assurance_requirements', '')
                df_combined.loc[pandas_index, 'AI_Financial_Integration'] = ai_data.get('financial_integration', '')
                
                exec_summary = str(ai_data.get('executive_summary', ''))
                df_combined.loc[pandas_index, 'AI_Executive_Summary'] = exec_summary[:300] + '...' if len(exec_summary) > 300 else exec_summary
                
                key_reqs = ai_data.get('key_requirements', [])
                df_combined.loc[pandas_index, 'AI_Top_3_Key_Requirements'] = ', '.join(key_reqs[:3]) if isinstance(key_reqs, list) else ''
                
                scope = str(ai_data.get('scope_and_applicability', ''))
                df_combined.loc[pandas_index, 'AI_Scope_and_Applicability'] = scope[:200] + '...' if len(scope) > 200 else scope
                
                df_combined.loc[pandas_index, 'AI_Processing_Status'] = 'AI Processed'
                df_combined.loc[pandas_index, 'AI_Processing_Date'] = ai_data.get('processing_metadata', {}).get('processing_date', '')
                df_combined.loc[pandas_index, 'AI_Model_Used'] = ai_data.get('processing_metadata', {}).get('ai_model_used', '')
                df_combined.loc[pandas_index, 'AI_Total_Pages_Analyzed'] = ai_data.get('total_pages_analyzed', 0)
                
                gaps = ai_data.get('key_gaps_identified', [])
                df_combined.loc[pandas_index, 'AI_Key_Gaps_Identified'] = ', '.join(gaps) if isinstance(gaps, list) else str(gaps)
                
            else:
                print(f"    ⚠️  Row {row_number} is outside DataFrame range (pandas index {pandas_index})")
        
        # Convert back to list of dictionaries for statistics
        combined_data = df_combined.to_dict('records')
        # Count processed rows for statistics
        ai_processed_count = len([row for row in combined_data if row.get('AI_Processing_Status') == 'AI Processed'])
        print(f"  ✅ Successfully added AI data to {ai_processed_count} rows")
        
        # Create Excel with original + AI columns
        excel_filename = output_dir / "combined_original_and_ai_analysis.xlsx"
        
        with pd.ExcelWriter(excel_filename, engine='openpyxl') as writer:
            # Main combined sheet
            df_combined.to_excel(writer, sheet_name='Combined_Analysis', index=False)
            
            # Statistics sheet
            processed_count = len([r for r in combined_data if r.get('AI_Processing_Status') == 'AI Processed'])
            no_esg_count = len([r for r in combined_data if r.get('AI_Processing_Status') == 'No ESG validated file found'])
            avg_confidence = sum(r.get('AI_Confidence_Score', 0) for r in combined_data if r.get('AI_Confidence_Score', 0) > 0) / max(processed_count, 1)
            
            stats_data = {
                'Metric': [
                    'Total Regulations in Original Excel',
                    'Regulations with AI Analysis',
                    'Regulations with No ESG Validated Files',
                    'AI Processing Coverage Rate',
                    'Average AI Confidence Score',
                    'High Confidence Regulations (>0.7)',
                    'Medium Confidence Regulations (0.4-0.7)',
                    'Low Confidence Regulations (<0.4)',
                    'Processing Date'
                ],
                'Value': [
                    len(df_combined),
                    processed_count,
                    no_esg_count,
                    f"{(processed_count/len(df_combined)*100):.1f}%",
                    f"{avg_confidence:.3f}",
                    len([r for r in combined_data if r.get('AI_Confidence_Score', 0) > 0.7]),
                    len([r for r in combined_data if 0.4 <= r.get('AI_Confidence_Score', 0) <= 0.7]),
                    len([r for r in combined_data if 0 < r.get('AI_Confidence_Score', 0) < 0.4]),
                    datetime.now().isoformat()
                ]
            }
            
            stats_df = pd.DataFrame(stats_data)
            stats_df.to_excel(writer, sheet_name='Coverage_Statistics', index=False)
            
            # Auto-adjust column widths for both sheets
            for sheet_name in writer.sheets:
                worksheet = writer.sheets[sheet_name]
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
        
        print(f"  ✅ Combined Excel created: {excel_filename}")
        print(f"  📊 {len(df_combined)} total regulations, {processed_count} with AI analysis ({(processed_count/len(df_combined)*100):.1f}% coverage)")
        
        return str(excel_filename)
        
    except Exception as e:
        print(f"  ❌ Error creating combined Excel: {e}")
        return None


def extract_row_index_from_country_name(country_folder_name: str) -> int:
    """Helper function to extract row index from country folder name"""
    try:
        match = re.match(r'Row(\d+)_', country_folder_name)
        if match:
            row_num = int(match.group(1))
            print(f"🔍 DEBUG: Extracted row {row_num} from folder '{country_folder_name}'")
            return row_num
        else:
            print(f"🔍 DEBUG: No row pattern found in folder '{country_folder_name}'")
            return None
    except Exception as e:
        print(f"🔍 DEBUG: Error extracting row from '{country_folder_name}': {e}")
        return None


def create_countries_excel_summary(results: Dict[str, Any], output_dir: Path) -> str:
    """Create Excel file with all countries as rows and key information as columns"""
    try:
        print("\n📊 Creating Excel summary of all countries...")
        
        # Prepare data for Excel
        excel_data = []
        
        for country, country_info in results['country_summaries'].items():
            # Load the detailed country data to extract more information
            country_file = Path(country_info['file'])
            if country_file.exists():
                try:
                    with open(country_file, 'r', encoding='utf-8') as f:
                        country_data = json.load(f)
                    
                    # Extract key information for Excel row
                    row = {
                        'Country': country,
                        'Unique_ID': country_data.get('unique_id', ''),
                        'Regulation_Name': country_data.get('regulation_name', ''),
                        'Issuing_Body': country_data.get('issuing_body', ''),
                        'Regulation_Type': country_data.get('regulation_type', ''),
                        'Publication_Date': country_data.get('publication_date', ''),
                        'Regulation_Status': country_data.get('regulation_status', ''),
                        'Effective_Dates': country_data.get('effective_dates', ''),
                        'ESG_Focus_Areas': ', '.join(country_data.get('esg_focus_areas', [])) if isinstance(country_data.get('esg_focus_areas'), list) else str(country_data.get('esg_focus_areas', '')),
                        'Reporting_Frequency': country_data.get('reporting_frequency', ''),
                        'Document_Sources': country_data.get('document_sources', 0),
                        'Total_Pages_Analyzed': country_data.get('total_pages_analyzed', 0),
                        'Confidence_Score': country_data.get('confidence_score', 0),
                        'Filing_Mechanisms': country_data.get('filing_mechanisms', ''),
                        'Assurance_Requirements': country_data.get('assurance_requirements', ''),
                        'Financial_Integration': country_data.get('financial_integration', ''),
                        'Primary_Source_URLs': ', '.join(country_data.get('primary_source_urls', [])[:3]) if isinstance(country_data.get('primary_source_urls'), list) else str(country_data.get('primary_source_urls', ''))[:200],
                        'Last_Updated': country_data.get('last_updated', ''),
                        'Key_Requirements_Count': len(country_data.get('key_requirements', [])) if isinstance(country_data.get('key_requirements'), list) else 0,
                        'Key_Gaps_Identified': ', '.join(country_data.get('key_gaps_identified', [])) if isinstance(country_data.get('key_gaps_identified'), list) else str(country_data.get('key_gaps_identified', '')),
                        
                        # Executive summary (truncated for Excel)
                        'Executive_Summary': str(country_data.get('executive_summary', ''))[:500] + '...' if len(str(country_data.get('executive_summary', ''))) > 500 else str(country_data.get('executive_summary', '')),
                        
                        # Key requirements (first 3 as comma-separated)
                        'Top_3_Key_Requirements': ', '.join(country_data.get('key_requirements', [])[:3]) if isinstance(country_data.get('key_requirements'), list) else str(country_data.get('key_requirements', ''))[:300],
                        
                        # Scope and applicability (truncated)
                        'Scope_and_Applicability': str(country_data.get('scope_and_applicability', ''))[:300] + '...' if len(str(country_data.get('scope_and_applicability', ''))) > 300 else str(country_data.get('scope_and_applicability', '')),
                        
                        # Processing metadata
                        'Processing_Date': country_data.get('processing_metadata', {}).get('processing_date', ''),
                        'Source_Documents_Total': country_data.get('processing_metadata', {}).get('source_documents_total', 0),
                        'ESG_Documents_Analyzed': country_data.get('processing_metadata', {}).get('esg_documents_analyzed', 0),
                        'AI_Model_Used': country_data.get('processing_metadata', {}).get('ai_model_used', ''),
                    }
                    
                    excel_data.append(row)
                    
                except Exception as e:
                    print(f"  Warning: Could not load detailed data for {country}: {e}")
                    # Create basic row with available info
                    row = {
                        'Country': country,
                        'Unique_ID': country_info.get('unique_id', ''),
                        'Regulation_Name': country_info.get('regulation_name', ''),
                        'Confidence_Score': country_info.get('confidence_score', 0),
                        'Document_Sources': country_info.get('document_sources', 0),
                        'Processing_Status': 'Data Load Error'
                    }
                    excel_data.append(row)
            else:
                print(f"  Warning: Country file not found for {country}")
        
        if not excel_data:
            print("  No data available for Excel export")
            return None
        
        # Create DataFrame and Excel file
        df = pd.DataFrame(excel_data)
        
        # Sort by confidence score (descending) and then by country name
        df = df.sort_values(['Confidence_Score', 'Country'], ascending=[False, True])
        
        # Create Excel file with formatting
        excel_filename = output_dir / "countries_regulatory_summary.xlsx"
        
        with pd.ExcelWriter(excel_filename, engine='openpyxl') as writer:
            # Main summary sheet
            df.to_excel(writer, sheet_name='Countries_Summary', index=False)
            
            # Get the workbook and worksheet
            workbook = writer.book
            worksheet = writer.sheets['Countries_Summary']
            
            # Auto-adjust column widths
            for column in worksheet.columns:
                max_length = 0
                column_letter = column[0].column_letter
                for cell in column:
                    try:
                        if len(str(cell.value)) > max_length:
                            max_length = len(str(cell.value))
                    except:
                        pass
                adjusted_width = min(max_length + 2, 50)  # Cap at 50 characters
                worksheet.column_dimensions[column_letter].width = adjusted_width
            
            # Create a statistics sheet
            stats_data = {
                'Metric': [
                    'Total Countries Processed',
                    'Successful AI Consolidations', 
                    'Failed Consolidations',
                    'Average Confidence Score',
                    'Countries with High Confidence (>0.7)',
                    'Countries with Medium Confidence (0.4-0.7)',
                    'Countries with Low Confidence (<0.4)',
                    'Total Document Sources',
                    'Average Documents per Country',
                    'Processing Date'
                ],
                'Value': [
                    len(excel_data),
                    results['consolidation_metadata']['successful_consolidations'],
                    results['consolidation_metadata']['failed_consolidations'],
                    f"{df['Confidence_Score'].mean():.3f}" if 'Confidence_Score' in df.columns else 'N/A',
                    len(df[df['Confidence_Score'] > 0.7]) if 'Confidence_Score' in df.columns else 'N/A',
                    len(df[(df['Confidence_Score'] >= 0.4) & (df['Confidence_Score'] <= 0.7)]) if 'Confidence_Score' in df.columns else 'N/A',
                    len(df[df['Confidence_Score'] < 0.4]) if 'Confidence_Score' in df.columns else 'N/A',
                    df['Document_Sources'].sum() if 'Document_Sources' in df.columns else 'N/A',
                    f"{df['Document_Sources'].mean():.1f}" if 'Document_Sources' in df.columns else 'N/A',
                    results['consolidation_metadata']['processing_date']
                ]
            }
            
            stats_df = pd.DataFrame(stats_data)
            stats_df.to_excel(writer, sheet_name='Statistics', index=False)
            
            # Auto-adjust stats sheet columns
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
        print(f"  📊 Included {len(excel_data)} countries with detailed regulatory information")
        print(f"  📋 Sheets: Countries_Summary, Statistics")
        
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
        description="AI-Powered Metadata Results Consolidator - Step 3 of Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This is STEP 3 of the regulation processing pipeline:
1. regulation_pipeline.py - Scrapes regulations
2. file_metadata_processor.py - Analyzes documents  
3. ai_consolidate_metadata.py - AI consolidates into structured summaries

Examples:
  python ai_consolidate_metadata.py
  python ai_consolidate_metadata.py --input file_metadata_analysis_results --output ai_consolidated_summaries
  python ai_consolidate_metadata.py --model gpt-4 --temperature 0.2

Output:
  - Individual country JSON files with AI-consolidated regulatory summaries
  - Excel file with all countries as rows and key regulatory data as columns
  - Global consolidation summary with statistics
        """
    )
    
    parser.add_argument("--input", "-i", default="file_metadata_analysis_results",
                       help="Input directory from file_metadata_processor.py")
    parser.add_argument("--output", "-o", default="ai_consolidated_summaries",
                       help="Output directory for AI-consolidated summaries")
    parser.add_argument("--model", default="gpt-4o-mini",
                       help="OpenAI model to use (default: gpt-4o-mini)")
    parser.add_argument("--temperature", type=float, default=0.1,
                       help="AI temperature setting (default: 0.1)")
    parser.add_argument("--max-context", type=int, default=15000,
                       help="Maximum context length for AI processing (default: 15000)")
    parser.add_argument("--original-excel", type=str,
                       help="Path to original Excel file for merging with AI results")
    
    args = parser.parse_args()
    
    # Check for required environment variables
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Error: OPENAI_API_KEY environment variable not set")
        print("Set it with: export OPENAI_API_KEY='your-api-key'")
        return False
    
    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True)
    
    if not input_dir.exists():
        print(f"❌ Input directory not found: {input_dir}")
        print("Run file_metadata_processor.py first!")
        return False
    
    print("🚀 AI-POWERED METADATA CONSOLIDATION - STEP 3")
    print("=" * 60)
    print(f"📂 Input: {input_dir}")
    print(f"📁 Output: {output_dir}")
    print(f"🤖 AI Model: {args.model}")
    print(f"🌡️  Temperature: {args.temperature}")
    print()
    
    try:
        # Initialize AI consolidator
        consolidator = AIMetadataConsolidator(args.model, args.temperature)
        consolidator.max_context_length = args.max_context
        consolidator.original_excel_path = args.original_excel  # Store for merging
        
        # Find all country folders
        country_folders = [d for d in input_dir.iterdir() if d.is_dir()]
        
        if not country_folders:
            print(f"❌ No country folders found in {input_dir}")
            return False
        
        print(f"Found {len(country_folders)} countries to process")
        
        # Show folder name → row mapping for verification
        print("\n📋 Folder → Excel Row Mapping:")
        for folder in country_folders:
            row_idx = extract_row_index_from_country_name(folder.name)
            if row_idx:
                clean_country = re.sub(r'^Row\d+_', '', folder.name)
                print(f"  {folder.name} → Excel row {row_idx} ({clean_country})")
            else:
                print(f"  {folder.name} → No row number detected")
        
        results = {
            'consolidation_metadata': {
                'processing_date': datetime.now().isoformat(),
                'ai_model': args.model,
                'temperature': args.temperature,
                'countries_processed': 0,
                'successful_consolidations': 0,
                'failed_consolidations': 0
            },
            'country_summaries': {}
        }
        
        # Process each country
        for country_folder in country_folders:
            print(f"\n📂 Processing country folder: {country_folder.name}")
            try:
                # Load country data
                country_data = consolidator.load_country_data(country_folder)
                
                # AI consolidation
                consolidated_summary = consolidator.consolidate_country_with_ai(country_data)
                
                if consolidated_summary:
                    country = country_data['country']
                    
                    # Save individual country summary
                    country_file = output_dir / f"{country}_ai_consolidated.json"
                    with open(country_file, 'w', encoding='utf-8') as f:
                        json.dump(consolidated_summary, f, indent=2, ensure_ascii=False)
                    
                    # Add to results
                    results['country_summaries'][country] = {
                        'file': str(country_file),
                        'unique_id': consolidated_summary.get('unique_id'),
                        'regulation_name': consolidated_summary.get('regulation_name'),
                        'confidence_score': consolidated_summary.get('confidence_score'),
                        'document_sources': consolidated_summary.get('document_sources')
                    }
                    
                    results['consolidation_metadata']['successful_consolidations'] += 1
                    
                    confidence = consolidated_summary.get('confidence_score', 0)
                    print(f"  ✅ {country}: {consolidated_summary.get('document_sources', 0)} docs, confidence: {confidence:.2f}")
                else:
                    results['consolidation_metadata']['failed_consolidations'] += 1
                    print(f"  ❌ {country_folder.name}: No ESG content found")
                
                results['consolidation_metadata']['countries_processed'] += 1
                
            except Exception as e:
                print(f"  💥 Error processing {country_folder.name}: {e}")
                results['consolidation_metadata']['failed_consolidations'] += 1
                continue
        
        # Save consolidated results summary
        summary_file = output_dir / "ai_consolidation_summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        # Create Excel summary of all countries
        excel_file = create_countries_excel_summary(results, output_dir)
        
        # Create combined Excel with original + AI data if original Excel provided
        combined_excel_file = None
        if hasattr(consolidator, 'original_excel_path') and consolidator.original_excel_path:
            print(f"\n🔗 STARTING EXCEL MERGE")
            print(f"  🔍 DEBUG: results['country_summaries'] has {len(results.get('country_summaries', {}))} countries")
            for country, info in results.get('country_summaries', {}).items():
                print(f"    - {country}: file={info.get('file', 'MISSING')}")
            combined_excel_file = merge_with_original_excel(results, consolidator.original_excel_path, output_dir)
        
        # Print final summary
        metadata = results['consolidation_metadata']
        print(f"\n🎉 AI CONSOLIDATION COMPLETED")
        print("=" * 60)
        print(f"📊 Countries processed: {metadata['countries_processed']}")
        print(f"✅ Successful consolidations: {metadata['successful_consolidations']}")
        print(f"❌ Failed consolidations: {metadata['failed_consolidations']}")
        print(f"📁 Output directory: {output_dir}")
        print(f"📋 Summary file: {summary_file}")
        if excel_file:
            print(f"📊 Excel summary: {excel_file}")
        if combined_excel_file:
            print(f"🔗 Combined Excel (Original + AI): {combined_excel_file}")
        
        return True
        
    except Exception as e:
        print(f"💥 Critical error: {e}")
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)