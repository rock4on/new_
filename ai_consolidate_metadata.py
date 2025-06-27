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
    
    def load_country_data(self, country_folder: Path) -> Dict[str, Any]:
        """Load all metadata analysis data for a country"""
        country_name = country_folder.name
        print(f"📂 Loading data for: {country_name}")
        
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
                        'summary': reg_summary
                    })
            
            # Load all document analyses
            analysis_files = list(reg_folder.glob("*_analysis.json"))
            for analysis_file in analysis_files:
                try:
                    with open(analysis_file, 'r', encoding='utf-8') as f:
                        doc_analysis = json.load(f)
                        doc_analysis['regulation_folder'] = regulation_name
                        all_documents.append(doc_analysis)
                except Exception as e:
                    print(f"  Warning: Could not load {analysis_file}: {e}")
        
        return {
            'country': country_name,
            'country_summary': country_summary,
            'regulation_contexts': regulation_contexts,
            'all_documents': all_documents,
            'esg_documents': [d for d in all_documents if d.get('esg_relevant', False)]
        }
    
    def prepare_analysis_text(self, country_data: Dict[str, Any]) -> Dict[str, str]:
        """Prepare text for AI analysis"""
        esg_documents = country_data['esg_documents']
        
        # Create regulation context summary
        regulation_context = f"Country: {country_data['country']}\n"
        regulation_context += f"Total Regulations: {len(country_data['regulation_contexts'])}\n"
        regulation_context += f"ESG Relevant Documents: {len(esg_documents)}\n\n"
        
        for reg_ctx in country_data['regulation_contexts']:
            regulation_context += f"Regulation: {reg_ctx['name']}\n"
            if reg_ctx.get('summary'):
                regulation_context += f"Summary: {reg_ctx['summary']}\n"
            regulation_context += "\n"
        
        # Create document analyses text (focus on ESG relevant documents)
        document_analyses = ""
        for i, doc in enumerate(esg_documents[:20], 1):  # Limit to top 20 ESG documents
            document_analyses += f"\n--- DOCUMENT {i} ---\n"
            document_analyses += f"File: {doc.get('file_name', 'Unknown')}\n"
            document_analyses += f"Type: {doc.get('file_type', 'Unknown')}\n"
            document_analyses += f"ESG Score: {doc.get('esg_match_score', 0)}\n"
            document_analyses += f"Regulation: {doc.get('regulation_folder', 'Unknown')}\n"
            document_analyses += f"Source URL: {doc.get('source_url', 'Unknown')}\n"
            
            # Include metadata if available
            if doc.get('metadata'):
                document_analyses += f"Metadata: {json.dumps(doc['metadata'], indent=2)}\n"
            
            # Include relevant excerpts from extracted text (limit length)
            extracted_text = doc.get('extracted_text', '')
            if extracted_text:
                # Take first 2000 characters as sample
                text_sample = extracted_text[:2000]
                if len(extracted_text) > 2000:
                    text_sample += "... [truncated]"
                document_analyses += f"Content Sample: {text_sample}\n"
            
            document_analyses += "\n"
        
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
            # Prepare analysis text
            analysis_data = self.prepare_analysis_text(country_data)
            
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
                'source_country_folder': country
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
                "processing_date": datetime.now().isoformat()
            }
        }


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
        
        # Find all country folders
        country_folders = [d for d in input_dir.iterdir() if d.is_dir()]
        
        if not country_folders:
            print(f"❌ No country folders found in {input_dir}")
            return False
        
        print(f"Found {len(country_folders)} countries to process")
        
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
        
        # Print final summary
        metadata = results['consolidation_metadata']
        print(f"\n🎉 AI CONSOLIDATION COMPLETED")
        print("=" * 60)
        print(f"📊 Countries processed: {metadata['countries_processed']}")
        print(f"✅ Successful consolidations: {metadata['successful_consolidations']}")
        print(f"❌ Failed consolidations: {metadata['failed_consolidations']}")
        print(f"📁 Output directory: {output_dir}")
        print(f"📋 Summary file: {summary_file}")
        
        return True
        
    except Exception as e:
        print(f"💥 Critical error: {e}")
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)