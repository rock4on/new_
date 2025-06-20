#!/usr/bin/env python3
"""
Regulation Consolidator

Processes relevant_regulations.json and uses LLM to create a single 
comprehensive regulation that summarizes all sub-regulations found.
"""

import json
import os
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field

# Configuration
DOWNLOADS_DIR = "downloads"
INPUT_FILE = "relevant_regulations.json"
OUTPUT_FILE = "consolidated_regulation.json"

class ConsolidatedRegulation(BaseModel):
    """Pydantic model for the final consolidated regulation"""
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


class RegulationConsolidator:
    def __init__(self):
        self.llm = ChatOpenAI(
            api_key=os.getenv("OPENAI_API_KEY", "your-api-key-here"),
            base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
            temperature=0.1,
            model="gpt-4o-mini"
        )
        
        self.parser = JsonOutputParser(pydantic_object=ConsolidatedRegulation)
        
        self.prompt = ChatPromptTemplate.from_template("""
You are an expert regulatory analyst. Analyze the following individual regulations and create ONE comprehensive consolidated regulation that summarizes all the sub-regulations.

INDIVIDUAL REGULATIONS TO CONSOLIDATE:
{regulations_text}

INSTRUCTIONS:
1. Create a SINGLE consolidated regulation that represents the combined regulatory landscape
2. Merge similar requirements across regulations
3. Identify the primary jurisdiction, authority, and scope
4. Create comprehensive summaries that capture all key requirements
5. For dates, use the most relevant or create ranges
6. For IDs, create a new consolidated ID in format: REGION_TOPIC_YEAR_CONSOLIDATED
7. Ensure the consolidated regulation captures the essence of all sub-regulations
8. If multiple countries/jurisdictions, indicate "Multiple" or create summary descriptions

{format_instructions}
""")
        
        self.chain = self.prompt | self.llm | self.parser

    def consolidate_regulations_with_llm(self, regulations: List[Dict]) -> Dict:
        """Use LLM to consolidate multiple regulations into one comprehensive regulation"""
        
        # Prepare regulations text for LLM
        regulations_text = ""
        for i, reg in enumerate(regulations, 1):
            regulations_text += f"\n--- REGULATION {i} ---\n"
            regulations_text += f"Name: {reg.get('regulation_name', 'Unknown')}\n"
            regulations_text += f"Country: {reg.get('country', 'Unknown')}\n"
            regulations_text += f"Authority: {reg.get('issuing_body', 'Unknown')}\n"
            regulations_text += f"Summary: {reg.get('summary', 'Unknown')}\n"
            regulations_text += f"Applicability: {reg.get('applicability', 'Unknown')}\n"
            regulations_text += f"Effective Date: {reg.get('effective_date', 'Unknown')}\n"
            regulations_text += f"Full Details: {json.dumps(reg, indent=2)}\n"
        
        # Use LLM to create consolidated regulation
        try:
            result = self.chain.invoke({
                "regulations_text": regulations_text,
                "format_instructions": self.parser.get_format_instructions()
            })
            return result
            
        except Exception as e:
            print(f"Error in LLM consolidation: {e}")
            # Fallback to manual consolidation
            return self.manual_consolidation_fallback(regulations)
    
    def manual_consolidation_fallback(self, regulations: List[Dict]) -> Dict:
        """Fallback manual consolidation if LLM fails"""
        countries = list(set(r.get('country', 'Unknown') for r in regulations))
        authorities = list(set(r.get('issuing_body', 'Unknown') for r in regulations))
        
        return {
            "unique_id": f"MULTI_{datetime.now().year}_CONSOLIDATED",
            "country": "Multiple" if len(countries) > 1 else countries[0],
            "jurisdiction": "Multi-jurisdictional" if len(countries) > 1 else "Single",
            "issuing_body": "; ".join(authorities[:3]) + ("..." if len(authorities) > 3 else ""),
            "tag": "Multiple Disclosure Areas",
            "regulation_name": f"Consolidated Regulatory Framework ({len(regulations)} regulations)",
            "publication_date": f"{datetime.now().year}",
            "regulation_status": "Consolidated",
            "summary": f"Comprehensive regulatory framework covering {len(regulations)} related regulations",
            "applicability": "Multiple entities and sectors",
            "scoping_threshold": "Various thresholds apply",
            "effective_date": f"{datetime.now().year}",
            "timeline_details": "Multiple implementation phases",
            "financial_integration": "Yes",
            "filing_mechanism": "Multiple mechanisms",
            "reporting_frequency": "Various frequencies",
            "assurance_requirement": "Various requirements",
            "penalties": "Multiple penalty frameworks",
            "full_text_link": regulations[0].get('full_text_link', 'Unknown') if regulations else 'Unknown',
            "translated_flag": any(r.get('translated_flag', False) for r in regulations),
            "source_url": "Multiple sources",
            "last_scraped": datetime.now().strftime("%Y-%m-%d"),
            "change_detected": False
        }


def consolidate_regulations():
    """Main consolidation function"""
    
    downloads_path = Path(DOWNLOADS_DIR)
    input_file = downloads_path / INPUT_FILE
    output_file = downloads_path / OUTPUT_FILE
    
    # Check if input file exists
    if not input_file.exists():
        print(f"Error: Input file not found: {input_file}")
        print("Run the LLM filter first to generate relevant_regulations.json")
        return
    
    # Load relevant regulations
    print(f"Loading regulations from: {input_file}")
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        regulations = data.get("regulations", [])
        metadata = data.get("metadata", {})
        
        print(f"Found {len(regulations)} individual regulations to consolidate")
        
    except Exception as e:
        print(f"Error reading input file: {e}")
        return
    
    if not regulations:
        print("No regulations found to consolidate")
        return
    
    # Initialize consolidator and process
    print("Creating comprehensive consolidated regulation...")
    consolidator = RegulationConsolidator()
    
    try:
        # Use LLM to create ONE consolidated regulation
        consolidated_regulation = consolidator.consolidate_regulations_with_llm(regulations)
        
        # Create final output structure
        output_data = {
            "metadata": {
                "consolidation_date": datetime.now().isoformat(),
                "source_file": INPUT_FILE,
                "input_regulations_count": len(regulations),
                "consolidation_method": "LLM_COMPREHENSIVE_SUMMARY",
                "description": "Single comprehensive regulation summarizing all sub-regulations"
            },
            "consolidated_regulation": consolidated_regulation
        }
        
        # Save consolidated file
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"Consolidation complete!")
        print(f"Input: {len(regulations)} individual regulations")
        print(f"Output: 1 comprehensive consolidated regulation")
        print(f"Saved to: {output_file}")
        print(f"Regulation name: {consolidated_regulation.get('regulation_name', 'Unknown')}")
        print(f"Coverage: {consolidated_regulation.get('country', 'Unknown')}")
        
    except Exception as e:
        print(f"Error during consolidation: {e}")
        return

def main():
    """Main function"""
    print("Comprehensive Regulation Consolidator")
    print("=" * 50)
    print("Creates ONE comprehensive regulation from all sub-regulations")
    print()
    
    # Run consolidation
    consolidate_regulations()

if __name__ == "__main__":
    main()