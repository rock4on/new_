#!/usr/bin/env python3
"""
Country Summary Collector

Gets an array of all {country_name}_summary.json files from the file_metadata_processor output.
"""

import json
from pathlib import Path
from typing import List, Dict, Any


def get_country_summary_files(output_dir: str = "file_metadata_analysis_results") -> List[Path]:
    """
    Get all {country_name}_summary.json files from the output directory.
    
    Args:
        output_dir: Directory where file_metadata_processor saves results
        
    Returns:
        List of Path objects pointing to summary files
    """
    output_path = Path(output_dir)
    
    if not output_path.exists():
        print(f"Output directory does not exist: {output_path}")
        return []
    
    # Find all *_summary.json files
    summary_files = list(output_path.glob("**/*_summary.json"))
    
    # Filter to only country summaries (exclude other summary types)
    country_summaries = [f for f in summary_files if not f.name.startswith('esg_') and not f.name.startswith('final_')]
    
    return country_summaries


def load_country_summaries(output_dir: str = "file_metadata_analysis_results") -> List[Dict[str, Any]]:
    """
    Load all country summary data into memory.
    
    Args:
        output_dir: Directory where file_metadata_processor saves results
        
    Returns:
        List of dictionaries containing country summary data
    """
    summary_files = get_country_summary_files(output_dir)
    summaries = []
    
    for file_path in summary_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # Add file path info to the data
                data['summary_file_path'] = str(file_path)
                summaries.append(data)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
    
    return summaries


def save_consolidated_summaries(summaries: List[Dict[str, Any]], output_file: str = "consolidated_country_summaries.json") -> str:
    """
    Save all country summaries into a single consolidated JSON file.
    
    Args:
        summaries: List of country summary dictionaries
        output_file: Output filename for consolidated data
        
    Returns:
        Path to the saved file
    """
    from datetime import datetime
    
    consolidated_data = {
        'consolidation_metadata': {
            'created_at': datetime.now().isoformat(),
            'total_countries': len(summaries),
            'source_files': [s.get('summary_file_path', '') for s in summaries],
            'consolidated_by': 'get_country_summaries.py'
        },
        'global_statistics': {
            'total_countries': len(summaries),
            'total_regulations': sum(s.get('total_regulations', 0) for s in summaries),
            'total_documents_processed': sum(s.get('total_documents_processed', 0) for s in summaries),
            'total_esg_relevant_documents': sum(s.get('total_esg_relevant_documents', 0) for s in summaries),
            'total_documents_found': sum(s.get('total_documents_found', 0) for s in summaries)
        },
        'country_summaries': summaries
    }
    
    output_path = Path(output_file)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(consolidated_data, f, indent=2, ensure_ascii=False)
    
    return str(output_path)


def main():
    """Main function to demonstrate usage"""
    
    # Get file paths
    print("🔍 Finding country summary files...")
    summary_files = get_country_summary_files()
    
    if not summary_files:
        print("❌ No country summary files found.")
        print("💡 Make sure file_metadata_processor.py has been run first.")
        return
    
    print(f"✅ Found {len(summary_files)} country summary files:")
    for file_path in summary_files:
        print(f"  📄 {file_path}")
    
    # Load the data
    print("\n📊 Loading country summary data...")
    summaries = load_country_summaries()
    
    if summaries:
        print(f"✅ Loaded {len(summaries)} country summaries")
        
        # Show basic stats
        total_regulations = sum(s.get('total_regulations', 0) for s in summaries)
        total_docs = sum(s.get('total_documents_processed', 0) for s in summaries)
        total_esg = sum(s.get('total_esg_relevant_documents', 0) for s in summaries)
        
        print(f"\n📈 Summary Statistics:")
        print(f"  🌍 Countries: {len(summaries)}")
        print(f"  📋 Total Regulations: {total_regulations}")
        print(f"  📄 Total Documents: {total_docs}")
        print(f"  🎯 ESG Relevant: {total_esg}")
        
        # Show country breakdown
        print(f"\n🌍 Countries found:")
        for summary in summaries:
            country = summary.get('country', 'Unknown')
            regs = summary.get('total_regulations', 0)
            docs = summary.get('total_documents_processed', 0)
            esg = summary.get('total_esg_relevant_documents', 0)
            print(f"  {country}: {regs} regulations, {docs} docs, {esg} ESG relevant")
        
        # Save consolidated file
        print(f"\n💾 Saving consolidated summaries...")
        consolidated_file = save_consolidated_summaries(summaries)
        print(f"✅ Consolidated summaries saved to: {consolidated_file}")
    
    return summary_files


if __name__ == "__main__":
    main()