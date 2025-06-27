#!/usr/bin/env python3
"""
Consolidate Metadata Analysis Results
Consolidates file metadata analysis results from file_metadata_analysis_results folder
into single JSON files per country, similar to consolidate_regulations.py
"""

import json
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
import argparse


class MetadataResultsConsolidator:
    """Consolidates metadata analysis results into single files per country"""
    
    def __init__(self, input_dir: str = "file_metadata_analysis_results", 
                 output_dir: str = "consolidated_metadata_results"):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        print(f"📂 Input directory: {self.input_dir}")
        print(f"📁 Output directory: {self.output_dir}")
    
    def load_json_file(self, file_path: Path) -> Dict[str, Any]:
        """Load JSON file with error handling"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Warning: Could not load {file_path}: {e}")
            return {}
    
    def collect_country_data(self, country_folder: Path) -> Dict[str, Any]:
        """Collect all data for a specific country"""
        country_name = country_folder.name
        print(f"\n🌍 Processing country: {country_name}")
        
        # Load country summary if it exists
        country_summary_file = country_folder / f"{country_name}_summary.json"
        country_summary = {}
        if country_summary_file.exists():
            country_summary = self.load_json_file(country_summary_file)
            print(f"  ✅ Loaded country summary: {country_summary_file.name}")
        
        # Find all regulation folders
        regulation_folders = [d for d in country_folder.iterdir() if d.is_dir()]
        print(f"  📋 Found {len(regulation_folders)} regulation folders")
        
        regulations_data = []
        all_documents = []
        
        # Process each regulation folder
        for reg_folder in regulation_folders:
            regulation_name = reg_folder.name
            print(f"    🔄 Processing regulation: {regulation_name}")
            
            # Load regulation summary
            reg_summary_file = reg_folder / "regulation_summary.json"
            regulation_summary = {}
            if reg_summary_file.exists():
                regulation_summary = self.load_json_file(reg_summary_file)
            
            # Collect all document analyses in this regulation
            document_analyses = []
            analysis_files = list(reg_folder.glob("*_analysis.json"))
            
            print(f"      📄 Found {len(analysis_files)} document analyses")
            
            for analysis_file in analysis_files:
                document_analysis = self.load_json_file(analysis_file)
                if document_analysis:
                    # Add regulation context to each document
                    document_analysis['regulation_name'] = regulation_name
                    document_analysis['country'] = country_name
                    document_analyses.append(document_analysis)
                    all_documents.append(document_analysis)
            
            # Create regulation entry
            regulation_entry = {
                'regulation_name': regulation_name,
                'regulation_folder': regulation_name,
                'regulation_summary': regulation_summary,
                'document_analyses': document_analyses,
                'document_count': len(document_analyses),
                'esg_relevant_count': len([d for d in document_analyses if d.get('esg_relevant', False)])
            }
            
            regulations_data.append(regulation_entry)
            print(f"      ✅ Processed {len(document_analyses)} documents, {regulation_entry['esg_relevant_count']} ESG relevant")
        
        # Calculate consolidated statistics
        total_documents = len(all_documents)
        esg_relevant_documents = len([d for d in all_documents if d.get('esg_relevant', False)])
        
        # Create consolidated country data
        consolidated_data = {
            'country': country_name,
            'consolidated_at': datetime.now().isoformat(),
            'source_directory': str(country_folder),
            'statistics': {
                'total_regulations': len(regulations_data),
                'total_documents': total_documents,
                'esg_relevant_documents': esg_relevant_documents,
                'esg_relevance_rate': f"{(esg_relevant_documents/total_documents*100):.1f}%" if total_documents > 0 else "0%",
                'regulations_with_content': len([r for r in regulations_data if r['document_count'] > 0])
            },
            'country_summary': country_summary,
            'regulations': regulations_data,
            'all_documents': all_documents,
            'esg_relevant_documents': [d for d in all_documents if d.get('esg_relevant', False)]
        }
        
        return consolidated_data
    
    def create_country_reports(self, country_data: Dict[str, Any]) -> Dict[str, str]:
        """Create different report formats for a country"""
        country_name = country_data['country']
        
        reports = {}
        
        # 1. Full consolidated report (everything)
        full_report_file = self.output_dir / f"{country_name}_full_consolidated.json"
        with open(full_report_file, 'w', encoding='utf-8') as f:
            json.dump(country_data, f, indent=2, ensure_ascii=False)
        reports['full'] = str(full_report_file)
        
        # 2. ESG-only report (only ESG relevant documents)
        esg_data = country_data.copy()
        esg_data['regulations'] = []
        for reg in country_data['regulations']:
            esg_docs = [d for d in reg['document_analyses'] if d.get('esg_relevant', False)]
            if esg_docs:
                esg_reg = reg.copy()
                esg_reg['document_analyses'] = esg_docs
                esg_reg['document_count'] = len(esg_docs)
                esg_reg['esg_relevant_count'] = len(esg_docs)
                esg_data['regulations'].append(esg_reg)
        
        esg_data['all_documents'] = country_data['esg_relevant_documents']
        esg_data['statistics']['total_documents'] = len(esg_data['all_documents'])
        esg_data['statistics']['esg_relevant_documents'] = len(esg_data['all_documents'])
        esg_data['statistics']['esg_relevance_rate'] = "100%"
        
        esg_report_file = self.output_dir / f"{country_name}_esg_only.json"
        with open(esg_report_file, 'w', encoding='utf-8') as f:
            json.dump(esg_data, f, indent=2, ensure_ascii=False)
        reports['esg_only'] = str(esg_report_file)
        
        # 3. Summary report (statistics and metadata only, no full text)
        summary_data = {
            'country': country_data['country'],
            'consolidated_at': country_data['consolidated_at'],
            'statistics': country_data['statistics'],
            'country_summary': country_data['country_summary'],
            'regulations_summary': []
        }
        
        for reg in country_data['regulations']:
            reg_summary = {
                'regulation_name': reg['regulation_name'],
                'document_count': reg['document_count'],
                'esg_relevant_count': reg['esg_relevant_count'],
                'regulation_summary': reg['regulation_summary'],
                'documents_metadata': []
            }
            
            # Include metadata but not full extracted text
            for doc in reg['document_analyses']:
                doc_meta = {
                    'file_name': doc.get('file_name'),
                    'file_type': doc.get('file_type'),
                    'file_size': doc.get('file_size'),
                    'text_length': doc.get('text_length'),
                    'word_count': doc.get('word_count'),
                    'esg_relevant': doc.get('esg_relevant'),
                    'esg_match_score': doc.get('esg_match_score'),
                    'source_url': doc.get('source_url'),
                    'processed_at': doc.get('processed_at'),
                    'metadata': doc.get('metadata')
                }
                reg_summary['documents_metadata'].append(doc_meta)
            
            summary_data['regulations_summary'].append(reg_summary)
        
        summary_report_file = self.output_dir / f"{country_name}_summary_only.json"
        with open(summary_report_file, 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, indent=2, ensure_ascii=False)
        reports['summary_only'] = str(summary_report_file)
        
        return reports
    
    def consolidate_all_countries(self) -> Dict[str, Any]:
        """Process all countries and create consolidated reports"""
        
        if not self.input_dir.exists():
            print(f"❌ Input directory not found: {self.input_dir}")
            return {}
        
        print(f"🚀 Starting metadata results consolidation")
        print("=" * 60)
        
        # Find all country folders
        country_folders = [d for d in self.input_dir.iterdir() if d.is_dir()]
        
        if not country_folders:
            print(f"❌ No country folders found in {self.input_dir}")
            return {}
        
        print(f"Found {len(country_folders)} country folders to consolidate")
        
        consolidation_results = {
            'consolidation_metadata': {
                'consolidated_at': datetime.now().isoformat(),
                'input_directory': str(self.input_dir),
                'output_directory': str(self.output_dir),
                'countries_processed': 0,
                'total_regulations': 0,
                'total_documents': 0,
                'total_esg_relevant': 0
            },
            'countries': {}
        }
        
        # Process each country
        for country_folder in country_folders:
            try:
                country_data = self.collect_country_data(country_folder)
                
                if country_data['statistics']['total_documents'] == 0:
                    print(f"  ⏭️  Skipping {country_folder.name} - no documents found")
                    continue
                
                # Create different report formats
                reports = self.create_country_reports(country_data)
                
                # Update consolidation results
                country_name = country_data['country']
                consolidation_results['countries'][country_name] = {
                    'statistics': country_data['statistics'],
                    'report_files': reports
                }
                
                # Update totals
                consolidation_results['consolidation_metadata']['countries_processed'] += 1
                consolidation_results['consolidation_metadata']['total_regulations'] += country_data['statistics']['total_regulations']
                consolidation_results['consolidation_metadata']['total_documents'] += country_data['statistics']['total_documents']
                consolidation_results['consolidation_metadata']['total_esg_relevant'] += country_data['statistics']['esg_relevant_documents']
                
                print(f"  ✅ Consolidated {country_name}:")
                print(f"      📋 {country_data['statistics']['total_regulations']} regulations")
                print(f"      📄 {country_data['statistics']['total_documents']} documents")
                print(f"      🎯 {country_data['statistics']['esg_relevant_documents']} ESG relevant")
                print(f"      📁 Reports: {len(reports)} files created")
                
            except Exception as e:
                print(f"❌ Error processing {country_folder.name}: {e}")
                continue
        
        # Save consolidation summary
        summary_file = self.output_dir / "consolidation_summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(consolidation_results, f, indent=2, ensure_ascii=False)
        
        return consolidation_results
    
    def print_final_summary(self, results: Dict[str, Any]):
        """Print final consolidation summary"""
        metadata = results['consolidation_metadata']
        
        print(f"\n🎉 CONSOLIDATION COMPLETED")
        print("=" * 60)
        print(f"📊 Countries processed: {metadata['countries_processed']}")
        print(f"📋 Total regulations: {metadata['total_regulations']}")
        print(f"📄 Total documents: {metadata['total_documents']}")
        print(f"🎯 Total ESG relevant: {metadata['total_esg_relevant']}")
        
        if metadata['total_documents'] > 0:
            esg_rate = (metadata['total_esg_relevant'] / metadata['total_documents'] * 100)
            print(f"📈 Overall ESG rate: {esg_rate:.1f}%")
        
        print(f"\n📁 Output directory: {self.output_dir}")
        print(f"📊 Consolidation summary: {self.output_dir}/consolidation_summary.json")
        
        print(f"\n📋 REPORT FILES CREATED:")
        for country, country_info in results['countries'].items():
            print(f"\n🌍 {country}:")
            for report_type, file_path in country_info['report_files'].items():
                file_size = Path(file_path).stat().st_size / 1024  # KB
                print(f"  📄 {report_type}: {Path(file_path).name} ({file_size:.1f} KB)")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Consolidate metadata analysis results into single JSON files per country",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python consolidate_metadata_results.py
  python consolidate_metadata_results.py --input file_metadata_analysis_results --output my_consolidated_results

Output files created per country:
  - {country}_full_consolidated.json     # Complete data with full text
  - {country}_esg_only.json             # Only ESG relevant documents  
  - {country}_summary_only.json         # Metadata only, no full text
  - consolidation_summary.json          # Overall summary
        """
    )
    
    parser.add_argument("--input", "-i", default="file_metadata_analysis_results",
                       help="Input directory containing metadata analysis results")
    parser.add_argument("--output", "-o", default="consolidated_metadata_results", 
                       help="Output directory for consolidated files")
    
    args = parser.parse_args()
    
    # Initialize consolidator
    consolidator = MetadataResultsConsolidator(args.input, args.output)
    
    # Run consolidation
    results = consolidator.consolidate_all_countries()
    
    if results and results['consolidation_metadata']['countries_processed'] > 0:
        consolidator.print_final_summary(results)
        print("\n✅ Consolidation completed successfully!")
    else:
        print("\n❌ No countries processed!")
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)