#!/usr/bin/env python3
"""
Main Regulation Processing Pipeline
Orchestrates the complete workflow:
1. Scrapes regulations using regulation_pipeline.py
2. Analyzes scraped documents using file_metadata_processor.py
"""

import sys
import os
import argparse
from pathlib import Path
from datetime import datetime
import time

# Import the pipeline modules
from regulation_pipeline import RegulationPipeline
from file_metadata_processor import FileMetadataProcessor


def print_banner():
    """Print a nice banner for the main pipeline"""
    print("=" * 80)
    print("🚀 REGULATION PROCESSING PIPELINE")
    print("=" * 80)
    print("This pipeline will:")
    print("1. 📊 Scrape regulations from Excel file using regulation_pipeline.py")
    print("2. 📄 Analyze scraped documents using file_metadata_processor.py")
    print("3. 📊 Generate comprehensive reports and metadata")
    print("=" * 80)


def check_prerequisites():
    """Check if all prerequisites are met before starting"""
    print("\n🔍 Checking prerequisites...")
    
    # Check if required files exist
    required_files = [
        "regulation_pipeline.py",
        "file_metadata_processor.py",
        "excel_reader.py",  # Required by regulation_pipeline
        "metadata.py",      # Required by file_metadata_processor
        "esg_filter.py"     # Required by file_metadata_processor
    ]
    
    missing_files = []
    for file in required_files:
        if not Path(file).exists():
            missing_files.append(file)
    
    if missing_files:
        print(f"❌ Missing required files: {', '.join(missing_files)}")
        return False
    
    print("✅ All required Python files found")
    
    # Check if scrapy project exists
    if not Path("documents").exists():
        print("❌ Scrapy project 'documents' not found")
        print("   Create the scrapy project first")
        return False
    
    print("✅ Scrapy project 'documents' found")
    
    return True


def run_regulation_pipeline(excel_file, regulation_column, sources_column, country_column):
    """Run the regulation scraping pipeline"""
    print(f"\n{'='*60}")
    print("🔄 STEP 1: REGULATION SCRAPING")
    print(f"{'='*60}")
    
    try:
        # Initialize pipeline
        pipeline = RegulationPipeline(excel_file)
        
        # Run the complete pipeline
        success = pipeline.run_pipeline(regulation_column, sources_column, country_column)
        
        if success:
            print("✅ Regulation scraping completed successfully!")
            return True
        else:
            print("❌ Regulation scraping failed!")
            return False
            
    except Exception as e:
        print(f"❌ Error during regulation scraping: {e}")
        return False


def run_file_metadata_processor(input_dir="regulation_scraping_results", 
                               output_dir="file_metadata_analysis_results",
                               max_workers=6, use_llm=True):
    """Run the file metadata processing pipeline"""
    print(f"\n{'='*60}")
    print("🔄 STEP 2: DOCUMENT ANALYSIS & METADATA EXTRACTION")
    print(f"{'='*60}")
    
    try:
        # Check if input directory exists
        if not Path(input_dir).exists():
            print(f"❌ Input directory not found: {input_dir}")
            print("   Run regulation scraping first!")
            return False
        
        # Initialize processor
        processor = FileMetadataProcessor(
            input_dir=input_dir,
            output_dir=output_dir,
            max_workers=max_workers
        )
        
        # Run the analysis
        results = processor.process_all_regulations(use_llm=use_llm)
        
        if results:
            print("✅ Document analysis completed successfully!")
            return True
        else:
            print("❌ Document analysis failed!")
            return False
            
    except Exception as e:
        print(f"❌ Error during document analysis: {e}")
        return False


def print_final_summary(scraping_success, analysis_success, start_time):
    """Print final pipeline summary"""
    end_time = datetime.now()
    total_duration = (end_time - start_time).total_seconds()
    
    print(f"\n{'='*80}")
    print("🎯 PIPELINE SUMMARY")
    print(f"{'='*80}")
    print(f"⏰ Started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"⏰ Completed at: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"⏱️  Total duration: {total_duration//3600:.0f}h {(total_duration%3600)//60:.0f}m {total_duration%60:.0f}s")
    print()
    
    # Step results
    print("📊 STEP RESULTS:")
    scraping_icon = "✅" if scraping_success else "❌"
    analysis_icon = "✅" if analysis_success else "❌"
    
    print(f"  {scraping_icon} Step 1: Regulation Scraping - {'SUCCESS' if scraping_success else 'FAILED'}")
    print(f"  {analysis_icon} Step 2: Document Analysis - {'SUCCESS' if analysis_success else 'FAILED'}")
    
    # Overall result
    overall_success = scraping_success and analysis_success
    overall_icon = "🎉" if overall_success else "💥"
    print(f"\n{overall_icon} OVERALL RESULT: {'SUCCESS' if overall_success else 'FAILED'}")
    
    # Output locations
    if scraping_success:
        print(f"\n📁 Scraped documents: regulation_scraping_results/")
        print(f"📊 Scraping report: regulation_scraping_results/pipeline_final_report.json")
    
    if analysis_success:
        print(f"📁 Analysis results: file_metadata_analysis_results/")
        print(f"📊 Analysis report: file_metadata_analysis_results/final_metadata_analysis_summary.json")
    
    print("=" * 80)
    
    return overall_success


def main():
    """Main function to orchestrate the complete pipeline"""
    parser = argparse.ArgumentParser(
        description="Complete Regulation Processing Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py regulations.xlsx
  python main.py regulations.xlsx --regulation-col "Regulation Name" --sources-col "Sources" --country-col "Country"
  python main.py regulations.xlsx --max-workers 4 --no-llm
  python main.py regulations.xlsx --skip-scraping  # Only run analysis
  python main.py regulations.xlsx --skip-analysis  # Only run scraping
        """
    )
    
    # Required arguments
    parser.add_argument("excel_file", help="Excel file containing regulations and sources")
    
    # Optional arguments for Excel columns
    parser.add_argument("--regulation-col", default="Regulation Name", 
                       help="Column name for regulation names (default: 'Regulation Name')")
    parser.add_argument("--sources-col", default="Sources", 
                       help="Column name for source URLs (default: 'Sources')")
    parser.add_argument("--country-col", default="Country", 
                       help="Column name for countries (default: 'Country')")
    
    # Optional arguments for processing
    parser.add_argument("--max-workers", type=int, default=6, 
                       help="Maximum number of concurrent workers for analysis (default: 6)")
    parser.add_argument("--no-llm", action="store_true", 
                       help="Skip LLM metadata extraction (faster but less detailed)")
    
    # Pipeline control
    parser.add_argument("--skip-scraping", action="store_true", 
                       help="Skip regulation scraping (only run analysis)")
    parser.add_argument("--skip-analysis", action="store_true", 
                       help="Skip document analysis (only run scraping)")
    
    # Output directories
    parser.add_argument("--scraping-output", default="regulation_scraping_results",
                       help="Output directory for scraped documents")
    parser.add_argument("--analysis-output", default="file_metadata_analysis_results",
                       help="Output directory for analysis results")
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.skip_scraping and args.skip_analysis:
        print("❌ Cannot skip both scraping and analysis!")
        sys.exit(1)
    
    if not Path(args.excel_file).exists():
        print(f"❌ Excel file not found: {args.excel_file}")
        sys.exit(1)
    
    # Print banner
    print_banner()
    
    # Record start time
    start_time = datetime.now()
    
    # Check prerequisites
    if not check_prerequisites():
        print("❌ Prerequisites not met. Please fix the issues above.")
        sys.exit(1)
    
    # Initialize success tracking
    scraping_success = True
    analysis_success = True
    
    # Step 1: Run regulation scraping
    if not args.skip_scraping:
        scraping_success = run_regulation_pipeline(
            args.excel_file, 
            args.regulation_col, 
            args.sources_col, 
            args.country_col
        )
        
        if not scraping_success:
            print("❌ Scraping failed. Stopping pipeline.")
            print_final_summary(scraping_success, False, start_time)
            sys.exit(1)
    else:
        print(f"\n⏭️  Skipping regulation scraping (--skip-scraping flag)")
        # Check if scraping results exist
        if not Path(args.scraping_output).exists():
            print(f"❌ Scraping output directory not found: {args.scraping_output}")
            print("   Cannot skip scraping without existing results!")
            sys.exit(1)
    
    # Step 2: Run document analysis
    if not args.skip_analysis:
        analysis_success = run_file_metadata_processor(
            input_dir=args.scraping_output,
            output_dir=args.analysis_output,
            max_workers=args.max_workers,
            use_llm=not args.no_llm
        )
    else:
        print(f"\n⏭️  Skipping document analysis (--skip-analysis flag)")
    
    # Print final summary
    overall_success = print_final_summary(scraping_success, analysis_success, start_time)
    
    # Exit with appropriate code
    sys.exit(0 if overall_success else 1)


if __name__ == "__main__":
    main()