#!/usr/bin/env python3
"""
Regulation Scraping Pipeline
Orchestrates the complete process:
1. Read Excel file with regulations and sources
2. For each regulation, scrape content using Scrapy with FlareSolverr fallback
3. Store each regulation's content in its own folder
"""

import os
import sys
import subprocess
import json
import time
from pathlib import Path
from datetime import datetime
from excel_reader import ExcelReader

class RegulationPipeline:
    def __init__(self, excel_file, flaresolverr_url="http://localhost:8191/v1"):
        self.excel_file = Path(excel_file)
        self.flaresolverr_url = flaresolverr_url
        self.pipeline_start_time = datetime.now()
        
        # Create main output directory
        self.output_dir = Path("regulation_scraping_results")
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize components
        self.excel_reader = ExcelReader(self.excel_file)
        self.regulations = []
        self.results = []
        
        print(f"🚀 Regulation Pipeline initialized")
        print(f"📊 Excel file: {self.excel_file}")
        print(f"📁 Output directory: {self.output_dir}")
    
    def check_prerequisites(self):
        """Check if all required components are available"""
        print(f"\n🔍 Checking prerequisites...")
        
        # Check if Excel file exists
        if not self.excel_file.exists():
            print(f"❌ Excel file not found: {self.excel_file}")
            return False
        print(f"✅ Excel file found: {self.excel_file}")
        
        # Check if FlareSolverr is running
        try:
            import requests
            response = requests.get(f"{self.flaresolverr_url.replace('/v1', '')}", timeout=5)
            if response.status_code in [200, 405]:  # 405 is expected for GET to FlareSolverr
                print("✅ FlareSolverr is running")
            else:
                print(f"⚠️  FlareSolverr status unclear (HTTP {response.status_code})")
        except Exception as e:
            print(f"❌ FlareSolverr is not accessible: {e}")
            print("Start FlareSolverr with: docker run -d -p 8191:8191 ghcr.io/flaresolverr/flaresolverr:latest")
            return False
        
        # Check required Python packages
        required_packages = ['scrapy', 'pandas', 'requests', 'bs4', 'html2text']
        missing_packages = []
        
        for package in required_packages:
            try:
                __import__(package)
                print(f"✅ {package} is available")
            except ImportError:
                missing_packages.append(package)
                print(f"❌ {package} is missing")
        
        if missing_packages:
            print(f"Install missing packages with: pip install {' '.join(missing_packages)}")
            return False
        
        # Check if scrapy project is properly set up
        if not Path("documents").exists():
            print("❌ Scrapy project 'documents' not found")
            return False
        print("✅ Scrapy project found")
        
        print("✅ All prerequisites met!")
        return True
    
    def load_regulations(self, regulation_column='Regulation Name', sources_column='Sources', country_column='Country'):
        """Load regulation data from Excel file"""
        print(f"\n📊 Loading regulations from Excel...")
        
        self.regulations = self.excel_reader.read_excel(regulation_column, sources_column, country_column)
        
        if not self.regulations:
            print("❌ No regulations found in Excel file")
            return False
        
        self.excel_reader.print_summary()
        
        # Save extracted regulation data
        regulation_data_file = self.output_dir / "extracted_regulations.json"
        self.excel_reader.save_extracted_data(regulation_data_file)
        
        return True
    
    def scrape_regulation(self, regulation):
        """Scrape content for a single regulation"""
        reg_name = regulation['name']
        country = regulation['country']
        urls = regulation['urls']
        
        print(f"\n🔄 Scraping regulation: [{country}] {reg_name}")
        print(f"🎯 URLs to scrape: {len(urls)}")
        
        # Create country-based folder structure
        safe_country = self.safe_folder_name(country)
        safe_reg_name = self.safe_folder_name(reg_name)
        
        country_dir = self.output_dir / safe_country
        country_dir.mkdir(exist_ok=True)
        
        reg_output_dir = country_dir / safe_reg_name
        
        # Check if regulation folder already exists and has content
        if reg_output_dir.exists():
            existing_files = list(reg_output_dir.glob('*'))
            if len(existing_files) > 1:  # More than just the regulation_info.json
                print(f"⏭️  SKIPPING: [{country}] {reg_name} - Folder already exists with {len(existing_files)} files")
                return {
                    'regulation_name': reg_name,
                    'country': country,
                    'status': 'skipped_existing',
                    'scraping_duration': 0,
                    'output_folder': str(reg_output_dir),
                    'country_folder': str(country_dir),
                    'scraped_content_count': self.count_scraped_content(reg_output_dir),
                    'error_message': 'Folder already exists with content'
                }
        
        reg_output_dir.mkdir(exist_ok=True)
        
        # Save regulation info
        reg_info = {
            'regulation_name': reg_name,
            'country': country,
            'sources_text': regulation['sources_text'],
            'urls': urls,
            'scraping_started_at': datetime.now().isoformat(),
            'output_folder': str(reg_output_dir),
            'country_folder': str(country_dir)
        }
        
        reg_info_file = reg_output_dir / "regulation_info.json"
        with open(reg_info_file, 'w', encoding='utf-8') as f:
            json.dump(reg_info, f, indent=2, ensure_ascii=False)
        
        # Run Scrapy spider for this regulation
        start_time = time.time()
        
        try:
            # Prepare Scrapy command
            # Escape URLs properly for shell
            urls_str = ",".join(urls)
            
            cmd = [
                'scrapy', 'crawl', 'regulation',
                '-a', f'regulation_name={reg_name}',
                '-a', f'country={country}',
                '-a', f'start_urls={urls_str}',
                '-s', f'FILES_STORE={reg_output_dir}',
                '-s', 'LOG_LEVEL=INFO'
            ]
            
            print(f"🚀 Running: {' '.join(cmd)}")
            
            # Run the spider
            result = subprocess.run(
                cmd,
                cwd=Path.cwd(),
                capture_output=True,
                text=True,
                timeout=1800  # 30 minute timeout per regulation
            )
            
            scraping_time = time.time() - start_time
            
            # Process results
            if result.returncode == 0:
                print(f"✅ Scraping completed for {reg_name} ({scraping_time:.1f}s)")
                status = "completed"
                error_message = None
            else:
                print(f"❌ Scraping failed for {reg_name}")
                print(f"Error output: {result.stderr}")
                status = "failed"
                error_message = result.stderr
            
            # Update regulation info with results
            reg_info.update({
                'scraping_completed_at': datetime.now().isoformat(),
                'scraping_duration_seconds': scraping_time,
                'status': status,
                'scrapy_returncode': result.returncode,
                'scrapy_stdout': result.stdout,
                'scrapy_stderr': result.stderr if result.stderr else None,
                'error_message': error_message
            })
            
            # Save updated info
            with open(reg_info_file, 'w', encoding='utf-8') as f:
                json.dump(reg_info, f, indent=2, ensure_ascii=False)
            
            # Count scraped content
            scraped_files = self.count_scraped_content(reg_output_dir)
            reg_info['scraped_content_count'] = scraped_files
            
            return {
                'regulation_name': reg_name,
                'country': country,
                'status': status,
                'scraping_duration': scraping_time,
                'output_folder': str(reg_output_dir),
                'country_folder': str(country_dir),
                'scraped_content_count': scraped_files,
                'error_message': error_message
            }
            
        except subprocess.TimeoutExpired:
            print(f"⏰ Scraping timeout for [{country}] {reg_name}")
            return {
                'regulation_name': reg_name,
                'country': country,
                'status': 'timeout',
                'scraping_duration': time.time() - start_time,
                'output_folder': str(reg_output_dir),
                'country_folder': str(country_dir),
                'error_message': 'Scraping timeout (30 minutes)'
            }
        except Exception as e:
            print(f"❌ Error scraping [{country}] {reg_name}: {e}")
            return {
                'regulation_name': reg_name,
                'country': country,
                'status': 'error',
                'scraping_duration': time.time() - start_time,
                'output_folder': str(reg_output_dir),
                'country_folder': str(country_dir),
                'error_message': str(e)
            }
    
    def count_scraped_content(self, folder):
        """Count scraped content files in a folder"""
        try:
            content_count = {
                'html_files': len(list(folder.glob('*.html'))),
                'pdf_files': len(list(folder.glob('*.pdf'))),
                'txt_files': len(list(folder.glob('*.txt'))),
                'json_files': len(list(folder.glob('*.json'))),
                'total_files': len(list(folder.glob('*')))
            }
            return content_count
        except Exception as e:
            print(f"⚠️  Error counting files in {folder}: {e}")
            return {'error': str(e)}
    
    def safe_folder_name(self, name):
        """Convert regulation name to safe folder name"""
        import re
        safe_name = re.sub(r'[<>:"/\\|?*]', '_', name)
        safe_name = re.sub(r'\s+', '_', safe_name)
        safe_name = safe_name.strip('._')
        return safe_name[:100]  # Limit length
    
    def run_pipeline(self, regulation_column='Regulation Name', sources_column='Sources', country_column='Country'):
        """Run the complete pipeline"""
        print(f"🚀 Starting Regulation Scraping Pipeline")
        print(f"⏰ Started at: {self.pipeline_start_time}")
        print("=" * 60)
        
        # Step 1: Check prerequisites
        if not self.check_prerequisites():
            print("❌ Prerequisites not met. Exiting.")
            return False
        
        # Step 2: Load regulations from Excel
        if not self.load_regulations(regulation_column, sources_column, country_column):
            print("❌ Failed to load regulations. Exiting.")
            return False
        
        # Step 3: Scrape each regulation
        print(f"\n🔄 Starting to scrape {len(self.regulations)} regulations...")
        
        for i, regulation in enumerate(self.regulations, 1):
            print(f"\n{'='*60}")
            print(f"📋 Processing regulation {i}/{len(self.regulations)}")
            
            result = self.scrape_regulation(regulation)
            self.results.append(result)
            
            print(f"📊 Result: {result['status']}")
            if result.get('scraped_content_count'):
                content = result['scraped_content_count']
                print(f"📄 Content: {content.get('total_files', 0)} files total")
            
            # Small delay between regulations
            if i < len(self.regulations):
                print("⏸️  Waiting 3 seconds before next regulation...")
                time.sleep(3)
        
        # Step 4: Generate final report
        self.generate_final_report()
        
        print(f"\n🎉 Pipeline completed!")
        return True
    
    def generate_final_report(self):
        """Generate final pipeline report"""
        pipeline_end_time = datetime.now()
        total_duration = (pipeline_end_time - self.pipeline_start_time).total_seconds()
        
        # Calculate statistics
        completed_count = sum(1 for r in self.results if r['status'] == 'completed')
        skipped_count = sum(1 for r in self.results if r['status'] == 'skipped_existing')
        failed_count = sum(1 for r in self.results if r['status'] in ['failed', 'error', 'timeout'])
        total_files_scraped = 0
        
        for result in self.results:
            if result.get('scraped_content_count') and isinstance(result['scraped_content_count'], dict):
                total_files_scraped += result['scraped_content_count'].get('total_files', 0)
        
        # Create final report
        final_report = {
            'pipeline_summary': {
                'excel_file': str(self.excel_file),
                'total_regulations': len(self.regulations),
                'completed_regulations': completed_count,
                'skipped_regulations': skipped_count,
                'failed_regulations': failed_count,
                'success_rate': f"{(completed_count/len(self.regulations)*100):.1f}%" if self.regulations else "0%",
                'skip_rate': f"{(skipped_count/len(self.regulations)*100):.1f}%" if self.regulations else "0%",
                'total_files_scraped': total_files_scraped,
                'pipeline_started_at': self.pipeline_start_time.isoformat(),
                'pipeline_completed_at': pipeline_end_time.isoformat(),
                'total_duration_seconds': total_duration,
                'total_duration_formatted': f"{total_duration//3600:.0f}h {(total_duration%3600)//60:.0f}m {total_duration%60:.0f}s"
            },
            'regulation_results': self.results,
            'output_directory': str(self.output_dir)
        }
        
        # Save final report
        report_file = self.output_dir / "pipeline_final_report.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(final_report, f, indent=2, ensure_ascii=False)
        
        # Print summary
        print(f"\n📊 FINAL PIPELINE REPORT")
        print("=" * 60)
        print(f"📂 Excel file: {self.excel_file}")
        print(f"📋 Total regulations: {len(self.regulations)}")
        print(f"✅ Completed: {completed_count}")
        print(f"❌ Failed: {failed_count}")
        print(f"📈 Success rate: {(completed_count/len(self.regulations)*100):.1f}%")
        print(f"📄 Total files scraped: {total_files_scraped}")
        print(f"⏱️  Total duration: {final_report['pipeline_summary']['total_duration_formatted']}")
        print(f"📁 Output directory: {self.output_dir}")
        print(f"📊 Full report: {report_file}")
        
        # List regulation folders
        print(f"\n📁 REGULATION FOLDERS BY COUNTRY:")
        countries = {}
        for result in self.results:
            country = result.get('country', 'Unknown')
            if country not in countries:
                countries[country] = []
            countries[country].append(result)
        
        for country, regulations in countries.items():
            print(f"\n🌍 {country}:")
            for result in regulations:
                status_icon = "✅" if result['status'] == 'completed' else "❌"
                folder_name = Path(result['output_folder']).name
                content_count = result.get('scraped_content_count', {})
                if isinstance(content_count, dict):
                    file_count = content_count.get('total_files', 0)
                else:
                    file_count = 0
                print(f"  {status_icon} {folder_name} ({file_count} files)")

def main():
    if len(sys.argv) < 2:
        print("Usage: python regulation_pipeline.py <excel_file> [regulation_column] [sources_column] [country_column]")
        print("Example: python regulation_pipeline.py regulations.xlsx 'Regulation Name' 'Sources' 'Country'")
        print("\nThe pipeline will:")
        print("1. Read the Excel file and extract regulations with their source URLs and countries")
        print("2. For each regulation, create a country-based folder structure")
        print("3. Scrape content from source URLs using Scrapy (depth limit 2-3)")
        print("4. Use FlareSolverr fallback for 403 errors or Cloudflare protection")
        print("5. Store HTML, PDF, and text content in country/regulation-specific folders")
        print("\nFolder structure: regulation_scraping_results/Country/Regulation_Name/")
        print("\nPrerequisites:")
        print("- FlareSolverr running on localhost:8191")
        print("- Required packages: scrapy, pandas, requests, beautifulsoup4, html2text")
        sys.exit(1)
    
    excel_file = sys.argv[1]
    regulation_column = sys.argv[2] if len(sys.argv) > 2 else 'Regulation Name'
    sources_column = sys.argv[3] if len(sys.argv) > 3 else 'Sources'
    country_column = sys.argv[4] if len(sys.argv) > 4 else 'Country'
    
    # Initialize and run pipeline
    pipeline = RegulationPipeline(excel_file)
    success = pipeline.run_pipeline(regulation_column, sources_column, country_column)
    
    if success:
        print("\n🎉 Pipeline completed successfully!")
        sys.exit(0)
    else:
        print("\n❌ Pipeline failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()