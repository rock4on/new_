#!/usr/bin/env python3
"""
Bulk Document Processor

Processes multiple URLs in bulk:
1. Crawls each URL with Scrapy to download PDFs
2. Filters downloaded documents with LLM
3. Organizes results by URL/domain

Usage: Just edit the configuration below and run: python3 bulk_processor.py
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path
from datetime import datetime
from urllib.parse import urlparse
import json

# =============================================================================
# CONFIGURATION - Edit these settings
# =============================================================================

# List of URLs to crawl
URLS_TO_CRAWL = [
    "https://example.com/documents",
    "https://another-site.com/resources", 
    "https://third-site.org/publications",
    # Add more URLs here...
]

# LLM filtering criteria
LLM_CRITERIA = "documents related to financial regulations, compliance, and legal requirements"

# LLM API Configuration
API_KEY = "your-api-key-here"
BASE_URL = "https://api.openai.com/v1"  # Change to your custom endpoint
MODEL = "gpt-4o-mini"
CONFIDENCE_THRESHOLD = 0.7

# Processing options
CLEAN_DOWNLOADS_BETWEEN_RUNS = True  # Clean downloads folder between URLs
ORGANIZE_BY_DOMAIN = True            # Create separate folders per domain
MAX_CONCURRENT_URLS = 1              # Process URLs one at a time

# =============================================================================
# Bulk Processor Class
# =============================================================================

class BulkProcessor:
    def __init__(self):
        self.base_dir = Path.cwd()
        self.results_dir = self.base_dir / "bulk_results"
        self.downloads_dir = self.base_dir / "downloads"
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create results directory structure
        self.results_dir.mkdir(exist_ok=True)
        self.session_dir = self.results_dir / f"session_{self.timestamp}"
        self.session_dir.mkdir(exist_ok=True)
        
        # Initialize tracking
        self.processed_urls = []
        self.failed_urls = []
        self.total_relevant_docs = 0
        self.total_docs_downloaded = 0
    
    def get_domain_name(self, url):
        """Extract clean domain name from URL"""
        parsed = urlparse(url)
        domain = parsed.netloc.replace("www.", "")
        # Clean domain for folder name
        return "".join(c for c in domain if c.isalnum() or c in ".-").strip(".-")
    
    def run_scrapy_spider(self, url):
        """Run scrapy spider for a specific URL"""
        print(f"🕷️  Crawling: {url}")
        
        try:
            # Run scrapy spider
            cmd = [
                "scrapy", "crawl", "docs", 
                "-a", f"start_url={url}",
                "-s", "LOG_LEVEL=INFO"
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.base_dir)
            
            if result.returncode != 0:
                print(f"❌ Scrapy failed for {url}")
                print(f"Error: {result.stderr}")
                return False
            
            print(f"✅ Scrapy completed for {url}")
            return True
            
        except Exception as e:
            print(f"❌ Error running scrapy for {url}: {e}")
            return False
    
    def get_realtime_results(self):
        """Get results from real-time LLM processing"""
        print("📊 Collecting real-time analysis results...")
        
        try:
            results_file = self.downloads_dir / "live_regulatory_analysis.json"
            if results_file.exists():
                with open(results_file, 'r', encoding='utf-8') as f:
                    analysis_data = json.load(f)
                
                relevant_count = analysis_data.get("metadata", {}).get("total_relevant_documents", 0)
                total_processed = analysis_data.get("metadata", {}).get("total_processed", 0)
                
                print(f"✅ Real-time analysis complete: {relevant_count} relevant docs out of {total_processed} processed")
                return relevant_count, total_processed, analysis_data
            else:
                print("⚠️ No real-time analysis results found")
                return 0, 0, None
                
        except Exception as e:
            print(f"❌ Error reading real-time results: {e}")
            return 0, 0, None
    
    def organize_results(self, url):
        """Organize results for current URL"""
        domain = self.get_domain_name(url)
        
        if ORGANIZE_BY_DOMAIN:
            url_results_dir = self.session_dir / domain
        else:
            url_results_dir = self.session_dir / f"url_{len(self.processed_urls)+1}"
        
        url_results_dir.mkdir(exist_ok=True)
        
        # Count documents
        relevant_dir = self.downloads_dir / "relevant"
        irrelevant_dir = self.downloads_dir / "irrelevant"
        
        relevant_count = len(list(relevant_dir.glob("*.pdf"))) if relevant_dir.exists() else 0
        irrelevant_count = len(list(irrelevant_dir.glob("*.pdf"))) if irrelevant_dir.exists() else 0
        total_count = relevant_count + irrelevant_count
        
        # Move results to organized folders
        if relevant_dir.exists() and relevant_count > 0:
            target_relevant = url_results_dir / "relevant"
            if target_relevant.exists():
                shutil.rmtree(target_relevant)
            shutil.move(str(relevant_dir), str(target_relevant))
        
        if irrelevant_dir.exists() and irrelevant_count > 0:
            target_irrelevant = url_results_dir / "irrelevant"
            if target_irrelevant.exists():
                shutil.rmtree(target_irrelevant)
            shutil.move(str(irrelevant_dir), str(target_irrelevant))
        
        # Move analysis report
        analysis_file = self.downloads_dir / "relevance_analysis.json"
        if analysis_file.exists():
            shutil.move(str(analysis_file), str(url_results_dir / "relevance_analysis.json"))
        
        # Create URL summary
        summary = {
            "url": url,
            "domain": domain,
            "processed_at": datetime.now().isoformat(),
            "total_documents": total_count,
            "relevant_documents": relevant_count,
            "irrelevant_documents": irrelevant_count,
            "criteria": LLM_CRITERIA,
            "model_used": MODEL,
            "confidence_threshold": CONFIDENCE_THRESHOLD
        }
        
        with open(url_results_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        
        # Update totals
        self.total_docs_downloaded += total_count
        self.total_relevant_docs += relevant_count
        
        print(f"📁 Results organized in: {url_results_dir}")
        print(f"📊 Found {relevant_count}/{total_count} relevant documents")
        
        return relevant_count, total_count
    
    def organize_results_new(self, url, analysis_data):
        """Organize results using real-time analysis data"""
        domain = self.get_domain_name(url)
        
        if ORGANIZE_BY_DOMAIN:
            url_results_dir = self.session_dir / domain
        else:
            url_results_dir = self.session_dir / f"url_{len(self.processed_urls)+1}"
        
        url_results_dir.mkdir(exist_ok=True)
        
        # Get counts from metadata
        metadata = analysis_data.get("metadata", {})
        relevant_count = metadata.get("total_relevant_documents", 0)
        total_count = metadata.get("total_processed", 0)
        
        # Copy the real-time analysis file
        analysis_source = self.downloads_dir / "live_regulatory_analysis.json"
        if analysis_source.exists():
            shutil.copy(str(analysis_source), str(url_results_dir / "regulatory_analysis.json"))
        
        # Create URL summary with enhanced info
        summary = {
            "url": url,
            "domain": domain,
            "processed_at": datetime.now().isoformat(),
            "total_documents": total_count,
            "relevant_documents": relevant_count,
            "irrelevant_documents": total_count - relevant_count,
            "criteria": analysis_data.get("metadata", {}).get("criteria", "Unknown"),
            "model_used": analysis_data.get("metadata", {}).get("model", "Unknown"),
            "confidence_threshold": analysis_data.get("metadata", {}).get("confidence_threshold", 0.7),
            "processing_method": "Real-time LLM analysis during crawling"
        }
        
        with open(url_results_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        
        # Update totals
        self.total_docs_downloaded += total_count
        self.total_relevant_docs += relevant_count
        
        print(f"📁 Results organized in: {url_results_dir}")
        print(f"📊 Found {relevant_count}/{total_count} relevant documents")
        print(f"⚡ Real-time processing: No separate LLM filtering step needed!")
        
        return relevant_count, total_count
    
    def clean_downloads(self):
        """Clean downloads directory"""
        if self.downloads_dir.exists():
            shutil.rmtree(self.downloads_dir)
        self.downloads_dir.mkdir(exist_ok=True)
    
    def process_all_urls(self):
        """Process all URLs in the list"""
        print("🚀 Starting bulk document processing...")
        print(f"📝 Processing {len(URLS_TO_CRAWL)} URLs")
        print(f"🎯 Criteria: {LLM_CRITERIA}")
        print(f"📁 Results will be saved to: {self.session_dir}")
        print("=" * 80)
        
        for i, url in enumerate(URLS_TO_CRAWL, 1):
            print(f"\n📍 Processing {i}/{len(URLS_TO_CRAWL)}: {url}")
            print("-" * 60)
            
            try:
                # Clean downloads if configured
                if CLEAN_DOWNLOADS_BETWEEN_RUNS and i > 1:
                    print("🧹 Cleaning downloads directory...")
                    self.clean_downloads()
                
                # Step 1: Run scrapy spider (with real-time LLM processing)
                if not self.run_scrapy_spider(url):
                    self.failed_urls.append(url)
                    continue
                
                # Step 2: Get real-time analysis results
                relevant_count, total_count, analysis_data = self.get_realtime_results()
                if analysis_data is None:
                    self.failed_urls.append(url)
                    continue
                
                # Step 3: Organize results
                relevant_count, total_count = self.organize_results_new(url, analysis_data)
                
                self.processed_urls.append({
                    "url": url,
                    "relevant_docs": relevant_count,
                    "total_docs": total_count
                })
                
                print(f"✅ Completed processing: {url}")
                
            except Exception as e:
                print(f"❌ Failed to process {url}: {e}")
                self.failed_urls.append(url)
        
        # Generate final report
        self.generate_final_report()
    
    def generate_final_report(self):
        """Generate final processing report"""
        print("\n" + "=" * 80)
        print("📊 BULK PROCESSING COMPLETE")
        print("=" * 80)
        
        report = {
            "session_timestamp": self.timestamp,
            "criteria": LLM_CRITERIA,
            "model_used": MODEL,
            "confidence_threshold": CONFIDENCE_THRESHOLD,
            "total_urls_attempted": len(URLS_TO_CRAWL),
            "successful_urls": len(self.processed_urls),
            "failed_urls": len(self.failed_urls),
            "total_documents_downloaded": self.total_docs_downloaded,
            "total_relevant_documents": self.total_relevant_docs,
            "processed_urls": self.processed_urls,
            "failed_urls": self.failed_urls,
            "results_directory": str(self.session_dir)
        }
        
        # Save report
        report_file = self.session_dir / "final_report.json"
        with open(report_file, "w") as f:
            json.dump(report, f, indent=2)
        
        # Print summary
        print(f"✅ Successfully processed: {len(self.processed_urls)}/{len(URLS_TO_CRAWL)} URLs")
        print(f"📄 Total documents downloaded: {self.total_docs_downloaded}")
        print(f"🎯 Total relevant documents: {self.total_relevant_docs}")
        print(f"📁 Results saved to: {self.session_dir}")
        print(f"📊 Full report: {report_file}")
        
        if self.failed_urls:
            print(f"❌ Failed URLs: {len(self.failed_urls)}")
            for url in self.failed_urls:
                print(f"   - {url}")


def main():
    """Main function"""
    if not URLS_TO_CRAWL:
        print("❌ No URLs configured. Please edit URLS_TO_CRAWL in the script.")
        return
    
    if API_KEY == "your-api-key-here":
        print("❌ Please configure your API key in the script.")
        return
    
    processor = BulkProcessor()
    processor.process_all_urls()


if __name__ == "__main__":
    main()