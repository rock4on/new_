#!/usr/bin/env python3
"""
Batch PDF Downloader using FlareSolverr
Usage: python batch_pdf_downloader.py urls.txt
or: python batch_pdf_downloader.py url1 url2 url3 ...
"""

import sys
import time
import random
from pathlib import Path
from pdf_downloader import PDFDownloader

def load_urls_from_file(filename):
    """Load URLs from a text file (one URL per line)"""
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            urls = [line.strip() for line in f if line.strip() and not line.startswith('#')]
        return urls
    except Exception as e:
        print(f"❌ Error reading file {filename}: {e}")
        return []

def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python batch_pdf_downloader.py urls.txt")
        print("  python batch_pdf_downloader.py url1 url2 url3 ...")
        print("\nFor file input, create a text file with one URL per line:")
        print("  https://example.com/doc1.pdf")
        print("  https://example.com/doc2.pdf")
        print("  # This is a comment")
        sys.exit(1)
    
    # Determine if first argument is a file or URL
    first_arg = sys.argv[1]
    if Path(first_arg).exists():
        # Load URLs from file
        urls = load_urls_from_file(first_arg)
        print(f"📁 Loaded {len(urls)} URLs from {first_arg}")
    else:
        # Use command line arguments as URLs
        urls = sys.argv[1:]
        print(f"📝 Processing {len(urls)} URLs from command line")
    
    if not urls:
        print("❌ No URLs to process!")
        sys.exit(1)
    
    print(f"🚀 Batch PDF Downloader with FlareSolverr")
    print(f"📊 Total URLs: {len(urls)}")
    
    downloader = PDFDownloader()
    
    # Check if FlareSolverr is running
    try:
        import requests
        test_response = requests.get("http://localhost:8191/v1", timeout=5)
        if test_response.status_code == 405:
            print("✅ FlareSolverr is running")
        else:
            print("⚠️  FlareSolverr may not be running properly")
    except:
        print("❌ FlareSolverr is not running! Start it with:")
        print("docker run -d -p 8191:8191 ghcr.io/flaresolverr/flaresolverr:latest")
        sys.exit(1)
    
    # Create results directory
    results_dir = Path("downloads/batch_results")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    successful = 0
    failed = 0
    results = []
    
    for i, url in enumerate(urls, 1):
        print(f"\n{'='*60}")
        print(f"📥 Processing {i}/{len(urls)}: {url}")
        print(f"{'='*60}")
        
        # Generate filename from URL index and hash
        import hashlib
        url_hash = hashlib.sha1(url.encode()).hexdigest()[:8]
        filename = f"batch_{i:03d}_{url_hash}.pdf"
        
        try:
            success = downloader.download_pdf(url, filename)
            
            if success:
                successful += 1
                status = "✅ SUCCESS"
                print(f"✅ Successfully downloaded: {filename}")
            else:
                failed += 1
                status = "❌ FAILED"
                print(f"❌ Failed to download: {url}")
            
            results.append({
                "index": i,
                "url": url,
                "filename": filename,
                "status": status,
                "success": success
            })
            
        except Exception as e:
            failed += 1
            status = f"❌ ERROR: {e}"
            print(f"❌ Error processing {url}: {e}")
            
            results.append({
                "index": i,
                "url": url,
                "filename": filename,
                "status": status,
                "success": False
            })
        
        # Add delay between downloads to be polite
        if i < len(urls):  # Don't wait after the last one
            delay = random.uniform(5, 10)
            print(f"⏱️  Waiting {delay:.1f} seconds before next download...")
            time.sleep(delay)
    
    # Save results summary
    print(f"\n{'='*60}")
    print(f"📊 BATCH DOWNLOAD SUMMARY")
    print(f"{'='*60}")
    print(f"✅ Successful: {successful}")
    print(f"❌ Failed: {failed}")
    print(f"📊 Total: {len(urls)}")
    print(f"📈 Success rate: {(successful/len(urls)*100):.1f}%")
    
    # Save detailed results
    results_file = results_dir / f"batch_results_{int(time.time())}.txt"
    with open(results_file, 'w', encoding='utf-8') as f:
        f.write("Batch PDF Download Results\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Total URLs: {len(urls)}\n")
        f.write(f"Successful: {successful}\n")
        f.write(f"Failed: {failed}\n")
        f.write(f"Success rate: {(successful/len(urls)*100):.1f}%\n\n")
        f.write("Detailed Results:\n")
        f.write("-" * 30 + "\n")
        
        for result in results:
            f.write(f"{result['index']:3d}. {result['status']}\n")
            f.write(f"     URL: {result['url']}\n")
            f.write(f"     File: {result['filename']}\n\n")
    
    print(f"📄 Detailed results saved to: {results_file}")
    
    # List successful downloads
    successful_files = [r for r in results if r['success']]
    if successful_files:
        print(f"\n📁 Successfully downloaded files:")
        for result in successful_files:
            file_path = Path("downloads") / result['filename']
            if file_path.exists():
                size = file_path.stat().st_size
                print(f"  {result['filename']} ({size} bytes)")
    
    if failed > 0:
        print(f"\n❌ Failed downloads:")
        failed_results = [r for r in results if not r['success']]
        for result in failed_results:
            print(f"  {result['url']} - {result['status']}")
    
    print(f"\n🎉 Batch download completed!")

if __name__ == "__main__":
    main()