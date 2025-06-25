#!/usr/bin/env python3
"""
Simple test script to verify the regulation spider works
"""

import subprocess
import sys
from pathlib import Path

def test_spider_with_valid_urls():
    """Test the spider with known valid URLs"""
    
    # Test URLs that should work
    test_urls = [
        "https://httpbin.org/html",
        "https://example.com"
    ]
    
    regulation_name = "TEST_REGULATION"
    urls_str = ",".join(test_urls)
    
    print(f"🧪 Testing spider with URLs: {test_urls}")
    
    # Create test output directory
    test_output = Path("test_regulation_output")
    test_output.mkdir(exist_ok=True)
    
    cmd = [
        'scrapy', 'crawl', 'regulation',
        '-a', f'regulation_name={regulation_name}',
        '-a', f'start_urls={urls_str}',
        '-s', f'FILES_STORE={test_output}',
        '-s', 'LOG_LEVEL=DEBUG',
        '-s', 'CLOSESPIDER_PAGECOUNT=5'  # Limit pages for testing
    ]
    
    print(f"🚀 Running command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        
        print(f"Return code: {result.returncode}")
        print("\n📋 STDOUT:")
        print(result.stdout[-2000:])  # Last 2000 characters
        
        if result.stderr:
            print("\n❌ STDERR:")
            print(result.stderr[-1000:])  # Last 1000 characters
        
        # Check if any files were created
        created_files = list(test_output.glob('**/*'))
        print(f"\n📁 Created files ({len(created_files)}):")
        for f in created_files[:10]:  # Show first 10 files
            print(f"  {f}")
        
        if result.returncode == 0:
            print("✅ Spider test completed successfully")
        else:
            print("❌ Spider test failed")
            
    except subprocess.TimeoutExpired:
        print("⏰ Spider test timed out after 2 minutes")
    except Exception as e:
        print(f"❌ Error running spider test: {e}")

def test_url_parsing():
    """Test URL parsing logic"""
    print("\n🧪 Testing URL parsing logic...")
    
    test_cases = [
        "https://example.com,https://google.com",
        "www.example.com,google.com",
        "example.com,https://test.com",
        "https://site1.com, https://site2.com, www.site3.com"
    ]
    
    for test_case in test_cases:
        print(f"\nInput: {test_case}")
        
        # Simulate the spider's URL processing
        if isinstance(test_case, str):
            urls = [url.strip() for url in test_case.split(',') if url.strip()]
        else:
            urls = test_case
        
        validated_urls = []
        for url in urls:
            if not url.startswith(('http://', 'https://')):
                if url.startswith('www.'):
                    url = f'https://{url}'
                elif '.' in url:
                    url = f'https://{url}'
                else:
                    print(f"  Skipping invalid URL: {url}")
                    continue
            validated_urls.append(url)
        
        print(f"  Parsed URLs: {validated_urls}")

if __name__ == "__main__":
    print("🧪 Spider Test Script")
    print("=" * 50)
    
    test_url_parsing()
    
    print("\n" + "=" * 50)
    input("Press Enter to test the actual spider (or Ctrl+C to skip)...")
    
    test_spider_with_valid_urls()