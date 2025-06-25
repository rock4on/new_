#!/usr/bin/env python3
"""
Debug script to test pipeline components individually
"""

import sys
import json
import subprocess
from pathlib import Path
import requests

def test_scrapy_setup():
    """Test if Scrapy is properly set up"""
    print("🔍 Testing Scrapy setup...")
    
    try:
        result = subprocess.run(['scrapy', 'list'], capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            spiders = result.stdout.strip().split('\n')
            print(f"✅ Scrapy is working. Available spiders: {spiders}")
            
            if 'regulation' in spiders:
                print("✅ regulation spider found")
            else:
                print("❌ regulation spider NOT found")
                print("Available spiders:", spiders)
            
            if 'docs' in spiders:
                print("✅ docs spider found")
            else:
                print("❌ docs spider NOT found")
        else:
            print(f"❌ Scrapy error: {result.stderr}")
    except Exception as e:
        print(f"❌ Scrapy test failed: {e}")

def test_flaresolverr():
    """Test FlareSolverr connection"""
    print("\n🔍 Testing FlareSolverr...")
    
    try:
        response = requests.get("http://localhost:8191/v1", timeout=5)
        if response.status_code in [200, 405]:
            print("✅ FlareSolverr is running")
        else:
            print(f"⚠️  FlareSolverr status: {response.status_code}")
    except Exception as e:
        print(f"❌ FlareSolverr not accessible: {e}")

def test_excel_reader():
    """Test Excel reader with sample data"""
    print("\n🔍 Testing Excel reader...")
    
    try:
        from excel_reader import ExcelReader
        print("✅ Excel reader imports successfully")
        
        # Create a simple test
        print("Create a test Excel file to verify Excel reading works")
        
    except Exception as e:
        print(f"❌ Excel reader test failed: {e}")

def test_downloaders():
    """Test HTML and PDF downloaders"""
    print("\n🔍 Testing downloaders...")
    
    try:
        from html_downloader import HTMLDownloader
        from pdf_downloader import PDFDownloader
        print("✅ Downloaders import successfully")
        
        # Test with a simple URL
        html_downloader = HTMLDownloader()
        print("✅ HTML downloader initialized")
        
        pdf_downloader = PDFDownloader()
        print("✅ PDF downloader initialized")
        
    except Exception as e:
        print(f"❌ Downloader test failed: {e}")

def test_spider_directly():
    """Test running the spider directly"""
    print("\n🔍 Testing spider directly...")
    
    try:
        test_url = "https://httpbin.org/html"  # Simple test URL
        cmd = [
            'scrapy', 'crawl', 'regulation',
            '-a', f'regulation_name=TEST',
            '-a', f'start_urls={test_url}',
            '-s', 'CLOSESPIDER_PAGECOUNT=2',  # Limit to 2 pages
            '-L', 'DEBUG'
        ]
        
        print(f"🚀 Running: {' '.join(cmd)}")
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        
        print(f"Return code: {result.returncode}")
        print(f"STDOUT (last 1000 chars): {result.stdout[-1000:]}")
        if result.stderr:
            print(f"STDERR (last 1000 chars): {result.stderr[-1000:]}")
            
    except subprocess.TimeoutExpired:
        print("⏰ Spider test timed out")
    except Exception as e:
        print(f"❌ Spider test failed: {e}")

def check_project_structure():
    """Check if all required files exist"""
    print("\n🔍 Checking project structure...")
    
    required_files = [
        'excel_reader.py',
        'html_downloader.py', 
        'pdf_downloader.py',
        'regulation_pipeline.py',
        'documents/spiders/regulation_spider.py',
        'documents/items.py',
        'documents/settings.py'
    ]
    
    for file_path in required_files:
        if Path(file_path).exists():
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path} - MISSING")

def main():
    print("🔧 Pipeline Debug Tool")
    print("=" * 50)
    
    check_project_structure()
    test_flaresolverr()
    test_scrapy_setup()
    test_excel_reader()
    test_downloaders()
    test_spider_directly()
    
    print("\n" + "=" * 50)
    print("🔧 Debug completed. Check the results above.")

if __name__ == "__main__":
    main()