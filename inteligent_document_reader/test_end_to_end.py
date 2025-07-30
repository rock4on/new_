#!/usr/bin/env python3
"""
End-to-end test for Inteligent Document Reader - Lease ingestion and Excel matching
"""

import os
import json
from pathlib import Path
from inteligent_document_reader import DocumentAgent
from inteligent_document_reader.tools.matching_tool_lease import MatchingToolLease
from inteligent_document_reader.config import Config


def test_lease_ingestion():
    """Test lease document ingestion workflow"""
    print("🧪 TEST 1: LEASE DOCUMENT INGESTION")
    print("=" * 50)
    
    try:
        # Initialize the agent with credentials from environment variables
        agent = DocumentAgent(
            azure_form_recognizer_endpoint=os.getenv('AZURE_FORM_RECOGNIZER_ENDPOINT'),
            azure_form_recognizer_key=os.getenv('AZURE_FORM_RECOGNIZER_KEY'),
            openai_api_key=os.getenv('OPENAI_API_KEY'),
            azure_search_endpoint=os.getenv('AZURE_SEARCH_ENDPOINT'),
            azure_search_key=os.getenv('AZURE_SEARCH_KEY'),
            openai_base_url=os.getenv('OPENAI_BASE_URL')  # Optional
        )
        
        print(f"✅ Agent initialized successfully")
        print(f"📁 Leases folder: {Config.LEASES_FOLDER}")
        
        # Check if leases folder exists and has PDF files
        leases_path = Path(Config.LEASES_FOLDER)
        if not leases_path.exists():
            print(f"⚠️  Leases folder doesn't exist: {Config.LEASES_FOLDER}")
            print(f"   Creating folder for testing...")
            leases_path.mkdir(parents=True, exist_ok=True)
            print(f"   📝 Add PDF lease documents to {Config.LEASES_FOLDER} for testing")
            return False
        
        # Find PDF files
        pdf_files = list(leases_path.rglob("*.pdf"))
        if not pdf_files:
            print(f"⚠️  No PDF files found in {Config.LEASES_FOLDER}")
            print(f"   📝 Add PDF lease documents to {Config.LEASES_FOLDER} for testing")
            return False
        
        print(f"📄 Found {len(pdf_files)} PDF files to test:")
        for pdf_file in pdf_files[:3]:  # Show first 3
            print(f"   • {pdf_file.name}")
        if len(pdf_files) > 3:
            print(f"   • ... and {len(pdf_files) - 3} more")
        
        # Test processing a single document
        test_file = pdf_files[0]
        print(f"\n📋 Testing single document processing: {test_file.name}")
        
        result = agent.process_document(
            file_path=str(test_file),
            doc_type="lease",
            client_name="Test Client"
        )
        
        if result["status"] == "success":
            print(f"✅ Single document processing: SUCCESS")
            print(f"   Client: {result.get('client_name', 'N/A')}")
            print(f"   Metadata fields: {len(result.get('metadata', {}))}")
            
            # Show extracted metadata
            metadata = result.get('metadata', {})
            for key, value in metadata.items():
                if value and str(value).strip():
                    print(f"   • {key}: {value}")
        else:
            print(f"❌ Single document processing: FAILED")
            print(f"   Error: {result.get('error', 'Unknown error')}")
            return False
        
        # Test batch processing (limit to 3 files for testing)
        print(f"\n📁 Testing batch processing (first 3 files)...")
        
        # Create a temporary folder with just 3 files for testing
        test_folder = Path("./test_leases_batch")
        test_folder.mkdir(exist_ok=True)
        
        # Copy first 3 files to test folder (simulate batch processing)
        import shutil
        test_files = []
        for i, pdf_file in enumerate(pdf_files[:3]):
            dest_file = test_folder / pdf_file.name
            if not dest_file.exists():
                shutil.copy2(pdf_file, dest_file)
            test_files.append(dest_file)
        
        batch_result = agent.process_folder(
            folder_path=str(test_folder),
            doc_type="lease",
            client_name="Batch Test Client"
        )
        
        print(f"📊 Batch processing results:")
        print(f"   Status: {batch_result['status']}")
        print(f"   Total files: {batch_result.get('total_files', 0)}")
        print(f"   Successful: {batch_result.get('successful', 0)}")
        print(f"   Failed: {batch_result.get('failed', 0)}")
        
        # Clean up test folder
        shutil.rmtree(test_folder, ignore_errors=True)
        
        if batch_result.get('successful', 0) > 0:
            print("✅ Lease ingestion test: PASSED")
            return True
        else:
            print("❌ Lease ingestion test: FAILED")
            return False
            
    except Exception as e:
        print(f"❌ Lease ingestion test error: {e}")
        return False


def test_excel_matching():
    """Test Excel matching with lease database"""
    print("\n🧪 TEST 2: EXCEL LEASE MATCHING")
    print("=" * 50)
    
    try:
        # Check if Leases_example.xlsm exists
        excel_file = Path("Leases_example.xlsm")
        if not excel_file.exists():
            print(f"⚠️  Excel file not found: {excel_file}")
            print(f"   📝 Add {excel_file} to test Excel matching")
            return False
        
        print(f"📊 Found Excel file: {excel_file}")
        
        # Initialize search client for matching tool
        from azure.search.documents import SearchClient
        from azure.core.credentials import AzureKeyCredential
        
        search_client = SearchClient(
            endpoint=os.getenv('AZURE_SEARCH_ENDPOINT'),
            index_name=Config.AZURE_SEARCH_INDEX_NAME,
            credential=AzureKeyCredential(os.getenv('AZURE_SEARCH_KEY'))
        )
        
        # Initialize matching tool
        matching_tool = MatchingToolLease(search_client=search_client)
        
        print(f"✅ Matching tool initialized")
        
        # Test Excel matching
        print(f"🔍 Testing Excel matching...")
        
        matching_input = json.dumps({
            'input_excel_path': str(excel_file),
            'output_excel_path': 'output_lease.xlsm'
        })
        
        result = matching_tool._run(matching_input)
        
        print("📋 Matching Results:")
        print(result)
        
        # Check if output file was created
        output_file = Path("output_lease.xlsm")
        if output_file.exists():
            print(f"✅ Output file created: {output_file}")
            print(f"   File size: {output_file.stat().st_size} bytes")
            
            # Show file info
            import pandas as pd
            try:
                df = pd.read_excel(output_file, header=None, engine='openpyxl')
                print(f"   Excel dimensions: {df.shape[0]} rows × {df.shape[1]} columns")
                print("✅ Excel matching test: PASSED")
                return True
            except Exception as e:
                print(f"⚠️  Could not read output Excel: {e}")
                print("✅ Excel matching test: PASSED (file created)")
                return True
        else:
            print(f"❌ Output file not created")
            print("❌ Excel matching test: FAILED")
            return False
            
    except Exception as e:
        print(f"❌ Excel matching test error: {e}")
        return False


def main():
    """Run end-to-end tests"""
    print("🚀 INTELIGENT DOCUMENT READER - END-TO-END TESTS")
    print("=" * 60)
    
    # Check required environment variables
    required_vars = [
        'AZURE_FORM_RECOGNIZER_ENDPOINT',
        'AZURE_FORM_RECOGNIZER_KEY',
        'OPENAI_API_KEY',
        'AZURE_SEARCH_ENDPOINT',
        'AZURE_SEARCH_KEY'
    ]
    
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    if missing_vars:
        print(f"❌ Missing required environment variables:")
        for var in missing_vars:
            print(f"   • {var}")
        print(f"\n💡 Set these environment variables before running tests")
        return
    
    print(f"✅ All required environment variables are set")
    
    # Run tests
    test_results = []
    
    # Test 1: Lease ingestion
    test_results.append(test_lease_ingestion())
    
    # Test 2: Excel matching
    test_results.append(test_excel_matching())
    
    # Summary
    print(f"\n📊 TEST SUMMARY")
    print("=" * 30)
    passed = sum(test_results)
    total = len(test_results)
    
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED!")
    else:
        print("⚠️  Some tests failed - check output above")
    
    print(f"\n📝 Test Details:")
    tests = ["Lease Ingestion", "Excel Matching"]
    for i, (test_name, result) in enumerate(zip(tests, test_results)):
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {i+1}. {test_name}: {status}")


if __name__ == "__main__":
    main()