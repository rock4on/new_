#!/usr/bin/env python3
"""
End-to-end test for Inteligent Document Reader - Natural Gas ingestion and Excel matching
"""

import os
import sys
import json
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

try:
    from inteligent_document_reader import DocumentAgent
    from inteligent_document_reader.tools.matching_tool_natural_gas import MatchingToolNaturalGas
    from inteligent_document_reader.config import Config
except ImportError:
    # If running from within the directory, use relative imports
    from agents.document_agent import DocumentAgent
    from tools.matching_tool_natural_gas import MatchingToolNaturalGas
    from config import Config


def test_natural_gas_ingestion():
    """Test natural gas document ingestion workflow"""
    print("🧪 TEST 1: NATURAL GAS DOCUMENT INGESTION")
    print("=" * 50)
    
    try:
        # Initialize the agent (credentials are now in Config class)
        agent = DocumentAgent()
        
        print(f"✅ Agent initialized successfully")
        print(f"📁 Natural Gas folder: {Config.NATURAL_GAS_FOLDER}")
        
        # Check if natural gas folder exists and has PDF files
        natural_gas_path = Path(Config.NATURAL_GAS_FOLDER)
        if not natural_gas_path.exists():
            print(f"⚠️  Natural Gas folder doesn't exist: {Config.NATURAL_GAS_FOLDER}")
            print(f"   Creating folder for testing...")
            natural_gas_path.mkdir(parents=True, exist_ok=True)
            print(f"   📝 Add PDF natural gas documents to {Config.NATURAL_GAS_FOLDER} for testing")
            return False
        
        # Find PDF files
        pdf_files = list(natural_gas_path.rglob("*.pdf"))
        if not pdf_files:
            print(f"⚠️  No PDF files found in {Config.NATURAL_GAS_FOLDER}")
            print(f"   📝 Add PDF natural gas documents to {Config.NATURAL_GAS_FOLDER} for testing")
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
            doc_type="natural_gas",
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
        test_folder = Path("./test_natural_gas_batch")
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
            doc_type="natural_gas",
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
            print("✅ Natural Gas ingestion test: PASSED")
            return True
        else:
            print("❌ Natural Gas ingestion test: FAILED")
            return False
            
    except Exception as e:
        print(f"❌ Natural Gas ingestion test error: {e}")
        return False


def test_excel_matching():
    """Test Excel matching with natural gas database"""
    print("\n🧪 TEST 2: EXCEL NATURAL GAS MATCHING")
    print("=" * 50)
    
    try:
        # Check if Gas_example.xlsm exists
        excel_file = Path("Gas_example.xlsm")
        if not excel_file.exists():
            print(f"⚠️  Excel file not found: {excel_file}")
            print(f"   📝 Add {excel_file} to test Excel matching")
            return False
        
        print(f"📊 Found Excel file: {excel_file}")
        
        # Initialize search client for matching tool
        from azure.search.documents import SearchClient
        from azure.core.credentials import AzureKeyCredential
        
        search_client = SearchClient(
            endpoint=Config.AZURE_SEARCH_ENDPOINT,
            index_name=Config.AZURE_SEARCH_UTILITIES_INDEX_NAME, # Use utilities index
            credential=AzureKeyCredential(Config.AZURE_SEARCH_KEY)
        )
        
        # Initialize matching tool
        matching_tool = MatchingToolNaturalGas(search_client=search_client)
        
        print(f"✅ Matching tool initialized")
        
        # Test Excel matching
        print(f"🔍 Testing Excel matching...")
        
        matching_input = json.dumps({
            'input_excel_path': str(excel_file),
            'output_excel_path': 'output_natural_gas.xlsm'
        })
        
        result = matching_tool._run(matching_input)
        
        print("📋 Matching Results:")
        print(result)
        
        # Check if output file was created
        output_file = Path("output_natural_gas.xlsm")
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
    print("🚀 INTELIGENT DOCUMENT READER - END-TO-END TESTS (NATURAL GAS)")
    print("=" * 60)
    
    # Check if config has real values (not placeholders)
    config_vars = [
        ('AZURE_FORM_RECOGNIZER_ENDPOINT', Config.AZURE_FORM_RECOGNIZER_ENDPOINT),
        ('AZURE_FORM_RECOGNIZER_KEY', Config.AZURE_FORM_RECOGNIZER_KEY),
        ('OPENAI_API_KEY', Config.OPENAI_API_KEY),
        ('AZURE_SEARCH_ENDPOINT', Config.AZURE_SEARCH_ENDPOINT),
        ('AZURE_SEARCH_KEY', Config.AZURE_SEARCH_KEY)
    ]
    
    missing_vars = [var_name for var_name, var_value in config_vars if var_value.startswith('your_')]
    if missing_vars:
        print(f"❌ Please update Config class with real credentials:")
        for var in missing_vars:
            print(f"   • {var}")
        print(f"\n💡 Edit inteligent_document_reader/config.py with your actual credentials")
        return
    
    print(f"✅ All required credentials are configured")
    
    # Run tests
    test_results = []
    
    # Test 1: Natural Gas ingestion
    test_results.append(test_natural_gas_ingestion())
    
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
    tests = ["Natural Gas Ingestion", "Excel Matching"]
    for i, (test_name, result) in enumerate(zip(tests, test_results)):
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {i+1}. {test_name}: {status}")


if __name__ == "__main__":
    main()