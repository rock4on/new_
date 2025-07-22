#!/usr/bin/env python3
"""
Connection test script for the Lease Document Agent
Tests all required services independently to isolate connection issues
"""

import sys
from pathlib import Path

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

from config import get_config


def test_openai_connection(config):
    """Test OpenAI API connection"""
    print("🔍 Testing OpenAI Connection...")
    print(f"   Model: {config.OPENAI_MODEL}")
    if config.OPENAI_BASE_URL:
        print(f"   Base URL: {config.OPENAI_BASE_URL}")
    
    try:
        import openai
        
        # Initialize client
        openai_kwargs = {"api_key": config.OPENAI_API_KEY}
        if config.OPENAI_BASE_URL:
            openai_kwargs["base_url"] = config.OPENAI_BASE_URL
        
        client = openai.OpenAI(**openai_kwargs)
        
        # Test 1: List models
        print("   Testing model list...")
        models = client.models.list()
        print(f"   ✅ Models available: {len(list(models.data))}")
        
        # Test 2: Simple completion
        print("   Testing chat completion...")
        response = client.chat.completions.create(
            model=config.OPENAI_MODEL,
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=10
        )
        print(f"   ✅ Chat completion successful: {response.choices[0].message.content[:20]}...")
        
        # Test 3: Embedding
        print("   Testing embedding generation...")
        embedding_response = client.embeddings.create(
            model=config.OPENAI_EMBEDDING_MODEL,
            input="test"
        )
        print(f"   ✅ Embedding successful: {len(embedding_response.data[0].embedding)} dimensions")
        
        return True
        
    except Exception as e:
        print(f"   ❌ OpenAI connection failed: {e}")
        return False


def test_azure_form_recognizer(config):
    """Test Azure Form Recognizer connection"""
    print("\n🔍 Testing Azure Form Recognizer Connection...")
    print(f"   Endpoint: {config.AZURE_FORM_RECOGNIZER_ENDPOINT}")
    
    try:
        from azure.ai.formrecognizer import DocumentAnalysisClient
        from azure.core.credentials import AzureKeyCredential
        
        client = DocumentAnalysisClient(
            endpoint=config.AZURE_FORM_RECOGNIZER_ENDPOINT,
            credential=AzureKeyCredential(config.AZURE_FORM_RECOGNIZER_KEY)
        )
        
        # We can't easily test without a document, but the client initialization
        # will fail if the endpoint/key are completely wrong
        print("   ✅ Client initialized successfully")
        print("   ⚠️  Note: Full test requires a PDF document")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Azure Form Recognizer connection failed: {e}")
        return False


def test_azure_search(config):
    """Test Azure AI Search connection"""
    print("\n🔍 Testing Azure AI Search Connection...")
    print(f"   Endpoint: {config.AZURE_SEARCH_ENDPOINT}")
    print(f"   Index: {config.AZURE_SEARCH_INDEX_NAME}")
    
    try:
        from azure.search.documents import SearchClient
        from azure.core.credentials import AzureKeyCredential
        
        client = SearchClient(
            endpoint=config.AZURE_SEARCH_ENDPOINT,
            index_name=config.AZURE_SEARCH_INDEX_NAME,
            credential=AzureKeyCredential(config.AZURE_SEARCH_KEY)
        )
        
        # Test connection with a simple search
        try:
            results = list(client.search("*", top=1))
            print(f"   ✅ Search successful: Found index with {len(results)} test results")
        except Exception as search_e:
            if "404" in str(search_e) or "not found" in str(search_e).lower():
                print(f"   ⚠️  Index '{config.AZURE_SEARCH_INDEX_NAME}' not found (will be created on first upload)")
                print("   ✅ Service connection successful")
            else:
                raise search_e
        
        return True
        
    except Exception as e:
        print(f"   ❌ Azure Search connection failed: {e}")
        return False


def test_langchain_integration(config):
    """Test LangChain integration"""
    print("\n🔍 Testing LangChain Integration...")
    
    try:
        from langchain_openai import ChatOpenAI
        
        # Initialize LangChain LLM
        langchain_kwargs = {
            "model": config.OPENAI_MODEL,
            "api_key": config.OPENAI_API_KEY,
            "temperature": config.OPENAI_TEMPERATURE
        }
        if config.OPENAI_BASE_URL:
            langchain_kwargs["base_url"] = config.OPENAI_BASE_URL
        
        llm = ChatOpenAI(**langchain_kwargs)
        
        # Test with a simple invocation
        response = llm.invoke("Hello, respond with just 'OK'")
        print(f"   ✅ LangChain integration successful: {response.content}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ LangChain integration failed: {e}")
        return False


def main():
    """Run all connection tests"""
    print("🤖 LEASE DOCUMENT AGENT - CONNECTION TESTS")
    print("=" * 50)
    
    # Load configuration
    try:
        config = get_config()
        is_valid, missing_fields = config.validate()
        
        if not is_valid:
            print("❌ Configuration is incomplete!")
            print("Missing required fields:")
            for field in missing_fields:
                print(f"   - {field}")
            print(f"\n💡 Run 'python config.py' to create a sample configuration")
            return False
        
        print("✅ Configuration loaded successfully")
        
    except Exception as e:
        print(f"❌ Failed to load configuration: {e}")
        return False
    
    # Run all tests
    tests = [
        ("OpenAI API", test_openai_connection),
        ("Azure Form Recognizer", test_azure_form_recognizer),
        ("Azure AI Search", test_azure_search),
        ("LangChain Integration", test_langchain_integration)
    ]
    
    results = {}
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func(config)
        except Exception as e:
            print(f"   ❌ {test_name} test crashed: {e}")
            results[test_name] = False
    
    # Summary
    print("\n📊 TEST SUMMARY")
    print("=" * 30)
    
    passed = 0
    total = len(tests)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Agent should work correctly.")
        return True
    else:
        print("⚠️  Some tests failed. Fix the issues before using the agent.")
        
        # Provide troubleshooting tips
        print("\n💡 Troubleshooting Tips:")
        if not results.get("OpenAI API"):
            print("   - Check OpenAI API key and permissions")
            print("   - Verify base URL if using Azure OpenAI")
            print("   - Test with curl: curl -H 'Authorization: Bearer YOUR_KEY' https://api.openai.com/v1/models")
        
        if not results.get("Azure Form Recognizer"):
            print("   - Check Azure Form Recognizer endpoint and key")
            print("   - Ensure service is deployed and running")
            print("   - Verify region matches your key")
        
        if not results.get("Azure AI Search"):
            print("   - Check Azure Search endpoint and key")
            print("   - Verify search service is running")
            print("   - Index will be created automatically on first upload")
        
        return False


if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)