#!/usr/bin/env python3
"""
Example usage of the Lease Document Agent
Demonstrates how to use the agentic framework for various document processing tasks
"""

import os
import sys
from pathlib import Path
import json

# Add the current directory to the path to import the lease_agent
sys.path.append(str(Path(__file__).parent))

from lease_agent import LeaseDocumentAgent, create_lease_agent_from_env


def main():
    """Demonstrate various agent capabilities"""
    
    print("🤖 Lease Document Agent - Example Usage")
    print("=" * 50)
    
    # Method 1: Create agent from environment variables
    try:
        agent = create_lease_agent_from_env()
        print("✅ Agent created successfully from environment variables")
    except Exception as e:
        print(f"❌ Failed to create agent from environment: {e}")
        print("\n💡 Make sure these environment variables are set:")
        print("   - AZURE_FORM_RECOGNIZER_ENDPOINT")
        print("   - AZURE_FORM_RECOGNIZER_KEY")
        print("   - OPENAI_API_KEY")
        print("   - AZURE_SEARCH_ENDPOINT")
        print("   - AZURE_SEARCH_KEY")
        print("   - AZURE_SEARCH_INDEX_NAME (optional, defaults to 'lease-documents')")
        
        # Method 2: Create agent with explicit parameters (example)
        print("\n🔧 Alternative: Create agent with explicit parameters")
        print("agent = LeaseDocumentAgent(")
        print("    azure_endpoint='your_azure_endpoint',")
        print("    azure_key='your_azure_key',")
        print("    openai_api_key='your_openai_key',")
        print("    azure_search_endpoint='your_search_endpoint',")
        print("    azure_search_key='your_search_key',")
        print("    search_index_name='your_index_name'")
        print(")")
        return
    
    # Example 1: Process a single document
    print("\n📄 Example 1: Processing a single document")
    print("-" * 40)
    
    # Assuming you have a PDF file in the leases directory
    sample_pdf = Path("../leases/sample_lease.pdf")  # Adjust path as needed
    
    if sample_pdf.exists():
        print(f"Processing document: {sample_pdf}")
        result = agent.process_document(
            file_path=str(sample_pdf),
            client_name="Example Client"
        )
        print("Result:", json.dumps(result, indent=2))
    else:
        print(f"⚠️  Sample file not found at {sample_pdf}")
        print("   Create a sample PDF file or adjust the path")
    
    # Example 2: Search for documents by location
    print("\n🏢 Example 2: Finding documents by location")
    print("-" * 40)
    
    result = agent.find_documents_by_location("New York")
    print("Location search result:", json.dumps(result, indent=2))
    
    # Example 3: Vector search for specific fields
    print("\n🔍 Example 3: Vector search for specific fields")
    print("-" * 40)
    
    result = agent.search_for_fields(
        search_query="office building lease agreements",
        fields=["location", "lease_start_date", "lease_end_date", "building_area"]
    )
    print("Field search result:", json.dumps(result, indent=2))
    
    # Example 4: Ask general questions
    print("\n❓ Example 4: Asking general questions")
    print("-" * 40)
    
    questions = [
        "What lease documents do we have for office buildings?",
        "Find all leases that expire in 2024",
        "Which client has the largest building area?",
        "List all documents with missing location information"
    ]
    
    for question in questions:
        print(f"\nQuestion: {question}")
        result = agent.ask_question(question)
        print("Answer:", result.get("result", result.get("error", "No response")))
    
    # Example 5: Custom workflow
    print("\n🔄 Example 5: Custom workflow demonstration")
    print("-" * 40)
    
    workflow_query = """
    I need to analyze all lease documents to create a summary report:
    1. Find all documents in the vector store
    2. Extract key information: location, lease dates, building type, area
    3. Identify any missing information
    4. Provide a summary of findings
    """
    
    result = agent.ask_question(workflow_query)
    print("Workflow result:", json.dumps(result, indent=2))
    
    print("\n✅ Example usage completed!")
    print("\n💡 The agent uses ReAct pattern to:")
    print("   - Think about each step")
    print("   - Choose appropriate tools")
    print("   - Execute actions")
    print("   - Observe results")
    print("   - Provide final answers")


def batch_process_example():
    """Example of processing multiple documents"""
    
    print("\n📁 Batch Processing Example")
    print("-" * 40)
    
    try:
        agent = create_lease_agent_from_env()
    except Exception as e:
        print(f"❌ Cannot create agent: {e}")
        return
    
    # Process all PDFs in the leases directory
    leases_dir = Path("../leases")
    
    if not leases_dir.exists():
        print(f"⚠️  Leases directory not found: {leases_dir}")
        return
    
    pdf_files = list(leases_dir.glob("**/*.pdf"))
    
    if not pdf_files:
        print("⚠️  No PDF files found in leases directory")
        return
    
    print(f"📄 Found {len(pdf_files)} PDF files to process")
    
    results = []
    
    for pdf_file in pdf_files[:3]:  # Process first 3 files as example
        print(f"\nProcessing: {pdf_file.name}")
        
        # Extract client name from directory structure if applicable
        client_name = pdf_file.parent.name if pdf_file.parent.name != "leases" else "Unknown"
        
        result = agent.process_document(
            file_path=str(pdf_file),
            client_name=client_name
        )
        
        results.append(result)
        print(f"Status: {result['status']}")
    
    # Save results
    results_file = Path("batch_processing_results.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved to: {results_file}")


def batch_ingest_example():
    """Example of the new batch ingestion functionality"""
    
    print("\n📦 Batch Ingestion Example")
    print("-" * 40)
    
    try:
        agent = create_lease_agent_from_env()
    except Exception as e:
        print(f"❌ Cannot create agent: {e}")
        return
    
    # Example 1: Ingest all PDFs from default folder (no path needed)
    print("\n🚀 Example 1: Ingest all PDFs from default folder")
    print("   Simply call agent.ingest_all() - no path needed!")
    
    result = agent.ingest_all()
    
    print(f"Status: {result['status']}")
    print(f"Total files found: {result['total_files']}")
    print(f"Successfully processed: {result['successful']}")
    print(f"Failed: {result['failed']}")
    print(f"Folder processed: {result['folder_path']}")
    
    # Example 2: Batch ingest with specific client name
    print("\n🏢 Example 2: Batch ingest with specific client name")
    result_with_client = agent.ingest_all(client_name="Batch Client Example")
    
    print(f"Status: {result_with_client['status']}")
    print(f"All documents assigned to client: 'Batch Client Example'")
    
    # Example 3: Batch ingest from specific folder
    print("\n📁 Example 3: Batch ingest from specific folder")
    custom_folder = "../custom_leases"  # Example custom path
    
    result_custom = agent.batch_ingest_pdfs(
        folder_path=custom_folder,
        client_name="Custom Folder Client"
    )
    
    if result_custom['status'] == 'error':
        print(f"⚠️  Custom folder example: {result_custom['error']}")
    else:
        print(f"Status: {result_custom['status']}")
        print(f"Total files: {result_custom['total_files']}")
    
    print("\n✅ Batch ingestion examples completed!")
    print("\n💡 Key benefits of batch ingestion:")
    print("   • No need to specify individual file paths")
    print("   • Automatically discovers all PDFs in folders")
    print("   • Supports recursive folder scanning")
    print("   • Handles errors gracefully for individual files")
    print("   • Provides detailed progress reporting")


if __name__ == "__main__":
    # Run main examples
    main()
    
    # Run new batch ingestion example
    print("\n" + "=" * 50)
    batch_ingest_example()
    
    # Optionally run original batch processing example
    print("\n" + "=" * 50)
    batch_process_example()