#!/usr/bin/env python3
"""
Example Usage: Integrated Lease and Utilities Document Analysis System
Shows how to use the enhanced lease_agent.py with natural gas and electricity support
"""

import os
from lease_agent import create_lease_agent_from_config
from config import get_config

def main():
    """Demonstrate the integrated lease and utilities system"""
    
    print("🏢 INTEGRATED LEASE & UTILITIES DOCUMENT ANALYSIS SYSTEM")
    print("=" * 60)
    
    # Initialize the agent (now supports both lease and utilities documents)
    config = get_config()
    agent = create_lease_agent_from_config(config)
    
    print("\n🎯 EXAMPLE 1: Basic Conversational Interface")
    print("-" * 50)
    
    # Basic conversation
    result = agent.ask_question("Hi, what can you help me with?")
    print("Agent:", result['result'])
    
    print("\n🎯 EXAMPLE 2: Process Mixed Document Types")
    print("-" * 50)
    
    # The agent can now handle both lease documents and utilities documents
    # It will automatically detect the document type and use appropriate tools
    
    # Process lease documents (existing functionality)
    lease_result = agent.ask_question("Find all lease documents for client2")
    print("Lease Search Result:", lease_result['result'][:500] + "...")
    
    # Process utilities documents (new functionality)
    utilities_result = agent.ask_question("Find all natural gas and electricity documents for Chicago")
    print("Utilities Search Result:", utilities_result['result'][:500] + "...")
    
    print("\n🎯 EXAMPLE 3: Location-Based Analysis (Works for Both Document Types)")
    print("-" * 50)
    
    # This will search across BOTH lease documents AND utilities documents
    mixed_result = agent.ask_question("Show me all documents (leases and utilities) for Helsinki")
    print("Mixed Search Result:", mixed_result['result'][:500] + "...")
    
    print("\n🎯 EXAMPLE 4: Utilities-Specific Queries")
    print("-" * 50)
    
    # Natural gas consumption analysis
    gas_result = agent.ask_question("What's the total natural gas consumption across all locations?")
    print("Gas Analysis:", gas_result['result'][:400] + "...")
    
    # Electricity bill analysis
    electricity_result = agent.ask_question("Find electricity bills with consumption over 1000 kWh")
    print("Electricity Analysis:", electricity_result['result'][:400] + "...")
    
    # Vendor analysis
    vendor_result = agent.ask_question("Which vendors supply the most utilities to our locations?")
    print("Vendor Analysis:", vendor_result['result'][:400] + "...")
    
    print("\n🎯 EXAMPLE 5: Direct Method Calls")
    print("-" * 50)
    
    # Direct method calls for specific functionality
    
    # Process a utilities document specifically  
    utilities_doc_result = agent.process_utilities_document(
        file_path="/path/to/gas_bill.pdf",
        utility_type="natural_gas",
        client_name="client2"
    )
    print("Utilities Processing Result:", utilities_doc_result)
    
    # Find utilities by location
    utilities_location_result = agent.find_utilities_by_location("Chicago")
    print("Utilities by Location:", utilities_location_result['status'])
    
    # Search utilities for specific fields
    utilities_fields_result = agent.search_utilities_for_fields(
        search_query="high consumption 2024",
        fields=["vendor_name", "consumption_amount", "unit_of_measure", "location"]
    )
    print("Utilities Field Search:", utilities_fields_result['status'])
    
    print("\n🎯 EXAMPLE 6: Batch Processing (Separate Folders)")
    print("-" * 50)
    
    # The system now has 3 separate folders and 3 separate ingestion tools
    print("📁 Folder structure:")
    print("   /leases - for lease documents")
    print("   /electric - for electric bills") 
    print("   /natural_gas - for natural gas bills")
    print()
    
    # Batch ingest lease documents
    lease_batch_result = agent.ask_question("Ingest all lease documents")
    print("Lease Batch Processing:", lease_batch_result['result'][:300] + "...")
    
    # Batch ingest electric documents  
    electric_batch_result = agent.ask_question("Process all electric bills")
    print("Electric Batch Processing:", electric_batch_result['result'][:300] + "...")
    
    # Batch ingest natural gas documents
    gas_batch_result = agent.ask_question("Ingest natural gas documents") 
    print("Natural Gas Batch Processing:", gas_batch_result['result'][:300] + "...")
    
    print("\n🎯 EXAMPLE 7: Advanced Analysis Queries")
    print("-" * 50)
    
    # Complex analysis that combines both document types
    advanced_queries = [
        "Compare lease costs vs utilities costs for Chicago locations",
        "Find locations with both office leases and high electricity consumption",
        "Which clients have the most lease agreements and highest gas usage?",
        "Show me all documents expiring in 2024 (both leases and utility contracts)",
        "Analyze the relationship between building area and electricity consumption"
    ]
    
    for query in advanced_queries:
        result = agent.ask_question(query)
        print(f"\nQuery: {query}")
        print(f"Result: {result['result'][:300]}...")


def interactive_demo():
    """Interactive demo where user can ask questions"""
    
    print("\n🤖 INTERACTIVE DEMO MODE")
    print("=" * 40)
    print("Ask questions about leases, natural gas, or electricity documents!")
    print("Type 'quit' to exit\n")
    
    config = get_config()
    agent = create_lease_agent_from_config(config)
    
    while True:
        user_input = input("You: ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("Goodbye! 👋")
            break
        
        if not user_input:
            continue
            
        try:
            result = agent.ask_question(user_input)
            print(f"\nAgent: {result['result']}\n")
            print("-" * 50)
        except Exception as e:
            print(f"Error: {e}")


def example_queries():
    """Show example queries that work with the integrated system"""
    
    print("\n📝 EXAMPLE QUERIES YOU CAN TRY:")
    print("=" * 40)
    
    lease_queries = [
        "Ingest all lease documents",  # processes /leases folder
        "Find all office leases in Chicago",
        "What leases expire in 2024?", 
        "Show me leases for client2",
        "Which buildings have the largest area?",
        "Give me a complete lease portfolio summary"
    ]
    
    utilities_queries = [
        "Process all electric bills",  # processes /electric folder
        "Ingest natural gas documents",  # processes /natural_gas folder
        "Find natural gas bills for Chicago",
        "What's the electricity consumption for client2?",
        "Which vendor supplies the most gas?",
        "Show me high consumption electricity bills",
        "Compare gas vs electricity usage by location"
    ]
    
    mixed_queries = [
        "Find all documents for Helsinki (leases and utilities)",
        "Show me everything for client2",
        "Which locations have both leases and utilities?",
        "Compare building costs vs utilities costs",
        "Analyze energy efficiency: large buildings vs consumption"
    ]
    
    print("\n🏢 LEASE DOCUMENT QUERIES:")
    for i, query in enumerate(lease_queries, 1):
        print(f"{i}. {query}")
    
    print("\n⚡ UTILITIES DOCUMENT QUERIES:")
    for i, query in enumerate(utilities_queries, 1):
        print(f"{i}. {query}")
    
    print("\n🔄 MIXED ANALYSIS QUERIES:")
    for i, query in enumerate(mixed_queries, 1):
        print(f"{i}. {query}")


if __name__ == "__main__":
    print("Choose a demo mode:")
    print("1. Run example usage")
    print("2. Interactive demo")
    print("3. Show example queries")
    
    choice = input("Enter choice (1-3): ").strip()
    
    if choice == "1":
        main()
    elif choice == "2":
        interactive_demo()
    elif choice == "3":
        example_queries()
    else:
        print("Invalid choice. Running example usage...")
        main()