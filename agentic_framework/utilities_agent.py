#!/usr/bin/env python3
"""
Utilities Document Agent Factory
Separated utilities processing functionality for better code organization
"""

from typing import List, Dict, Any, Optional
from utilities_tools import (
    UtilitiesOCRTool, 
    UtilitiesIngestionTool, 
    UtilitiesMatchingTool, 
    UtilitiesVectorSearchTool
)


def create_utilities_tools(azure_endpoint: str, azure_key: str, openai_client, search_client, embedding_model: str) -> List:
    """
    Create utilities-specific tools for the agent
    
    Args:
        azure_endpoint: Azure Form Recognizer endpoint
        azure_key: Azure Form Recognizer key
        openai_client: OpenAI client instance
        search_client: Azure Search client instance
        embedding_model: Embedding model to use
        
    Returns:
        List of utilities tools
    """
    return [
        UtilitiesOCRTool(azure_endpoint=azure_endpoint, azure_key=azure_key),
        UtilitiesIngestionTool(
            openai_client=openai_client,
            search_client=search_client,
            embedding_model=embedding_model
        ),
        UtilitiesMatchingTool(search_client=search_client),
        UtilitiesVectorSearchTool(
            openai_client=openai_client,
            search_client=search_client,
            embedding_model=embedding_model
        )
    ]


class UtilitiesAnalysisTool:
    """Tool for comprehensive utilities portfolio analysis"""
    
    def __init__(self, search_client):
        self.search_client = search_client
    
    def analyze_utilities_portfolio(self, analysis_type: str = "all") -> str:
        """Perform comprehensive utilities analysis"""
        try:
            print(f"   📊 Performing utilities analysis: {analysis_type}")
            
            # Get all utilities documents
            search_results = self.search_client.search(
                search_text="*",
                filter="doc_type eq 'NaturalGas' or doc_type eq 'Electricity'",
                select=["filename", "location", "client_name", "vendor_name", "invoice_date",
                       "measurement_period_start", "measurement_period_end", "consumption_amount", 
                       "unit_of_measure", "doc_type", "processed_at"],
                top=1000
            )
            
            all_results = []
            for result in search_results:
                all_results.append({
                    "filename": result.get("filename"),
                    "location": result.get("location"),
                    "client_name": result.get("client_name"),
                    "vendor_name": result.get("vendor_name"),
                    "invoice_date": result.get("invoice_date"),
                    "measurement_period_start": result.get("measurement_period_start"),
                    "measurement_period_end": result.get("measurement_period_end"),
                    "consumption_amount": result.get("consumption_amount"),
                    "unit_of_measure": result.get("unit_of_measure"),
                    "doc_type": result.get("doc_type"),
                    "processed_at": result.get("processed_at")
                })
            
            # Group by filename to avoid counting pages as separate documents
            document_map = {}
            for result in all_results:
                filename = result.get("filename", "Unknown")
                if filename not in document_map:
                    document_map[filename] = result
            
            documents = list(document_map.values())
            
            if not documents:
                return "No utilities documents found in the system. Please upload utilities documents first."
            
            # Comprehensive analysis
            analysis = f"⚡ COMPREHENSIVE UTILITIES PORTFOLIO ANALYSIS\n"
            analysis += f"{'='*60}\n\n"
            
            # Basic statistics
            total_docs = len(documents)
            clients = set()
            locations = set()
            doc_types = {}
            vendors = set()
            total_consumption_by_type = {}
            
            for doc in documents:
                # Client analysis
                if doc.get("client_name"):
                    clients.add(doc["client_name"])
                
                # Location analysis
                if doc.get("location"):
                    locations.add(doc["location"])
                
                # Vendor analysis
                if doc.get("vendor_name"):
                    vendors.add(doc["vendor_name"])
                
                # Document type analysis
                if doc.get("doc_type"):
                    doc_type = doc["doc_type"]
                    doc_types[doc_type] = doc_types.get(doc_type, 0) + 1
                
                # Consumption analysis
                consumption = doc.get("consumption_amount")
                unit = doc.get("unit_of_measure", "")
                doc_type = doc.get("doc_type", "Unknown")
                
                if consumption and str(consumption).replace(".", "").replace(",", "").isdigit():
                    key = f"{doc_type}_{unit}"
                    total_consumption_by_type[key] = total_consumption_by_type.get(key, 0) + float(str(consumption).replace(",", ""))
            
            # Portfolio Overview
            analysis += "📊 UTILITIES PORTFOLIO OVERVIEW\n"
            analysis += "-" * 30 + "\n"
            analysis += f"• Total utilities documents: {total_docs}\n"
            analysis += f"• Unique clients: {len(clients)}\n"
            analysis += f"• Unique locations: {len(locations)}\n"
            analysis += f"• Unique vendors: {len(vendors)}\n"
            analysis += f"• Document types: {len(doc_types)}\n"
            
            # Document Type Analysis
            if doc_types:
                analysis += f"\n⚡ DOCUMENT TYPE ANALYSIS\n"
                analysis += "-" * 30 + "\n"
                sorted_types = sorted(doc_types.items(), key=lambda x: x[1], reverse=True)
                for doc_type, count in sorted_types:
                    percentage = (count / total_docs) * 100
                    analysis += f"• {doc_type}: {count} documents ({percentage:.1f}%)\n"
            
            # Consumption Analysis
            if total_consumption_by_type:
                analysis += f"\n📈 CONSUMPTION ANALYSIS\n"
                analysis += "-" * 30 + "\n"
                for key, total in sorted(total_consumption_by_type.items()):
                    doc_type, unit = key.split('_', 1)
                    analysis += f"• {doc_type}: {total:,.1f} {unit}\n"
            
            # Client Analysis
            if len(clients) > 0:
                analysis += f"\n👥 CLIENT ANALYSIS\n"
                analysis += "-" * 30 + "\n"
                client_counts = {}
                for doc in documents:
                    client = doc.get("client_name", "Unknown")
                    client_counts[client] = client_counts.get(client, 0) + 1
                
                sorted_clients = sorted(client_counts.items(), key=lambda x: x[1], reverse=True)
                for client, count in sorted_clients[:10]:
                    analysis += f"• {client}: {count} utilities document(s)\n"
            
            # Vendor Analysis
            if len(vendors) > 0:
                analysis += f"\n🏪 VENDOR ANALYSIS\n"
                analysis += "-" * 30 + "\n"
                vendor_counts = {}
                for doc in documents:
                    vendor = doc.get("vendor_name", "Unknown")
                    vendor_counts[vendor] = vendor_counts.get(vendor, 0) + 1
                
                sorted_vendors = sorted(vendor_counts.items(), key=lambda x: x[1], reverse=True)
                for vendor, count in sorted_vendors[:10]:
                    analysis += f"• {vendor}: {count} document(s)\n"
            
            # Location Analysis
            if len(locations) > 0:
                analysis += f"\n📍 LOCATION ANALYSIS\n"
                analysis += "-" * 30 + "\n"
                location_counts = {}
                for doc in documents:
                    location = doc.get("location", "Unknown")
                    # Extract city/state from location
                    city = location.split(",")[0] if "," in location else location
                    location_counts[city] = location_counts.get(city, 0) + 1
                
                sorted_locations = sorted(location_counts.items(), key=lambda x: x[1], reverse=True)
                for location, count in sorted_locations[:10]:
                    analysis += f"• {location}: {count} utilities document(s)\n"
            
            analysis += f"\n💡 KEY INSIGHTS\n"
            analysis += "-" * 30 + "\n"
            
            if doc_types:
                most_common_type = max(doc_types, key=doc_types.get)
                analysis += f"• ⚡ Most common document type: {most_common_type} ({doc_types[most_common_type]} documents)\n"
            if len(clients) > 0:
                top_client = max(client_counts, key=client_counts.get)
                analysis += f"• 👑 Client with most utilities documents: {top_client} ({client_counts[top_client]} documents)\n"
            if len(vendors) > 0:
                top_vendor = max(vendor_counts, key=vendor_counts.get)
                analysis += f"• 🏪 Most common vendor: {top_vendor} ({vendor_counts[top_vendor]} documents)\n"
            if total_consumption_by_type:
                analysis += f"• 📊 Total document types tracked: {len(total_consumption_by_type)}\n"
            
            return analysis
            
        except Exception as e:
            return f"Error performing utilities analysis: {str(e)}"


def add_utilities_capabilities_to_prompt() -> str:
    """
    Returns additional prompt text for utilities capabilities
    """
    return """
UTILITIES ANALYSIS CAPABILITIES:
- Process natural gas and electricity invoices/bills
- Extract consumption data, vendor information, and billing periods
- Match utilities documents by location or client name
- Analyze consumption patterns and trends across locations
- Compare utilities costs and usage between clients
- Search for specific utilities information using vector search

UTILITIES TOOLS AVAILABLE:
- utilities_ocr_extractor: Extract text from utilities PDF documents
- utilities_vector_store_ingest: Store utilities documents in vector database
- utilities_location_matcher: Find utilities documents by location/client
- utilities_vector_search_fields: Search utilities documents for specific information

For utilities analysis, use the utilities-specific tools to ensure proper filtering and analysis of natural gas and electricity data.
"""