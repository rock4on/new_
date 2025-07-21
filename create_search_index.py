#!/usr/bin/env python3
"""
Azure AI Search Index Creation Utility

Creates an Azure AI Search index for electricity documents with vector search capabilities.
This utility sets up the index schema with all required fields for document vectorization
and search functionality.

Usage:
    python create_search_index.py --search-endpoint "https://yoursearch.search.windows.net" --search-key "your-key"
    
Environment Variables:
    AZURE_SEARCH_ENDPOINT: Azure AI Search endpoint
    AZURE_SEARCH_KEY: Azure AI Search admin key
    AZURE_SEARCH_INDEX_NAME: Index name (default: electricity-documents)
"""

import os
import sys
import argparse
from typing import Optional

try:
    from azure.search.documents.indexes import SearchIndexClient
    from azure.core.credentials import AzureKeyCredential
    from azure.search.documents.indexes.models import (
        SearchIndex, SearchField, SimpleField, SearchableField, VectorSearch,
        HnswAlgorithmConfiguration, HnswParameters, VectorSearchProfile,
        VectorSearchAlgorithmMetric
    )
except ImportError as e:
    print(f"❌ Missing required dependency: {e}")
    print("💡 Install with: pip install azure-search-documents")
    sys.exit(1)


def create_electricity_documents_index(
    search_endpoint: str,
    search_key: str,
    index_name: str = "electricity-documents"
) -> bool:
    """
    Create the electricity documents index in Azure AI Search.
    
    Args:
        search_endpoint: Azure AI Search endpoint
        search_key: Azure AI Search admin key
        index_name: Name of the index to create
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Initialize the search index client
        search_index_client = SearchIndexClient(
            endpoint=search_endpoint,
            credential=AzureKeyCredential(search_key)
        )
        
        print(f"🔍 Creating index: {index_name}")
        
        # Define the index fields
        fields = [
            SimpleField("id", "Edm.String", key=True),
            SearchableField("content", "Edm.String"),
            SearchField(
                name="embedding",
                type="Collection(Edm.Single)",
                searchable=True,
                vector_search_dimensions=1536,
                vector_search_profile_name="ada-vector-profile"
            ),
            SimpleField("filename", "Edm.String", filterable=True, sortable=True),
            SimpleField("page_no", "Edm.Int32", filterable=True, sortable=True, facetable=True),
            SimpleField("doc_type", "Edm.String", filterable=True, sortable=True, facetable=True),
            SimpleField("client_name", "Edm.String", filterable=True, sortable=True, facetable=True),
            SimpleField("language", "Edm.String", filterable=True, facetable=True),
            SimpleField("isTranslated", "Edm.String", filterable=True, facetable=True),
            SearchableField("description", "Edm.String", filterable=True),
            SearchableField("location", "Edm.String", filterable=True),
            SimpleField("lease_start_date", "Edm.String", filterable=True, sortable=True, facetable=True),
            SimpleField("lease_end_date", "Edm.String", filterable=True, sortable=True, facetable=True),
            SimpleField("building_area", "Edm.String", filterable=True, sortable=True, facetable=True),
            SimpleField("area_unit", "Edm.String", filterable=True, facetable=True),
            SimpleField("building_type", "Edm.String", filterable=True, facetable=True),
            SimpleField("processed_at", "Edm.DateTimeOffset", filterable=True, sortable=True),
        ]
        
        # Configure vector search
        vector_search = VectorSearch(
            algorithms=[
                HnswAlgorithmConfiguration(
                    name="ada-hnsw",
                    parameters=HnswParameters(
                        metric=VectorSearchAlgorithmMetric.COSINE,
                        m=4,
                        ef_construction=400,
                        ef_search=500
                    )
                )
            ],
            profiles=[
                VectorSearchProfile(
                    name="ada-vector-profile",
                    algorithm_configuration_name="ada-hnsw"
                )
            ]
        )
        
        # Create the search index
        index = SearchIndex(
            name=index_name,
            fields=fields,
            vector_search=vector_search
        )
        
        # Check if index already exists
        try:
            existing_index = search_index_client.get_index(index_name)
            print(f"⚠️  Index '{index_name}' already exists")
            
            # Ask user if they want to delete and recreate
            response = input("Do you want to delete and recreate the index? (y/N): ").lower().strip()
            if response == 'y' or response == 'yes':
                print(f"🗑️  Deleting existing index: {index_name}")
                search_index_client.delete_index(index_name)
                print(f"✅ Index deleted successfully")
            else:
                print("ℹ️  Keeping existing index")
                return True
                
        except Exception:
            # Index doesn't exist, which is fine
            pass
        
        # Create the new index
        print(f"🏗️  Creating new index: {index_name}")
        result = search_index_client.create_index(index)
        
        print(f"✅ Successfully created index: {result.name}")
        print(f"📊 Index contains {len(result.fields)} fields")
        print(f"🔤 Vector search enabled with 1536 dimensions")
        
        # Display field summary
        print(f"\n📋 Index Fields:")
        for field in result.fields:
            field_type = field.type
            properties = []
            if getattr(field, 'key', False):
                properties.append("key")
            if getattr(field, 'searchable', False):
                properties.append("searchable")
            if getattr(field, 'filterable', False):
                properties.append("filterable")
            if getattr(field, 'sortable', False):
                properties.append("sortable")
            if getattr(field, 'facetable', False):
                properties.append("facetable")
            if getattr(field, 'vector_search_dimensions', None):
                properties.append(f"vector({field.vector_search_dimensions}d)")
            
            props_str = f" ({', '.join(properties)})" if properties else ""
            print(f"   • {field.name}: {field_type}{props_str}")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to create index: {e}")
        return False


def main():
    """Command line interface for index creation"""
    parser = argparse.ArgumentParser(
        description='Create Azure AI Search index for electricity documents'
    )
    parser.add_argument(
        '--search-endpoint', 
        help='Azure AI Search endpoint (or set AZURE_SEARCH_ENDPOINT env var)'
    )
    parser.add_argument(
        '--search-key', 
        help='Azure AI Search admin key (or set AZURE_SEARCH_KEY env var)'
    )
    parser.add_argument(
        '--index-name', 
        default='electricity-documents',
        help='Name of the index to create (default: electricity-documents)'
    )
    
    args = parser.parse_args()
    
    # Get configuration from arguments or environment
    search_endpoint = args.search_endpoint or os.getenv('AZURE_SEARCH_ENDPOINT')
    search_key = args.search_key or os.getenv('AZURE_SEARCH_KEY')
    index_name = args.index_name or os.getenv('AZURE_SEARCH_INDEX_NAME', 'electricity-documents')
    
    # Validate required parameters
    if not search_endpoint:
        print("❌ Azure Search endpoint is required")
        print("💡 Use --search-endpoint or set AZURE_SEARCH_ENDPOINT environment variable")
        sys.exit(1)
    
    if not search_key:
        print("❌ Azure Search admin key is required")
        print("💡 Use --search-key or set AZURE_SEARCH_KEY environment variable")
        sys.exit(1)
    
    # Ensure endpoint is properly formatted
    if not search_endpoint.startswith('https://'):
        search_endpoint = f"https://{search_endpoint}"
    if not search_endpoint.endswith('.search.windows.net'):
        if not search_endpoint.endswith('.search.windows.net/'):
            search_endpoint = f"{search_endpoint}.search.windows.net"
    
    print(f"🔗 Azure Search Endpoint: {search_endpoint}")
    print(f"📝 Index Name: {index_name}")
    print()
    
    # Create the index
    success = create_electricity_documents_index(
        search_endpoint=search_endpoint,
        search_key=search_key,
        index_name=index_name
    )
    
    if success:
        print(f"\n🎉 Index creation completed successfully!")
        print(f"🔍 You can now use this index with the lease document processor")
        print(f"💡 Set environment variable: AZURE_SEARCH_INDEX_NAME={index_name}")
    else:
        print(f"\n❌ Index creation failed")
        sys.exit(1)


if __name__ == "__main__":
    main()