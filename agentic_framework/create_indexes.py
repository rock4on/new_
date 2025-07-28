#!/usr/bin/env python3
"""
Index Creation Script for Lease and Utilities Document Processing
Creates Azure AI Search indexes if they don't exist
"""

import os
import sys
from typing import Optional
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchIndex,
    SearchField,
    SearchFieldDataType,
    SimpleField,
    SearchableField,
    VectorSearch,
    VectorSearchProfile,
    HnswAlgorithmConfiguration,
)
from azure.core.credentials import AzureKeyCredential


def create_lease_index(index_client: SearchIndexClient, index_name: str) -> bool:
    """Create the lease documents index"""
    try:
        print(f"📄 Creating lease index: {index_name}")
        
        # Define the fields for lease documents
        fields = [
            SimpleField(name="id", type=SearchFieldDataType.String, key=True),
            SearchableField(name="content", type=SearchFieldDataType.String),
            SearchField(
                name="embedding", 
                type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                searchable=True, 
                vector_search_dimensions=1536,
                vector_search_profile_name="myHnswProfile"
            ),
            SearchableField(name="filename", type=SearchFieldDataType.String),
            SimpleField(name="page_no", type=SearchFieldDataType.Int32),
            SimpleField(name="doc_type", type=SearchFieldDataType.String, filterable=True),
            SearchableField(name="client_name", type=SearchFieldDataType.String, filterable=True),
            SimpleField(name="language", type=SearchFieldDataType.String),
            SimpleField(name="isTranslated", type=SearchFieldDataType.String),
            
            # Lease-specific fields
            SearchableField(name="location", type=SearchFieldDataType.String),
            SimpleField(name="lease_start_date", type=SearchFieldDataType.String),
            SimpleField(name="lease_end_date", type=SearchFieldDataType.String),
            SimpleField(name="building_area", type=SearchFieldDataType.String),
            SimpleField(name="area_unit", type=SearchFieldDataType.String),
            SimpleField(name="building_type", type=SearchFieldDataType.String),
            SimpleField(name="processed_at", type=SearchFieldDataType.String),
        ]
        
        # Configure vector search
        vector_search = VectorSearch(
            profiles=[
                VectorSearchProfile(
                    name="myHnswProfile",
                    algorithm_configuration_name="myHnsw",
                )
            ],
            algorithms=[
                HnswAlgorithmConfiguration(name="myHnsw")
            ],
        )
        
        # Create the search index
        index = SearchIndex(
            name=index_name,
            fields=fields,
            vector_search=vector_search
        )
        
        result = index_client.create_index(index)
        print(f"✅ Successfully created lease index: {index_name}")
        return True
        
    except Exception as e:
        if "already exists" in str(e).lower():
            print(f"ℹ️  Lease index {index_name} already exists")
            return True
        else:
            print(f"❌ Failed to create lease index {index_name}: {e}")
            return False


def create_utilities_index(index_client: SearchIndexClient, index_name: str) -> bool:
    """Create the utilities documents index"""
    try:
        print(f"⚡ Creating utilities index: {index_name}")
        
        # Define the fields for utilities documents
        fields = [
            SimpleField(name="id", type=SearchFieldDataType.String, key=True),
            SearchableField(name="content", type=SearchFieldDataType.String),
            SearchField(
                name="embedding", 
                type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                searchable=True, 
                vector_search_dimensions=1536,
                vector_search_profile_name="myHnswProfile"
            ),
            SearchableField(name="filename", type=SearchFieldDataType.String),
            SimpleField(name="page_no", type=SearchFieldDataType.Int32),
            SimpleField(name="doc_type", type=SearchFieldDataType.String, filterable=True),
            SearchableField(name="client_name", type=SearchFieldDataType.String, filterable=True),
            SimpleField(name="language", type=SearchFieldDataType.String),
            SimpleField(name="isTranslated", type=SearchFieldDataType.String),
            
            # Utilities-specific fields
            SearchableField(name="location", type=SearchFieldDataType.String),
            SearchableField(name="vendor_name", type=SearchFieldDataType.String),
            SimpleField(name="account_or_invoice_number", type=SearchFieldDataType.String),
            SimpleField(name="invoice_date", type=SearchFieldDataType.String),
            SimpleField(name="measurement_period_start", type=SearchFieldDataType.String),
            SimpleField(name="measurement_period_end", type=SearchFieldDataType.String),
            SimpleField(name="consumption_amount", type=SearchFieldDataType.String),
            SimpleField(name="unit_of_measure", type=SearchFieldDataType.String),
            SimpleField(name="processed_at", type=SearchFieldDataType.String),
        ]
        
        # Configure vector search
        vector_search = VectorSearch(
            profiles=[
                VectorSearchProfile(
                    name="myHnswProfile",
                    algorithm_configuration_name="myHnsw",
                )
            ],
            algorithms=[
                HnswAlgorithmConfiguration(name="myHnsw")
            ],
        )
        
        # Create the search index
        index = SearchIndex(
            name=index_name,
            fields=fields,
            vector_search=vector_search
        )
        
        result = index_client.create_index(index)
        print(f"✅ Successfully created utilities index: {index_name}")
        return True
        
    except Exception as e:
        if "already exists" in str(e).lower():
            print(f"ℹ️  Utilities index {index_name} already exists")
            return True
        else:
            print(f"❌ Failed to create utilities index {index_name}: {e}")
            return False


def main():
    """Main function to create indexes"""
    print("🏗️  Azure AI Search Index Creator")
    print("=" * 50)
    
    # Get configuration
    try:
        from config import get_config
        config = get_config()
        
        azure_search_endpoint = config.AZURE_SEARCH_ENDPOINT
        azure_search_key = config.AZURE_SEARCH_KEY
        base_index_name = config.AZURE_SEARCH_INDEX_NAME
        
    except ImportError:
        print("❌ Could not import config. Make sure config.py exists and is properly configured.")
        print("\nAlternatively, you can set environment variables:")
        print("  AZURE_SEARCH_ENDPOINT")
        print("  AZURE_SEARCH_KEY")
        print("  AZURE_SEARCH_INDEX_NAME")
        
        # Try environment variables as fallback
        azure_search_endpoint = os.getenv('AZURE_SEARCH_ENDPOINT')
        azure_search_key = os.getenv('AZURE_SEARCH_KEY')
        base_index_name = os.getenv('AZURE_SEARCH_INDEX_NAME', 'lease-documents')
        
        if not azure_search_endpoint or not azure_search_key:
            print("❌ Missing required Azure Search configuration")
            sys.exit(1)
    
    except Exception as e:
        print(f"❌ Error loading configuration: {e}")
        sys.exit(1)
    
    print(f"📡 Azure Search Endpoint: {azure_search_endpoint}")
    print(f"🔑 Azure Search Key: {azure_search_key[:10]}...{azure_search_key[-4:]}")
    print(f"📋 Base Index Name: {base_index_name}")
    print()
    
    # Generate index names
    lease_index_name = base_index_name
    utilities_index_name = base_index_name.replace('-documents', '-utilities') if '-documents' in base_index_name else f"{base_index_name}-utilities"
    
    print(f"📄 Lease Index: {lease_index_name}")
    print(f"⚡ Utilities Index: {utilities_index_name}")
    print()
    
    # Initialize the search index client
    try:
        index_client = SearchIndexClient(
            endpoint=azure_search_endpoint,
            credential=AzureKeyCredential(azure_search_key)
        )
        print("✅ Connected to Azure Search service")
    except Exception as e:
        print(f"❌ Failed to connect to Azure Search: {e}")
        sys.exit(1)
    
    # Create indexes
    success_count = 0
    
    # Create lease index
    if create_lease_index(index_client, lease_index_name):
        success_count += 1
    
    # Create utilities index
    if create_utilities_index(index_client, utilities_index_name):
        success_count += 1
    
    print()
    print("=" * 50)
    if success_count == 2:
        print("🎉 All indexes created/verified successfully!")
        print()
        print("📋 Summary:")
        print(f"   📄 Lease Index: {lease_index_name}")
        print(f"   ⚡ Utilities Index: {utilities_index_name}")
        print()
        print("✅ Your system is ready for document ingestion!")
    else:
        print(f"⚠️  Only {success_count}/2 indexes were created successfully")
        print("❌ Some indexes may need manual attention")
        sys.exit(1)


if __name__ == "__main__":
    main()