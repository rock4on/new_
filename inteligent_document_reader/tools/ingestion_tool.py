"""
Ingestion Tool for storing documents with embeddings in Azure AI Search
"""

import json
import time
import hashlib
from datetime import datetime
from typing import Any
from langchain.tools import BaseTool
from pydantic import Field


class IngestionTool(BaseTool):
    """Tool for ingesting documents with embeddings into Azure AI Search"""
    
    name: str = "ingestion_tool"
    description: str = "Ingests document with text, metadata, and embedding into Azure AI Search index. Input should be JSON with 'text', 'filename', 'doc_type', 'metadata', and 'embedding'."
    search_client: Any = Field(default=None, exclude=True)
    
    def __init__(self, search_client, **kwargs):
        super().__init__(**kwargs)
        object.__setattr__(self, 'search_client', search_client)
    
    def _run(self, input_data: str) -> str:
        """Ingest document into Azure AI Search"""
        try:
            data = json.loads(input_data)
            text = data.get('text', '')
            filename = data.get('filename', 'unknown')
            doc_type = data.get('doc_type', 'lease')
            metadata = data.get('metadata', {})
            embedding = data.get('embedding', [])
            
            if not text:
                return "Error: No text provided for ingestion"
            
            if not embedding:
                return "Error: No embedding provided for ingestion"
            
            print(f"   📤 Ingesting {doc_type} document: {filename}")
            
            # Create document ID
            doc_id = hashlib.md5(f"{filename}_{metadata.get('page_no', 1)}_{time.time()}".encode()).hexdigest()
            
            # Determine index schema based on document type
            if doc_type == 'lease':
                search_document = {
                    "id": doc_id,
                    "content": text,
                    "embedding": embedding,
                    "filename": filename,
                    "page_no": metadata.get('page_no', 1),
                    "doc_type": "Lease",
                    "client_name": metadata.get('client_name', ''),
                    "language": "en",
                    "isTranslated": "no",
                    "location": metadata.get('location', ''),
                    "lease_start_date": metadata.get('lease_start_date', ''),
                    "lease_end_date": metadata.get('lease_end_date', ''),
                    "building_area": metadata.get('building_area', ''),
                    "area_unit": metadata.get('area_unit', ''),
                    "building_type": metadata.get('building_type', ''),
                    "description": metadata.get('description', ''),
                    "processed_at": datetime.now().isoformat()
                }
            else:  # electricity or natural_gas
                # Determine document type from filename or metadata
                determined_doc_type = "Electricity" if "electricity" in filename.lower() or doc_type == "electricity" else "NaturalGas"
                
                search_document = {
                    "id": doc_id,
                    "content": text,
                    "embedding": embedding,
                    "filename": filename,
                    "page_no": metadata.get('page_no', 1),
                    "doc_type": determined_doc_type,
                    "client_name": metadata.get('client_name', ''),
                    "language": "en",
                    "isTranslated": "no",
                    # Utilities-specific fields
                    "vendor_name": metadata.get('vendor_name', ''),
                    "account_or_invoice_number": metadata.get('account_or_invoice_number', ''),
                    "invoice_date": metadata.get('invoice_date', ''),
                    "location": metadata.get('location', ''),
                    "measurement_period_start": metadata.get('measurement_period_start', ''),
                    "measurement_period_end": metadata.get('measurement_period_end', ''),
                    "consumption_amount": metadata.get('consumption_amount'),
                    "unit_of_measure": metadata.get('unit_of_measure', ''),
                    "processed_at": datetime.now().isoformat()
                }
            
            # Upload to search index
            result = self.search_client.upload_documents([search_document])
            
            if result and len(result) > 0 and result[0].succeeded:
                extracted_fields = []
                if doc_type == 'lease':
                    extracted_fields = [f for f in ['location', 'lease_start_date', 'lease_end_date', 'building_area', 'building_type'] 
                                      if metadata.get(f)]
                else:
                    extracted_fields = [f for f in ['vendor_name', 'location', 'invoice_date', 'consumption_amount', 'unit_of_measure'] 
                                      if metadata.get(f)]
                
                success_msg = f"Successfully ingested {doc_type} document {filename} with ID {doc_id}"
                if extracted_fields:
                    success_msg += f" and extracted fields: {', '.join(extracted_fields)}"
                
                print(f"   ✅ {success_msg}")
                return success_msg
            else:
                error_msg = f"Failed to ingest document {filename} into search index"
                print(f"   ❌ {error_msg}")
                return error_msg
                
        except json.JSONDecodeError:
            return "Error: Invalid JSON input format"
        except Exception as e:
            error_msg = f"Error ingesting document: {str(e)}"
            print(f"   ❌ {error_msg}")
            return error_msg