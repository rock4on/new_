#!/usr/bin/env python3
"""
Utilities Tools for Natural Gas and Electricity Document Processing
Integrated with LangChain ReAct Agent System
"""

import os
import json
import time
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime

from langchain.tools import BaseTool
from pydantic import BaseModel, Field

from utilities_models import NaturalGas_Electricity_Information, UTILITIES_FIELDS


class UtilitiesOCRTool(BaseTool):
    """Tool for extracting text from utilities PDF documents using Azure Form Recognizer"""
    
    name: str = "utilities_ocr_extractor"
    description: str = "Extracts text from utilities PDF documents (natural gas/electricity invoices) using Azure Form Recognizer OCR. Input should be the file path to a PDF document."
    azure_client: Any = Field(default=None, exclude=True)
    
    def __init__(self, azure_endpoint: str, azure_key: str, **kwargs):
        super().__init__(**kwargs)
        try:
            from azure.ai.formrecognizer import DocumentAnalysisClient
            from azure.core.credentials import AzureKeyCredential
            from azure.core.pipeline.transport import RequestsTransport
            
            print(f"   🔍 Testing Azure Form Recognizer connection for utilities...")
            
            # Disable SSL verification for Azure OCR transport
            transport = RequestsTransport(connection_verify=False)
            azure_client = DocumentAnalysisClient(
                endpoint=azure_endpoint,
                credential=AzureKeyCredential(azure_key),
                transport=transport
            )
            
            object.__setattr__(self, 'azure_client', azure_client)
            print(f"   ✅ Utilities Azure Form Recognizer tool initialized")
            
        except Exception as e:
            print(f"   ❌ Utilities Azure Form Recognizer initialization failed: {e}")
            raise ConnectionError(f"Utilities Azure Form Recognizer initialization failed: {e}")
    
    def _run(self, file_path: str) -> str:
        """Extract text from utilities PDF file"""
        try:
            pdf_path = Path(file_path)
            if not pdf_path.exists():
                return f"Error: File not found at {file_path}"
            
            with open(pdf_path, "rb") as f:
                poller = self.azure_client.begin_analyze_document("prebuilt-document", document=f)
                result = poller.result()
            
            # Extract all text
            full_text = ""
            for page in result.pages:
                for line in page.lines:
                    full_text += line.content + "\n"
            
            return f"Successfully extracted {len(full_text)} characters of text from utilities document {pdf_path.name}"
            
        except Exception as e:
            return f"Error extracting text from utilities document: {str(e)}"


class UtilitiesIngestionTool(BaseTool):
    """Tool for ingesting utilities text into vector store with embeddings"""
    
    name: str = "utilities_vector_store_ingest"
    description: str = "Ingests utilities document text content into vector store with embeddings. Input should be JSON with 'text', 'filename', 'metadata' fields including utilities-specific information."
    openai_client: Any = Field(default=None, exclude=True)
    search_client: Any = Field(default=None, exclude=True)
    embedding_model: str = Field(default="text-embedding-ada-002", exclude=True)
    
    def __init__(self, openai_client, search_client, embedding_model: str = "text-embedding-ada-002", **kwargs):
        super().__init__(**kwargs)
        object.__setattr__(self, 'openai_client', openai_client)
        object.__setattr__(self, 'search_client', search_client)
        object.__setattr__(self, 'embedding_model', embedding_model)
    
    def _generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for text"""
        try:
            print(f"   🔤 Generating embedding for utilities text ({len(text)} chars)...")
            response = self.openai_client.embeddings.create(
                model=self.embedding_model,
                input=text[:8000]  # Limit text length to avoid token limits
            )
            embedding = response.data[0].embedding
            print(f"   ✅ Utilities embedding generated: {len(embedding)} dimensions")
            return embedding
        except Exception as e:
            print(f"   ❌ Failed to generate utilities embedding: {e}")
            return []
    
    def _run(self, input_data: str) -> str:
        """Ingest utilities text into vector store"""
        try:
            # Parse input JSON
            data = json.loads(input_data)
            text = data.get('text', '')
            filename = data.get('filename', 'unknown')
            metadata = data.get('metadata', {})
            
            if not text:
                return "Error: No text provided for utilities ingestion"
            
            # Generate embedding
            embedding = self._generate_embedding(text)
            if not embedding:
                return "Error: Failed to generate utilities embedding"
            
            # Create document for indexing - matches full.py schema exactly
            doc_id = hashlib.md5(f"{metadata.get('client_name', '')}_{filename}_{metadata.get('page_no', 1)}".encode()).hexdigest()
            
            # Determine document type from filename - matches full.py logic exactly
            doc_type = "Electricity" if "electricity" in metadata.get('source_file', filename).lower() else "NaturalGas"
            
            search_document = {
                "id": doc_id,
                "content": text,
                "embedding": embedding,
                "filename": filename,
                "page_no": metadata.get('page_no', 1),
                "doc_type": doc_type,
                "client_name": metadata.get('client_name', ''),
                "language": "en",
                "isTranslated": "no",
                # Add natural gas/electricity information as metadata - exact same fields as full.py
                "vendor_name": metadata.get('vendor_name'),
                "account_or_invoice_number": metadata.get('account_or_invoice_number'),
                "invoice_date": metadata.get('invoice_date'),
                "location": metadata.get('location'),
                "measurement_period_start": metadata.get('measurement_period_start'),
                "measurement_period_end": metadata.get('measurement_period_end'),
                "consumption_amount": metadata.get('consumption_amount'),
                "unit_of_measure": metadata.get('unit_of_measure'),
                "processed_at": datetime.now().isoformat()
            }
            
            # Upload to search index
            result = self.search_client.upload_documents([search_document])
            
            if result and len(result) > 0 and result[0].succeeded:
                return f"Successfully ingested utilities document {filename} into vector store with ID {doc_id}"
            else:
                return f"Failed to ingest utilities document {filename} into vector store"
                
        except json.JSONDecodeError:
            return "Error: Invalid JSON input format"
        except Exception as e:
            return f"Error ingesting utilities document into vector store: {str(e)}"


class UtilitiesMatchingTool(BaseTool):
    """Tool for matching utilities documents by location or client name"""
    
    name: str = "utilities_location_matcher"
    description: str = "Finds utilities documents (natural gas/electricity) matching a specific location OR client name. Input should be the location or client name to search for."
    search_client: Any = Field(default=None, exclude=True)
    
    def __init__(self, search_client, **kwargs):
        super().__init__(**kwargs)
        object.__setattr__(self, 'search_client', search_client)
    
    def _run(self, location: str) -> str:
        """Find utilities documents matching the specified location or client"""
        try:
            # Search for utilities documents with matching location or client name
            search_results = self.search_client.search(
                search_text=f"{location}",
                filter="doc_type eq 'NaturalGas' or doc_type eq 'Electricity'",
                select=["id", "filename", "location", "client_name", "vendor_name", "invoice_date", 
                       "measurement_period_start", "measurement_period_end", "consumption_amount", 
                       "unit_of_measure", "doc_type", "processed_at"],
                top=100
            )
            
            # Also try a filter-based search for exact client matches
            try:
                filtered_results = self.search_client.search(
                    search_text="*",
                    filter=f"(doc_type eq 'NaturalGas' or doc_type eq 'Electricity') and client_name eq '{location}'",
                    select=["id", "filename", "location", "client_name", "vendor_name", "invoice_date", 
                           "measurement_period_start", "measurement_period_end", "consumption_amount", 
                           "unit_of_measure", "doc_type", "processed_at"],
                    top=100
                )
                # Combine results
                all_results = list(search_results) + list(filtered_results)
            except:
                # If filter fails, just use text search results
                all_results = list(search_results)
            
            matches = []
            for result in all_results:
                matches.append({
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
                    "score": result.get("@search.score", 0)
                })
            
            if matches:
                # Group by filename to avoid counting pages as separate documents
                documents = {}
                for match in matches:
                    filename = match.get("filename", "Unknown")
                    if filename not in documents:
                        documents[filename] = match
                
                unique_docs = list(documents.values())
                search_type = "client" if any(location.lower() in doc.get("client_name", "").lower() for doc in unique_docs) else "location/keyword"
                analysis = f"Found {len(unique_docs)} utilities documents matching {search_type} '{location}':\n\n"
                
                # Group by client and document type
                clients = {}
                doc_types = {}
                total_consumption = {}
                
                for doc in unique_docs:
                    client = doc.get("client_name", "Unknown")
                    if client not in clients:
                        clients[client] = []
                    clients[client].append(doc)
                    
                    # Track document types  
                    doc_type = doc.get("doc_type", "Unknown")
                    doc_types[doc_type] = doc_types.get(doc_type, 0) + 1
                    
                    # Track consumption by document type
                    consumption = doc.get("consumption_amount")
                    unit = doc.get("unit_of_measure", "")
                    if consumption and str(consumption).replace(".", "").replace(",", "").isdigit():
                        key = f"{doc_type}_{unit}"
                        total_consumption[key] = total_consumption.get(key, 0) + float(str(consumption).replace(",", ""))
                
                # Summary analysis
                analysis += "📊 UTILITIES SUMMARY:\n"
                analysis += f"• Total utilities documents: {len(unique_docs)}\n"
                analysis += f"• Unique clients: {len(clients)}\n"
                if doc_types:
                    analysis += f"• Document types: {', '.join([f'{k} ({v})' for k, v in doc_types.items()])}\n"
                if total_consumption:
                    analysis += f"• Total consumption by type:\n"
                    for key, value in total_consumption.items():
                        util_type, unit = key.split('_', 1)
                        analysis += f"  - {util_type}: {value:,.1f} {unit}\n"
                
                analysis += f"\n📋 DETAILED RESULTS:\n"
                
                for client, client_docs in clients.items():
                    analysis += f"\n🏢 Client: {client} ({len(client_docs)} utilities documents)\n"
                    for doc in client_docs:
                        analysis += f"  • {doc.get('filename', 'Unknown file')} ({doc.get('doc_type', 'Unknown type')})\n"
                        analysis += f"    Location: {doc.get('location', 'Not specified')}\n"
                        analysis += f"    Vendor: {doc.get('vendor_name', 'Not specified')}\n"
                        if doc.get('invoice_date'):
                            analysis += f"    Invoice Date: {doc.get('invoice_date')}\n"
                        if doc.get('measurement_period_start'):
                            analysis += f"    Period: {doc.get('measurement_period_start')} to {doc.get('measurement_period_end', 'Unknown')}\n"
                        if doc.get('consumption_amount'):
                            analysis += f"    Consumption: {doc.get('consumption_amount')} {doc.get('unit_of_measure', '')}\n"
                        analysis += "\n"
                
                return analysis
            else:
                return f"No utilities documents found for location '{location}'. Try searching with different keywords or check if utilities documents have been uploaded to the system."
                
        except Exception as e:
            return f"Error searching for utilities documents: {str(e)}"


class UtilitiesVectorSearchTool(BaseTool):
    """Tool for performing vector search on utilities documents to extract specific fields"""
    
    name: str = "utilities_vector_search_fields"
    description: str = "Performs vector search on utilities documents to find relevant documents and extract specific fields. Input should be JSON with 'query' and 'fields' (list of field names to extract)."
    openai_client: Any = Field(default=None, exclude=True)
    search_client: Any = Field(default=None, exclude=True)
    embedding_model: str = Field(default="text-embedding-ada-002", exclude=True)
    
    def __init__(self, openai_client, search_client, embedding_model: str = "text-embedding-ada-002", **kwargs):
        super().__init__(**kwargs)
        object.__setattr__(self, 'openai_client', openai_client)
        object.__setattr__(self, 'search_client', search_client)
        object.__setattr__(self, 'embedding_model', embedding_model)
    
    def _generate_query_embedding(self, query: str) -> List[float]:
        """Generate embedding for search query"""
        try:
            print(f"   🔤 Generating utilities query embedding for: '{query[:100]}...'")
            response = self.openai_client.embeddings.create(
                model=self.embedding_model, 
                input=query
            )
            embedding = response.data[0].embedding
            print(f"   ✅ Utilities query embedding generated: {len(embedding)} dimensions")
            return embedding
        except Exception as e:
            print(f"   ❌ Failed to generate utilities query embedding: {e}")
            return []
    
    def _run(self, input_data: str) -> str:
        """Perform vector search on utilities documents and extract fields"""
        try:
            from azure.search.documents.models import VectorizedQuery
            
            # Parse input JSON
            data = json.loads(input_data)
            query = data.get('query', '')
            fields = data.get('fields', ['location', 'vendor_name', 'invoice_date', 'consumption_amount', 'utility_type'])
            
            if not query:
                return "Error: No search query provided"
            
            # Generate query embedding
            query_embedding = self._generate_query_embedding(query)
            if not query_embedding:
                return "Error: Failed to generate utilities query embedding"
            
            # Perform vector search on utilities documents only
            vector_query = VectorizedQuery(vector=query_embedding, k_nearest_neighbors=10, fields="embedding")
            
            search_results = self.search_client.search(
                search_text=query,
                vector_queries=[vector_query],
                filter="doc_type eq 'NaturalGas' or doc_type eq 'Electricity'",
                select=["id", "filename", "content", "location", "client_name", "vendor_name", 
                       "invoice_date", "measurement_period_start", "measurement_period_end", 
                       "consumption_amount", "unit_of_measure", "doc_type"],
                top=15
            )
            
            results = []
            for result in search_results:
                doc_data = {
                    "filename": result.get("filename"),
                    "client_name": result.get("client_name"),
                    "doc_type": result.get("doc_type"),
                    "score": result.get("@search.score", 0),
                    "content_preview": (result.get("content", "")[:200] + "...") if result.get("content") else ""
                }
                
                # Extract all utilities-specific fields
                for field in ["location", "vendor_name", "invoice_date", "measurement_period_start", 
                             "measurement_period_end", "consumption_amount", "unit_of_measure"]:
                    doc_data[field] = result.get(field, "")
                
                results.append(doc_data)
            
            if results:
                # Group by filename to avoid counting pages as separate documents
                documents = {}
                for result in results:
                    filename = result.get("filename", "Unknown")
                    if filename not in documents:
                        documents[filename] = result
                
                unique_docs = list(documents.values())
                
                # Provide comprehensive analysis
                analysis = f"🔍 UTILITIES SEARCH RESULTS for '{query}'\n"
                analysis += f"Found {len(unique_docs)} relevant utilities documents (deduplicated from {len(results)} pages):\n\n"
                
                # Analyze the unique documents
                clients = set()
                locations = set()
                doc_types = {}
                vendors = set()
                total_consumption = {}
                
                for doc in unique_docs:
                    if doc.get("client_name"):
                        clients.add(doc["client_name"])
                    if doc.get("location"):
                        locations.add(doc["location"])
                    if doc.get("vendor_name"):
                        vendors.add(doc["vendor_name"])
                    if doc.get("doc_type"):
                        doc_type = doc["doc_type"]
                        doc_types[doc_type] = doc_types.get(doc_type, 0) + 1
                    
                    # Parse consumption
                    consumption = doc.get("consumption_amount")
                    unit = doc.get("unit_of_measure", "")
                    doc_type = doc.get("doc_type", "Unknown")
                    if consumption and str(consumption).replace(".", "").replace(",", "").isdigit():
                        key = f"{doc_type}_{unit}"
                        total_consumption[key] = total_consumption.get(key, 0) + float(str(consumption).replace(",", ""))
                
                # Summary
                analysis += "📊 UTILITIES SUMMARY:\n"
                analysis += f"• {len(clients)} unique clients\n"
                analysis += f"• {len(locations)} unique locations\n"
                analysis += f"• {len(vendors)} unique vendors\n"
                if doc_types:
                    analysis += f"• Document types: {', '.join([f'{k} ({v})' for k, v in doc_types.items()])}\n"
                if total_consumption:
                    analysis += f"• Total consumption by type:\n"
                    for key, value in total_consumption.items():
                        doc_type, unit = key.split('_', 1)
                        analysis += f"  - {doc_type}: {value:,.1f} {unit}\n"
                
                # Top results with detailed info
                analysis += f"\n📋 TOP MATCHING UTILITIES DOCUMENTS:\n"
                for i, doc in enumerate(unique_docs[:8], 1):
                    analysis += f"\n{i}. {doc.get('filename', 'Unknown file')} (Score: {doc['score']:.2f})\n"
                    analysis += f"   Type: {doc.get('doc_type', 'Unknown')}\n"
                    analysis += f"   Client: {doc.get('client_name', 'Unknown')}\n"
                    analysis += f"   Location: {doc.get('location', 'Not specified')}\n"
                    analysis += f"   Vendor: {doc.get('vendor_name', 'Not specified')}\n"
                    if doc.get('invoice_date'):
                        analysis += f"   Invoice Date: {doc['invoice_date']}\n"
                    if doc.get('measurement_period_start'):
                        analysis += f"   Period: {doc['measurement_period_start']} to {doc.get('measurement_period_end', 'Unknown')}\n"
                    if doc.get('consumption_amount'):
                        analysis += f"   Consumption: {doc['consumption_amount']} {doc.get('unit_of_measure', '')}\n"
                
                # Extract specific requested fields if provided
                if fields and any(f != "" for f in fields):
                    analysis += f"\n🎯 REQUESTED FIELDS SUMMARY:\n"
                    for field in fields:
                        field_values = [doc.get(field) for doc in unique_docs if doc.get(field)]
                        if field_values:
                            unique_values = list(set(field_values))
                            analysis += f"• {field}: {len(unique_values)} unique values\n"
                            if len(unique_values) <= 5:
                                analysis += f"  Values: {', '.join(str(v) for v in unique_values)}\n"
                            else:
                                analysis += f"  Top values: {', '.join(str(v) for v in unique_values[:5])}...\n"
                
                return analysis
            else:
                return f"No utilities documents found matching '{query}'. Try different keywords or check if relevant utilities documents have been uploaded."
                
        except json.JSONDecodeError:
            return "Error: Invalid JSON input format. Please provide JSON with 'query' and optional 'fields' array."
        except Exception as e:
            return f"Error performing utilities vector search: {str(e)}"