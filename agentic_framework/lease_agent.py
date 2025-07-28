#!/usr/bin/env python3
"""
Agentic Lease Document Processing Framework
Using LangChain ReAct pattern with custom tools for document processing

This agent can:
- Extract text using Azure OCR
- Ingest text into vector store
- Match relevant documents based on location
- Perform vector search to extract desired fields
"""

import os
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
import hashlib
import uuid

# LangChain imports
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import BaseTool
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.schema import BaseMessage, HumanMessage, AIMessage

# Azure and OpenAI imports
from azure.ai.formrecognizer import DocumentAnalysisClient
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.models import VectorizedQuery
from azure.core.pipeline.transport import RequestsTransport
import openai
import ssl
import urllib3
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

# Import utilities models and agent factory
from utilities_models import NaturalGas_Electricity_Information, UTILITIES_FIELDS
from utilities_agent import create_utilities_tools, UtilitiesAnalysisTool
from batch_ingestion_tools import create_batch_ingestion_tools


class LeaseInformation(BaseModel):
    """Pydantic model for structured lease information"""
    description: Optional[str] = None
    location: Optional[str] = None
    lease_start_date: Optional[str] = None
    lease_end_date: Optional[str] = None
    building_area: Optional[str] = None
    area_unit: Optional[str] = None
    building_type: Optional[str] = None




class AzureOCRTool(BaseTool):
    """Tool for extracting text from PDF documents using Azure Form Recognizer"""
    
    name: str = "azure_ocr_extractor"
    description: str = "Extracts text from PDF documents using Azure Form Recognizer OCR. Input should be the file path to a PDF document."
    azure_client: Any = Field(default=None, exclude=True)
    
    def __init__(self, azure_endpoint: str, azure_key: str, **kwargs):
        super().__init__(**kwargs)
        try:
            print(f"   🔍 Testing Azure Form Recognizer connection...")
            
            azure_client = DocumentAnalysisClient(
                endpoint=azure_endpoint,
                credential=AzureKeyCredential(azure_key)
            )
            
            # Test the connection by checking the service
            # Note: We can't easily test without a document, so we just initialize
            object.__setattr__(self, 'azure_client', azure_client)
            print(f"   ✅ Azure Form Recognizer tool initialized")
            
        except Exception as e:
            print(f"   ❌ Azure Form Recognizer initialization failed: {e}")
            print(f"   Endpoint: {azure_endpoint}")
            print(f"   Key: {azure_key[:10]}...{azure_key[-4:] if len(azure_key) > 10 else 'invalid'}")
            raise ConnectionError(f"Azure Form Recognizer initialization failed: {e}")
    
    def _run(self, file_path: str) -> str:
        """Extract text from PDF file"""
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
            
            return f"Successfully extracted {len(full_text)} characters of text from {pdf_path.name}"
            
        except Exception as e:
            return f"Error extracting text: {str(e)}"


class VectorStoreIngestionTool(BaseTool):
    """Tool for ingesting text into vector store with embeddings"""
    
    name: str = "vector_store_ingest"
    description: str = "Ingests text content into vector store with embeddings. Input should be JSON with 'text', 'filename', 'metadata' fields."
    openai_client: Any = Field(default=None, exclude=True)
    search_client: Any = Field(default=None, exclude=True)
    embedding_model: str = Field(default="text-embedding-ada-002", exclude=True)
    
    def __init__(self, openai_client, search_client: SearchClient, embedding_model: str = "text-embedding-ada-002", **kwargs):
        super().__init__(**kwargs)
        object.__setattr__(self, 'openai_client', openai_client)
        object.__setattr__(self, 'search_client', search_client)
        object.__setattr__(self, 'embedding_model', embedding_model)
    
    def _generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for text"""
        try:
            print(f"   🔤 Generating embedding for text ({len(text)} chars)...")
            response = self.openai_client.embeddings.create(
                model=self.embedding_model,
                input=text[:8000]  # Limit text length to avoid token limits
            )
            embedding = response.data[0].embedding
            print(f"   ✅ Embedding generated: {len(embedding)} dimensions")
            return embedding
        except Exception as e:
            print(f"   ❌ Failed to generate embedding: {e}")
            print(f"   Text length: {len(text)} chars")
            print(f"   Text preview: {text[:100]}...")
            return []
    
    def _run(self, input_data: str) -> str:
        """Ingest text into vector store"""
        try:
            # Parse input JSON
            data = json.loads(input_data)
            text = data.get('text', '')
            filename = data.get('filename', 'unknown')
            metadata = data.get('metadata', {})
            
            if not text:
                return "Error: No text provided for ingestion"
            
            # Generate embedding
            embedding = self._generate_embedding(text)
            if not embedding:
                return "Error: Failed to generate embedding"
            
            # Create document for indexing
            doc_id = hashlib.md5(f"{filename}_{metadata.get('page_no', 1)}_{time.time()}".encode()).hexdigest()
            
            search_document = {
                "id": doc_id,
                "content": text,
                "embedding": embedding,
                "filename": filename,
                "page_no": metadata.get('page_no', 1),
                "doc_type": metadata.get('doc_type', 'Lease'),
                "client_name": metadata.get('client_name', ''),
                "language": "en",
                "isTranslated": "no",
                "location": metadata.get('location', ''),
                "lease_start_date": metadata.get('lease_start_date', ''),
                "lease_end_date": metadata.get('lease_end_date', ''),
                "building_area": metadata.get('building_area', ''),
                "area_unit": metadata.get('area_unit', ''),
                "building_type": metadata.get('building_type', ''),
                "processed_at": datetime.now().isoformat()
            }
            
            # Upload to search index
            result = self.search_client.upload_documents([search_document])
            
            if result and len(result) > 0 and result[0].succeeded:
                return f"Successfully ingested document {filename} into vector store with ID {doc_id}"
            else:
                return f"Failed to ingest document {filename} into vector store"
                
        except json.JSONDecodeError:
            return "Error: Invalid JSON input format"
        except Exception as e:
            return f"Error ingesting into vector store: {str(e)}"


class LocationMatchingTool(BaseTool):
    """Tool for matching documents by location or client name"""
    
    name: str = "location_matcher"
    description: str = "Finds lease documents matching a specific location OR client name. Input should be the location or client name to search for (e.g., 'Chicago', 'client2', 'ABC Corporation')."
    search_client: Any = Field(default=None, exclude=True)
    
    def __init__(self, search_client: SearchClient, **kwargs):
        super().__init__(**kwargs)
        object.__setattr__(self, 'search_client', search_client)
    
    def _run(self, location: str) -> str:
        """Find documents matching the specified location or client"""
        try:
            # Search for documents with matching location or client name
            # Use multiple search approaches to catch client names
            search_results = self.search_client.search(
                search_text=f"{location}",
                select=["id", "filename", "location", "client_name", "lease_start_date", 
                       "lease_end_date", "building_area", "area_unit", "building_type", "processed_at"],
                top=100  # Get more results to handle pagination
            )
            
            # Also try a filter-based search for exact client matches
            try:
                filtered_results = self.search_client.search(
                    search_text="*",
                    filter=f"client_name eq '{location}'",
                    select=["id", "filename", "location", "client_name", "lease_start_date", 
                           "lease_end_date", "building_area", "area_unit", "building_type", "processed_at"],
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
                    "lease_start_date": result.get("lease_start_date"),
                    "lease_end_date": result.get("lease_end_date"),
                    "building_area": result.get("building_area"),
                    "area_unit": result.get("area_unit"),
                    "building_type": result.get("building_type"),
                    "score": result.get("@search.score", 0)
                })
            
            if matches:
                # Group by filename to avoid counting pages as separate documents
                documents = {}
                for match in matches:
                    filename = match.get("filename", "Unknown")
                    if filename not in documents:
                        documents[filename] = match  # Take first page data for the document
                
                # Analyze the unique documents (not pages)
                unique_docs = list(documents.values())
                search_type = "client" if any(location.lower() in doc.get("client_name", "").lower() for doc in unique_docs) else "location/keyword"
                analysis = f"Found {len(unique_docs)} lease documents matching {search_type} '{location}':\n\n"
                
                # Group by client
                clients = {}
                total_area = 0
                building_types = {}
                
                for doc in unique_docs:
                    client = doc.get("client_name", "Unknown")
                    if client not in clients:
                        clients[client] = []
                    clients[client].append(doc)
                    
                    # Track building area (only once per document)
                    area = doc.get("building_area")
                    if area and str(area).replace(".", "").replace(",", "").isdigit():
                        total_area += float(str(area).replace(",", ""))
                    
                    # Track building types (only once per document)
                    btype = doc.get("building_type")
                    if btype:
                        building_types[btype] = building_types.get(btype, 0) + 1
                
                # Detailed analysis
                analysis += "📊 SUMMARY:\n"
                analysis += f"• Total lease documents: {len(unique_docs)} (deduplicated from {len(matches)} pages)\n"
                analysis += f"• Unique clients: {len(clients)}\n"
                if total_area > 0:
                    analysis += f"• Total building area: {total_area:,.0f} sq ft\n"
                if building_types:
                    analysis += f"• Building types: {', '.join([f'{k} ({v})' for k, v in building_types.items()])}\n"
                
                analysis += f"\n📋 DETAILED RESULTS:\n"
                
                for client, client_docs in clients.items():
                    analysis += f"\n🏢 Client: {client} ({len(client_docs)} lease documents)\n"
                    for doc in client_docs:
                        analysis += f"  • {doc.get('filename', 'Unknown file')}\n"
                        analysis += f"    Location: {doc.get('location', 'Not specified')}\n"
                        if doc.get('lease_start_date'):
                            analysis += f"    Lease: {doc.get('lease_start_date')} to {doc.get('lease_end_date', 'Unknown')}\n"
                        if doc.get('building_area'):
                            analysis += f"    Area: {doc.get('building_area')} {doc.get('area_unit', '')}\n"
                        if doc.get('building_type'):
                            analysis += f"    Type: {doc.get('building_type')}\n"
                        analysis += "\n"
                
                return analysis
            else:
                return f"No lease documents found for location '{location}'. Try searching with different keywords or check if documents have been uploaded to the system."
                
        except Exception as e:
            return f"Error searching for location: {str(e)}"


class VectorSearchTool(BaseTool):
    """Tool for performing vector search to extract desired fields"""
    
    name: str = "vector_search_fields"
    description: str = "Performs vector search to find relevant documents and extract specific fields. Input should be JSON with 'query' and 'fields' (list of field names to extract)."
    openai_client: Any = Field(default=None, exclude=True)
    search_client: Any = Field(default=None, exclude=True)
    embedding_model: str = Field(default="text-embedding-ada-002", exclude=True)
    
    def __init__(self, openai_client, search_client: SearchClient, embedding_model: str = "text-embedding-ada-002", **kwargs):
        super().__init__(**kwargs)
        object.__setattr__(self, 'openai_client', openai_client)
        object.__setattr__(self, 'search_client', search_client)
        object.__setattr__(self, 'embedding_model', embedding_model)
    
    def _generate_query_embedding(self, query: str) -> List[float]:
        """Generate embedding for search query"""
        try:
            print(f"   🔤 Generating query embedding for: '{query[:100]}...'")
            response = self.openai_client.embeddings.create(
                model=self.embedding_model, 
                input=query
            )
            embedding = response.data[0].embedding
            print(f"   ✅ Query embedding generated: {len(embedding)} dimensions")
            return embedding
        except Exception as e:
            print(f"   ❌ Failed to generate query embedding: {e}")
            print(f"   Query: '{query}'")
            print(f"   Query length: {len(query)} chars")
            print(f"   OpenAI client: {type(self.openai_client)}")
            print(f"   Has API key: {'Yes' if hasattr(self.openai_client, 'api_key') else 'No'}")
            return []
    
    def _run(self, input_data: str) -> str:
        """Perform vector search and extract fields"""
        try:
            # Parse input JSON
            data = json.loads(input_data)
            query = data.get('query', '')
            fields = data.get('fields', ['location', 'lease_start_date', 'lease_end_date', 'building_area', 'building_type'])
            
            if not query:
                return "Error: No search query provided"
            
            # Generate query embedding
            query_embedding = self._generate_query_embedding(query)
            if not query_embedding:
                return "Error: Failed to generate query embedding"
            
            # Perform vector search
            vector_query = VectorizedQuery(vector=query_embedding, k_nearest_neighbors=10, fields="embedding")
            
            search_results = self.search_client.search(
                search_text=query,
                vector_queries=[vector_query],
                select=["id", "filename", "content", "location", "lease_start_date", "lease_end_date", 
                       "building_area", "area_unit", "building_type", "client_name"],
                top=15
            )
            
            results = []
            for result in search_results:
                doc_data = {
                    "filename": result.get("filename"),
                    "client_name": result.get("client_name"),
                    "score": result.get("@search.score", 0),
                    "content_preview": (result.get("content", "")[:200] + "...") if result.get("content") else ""
                }
                
                # Extract all relevant fields
                for field in ["location", "lease_start_date", "lease_end_date", "building_area", "area_unit", "building_type"]:
                    doc_data[field] = result.get(field, "")
                
                results.append(doc_data)
            
            if results:
                # Group by filename to avoid counting pages as separate documents
                documents = {}
                for result in results:
                    filename = result.get("filename", "Unknown")
                    if filename not in documents:
                        documents[filename] = result  # Take the highest scoring page for each document
                
                unique_docs = list(documents.values())
                
                # Provide comprehensive analysis
                analysis = f"🔍 SEARCH RESULTS for '{query}'\n"
                analysis += f"Found {len(unique_docs)} relevant documents (deduplicated from {len(results)} pages):\n\n"
                
                # Analyze the unique documents
                clients = set()
                locations = set()
                building_types = {}
                total_area = 0
                lease_dates = []
                
                for doc in unique_docs:
                    if doc.get("client_name"):
                        clients.add(doc["client_name"])
                    if doc.get("location"):
                        locations.add(doc["location"])
                    if doc.get("building_type"):
                        btype = doc["building_type"]
                        building_types[btype] = building_types.get(btype, 0) + 1
                    
                    # Parse area (only once per document)
                    area = doc.get("building_area")
                    if area and str(area).replace(".", "").replace(",", "").isdigit():
                        total_area += float(str(area).replace(",", ""))
                    
                    # Collect lease dates
                    if doc.get("lease_start_date") and doc.get("lease_end_date"):
                        lease_dates.append((doc["lease_start_date"], doc["lease_end_date"]))
                
                # Summary
                analysis += "📊 SUMMARY:\n"
                analysis += f"• {len(clients)} unique clients\n"
                analysis += f"• {len(locations)} unique locations\n"
                if building_types:
                    analysis += f"• Building types: {', '.join([f'{k} ({v})' for k, v in building_types.items()])}\n"
                if total_area > 0:
                    avg_area = total_area / len([doc for doc in unique_docs if doc.get("building_area")])
                    analysis += f"• Total area: {total_area:,.0f} sq ft (avg: {avg_area:,.0f} sq ft)\n"
                
                # Top results with detailed info
                analysis += f"\n📋 TOP MATCHING DOCUMENTS:\n"
                for i, doc in enumerate(unique_docs[:8], 1):
                    analysis += f"\n{i}. {doc.get('filename', 'Unknown file')} (Score: {doc['score']:.2f})\n"
                    analysis += f"   Client: {doc.get('client_name', 'Unknown')}\n"
                    analysis += f"   Location: {doc.get('location', 'Not specified')}\n"
                    if doc.get('lease_start_date'):
                        analysis += f"   Lease Period: {doc['lease_start_date']} to {doc.get('lease_end_date', 'Unknown')}\n"
                    if doc.get('building_area'):
                        analysis += f"   Area: {doc['building_area']} {doc.get('area_unit', '')}\n"
                    if doc.get('building_type'):
                        analysis += f"   Type: {doc['building_type']}\n"
                
                # Extract specific requested fields if provided
                if fields and any(f != "" for f in fields):
                    analysis += f"\n🎯 REQUESTED FIELDS SUMMARY:\n"
                    for field in fields:
                        field_values = [doc.get(field) for doc in unique_docs if doc.get(field)]
                        if field_values:
                            unique_values = list(set(field_values))
                            analysis += f"• {field}: {len(unique_values)} unique values\n"
                            if len(unique_values) <= 5:
                                analysis += f"  Values: {', '.join(unique_values)}\n"
                            else:
                                analysis += f"  Top values: {', '.join(unique_values[:5])}...\n"
                
                return analysis
            else:
                return f"No documents found matching '{query}'. Try different keywords or check if relevant documents have been uploaded."
                
        except json.JSONDecodeError:
            return "Error: Invalid JSON input format. Please provide JSON with 'query' and optional 'fields' array."
        except Exception as e:
            return f"Error performing vector search: {str(e)}"


class LeaseAnalysisTool(BaseTool):
    """Tool for comprehensive lease portfolio analysis"""
    
    name: str = "lease_analysis"
    description: str = "Performs comprehensive analysis of the entire lease portfolio. Input can be 'all' for full analysis or specific criteria like 'expiring_soon', 'by_client', 'by_location', etc."
    search_client: Any = Field(default=None, exclude=True)
    
    def __init__(self, search_client: SearchClient, **kwargs):
        super().__init__(**kwargs)
        object.__setattr__(self, 'search_client', search_client)
    
    def _run(self, analysis_type: str = "all") -> str:
        """Perform comprehensive lease analysis"""
        try:
            print(f"   📊 Performing lease analysis: {analysis_type}")
            
            # Get all documents
            search_results = self.search_client.search(
                search_text="*",
                select=["filename", "location", "client_name", "lease_start_date", 
                       "lease_end_date", "building_area", "area_unit", "building_type", 
                       "processed_at"],
                top=1000  # Get more documents for analysis
            )
            
            all_results = []
            for result in search_results:
                all_results.append({
                    "filename": result.get("filename"),
                    "location": result.get("location"),
                    "client_name": result.get("client_name"),
                    "lease_start_date": result.get("lease_start_date"),
                    "lease_end_date": result.get("lease_end_date"),
                    "building_area": result.get("building_area"),
                    "area_unit": result.get("area_unit"),
                    "building_type": result.get("building_type"),
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
                return "No lease documents found in the system. Please upload lease documents first."
            
            # Comprehensive analysis
            analysis = f"🏢 COMPREHENSIVE LEASE PORTFOLIO ANALYSIS\n"
            analysis += f"{'='*60}\n\n"
            
            # Basic statistics
            total_docs = len(documents)
            clients = set()
            locations = set()
            building_types = {}
            total_area = 0
            area_count = 0
            lease_dates = []
            
            from datetime import datetime, timedelta
            today = datetime.now()
            expiring_soon = []
            expired = []
            
            for doc in documents:
                # Client analysis
                if doc.get("client_name"):
                    clients.add(doc["client_name"])
                
                # Location analysis
                if doc.get("location"):
                    locations.add(doc["location"])
                
                # Building type analysis
                if doc.get("building_type"):
                    btype = doc["building_type"]
                    building_types[btype] = building_types.get(btype, 0) + 1
                
                # Area analysis
                area = doc.get("building_area")
                if area and str(area).replace(".", "").replace(",", "").isdigit():
                    total_area += float(str(area).replace(",", ""))
                    area_count += 1
                
                # Date analysis
                if doc.get("lease_end_date"):
                    try:
                        # Try to parse different date formats
                        end_date_str = doc["lease_end_date"]
                        end_date = None
                        
                        for fmt in ["%Y-%m-%d", "%m/%d/%Y", "%m-%d-%Y", "%B %d, %Y"]:
                            try:
                                end_date = datetime.strptime(end_date_str, fmt)
                                break
                            except ValueError:
                                continue
                        
                        if end_date:
                            days_until_expiry = (end_date - today).days
                            if days_until_expiry < 0:
                                expired.append((doc, abs(days_until_expiry)))
                            elif days_until_expiry <= 365:  # Expiring within a year
                                expiring_soon.append((doc, days_until_expiry))
                    except:
                        pass
            
            # Portfolio Overview
            analysis += "📊 PORTFOLIO OVERVIEW\n"
            analysis += "-" * 30 + "\n"
            analysis += f"• Total lease documents: {total_docs}\n"
            analysis += f"• Unique clients: {len(clients)}\n"
            analysis += f"• Unique locations: {len(locations)}\n"
            analysis += f"• Building types: {len(building_types)}\n"
            if area_count > 0:
                avg_area = total_area / area_count
                analysis += f"• Total leased area: {total_area:,.0f} sq ft\n"
                analysis += f"• Average area per lease: {avg_area:,.0f} sq ft\n"
            
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
                    analysis += f"• {client}: {count} lease(s)\n"
            
            # Building Type Analysis
            if building_types:
                analysis += f"\n🏗️  BUILDING TYPE ANALYSIS\n"
                analysis += "-" * 30 + "\n"
                sorted_types = sorted(building_types.items(), key=lambda x: x[1], reverse=True)
                for btype, count in sorted_types:
                    percentage = (count / total_docs) * 100
                    analysis += f"• {btype}: {count} leases ({percentage:.1f}%)\n"
            
            # Lease Expiry Analysis
            if expiring_soon or expired:
                analysis += f"\n⏰ LEASE EXPIRY ANALYSIS\n"
                analysis += "-" * 30 + "\n"
                
                if expired:
                    analysis += f"🔴 EXPIRED LEASES ({len(expired)}):\n"
                    for doc, days_expired in sorted(expired, key=lambda x: x[1], reverse=True)[:5]:
                        analysis += f"  • {doc.get('filename', 'Unknown')} - Expired {days_expired} days ago\n"
                        analysis += f"    Client: {doc.get('client_name', 'Unknown')}\n"
                        analysis += f"    Location: {doc.get('location', 'Unknown')}\n"
                
                if expiring_soon:
                    analysis += f"\n🟡 EXPIRING SOON ({len(expiring_soon)}):\n"
                    for doc, days_left in sorted(expiring_soon, key=lambda x: x[1])[:10]:
                        analysis += f"  • {doc.get('filename', 'Unknown')} - {days_left} days remaining\n"
                        analysis += f"    Client: {doc.get('client_name', 'Unknown')}\n"
                        analysis += f"    Location: {doc.get('location', 'Unknown')}\n"
            
            # Location Hotspots
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
                    analysis += f"• {location}: {count} lease(s)\n"
            
            analysis += f"\n💡 KEY INSIGHTS\n"
            analysis += "-" * 30 + "\n"
            
            if expired:
                analysis += f"• ⚠️  {len(expired)} leases have already expired\n"
            if expiring_soon:
                analysis += f"• 📅 {len(expiring_soon)} leases expire within the next year\n"
            if building_types:
                most_common_type = max(building_types, key=building_types.get)
                analysis += f"• 🏗️  Most common building type: {most_common_type} ({building_types[most_common_type]} leases)\n"
            if len(clients) > 0:
                top_client = max(client_counts, key=client_counts.get)
                analysis += f"• 👑 Largest client: {top_client} ({client_counts[top_client]} leases)\n"
            
            return analysis
            
        except Exception as e:
            return f"Error performing lease analysis: {str(e)}"


class MatchDataTool(BaseTool):
    """Tool for matching Excel data with vector database data based on location"""
    
    name: str = "match_data"
    description: str = "Matches data from EGA.xlsx with vector database entries based on location. Input should be JSON with 'excel_path' (path to Excel file), optional 'location_column' (default: auto-detect), and optional 'column_2' (default: second column)."
    search_client: Any = Field(default=None, exclude=True)
    
    def __init__(self, search_client: SearchClient, **kwargs):
        super().__init__(**kwargs)
        object.__setattr__(self, 'search_client', search_client)
    
    def _load_excel_data(self, excel_path: str):
        """Load data from Excel file"""
        try:
            import pandas as pd
            df = pd.read_excel(excel_path)
            return df
        except Exception as e:
            return None, f"Error loading Excel file: {e}"
    
    def _fuzzy_search_vector_by_location(self, location: str):
        """Search vector database using same logic as LocationMatchingTool"""
        try:
            # Use the SAME search logic as LocationMatchingTool that works for Helsinki
            # Strategy 1: Simple text search across all fields (this is what works!)
            search_results = self.search_client.search(
                search_text=f"{location}",
                select=["id", "filename", "content", "location", "client_name", 
                       "lease_start_date", "lease_end_date", "building_area", 
                       "area_unit", "building_type", "processed_at", "page_no"],
                top=100
            )
            
            # Strategy 2: Also try filter-based search for exact client matches
            try:
                filtered_results = self.search_client.search(
                    search_text="*",
                    filter=f"client_name eq '{location}'",
                    select=["id", "filename", "content", "location", "client_name", 
                           "lease_start_date", "lease_end_date", "building_area", 
                           "area_unit", "building_type", "processed_at", "page_no"],
                    top=100
                )
                # Combine results
                all_results = list(search_results) + list(filtered_results)
            except:
                # If filter fails, just use text search results
                all_results = list(search_results)
            
            # Use EXACT same logic as LocationMatchingTool
            matches = []
            for result in all_results:
                matches.append({
                    "id": result.get("id"),
                    "filename": result.get("filename"),
                    "location": result.get("location"),
                    "client_name": result.get("client_name"),
                    "lease_start_date": result.get("lease_start_date"),
                    "lease_end_date": result.get("lease_end_date"),
                    "building_area": result.get("building_area"),
                    "area_unit": result.get("area_unit"),
                    "building_type": result.get("building_type"),
                    "processed_at": result.get("processed_at"),
                    "content_preview": (result.get("content", "")[:100] + "...") if result.get("content") else "",
                    "search_score": result.get("@search.score", 0)
                })
            
            # Group by filename to avoid counting pages as separate documents (EXACT same logic)
            documents = {}
            for match in matches:
                filename = match.get("filename", "Unknown")
                if filename not in documents:
                    documents[filename] = match  # Take first page data for the document (SAME AS LocationMatchingTool)
            
            # Return the unique documents (not pages)
            sorted_matches = list(documents.values())
            
            return sorted_matches
            
        except Exception as e:
            print(f"   ❌ Error in location search: {e}")
            return []
    
    
    def _run(self, input_data: str) -> str:
        """Match Excel data with vector database data by searching column 2 locations"""
        try:
            # Parse input JSON
            data = json.loads(input_data)
            excel_path = data.get('excel_path', 'EGA.xlsx')
            
            # Load Excel data
            import pandas as pd
            try:
                excel_df = pd.read_excel(excel_path)
            except Exception as e:
                return f"❌ Error loading Excel file '{excel_path}': {e}"
            
            if excel_df.empty:
                return f"❌ Excel file '{excel_path}' is empty"
            
            if len(excel_df.columns) < 2:
                return f"❌ Excel file needs at least 2 columns. Found: {list(excel_df.columns)}"
            
            # Use column 2 (index 1) as the location to search for
            column_2 = excel_df.columns[1]
            
            print(f"   📊 Processing Excel file: {excel_path}")
            print(f"   🔍 Searching using column 2: {column_2}")
            
            # Process each row - extract column 2 and do location search
            results = []
            total_matches = 0
            matched_rows = 0
            
            for index, row in excel_df.iterrows():
                # Get all row data for display
                row_data = dict(row)
                
                # Extract location from column 2 to search with
                search_location = str(row[column_2]) if pd.notna(row[column_2]) else ""
                
                if not search_location.strip():
                    results.append({
                        "excel_row": index + 1,
                        "search_location": search_location,
                        "excel_data": row_data,
                        "vector_matches": 0,
                        "matches": []
                    })
                    continue
                
                print(f"   🔍 Row {index + 1}: Searching for '{search_location}'")
                
                # Do location search using the same logic as LocationMatchingTool
                vector_matches = self._fuzzy_search_vector_by_location(search_location)
                
                if vector_matches:
                    matched_rows += 1
                    total_matches += len(vector_matches)
                    print(f"      ✅ Found {len(vector_matches)} matches")
                else:
                    print(f"      ❌ No matches found")
                
                # Store result
                results.append({
                    "excel_row": index + 1,
                    "search_location": search_location,
                    "excel_data": row_data,
                    "vector_matches": len(vector_matches),
                    "matches": vector_matches[:5]  # Top 5 matches for display
                })
            
            # Format response
            response = f"📊 MATCH DATA RESULTS\n"
            response += f"{'='*50}\n\n"
            response += f"📈 SUMMARY:\n"
            response += f"   Excel rows processed: {len(results)}\n"
            response += f"   Rows with vector matches: {matched_rows}\n"
            response += f"   Total vector matches: {total_matches}\n"
            response += f"   Match rate: {(matched_rows/len(results)*100):.1f}%\n\n"
            
            response += f"📋 DETAILED RESULTS:\n"
            for result in results[:10]:  # Show first 10 results
                response += f"\n--- Row {result['excel_row']} ---\n"
                response += f"🔍 Searched for: '{result['search_location']}'\n"
                response += f"📊 Excel Data: {dict(list(result['excel_data'].items())[:3])}{'...' if len(result['excel_data']) > 3 else ''}\n"
                
                if result['matches']:
                    response += f"✅ Vector Matches ({result['vector_matches']}):\n"
                    for i, match in enumerate(result['matches'], 1):
                        response += f"  {i}. {match['filename']} (Score: {match['search_score']:.2f})\n"
                        response += f"     Location: {match['location']}\n"
                        response += f"     Client: {match['client_name']}\n"
                        if match['lease_start_date']:
                            response += f"     Lease: {match['lease_start_date']} - {match['lease_end_date']}\n"
                        if match['building_area']:
                            response += f"     Area: {match['building_area']} {match.get('area_unit', '')}\n"
                else:
                    response += f"❌ No vector matches found\n"
            
            if len(results) > 10:
                response += f"\n... and {len(results) - 10} more rows\n"
            
            return response
            
        except json.JSONDecodeError:
            return "❌ Error: Invalid JSON input format. Use: {'excel_path': 'EGA.xlsx'}"
        except Exception as e:
            return f"❌ Error matching data: {str(e)}"



class LeaseDocumentAgent:
    """Main agent class for lease document processing"""
    
    def __init__(self, 
                 azure_endpoint: str,
                 azure_key: str,
                 openai_api_key: str,
                 azure_search_endpoint: str,
                 azure_search_key: str,
                 search_index_name: str = "lease-documents",
                 openai_base_url: Optional[str] = None,
                 openai_model: str = "gpt-4o-mini",
                 openai_temperature: float = 0.1,
                 openai_embedding_model: str = "text-embedding-ada-002"):
        
        # Store configuration
        self.embedding_model = openai_embedding_model
        
        print("🔧 Initializing Lease Document Agent...")
        print(f"   Azure Form Recognizer: {azure_endpoint[:50]}...")
        print(f"   Azure Search: {azure_search_endpoint[:50]}...")
        print(f"   OpenAI Model: {openai_model}")
        print(f"   OpenAI Embedding Model: {openai_embedding_model}")
        if openai_base_url:
            print(f"   OpenAI Base URL: {openai_base_url[:50]}...")
        
        # Initialize OpenAI client with debugging
        try:
            print("🔍 Testing OpenAI connection...")
            openai_kwargs = {"api_key": openai_api_key}
            if openai_base_url:
                openai_kwargs["base_url"] = openai_base_url
            self.openai_client = openai.OpenAI(**openai_kwargs)
            
            # Test connection with a simple request
            test_response = self.openai_client.models.list()
            print("✅ OpenAI connection successful")
            
        except Exception as e:
            print(f"❌ OpenAI connection failed: {e}")
            print(f"   API Key: {openai_api_key[:10]}...{openai_api_key[-4:] if len(openai_api_key) > 10 else 'invalid'}")
            if openai_base_url:
                print(f"   Base URL: {openai_base_url}")
            raise ConnectionError(f"OpenAI connection failed: {e}")
        
        # Initialize Azure Search client with debugging
        try:
            print("🔍 Testing Azure Search connection...")
            
            self.search_client = SearchClient(
                endpoint=azure_search_endpoint,
                index_name=search_index_name,
                credential=AzureKeyCredential(azure_search_key)
            )
            
            # Test connection by trying to get index info
            try:
                # Simple test to verify connection
                search_results = list(self.search_client.search("*", top=1))
                print("✅ Azure Search connection successful")
            except Exception as search_e:
                if "404" in str(search_e) or "index" in str(search_e).lower():
                    print(f"⚠️  Azure Search connected but index '{search_index_name}' may not exist: {search_e}")
                    print("   This is OK - index will be created when first document is uploaded")
                else:
                    raise search_e
            
        except Exception as e:
            print(f"❌ Azure Search connection failed: {e}")
            print(f"   Endpoint: {azure_search_endpoint}")
            print(f"   Index: {search_index_name}")
            print(f"   Key: {azure_search_key[:10]}...{azure_search_key[-4:] if len(azure_search_key) > 10 else 'invalid'}")
            raise ConnectionError(f"Azure Search connection failed: {e}")
        
        # Initialize tools with debugging
        try:
            print("🛠️  Initializing tools...")
            self.tools = [
                AzureOCRTool(azure_endpoint=azure_endpoint, azure_key=azure_key),
                VectorStoreIngestionTool(
                    openai_client=self.openai_client,
                    search_client=self.search_client,
                    embedding_model=self.embedding_model
                ),
                LocationMatchingTool(search_client=self.search_client),
                VectorSearchTool(
                    openai_client=self.openai_client,
                    search_client=self.search_client,
                    embedding_model=self.embedding_model
                ),
                LeaseAnalysisTool(search_client=self.search_client),
                MatchDataTool(search_client=self.search_client)
            ]
            
            # Add utilities tools
            utilities_tools = create_utilities_tools(
                azure_endpoint=azure_endpoint,
                azure_key=azure_key,
                openai_client=self.openai_client,
                search_client=self.search_client,
                embedding_model=self.embedding_model
            )
            self.tools.extend(utilities_tools)
            
            # Add the three separate batch ingestion tools
            batch_ingestion_tools = create_batch_ingestion_tools(agent_instance=self)
            self.tools.extend(batch_ingestion_tools)
            print(f"✅ Initialized {len(self.tools)} tools successfully")
            
        except Exception as e:
            print(f"❌ Tool initialization failed: {e}")
            raise ConnectionError(f"Tool initialization failed: {e}")
        
        # Initialize LangChain LLM with debugging
        try:
            print("🤖 Initializing LangChain LLM...")
            langchain_kwargs = {
                "model": openai_model,
                "api_key": openai_api_key,
                "temperature": openai_temperature
            }
            if openai_base_url:
                langchain_kwargs["base_url"] = openai_base_url
            
            # Create LLM for ReAct agent
            self.llm = ChatOpenAI(**langchain_kwargs)
            
            # Test LLM
            test_response = self.llm.invoke("Hello")
            print("✅ LangChain LLM initialized successfully")
            
        except Exception as e:
            print(f"❌ LangChain LLM initialization failed: {e}")
            raise ConnectionError(f"LangChain LLM initialization failed: {e}")
        
        # Create ReAct agent with debugging
        try:
            print("🎯 Creating ReAct agent...")
            self._create_agent()
            print("✅ Agent initialization completed successfully!")
            
        except Exception as e:
            print(f"❌ Agent creation failed: {e}")
            raise ConnectionError(f"Agent creation failed: {e}")
    
    def _create_agent(self):
        """Create the ReAct agent with custom prompt"""
        
        react_prompt = PromptTemplate.from_template("""
You are an expert document analyst specializing in lease documents and utilities data (natural gas and electricity). You can have conversations and use tools to analyze lease documents, natural gas invoices, and electricity bills.

Available tools: {tools}

IMPORTANT: Always follow this exact format:
- For simple conversations (greetings, questions about capabilities): Use "Final Answer:" directly
- For lease analysis (searching, analyzing data): Use "Action:" with a tool, then "Final Answer:" with your analysis
- For utilities analysis (natural gas/electricity documents): Use utilities-specific tools like "utilities_location_matcher" or "utilities_vector_search_fields"
- For batch ingestion (processing PDF files): Use specific ingestion tools:
  - "batch_ingest_leases" for lease documents in /leases folder
  - "batch_ingest_electric" for electric documents in /electric folder  
  - "batch_ingest_natural_gas" for natural gas documents in /natural_gas folder

Examples:

Simple conversation:
Question: Hi
Thought: This is a greeting, no tools needed.
Final Answer: Hello! I'm your lease document analyst. I can help you search and analyze lease documents, and ingest new PDF documents. What would you like to know?

Batch ingestion:
Question: Ingest all lease documents
Thought: I need to use the lease batch ingestion tool to process all lease PDF files.
Action: batch_ingest_leases
Action Input: {{}}
Observation: [lease ingestion results]
Thought: The lease PDFs have been processed and are ready for analysis.
Final Answer: [summary of lease ingestion results]

Question: Process all electric bills
Thought: I need to use the electric batch ingestion tool to process all electric PDF files.
Action: batch_ingest_electric
Action Input: {{}}
Observation: [electric ingestion results]
Thought: The electric PDFs have been processed and are ready for analysis.
Final Answer: [summary of electric ingestion results]

Lease analysis:
Question: Find leases for client2
Thought: I need to search for lease documents for client2.
Action: location_matcher
Action Input: client2
Observation: [search results]
Thought: Now I can analyze the results.
Final Answer: [detailed analysis of client2's leases]

Utilities analysis:
Question: Find natural gas consumption for Chicago
Thought: I need to search for utilities documents related to Chicago.
Action: utilities_location_matcher
Action Input: Chicago
Observation: [utilities search results]
Thought: Now I can analyze the utilities data.
Final Answer: [detailed analysis of Chicago's natural gas and electricity consumption]

You have access to these tools: {tool_names}

Question: {input}
Thought: {agent_scratchpad}
""")
        
        # Create the agent
        self.agent = create_react_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=react_prompt
        )
        
        # Create agent executor
        self.agent_executor = AgentExecutor(
            agent=self.agent,
            tools=self.tools,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=10
        )
    
    def process_document(self, file_path: str, client_name: str = None) -> Dict[str, Any]:
        """Process a single document end-to-end"""
        
        query = f"""
        Process the lease document at {file_path} for client {client_name or 'unknown'}:
        1. Extract text using Azure OCR
        2. Ingest the text into the vector store with appropriate metadata
        3. Extract lease information fields (location, dates, building area, etc.)
        
        Return a summary of the processing results.
        """
        
        try:
            result = self.agent_executor.invoke({"input": query})
            return {
                "status": "success",
                "file_path": file_path,
                "client_name": client_name,
                "result": result["output"]
            }
        except Exception as e:
            return {
                "status": "error",
                "file_path": file_path,
                "client_name": client_name,
                "error": str(e)
            }
    
    def find_documents_by_location(self, location: str) -> Dict[str, Any]:
        """Find all documents matching a specific location"""
        
        query = f"Find all lease documents that match the location: {location}"
        
        try:
            result = self.agent_executor.invoke({"input": query})
            return {
                "status": "success",
                "location": location,
                "result": result["output"]
            }
        except Exception as e:
            return {
                "status": "error",
                "location": location,
                "error": str(e)
            }
    
    def search_for_fields(self, search_query: str, fields: List[str]) -> Dict[str, Any]:
        """Search for documents and extract specific fields"""
        
        query = f"""
        Search for documents related to: {search_query}
        Extract the following fields from the most relevant documents: {', '.join(fields)}
        """
        
        try:
            result = self.agent_executor.invoke({"input": query})
            return {
                "status": "success",
                "search_query": search_query,
                "fields": fields,
                "result": result["output"]
            }
        except Exception as e:
            return {
                "status": "error",
                "search_query": search_query,
                "fields": fields,
                "error": str(e)
            }
    
    def ask_question(self, question: str) -> Dict[str, Any]:
        """Ask the agent any question about lease documents"""
        
        try:
            # Check if this is a simple conversational question that doesn't need tools
            simple_queries = [
                'hi', 'hello', 'hey', 'good morning', 'good afternoon', 'good evening',
                'how are you', 'what can you do', 'help', 'what do you do',
                'thanks', 'thank you', 'bye', 'goodbye'
            ]
            
            question_lower = question.lower().strip()
            
            # Handle simple conversational responses directly
            if any(greeting in question_lower for greeting in simple_queries[:6]):  # greetings
                return {
                    "status": "success",
                    "question": question,
                    "result": "Hello! I'm your document analyst for lease documents and utilities data. I can help you search, analyze, and get insights from your lease documents, natural gas bills, and electricity invoices. What would you like to know?"
                }
            elif any(capability in question_lower for capability in ['what can you do', 'help', 'what do you do']):
                return {
                    "status": "success", 
                    "question": question,
                    "result": """I can help you with comprehensive document analysis including:

LEASE DOCUMENTS:
• Finding leases by client name or location
• Analyzing lease portfolios for insights and trends  
• Tracking lease expirations and renewal opportunities
• Processing PDF documents with OCR from /leases folder
• Searching for specific lease terms
• Generating portfolio summaries and reports

UTILITIES DOCUMENTS (Natural Gas & Electricity):
• Processing utilities invoices and bills from separate folders
• Finding utilities documents by location or client
• Analyzing consumption patterns and trends
• Tracking vendor information and costs
• Extracting meter readings and billing periods
• Comparing utilities usage across locations

DOCUMENT PROCESSING:
• Place lease PDFs in /leases folder (with optional client subfolders)
• Place electric bills in /electric folder (with optional client subfolders)
• Place natural gas bills in /natural_gas folder (with optional client subfolders)

Just ask me about any documents or analysis you need! For example:
LEASE EXAMPLES:
- "Ingest all lease documents" (processes /leases folder)
- "Find all leases for client2"
- "Show me office buildings in Chicago"

UTILITIES EXAMPLES:
- "Process all electric bills" (processes /electric folder)
- "Ingest natural gas documents" (processes /natural_gas folder)
- "Find natural gas consumption for Chicago"
- "Compare utilities costs by location"
"""
                }
            elif any(thanks in question_lower for thanks in ['thanks', 'thank you']):
                return {
                    "status": "success",
                    "question": question, 
                    "result": "You're welcome! Let me know if you need any other lease analysis or have questions about your portfolio."
                }
            elif any(bye in question_lower for bye in ['bye', 'goodbye']):
                return {
                    "status": "success",
                    "question": question,
                    "result": "Goodbye! Feel free to come back anytime you need lease document analysis."
                }
            
            # For lease-related queries, use the agent
            result = self.agent_executor.invoke({"input": question})
            return {
                "status": "success",
                "question": question,
                "result": result["output"]
            }
        except Exception as e:
            return {
                "status": "error",
                "question": question,
                "error": str(e)
            }
    
    def batch_ingest_pdfs(self, folder_path: Optional[str] = None, client_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Batch ingest all PDF documents from a folder
        
        Args:
            folder_path: Path to folder containing PDFs. If None, uses DEFAULT_INPUT_FOLDER from config
            client_name: Optional client name to associate with all documents
            
        Returns:
            Dictionary with processing results for each file
        """
        from config import get_config
        
        # Use provided folder or default from config
        if folder_path is None:
            config = get_config()
            folder_path = config.DEFAULT_INPUT_FOLDER
        
        folder = Path(folder_path)
        
        if not folder.exists():
            return {
                "status": "error",
                "error": f"Folder does not exist: {folder_path}",
                "results": []
            }
        
        # Find all PDF files recursively
        pdf_files = list(folder.rglob("*.pdf"))
        
        if not pdf_files:
            return {
                "status": "warning",
                "message": f"No PDF files found in folder: {folder_path}",
                "results": []
            }
        
        print(f"📁 Found {len(pdf_files)} PDF files to process in {folder_path}")
        
        results = []
        successful = 0
        failed = 0
        
        for pdf_file in pdf_files:
            print(f"\n📄 Processing: {pdf_file.name}")
            
            # Determine client name from folder structure if not provided
            effective_client_name = client_name
            if not effective_client_name:
                # Try to extract client name from parent directory
                if pdf_file.parent.name != folder.name:
                    effective_client_name = pdf_file.parent.name
                else:
                    effective_client_name = "Unknown"
            
            try:
                # Process individual document
                result = self.process_document(
                    file_path=str(pdf_file),
                    client_name=effective_client_name
                )
                
                results.append(result)
                
                if result["status"] == "success":
                    successful += 1
                    print(f"   ✅ Successfully processed {pdf_file.name}")
                else:
                    failed += 1
                    print(f"   ❌ Failed to process {pdf_file.name}: {result.get('error', 'Unknown error')}")
                    
            except Exception as e:
                failed += 1
                error_result = {
                    "status": "error",
                    "file_path": str(pdf_file),
                    "client_name": effective_client_name,
                    "error": str(e)
                }
                results.append(error_result)
                print(f"   ❌ Exception processing {pdf_file.name}: {e}")
        
        return {
            "status": "completed" if failed == 0 else "partial_success" if successful > 0 else "failed",
            "folder_path": str(folder_path),
            "total_files": len(pdf_files),
            "successful": successful,
            "failed": failed,
            "results": results
        }
    
    def ingest_all(self, client_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Simple method to ingest all PDF documents from the default folder
        No path needed - uses the DEFAULT_INPUT_FOLDER from config
        
        Args:
            client_name: Optional client name to associate with all documents
            
        Returns:
            Dictionary with processing results for each file
        """
        return self.batch_ingest_pdfs(folder_path=None, client_name=client_name)
    
    def process_utilities_document(self, file_path: str, utility_type: str = "auto", client_name: str = None) -> Dict[str, Any]:
        """Process a utilities document (natural gas or electricity) end-to-end"""
        
        query = f"""
        Process the utilities document at {file_path} for client {client_name or 'unknown'}:
        1. Extract text using utilities OCR extractor
        2. Determine if this is a natural gas or electricity document
        3. Ingest the text into the vector store with appropriate utilities metadata
        4. Extract utilities information fields (vendor, location, consumption, dates, etc.)
        
        Return a summary of the processing results.
        """
        
        try:
            result = self.agent_executor.invoke({"input": query})
            return {
                "status": "success",
                "file_path": file_path,
                "utility_type": utility_type,
                "client_name": client_name,
                "result": result["output"]
            }
        except Exception as e:
            return {
                "status": "error",
                "file_path": file_path,
                "utility_type": utility_type,
                "client_name": client_name,
                "error": str(e)
            }
    
    def find_utilities_by_location(self, location: str) -> Dict[str, Any]:
        """Find all utilities documents matching a specific location"""
        
        query = f"Find all utilities documents (natural gas and electricity) that match the location: {location}"
        
        try:
            result = self.agent_executor.invoke({"input": query})
            return {
                "status": "success",
                "location": location,
                "result": result["output"]
            }
        except Exception as e:
            return {
                "status": "error",
                "location": location,
                "error": str(e)
            }
    
    def search_utilities_for_fields(self, search_query: str, fields: List[str]) -> Dict[str, Any]:
        """Search utilities documents and extract specific fields"""
        
        query = f"""
        Search for utilities documents (natural gas and electricity) related to: {search_query}
        Extract the following fields from the most relevant documents: {', '.join(fields)}
        """
        
        try:
            result = self.agent_executor.invoke({"input": query})
            return {
                "status": "success",
                "search_query": search_query,
                "fields": fields,
                "result": result["output"]
            }
        except Exception as e:
            return {
                "status": "error",
                "search_query": search_query,
                "fields": fields,
                "error": str(e)
            }


def create_lease_agent_from_env() -> LeaseDocumentAgent:
    """Create agent instance from environment variables"""
    
    required_env_vars = [
        'AZURE_FORM_RECOGNIZER_ENDPOINT',
        'AZURE_FORM_RECOGNIZER_KEY',
        'OPENAI_API_KEY',
        'AZURE_SEARCH_ENDPOINT',
        'AZURE_SEARCH_KEY'
    ]
    
    missing_vars = [var for var in required_env_vars if not os.getenv(var)]
    if missing_vars:
        raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")
    
    return LeaseDocumentAgent(
        azure_endpoint=os.getenv('AZURE_FORM_RECOGNIZER_ENDPOINT'),
        azure_key=os.getenv('AZURE_FORM_RECOGNIZER_KEY'),
        openai_api_key=os.getenv('OPENAI_API_KEY'),
        azure_search_endpoint=os.getenv('AZURE_SEARCH_ENDPOINT'),
        azure_search_key=os.getenv('AZURE_SEARCH_KEY'),
        search_index_name=os.getenv('AZURE_SEARCH_INDEX_NAME', 'lease-documents')
    )


def create_lease_agent_from_config(config) -> LeaseDocumentAgent:
    """Create agent instance from configuration object"""
    
    # Validate configuration
    is_valid, missing_fields = config.validate()
    if not is_valid:
        raise ValueError(f"Configuration is incomplete. Missing fields: {', '.join(missing_fields)}")
    
    return LeaseDocumentAgent(
        azure_endpoint=config.AZURE_FORM_RECOGNIZER_ENDPOINT,
        azure_key=config.AZURE_FORM_RECOGNIZER_KEY,
        openai_api_key=config.OPENAI_API_KEY,
        azure_search_endpoint=config.AZURE_SEARCH_ENDPOINT,
        azure_search_key=config.AZURE_SEARCH_KEY,
        search_index_name=config.AZURE_SEARCH_INDEX_NAME,
        openai_base_url=config.OPENAI_BASE_URL or None,
        openai_model=config.OPENAI_MODEL,
        openai_temperature=config.OPENAI_TEMPERATURE,
        openai_embedding_model=config.OPENAI_EMBEDDING_MODEL
    )