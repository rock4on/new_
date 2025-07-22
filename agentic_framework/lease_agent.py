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
import openai
from pydantic import BaseModel, Field


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
        object.__setattr__(self, 'azure_client', DocumentAnalysisClient(
            endpoint=azure_endpoint,
            credential=AzureKeyCredential(azure_key)
        ))
    
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
    
    def __init__(self, openai_client, search_client: SearchClient, **kwargs):
        super().__init__(**kwargs)
        object.__setattr__(self, 'openai_client', openai_client)
        object.__setattr__(self, 'search_client', search_client)
    
    def _generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for text"""
        try:
            response = self.openai_client.embeddings.create(
                model="text-embedding-ada-002",
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
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
    """Tool for matching documents by location"""
    
    name: str = "location_matcher"
    description: str = "Finds documents matching a specific location. Input should be the location string to search for."
    search_client: Any = Field(default=None, exclude=True)
    
    def __init__(self, search_client: SearchClient, **kwargs):
        super().__init__(**kwargs)
        object.__setattr__(self, 'search_client', search_client)
    
    def _run(self, location: str) -> str:
        """Find documents matching the specified location"""
        try:
            # Search for documents with matching location
            search_results = self.search_client.search(
                search_text="*",
                filter=f"search.ismatch('{location}', 'location')",
                select=["id", "filename", "location", "client_name", "doc_type", "processed_at"],
                top=10
            )
            
            matches = []
            for result in search_results:
                matches.append({
                    "id": result.get("id"),
                    "filename": result.get("filename"),
                    "location": result.get("location"),
                    "client_name": result.get("client_name"),
                    "doc_type": result.get("doc_type"),
                    "processed_at": result.get("processed_at")
                })
            
            if matches:
                return f"Found {len(matches)} documents matching location '{location}': {json.dumps(matches, indent=2)}"
            else:
                return f"No documents found matching location '{location}'"
                
        except Exception as e:
            return f"Error searching for location: {str(e)}"


class VectorSearchTool(BaseTool):
    """Tool for performing vector search to extract desired fields"""
    
    name: str = "vector_search_fields"
    description: str = "Performs vector search to find relevant documents and extract specific fields. Input should be JSON with 'query' and 'fields' (list of field names to extract)."
    openai_client: Any = Field(default=None, exclude=True)
    search_client: Any = Field(default=None, exclude=True)
    
    def __init__(self, openai_client, search_client: SearchClient, **kwargs):
        super().__init__(**kwargs)
        object.__setattr__(self, 'openai_client', openai_client)
        object.__setattr__(self, 'search_client', search_client)
    
    def _generate_query_embedding(self, query: str) -> List[float]:
        """Generate embedding for search query"""
        try:
            response = self.openai_client.embeddings.create(
                model="text-embedding-ada-002",
                input=query
            )
            return response.data[0].embedding
        except Exception as e:
            return []
    
    def _run(self, input_data: str) -> str:
        """Perform vector search and extract fields"""
        try:
            # Parse input JSON
            data = json.loads(input_data)
            query = data.get('query', '')
            fields = data.get('fields', [])
            
            if not query:
                return "Error: No search query provided"
            
            # Generate query embedding
            query_embedding = self._generate_query_embedding(query)
            if not query_embedding:
                return "Error: Failed to generate query embedding"
            
            # Perform vector search
            vector_query = VectorizedQuery(vector=query_embedding, k_nearest_neighbors=5, fields="embedding")
            
            search_results = self.search_client.search(
                search_text=query,
                vector_queries=[vector_query],
                select=["id", "filename", "content", "location", "lease_start_date", "lease_end_date", 
                       "building_area", "area_unit", "building_type", "client_name"],
                top=5
            )
            
            results = []
            for result in search_results:
                doc_data = {
                    "id": result.get("id"),
                    "filename": result.get("filename"),
                    "score": result.get("@search.score", 0)
                }
                
                # Extract requested fields
                for field in fields:
                    doc_data[field] = result.get(field, "")
                
                results.append(doc_data)
            
            if results:
                return f"Found {len(results)} relevant documents: {json.dumps(results, indent=2)}"
            else:
                return f"No documents found for query '{query}'"
                
        except json.JSONDecodeError:
            return "Error: Invalid JSON input format"
        except Exception as e:
            return f"Error performing vector search: {str(e)}"


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
                 openai_temperature: float = 0.1):
        
        # Initialize OpenAI client
        openai_kwargs = {"api_key": openai_api_key}
        if openai_base_url:
            openai_kwargs["base_url"] = openai_base_url
        self.openai_client = openai.OpenAI(**openai_kwargs)
        
        # Initialize Azure Search client
        self.search_client = SearchClient(
            endpoint=azure_search_endpoint,
            index_name=search_index_name,
            credential=AzureKeyCredential(azure_search_key)
        )
        
        # Initialize tools
        self.tools = [
            AzureOCRTool(azure_endpoint=azure_endpoint, azure_key=azure_key),
            VectorStoreIngestionTool(
                openai_client=self.openai_client,
                search_client=self.search_client
            ),
            LocationMatchingTool(search_client=self.search_client),
            VectorSearchTool(
                openai_client=self.openai_client,
                search_client=self.search_client
            )
        ]
        
        # Initialize LangChain LLM
        langchain_kwargs = {
            "model": openai_model,
            "api_key": openai_api_key,
            "temperature": openai_temperature
        }
        if openai_base_url:
            langchain_kwargs["base_url"] = openai_base_url
        self.llm = ChatOpenAI(**langchain_kwargs)
        
        # Create ReAct agent
        self._create_agent()
    
    def _create_agent(self):
        """Create the ReAct agent with custom prompt"""
        
        react_prompt = PromptTemplate.from_template("""
You are a lease document processing agent with access to specialized tools for document analysis.

You have access to the following tools:
{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

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
        openai_temperature=config.OPENAI_TEMPERATURE
    )