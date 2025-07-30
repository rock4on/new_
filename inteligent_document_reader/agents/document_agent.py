"""
Main Document Agent that orchestrates the document processing workflow
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, List, Optional

import openai
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential

from ..tools import OCRTool, ExtractionTool, EmbeddingTool, IngestionTool, MatchingTool
from ..config import Config


class DocumentAgent:
    """
    Main agent that orchestrates document processing workflow:
    1. Read text using Azure OCR
    2. Extract metadata with OpenAI using forced Pydantic models  
    3. Create embedding
    4. Ingest into index
    """
    
    def __init__(self):
        self.config = Config()
        
        print("🤖 Initializing Inteligent Document Reader...")
        
        # Initialize OpenAI client
        self._init_openai_client()
        
        # Initialize Azure Search clients
        self._init_search_clients()
        
        # Initialize tools
        self._init_tools()
        
        print("✅ Inteligent Document Reader initialized successfully!")
    
    def _init_openai_client(self):
        """Initialize OpenAI client"""
        try:
            print("🔍 Initializing OpenAI client...")
            openai_kwargs = {"api_key": self.config.OPENAI_API_KEY}
            if self.config.OPENAI_BASE_URL:
                openai_kwargs["base_url"] = self.config.OPENAI_BASE_URL
            
            self.openai_client = openai.OpenAI(**openai_kwargs)
            
            # Test connection
            test_response = self.openai_client.models.list()
            print("✅ OpenAI client initialized")
            
        except Exception as e:
            print(f"❌ OpenAI initialization failed: {e}")
            raise ConnectionError(f"OpenAI initialization failed: {e}")
    
    def _init_search_clients(self):
        """Initialize Azure Search clients"""
        try:
            print("🔍 Initializing Azure Search clients...")
            
            # Main documents search client
            self.search_client = SearchClient(
                endpoint=self.config.AZURE_SEARCH_ENDPOINT,
                index_name=self.config.AZURE_SEARCH_INDEX_NAME,
                credential=AzureKeyCredential(self.config.AZURE_SEARCH_KEY)
            )
            
            # Utilities search client (separate index)
            self.utilities_search_client = SearchClient(
                endpoint=self.config.AZURE_SEARCH_ENDPOINT,
                index_name=self.config.AZURE_SEARCH_UTILITIES_INDEX_NAME,
                credential=AzureKeyCredential(self.config.AZURE_SEARCH_KEY)
            )
            
            print("✅ Azure Search clients initialized")
            
        except Exception as e:
            print(f"❌ Azure Search initialization failed: {e}")
            raise ConnectionError(f"Azure Search initialization failed: {e}")
    
    def _init_tools(self):
        """Initialize processing tools"""
        try:
            print("🛠️  Initializing processing tools...")
            
            self.ocr_tool = OCRTool(
                azure_endpoint=self.config.AZURE_FORM_RECOGNIZER_ENDPOINT,
                azure_key=self.config.AZURE_FORM_RECOGNIZER_KEY
            )
            
            self.extraction_tool = ExtractionTool(
                openai_client=self.openai_client
            )
            
            self.embedding_tool = EmbeddingTool(
                openai_client=self.openai_client,
                embedding_model=self.config.OPENAI_EMBEDDING_MODEL
            )
            
            self.ingestion_tool = IngestionTool(
                search_client=self.search_client
            )
            
            self.matching_tool = MatchingTool(
                search_client=self.search_client,
                utilities_search_client=self.utilities_search_client
            )
            
            print("✅ Processing tools initialized")
            
        except Exception as e:
            print(f"❌ Tools initialization failed: {e}")
            raise ConnectionError(f"Tools initialization failed: {e}")
    
    def process_document(self, file_path: str, doc_type: str, client_name: str = None) -> Dict[str, Any]:
        """
        Process a single document through the full workflow:
        OCR -> Extraction -> Embedding -> Ingestion
        
        Args:
            file_path: Path to PDF document
            doc_type: Type of document (lease, electricity, natural_gas)
            client_name: Optional client name
            
        Returns:
            Dictionary with processing results
        """
        try:
            print(f"\n📄 Processing {doc_type} document: {Path(file_path).name}")
            
            # Step 1: OCR - Extract text
            print("1️⃣ Extracting text with Azure OCR...")
            text = self.ocr_tool._run(file_path)
            if text.startswith("Error:"):
                return {"status": "error", "step": "ocr", "error": text}
            
            # Step 2: Extraction - Extract metadata with OpenAI forced Pydantic models
            print("2️⃣ Extracting metadata with OpenAI...")
            extraction_input = json.dumps({"text": text, "doc_type": doc_type})
            extraction_result = self.extraction_tool._run(extraction_input)
            if extraction_result.startswith("Error:"):
                return {"status": "error", "step": "extraction", "error": extraction_result}
            
            # Parse extracted metadata from the extraction tool's output
            metadata = {"client_name": client_name or "Unknown"}
            
            # Extract metadata from the structured output
            # This is a simplified parsing - in production you'd want more robust parsing
            if doc_type == "lease":
                # Parse lease fields from extraction result
                lines = extraction_result.split('\n')
                for line in lines:
                    if ':' in line and '•' in line:
                        field = line.split('•')[1].split(':')[0].strip().lower().replace(' ', '_')
                        value = line.split(':', 1)[1].strip()
                        if value != "Not found":
                            metadata[field] = value
            else:
                # Parse utilities fields from extraction result  
                lines = extraction_result.split('\n')
                for line in lines:
                    if ':' in line and '•' in line:
                        field = line.split('•')[1].split(':')[0].strip().lower().replace(' ', '_')
                        value = line.split(':', 1)[1].strip()
                        if value != "Not found":
                            metadata[field] = value
            
            # Step 3: Embedding - Create vector embedding
            print("3️⃣ Creating vector embedding...")
            embedding_json = self.embedding_tool._run(text)
            if embedding_json.startswith("Error:"):
                return {"status": "error", "step": "embedding", "error": embedding_json}
            
            embedding = json.loads(embedding_json)
            
            # Step 4: Ingestion - Store in search index
            print("4️⃣ Ingesting into search index...")
            ingestion_input = json.dumps({
                "text": text,
                "filename": Path(file_path).name,
                "doc_type": doc_type,
                "metadata": metadata,
                "embedding": embedding
            })
            
            # Choose the right search client based on document type
            if doc_type == "lease":
                self.ingestion_tool.search_client = self.search_client
            else:
                self.ingestion_tool.search_client = self.utilities_search_client
            
            ingestion_result = self.ingestion_tool._run(ingestion_input)
            if ingestion_result.startswith("Error:"):
                return {"status": "error", "step": "ingestion", "error": ingestion_result}
            
            print(f"✅ Successfully processed {doc_type} document: {Path(file_path).name}")
            
            return {
                "status": "success",
                "file_path": file_path,
                "doc_type": doc_type,
                "client_name": client_name,
                "metadata": metadata,
                "message": ingestion_result
            }
            
        except Exception as e:
            error_msg = f"Unexpected error processing document: {str(e)}"
            print(f"❌ {error_msg}")
            return {"status": "error", "step": "unexpected", "error": error_msg}
    
    def process_folder(self, folder_path: str, doc_type: str, client_name: str = None) -> Dict[str, Any]:
        """
        Process all PDF documents in a folder
        
        Args:
            folder_path: Path to folder containing PDFs
            doc_type: Type of documents (lease, electricity, natural_gas)
            client_name: Optional client name for all documents
            
        Returns:
            Dictionary with batch processing results
        """
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
            # Determine client name from folder structure if not provided
            effective_client_name = client_name
            if not effective_client_name:
                if pdf_file.parent.name != folder.name:
                    effective_client_name = pdf_file.parent.name
                else:
                    effective_client_name = "Unknown"
            
            result = self.process_document(
                file_path=str(pdf_file),
                doc_type=doc_type,
                client_name=effective_client_name
            )
            
            results.append(result)
            
            if result["status"] == "success":
                successful += 1
            else:
                failed += 1
        
        return {
            "status": "completed" if failed == 0 else "partial_success" if successful > 0 else "failed",
            "folder_path": str(folder_path),
            "doc_type": doc_type,
            "total_files": len(pdf_files),
            "successful": successful,
            "failed": failed,
            "results": results
        }
    
    def process_leases(self, folder_path: str = None, client_name: str = None) -> Dict[str, Any]:
        """Process lease documents from default or specified folder"""
        folder_path = folder_path or self.config.LEASES_FOLDER
        return self.process_folder(folder_path, "lease", client_name)
    
    def process_electricity(self, folder_path: str = None, client_name: str = None) -> Dict[str, Any]:
        """Process electricity bills from default or specified folder"""
        folder_path = folder_path or self.config.ELECTRICITY_FOLDER
        return self.process_folder(folder_path, "electricity", client_name)
    
    def process_natural_gas(self, folder_path: str = None, client_name: str = None) -> Dict[str, Any]:
        """Process natural gas bills from default or specified folder"""
        folder_path = folder_path or self.config.NATURAL_GAS_FOLDER
        return self.process_folder(folder_path, "natural_gas", client_name)
    
    def match_excel(self, excel_path: str) -> str:
        """Match Excel data with vector database by location"""
        matching_input = json.dumps({"excel_path": excel_path})
        return self.matching_tool._run(matching_input)
    
