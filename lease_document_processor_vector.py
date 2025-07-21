#!/usr/bin/env python3
"""
Lease Document Processor with Azure Form Recognizer and AI Search Integration

Processes PDF lease documents using Azure Form Recognizer OCR to extract structured lease information.
Uses Azure Document Intelligence for text extraction, regex/AI for information extraction, and
Azure AI Search for vectorization and document indexing.

Features:
- Azure Form Recognizer for high-quality OCR
- Regex-based extraction with AI fallback
- OpenAI embeddings for document vectorization
- Azure AI Search integration with structured fields
- Client-based folder organization support
- Page-by-page processing and indexing
- Detailed timing measurements for each document
- Excel output with multiple sheets (Lease Information, Processing Times, Summary)
- Thread-safe operations with configurable worker threads

Document Structure:
- Supports client-organized folders: leases/client_name/documents.pdf
- Extracts client name from folder structure
- Indexes documents with fields: id, content, embedding, filename, page_no, doc_type, client_name, language, isTranslated

Default behavior:
- Processes all PDF files in 'leases' folder and client subdirectories
- Outputs results to 'lease_results.xlsx'
- Uses Azure OCR for text extraction and regex/AI for information extraction
- Vectorizes and uploads documents to Azure AI Search if configured
- Uses 4 worker threads for parallel processing
"""

import os
import sys
import re
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import csv
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd

# Import required dependencies first
try:
    from azure.ai.formrecognizer import DocumentAnalysisClient
    from azure.core.credentials import AzureKeyCredential
    from azure.core.pipeline.transport import RequestsTransport
    from azure.search.documents import SearchClient
    from azure.search.documents.indexes import SearchIndexClient
    from azure.search.documents.models import VectorizedQuery
    import openai
    from pydantic import BaseModel, Field
    import uuid
    import hashlib
except ImportError as e:
    print(f"❌ Missing required dependency: {e}")
    print("💡 Install with: pip install azure-ai-formrecognizer azure-core azure-search-documents openai pydantic")
    sys.exit(1)


class LeaseInformation(BaseModel):
    """Pydantic model for structured lease information extraction"""
    
    description: Optional[str] = Field(
        None, 
        description="Description of supporting documentation for lease details (e.g. signed lease agreement)"
    )
    location: Optional[str] = Field(
        None, 
        description="Location per underlying support"
    )
    lease_start_date: Optional[str] = Field(
        None, 
        description="Lease start date in YYYY-MM-DD format if possible"
    )
    lease_end_date: Optional[str] = Field(
        None, 
        description="Lease end date in YYYY-MM-DD format if possible"
    )
    building_area: Optional[str] = Field(
        None, 
        description="Building area per underlying support (numeric value)"
    )
    area_unit: Optional[str] = Field(
        None, 
        description="Unit of measure of building area (e.g., sq ft, sq m, sqft)"
    )
    building_type: Optional[str] = Field(
        None, 
        description="Building Type (i.e., Office vs. Warehouse)"
    )


class LeaseDocumentProcessor:
    """Extract structured lease information from PDF documents using Azure Form Recognizer"""
    
    def __init__(self, azure_endpoint: Optional[str] = None, azure_key: Optional[str] = None, 
                 openai_api_key: Optional[str] = None, azure_search_endpoint: Optional[str] = None,
                 azure_search_key: Optional[str] = None, search_index_name: Optional[str] = None,
                 max_workers: int = 4, ssl_verify: bool = False):
        """
        Initialize the lease document processor.
        
        Args:
            azure_endpoint: Azure Form Recognizer endpoint (or from environment)
            azure_key: Azure Form Recognizer key (or from environment)
            openai_api_key: OpenAI API key (or from environment)
            azure_search_endpoint: Azure AI Search endpoint (or from environment)
            azure_search_key: Azure AI Search key (or from environment)
            search_index_name: Azure AI Search index name (or from environment)
            max_workers: Maximum number of worker threads for parallel processing
            ssl_verify: Whether to verify SSL certificates
        """
        # Azure Form Recognizer configuration
        self.azure_endpoint = azure_endpoint or os.getenv('AZURE_FORM_RECOGNIZER_ENDPOINT', '')
        self.azure_key = azure_key or os.getenv('AZURE_FORM_RECOGNIZER_KEY', '')
        
        if not self.azure_endpoint or not self.azure_key:
            raise ValueError("Azure Form Recognizer endpoint and key are required. Set AZURE_FORM_RECOGNIZER_ENDPOINT and AZURE_FORM_RECOGNIZER_KEY environment variables or pass parameters.")
        
        # Set up Azure client with optional SSL verification
        if not ssl_verify:
            transport = RequestsTransport(connection_verify=False)
            self.azure_client = DocumentAnalysisClient(
                endpoint=self.azure_endpoint, 
                credential=AzureKeyCredential(self.azure_key),
                transport=transport
            )
        else:
            self.azure_client = DocumentAnalysisClient(
                endpoint=self.azure_endpoint, 
                credential=AzureKeyCredential(self.azure_key)
            )
        
        # Initialize OpenAI client
        self.client = openai.OpenAI(
            api_key=openai_api_key or os.getenv('OPENAI_API_KEY')
        )
        
        # Check if API key is available
        if not (openai_api_key or os.getenv('OPENAI_API_KEY')):
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY environment variable or pass openai_api_key parameter.")
        
        # Azure AI Search configuration
        self.azure_search_endpoint = azure_search_endpoint or os.getenv('AZURE_SEARCH_ENDPOINT', '')
        self.azure_search_key = azure_search_key or os.getenv('AZURE_SEARCH_KEY', '')
        self.search_index_name = search_index_name or os.getenv('AZURE_SEARCH_INDEX_NAME', 'electricity-documents')
        
        # Initialize Azure AI Search clients if configured
        self.search_client = None
        self.search_index_client = None
        if self.azure_search_endpoint and self.azure_search_key:
            try:
                self.search_client = SearchClient(
                    endpoint=self.azure_search_endpoint,
                    index_name=self.search_index_name,
                    credential=AzureKeyCredential(self.azure_search_key)
                )
                self.search_index_client = SearchIndexClient(
                    endpoint=self.azure_search_endpoint,
                    credential=AzureKeyCredential(self.azure_search_key)
                )
            except Exception as e:
                print(f"⚠️  Azure AI Search initialization failed: {e}")
                print("   Vectorization and search indexing will be disabled")
        
        # Threading settings
        self.max_workers = max_workers
        self.lock = threading.Lock()
        
        # Define the lease information fields we want to extract
        self.lease_fields = {
            'description': 'Description of supporting documentation for lease details (e.g. signed lease agreement)',
            'location': 'Location per underlying support',
            'lease_start_date': 'Lease start date',
            'lease_end_date': 'Lease end date',
            'building_area': 'Building area per underlying support',
            'area_unit': 'Unit of measure of building area',
            'building_type': 'Building Type (i.e., Office vs. Warehouse)'
        }
    
    def extract_text_azure(self, pdf_path: Path) -> str:
        """
        Extract text from PDF using Azure Form Recognizer.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Extracted text as string
        """
        try:
            with open(pdf_path, "rb") as f:
                poller = self.azure_client.begin_analyze_document("prebuilt-document", document=f)
                result = poller.result()
            
            # Gather all text for flexible regex matching
            full_text = ""
            for page in result.pages:
                for line in page.lines:
                    full_text += line.content + "\n"
            
            return full_text
            
        except Exception as e:
            with self.lock:
                print(f"❌ Azure OCR extraction failed: {e}")
            return ""
    
    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for text using OpenAI's text-embedding model.
        
        Args:
            text: Text to generate embedding for
            
        Returns:
            List of float values representing the embedding
        """
        try:
            response = self.client.embeddings.create(
                model="text-embedding-ada-002",
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            with self.lock:
                print(f"❌ Embedding generation failed: {e}")
            return []
    
    def extract_page_number_from_text(self, text: str, page_index: int) -> int:
        """
        Extract page number from text or use page index as fallback.
        
        Args:
            text: Text content of the page
            page_index: Zero-based page index from PDF processing
            
        Returns:
            Page number (1-based)
        """
        # Try to find page number in text
        page_patterns = [
            r"Page\s+(\d+)",
            r"Page\s+(\d+)\s+of\s+\d+",
            r"^\s*(\d+)\s*$"  # Standalone number at beginning or end of line
        ]
        
        for pattern in page_patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
            if match:
                try:
                    return int(match.group(1))
                except ValueError:
                    continue
        
        # Fallback to page index + 1 (convert to 1-based)
        return page_index + 1
    
    def extract_lease_info_regex(self, text: str) -> Dict[str, Any]:
        """
        Extract lease information using regex patterns similar to azure_ocr.py logic.
        
        Args:
            text: Raw text from PDF
            
        Returns:
            Dictionary with extracted lease information
        """
        extracted = {}
        
        # 1. Description - look for lease-related terms
        desc_match = re.search(r"(lease agreement|lease contract|rental agreement|tenancy agreement)", text, re.IGNORECASE)
        extracted['description'] = desc_match.group(1) if desc_match else None
        
        # 2. Location - look for address patterns
        location_patterns = [
            r"([A-Za-z0-9\s,.-]+(?:Street|St|Drive|Dr|Avenue|Ave|Road|Rd|Boulevard|Blvd|Lane|Ln|Court|Ct|Circle|Cir|Way|Place|Pl)[A-Za-z0-9\s,.-]*\d{5})",
            r"([A-Za-z0-9\s,.-]+(?:Street|St|Drive|Dr|Avenue|Ave|Road|Rd|Boulevard|Blvd|Lane|Ln|Court|Ct|Circle|Cir|Way|Place|Pl)[A-Za-z0-9\s,.-]*)",
            r"Property Address:?\s*([A-Za-z0-9\s,.-]+)",
            r"Premises:?\s*([A-Za-z0-9\s,.-]+)",
            r"Located at:?\s*([A-Za-z0-9\s,.-]+)"
        ]
        
        location = None
        for pattern in location_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                location = match.group(1).strip()
                break
        extracted['location'] = location
        
        # 3. Lease start date
        start_date_patterns = [
            r"(?:lease\s+)?(?:start|commencement|beginning)\s+date:?\s*([A-Za-z]+\s+\d{1,2},?\s*\d{4})",
            r"(?:lease\s+)?(?:start|commencement|beginning)\s+date:?\s*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})",
            r"(?:term\s+)?(?:commences|begins):?\s*([A-Za-z]+\s+\d{1,2},?\s*\d{4})",
            r"(?:term\s+)?(?:commences|begins):?\s*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})"
        ]
        
        start_date = None
        for pattern in start_date_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                start_date = match.group(1).strip()
                break
        extracted['lease_start_date'] = start_date
        
        # 4. Lease end date
        end_date_patterns = [
            r"(?:lease\s+)?(?:end|expiration|termination)\s+date:?\s*([A-Za-z]+\s+\d{1,2},?\s*\d{4})",
            r"(?:lease\s+)?(?:end|expiration|termination)\s+date:?\s*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})",
            r"(?:term\s+)?(?:expires|ends):?\s*([A-Za-z]+\s+\d{1,2},?\s*\d{4})",
            r"(?:term\s+)?(?:expires|ends):?\s*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})"
        ]
        
        end_date = None
        for pattern in end_date_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                end_date = match.group(1).strip()
                break
        extracted['lease_end_date'] = end_date
        
        # 5. Building area
        area_patterns = [
            r"(?:building|premises|rental|leased)\s+area:?\s*([\d,]+\.?\d*)",
            r"(?:square\s+feet|sq\.?\s*ft\.?|sqft):?\s*([\d,]+\.?\d*)",
            r"(?:square\s+meters|sq\.?\s*m\.?|sqm):?\s*([\d,]+\.?\d*)",
            r"area:?\s*([\d,]+\.?\d*)\s*(?:square\s+feet|sq\.?\s*ft\.?|sqft)",
            r"area:?\s*([\d,]+\.?\d*)\s*(?:square\s+meters|sq\.?\s*m\.?|sqm)"
        ]
        
        building_area = None
        area_unit = None
        for pattern in area_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                building_area = match.group(1).replace(',', '')
                # Determine unit from the pattern
                if re.search(r"(?:square\s+feet|sq\.?\s*ft\.?|sqft)", pattern, re.IGNORECASE):
                    area_unit = "sq ft"
                elif re.search(r"(?:square\s+meters|sq\.?\s*m\.?|sqm)", pattern, re.IGNORECASE):
                    area_unit = "sq m"
                break
        
        extracted['building_area'] = building_area
        extracted['area_unit'] = area_unit
        
        # 6. Building type
        type_patterns = [
            r"(?:building|property|premises)\s+type:?\s*(office|warehouse|retail|industrial|commercial|residential)",
            r"(?:use|usage):?\s*(office|warehouse|retail|industrial|commercial|residential)",
            r"(?:zoned|classified)\s+as:?\s*(office|warehouse|retail|industrial|commercial|residential)"
        ]
        
        building_type = None
        for pattern in type_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                building_type = match.group(1).capitalize()
                break
        extracted['building_type'] = building_type
        
        return extracted
    
    def extract_lease_info_ai(self, text: str) -> Dict[str, Any]:
        """
        Use AI with structured outputs to extract lease information from text.
        
        Args:
            text: Raw text from PDF
            
        Returns:
            Dictionary with extracted lease information
        """
        prompt = f"""
        Extract lease information from the following document text. Identify and extract all available fields.
        
        For dates, use YYYY-MM-DD format if possible.
        For building area, extract the numeric value only.
        For area unit, use standard units like "sq ft", "sq m", "sqft", etc.
        
        Document text:
        {text}
        """
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a document analyst specializing in lease agreement analysis. Extract information accurately from the provided text."},
                    {"role": "user", "content": prompt}
                ],
                response_format=LeaseInformation,
                temperature=0.1,
                max_tokens=1000
            )
            
            # Parse the structured response
            lease_info = response.choices[0].message.parsed
            
            # Convert Pydantic model to dictionary
            extracted_data = lease_info.model_dump()
            
            # Check if AI extracted any meaningful data
            has_data = any(value is not None and str(value).strip() for value in extracted_data.values())
            
            if not has_data:
                print("⚠️  AI extraction completed but no lease information was found in the document")
                print("📄 This may indicate:")
                print("   - Document doesn't contain lease information")
                print("   - OCR quality is poor") 
                print("   - Document format is not recognized")
            else:
                # Print what was successfully extracted
                extracted_fields = [field for field, value in extracted_data.items() 
                                  if value is not None and str(value).strip()]
                print(f"✅ AI successfully extracted: {', '.join(extracted_fields)}")
            
            return extracted_data
            
        except Exception as e:
            print(f"❌ AI extraction failed with error: {e}")
            print("📄 This could be due to:")
            print("   - OpenAI API issues")
            print("   - Invalid API key")
            print("   - Network connectivity problems")
            print("   - Text format issues")
            return {field: None for field in self.lease_fields.keys()}
    
    def upload_to_search_index(self, document_data: Dict[str, Any], client_name: str, 
                              filename: str, page_no: int, content: str, embedding: List[float]) -> bool:
        """
        Upload document to Azure AI Search with vectorization.
        
        Args:
            document_data: Extracted lease information
            client_name: Name of the client (from folder structure)
            filename: Name of the PDF file
            page_no: Page number in the document
            content: Text content of the page
            embedding: Vector embedding of the content
            
        Returns:
            True if upload successful, False otherwise
        """
        if not self.search_client or not embedding:
            return False
        
        try:
            # Generate unique document ID
            doc_id = hashlib.md5(f"{client_name}_{filename}_{page_no}".encode()).hexdigest()
            
            # Create document for indexing
            search_document = {
                "id": doc_id,
                "content": content,
                "embedding": embedding,
                "filename": filename,
                "page_no": page_no,
                "doc_type": "Electricity",
                "client_name": client_name,
                "language": "en",
                "isTranslated": "no",
                # Add lease information as metadata
                "description": document_data.get('description'),
                "location": document_data.get('location'),
                "lease_start_date": document_data.get('lease_start_date'),
                "lease_end_date": document_data.get('lease_end_date'),
                "building_area": document_data.get('building_area'),
                "area_unit": document_data.get('area_unit'),
                "building_type": document_data.get('building_type'),
                "processed_at": datetime.now().isoformat()
            }
            
            # Upload to search index
            result = self.search_client.upload_documents([search_document])
            
            if result and len(result) > 0 and result[0].succeeded:
                with self.lock:
                    print(f"✅ Uploaded to search index: {filename} (page {page_no})")
                return True
            else:
                with self.lock:
                    print(f"❌ Failed to upload to search index: {filename} (page {page_no})")
                return False
                
        except Exception as e:
            with self.lock:
                print(f"❌ Search index upload failed: {e}")
            return False
    
    def process_pdf(self, pdf_path: Path, extraction_method: str = 'regex', client_name: str = None) -> Dict[str, Any]:
        """
        Process a PDF lease document and extract structured information.
        
        Args:
            pdf_path: Path to PDF file
            extraction_method: Extraction method to use ('regex' or 'ai')
            client_name: Name of the client (extracted from folder structure)
            
        Returns:
            Dictionary with extracted lease information
        """
        start_time = time.time()
        
        with self.lock:
            print(f"📄 Processing lease document: {pdf_path}")
        
        # Extract text using Azure Form Recognizer with page-by-page processing
        ocr_start_time = time.time()
        
        try:
            with open(pdf_path, "rb") as f:
                poller = self.azure_client.begin_analyze_document("prebuilt-document", document=f)
                result = poller.result()
        except Exception as e:
            with self.lock:
                print(f"❌ Azure OCR extraction failed: {e}")
            return {field: None for field in self.lease_fields.keys()}
        
        if not result.pages:
            with self.lock:
                print("❌ No pages extracted from PDF")
            return {field: None for field in self.lease_fields.keys()}
        
        # Process each page and prepare for vectorization
        all_text = ""
        vectorization_tasks = []
        
        for page_idx, page in enumerate(result.pages):
            page_text = ""
            for line in page.lines:
                page_text += line.content + "\n"
            
            all_text += page_text + "\n"
            
            # Prepare page for vectorization if we have search client and client name
            if self.search_client and client_name and page_text.strip():
                page_number = self.extract_page_number_from_text(page_text, page_idx)
                vectorization_tasks.append({
                    'page_number': page_number,
                    'content': page_text.strip(),
                    'filename': pdf_path.name
                })
        
        ocr_time = time.time() - ocr_start_time
        
        if not all_text.strip():
            with self.lock:
                print("❌ No text extracted from PDF")
            return {field: None for field in self.lease_fields.keys()}
        
        with self.lock:
            print(f"📝 Extracted {len(all_text)} characters from {len(result.pages)} pages in {ocr_time:.2f}s")
        
        # Extract lease information using selected method
        extraction_start_time = time.time()
        
        if extraction_method == 'regex':
            with self.lock:
                print("🔍 Using regex for lease information extraction...")
            lease_info = self.extract_lease_info_regex(all_text)
        else:  # ai
            with self.lock:
                print("🤖 Using AI for lease information extraction...")
            lease_info = self.extract_lease_info_ai(all_text)
        
        extraction_time = time.time() - extraction_start_time
        
        # Process vectorization and upload to Azure AI Search
        vectorization_time = 0
        upload_time = 0
        uploaded_pages = 0
        
        if vectorization_tasks and client_name:
            vectorization_start = time.time()
            
            with self.lock:
                print(f"🔤 Starting vectorization for {len(vectorization_tasks)} pages...")
            
            for task in vectorization_tasks:
                try:
                    # Generate embedding
                    embedding = self.generate_embedding(task['content'])
                    
                    if embedding:
                        # Upload to search index
                        upload_success = self.upload_to_search_index(
                            lease_info, 
                            client_name,
                            task['filename'],
                            task['page_number'],
                            task['content'],
                            embedding
                        )
                        
                        if upload_success:
                            uploaded_pages += 1
                            
                except Exception as e:
                    with self.lock:
                        print(f"❌ Failed to process page {task['page_number']}: {e}")
            
            vectorization_time = time.time() - vectorization_start
            upload_time = vectorization_time  # Combined time for simplicity
            
            with self.lock:
                print(f"📤 Uploaded {uploaded_pages}/{len(vectorization_tasks)} pages to search index in {vectorization_time:.2f}s")
        
        # Report missing fields
        missing_fields = [field for field, value in lease_info.items() 
                         if value is None or not str(value).strip()]
        if missing_fields:
            with self.lock:
                print(f"⚠️  Missing or empty fields: {', '.join(missing_fields)}")
        
        # Report successfully extracted fields
        extracted_fields = [field for field, value in lease_info.items() 
                           if value is not None and str(value).strip()]
        if extracted_fields:
            with self.lock:
                print(f"✅ Successfully extracted fields: {', '.join(extracted_fields)}")
        else:
            with self.lock:
                print("❌ No lease information could be extracted from this document")
        
        total_time = time.time() - start_time
        
        # Add metadata
        lease_info['_metadata'] = {
            'source_file': str(pdf_path),
            'processing_method': extraction_method,
            'ocr_method': 'azure',
            'text_length': len(all_text),
            'pages_processed': len(result.pages) if result else 0,
            'pages_uploaded': uploaded_pages,
            'client_name': client_name,
            'processed_at': datetime.now().isoformat(),
            'ocr_time_seconds': round(ocr_time, 2),
            'extraction_time_seconds': round(extraction_time, 2),
            'vectorization_time_seconds': round(vectorization_time, 2),
            'upload_time_seconds': round(upload_time, 2),
            'total_time_seconds': round(total_time, 2)
        }
        
        with self.lock:
            if vectorization_time > 0:
                print(f"⏱️  Total processing time: {total_time:.2f}s (OCR: {ocr_time:.2f}s, Extraction: {extraction_time:.2f}s, Vectorization: {vectorization_time:.2f}s)")
            else:
                print(f"⏱️  Total processing time: {total_time:.2f}s (OCR: {ocr_time:.2f}s, Extraction: {extraction_time:.2f}s)")
        
        return lease_info
    
    def process_single_pdf_with_index(self, pdf_path: Path, file_index: int, extraction_method: str, client_name: str = None) -> Dict[str, Any]:
        """
        Process a single PDF file with thread-safe operations.
        
        Args:
            pdf_path: Path to PDF file
            file_index: Index of the file in the batch
            extraction_method: Extraction method to use
            client_name: Name of the client (from folder structure)
            
        Returns:
            Dictionary with extracted lease information
        """
        try:
            lease_info = self.process_pdf(pdf_path, extraction_method=extraction_method, client_name=client_name)
            lease_info['_metadata']['file_index'] = file_index
            return lease_info
        except Exception as e:
            with self.lock:
                print(f"❌ Failed to process {pdf_path.name}: {e}")
            # Add error record
            error_record = {field: None for field in self.lease_fields.keys()}
            error_record['_metadata'] = {
                'source_file': str(pdf_path),
                'error': str(e),
                'file_index': file_index,
                'processed_at': datetime.now().isoformat()
            }
            return error_record

    def process_directory(self, directory_path: Path, output_path: Optional[Path] = None, 
                         extraction_method: str = 'regex') -> List[Dict[str, Any]]:
        """
        Process all PDF files in a directory structure organized by client folders.
        
        Args:
            directory_path: Path to directory containing client subdirectories
            output_path: Path to save results (optional)
            extraction_method: Extraction method to use ('regex' or 'ai')
            
        Returns:
            List of extracted lease information dictionaries
        """
        # Collect all PDF files from client subdirectories
        pdf_files_with_clients = []
        
        # Check if directory has client subdirectories
        client_dirs = [d for d in directory_path.iterdir() if d.is_dir()]
        
        if client_dirs:
            # Process client-organized structure
            for client_dir in client_dirs:
                client_name = client_dir.name
                pdf_files = list(client_dir.glob('*.pdf'))
                
                for pdf_file in pdf_files:
                    pdf_files_with_clients.append((pdf_file, client_name))
                
                if pdf_files:
                    print(f"📁 Found {len(pdf_files)} PDF files for client: {client_name}")
        else:
            # Fallback to processing PDFs directly in the main directory
            pdf_files = list(directory_path.glob('*.pdf'))
            pdf_files_with_clients = [(pdf_file, None) for pdf_file in pdf_files]
            
            if pdf_files:
                print(f"📁 Found {len(pdf_files)} PDF files in main directory (no client folders)")
        
        if not pdf_files_with_clients:
            print(f"❌ No PDF files found in {directory_path} or its client subdirectories")
            return []
        
        print(f"📁 Total: {len(pdf_files_with_clients)} PDF files to process")
        print(f"⚡ Using {self.max_workers} worker threads for parallel processing")
        
        start_time = time.time()
        results = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_pdf = {
                executor.submit(self.process_single_pdf_with_index, pdf_path, i, extraction_method, client_name): (pdf_path, i, client_name)
                for i, (pdf_path, client_name) in enumerate(pdf_files_with_clients, 1)
            }
            
            # Process completed tasks
            for future in as_completed(future_to_pdf):
                pdf_path, file_index, client_name = future_to_pdf[future]
                try:
                    result = future.result()
                    results.append(result)
                    with self.lock:
                        client_info = f" (client: {client_name})" if client_name else ""
                        print(f"✅ Completed {len(results)}/{len(pdf_files_with_clients)}: {pdf_path.name}{client_info}")
                except Exception as e:
                    with self.lock:
                        print(f"❌ Error processing {pdf_path.name}: {e}")
        
        # Sort results by file index to maintain order
        results.sort(key=lambda x: x['_metadata'].get('file_index', 0))
        
        total_time = time.time() - start_time
        avg_time = total_time / len(pdf_files_with_clients) if pdf_files_with_clients else 0
        
        # Calculate vectorization stats
        total_pages_uploaded = sum(r['_metadata'].get('pages_uploaded', 0) for r in results)
        total_pages_processed = sum(r['_metadata'].get('pages_processed', 0) for r in results)
        
        print(f"\n📊 Processing Summary:")
        print(f"   Total files: {len(pdf_files_with_clients)}")
        print(f"   Total pages processed: {total_pages_processed}")
        if total_pages_uploaded > 0:
            print(f"   Total pages uploaded to search: {total_pages_uploaded}")
        print(f"   Total time: {total_time:.2f}s")
        print(f"   Average time per file: {avg_time:.2f}s")
        print(f"   Worker threads: {self.max_workers}")
        
        # Save results if output path provided
        if output_path:
            self.save_results(results, output_path)
        
        return results
    
    def save_results(self, results: List[Dict[str, Any]], output_path: Path):
        """
        Save extraction results to file.
        
        Args:
            results: List of lease information dictionaries
            output_path: Path to save results
        """
        if output_path.suffix.lower() == '.json':
            # Save as JSON
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"💾 Results saved to JSON: {output_path}")
            
        elif output_path.suffix.lower() in ['.xlsx', '.xls']:
            # Save as Excel with multiple sheets
            if not results:
                print("❌ No results to save")
                return
            
            # Prepare data for the main sheet
            main_data = []
            timing_data = []
            
            for result in results:
                # Main lease information
                row = {field: result.get(field) for field in self.lease_fields.keys()}
                if '_metadata' in result:
                    row.update({
                        'source_file': result['_metadata'].get('source_file', ''),
                        'processing_method': result['_metadata'].get('processing_method', ''),
                        'ocr_method': result['_metadata'].get('ocr_method', ''),
                        'processed_at': result['_metadata'].get('processed_at', ''),
                        'file_index': result['_metadata'].get('file_index', '')
                    })
                main_data.append(row)
                
                # Timing information
                if '_metadata' in result:
                    timing_row = {
                        'file_index': result['_metadata'].get('file_index', ''),
                        'source_file': result['_metadata'].get('source_file', ''),
                        'ocr_time_seconds': result['_metadata'].get('ocr_time_seconds', ''),
                        'extraction_time_seconds': result['_metadata'].get('extraction_time_seconds', ''),
                        'total_time_seconds': result['_metadata'].get('total_time_seconds', ''),
                        'text_length': result['_metadata'].get('text_length', ''),
                        'processed_at': result['_metadata'].get('processed_at', '')
                    }
                    timing_data.append(timing_row)
            
            # Create DataFrames
            main_df = pd.DataFrame(main_data)
            timing_df = pd.DataFrame(timing_data)
            
            # Create Excel writer
            with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
                # Main lease information sheet
                main_df.to_excel(writer, sheet_name='Lease Information', index=False)
                
                # Timing analysis sheet
                if timing_data:
                    timing_df.to_excel(writer, sheet_name='Processing Times', index=False)
                
                # Summary sheet
                summary_data = {
                    'Metric': [
                        'Total Files Processed',
                        'Average OCR Time (seconds)',
                        'Average Extraction Time (seconds)',
                        'Average Total Time (seconds)',
                        'Total Processing Time (seconds)',
                        'Files with Errors'
                    ],
                    'Value': [
                        len(results),
                        timing_df['ocr_time_seconds'].mean() if not timing_df.empty else 0,
                        timing_df['extraction_time_seconds'].mean() if not timing_df.empty else 0,
                        timing_df['total_time_seconds'].mean() if not timing_df.empty else 0,
                        timing_df['total_time_seconds'].sum() if not timing_df.empty else 0,
                        sum(1 for r in results if r.get('_metadata', {}).get('error'))
                    ]
                }
                summary_df = pd.DataFrame(summary_data)
                summary_df.to_excel(writer, sheet_name='Summary', index=False)
            
            print(f"💾 Results saved to Excel: {output_path}")
            print(f"📊 Created 3 sheets: Lease Information, Processing Times, Summary")
            
        elif output_path.suffix.lower() == '.csv':
            # Save as CSV (legacy support)
            if not results:
                print("❌ No results to save")
                return
            
            # Get all field names (excluding metadata)
            fieldnames = [field for field in self.lease_fields.keys()]
            fieldnames.extend(['source_file', 'processing_method', 'ocr_method', 'processed_at'])
            
            with open(output_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                
                for result in results:
                    # Flatten the result
                    row = {field: result.get(field) for field in self.lease_fields.keys()}
                    if '_metadata' in result:
                        row.update({
                            'source_file': result['_metadata'].get('source_file'),
                            'processing_method': result['_metadata'].get('processing_method'),
                            'ocr_method': result['_metadata'].get('ocr_method'),
                            'processed_at': result['_metadata'].get('processed_at')
                        })
                    writer.writerow(row)
            
            print(f"💾 Results saved to CSV: {output_path}")
        
        else:
            # Save as text
            with open(output_path, 'w', encoding='utf-8') as f:
                for i, result in enumerate(results, 1):
                    f.write(f"=== Document {i} ===\n")
                    f.write(f"Source: {result.get('_metadata', {}).get('source_file', 'Unknown')}\n")
                    f.write(f"Processing method: {result.get('_metadata', {}).get('processing_method', 'Unknown')}\n")
                    f.write(f"OCR method: {result.get('_metadata', {}).get('ocr_method', 'Unknown')}\n")
                    f.write(f"Processed at: {result.get('_metadata', {}).get('processed_at', 'Unknown')}\n\n")
                    
                    for field, description in self.lease_fields.items():
                        value = result.get(field, 'Not found')
                        f.write(f"{description}: {value}\n")
                    
                    f.write("\n" + "="*80 + "\n\n")
            
            print(f"💾 Results saved to text: {output_path}")


def main():
    """Command line interface"""
    parser = argparse.ArgumentParser(description='Extract structured lease information from PDF documents using Azure Form Recognizer')
    parser.add_argument('input', nargs='?', default='leases', help='Path to PDF file or directory containing PDF files (default: leases)')
    parser.add_argument('-o', '--output', default='lease_results.xlsx', help='Output file path (XLSX, JSON, CSV, or TXT) (default: lease_results.xlsx)')
    parser.add_argument('-m', '--method', choices=['regex', 'ai'], 
                       default='regex', help='Extraction method to use (default: regex)')
    parser.add_argument('--azure-endpoint', help='Azure Form Recognizer endpoint (or set AZURE_FORM_RECOGNIZER_ENDPOINT env var)')
    parser.add_argument('--azure-key', help='Azure Form Recognizer key (or set AZURE_FORM_RECOGNIZER_KEY env var)')
    parser.add_argument('--openai-key', help='OpenAI API key (or set OPENAI_API_KEY env var)')
    parser.add_argument('--search-endpoint', help='Azure AI Search endpoint (or set AZURE_SEARCH_ENDPOINT env var)')
    parser.add_argument('--search-key', help='Azure AI Search key (or set AZURE_SEARCH_KEY env var)')
    parser.add_argument('--search-index', help='Azure AI Search index name (or set AZURE_SEARCH_INDEX_NAME env var, default: electricity-documents)')
    parser.add_argument('-w', '--workers', type=int, default=4, help='Number of worker threads for parallel processing (default: 4)')
    parser.add_argument('--ssl-verify', action='store_true', help='Verify SSL certificates (default: False)')
    
    args = parser.parse_args()
    
    # Check input path
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"❌ Input path not found: {input_path}")
        if args.input == 'leases':
            print("💡 Create a 'leases' folder and place PDF files there, or specify a different path.")
        sys.exit(1)
    
    # Initialize processor
    try:
        processor = LeaseDocumentProcessor(
            azure_endpoint=args.azure_endpoint,
            azure_key=args.azure_key,
            openai_api_key=args.openai_key,
            azure_search_endpoint=args.search_endpoint,
            azure_search_key=args.search_key,
            search_index_name=args.search_index,
            max_workers=args.workers,
            ssl_verify=args.ssl_verify
        )
    except Exception as e:
        print(f"❌ Failed to initialize processor: {e}")
        sys.exit(1)
    
    # Set output path
    output_path = Path(args.output)
    
    try:
        if input_path.is_file():
            # Process single file - extract client name from parent directory if applicable
            client_name = None
            if input_path.parent.name != 'leases':  # If not in root leases folder
                client_name = input_path.parent.name
            
            result = processor.process_pdf(
                input_path,
                extraction_method=args.method,
                client_name=client_name
            )
            
            # Save results to file
            processor.save_results([result], output_path)
            
            # Also print results to console
            print("\n🎉 Extraction Results:")
            print("=" * 50)
            for field, description in processor.lease_fields.items():
                value = result.get(field, 'Not found')
                print(f"{description}: {value}")
        
        else:
            # Process directory
            results = processor.process_directory(
                input_path,
                output_path=output_path,
                extraction_method=args.method
            )
            
            # Print summary to console
            print(f"\n🎉 Processed {len(results)} documents")
            print(f"📄 Results saved to: {output_path}")
        
        print(f"\n✅ Processing complete!")
        print(f"📄 Results saved to: {output_path}")
        
    except Exception as e:
        print(f"❌ Processing failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()