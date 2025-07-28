#!/usr/bin/env python3
"""
Separate Batch Ingestion Tools for Leases, Electric, and Natural Gas Documents
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Optional, Any

from langchain.tools import BaseTool
from pydantic import BaseModel, Field


class BatchIngestLeasesTool(BaseTool):
    """Tool for batch ingesting lease PDF documents from the /leases folder"""
    
    name: str = "batch_ingest_leases"
    description: str = "Automatically discovers and ingests all lease PDF documents from the /leases folder. Uses OCR to extract text and stores in vector database. Input can be empty '{}' or JSON with 'client_name' field."
    agent_instance: Any = Field(default=None, exclude=True)
    
    def __init__(self, agent_instance, **kwargs):
        super().__init__(**kwargs)
        object.__setattr__(self, 'agent_instance', agent_instance)
    
    def _run(self, input_data: str = "{}") -> str:
        """Run batch ingestion of all lease PDFs in /leases folder"""
        try:
            print("   📁 Starting batch ingestion of lease documents from /leases folder...")
            
            # Parse input for optional client name
            client_name = None
            if input_data.strip() and input_data.strip() != "{}":
                try:
                    data = json.loads(input_data)
                    client_name = data.get('client_name')
                except json.JSONDecodeError:
                    client_name = input_data.strip()
            
            # Set specific folder path for leases
            leases_folder = Path("./leases")
            
            # Call the agent's batch_ingest_pdfs method with specific folder
            result = self.agent_instance.batch_ingest_pdfs(
                folder_path=str(leases_folder), 
                client_name=client_name
            )
            
            # Format response for the agent
            if result['status'] == 'completed':
                return f"✅ Successfully ingested all lease PDFs! Processed {result['successful']} files from {result['folder_path']}. All lease documents are now searchable in the vector database."
            elif result['status'] == 'partial_success':
                return f"⚠️ Partially successful lease ingestion: {result['successful']} files succeeded, {result['failed']} failed from {result['folder_path']}. Successfully processed lease documents are searchable."
            elif result['status'] == 'warning':
                return f"⚠️ No lease PDF files found in /leases folder. Please place lease PDF files in the /leases folder or its client subdirectories."
            elif result['status'] == 'failed':
                return f"❌ Lease ingestion failed: all {result['failed']} files failed to process from {result['folder_path']}. Check file permissions and formats."
            else:
                return f"❌ Lease ingestion error: {result.get('error', 'Unknown error occurred during lease processing')}"
                
        except Exception as e:
            return f"❌ Error during lease ingestion: {str(e)}"


class BatchIngestElectricTool(BaseTool):
    """Tool for batch ingesting electric PDF documents from the /electric folder"""
    
    name: str = "batch_ingest_electric"
    description: str = "Automatically discovers and ingests all electric/electricity PDF documents from the /electric folder. Uses OCR to extract text and stores in vector database. Input can be empty '{}' or JSON with 'client_name' field."
    agent_instance: Any = Field(default=None, exclude=True)
    
    def __init__(self, agent_instance, **kwargs):
        super().__init__(**kwargs)
        object.__setattr__(self, 'agent_instance', agent_instance)
    
    def _batch_ingest_utilities_pdfs(self, folder_path: str, client_name: str = None, utility_type: str = "auto") -> Dict[str, Any]:
        """Batch process utilities PDFs using process_utilities_document instead of process_document"""
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
        
        print(f"📁 Found {len(pdf_files)} utilities PDF files to process in {folder_path}")
        
        results = []
        successful = 0
        failed = 0
        
        for pdf_file in pdf_files:
            print(f"\n📄 Processing utilities document: {pdf_file.name}")
            
            # Determine client name from folder structure if not provided
            effective_client_name = client_name
            if not effective_client_name:
                if pdf_file.parent.name != folder.name:
                    effective_client_name = pdf_file.parent.name
                else:
                    effective_client_name = "Unknown"
            
            try:
                # Use process_utilities_document instead of process_document!
                result = self.agent_instance.process_utilities_document(
                    file_path=str(pdf_file),
                    utility_type=utility_type,
                    client_name=effective_client_name
                )
                
                results.append(result)
                
                if result["status"] == "success":
                    successful += 1
                    print(f"   ✅ Successfully processed utilities document {pdf_file.name}")
                else:
                    failed += 1
                    print(f"   ❌ Failed to process utilities document {pdf_file.name}: {result.get('error', 'Unknown error')}")
                    
            except Exception as e:
                failed += 1
                error_result = {
                    "status": "error",
                    "file_path": str(pdf_file),
                    "client_name": effective_client_name,
                    "utility_type": utility_type,
                    "error": str(e)
                }
                results.append(error_result)
                print(f"   ❌ Exception processing utilities document {pdf_file.name}: {e}")
        
        return {
            "status": "completed" if failed == 0 else "partial_success" if successful > 0 else "failed",
            "folder_path": str(folder_path),
            "total_files": len(pdf_files),
            "successful": successful,
            "failed": failed,
            "results": results
        }
    
    def _run(self, input_data: str = "{}") -> str:
        """Run batch ingestion of all electric PDFs in /electric folder"""
        try:
            print("   ⚡ Starting batch ingestion of electric documents from /electric folder...")
            
            # Parse input for optional client name
            client_name = None
            if input_data.strip() and input_data.strip() != "{}":
                try:
                    data = json.loads(input_data)
                    client_name = data.get('client_name')
                except json.JSONDecodeError:
                    client_name = input_data.strip()
            
            # Set specific folder path for electric documents
            electric_folder = Path("./electric")
            
            # Use custom utilities batch processing instead of generic batch_ingest_pdfs
            result = self._batch_ingest_utilities_pdfs(
                folder_path=str(electric_folder), 
                client_name=client_name,
                utility_type="electricity"
            )
            
            # Format response for the agent
            if result['status'] == 'completed':
                return f"✅ Successfully ingested all electric PDFs! Processed {result['successful']} files from {result['folder_path']}. All electric documents are now searchable in the vector database."
            elif result['status'] == 'partial_success':
                return f"⚠️ Partially successful electric ingestion: {result['successful']} files succeeded, {result['failed']} failed from {result['folder_path']}. Successfully processed electric documents are searchable."
            elif result['status'] == 'warning':
                return f"⚠️ No electric PDF files found in /electric folder. Please place electric/electricity PDF files in the /electric folder or its client subdirectories."
            elif result['status'] == 'failed':
                return f"❌ Electric ingestion failed: all {result['failed']} files failed to process from {result['folder_path']}. Check file permissions and formats."
            else:
                return f"❌ Electric ingestion error: {result.get('error', 'Unknown error occurred during electric processing')}"
                
        except Exception as e:
            return f"❌ Error during electric ingestion: {str(e)}"


class BatchIngestNaturalGasTool(BaseTool):
    """Tool for batch ingesting natural gas PDF documents from the /natural_gas folder"""
    
    name: str = "batch_ingest_natural_gas"
    description: str = "Automatically discovers and ingests all natural gas PDF documents from the /natural_gas folder. Uses OCR to extract text and stores in vector database. Input can be empty '{}' or JSON with 'client_name' field."
    agent_instance: Any = Field(default=None, exclude=True)
    
    def __init__(self, agent_instance, **kwargs):
        super().__init__(**kwargs)
        object.__setattr__(self, 'agent_instance', agent_instance)
    
    def _batch_ingest_utilities_pdfs(self, folder_path: str, client_name: str = None, utility_type: str = "auto") -> Dict[str, Any]:
        """Batch process utilities PDFs using process_utilities_document instead of process_document"""
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
        
        print(f"📁 Found {len(pdf_files)} utilities PDF files to process in {folder_path}")
        
        results = []
        successful = 0
        failed = 0
        
        for pdf_file in pdf_files:
            print(f"\n📄 Processing utilities document: {pdf_file.name}")
            
            # Determine client name from folder structure if not provided
            effective_client_name = client_name
            if not effective_client_name:
                if pdf_file.parent.name != folder.name:
                    effective_client_name = pdf_file.parent.name
                else:
                    effective_client_name = "Unknown"
            
            try:
                # Use process_utilities_document instead of process_document!
                result = self.agent_instance.process_utilities_document(
                    file_path=str(pdf_file),
                    utility_type=utility_type,
                    client_name=effective_client_name
                )
                
                results.append(result)
                
                if result["status"] == "success":
                    successful += 1
                    print(f"   ✅ Successfully processed utilities document {pdf_file.name}")
                else:
                    failed += 1
                    print(f"   ❌ Failed to process utilities document {pdf_file.name}: {result.get('error', 'Unknown error')}")
                    
            except Exception as e:
                failed += 1
                error_result = {
                    "status": "error",
                    "file_path": str(pdf_file),
                    "client_name": effective_client_name,
                    "utility_type": utility_type,
                    "error": str(e)
                }
                results.append(error_result)
                print(f"   ❌ Exception processing utilities document {pdf_file.name}: {e}")
        
        return {
            "status": "completed" if failed == 0 else "partial_success" if successful > 0 else "failed",
            "folder_path": str(folder_path),
            "total_files": len(pdf_files),
            "successful": successful,
            "failed": failed,
            "results": results
        }
    
    def _run(self, input_data: str = "{}") -> str:
        """Run batch ingestion of all natural gas PDFs in /natural_gas folder"""
        try:
            print("   🔥 Starting batch ingestion of natural gas documents from /natural_gas folder...")
            
            # Parse input for optional client name
            client_name = None
            if input_data.strip() and input_data.strip() != "{}":
                try:
                    data = json.loads(input_data)
                    client_name = data.get('client_name')
                except json.JSONDecodeError:
                    client_name = input_data.strip()
            
            # Set specific folder path for natural gas documents
            natural_gas_folder = Path("./natural_gas")
            
            # Use custom utilities batch processing instead of generic batch_ingest_pdfs
            result = self._batch_ingest_utilities_pdfs(
                folder_path=str(natural_gas_folder), 
                client_name=client_name,
                utility_type="natural_gas"
            )
            
            # Format response for the agent
            if result['status'] == 'completed':
                return f"✅ Successfully ingested all natural gas PDFs! Processed {result['successful']} files from {result['folder_path']}. All natural gas documents are now searchable in the vector database."
            elif result['status'] == 'partial_success':
                return f"⚠️ Partially successful natural gas ingestion: {result['successful']} files succeeded, {result['failed']} failed from {result['folder_path']}. Successfully processed natural gas documents are searchable."
            elif result['status'] == 'warning':
                return f"⚠️ No natural gas PDF files found in /natural_gas folder. Please place natural gas PDF files in the /natural_gas folder or its client subdirectories."
            elif result['status'] == 'failed':
                return f"❌ Natural gas ingestion failed: all {result['failed']} files failed to process from {result['folder_path']}. Check file permissions and formats."
            else:
                return f"❌ Natural gas ingestion error: {result.get('error', 'Unknown error occurred during natural gas processing')}"
                
        except Exception as e:
            return f"❌ Error during natural gas ingestion: {str(e)}"


def create_batch_ingestion_tools(agent_instance) -> List:
    """
    Create all three batch ingestion tools
    
    Args:
        agent_instance: The main agent instance
        
    Returns:
        List of batch ingestion tools
    """
    return [
        BatchIngestLeasesTool(agent_instance=agent_instance),
        BatchIngestElectricTool(agent_instance=agent_instance),
        BatchIngestNaturalGasTool(agent_instance=agent_instance)
    ]