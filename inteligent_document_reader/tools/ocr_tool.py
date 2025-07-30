"""
OCR Tool for text extraction from PDF documents using Azure Form Recognizer
"""

from pathlib import Path
from typing import Any
from langchain.tools import BaseTool
from pydantic import Field

from azure.ai.formrecognizer import DocumentAnalysisClient
from azure.core.credentials import AzureKeyCredential
from azure.core.pipeline.transport import RequestsTransport


class OCRTool(BaseTool):
    """Tool for extracting text from PDF documents using Azure Form Recognizer OCR"""
    
    name: str = "ocr_extractor"
    description: str = "Extracts text from PDF documents using Azure Form Recognizer OCR. Input should be the file path to a PDF document."
    azure_client: Any = Field(default=None, exclude=True)
    
    def __init__(self, azure_endpoint: str, azure_key: str, **kwargs):
        super().__init__(**kwargs)
        try:
            print(f"   🔍 Initializing Azure Form Recognizer OCR...")
            
            # Disable SSL verification for Azure OCR transport
            transport = RequestsTransport(connection_verify=False)
            azure_client = DocumentAnalysisClient(
                endpoint=azure_endpoint,
                credential=AzureKeyCredential(azure_key),
                transport=transport
            )
            
            object.__setattr__(self, 'azure_client', azure_client)
            print(f"   ✅ Azure Form Recognizer OCR tool initialized")
            
        except Exception as e:
            print(f"   ❌ Azure Form Recognizer initialization failed: {e}")
            raise ConnectionError(f"Azure Form Recognizer initialization failed: {e}")
    
    def _run(self, file_path: str) -> str:
        """Extract text from PDF file"""
        try:
            pdf_path = Path(file_path)
            if not pdf_path.exists():
                return f"Error: File not found at {file_path}"
            
            print(f"   📄 Extracting text from: {pdf_path.name}")
            
            with open(pdf_path, "rb") as f:
                poller = self.azure_client.begin_analyze_document("prebuilt-document", document=f)
                result = poller.result()
            
            # Extract all text
            full_text = ""
            for page in result.pages:
                for line in page.lines:
                    full_text += line.content + "\n"
            
            print(f"   ✅ Extracted {len(full_text)} characters from {len(result.pages)} pages")
            return full_text
            
        except Exception as e:
            error_msg = f"Error extracting text: {str(e)}"
            print(f"   ❌ {error_msg}")
            return error_msg