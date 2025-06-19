#!/usr/bin/env python3
"""
LLM Document Filter Service

Scans downloaded PDFs, extracts text, and uses LLM to determine relevance
based on user-provided criteria. Keeps only relevant documents.
"""

import os
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any
import PyPDF2
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field


class RelevanceAnalysis(BaseModel):
    """Pydantic model for structured relevance analysis output"""
    relevant: bool = Field(description="Whether the document is relevant")
    confidence: float = Field(description="Confidence score between 0.0 and 1.0")
    reasoning: str = Field(description="Brief explanation of the decision")
    key_topics: List[str] = Field(description="Main topics found in the document")


# Configuration - Set your API key and endpoint here
DEFAULT_API_KEY = "your-api-key-here"
DEFAULT_BASE_URL = "https://api.openai.com/v1"  # Change to your custom endpoint


class DocumentProcessor:
    def __init__(self, downloads_dir: str = "downloads", api_key: str = None, base_url: str = None):
        self.downloads_dir = Path(downloads_dir)
        self.relevant_dir = self.downloads_dir / "relevant"
        self.irrelevant_dir = self.downloads_dir / "irrelevant"
        
        # Create directories
        self.relevant_dir.mkdir(exist_ok=True)
        self.irrelevant_dir.mkdir(exist_ok=True)
        
        # Setup LangChain ChatOpenAI with fallback priority:
        # 1. Command line arguments
        # 2. Environment variables  
        # 3. Default constants above
        self.llm = ChatOpenAI(
            api_key=api_key or os.getenv("OPENAI_API_KEY") or DEFAULT_API_KEY,
            base_url=base_url or os.getenv("OPENAI_BASE_URL") or DEFAULT_BASE_URL,
            temperature=0.1
        )
        
        # Setup output parser
        self.parser = JsonOutputParser(pydantic_object=RelevanceAnalysis)
        
        # Create prompt template
        self.prompt = ChatPromptTemplate.from_template("""
You are a document relevance analyzer. Analyze the following document excerpt and determine if it's relevant to the given criteria.

RELEVANCE CRITERIA:
{criteria}

DOCUMENT EXCERPT:
{document_text}

Based on the criteria, analyze this document's relevance.

{format_instructions}
""")
        
        # Create the chain
        self.chain = self.prompt | self.llm | self.parser
    
    def extract_pdf_text(self, pdf_path: Path, max_pages: int = 5) -> str:
        """Extract text from first few pages of PDF"""
        try:
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text = ""
                
                # Extract text from first max_pages pages
                for i, page in enumerate(reader.pages[:max_pages]):
                    text += page.extract_text() + "\n"
                
                return text.strip()
        except Exception as e:
            print(f"Error extracting text from {pdf_path}: {e}")
            return ""
    
    def check_relevance(self, document_text: str, relevance_criteria: str, model: str = "gpt-4o-mini") -> Dict[str, Any]:
        """Use LangChain LLM to determine if document is relevant"""
        
        # Truncate text if too long (keep first 4000 chars for context)
        if len(document_text) > 4000:
            document_text = document_text[:4000] + "...[truncated]"
        
        # Update model if different from default
        if model != "gpt-4o-mini":
            self.llm.model_name = model
        
        try:
            # Invoke the chain with the document text and criteria
            result = self.chain.invoke({
                "criteria": relevance_criteria,
                "document_text": document_text,
                "format_instructions": self.parser.get_format_instructions()
            })
            
            # Convert to dict format expected by the rest of the code
            return {
                "relevant": result["relevant"],
                "confidence": result["confidence"], 
                "reasoning": result["reasoning"],
                "key_topics": result["key_topics"]
            }
            
        except Exception as e:
            print(f"Error analyzing document relevance: {e}")
            return {
                "relevant": False,
                "confidence": 0.0,
                "reasoning": f"Error in analysis: {str(e)}",
                "key_topics": []
            }
    
    def process_documents(self, relevance_criteria: str, model: str = "gpt-4o-mini", 
                         confidence_threshold: float = 0.7):
        """Process all PDFs in downloads directory"""
        
        pdf_files = list(self.downloads_dir.glob("*.pdf"))
        if not pdf_files:
            print("No PDF files found in downloads directory")
            return
        
        results = []
        relevant_count = 0
        
        print(f"Processing {len(pdf_files)} documents...")
        print(f"Relevance criteria: {relevance_criteria}")
        print(f"Confidence threshold: {confidence_threshold}")
        print("-" * 80)
        
        for pdf_file in pdf_files:
            print(f"Processing: {pdf_file.name}")
            
            # Extract text
            text = self.extract_pdf_text(pdf_file)
            if not text:
                print(f"  ❌ Could not extract text, moving to irrelevant")
                pdf_file.rename(self.irrelevant_dir / pdf_file.name)
                continue
            
            # Check relevance
            analysis = self.check_relevance(text, relevance_criteria, model)
            
            # Decision based on relevance and confidence
            is_relevant = (analysis["relevant"] and 
                          analysis["confidence"] >= confidence_threshold)
            
            if is_relevant:
                # Move to relevant folder
                destination = self.relevant_dir / pdf_file.name
                pdf_file.rename(destination)
                relevant_count += 1
                status = "✅ RELEVANT"
            else:
                # Move to irrelevant folder  
                destination = self.irrelevant_dir / pdf_file.name
                pdf_file.rename(destination)
                status = "❌ IRRELEVANT"
            
            print(f"  {status}")
            print(f"  Confidence: {analysis['confidence']:.2f}")
            print(f"  Reasoning: {analysis['reasoning']}")
            print(f"  Topics: {', '.join(analysis['key_topics'])}")
            print()
            
            # Store results
            results.append({
                "filename": pdf_file.name,
                "relevant": is_relevant,
                "analysis": analysis,
                "moved_to": str(destination)
            })
        
        # Save results
        results_file = self.downloads_dir / "relevance_analysis.json"
        with open(results_file, 'w') as f:
            json.dump({
                "criteria": relevance_criteria,
                "model": model,
                "confidence_threshold": confidence_threshold,
                "total_documents": len(pdf_files),
                "relevant_documents": relevant_count,
                "results": results
            }, f, indent=2)
        
        print(f"✅ Processing complete!")
        print(f"📊 Results: {relevant_count}/{len(pdf_files)} documents marked as relevant")
        print(f"📁 Relevant documents: {self.relevant_dir}")
        print(f"📁 Irrelevant documents: {self.irrelevant_dir}")
        print(f"📄 Full analysis saved to: {results_file}")


def main():
    parser = argparse.ArgumentParser(description="Filter downloaded PDFs using LLM")
    parser.add_argument("criteria", help="Relevance criteria (what you're looking for)")
    parser.add_argument("--downloads-dir", default="downloads", help="Directory containing PDFs")
    parser.add_argument("--model", default="gpt-4o-mini", help="LLM model to use")
    parser.add_argument("--confidence", type=float, default=0.7, help="Confidence threshold (0.0-1.0)")
    parser.add_argument("--api-key", help="OpenAI API key (or set OPENAI_API_KEY env var)")
    parser.add_argument("--base-url", help="Custom API base URL")
    
    args = parser.parse_args()
    
    processor = DocumentProcessor(
        downloads_dir=args.downloads_dir,
        api_key=args.api_key,
        base_url=args.base_url
    )
    
    processor.process_documents(
        relevance_criteria=args.criteria,
        model=args.model,
        confidence_threshold=args.confidence
    )


if __name__ == "__main__":
    main()