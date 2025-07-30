"""
Embedding Tool for creating vector embeddings from text using OpenAI
"""

from typing import List, Any
from langchain.tools import BaseTool
from pydantic import Field


class EmbeddingTool(BaseTool):
    """Tool for creating vector embeddings from text using OpenAI"""
    
    name: str = "embedding_tool"
    description: str = "Creates vector embeddings from text using OpenAI embeddings. Input should be the text to embed."
    openai_client: Any = Field(default=None, exclude=True)
    embedding_model: str = Field(default="text-embedding-ada-002", exclude=True)
    
    def __init__(self, openai_client, embedding_model: str = "text-embedding-ada-002", **kwargs):
        super().__init__(**kwargs)
        object.__setattr__(self, 'openai_client', openai_client)
        object.__setattr__(self, 'embedding_model', embedding_model)
    
    def _run(self, text: str) -> str:
        """Generate embedding for text"""
        try:
            if not text:
                return "Error: No text provided for embedding"
            
            print(f"   🔤 Generating embedding for text ({len(text)} chars)...")
            
            # Limit text length to avoid token limits
            text_to_embed = text[:8000] if len(text) > 8000 else text
            
            response = self.openai_client.embeddings.create(
                model=self.embedding_model,
                input=text_to_embed
            )
            
            embedding = response.data[0].embedding
            print(f"   ✅ Embedding generated: {len(embedding)} dimensions")
            
            # Return embedding as JSON string for further processing
            import json
            return json.dumps(embedding)
            
        except Exception as e:
            error_msg = f"Error generating embedding: {str(e)}"
            print(f"   ❌ {error_msg}")
            return error_msg