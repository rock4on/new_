#!/usr/bin/env python3
"""
Test embedding generation specifically
"""

import sys
from pathlib import Path

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

from config import get_config
import openai


def test_embedding_generation():
    """Test embedding generation directly"""
    print("🔤 Testing Embedding Generation...")
    
    try:
        config = get_config()
        print(f"   Model: {config.OPENAI_EMBEDDING_MODEL}")
        
        # Initialize OpenAI client
        openai_kwargs = {"api_key": config.OPENAI_API_KEY}
        if config.OPENAI_BASE_URL:
            openai_kwargs["base_url"] = config.OPENAI_BASE_URL
            print(f"   Base URL: {config.OPENAI_BASE_URL}")
        
        client = openai.OpenAI(**openai_kwargs)
        
        # Test embedding generation
        test_texts = [
            "Hello world",
            "This is a test document about lease agreements",
            "office building lease Chicago downtown"
        ]
        
        for i, text in enumerate(test_texts, 1):
            print(f"\n   Test {i}: '{text}'")
            try:
                response = client.embeddings.create(
                    model=config.OPENAI_EMBEDDING_MODEL,
                    input=text
                )
                embedding = response.data[0].embedding
                print(f"   ✅ Success: {len(embedding)} dimensions")
                print(f"   Sample values: {embedding[:3]}")
                
            except Exception as e:
                print(f"   ❌ Failed: {e}")
                return False
        
        print(f"\n✅ All embedding tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Embedding test setup failed: {e}")
        return False


if __name__ == "__main__":
    success = test_embedding_generation()
    if not success:
        print(f"\n💡 Troubleshooting:")
        print(f"   - Check your OpenAI API key")
        print(f"   - Verify embedding model name")
        print(f"   - Check base URL if using Azure OpenAI")
        print(f"   - Ensure you have access to the embedding model")
        sys.exit(1)
    else:
        print(f"\n🎉 Embedding generation is working correctly!")