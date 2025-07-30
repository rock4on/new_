"""
Configuration constants for Inteligent Document Reader
"""


class Config:
    """Configuration constants class"""
    
    # Folder paths
    LEASES_FOLDER = './leases'
    ELECTRICITY_FOLDER = './electric'
    NATURAL_GAS_FOLDER = './natural_gas'
    
    # OpenAI settings
    OPENAI_MODEL = 'gpt-4o-mini'
    OPENAI_TEMPERATURE = 0.1
    OPENAI_MAX_TOKENS = 1000
    OPENAI_EMBEDDING_MODEL = 'text-embedding-ada-002'
    
    # Azure AI Search
    AZURE_SEARCH_INDEX_NAME = 'lease-documents'
    AZURE_SEARCH_UTILITIES_INDEX_NAME = 'lease-utilities'