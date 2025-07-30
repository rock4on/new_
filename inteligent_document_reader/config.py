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
    OPENAI_API_KEY = 'your_openai_api_key'
    OPENAI_BASE_URL = None  # Optional: set to custom endpoint if needed
    OPENAI_MODEL = 'gpt-4o-mini'
    OPENAI_TEMPERATURE = 0.1
    OPENAI_MAX_TOKENS = 1000
    OPENAI_EMBEDDING_MODEL = 'text-embedding-ada-002'
    
    # Azure Form Recognizer
    AZURE_FORM_RECOGNIZER_ENDPOINT = 'your_azure_form_recognizer_endpoint'
    AZURE_FORM_RECOGNIZER_KEY = 'your_azure_form_recognizer_key'
    
    # Azure AI Search
    AZURE_SEARCH_ENDPOINT = 'your_azure_search_endpoint'
    AZURE_SEARCH_KEY = 'your_azure_search_key'
    AZURE_SEARCH_INDEX_NAME = 'lease-documents'
    AZURE_SEARCH_UTILITIES_INDEX_NAME = 'lease-utilities'