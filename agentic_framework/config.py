#!/usr/bin/env python3
"""
Configuration file for the Lease Document Processing Agent
Centralized settings for easy management
"""


import os
from pathlib import Path
from typing import Dict, Any, Optional


class Config:
    """Configuration class for the Lease Document Agent"""
    
    # Azure Form Recognizer settings
    AZURE_FORM_RECOGNIZER_ENDPOINT: str = ""
    AZURE_FORM_RECOGNIZER_KEY: str = ""
    
    # OpenAI settings
    OPENAI_API_KEY: str = ""
    OPENAI_BASE_URL: str = ""  # For custom OpenAI endpoints (Azure OpenAI, etc.)
    OPENAI_MODEL: str = "gpt-4o-mini"
    OPENAI_EMBEDDING_MODEL: str = "text-embedding-ada-002"
    OPENAI_TEMPERATURE: float = 0.1
    
    # Azure AI Search settings
    AZURE_SEARCH_ENDPOINT: str = ""
    AZURE_SEARCH_KEY: str = ""
    AZURE_SEARCH_INDEX_NAME: str = "lease-documents"
    
    # Agent settings
    AGENT_MAX_ITERATIONS: int = 10
    AGENT_VERBOSE: bool = True
    
    # File processing settings
    DEFAULT_INPUT_FOLDER: str = "../leases"
    DEFAULT_OUTPUT_FOLDER: str = "./output"
    SUPPORTED_FILE_TYPES: list = [".pdf"]
    
    # Chat interface settings
    CHAT_HISTORY_FILE: str = "./chat_history.json"
    MAX_CHAT_HISTORY: int = 100
    SHOW_TOOL_EXECUTION: bool = True
    
    @classmethod
    def load_from_env(cls) -> 'Config':
        """Load configuration from environment variables"""
        config = cls()
        
        # Azure Form Recognizer
        config.AZURE_FORM_RECOGNIZER_ENDPOINT = os.getenv(
            'AZURE_FORM_RECOGNIZER_ENDPOINT', 
            config.AZURE_FORM_RECOGNIZER_ENDPOINT
        )
        config.AZURE_FORM_RECOGNIZER_KEY = os.getenv(
            'AZURE_FORM_RECOGNIZER_KEY', 
            config.AZURE_FORM_RECOGNIZER_KEY
        )
        
        # OpenAI
        config.OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', config.OPENAI_API_KEY)
        config.OPENAI_BASE_URL = os.getenv('OPENAI_BASE_URL', config.OPENAI_BASE_URL)
        config.OPENAI_MODEL = os.getenv('OPENAI_MODEL', config.OPENAI_MODEL)
        config.OPENAI_EMBEDDING_MODEL = os.getenv('OPENAI_EMBEDDING_MODEL', config.OPENAI_EMBEDDING_MODEL)
        config.OPENAI_TEMPERATURE = float(os.getenv('OPENAI_TEMPERATURE', str(config.OPENAI_TEMPERATURE)))
        
        # Azure AI Search
        config.AZURE_SEARCH_ENDPOINT = os.getenv('AZURE_SEARCH_ENDPOINT', config.AZURE_SEARCH_ENDPOINT)
        config.AZURE_SEARCH_KEY = os.getenv('AZURE_SEARCH_KEY', config.AZURE_SEARCH_KEY)
        config.AZURE_SEARCH_INDEX_NAME = os.getenv('AZURE_SEARCH_INDEX_NAME', config.AZURE_SEARCH_INDEX_NAME)
        
        # Agent settings
        config.AGENT_MAX_ITERATIONS = int(os.getenv('AGENT_MAX_ITERATIONS', str(config.AGENT_MAX_ITERATIONS)))
        config.AGENT_VERBOSE = os.getenv('AGENT_VERBOSE', 'True').lower() == 'true'
        
        # File processing
        config.DEFAULT_INPUT_FOLDER = os.getenv('DEFAULT_INPUT_FOLDER', config.DEFAULT_INPUT_FOLDER)
        config.DEFAULT_OUTPUT_FOLDER = os.getenv('DEFAULT_OUTPUT_FOLDER', config.DEFAULT_OUTPUT_FOLDER)
        
        # Chat interface
        config.CHAT_HISTORY_FILE = os.getenv('CHAT_HISTORY_FILE', config.CHAT_HISTORY_FILE)
        config.MAX_CHAT_HISTORY = int(os.getenv('MAX_CHAT_HISTORY', str(config.MAX_CHAT_HISTORY)))
        config.SHOW_TOOL_EXECUTION = os.getenv('SHOW_TOOL_EXECUTION', 'True').lower() == 'true'
        
        return config
    
    @classmethod
    def create_sample_config_file(cls, file_path: str = "config_sample.py") -> None:
        """Create a sample configuration file for user to fill in"""
        
        sample_content = '''#!/usr/bin/env python3
"""
Sample configuration file for the Lease Document Processing Agent
Copy this file to config_local.py and fill in your actual values
"""

from config import Config

# Create your custom configuration
config = Config()

# =============================================================================
# REQUIRED: Azure Form Recognizer Settings
# =============================================================================
config.AZURE_FORM_RECOGNIZER_ENDPOINT = "https://your-form-recognizer.cognitiveservices.azure.com/"
config.AZURE_FORM_RECOGNIZER_KEY = "your-form-recognizer-key-here"

# =============================================================================
# REQUIRED: OpenAI Settings
# =============================================================================
config.OPENAI_API_KEY = "sk-your-openai-api-key-here"
config.OPENAI_BASE_URL = ""  # Leave empty for standard OpenAI API, or set custom endpoint
                             # Example for Azure OpenAI: "https://your-resource.openai.azure.com/"
config.OPENAI_MODEL = "gpt-4o-mini"  # or "gpt-4o", "gpt-3.5-turbo"
config.OPENAI_EMBEDDING_MODEL = "text-embedding-ada-002"
config.OPENAI_TEMPERATURE = 0.1  # Lower = more deterministic

# =============================================================================
# REQUIRED: Azure AI Search Settings
# =============================================================================
config.AZURE_SEARCH_ENDPOINT = "https://your-search-service.search.windows.net"
config.AZURE_SEARCH_KEY = "your-search-admin-key-here"
config.AZURE_SEARCH_INDEX_NAME = "lease-documents"  # Your index name

# =============================================================================
# OPTIONAL: Agent Behavior Settings
# =============================================================================
config.AGENT_MAX_ITERATIONS = 10  # Maximum reasoning steps
config.AGENT_VERBOSE = True  # Show agent thinking process

# =============================================================================
# OPTIONAL: File Processing Settings
# =============================================================================
config.DEFAULT_INPUT_FOLDER = "../leases"  # Where to look for PDFs
config.DEFAULT_OUTPUT_FOLDER = "./output"  # Where to save results
config.SUPPORTED_FILE_TYPES = [".pdf"]  # File types to process

# =============================================================================
# OPTIONAL: Chat Interface Settings
# =============================================================================
config.CHAT_HISTORY_FILE = "./chat_history.json"  # Chat history storage
config.MAX_CHAT_HISTORY = 100  # Maximum messages to remember
config.SHOW_TOOL_EXECUTION = True  # Show when agent uses tools

# Export the configuration
__all__ = ['config']
'''
        
        with open(file_path, 'w') as f:
            f.write(sample_content)
        
        print(f"📝 Sample configuration file created: {file_path}")
        print("💡 Copy this to config_local.py and fill in your actual values")
    
    def validate(self) -> tuple[bool, list[str]]:
        """Validate the configuration and return any missing required fields"""
        
        required_fields = [
            ('AZURE_FORM_RECOGNIZER_ENDPOINT', self.AZURE_FORM_RECOGNIZER_ENDPOINT),
            ('AZURE_FORM_RECOGNIZER_KEY', self.AZURE_FORM_RECOGNIZER_KEY),
            ('OPENAI_API_KEY', self.OPENAI_API_KEY),
            ('AZURE_SEARCH_ENDPOINT', self.AZURE_SEARCH_ENDPOINT),
            ('AZURE_SEARCH_KEY', self.AZURE_SEARCH_KEY),
        ]
        
        missing_fields = []
        for field_name, field_value in required_fields:
            if not field_value or field_value.strip() == "":
                missing_fields.append(field_name)
        
        return len(missing_fields) == 0, missing_fields
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary (excluding sensitive keys)"""
        
        config_dict = {}
        for attr in dir(self):
            if not attr.startswith('_') and not callable(getattr(self, attr)):
                value = getattr(self, attr)
                # Mask sensitive information
                if 'KEY' in attr and value:
                    config_dict[attr] = f"{value[:8]}...{value[-4:]}" if len(value) > 12 else "***"
                else:
                    config_dict[attr] = value
        
        return config_dict
    
    def print_status(self) -> None:
        """Print configuration status"""
        
        is_valid, missing_fields = self.validate()
        
        print("🔧 Configuration Status")
        print("=" * 40)
        
        if is_valid:
            print("✅ Configuration is valid!")
        else:
            print("❌ Configuration is incomplete!")
            print("Missing required fields:")
            for field in missing_fields:
                print(f"   - {field}")
        
        print(f"\n📊 Current settings:")
        config_dict = self.to_dict()
        for key, value in config_dict.items():
            print(f"   {key}: {value}")


# Default configuration instance
default_config = Config()


def get_config(config_file: Optional[str] = None) -> Config:
    """
    Get configuration from various sources in order of precedence:
    1. Custom config file (if provided)
    2. config_local.py (if exists)
    3. Environment variables
    4. Default values
    """
    
    # Start with environment variables
    config = Config.load_from_env()
    
    # Try to load from custom config file
    if config_file and Path(config_file).exists():
        try:
            import importlib.util
            spec = importlib.util.spec_from_file_location("custom_config", config_file)
            custom_config_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(custom_config_module)
            
            if hasattr(custom_config_module, 'config'):
                config = custom_config_module.config
                print(f"✅ Loaded configuration from: {config_file}")
        except Exception as e:
            print(f"⚠️  Failed to load config file {config_file}: {e}")
    
    # Try to load from config_local.py
    elif Path("config_local.py").exists():
        try:
            from config_local import config as local_config
            config = local_config
            print("✅ Loaded configuration from: config_local.py")
        except Exception as e:
            print(f"⚠️  Failed to load config_local.py: {e}")
    
    return config


def create_sample_config():
    """Helper function to create sample configuration file"""
    Config.create_sample_config_file()


if __name__ == "__main__":
    # When run directly, create sample config and show status
    print("🤖 Lease Document Agent Configuration")
    print("=" * 50)
    
    # Create sample config file
    create_sample_config()
    
    # Load and validate current config
    config = get_config()
    config.print_status()
    
    print(f"\n💡 Next steps:")
    print(f"   1. Fill in your actual values in config_local.py")
    print(f"   2. Run the chat script: python chat_agent.py")