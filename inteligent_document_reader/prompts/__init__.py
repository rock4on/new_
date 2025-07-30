"""
Prompt templates for document processing - easily modifiable
"""

# System prompts for different document types
SYSTEM_PROMPTS = {
    "lease_extraction": "You are a document analyst specializing in lease agreement analysis. Extract information accurately from the provided text.",
    "electricity_extraction": "You are a document analyst specializing in electricity bill and invoice analysis. Extract information accurately from electricity utility bills, invoices, and statements.",
    "natural_gas_extraction": "You are a document analyst specializing in natural gas bill and invoice analysis. Extract information accurately from natural gas utility bills, invoices, and statements."
}

# User prompts with placeholders for text
USER_PROMPTS = {
    "lease_extraction": """Extract lease information from the following document text. Identify and extract all available fields.

For dates, use YYYY-MM-DD format if possible.
For building area, extract the numeric value only.
For area unit, use standard units like "sq ft", "sq m", "sqft", etc.

Document text:
{text}""",
    
    "electricity_extraction": """Extract electricity bill information from the following document text. This is an electricity utility bill, invoice, or statement.
Identify and extract all available fields related to electricity consumption and billing.

For dates, use YYYY-MM-DD format if possible.  
For consumption amount, extract the numeric value only.
For unit of measure, use standard electricity units like "kWh", "MWh", "kW", etc.
For vendor name, extract the full electricity company/utility name.

Document text:
{text}""",
    
    "natural_gas_extraction": """Extract natural gas bill information from the following document text. This is a natural gas utility bill, invoice, or statement.
Identify and extract all available fields related to natural gas consumption and billing.

For dates, use YYYY-MM-DD format if possible.  
For consumption amount, extract the numeric value only.
For unit of measure, use standard natural gas units like "therms", "m3", "ccf", "BTU", "mcf", etc.
For vendor name, extract the full natural gas company/utility name.

Document text:
{text}"""
}

# OpenAI model - can be modified
DEFAULT_MODEL = "gpt-4o-mini"

def get_system_prompt(doc_type: str) -> str:
    """Get system prompt for document type"""
    if doc_type == "lease":
        return SYSTEM_PROMPTS["lease_extraction"]
    elif doc_type == "electricity":
        return SYSTEM_PROMPTS["electricity_extraction"]
    elif doc_type == "natural_gas":
        return SYSTEM_PROMPTS["natural_gas_extraction"]
    else:
        return SYSTEM_PROMPTS["lease_extraction"]  # default

def get_user_prompt(doc_type: str, text: str) -> str:
    """Get formatted user prompt for document type"""
    if doc_type == "lease":
        return USER_PROMPTS["lease_extraction"].format(text=text)
    elif doc_type == "electricity":
        return USER_PROMPTS["electricity_extraction"].format(text=text)
    elif doc_type == "natural_gas":
        return USER_PROMPTS["natural_gas_extraction"].format(text=text)
    else:
        return USER_PROMPTS["lease_extraction"].format(text=text)  # default

def get_model() -> str:
    """Get OpenAI model"""
    return DEFAULT_MODEL

def update_prompt(prompt_type: str, doc_type: str, new_prompt: str):
    """Update a prompt template"""
    if prompt_type == "system":
        if doc_type == "lease":
            SYSTEM_PROMPTS["lease_extraction"] = new_prompt
        elif doc_type == "electricity":
            SYSTEM_PROMPTS["electricity_extraction"] = new_prompt
        elif doc_type == "natural_gas":
            SYSTEM_PROMPTS["natural_gas_extraction"] = new_prompt
    elif prompt_type == "user":
        if doc_type == "lease":
            USER_PROMPTS["lease_extraction"] = new_prompt
        elif doc_type == "electricity":
            USER_PROMPTS["electricity_extraction"] = new_prompt
        elif doc_type == "natural_gas":
            USER_PROMPTS["natural_gas_extraction"] = new_prompt

def update_model(new_model: str):
    """Update OpenAI model"""
    global DEFAULT_MODEL
    DEFAULT_MODEL = new_model

__all__ = [
    "SYSTEM_PROMPTS", 
    "USER_PROMPTS", 
    "DEFAULT_MODEL",
    "get_system_prompt",
    "get_user_prompt", 
    "get_model",
    "update_prompt",
    "update_model"
]