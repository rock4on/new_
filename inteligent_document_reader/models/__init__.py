"""
Pydantic models for document processing - same as agentic_framework
"""

from .lease import LeaseInformation
from .utilities import NaturalGas_Electricity_Information, UTILITIES_FIELDS

# Document type mapping
DOCUMENT_MODELS = {
    'lease': LeaseInformation,
    'electricity': NaturalGas_Electricity_Information,
    'natural_gas': NaturalGas_Electricity_Information
}

__all__ = [
    "LeaseInformation",
    "NaturalGas_Electricity_Information", 
    "UTILITIES_FIELDS",
    "DOCUMENT_MODELS"
]