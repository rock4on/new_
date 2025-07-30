"""
Inteligent Document Reader
A modular document processing system for leases, electricity bills, and natural gas bills
"""

__version__ = "1.0.0"

from .agents.document_agent import DocumentAgent

__all__ = ["DocumentAgent"]