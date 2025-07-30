"""
Lease document Pydantic model - same as agentic_framework
"""

from pydantic import BaseModel, Field
from typing import Optional


class LeaseInformation(BaseModel):
    """Pydantic model for structured lease information"""
    description: Optional[str] = Field(
        None,
        description="Description of the property or lease"
    )
    location: Optional[str] = Field(
        None,
        description="Property location or address"
    )
    lease_start_date: Optional[str] = Field(
        None,
        description="Lease start date in YYYY-MM-DD format if possible"
    )
    lease_end_date: Optional[str] = Field(
        None,
        description="Lease end date in YYYY-MM-DD format if possible"
    )
    building_area: Optional[str] = Field(
        None,
        description="Building area (numeric value only)"
    )
    area_unit: Optional[str] = Field(
        None,
        description="Unit of area (sq ft, sq m, etc.)"
    )
    building_type: Optional[str] = Field(
        None,
        description="Type of building (office, retail, warehouse, etc.)"
    )