#!/usr/bin/env python3
"""
Utilities Data Models for Natural Gas and Electricity Information
"""

from typing import Optional
from pydantic import BaseModel, Field


class NaturalGas_Electricity_Information(BaseModel):
    """Pydantic model for structured natural gas and electricity information extraction"""
    
    vendor_name: Optional[str] = Field(
        None,
        description="Name of the vendor or supplier"
    )
    
    account_or_invoice_number: Optional[str] = Field(
        None,
        description="Account or invoice number associated with the service"
    )
    
    invoice_date: Optional[str] = Field(
        None,
        description="Date of the invoice in YYYY-MM-DD format if possible"
    )
    
    location: Optional[str] = Field(
        None,
        description="Service or billing address for the natural gas service"
    )
    
    measurement_period_start: Optional[str] = Field(
        None,
        description="Start date of the billing or measurement period in YYYY-MM-DD format if possible"
    )
    
    measurement_period_end: Optional[str] = Field(
        None,
        description="End date of the billing or measurement period in YYYY-MM-DD format if possible"   
    )
    
    consumption_amount: Optional[float] = Field(
        None,
        description="Total consumption or usage for the period (numeric value only)"
    )
    
    unit_of_measure: Optional[str] = Field(
        None,
        description="Unit of consumption (e.g., therms, m3, etc.)"
    )


# Field mappings for utilities processing - matches full.py schema exactly
UTILITIES_FIELDS = {
    'vendor_name': 'Vendor Name',
    'account_or_invoice_number': 'Account or Invoice Number',
    'invoice_date': 'Invoice Date',
    'location': 'Service or Billing Address',
    'measurement_period_start': 'Measurement Period Start',
    'measurement_period_end': 'Measurement Period End',
    'consumption_amount': 'Consumption Amount',
    'unit_of_measure': 'Unit of Measure (e.g., therms, m3, etc.)'
}