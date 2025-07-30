"""
Extraction Tool for extracting structured information using OpenAI with forced Pydantic models
"""

from typing import Any, Dict
from langchain.tools import BaseTool
from pydantic import Field

from ..models import DOCUMENT_MODELS, LeaseInformation, NaturalGas_Electricity_Information
from ..prompts import get_system_prompt, get_user_prompt, get_model
from ..config import Config


class ExtractionTool(BaseTool):
    """Tool for extracting structured information from text using OpenAI with forced Pydantic models"""
    
    name: str = "extraction_tool"
    description: str = "Extracts structured information from document text using OpenAI with forced Pydantic response format. Input should be JSON with 'text' and 'doc_type' (lease/electricity/natural_gas)."
    openai_client: Any = Field(default=None, exclude=True)
    
    def __init__(self, openai_client, **kwargs):
        super().__init__(**kwargs)
        object.__setattr__(self, 'openai_client', openai_client)
    
    def extract_lease_info(self, text: str) -> Dict[str, Any]:
        """Extract lease information using forced Pydantic model"""
        try:
            print('   🤖 Extracting lease information with AI...')
            
            # Get prompts from prompts module
            system_prompt = get_system_prompt("lease")
            user_prompt = get_user_prompt("lease", text)
            model = get_model()
            
            response = self.openai_client.chat.completions.parse(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format=LeaseInformation,
                temperature=Config.OPENAI_TEMPERATURE,
                max_tokens=Config.OPENAI_MAX_TOKENS
            )
            
            # Parse the structured response
            lease_info = response.choices[0].message.parsed
            extracted_data = lease_info.model_dump()
            
            # Check if AI extracted any meaningful data
            has_data = any(value is not None and str(value).strip() for value in extracted_data.values())
            
            if has_data:
                extracted_fields = [field for field, value in extracted_data.items() 
                                  if value is not None and str(value).strip()]
                print(f"   ✅ AI successfully extracted lease fields: {', '.join(extracted_fields)}")
            else:
                print("   ⚠️  No lease information found in document")
            
            return extracted_data
            
        except Exception as e:
            print(f"   ❌ AI lease extraction failed: {e}")
            return {
                'description': None,
                'location': None,
                'lease_start_date': None,
                'lease_end_date': None,
                'building_area': None,
                'area_unit': None,
                'building_type': None
            }
    
    def extract_utilities_info(self, text: str, doc_type: str) -> Dict[str, Any]:
        """Extract utilities information using forced Pydantic model"""
        try:
            print(f'   🤖 Extracting {doc_type} information with AI...')
            
            # Get prompts from prompts module (doc_type should be "electricity" or "natural_gas")
            system_prompt = get_system_prompt(doc_type)
            user_prompt = get_user_prompt(doc_type, text)
            model = get_model()
            
            response = self.openai_client.chat.completions.parse(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format=NaturalGas_Electricity_Information,
                temperature=Config.OPENAI_TEMPERATURE,
                max_tokens=Config.OPENAI_MAX_TOKENS
            )
            
            # Parse the structured response
            utilities_info = response.choices[0].message.parsed
            extracted_data = utilities_info.model_dump()
            
            # Check if AI extracted any meaningful data
            has_data = any(value is not None and str(value).strip() for value in extracted_data.values())
            
            if has_data:
                extracted_fields = [field for field, value in extracted_data.items() 
                                  if value is not None and str(value).strip()]
                print(f"   ✅ AI successfully extracted utilities fields: {', '.join(extracted_fields)}")
            else:
                print("   ⚠️  No utilities information found in document")
            
            return extracted_data
            
        except Exception as e:
            print(f"   ❌ AI utilities extraction failed: {e}")
            return {
                'vendor_name': None,
                'account_or_invoice_number': None,
                'invoice_date': None,
                'location': None,
                'measurement_period_start': None,
                'measurement_period_end': None,
                'consumption_amount': None,
                'unit_of_measure': None
            }
    
    def _run(self, input_data: str) -> str:
        """Extract structured information based on document type"""
        try:
            import json
            data = json.loads(input_data)
            text = data.get('text', '')
            doc_type = data.get('doc_type', 'lease').lower()
            
            if not text:
                return "Error: No text provided for extraction"
            
            print(f"   📋 Extracting {doc_type} information from text ({len(text)} chars)")
            
            if doc_type == 'lease':
                extracted_info = self.extract_lease_info(text)
            elif doc_type in ['electricity', 'natural_gas']:
                extracted_info = self.extract_utilities_info(text, doc_type)
            else:
                return f"Error: Unknown document type '{doc_type}'. Supported: lease, electricity, natural_gas"
            
            # Format results
            result = f"📋 EXTRACTED {doc_type.upper()} INFORMATION:\n\n"
            for field, value in extracted_info.items():
                field_display = field.replace('_', ' ').title()
                if value is not None and str(value).strip():
                    result += f"• {field_display}: {value}\n"
                else:
                    result += f"• {field_display}: Not found\n"
            
            return result
            
        except json.JSONDecodeError:
            return "Error: Invalid JSON input format"
        except Exception as e:
            return f"Error during extraction: {str(e)}"