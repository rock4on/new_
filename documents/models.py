from pydantic import BaseModel, Field
from enum import Enum

class Reg(str, Enum):
    application = "Application Standard"
    general = "General Standard"
    climate = "Climate Standard"
    none = "None"

class RegulationModel(BaseModel):
    regulation: Reg = Field(
        description="Which SSBJ regulation family the document belongs to. "
                    "Choose exactly one of the enum values."
    )