# Inteligent Document Reader

A modular document processing system for leases, electricity bills, and natural gas bills with configurable folder paths.

## Architecture

Based on the agentic_framework with similar functionality but organized in a modular structure:

```
inteligent_document_reader/
├── agents/              # Document processing agents
├── tools/               # Processing tools (OCR, extraction, embedding, ingestion, matching)
├── prompts/             # AI prompt templates
├── models/              # Pydantic data models (same as agentic_framework)
├── config.py            # Configuration with defaults
├── example_usage.py     # Usage examples
└── README.md           # This file
```

## Workflow

The system implements the exact workflow you specified:

1. **OCR**: Read text using Azure Form Recognizer
2. **AI Extraction**: Extract metadata with OpenAI using forced Pydantic models
3. **Embedding**: Create vector embeddings with OpenAI
4. **Ingestion**: Store in Azure AI Search index

## Document Types Supported

- **Leases**: Uses `LeaseInformation` Pydantic model
- **Electricity Bills**: Uses `NaturalGas_Electricity_Information` Pydantic model  
- **Natural Gas Bills**: Uses `NaturalGas_Electricity_Information` Pydantic model

## Configurable Folder Paths

Default folder paths (defined in `Config` class):

- **Leases**: `./leases`
- **Electricity**: `./electric` 
- **Natural Gas**: `./natural_gas`

## Quick Start

### 1. Environment Setup

Set required environment variables:

```bash
export AZURE_FORM_RECOGNIZER_ENDPOINT="your_endpoint"
export AZURE_FORM_RECOGNIZER_KEY="your_key"
export OPENAI_API_KEY="your_openai_key"
export AZURE_SEARCH_ENDPOINT="your_search_endpoint"
export AZURE_SEARCH_KEY="your_search_key"
```

### 2. Basic Usage

```python
import os
from inteligent_document_reader import DocumentAgent

# Initialize agent with credentials
agent = DocumentAgent(
    azure_form_recognizer_endpoint=os.getenv('AZURE_FORM_RECOGNIZER_ENDPOINT'),
    azure_form_recognizer_key=os.getenv('AZURE_FORM_RECOGNIZER_KEY'),
    openai_api_key=os.getenv('OPENAI_API_KEY'),
    azure_search_endpoint=os.getenv('AZURE_SEARCH_ENDPOINT'),
    azure_search_key=os.getenv('AZURE_SEARCH_KEY')
)

# Process single document
result = agent.process_document(
    file_path="./leases/sample_lease.pdf",
    doc_type="lease",
    client_name="Example Client"
)

# Process entire folders (batch processing)
lease_results = agent.process_leases()           # Process ./leases folder
electricity_results = agent.process_electricity() # Process ./electric folder  
gas_results = agent.process_natural_gas()        # Process ./natural_gas folder

# Custom folder paths
custom_results = agent.process_folder(
    folder_path="./custom/path",
    doc_type="lease",
    client_name="Custom Client"
)

# Excel matching by location (like agentic_framework)
match_results = agent.match_excel("./EGA.xlsx")
print(match_results)
```

### 3. Run Example

```bash
python example_usage.py
```

## Features

### Same as Agentic Framework
- ✅ Azure OCR text extraction
- ✅ OpenAI AI metadata extraction with forced Pydantic models
- ✅ Vector embeddings and Azure AI Search storage
- ✅ Excel location matching functionality
- ✅ Support for leases, electricity, and natural gas documents
- ✅ Same Pydantic models (`LeaseInformation`, `NaturalGas_Electricity_Information`)

### Improvements
- ✅ Modular architecture with separate tools, agents, prompts
- ✅ Configurable folder paths with environment variables
- ✅ Clear separation of concerns
- ✅ Easy to extend and test
- ✅ Simple API for batch processing

## Configuration Options

All configuration via environment variables:

```bash
# Required - Azure Form Recognizer
AZURE_FORM_RECOGNIZER_ENDPOINT=https://your-ocr.cognitiveservices.azure.com/
AZURE_FORM_RECOGNIZER_KEY=your_ocr_key

# Required - OpenAI  
OPENAI_API_KEY=your_openai_key
OPENAI_MODEL=gpt-4o-mini

# Required - Azure AI Search
AZURE_SEARCH_ENDPOINT=https://your-search.search.windows.net
AZURE_SEARCH_KEY=your_search_key
AZURE_SEARCH_INDEX_NAME=lease-documents
AZURE_SEARCH_UTILITIES_INDEX_NAME=lease-utilities

# Optional - Folder Paths (defaults shown)
LEASES_FOLDER=./leases
ELECTRICITY_FOLDER=./electric  
NATURAL_GAS_FOLDER=./natural_gas
DEFAULT_INPUT_FOLDER=./leases
```

## API Reference

### DocumentAgent

Main class that orchestrates the document processing workflow.

#### Methods

- `process_document(file_path, doc_type, client_name)`: Process single document
- `process_folder(folder_path, doc_type, client_name)`: Process entire folder
- `process_leases(folder_path, client_name)`: Process lease documents
- `process_electricity(folder_path, client_name)`: Process electricity bills
- `process_natural_gas(folder_path, client_name)`: Process natural gas bills
- `match_excel(excel_path)`: Match Excel data by location

### Document Types

- `"lease"`: Lease documents
- `"electricity"`: Electricity bills/invoices
- `"natural_gas"`: Natural gas bills/invoices

## Folder Structure

Organize your documents like this:

```
./leases/
├── client1/
│   ├── lease1.pdf
│   └── lease2.pdf
└── client2/
    └── lease3.pdf

./electric/
├── client1/
│   └── electric_bill.pdf
└── client2/
    └── electricity_invoice.pdf

./natural_gas/
├── client1/
│   └── gas_bill.pdf
└── client2/
    └── natural_gas_invoice.pdf
```

The system will automatically detect client names from folder structure or use the provided `client_name` parameter.

## Comparison with Agentic Framework

| Feature | Agentic Framework | Inteligent Document Reader |
|---------|------------------|---------------------------|
| Document Processing | ✅ | ✅ Same functionality |
| Pydantic Models | ✅ | ✅ Exact same models |
| Excel Matching | ✅ | ✅ Same matching logic |
| Architecture | Monolithic | ✅ Modular |
| Folder Paths | Fixed | ✅ Configurable |
| Easy Testing | ❌ | ✅ Separated tools |
| Simple API | ❌ | ✅ Clean interface |

This module provides the exact same functionality as the agentic_framework but with a cleaner, more maintainable modular architecture and configurable folder paths.