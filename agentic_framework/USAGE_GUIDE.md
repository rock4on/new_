# 🏢⚡ Integrated Lease & Utilities Document Analysis System

## Overview

The enhanced lease_agent.py now supports **both lease documents AND utilities data** (natural gas and electricity bills/invoices). The system can process, search, and analyze all document types using a unified interface.

## 🚀 Quick Start

### 1. Basic Setup

```python
from lease_agent import create_lease_agent_from_config
from config import get_config

# Initialize the integrated agent
config = get_config()
agent = create_lease_agent_from_config(config)

# Ask questions about any document type
result = agent.ask_question("Find all documents for Chicago")
print(result['result'])
```

### 2. Document Types Supported

- **📄 Lease Documents**: Office leases, rental agreements, property contracts
- **⚡ Electricity Bills**: Electric utility invoices and consumption data  
- **🔥 Natural Gas Bills**: Gas utility invoices and usage information

## 💬 Natural Language Interface

### Lease Queries
```python
# Find lease documents
agent.ask_question("Find all office leases in Chicago")
agent.ask_question("What leases expire in 2024?")
agent.ask_question("Show me leases for client2")
agent.ask_question("Which buildings have the largest area?")
```

### Utilities Queries  
```python
# Find utilities documents
agent.ask_question("Find natural gas bills for Chicago")
agent.ask_question("What's the electricity consumption for client2?")
agent.ask_question("Which vendor supplies the most gas?")
agent.ask_question("Show me high consumption electricity bills")
```

### Mixed Analysis
```python
# Analyze both document types together
agent.ask_question("Find all documents for Helsinki (leases and utilities)")
agent.ask_question("Show me everything for client2")
agent.ask_question("Compare building costs vs utilities costs")
agent.ask_question("Which locations have both leases and utilities?")
```

## 🔧 Direct Method Calls

### Process Documents

```python
# Process lease document
lease_result = agent.process_document(
    file_path="/path/to/lease.pdf",
    client_name="client2"
)

# Process utilities document  
utilities_result = agent.process_utilities_document(
    file_path="/path/to/gas_bill.pdf",
    utility_type="natural_gas",
    client_name="client2"
)
```

### Search by Location

```python
# Find lease documents by location
lease_location = agent.find_documents_by_location("Chicago")

# Find utilities documents by location
utilities_location = agent.find_utilities_by_location("Chicago")
```

### Field-Specific Search

```python
# Search lease documents for specific fields
lease_fields = agent.search_for_fields(
    search_query="office buildings 2024",
    fields=["location", "building_area", "lease_end_date"]
)

# Search utilities documents for specific fields
utilities_fields = agent.search_utilities_for_fields(
    search_query="high consumption winter",
    fields=["vendor_name", "consumption_amount", "location"]
)
```

## 📊 Analysis Features

### Portfolio Analysis
```python
# Get comprehensive analysis (includes both leases and utilities)
result = agent.ask_question("Give me a complete portfolio analysis")
```

### Consumption Analysis  
```python
# Analyze utilities consumption patterns
result = agent.ask_question("Analyze electricity consumption trends by location")
result = agent.ask_question("Compare gas usage between different clients")
```

### Vendor Analysis
```python
# Analyze utilities vendors
result = agent.ask_question("Which utilities vendors do we use most?")
result = agent.ask_question("Show me vendor costs by location")
```

## 🔍 Advanced Search Examples

### Location Matching (Works for Both Document Types)
```python
# ESG-style location matching works across all document types
agent.ask_question("Helsinki")  # Finds both leases and utilities for Helsinki
agent.ask_question("client2")   # Finds all documents for client2
```

### Consumption-Based Queries
```python
agent.ask_question("Find locations with electricity usage over 10,000 kWh")
agent.ask_question("Show me natural gas consumption for winter months")
agent.ask_question("Which buildings use the most energy per square foot?")
```

### Time-Based Analysis
```python
agent.ask_question("What utilities contracts expire this year?")
agent.ask_question("Compare Q1 vs Q4 energy consumption")
agent.ask_question("Show me recent utilities invoices")
```

## 📁 Batch Processing (Separate Folders)

The system now uses **3 separate folders** for different document types:

### Folder Structure
```
/leases          - Place lease documents here
/electric        - Place electric bills here  
/natural_gas     - Place natural gas bills here
```

### Separate Ingestion Tools
```python
# Process lease documents from /leases folder
result = agent.ask_question("Ingest all lease documents")

# Process electric bills from /electric folder  
result = agent.ask_question("Process all electric bills")

# Process natural gas bills from /natural_gas folder
result = agent.ask_question("Ingest natural gas documents")

# Each folder can have client subfolders for organization:
# /leases/client1/document.pdf
# /electric/client2/bill.pdf
# /natural_gas/client3/invoice.pdf
```

## 🛠️ Available Tools

The agent automatically uses the appropriate tools based on your query:

### Lease Tools
- `azure_ocr_extractor`: Extract text from lease PDFs
- `vector_store_ingest`: Store lease documents  
- `location_matcher`: Find leases by location
- `vector_search_fields`: Search lease documents
- `lease_analysis`: Analyze lease portfolio

### Utilities Tools  
- `utilities_ocr_extractor`: Extract text from utilities PDFs
- `utilities_vector_store_ingest`: Store utilities documents
- `utilities_location_matcher`: Find utilities by location
- `utilities_vector_search_fields`: Search utilities documents

### Batch Ingestion Tools
- `batch_ingest_leases`: Process all PDFs in /leases folder
- `batch_ingest_electric`: Process all PDFs in /electric folder  
- `batch_ingest_natural_gas`: Process all PDFs in /natural_gas folder

### Shared Tools
- `match_data`: Match Excel data with vector database

## 📈 Data Schema

### Lease Information
- `description`, `location`, `lease_start_date`, `lease_end_date`
- `building_area`, `area_unit`, `building_type`

### Utilities Information (Natural Gas & Electricity)
- `vendor_name`, `account_or_invoice_number`, `invoice_date`
- `location`, `measurement_period_start`, `measurement_period_end`  
- `consumption_amount`, `unit_of_measure`

## 🎯 Use Cases

### 1. ESG Reporting
```python
# Get comprehensive ESG data for a location
agent.ask_question("Show me all ESG data for Helsinki - leases, gas, and electricity")
```

### 2. Cost Analysis
```python
# Analyze total occupancy costs
agent.ask_question("Compare lease costs vs utilities costs by location")
```

### 3. Efficiency Analysis
```python
# Analyze energy efficiency
agent.ask_question("Which buildings have the best energy efficiency per square foot?")
```

### 4. Vendor Management
```python
# Utilities vendor analysis
agent.ask_question("Show me all utilities vendors and their contract terms")
```

### 5. Compliance Tracking
```python
# Track expiring contracts
agent.ask_question("What leases and utilities contracts expire in the next 6 months?")
```

## 🚨 Important Notes

1. **Automatic Detection**: The system automatically detects document types and uses appropriate tools
2. **Location Matching**: Uses the same ESG matching logic for all document types
3. **Unified Search**: One query can search across lease documents, gas bills, and electricity invoices
4. **Schema Consistency**: Maintains exact same data structure as the original full.py implementation
5. **Vector Search**: Supports semantic search across all document types

## 🔧 Configuration

Make sure your `config.py` has all required Azure and OpenAI credentials:

```python
# Required for the integrated system
AZURE_FORM_RECOGNIZER_ENDPOINT = "your-endpoint"
AZURE_FORM_RECOGNIZER_KEY = "your-key"
AZURE_SEARCH_ENDPOINT = "your-search-endpoint"  
AZURE_SEARCH_KEY = "your-search-key"
OPENAI_API_KEY = "your-openai-key"
```

## 📞 Getting Help

```python
# Ask the agent what it can do
result = agent.ask_question("What can you help me with?")
print(result['result'])
```

The system will show you all available capabilities for both lease and utilities document analysis!