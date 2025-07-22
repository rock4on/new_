# Lease Document Processing Agent

An intelligent agent built with LangChain's ReAct framework for processing lease documents with Azure services and OpenAI.

## Features

The agent provides the following capabilities through specialized tools:

### 🔧 Available Tools

1. **Azure OCR Text Extraction** (`azure_ocr_extractor`)
   - Extracts text from PDF documents using Azure Form Recognizer
   - High-quality OCR for lease documents

2. **Vector Store Ingestion** (`vector_store_ingest`)
   - Ingests document text into Azure AI Search with OpenAI embeddings
   - Stores structured metadata for efficient retrieval

3. **Location-based Document Matching** (`location_matcher`)
   - Finds documents matching specific locations
   - Filters by location fields in the vector store

4. **Vector Search for Field Extraction** (`vector_search_fields`)
   - Performs semantic search across document collection
   - Extracts specific fields from relevant documents

### 🤖 Agent Capabilities

- **End-to-end Document Processing**: Upload PDF → Extract text → Generate embeddings → Store in vector database
- **Intelligent Query Processing**: Uses ReAct pattern to break down complex queries
- **Location-based Search**: Find all documents for specific addresses or locations
- **Field Extraction**: Extract specific lease information (dates, areas, types, etc.)
- **Conversational Interface**: Ask natural language questions about your document collection

## Quick Start

### Option 1: Automated Setup
```bash
python setup.py
```
Follow the prompts to install dependencies and configure the agent.

### Option 2: Manual Setup

1. Install required packages:
```bash
pip install -r requirements.txt
```

2. Create configuration file:
```bash
python config.py  # Creates config_sample.py
cp config_sample.py config_local.py
# Edit config_local.py with your actual values
```

3. Alternative: Set up environment variables:
```bash
export AZURE_FORM_RECOGNIZER_ENDPOINT="your_azure_endpoint"
export AZURE_FORM_RECOGNIZER_KEY="your_azure_key"
export OPENAI_API_KEY="your_openai_key"
export AZURE_SEARCH_ENDPOINT="your_search_endpoint"
export AZURE_SEARCH_KEY="your_search_key"
export AZURE_SEARCH_INDEX_NAME="lease-documents"  # optional
```

### Start Chatting
```bash
python chat_agent.py
```

## Usage

### Interactive Chat Interface

The easiest way to use the agent is through the interactive chat interface:

```bash
python chat_agent.py
```

This provides a conversational interface where you can:
- Ask natural language questions
- Process documents interactively
- Get help with `/help` command
- View chat history with `/history`
- Use various shortcuts and commands

Example chat session:
```
👤 You: Process the lease document at ../leases/client1/lease.pdf
🤖 Agent: I'll process that lease document for you using Azure OCR and store it in the vector database...

👤 You: Find all office buildings in Chicago
🤖 Agent: I found 3 office buildings in Chicago: ...

👤 You: What's the average lease area?
🤖 Agent: Based on the documents in the database, the average lease area is 5,420 sq ft...
```

### Chat Commands
- `/help` - Show detailed help
- `/history` - View recent chat history  
- `/status` - Check system status
- `/clear` - Clear chat history
- `/quit` - Exit chat

### Programmatic Usage

```python
from lease_agent import create_lease_agent_from_config
from config import get_config

# Create agent from configuration
config = get_config()
agent = create_lease_agent_from_config(config)

# Process a document
result = agent.process_document(
    file_path="path/to/lease.pdf",
    client_name="Client Name"
)

# Find documents by location
result = agent.find_documents_by_location("New York")

# Search for specific fields
result = agent.search_for_fields(
    search_query="office building leases",
    fields=["location", "lease_start_date", "building_area"]
)

# Ask natural language questions
result = agent.ask_question("What leases expire in 2024?")
```

### Advanced Usage

```python
from lease_agent import LeaseDocumentAgent

# Create agent with explicit parameters
agent = LeaseDocumentAgent(
    azure_endpoint="your_azure_endpoint",
    azure_key="your_azure_key",
    openai_api_key="your_openai_key",
    azure_search_endpoint="your_search_endpoint",
    azure_search_key="your_search_key",
    search_index_name="custom-index-name"
)

# Complex workflow example
result = agent.ask_question("""
    Analyze all lease documents and provide:
    1. Total number of leases
    2. Average building area
    3. Most common building types
    4. Leases expiring in the next 12 months
""")
```

## Example Queries

The agent can handle various types of queries:

### Document Processing
- "Process the lease document at /path/to/file.pdf for client ABC Corp"
- "Extract text from lease.pdf and add it to the vector store"

### Search and Analysis
- "Find all office building leases in downtown Chicago"
- "What lease documents do we have for retail spaces?"
- "List all leases that expire between 2024-01-01 and 2024-12-31"

### Data Extraction
- "Extract location, lease dates, and building area from all warehouse leases"
- "Find the client with the largest total leased area"
- "Which documents are missing location information?"

### Reporting
- "Generate a summary of all lease documents in the system"
- "Compare building areas across different property types"
- "Identify leases that need renewal in the next 6 months"

## ReAct Pattern

The agent uses the ReAct (Reasoning and Acting) pattern:

1. **Thought**: Analyzes the query and plans the approach
2. **Action**: Selects and executes the appropriate tool
3. **Observation**: Reviews the results from the tool
4. **Repeat**: Continues until the query is fully answered
5. **Final Answer**: Provides a comprehensive response

This allows the agent to:
- Break down complex queries into manageable steps
- Choose the right tools for each task
- Combine information from multiple sources
- Provide detailed, accurate responses

## Architecture

```
User Query → ReAct Agent → Tool Selection → Azure Services
                ↓              ↓
        Final Answer ← Result Processing ← Tool Execution
```

### Components

- **LangChain ReAct Agent**: Orchestrates the reasoning and tool execution
- **Custom Tools**: Specialized tools for document processing tasks
- **Azure Form Recognizer**: OCR for text extraction
- **Azure AI Search**: Vector storage and semantic search
- **OpenAI GPT**: Language understanding and response generation
- **OpenAI Embeddings**: Vector representations for semantic search

## Error Handling

The agent includes robust error handling:
- Validation of input parameters
- Graceful handling of Azure service errors
- Clear error messages for debugging
- Fallback strategies for failed operations

## Integration with Existing Code

This agent can work alongside your existing `lease_document_processor_vector.py`:
- Use the same Azure services and configuration
- Process documents that were already indexed
- Extend functionality with conversational queries
- Maintain compatibility with existing data structures