# Regulation Processing Pipeline

This repository contains a complete pipeline for analyzing regulatory documents:

1. **main.py** - Orchestrates the complete workflow (recommended)
2. **regulation_pipeline.py** - Scrapes regulation documents from Excel-defined sources
3. **file_metadata_processor.py** - Analyzes scraped documents and extracts metadata

## Quick Start

### Complete Pipeline (Recommended)
```bash
# Run both scraping and analysis in sequence
python main.py regulations.xlsx
```

### Individual Components
```bash
# Step 1: Scrape regulations
python regulation_pipeline.py regulations.xlsx

# Step 2: Analyze documents
python file_metadata_processor.py
```

## regulation_pipeline.py

### Overview
Orchestrates the complete regulation scraping process:
1. Reads Excel file containing regulations and source URLs
2. Uses Scrapy with FlareSolverr fallback for web scraping
3. Stores each regulation's content in organized country/regulation folders

### Prerequisites

#### Required Software
- **FlareSolverr**: Bypass Cloudflare protection
  ```bash
  docker run -d -p 8191:8191 ghcr.io/flaresolverr/flaresolverr:latest
  ```

#### Required Python Packages
```bash
pip install scrapy pandas requests beautifulsoup4 html2text
```

#### Required Project Structure
- Scrapy project folder named `documents` must exist in the current directory

### Setup

1. **Start FlareSolverr**:
   ```bash
   docker run -d -p 8191:8191 ghcr.io/flaresolverr/flaresolverr:latest
   ```

2. **Verify FlareSolverr is running**:
   - Should be accessible at `http://localhost:8191`

3. **Prepare Excel file** with columns:
   - `Regulation Name` (or custom column name)
   - `Sources` (URLs separated by commas/newlines)
   - `Country` (or custom column name)

### Usage

#### Basic Usage
```bash
python regulation_pipeline.py <excel_file>
```

#### Custom Column Names
```bash
python regulation_pipeline.py regulations.xlsx "Custom Regulation Column" "Custom Sources Column" "Custom Country Column"
```

#### Example
```bash
python regulation_pipeline.py regulations.xlsx "Regulation Name" "Sources" "Country"
```

### Output Structure
```
regulation_scraping_results/
├── Country1/
│   ├── Regulation_Name_1/
│   │   ├── regulation_info.json
│   │   ├── scraped_content.html
│   │   ├── document.pdf
│   │   └── ...
│   └── Regulation_Name_2/
│       └── ...
├── Country2/
│   └── ...
├── extracted_regulations.json
└── pipeline_final_report.json
```

### Features
- **Automatic prerequisites checking**
- **Folder-based organization** by country and regulation
- **Skip existing folders** to resume interrupted runs
- **Comprehensive logging** and error handling
- **30-minute timeout** per regulation
- **Final reporting** with statistics and success rates

---

## file_metadata_processor.py

### Overview
Processes scraped regulation documents and extracts metadata:
1. Reads documents from regulation_scraping_results folder
2. Extracts text from PDF, HTML, and TXT files
3. Scores documents for ESG relevance
4. Uses LLM to extract structured metadata from relevant documents

### Prerequisites

#### Required Python Packages
```bash
pip install PyPDF2 beautifulsoup4 html2text
```

#### Required Dependencies
- **metadata.py** - Contains `extract_metadata()` function
- **esg_filter.py** - Contains `esg_match_score()` and `ESG_KEYWORDS`

### Setup

1. **Ensure input directory exists**:
   - Default: `regulation_scraping_results/` (output from regulation_pipeline.py)

2. **Verify dependencies are available**:
   - `metadata.py` with `extract_metadata()` function
   - `esg_filter.py` with ESG filtering capabilities

### Usage

#### Basic Usage (processes all scraped regulations)
```bash
python file_metadata_processor.py
```

#### Programmatic Usage
```python
from file_metadata_processor import FileMetadataProcessor

processor = FileMetadataProcessor(
    input_dir="regulation_scraping_results",
    output_dir="file_metadata_analysis_results",
    max_workers=6
)

results = processor.process_all_regulations(use_llm=True)
```

### Output Structure
```
file_metadata_analysis_results/
├── Country1/
│   ├── Country1_summary.json
│   ├── Regulation_Name_1/
│   │   ├── regulation_summary.json
│   │   ├── document1_pdf_analysis.json
│   │   └── document2_html_analysis.json
│   └── ...
├── Country2/
│   └── ...
├── final_metadata_analysis_summary.json
└── .processed_folders.json
```

### Features
- **Concurrent processing** with configurable worker threads
- **ESG relevance scoring** (threshold: 30 points)
- **Intelligent text chunking** for large documents
- **Duplicate detection** using file hashing
- **Resume capability** - skips already processed folders
- **Comprehensive metadata extraction** for ESG-relevant documents
- **Progress tracking** with time estimates

### ESG Relevance
Documents are scored based on ESG keyword matching. Only documents scoring ≥30 points undergo metadata extraction to optimize processing time.

### Performance Features
- **Parallel file processing** (configurable workers)
- **Text chunk optimization** for LLM processing
- **File content caching** to avoid reprocessing duplicates
- **Incremental processing** with skip tracking

---

---

## main.py

### Overview
Orchestrates the complete regulation processing workflow by running both pipelines in sequence with comprehensive error handling and progress tracking.

### Prerequisites
- All prerequisites from both `regulation_pipeline.py` and `file_metadata_processor.py`
- FlareSolverr running on localhost:8191

### Usage

#### Basic Usage
```bash
python main.py regulations.xlsx
```

#### Custom Column Names
```bash
python main.py regulations.xlsx --regulation-col "Custom Regulation Column" --sources-col "Custom Sources Column" --country-col "Custom Country Column"
```

#### Performance Tuning
```bash
# Use 4 workers for analysis
python main.py regulations.xlsx --max-workers 4

# Skip LLM metadata extraction (faster)
python main.py regulations.xlsx --no-llm
```

#### Partial Execution
```bash
# Only run scraping
python main.py regulations.xlsx --skip-analysis

# Only run analysis (requires existing scraped data)
python main.py regulations.xlsx --skip-scraping
```

#### Custom Output Directories
```bash
python main.py regulations.xlsx --scraping-output "custom_scraping_dir" --analysis-output "custom_analysis_dir"
```

### Features
- **Automatic prerequisites checking**
- **Sequential pipeline execution** with error handling
- **Comprehensive progress tracking** and timing
- **Flexible execution options** (skip steps, custom parameters)
- **Final summary report** with success/failure status
- **Command-line help** with usage examples

### Command Line Options
- `--regulation-col`: Column name for regulation names (default: "Regulation Name")
- `--sources-col`: Column name for source URLs (default: "Sources")
- `--country-col`: Column name for countries (default: "Country")
- `--max-workers`: Concurrent workers for analysis (default: 6)
- `--no-llm`: Skip LLM metadata extraction
- `--skip-scraping`: Only run document analysis
- `--skip-analysis`: Only run regulation scraping
- `--scraping-output`: Custom scraping output directory
- `--analysis-output`: Custom analysis output directory

---

## Complete Workflow

### Option 1: Using main.py (Recommended)
```bash
# Start FlareSolverr
docker run -d -p 8191:8191 ghcr.io/flaresolverr/flaresolverr:latest

# Run complete pipeline
python main.py regulations.xlsx
```

### Option 2: Manual Steps
```bash
# Start FlareSolverr
docker run -d -p 8191:8191 ghcr.io/flaresolverr/flaresolverr:latest

# Step 1: Scrape regulations
python regulation_pipeline.py regulations.xlsx

# Step 2: Analyze documents
python file_metadata_processor.py
```

### Results
- **Scraping results**: `regulation_scraping_results/pipeline_final_report.json`
- **Analysis results**: `file_metadata_analysis_results/final_metadata_analysis_summary.json`
- **Pipeline summary**: Console output with timing and success rates

## Troubleshooting

### regulation_pipeline.py
- **FlareSolverr not accessible**: Ensure Docker container is running
- **Missing packages**: Run `pip install scrapy pandas requests beautifulsoup4 html2text`
- **Scrapy project not found**: Ensure `documents/` folder exists

### file_metadata_processor.py
- **Import errors**: Ensure `metadata.py` and `esg_filter.py` are available
- **No input files**: Run `regulation_pipeline.py` first
- **Memory issues**: Reduce `max_workers` parameter