# Regulation Scraping Pipeline

A comprehensive pipeline that reads Excel files containing regulation data and scrapes content from source URLs with automatic fallback to FlareSolverr for protected sites.

## Overview

The pipeline consists of several components:

1. **Excel Reader** (`excel_reader.py`) - Extracts regulation names and source URLs from Excel files
2. **HTML Downloader** (`html_downloader.py`) - Downloads HTML content using FlareSolverr
3. **PDF Downloader** (`pdf_downloader.py`) - Downloads PDF files using FlareSolverr  
4. **Regulation Spider** (`documents/spiders/regulation_spider.py`) - Scrapy spider with FlareSolverr fallback
5. **Pipeline Orchestrator** (`regulation_pipeline.py`) - Coordinates the entire process

## Features

- **Intelligent URL Extraction**: Extracts URLs from Excel cells containing text with hyperlinks
- **Multi-Strategy Scraping**: Uses normal Scrapy first, falls back to FlareSolverr for 403 errors
- **Depth-Limited Crawling**: Configurable depth limit (2-3 levels) for comprehensive content discovery
- **Content Type Detection**: Automatically handles HTML pages and PDF documents
- **Cloudflare Bypass**: Uses FlareSolverr to bypass Cloudflare protection and solve challenges
- **Organized Storage**: Each regulation gets its own folder with all scraped content
- **Comprehensive Logging**: Detailed logs and metadata for all scraped content
- **Error Handling**: Robust error handling with fallback mechanisms

## Prerequisites

### Required Software
- Python 3.7+
- FlareSolverr (Docker container)
- Scrapy project setup

### Required Python Packages
```bash
pip install scrapy pandas requests beautifulsoup4 html2text openpyxl
```

### FlareSolverr Setup
```bash
# Start FlareSolverr container
docker run -d -p 8191:8191 ghcr.io/flaresolverr/flaresolverr:latest

# Verify it's running
curl http://localhost:8191/v1
```

## Excel File Format

Your Excel file should have (at minimum) these columns:
- **Regulation Name**: Name of the regulation
- **Sources**: Text containing URLs or hyperlinks to source documents

Example:
| Regulation Name | Sources |
|----------------|---------|
| Banking Regulation 2024 | https://example.gov/banking-rules.pdf, https://sec.gov/documents/banking/ |
| Securities Act Amendment | Visit https://sec.gov.ph/regulations/securities for more details |

## Usage

### Basic Usage
```bash
python regulation_pipeline.py your_regulations.xlsx
```

### Custom Column Names
```bash
python regulation_pipeline.py regulations.xlsx "Regulation Title" "Source Links"
```

### Step-by-Step Process

1. **Check Prerequisites**: Verifies FlareSolverr is running and packages are installed
2. **Load Excel Data**: Extracts regulation names and URLs from the Excel file
3. **Create Folders**: Creates individual folders for each regulation
4. **Scrape Content**: For each regulation:
   - Attempts normal Scrapy crawling first
   - Falls back to FlareSolverr for 403 errors or Cloudflare protection
   - Downloads HTML pages, PDF files, and extracts text content
   - Follows links up to the specified depth limit
5. **Generate Report**: Creates comprehensive report with statistics

## Output Structure

```
regulation_scraping_results/
├── extracted_regulations.json          # Parsed Excel data
├── pipeline_final_report.json         # Complete pipeline report
├── Banking_Regulation_2024/           # Individual regulation folder
│   ├── regulation_info.json          # Regulation metadata
│   ├── scraping_summary.json         # Scraping results
│   ├── *.html                        # Downloaded HTML pages
│   ├── *.pdf                         # Downloaded PDF files
│   ├── *.txt                         # Extracted text content
│   └── *_metadata.json               # File metadata
└── Securities_Act_Amendment/
    └── ... (similar structure)
```

## Configuration

### Scrapy Settings (in regulation_spider.py)
- `DEPTH_LIMIT`: 3 (crawl depth limit)
- `DOWNLOAD_DELAY`: 10 seconds
- `CONCURRENT_REQUESTS`: 1 (conservative for government sites)
- `DOWNLOAD_TIMEOUT`: 180 seconds

### FlareSolverr Settings
- `maxTimeout`: 300000ms (5 minutes for complex challenges)
- Compatible with FlareSolverr v2 (no custom headers)

## Error Handling

The pipeline includes multiple fallback mechanisms:

1. **Normal Scrapy Request** → Success ✅
2. **403 Error Detected** → FlareSolverr Fallback
3. **FlareSolverr HTML Download** → Extract links and PDFs
4. **FlareSolverr PDF Download** → Direct PDF download
5. **All Methods Failed** → Log error and continue

## Monitoring

### Real-time Progress
The pipeline provides detailed console output showing:
- Prerequisites check results
- Excel parsing progress
- Individual regulation scraping status
- Content download progress
- Final statistics

### Log Files
- Scrapy generates detailed logs for each spider run
- Individual metadata files for each downloaded item
- Summary JSON files for each regulation
- Final pipeline report with complete statistics

## Troubleshooting

### Common Issues

**FlareSolverr Not Running**
```bash
# Check if container is running
docker ps | grep flaresolverr

# Start if not running
docker run -d -p 8191:8191 ghcr.io/flaresolverr/flaresolverr:latest
```

**Missing Dependencies**
```bash
pip install scrapy pandas requests beautifulsoup4 html2text openpyxl
```

**Excel File Not Found**
- Verify the file path is correct
- Ensure the file has .xlsx extension
- Check file permissions

**No URLs Found**
- Verify column names match exactly (case-sensitive)
- Check that cells contain actual URLs or hyperlinks
- Use the Excel reader standalone to debug: `python excel_reader.py your_file.xlsx`

### Performance Tuning

For large Excel files or many regulations:
- Increase the timeout values in `regulation_pipeline.py`
- Adjust `DOWNLOAD_DELAY` in the spider settings
- Monitor memory usage and adjust `MEMUSAGE_LIMIT_MB`

## Testing

### Test Individual Components

```bash
# Test Excel reader
python excel_reader.py test_regulations.xlsx

# Test HTML downloader
python html_downloader.py https://example.com

# Test PDF downloader  
python pdf_downloader.py https://example.com/document.pdf

# Test spider directly
scrapy crawl regulation -a regulation_name="Test" -a start_urls="https://example.com"
```

### Sample Excel File
Create a test Excel file with a few regulations to verify the pipeline works before running on your full dataset.

## Advanced Usage

### Custom Depth Limits
Modify `DEPTH_LIMIT` in `regulation_spider.py` to change crawling depth.

### Custom Output Directory
Modify `output_dir` in `RegulationPipeline.__init__()` to change the output location.

### Additional File Types
The spider can be extended to handle additional file types by modifying the file detection patterns.

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review the generated log files and error messages
3. Verify FlareSolverr is accessible at http://localhost:8191/v1