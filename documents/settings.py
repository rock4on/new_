BOT_NAME = "documents"
SPIDER_MODULES = ["documents.spiders"]
NEWSPIDER_MODULE = "documents.spiders"

# -- High-performance global settings -------------------------------------
CONCURRENT_REQUESTS = 32           # Total concurrent requests
CONCURRENT_REQUESTS_PER_DOMAIN = 16  # Per domain concurrency
DOWNLOAD_DELAY = 0.5               # Reduced delay
RANDOMIZE_DOWNLOAD_DELAY = 0.5     # Randomize delay ±50%

# Auto-throttling for dynamic adjustment
AUTOTHROTTLE_ENABLED = True
AUTOTHROTTLE_START_DELAY = 0.5
AUTOTHROTTLE_MAX_DELAY = 5
AUTOTHROTTLE_TARGET_CONCURRENCY = 8.0

# -- Real-time LLM processing ---------------------------------------------
FILES_STORE = "downloads"          # PDFs land here
ROBOTSTXT_OBEY = True

# -- JSON-only pipeline: In-memory analysis, no PDF storage --------------
ITEM_PIPELINES = {
    "documents.pipelines.SmartDownloadPipeline": 400,      # Download, analyze, save JSON only
    # FilesPipeline disabled - no PDF files saved to disk
}