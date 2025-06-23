BOT_NAME = "documents"
SPIDER_MODULES = ["documents.spiders"]
NEWSPIDER_MODULE = "documents.spiders"

# -- Conservative settings for government sites ---------------------------
CONCURRENT_REQUESTS = 1            # Single request at a time
CONCURRENT_REQUESTS_PER_DOMAIN = 1  # One request per domain
DOWNLOAD_DELAY = 15                # 15 second delay between requests
RANDOMIZE_DOWNLOAD_DELAY = 0.5     # Randomize delay ±50%

# Enhanced retry and connection settings
RETRY_ENABLED = True
RETRY_TIMES = 5                    # Retry failed requests 5 times
RETRY_HTTP_CODES = [500, 502, 503, 504, 408, 429, 403]  # Retry these HTTP codes
DOWNLOAD_TIMEOUT = 300             # 5 minute timeout for large PDFs
DOWNLOAD_MAXSIZE = 100 * 1024 * 1024  # 100MB max file size

# Auto-throttling for dynamic adjustment
AUTOTHROTTLE_ENABLED = True
AUTOTHROTTLE_START_DELAY = 0.5
AUTOTHROTTLE_MAX_DELAY = 5
AUTOTHROTTLE_TARGET_CONCURRENCY = 8.0

# Allow 403 responses to be processed (for debugging)
HTTPERROR_ALLOWED_CODES = [403, 404]

# -- Real-time LLM processing ---------------------------------------------
FILES_STORE = "downloads"          # PDFs land here
ROBOTSTXT_OBEY = True

# -- FlareSolverr configuration for Cloudflare bypass -------------------
# FlareSolverr handles Cloudflare challenges via HTTP API
# No special Scrapy configuration needed - handled in spider logic

# -- Standard pipeline: Download PDFs + save metadata --------------------
ITEM_PIPELINES = {
    "scrapy.pipelines.files.FilesPipeline": 400,           # Download PDFs
    "documents.pipelines.MetaPipeline": 500,               # Save metadata with URLs
}