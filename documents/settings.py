BOT_NAME = "documents"
SPIDER_MODULES = ["documents.spiders"]
NEWSPIDER_MODULE = "documents.spiders"

# -- Conservative settings for government sites ---------------------------
CONCURRENT_REQUESTS = 1            # Single request at a time
CONCURRENT_REQUESTS_PER_DOMAIN = 1  # One request per domain
DOWNLOAD_DELAY = 10                # 10 second delay between requests
RANDOMIZE_DOWNLOAD_DELAY = 0.5     # Randomize delay ±50%

# Auto-throttling for dynamic adjustment
AUTOTHROTTLE_ENABLED = True
AUTOTHROTTLE_START_DELAY = 0.5
AUTOTHROTTLE_MAX_DELAY = 5
AUTOTHROTTLE_TARGET_CONCURRENCY = 8.0

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