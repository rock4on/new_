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

# -- Playwright configuration for Cloudflare bypass -------------------
DOWNLOAD_HANDLERS = {
    "http": "scrapy_playwright.handler.ScrapyPlaywrightDownloadHandler",
    "https": "scrapy_playwright.handler.ScrapyPlaywrightDownloadHandler",
}

# Set async reactor (required for Playwright)
TWISTED_REACTOR = "twisted.internet.asyncioreactor.AsyncioSelectorReactor"

# Playwright browser settings
PLAYWRIGHT_BROWSER_TYPE = "chromium"
PLAYWRIGHT_DEFAULT_NAVIGATION_TIMEOUT = 30 * 1000  # 30 seconds in milliseconds
PLAYWRIGHT_LAUNCH_OPTIONS = {
    "headless": True,
    "args": [
        "--no-sandbox",
        "--disable-dev-shm-usage",
        "--disable-gpu",
        "--disable-web-security",
        "--disable-features=VizDisplayCompositor"
    ]
}

# -- Standard pipeline: Download PDFs + save metadata --------------------
ITEM_PIPELINES = {
    "scrapy.pipelines.files.FilesPipeline": 400,           # Download PDFs
    "documents.pipelines.MetaPipeline": 500,               # Save metadata with URLs
}