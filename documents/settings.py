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
PLAYWRIGHT_DEFAULT_NAVIGATION_TIMEOUT = 60 * 1000  # 60 seconds in milliseconds
PLAYWRIGHT_LAUNCH_OPTIONS = {
    "headless": False,
    "args": [
        "--no-sandbox",
        "--disable-dev-shm-usage",
        "--disable-blink-features=AutomationControlled",
        "--disable-extensions-file-access-check",
        "--disable-extensions-http-throttling",
        "--disable-extensions-except",
        "--disable-background-timer-throttling",
        "--disable-backgrounding-occluded-windows",
        "--disable-renderer-backgrounding",
        "--disable-features=TranslateUI,BlinkGenPropertyTrees",
        "--disable-ipc-flooding-protection",
        "--enable-features=NetworkService,NetworkServiceLogging",
        "--force-color-profile=srgb",
        "--metrics-recording-only",
        "--no-first-run",
        "--no-default-browser-check",
        "--no-pings",
        "--password-store=basic",
        "--use-mock-keychain",
        "--export-tagged-pdf"
    ]
}

# Additional context settings for stealth
PLAYWRIGHT_CONTEXTS = {
    "default": {
        "viewport": {"width": 1920, "height": 1080},
        "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
        "locale": "en-US",
        "timezone_id": "America/New_York",
        "permissions": ["geolocation"],
        "extra_http_headers": {
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept-Encoding": "gzip, deflate, br",
            "DNT": "1",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
        }
    }
}

# -- Standard pipeline: Download PDFs + save metadata --------------------
ITEM_PIPELINES = {
    "scrapy.pipelines.files.FilesPipeline": 400,           # Download PDFs
    "documents.pipelines.MetaPipeline": 500,               # Save metadata with URLs
}