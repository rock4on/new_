BOT_NAME = "documents"
SPIDER_MODULES = ["documents.spiders"]
NEWSPIDER_MODULE = "documents.spiders"

# -- scrapy-llm ------------------------------------------------------------
LLM_RESPONSE_MODEL = "documents.models.RegulationModel"
DOWNLOADER_MIDDLEWARES = {
    # add Playwright handler first if you need JS rendering
    # "scrapy_playwright.handler.ScrapyPlaywrightDownloadHandler": 543,
    "scrapy_llm.handler.LlmExtractorMiddleware": 550,
}

# Use GPT-4o-mini (fast) – change to any OpenAI-compatible name you like
LLM_MODEL = "gpt-4o-mini"
LLM_MODEL_TEMPERATURE = 0.0

# -- files & politeness ----------------------------------------------------
FILES_STORE = "downloads"          # PDFs land here
ROBOTSTXT_OBEY = True
DOWNLOAD_DELAY = 1
CONCURRENT_REQUESTS_PER_DOMAIN = 4

# -- pipelines -------------------------------------------------------------
ITEM_PIPELINES = {
    "documents.pipelines.MetaPipeline": 300,
    "scrapy.pipelines.files.FilesPipeline": 400,
}