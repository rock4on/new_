BOT_NAME = "documents"
SPIDER_MODULES = ["documents.spiders"]
NEWSPIDER_MODULE = "documents.spiders"

# -- removed scrapy-llm - using custom LLM post-processing instead

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