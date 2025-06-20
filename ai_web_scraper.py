#!/usr/bin/env python3
"""
AI Web Scraper - Intelligent site scraping with AI-powered content extraction

Usage:
    python ai_web_scraper.py "https://example.com" "search for financial regulations"
    
Or import and use programmatically:
    from ai_web_scraper import AIWebScraper
    scraper = AIWebScraper()
    results = scraper.scrape("https://example.com", "financial regulations")
"""

import asyncio
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Union
from urllib.parse import urljoin, urlparse

import aiohttp
import requests
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from playwright.async_api import async_playwright


@dataclass
class ScrapedContent:
    """Container for scraped content with metadata"""
    url: str
    title: str
    content: str
    links: List[str]
    relevance_score: float
    ai_summary: str
    timestamp: str


class AIWebScraper:
    """Intelligent web scraper with AI-powered content extraction"""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://api.openai.com/v1",
        planning_model: str = "gpt-4o",
        summary_model: str = "gpt-4o-mini",
        max_pages: int = 10,
        output_dir: str = "scraper_output"
    ):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url
        self.planning_model = planning_model
        self.summary_model = summary_model
        self.max_pages = max_pages
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        if not self.api_key:
            raise ValueError("OpenAI API key required. Set OPENAI_API_KEY env var or pass api_key parameter.")
        
        # Planning LLM for intelligent navigation and strategy
        self.planning_llm = ChatOpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            model=self.planning_model,
            temperature=0.2
        )
        
        # Summary LLM for content analysis and relevance scoring
        self.summary_llm = ChatOpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            model=self.summary_model,
            temperature=0.1
        )
        
        self.session = None
        self.browser = None
        self.scraped_urls = set()
        
    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession()
        self.playwright = await async_playwright().start()
        self.browser = await self.playwright.chromium.launch(headless=True)
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
        if self.browser:
            await self.browser.close()
        if self.playwright:
            await self.playwright.stop()
    
    def _get_site_map_urls(self, base_url: str) -> List[str]:
        """Extract potential URLs from robots.txt and sitemap.xml"""
        urls = []
        domain = f"{urlparse(base_url).scheme}://{urlparse(base_url).netloc}"
        
        # Try robots.txt
        try:
            robots_url = urljoin(domain, "/robots.txt")
            response = requests.get(robots_url, timeout=10)
            if response.status_code == 200:
                for line in response.text.split('\n'):
                    if line.startswith('Sitemap:'):
                        sitemap_url = line.split(':', 1)[1].strip()
                        urls.extend(self._parse_sitemap(sitemap_url))
        except Exception:
            pass
        
        return urls[:self.max_pages]
    
    def _parse_sitemap(self, sitemap_url: str) -> List[str]:
        """Parse sitemap.xml for URLs"""
        urls = []
        try:
            response = requests.get(sitemap_url, timeout=10)
            if response.status_code == 200:
                import re
                url_pattern = r'<loc>(.*?)</loc>'
                urls = re.findall(url_pattern, response.text)
        except Exception:
            pass
        return urls
    
    async def _extract_page_content(self, url: str) -> Dict[str, Union[str, List[str]]]:
        """Extract content from a single page using Playwright"""
        try:
            page = await self.browser.new_page()
            await page.goto(url, timeout=30000)
            await page.wait_for_load_state('networkidle', timeout=10000)
            
            # Extract title
            title = await page.title()
            
            # Extract main content (remove nav, footer, sidebar)
            content_selectors = [
                'main', 'article', '.content', '#content', '.main-content',
                '[role="main"]', '.post-content', '.entry-content'
            ]
            
            content = ""
            for selector in content_selectors:
                try:
                    element = await page.query_selector(selector)
                    if element:
                        content = await element.inner_text()
                        break
                except:
                    continue
            
            # Fallback to body if no main content found
            if not content:
                body = await page.query_selector('body')
                if body:
                    content = await body.inner_text()
            
            # Extract links
            links = []
            link_elements = await page.query_selector_all('a[href]')
            for link in link_elements:
                href = await link.get_attribute('href')
                if href:
                    full_url = urljoin(url, href)
                    if self._is_same_domain(url, full_url):
                        links.append(full_url)
            
            await page.close()
            
            return {
                "title": title,
                "content": content[:10000],  # Limit content length
                "links": list(set(links))[:20]  # Limit links
            }
            
        except Exception as e:
            print(f"Error extracting content from {url}: {e}")
            return {"title": "", "content": "", "links": []}
    
    def _is_same_domain(self, base_url: str, target_url: str) -> bool:
        """Check if target URL is from the same domain as base URL"""
        base_domain = urlparse(base_url).netloc
        target_domain = urlparse(target_url).netloc
        return base_domain == target_domain
    
    async def _evaluate_relevance(self, content: str, title: str, search_query: str) -> Dict[str, Union[float, str]]:
        """Use AI to evaluate content relevance and generate summary"""
        system_prompt = f"""
        You are an AI assistant that evaluates web content relevance.
        
        SEARCH QUERY: "{search_query}"
        
        Your task:
        1. Rate the relevance of the provided content to the search query on a scale of 0.0 to 1.0
        2. Provide a concise summary (2-3 sentences) of the content
        3. Focus on how well the content matches the search intent
        
        Respond in JSON format:
        {{
            "relevance_score": 0.0-1.0,
            "summary": "Brief summary here"
        }}
        """
        
        user_prompt = f"""
        TITLE: {title}
        
        CONTENT: {content[:3000]}  # Limit for API efficiency
        """
        
        try:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]
            
            response = await self.summary_llm.ainvoke(messages)
            result = json.loads(response.content)
            
            return {
                "relevance_score": float(result.get("relevance_score", 0.0)),
                "summary": result.get("summary", "No summary available")
            }
        except Exception as e:
            print(f"Error evaluating relevance: {e}")
            return {"relevance_score": 0.0, "summary": "Error processing content"}
    
    async def _plan_navigation(self, initial_urls: List[str], search_query: str) -> List[str]:
        """Use planning model to intelligently select URLs to explore"""
        system_prompt = f"""
        You are an AI web scraping strategist. Given a list of URLs and a search query, 
        select and prioritize the most promising URLs to explore.
        
        SEARCH QUERY: "{search_query}"
        
        Your task:
        1. Analyze the URL patterns and paths
        2. Identify URLs most likely to contain relevant content
        3. Prioritize based on URL structure, keywords, and typical site organization
        4. Return top URLs in order of priority
        
        Respond in JSON format:
        {{
            "priority_urls": ["url1", "url2", "url3", ...],
            "reasoning": "Brief explanation of selection strategy"
        }}
        """
        
        user_prompt = f"""
        Available URLs to choose from:
        {json.dumps(initial_urls[:50], indent=2)}  # Limit for API efficiency
        
        Select the most promising URLs for finding: {search_query}
        """
        
        try:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]
            
            response = await self.planning_llm.ainvoke(messages)
            result = json.loads(response.content)
            
            priority_urls = result.get("priority_urls", initial_urls[:10])
            reasoning = result.get("reasoning", "Default prioritization")
            
            print(f"🧠 Planning model reasoning: {reasoning}")
            return priority_urls
            
        except Exception as e:
            print(f"Planning error, using fallback: {e}")
            return initial_urls[:10]
    
    async def scrape(
        self,
        url: str,
        search_query: str,
        min_relevance: float = 0.3,
        strategy: str = "intelligent"
    ) -> List[ScrapedContent]:
        """
        Scrape website intelligently based on search query
        
        Args:
            url: Starting URL
            search_query: What to search for
            min_relevance: Minimum relevance score to include results
            strategy: 'intelligent' (AI-guided) or 'sitemap' (sitemap-based)
        """
        results = []
        urls_to_process = []
        
        # Determine URLs to process based on strategy
        if strategy == "sitemap":
            sitemap_urls = self._get_site_map_urls(url)
            if sitemap_urls:
                print(f"📋 Found {len(sitemap_urls)} URLs from sitemap")
                urls_to_process = await self._plan_navigation(sitemap_urls, search_query)
            else:
                urls_to_process = [url]
        else:
            urls_to_process = [url]
        
        print(f"🔍 Starting scrape of {len(urls_to_process)} URLs")
        print(f"🎯 Search query: '{search_query}'")
        print(f"📊 Min relevance: {min_relevance}")
        
        for i, current_url in enumerate(urls_to_process[:self.max_pages]):
            if current_url in self.scraped_urls:
                continue
                
            print(f"📄 Processing ({i+1}/{min(len(urls_to_process), self.max_pages)}): {current_url}")
            
            # Extract content
            page_data = await self._extract_page_content(current_url)
            
            if not page_data["content"]:
                continue
            
            # Evaluate relevance with AI
            evaluation = await self._evaluate_relevance(
                page_data["content"],
                page_data["title"],
                search_query
            )
            
            relevance_score = evaluation["relevance_score"]
            print(f"   ⭐ Relevance: {relevance_score:.2f}")
            
            if relevance_score >= min_relevance:
                scraped_content = ScrapedContent(
                    url=current_url,
                    title=page_data["title"],
                    content=page_data["content"],
                    links=page_data["links"],
                    relevance_score=relevance_score,
                    ai_summary=evaluation["summary"],
                    timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
                )
                results.append(scraped_content)
                
                # For intelligent strategy, use planning model to explore relevant links
                if strategy == "intelligent" and len(results) < self.max_pages and page_data["links"]:
                    candidate_links = [link for link in page_data["links"][:20] if link not in self.scraped_urls]
                    if candidate_links:
                        planned_links = await self._plan_navigation(candidate_links, search_query)
                        for link in planned_links[:3]:  # Add top planned links
                            if link not in self.scraped_urls and len(urls_to_process) < self.max_pages:
                                urls_to_process.append(link)
            
            self.scraped_urls.add(current_url)
            
            # Rate limiting
            await asyncio.sleep(1)
        
        # Sort by relevance
        results.sort(key=lambda x: x.relevance_score, reverse=True)
        
        # Save results
        await self._save_results(results, search_query)
        
        print(f"✅ Scraping complete! Found {len(results)} relevant pages")
        return results
    
    async def _save_results(self, results: List[ScrapedContent], search_query: str):
        """Save results to JSON file"""
        output_data = {
            "search_query": search_query,
            "total_results": len(results),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "results": [
                {
                    "url": r.url,
                    "title": r.title,
                    "relevance_score": r.relevance_score,
                    "ai_summary": r.ai_summary,
                    "content_preview": r.content[:500] + "..." if len(r.content) > 500 else r.content,
                    "links_count": len(r.links),
                    "timestamp": r.timestamp
                }
                for r in results
            ]
        }
        
        filename = f"scrape_results_{int(time.time())}.json"
        output_path = self.output_dir / filename
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Results saved to: {output_path}")


# Configuration - Edit these settings
TARGET_WEBSITE = "https://example.com"
SEARCH_CRITERIA = "financial regulations and compliance documents"
MIN_RELEVANCE = 0.3
MAX_PAGES = 15
STRATEGY = "intelligent"  # or "sitemap"

# Model Configuration
PLANNING_MODEL = "gpt-4o"  # For intelligent navigation and planning
SUMMARY_MODEL = "gpt-4o-mini"  # For content summarization and relevance scoring


async def main():
    """Main function - edit configuration above"""
    # Use command line args if provided, otherwise use config
    if len(sys.argv) >= 3:
        url = sys.argv[1]
        search_query = sys.argv[2]
        min_relevance = float(sys.argv[3]) if len(sys.argv) > 3 else MIN_RELEVANCE
    else:
        url = TARGET_WEBSITE
        search_query = SEARCH_CRITERIA
        min_relevance = MIN_RELEVANCE
        print(f"🎯 Using configured settings:")
        print(f"   Website: {url}")
        print(f"   Criteria: {search_query}")
        print(f"   Min relevance: {min_relevance}")
        print(f"   Strategy: {STRATEGY}")
        print()
    
    async with AIWebScraper(
        max_pages=MAX_PAGES,
        planning_model=PLANNING_MODEL,
        summary_model=SUMMARY_MODEL
    ) as scraper:
        results = await scraper.scrape(url, search_query, min_relevance, STRATEGY)
        
        print("\n" + "="*80)
        print("SUMMARY OF RESULTS")
        print("="*80)
        
        for i, result in enumerate(results[:5], 1):
            print(f"\n{i}. {result.title}")
            print(f"   URL: {result.url}")
            print(f"   Relevance: {result.relevance_score:.2f}")
            print(f"   Summary: {result.ai_summary}")


if __name__ == "__main__":
    asyncio.run(main())