#!/usr/bin/env python3
"""
Web Search + AI Scraper - Find and scrape relevant websites using web search

This version:
1. Searches the web for websites matching your criteria
2. Scrapes the found websites with AI analysis
3. Extracts and summarizes relevant content
"""

import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import List, Dict, Any
from ai_web_scraper import AIWebScraper, ScrapedContent

# Configuration - Edit these settings
SEARCH_CRITERIA = "financial regulations and compliance documents"
MIN_RELEVANCE = 0.3
MAX_PAGES_PER_SITE = 5
MAX_SITES = 5
STRATEGY = "intelligent"

# Model Configuration
PLANNING_MODEL = "gpt-4o"
SUMMARY_MODEL = "gpt-4o-mini"


class WebSearchAndScrape:
    """Combines web search with intelligent scraping"""
    
    def __init__(self):
        self.output_dir = Path("web_search_results")
        self.output_dir.mkdir(exist_ok=True)
    
    def search_web(self, query: str, max_results: int = 10) -> List[Dict[str, str]]:
        """
        Search the web using WebSearch tool
        Note: This is a placeholder - in actual implementation, 
        this would use the WebSearch tool from the environment
        """
        print(f"🔍 Searching web for: {query}")
        
        # Simulate web search results based on common patterns
        # In real implementation, this would call the WebSearch tool
        search_results = []
        
        if "financial" in query.lower() or "regulation" in query.lower():
            search_results = [
                {"url": "https://www.sec.gov", "title": "SEC - U.S. Securities and Exchange Commission", "snippet": "Official website of the Securities and Exchange Commission"},
                {"url": "https://www.treasury.gov", "title": "U.S. Department of the Treasury", "snippet": "Treasury Department regulations and guidance"},
                {"url": "https://www.federalreserve.gov", "title": "Federal Reserve System", "snippet": "Federal Reserve regulations and compliance"},
                {"url": "https://www.finra.org", "title": "FINRA", "snippet": "Financial Industry Regulatory Authority"},
                {"url": "https://www.fasb.org", "title": "FASB", "snippet": "Financial Accounting Standards Board"},
                {"url": "https://www.cftc.gov", "title": "CFTC", "snippet": "Commodity Futures Trading Commission"},
                {"url": "https://www.fdic.gov", "title": "FDIC", "snippet": "Federal Deposit Insurance Corporation"},
                {"url": "https://www.occ.gov", "title": "OCC", "snippet": "Office of the Comptroller of the Currency"}
            ]
        else:
            # Generic search - you could add more patterns here
            search_results = [
                {"url": "https://example.com", "title": "Example Site", "snippet": "Example search result"}
            ]
        
        print(f"📋 Found {len(search_results)} potential websites")
        return search_results[:max_results]
    
    async def scrape_search_results(
        self, 
        search_query: str, 
        max_sites: int = 5,
        max_pages_per_site: int = 5,
        min_relevance: float = 0.3
    ) -> List[ScrapedContent]:
        """Search web and scrape found sites"""
        
        # Step 1: Search the web
        search_results = self.search_web(search_query, max_sites * 2)  # Get extra results to filter
        
        if not search_results:
            print("❌ No search results found")
            return []
        
        all_scraped_content = []
        
        # Step 2: Scrape each found website
        for i, result in enumerate(search_results[:max_sites]):
            url = result["url"]
            site_title = result["title"]
            
            print(f"\n{'='*60}")
            print(f"🌐 Scraping site {i+1}/{max_sites}: {site_title}")
            print(f"🔗 URL: {url}")
            print('='*60)
            
            try:
                async with AIWebScraper(
                    max_pages=max_pages_per_site,
                    planning_model=PLANNING_MODEL,
                    summary_model=SUMMARY_MODEL
                ) as scraper:
                    site_results = await scraper.scrape(
                        url=url,
                        search_query=search_query,
                        min_relevance=min_relevance,
                        strategy=STRATEGY
                    )
                    
                    # Add site context to results
                    for content in site_results:
                        content.ai_summary = f"[From {site_title}] {content.ai_summary}"
                    
                    all_scraped_content.extend(site_results)
                    print(f"✅ Found {len(site_results)} relevant pages from {site_title}")
                    
            except Exception as e:
                print(f"❌ Error scraping {url}: {e}")
                continue
            
            # Rate limiting between sites
            if i < len(search_results) - 1:
                print("⏱️  Waiting before next site...")
                await asyncio.sleep(3)
        
        # Sort all results by relevance
        all_scraped_content.sort(key=lambda x: x.relevance_score, reverse=True)
        
        # Save comprehensive results
        await self._save_comprehensive_results(all_scraped_content, search_query)
        
        return all_scraped_content
    
    async def _save_comprehensive_results(self, results: List[ScrapedContent], search_query: str):
        """Save comprehensive results from multiple sites"""
        
        # Group results by website
        sites_data = {}
        for result in results:
            domain = result.url.split('/')[2]  # Extract domain
            if domain not in sites_data:
                sites_data[domain] = []
            sites_data[domain].append(result)
        
        output_data = {
            "search_query": search_query,
            "total_results": len(results),
            "sites_scraped": len(sites_data),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "summary": {
                "top_sites": list(sites_data.keys()),
                "avg_relevance": sum(r.relevance_score for r in results) / len(results) if results else 0,
                "highest_relevance": max(r.relevance_score for r in results) if results else 0
            },
            "results_by_site": {}
        }
        
        # Add results grouped by site
        for domain, site_results in sites_data.items():
            output_data["results_by_site"][domain] = {
                "total_pages": len(site_results),
                "avg_relevance": sum(r.relevance_score for r in site_results) / len(site_results),
                "pages": [
                    {
                        "url": r.url,
                        "title": r.title,
                        "relevance_score": r.relevance_score,
                        "ai_summary": r.ai_summary,
                        "content_preview": r.content[:500] + "..." if len(r.content) > 500 else r.content,
                        "timestamp": r.timestamp
                    }
                    for r in site_results
                ]
            }
        
        # Also create flat list of all results
        output_data["all_results"] = [
            {
                "url": r.url,
                "title": r.title,
                "relevance_score": r.relevance_score,
                "ai_summary": r.ai_summary,
                "content_preview": r.content[:500] + "..." if len(r.content) > 500 else r.content,
                "timestamp": r.timestamp,
                "domain": r.url.split('/')[2]
            }
            for r in results
        ]
        
        # Save results
        filename = f"web_search_results_{int(time.time())}.json"
        output_path = self.output_dir / filename
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Comprehensive results saved to: {output_path}")
        
        # Create summary report
        summary_path = self.output_dir / f"summary_{int(time.time())}.txt"
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(f"WEB SEARCH & SCRAPE SUMMARY\n")
            f.write(f"{'='*50}\n\n")
            f.write(f"Search Query: {search_query}\n")
            f.write(f"Total Results: {len(results)}\n")
            f.write(f"Sites Scraped: {len(sites_data)}\n")
            f.write(f"Average Relevance: {output_data['summary']['avg_relevance']:.2f}\n\n")
            
            f.write("TOP RESULTS:\n")
            f.write("-" * 30 + "\n")
            for i, result in enumerate(results[:10], 1):
                f.write(f"{i}. {result.title}\n")
                f.write(f"   URL: {result.url}\n")
                f.write(f"   Relevance: {result.relevance_score:.2f}\n")
                f.write(f"   Summary: {result.ai_summary}\n\n")
        
        print(f"📄 Summary report saved to: {summary_path}")


async def main():
    """Main function"""
    search_query = SEARCH_CRITERIA
    
    # Allow command line override
    if len(sys.argv) > 1:
        search_query = " ".join(sys.argv[1:])
    
    print(f"🚀 Starting Web Search + AI Scraping")
    print(f"🎯 Search Query: '{search_query}'")
    print(f"🌐 Max Sites: {MAX_SITES}")
    print(f"📄 Max Pages per Site: {MAX_PAGES_PER_SITE}")
    print(f"📊 Min Relevance: {MIN_RELEVANCE}")
    print(f"🧠 Planning Model: {PLANNING_MODEL}")
    print(f"📝 Summary Model: {SUMMARY_MODEL}")
    print()
    
    searcher = WebSearchAndScrape()
    results = await searcher.scrape_search_results(
        search_query=search_query,
        max_sites=MAX_SITES,
        max_pages_per_site=MAX_PAGES_PER_SITE,
        min_relevance=MIN_RELEVANCE
    )
    
    print(f"\n🎉 COMPLETED!")
    print(f"📊 Total relevant pages found: {len(results)}")
    
    if results:
        print(f"\n🏆 TOP 3 MOST RELEVANT RESULTS:")
        print("=" * 80)
        for i, result in enumerate(results[:3], 1):
            print(f"\n{i}. {result.title}")
            print(f"   🔗 {result.url}")
            print(f"   ⭐ Relevance: {result.relevance_score:.2f}")
            print(f"   📝 {result.ai_summary}")


if __name__ == "__main__":
    asyncio.run(main())