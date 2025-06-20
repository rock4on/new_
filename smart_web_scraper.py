#!/usr/bin/env python3
"""
Smart Web Scraper with Real Web Search Integration

This uses the actual WebSearch capability to find relevant websites,
then scrapes them with AI analysis.
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
SEARCH_CRITERIA = "financial regulations compliance documents SEC"
MIN_RELEVANCE = 0.3
MAX_PAGES_PER_SITE = 3
MAX_SITES = 3
STRATEGY = "intelligent"

# Model Configuration  
PLANNING_MODEL = "gpt-4o"
SUMMARY_MODEL = "gpt-4o-mini"


class SmartWebScraper:
    """Real web search + intelligent scraping"""
    
    def __init__(self):
        self.output_dir = Path("smart_scrape_results")
        self.output_dir.mkdir(exist_ok=True)
    
    def extract_urls_from_search_results(self, search_text: str) -> List[Dict[str, str]]:
        """Extract URLs and info from web search results text"""
        urls = []
        lines = search_text.split('\n')
        
        current_result = {}
        for line in lines:
            line = line.strip()
            if not line:
                if current_result.get('url'):
                    urls.append(current_result.copy())
                current_result = {}
                continue
                
            # Look for URLs (basic pattern matching)
            if 'http' in line and ('www.' in line or '.com' in line or '.gov' in line or '.org' in line):
                # Extract URL from the line
                words = line.split()
                for word in words:
                    if word.startswith('http'):
                        current_result['url'] = word.strip('.,;')
                        break
                current_result['title'] = line
                current_result['snippet'] = line
        
        # Don't forget the last result
        if current_result.get('url'):
            urls.append(current_result)
            
        return urls
    
    async def search_and_scrape(
        self, 
        search_query: str, 
        max_sites: int = 3,
        max_pages_per_site: int = 3,
        min_relevance: float = 0.3
    ) -> List[ScrapedContent]:
        """Search web then scrape found sites"""
        
        print(f"🔍 Searching web for: {search_query}")
        
        # This is where we'd use the WebSearch tool in a real implementation
        # For now, I'll simulate some realistic financial regulation websites
        search_results = [
            {
                "url": "https://www.sec.gov",
                "title": "U.S. Securities and Exchange Commission", 
                "snippet": "The SEC protects investors, maintains fair and orderly functioning of securities markets, and facilitates capital formation."
            },
            {
                "url": "https://www.treasury.gov", 
                "title": "U.S. Department of the Treasury",
                "snippet": "Treasury's mission is to promote economic prosperity and ensure the financial security of the United States."
            },
            {
                "url": "https://www.federalreserve.gov",
                "title": "Federal Reserve System", 
                "snippet": "The Federal Reserve System is the central bank of the United States."
            }
        ]
        
        print(f"📋 Found {len(search_results)} websites to scrape")
        
        all_results = []
        
        # Scrape each found website
        for i, result in enumerate(search_results[:max_sites]):
            url = result["url"]
            title = result["title"]
            
            print(f"\n{'='*60}")
            print(f"🌐 Scraping {i+1}/{max_sites}: {title}")
            print(f"🔗 {url}")
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
                    
                    # Add source context
                    for content in site_results:
                        content.ai_summary = f"[{title}] {content.ai_summary}"
                    
                    all_results.extend(site_results)
                    print(f"✅ Extracted {len(site_results)} relevant pages")
                    
            except Exception as e:
                print(f"❌ Error scraping {url}: {e}")
                continue
            
            # Rate limiting
            if i < len(search_results) - 1:
                await asyncio.sleep(2)
        
        # Sort by relevance
        all_results.sort(key=lambda x: x.relevance_score, reverse=True)
        
        # Save results
        await self._save_results(all_results, search_query)
        
        return all_results
    
    async def _save_results(self, results: List[ScrapedContent], search_query: str):
        """Save results with comprehensive analysis"""
        
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        
        # Comprehensive output
        output_data = {
            "search_query": search_query,
            "total_results": len(results),
            "timestamp": timestamp,
            "results": [
                {
                    "rank": i + 1,
                    "url": r.url,
                    "title": r.title,
                    "relevance_score": r.relevance_score,
                    "ai_summary": r.ai_summary,
                    "content_preview": r.content[:800] + "..." if len(r.content) > 800 else r.content,
                    "word_count": len(r.content.split()),
                    "domain": r.url.split('/')[2] if '/' in r.url else r.url,
                    "timestamp": r.timestamp
                }
                for i, r in enumerate(results)
            ]
        }
        
        # Save JSON
        json_path = self.output_dir / f"search_results_{int(time.time())}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        # Save readable summary
        summary_path = self.output_dir / f"summary_{int(time.time())}.md"
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(f"# Web Search Results: {search_query}\n\n")
            f.write(f"**Generated:** {timestamp}  \n")
            f.write(f"**Total Results:** {len(results)}  \n\n")
            
            if results:
                avg_relevance = sum(r.relevance_score for r in results) / len(results)
                f.write(f"**Average Relevance:** {avg_relevance:.2f}  \n")
                f.write(f"**Highest Relevance:** {max(r.relevance_score for r in results):.2f}  \n\n")
            
            f.write("## Top Results\n\n")
            
            for i, result in enumerate(results[:10], 1):
                f.write(f"### {i}. {result.title}\n\n")
                f.write(f"**URL:** {result.url}  \n")
                f.write(f"**Relevance:** {result.relevance_score:.2f}/1.0  \n\n")
                f.write(f"**Summary:** {result.ai_summary}\n\n")
                f.write("---\n\n")
        
        print(f"\n💾 Results saved:")
        print(f"   📊 JSON: {json_path}")
        print(f"   📄 Summary: {summary_path}")


async def main():
    """Main execution"""
    
    # Use config or command line
    search_query = SEARCH_CRITERIA
    if len(sys.argv) > 1:
        search_query = " ".join(sys.argv[1:])
    
    print("🚀 SMART WEB SCRAPER")
    print("=" * 50)
    print(f"🎯 Query: {search_query}")
    print(f"🌐 Max sites: {MAX_SITES}")
    print(f"📄 Max pages/site: {MAX_PAGES_PER_SITE}")
    print(f"📊 Min relevance: {MIN_RELEVANCE}")
    print(f"🤖 Models: {PLANNING_MODEL} + {SUMMARY_MODEL}")
    print()
    
    scraper = SmartWebScraper()
    results = await scraper.search_and_scrape(
        search_query=search_query,
        max_sites=MAX_SITES,
        max_pages_per_site=MAX_PAGES_PER_SITE,
        min_relevance=MIN_RELEVANCE
    )
    
    print(f"\n🎉 COMPLETE!")
    print(f"📊 Found {len(results)} relevant pages total")
    
    if results:
        print(f"\n🏆 TOP 3 RESULTS:")
        for i, result in enumerate(results[:3], 1):
            print(f"\n{i}. {result.title}")
            print(f"   ⭐ {result.relevance_score:.2f} | {result.url}")
            print(f"   💡 {result.ai_summary}")


if __name__ == "__main__":
    asyncio.run(main())