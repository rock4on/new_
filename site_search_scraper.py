#!/usr/bin/env python3
"""
Site-Specific Search Scraper

Uses WebSearch tool to find relevant pages within a specific website,
then scrapes those pages with AI analysis.

Example: Search for "financial regulations" only within "sec.gov"
"""

import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import List, Dict, Any
from urllib.parse import urlparse
from ai_web_scraper import AIWebScraper, ScrapedContent

# Configuration - Edit these settings
TARGET_WEBSITE = "https://www.sec.gov"
SEARCH_CRITERIA = "financial regulations compliance"
MIN_RELEVANCE = 0.3
MAX_PAGES = 10
STRATEGY = "intelligent"

# Model Configuration
PLANNING_MODEL = "gpt-4o"
SUMMARY_MODEL = "gpt-4o-mini"


class SiteSpecificSearcher:
    """Search within a specific website using WebSearch tool"""
    
    def __init__(self):
        self.output_dir = Path("site_search_results")
        self.output_dir.mkdir(exist_ok=True)
    
    def _extract_domain(self, url: str) -> str:
        """Extract clean domain from URL"""
        parsed = urlparse(url)
        domain = parsed.netloc
        
        # Remove www. prefix
        if domain.startswith('www.'):
            domain = domain[4:]
            
        return domain
    
    def search_within_site(self, website: str, search_terms: str) -> List[str]:
        """
        Use WebSearch to find pages within specific site
        
        In a real implementation, this would call:
        WebSearch(query=f"site:{domain} {search_terms}")
        
        For now, we simulate the results
        """
        domain = self._extract_domain(website)
        site_query = f"site:{domain} {search_terms}"
        
        print(f"🔍 Searching within {domain}")
        print(f"📝 Query: {site_query}")
        
        # Simulate WebSearch results - replace with actual WebSearch tool call
        if "sec.gov" in domain:
            # Realistic SEC.gov URLs that would match financial regulations search
            return [
                "https://www.sec.gov/rules-regulations",
                "https://www.sec.gov/rules-regulations/staff-guidance", 
                "https://www.sec.gov/compliance-disclosures",
                "https://www.sec.gov/compliance-disclosures/cf-disclosure-guidance",
                "https://www.sec.gov/investment/rules-guidance",
                "https://www.sec.gov/reportspubs/investor-publications",
                "https://www.sec.gov/enforce",
                "https://www.sec.gov/corpfin"
            ]
        elif "treasury.gov" in domain:
            return [
                "https://www.treasury.gov/resource-center/faqs",
                "https://www.treasury.gov/press-center/press-releases",
                "https://www.treasury.gov/initiatives/financial-stability"
            ]
        elif "federalreserve.gov" in domain:
            return [
                "https://www.federalreserve.gov/supervisionreg.htm",
                "https://www.federalreserve.gov/bankinforeg.htm",
                "https://www.federalreserve.gov/publications.htm"
            ]
        else:
            # For unknown sites, return the base URL
            return [website]
    
    async def search_and_scrape_site(
        self, 
        website: str, 
        search_terms: str,
        max_pages: int = 10,
        min_relevance: float = 0.3,
        strategy: str = "intelligent"
    ) -> List[ScrapedContent]:
        """Search within site then scrape found pages"""
        
        print(f"🎯 Target Website: {website}")
        print(f"🔍 Search Terms: {search_terms}")
        print(f"📊 Min Relevance: {min_relevance}")
        print()
        
        # Step 1: Use WebSearch to find relevant pages within the site
        relevant_urls = self.search_within_site(website, search_terms)
        
        if not relevant_urls:
            print("❌ No relevant pages found in site search")
            return []
        
        print(f"✅ Found {len(relevant_urls)} potentially relevant pages")
        
        # Step 2: Scrape each found page with AI analysis
        all_results = []
        
        # Limit pages to process
        urls_to_process = relevant_urls[:max_pages]
        
        print(f"\n📄 Processing {len(urls_to_process)} pages...")
        
        async with AIWebScraper(
            max_pages=1,  # Process one page at a time since we have specific URLs
            planning_model=PLANNING_MODEL,
            summary_model=SUMMARY_MODEL
        ) as scraper:
            
            for i, url in enumerate(urls_to_process, 1):
                print(f"\n[{i}/{len(urls_to_process)}] Processing: {url}")
                
                try:
                    # Extract content from this specific page
                    page_data = await scraper._extract_page_content(url)
                    
                    if not page_data["content"]:
                        print("   ⚠️  No content extracted")
                        continue
                    
                    # Evaluate relevance with AI
                    evaluation = await scraper._evaluate_relevance(
                        page_data["content"],
                        page_data["title"], 
                        search_terms
                    )
                    
                    relevance_score = evaluation["relevance_score"]
                    print(f"   ⭐ Relevance: {relevance_score:.2f}")
                    
                    if relevance_score >= min_relevance:
                        from ai_web_scraper import ScrapedContent
                        
                        scraped_content = ScrapedContent(
                            url=url,
                            title=page_data["title"],
                            content=page_data["content"],
                            links=page_data["links"],
                            relevance_score=relevance_score,
                            ai_summary=evaluation["summary"],
                            timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
                        )
                        all_results.append(scraped_content)
                        print(f"   ✅ Added to results")
                    else:
                        print(f"   ❌ Below relevance threshold")
                        
                except Exception as e:
                    print(f"   ❌ Error processing {url}: {e}")
                    continue
                
                # Rate limiting
                await asyncio.sleep(1)
        
        # Sort by relevance
        all_results.sort(key=lambda x: x.relevance_score, reverse=True)
        
        # Save results
        await self._save_results(all_results, website, search_terms)
        
        return all_results
    
    async def _save_results(self, results: List[ScrapedContent], website: str, search_terms: str):
        """Save results with site context"""
        
        domain = self._extract_domain(website)
        timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        
        output_data = {
            "target_website": website,
            "domain": domain,
            "search_terms": search_terms,
            "total_results": len(results),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "summary": {
                "avg_relevance": sum(r.relevance_score for r in results) / len(results) if results else 0,
                "highest_relevance": max(r.relevance_score for r in results) if results else 0,
                "total_content_words": sum(len(r.content.split()) for r in results)
            },
            "results": [
                {
                    "rank": i + 1,
                    "url": r.url,
                    "title": r.title,
                    "relevance_score": r.relevance_score,
                    "ai_summary": r.ai_summary,
                    "content_preview": r.content[:600] + "..." if len(r.content) > 600 else r.content,
                    "word_count": len(r.content.split()),
                    "timestamp": r.timestamp
                }
                for i, r in enumerate(results)
            ]
        }
        
        # Save JSON results
        json_filename = f"site_search_{domain}_{timestamp}.json"
        json_path = self.output_dir / json_filename
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        # Save readable report
        report_filename = f"report_{domain}_{timestamp}.md"
        report_path = self.output_dir / report_filename
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(f"# Site Search Report: {domain}\n\n")
            f.write(f"**Search Terms:** {search_terms}  \n")
            f.write(f"**Target Site:** {website}  \n")
            f.write(f"**Generated:** {output_data['timestamp']}  \n")
            f.write(f"**Total Results:** {len(results)}  \n\n")
            
            if results:
                f.write(f"**Average Relevance:** {output_data['summary']['avg_relevance']:.2f}  \n")
                f.write(f"**Highest Relevance:** {output_data['summary']['highest_relevance']:.2f}  \n")
                f.write(f"**Total Words Extracted:** {output_data['summary']['total_content_words']:,}  \n\n")
            
            f.write("## Results\n\n")
            
            for i, result in enumerate(results, 1):
                f.write(f"### {i}. {result.title}\n\n")
                f.write(f"**URL:** {result.url}  \n")
                f.write(f"**Relevance:** {result.relevance_score:.2f}/1.0  \n\n")
                f.write(f"**AI Summary:**  \n{result.ai_summary}\n\n")
                f.write("---\n\n")
        
        print(f"\n💾 Results saved:")
        print(f"   📊 JSON: {json_path}")
        print(f"   📄 Report: {report_path}")


async def main():
    """Main execution"""
    
    # Use config or command line args
    website = TARGET_WEBSITE
    search_terms = SEARCH_CRITERIA
    
    if len(sys.argv) >= 3:
        website = sys.argv[1]
        search_terms = " ".join(sys.argv[2:])
    elif len(sys.argv) == 2:
        search_terms = sys.argv[1]
    
    print("🚀 SITE-SPECIFIC SEARCH SCRAPER")
    print("=" * 50)
    print(f"🌐 Website: {website}")
    print(f"🔍 Search: {search_terms}")
    print(f"📄 Max pages: {MAX_PAGES}")
    print(f"📊 Min relevance: {MIN_RELEVANCE}")
    print(f"🤖 Models: {PLANNING_MODEL} + {SUMMARY_MODEL}")
    print()
    
    searcher = SiteSpecificSearcher()
    results = await searcher.search_and_scrape_site(
        website=website,
        search_terms=search_terms,
        max_pages=MAX_PAGES,
        min_relevance=MIN_RELEVANCE,
        strategy=STRATEGY
    )
    
    print(f"\n🎉 COMPLETE!")
    print(f"📊 Found {len(results)} relevant pages")
    
    if results:
        print(f"\n🏆 TOP RESULTS:")
        print("-" * 50)
        for i, result in enumerate(results[:5], 1):
            print(f"\n{i}. {result.title}")
            print(f"   ⭐ {result.relevance_score:.2f} | {result.url}")
            print(f"   💡 {result.ai_summary}")
    else:
        print("\n❌ No relevant results found. Try:")
        print("   • Lowering MIN_RELEVANCE")
        print("   • Adjusting search terms")
        print("   • Checking if the website is accessible")


if __name__ == "__main__":
    asyncio.run(main())