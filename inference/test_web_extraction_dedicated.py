#!/usr/bin/env python3
"""
Dedicated Web Extraction Test
Tests the web extraction service with various URLs and configurations
"""

import asyncio
import logging
import sys
import time
from typing import List, Dict, Any
from pathlib import Path

# Add inference paths
sys.path.insert(0, str(Path(__file__).parent / "server"))
sys.path.insert(0, str(Path(__file__).parent / "runner"))
sys.path.insert(0, str(Path(__file__).parent / "evaluation"))

from server.services.web_extraction_service import WebExtractionService
from server.config import settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('web_extraction_test.log')
    ]
)
logger = logging.getLogger(__name__)

class WebExtractionTester:
    """Comprehensive web extraction testing"""
    
    def __init__(self):
        self.service = WebExtractionService()
        self.test_results = []
        
    def get_test_urls(self) -> List[Dict[str, Any]]:
        """Get a variety of test URLs with different characteristics"""
        return [
            {
                "url": "https://httpbin.org/html",
                "name": "Simple HTML Test",
                "expected_content": ["Herman Melville", "Moby Dick"],
                "timeout": 30
            },
            {
                "url": "https://example.com",
                "name": "Basic Static Page",
                "expected_content": ["Example Domain", "domain"],
                "timeout": 30
            },
            {
                "url": "https://www.spinquanta.com/news-detail/quantum-computer-development-progress-challenges-and-future-prospects",
                "name": "Quantum Computing Article (Failed URL)",
                "expected_content": ["quantum", "computing", "development"],
                "timeout": 300
            },
            {
                "url": "https://arxiv.org/abs/2301.00001",
                "name": "ArXiv Paper",
                "expected_content": ["abstract", "title"],
                "timeout": 60
            },
            {
                "url": "https://news.ycombinator.com",
                "name": "Hacker News",
                "expected_content": ["Hacker News", "comments"],
                "timeout": 60
            }
        ]
    
    async def test_single_url(self, test_case: Dict[str, Any]) -> Dict[str, Any]:
        """Test extraction from a single URL"""
        logger.info(f"🔍 Testing: {test_case['name']}")
        logger.info(f"📍 URL: {test_case['url']}")
        
        start_time = time.time()
        result = {
            "name": test_case["name"],
            "url": test_case["url"],
            "success": False,
            "content_length": 0,
            "execution_time": 0,
            "error": None,
            "content_preview": "",
            "expected_found": [],
            "scrapy_stats": {}
        }
        
        try:
            # Test with different timeout configurations
            custom_settings = {
                'CLOSESPIDER_TIMEOUT': test_case['timeout'],
                'DOWNLOAD_TIMEOUT': min(120, test_case['timeout'] // 2),
                'DNS_TIMEOUT': 60,
                'CONCURRENT_REQUESTS': 1,
                'DOWNLOAD_DELAY': 2.0,
                'RANDOMIZE_DOWNLOAD_DELAY': True,
                'AUTOTHROTTLE_ENABLED': True,
                'AUTOTHROTTLE_START_DELAY': 1.0,
                'AUTOTHROTTLE_MAX_DELAY': 10.0,
                'AUTOTHROTTLE_TARGET_CONCURRENCY': 1.0,
                'RETRY_TIMES': 3,
                'RETRY_HTTP_CODES': [500, 502, 503, 504, 408, 429],
                'ROBOTSTXT_OBEY': False,
                'COOKIES_ENABLED': True,
                'REDIRECT_ENABLED': True,
                'USER_AGENT': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
            }
            
            content = await self.service.extract_content([test_case["url"]], custom_settings=custom_settings)
            
            if content and len(content) > 0:
                result["success"] = True
                result["content_length"] = len(content)
                result["content_preview"] = content[:500] + "..." if len(content) > 500 else content
                
                # Check for expected content
                content_lower = content.lower()
                for expected in test_case.get("expected_content", []):
                    if expected.lower() in content_lower:
                        result["expected_found"].append(expected)
                
                logger.info(f"✅ SUCCESS: {len(content)} chars extracted")
                logger.info(f"📝 Content preview: {content[:200]}...")
                logger.info(f"🎯 Expected content found: {result['expected_found']}")
            else:
                result["error"] = "No content extracted"
                logger.warning(f"❌ FAILED: No content extracted")
                
        except Exception as e:
            result["error"] = str(e)
            logger.error(f"❌ ERROR: {e}")
        
        result["execution_time"] = time.time() - start_time
        logger.info(f"⏱️  Execution time: {result['execution_time']:.2f}s")
        
        return result
    
    async def test_scrapy_direct(self, url: str) -> Dict[str, Any]:
        """Test Scrapy directly without the service wrapper"""
        logger.info(f"🕷️  Testing Scrapy directly on: {url}")
        
        try:
            import scrapy
            from scrapy.crawler import CrawlerRunner
            from scrapy.utils.log import configure_logging
            from twisted.internet import reactor, defer
            
            class TestSpider(scrapy.Spider):
                name = 'test_spider'
                
                def __init__(self, url):
                    self.start_urls = [url]
                    self.results = []
                
                def parse(self, response):
                    content = response.text
                    self.results.append({
                        'url': response.url,
                        'status': response.status,
                        'content_length': len(content),
                        'content_preview': content[:500]
                    })
                    return self.results[0]
            
            configure_logging({'LOG_LEVEL': 'INFO'})
            runner = CrawlerRunner({
                'DOWNLOAD_TIMEOUT': 120,
                'DNS_TIMEOUT': 60,
                'CONCURRENT_REQUESTS': 1,
                'ROBOTSTXT_OBEY': False,
                'USER_AGENT': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
            })
            
            spider = TestSpider(url)
            deferred = runner.crawl(spider)
            
            # This is a simplified test - in practice you'd need proper async handling
            return {"status": "direct_test_setup", "note": "Direct scrapy test requires reactor handling"}
            
        except Exception as e:
            logger.error(f"Direct scrapy test failed: {e}")
            return {"error": str(e)}
    
    async def test_network_connectivity(self) -> Dict[str, Any]:
        """Test basic network connectivity"""
        logger.info("🌐 Testing network connectivity...")
        
        import aiohttp
        
        test_urls = [
            "https://httpbin.org/get",
            "https://www.google.com",
            "https://example.com"
        ]
        
        results = {}
        
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as session:
            for url in test_urls:
                try:
                    start_time = time.time()
                    async with session.get(url) as response:
                        status = response.status
                        content_length = len(await response.text())
                        execution_time = time.time() - start_time
                        
                        results[url] = {
                            "status": status,
                            "content_length": content_length,
                            "execution_time": execution_time,
                            "success": status == 200
                        }
                        logger.info(f"✅ {url}: {status} ({content_length} chars, {execution_time:.2f}s)")
                        
                except Exception as e:
                    results[url] = {
                        "error": str(e),
                        "success": False
                    }
                    logger.error(f"❌ {url}: {e}")
        
        return results
    
    async def run_comprehensive_test(self):
        """Run comprehensive web extraction tests"""
        logger.info("🚀 Starting Comprehensive Web Extraction Test")
        logger.info("=" * 70)
        
        # Test network connectivity first
        logger.info("📡 Phase 1: Network Connectivity Test")
        connectivity_results = await self.test_network_connectivity()
        
        # Test web extraction service
        logger.info("\n🕷️  Phase 2: Web Extraction Service Test")
        test_urls = self.get_test_urls()
        
        for test_case in test_urls:
            logger.info(f"\n{'='*50}")
            result = await self.test_single_url(test_case)
            self.test_results.append(result)
            
            # Add delay between tests to avoid rate limiting
            await asyncio.sleep(2)
        
        # Generate summary
        logger.info("\n" + "=" * 70)
        logger.info("📊 TEST SUMMARY")
        logger.info("=" * 70)
        
        successful_tests = [r for r in self.test_results if r["success"]]
        failed_tests = [r for r in self.test_results if not r["success"]]
        
        logger.info(f"✅ Successful: {len(successful_tests)}/{len(self.test_results)}")
        logger.info(f"❌ Failed: {len(failed_tests)}/{len(self.test_results)}")
        
        if successful_tests:
            logger.info("\n✅ SUCCESSFUL TESTS:")
            for result in successful_tests:
                logger.info(f"   • {result['name']}: {result['content_length']} chars ({result['execution_time']:.2f}s)")
        
        if failed_tests:
            logger.info("\n❌ FAILED TESTS:")
            for result in failed_tests:
                error = result.get('error', 'Unknown error')
                logger.info(f"   • {result['name']}: {error}")
        
        logger.info(f"\n🌐 NETWORK CONNECTIVITY:")
        for url, result in connectivity_results.items():
            status = "✅ OK" if result.get("success") else f"❌ {result.get('error', 'Failed')}"
            logger.info(f"   • {url}: {status}")
        
        # Check if the problematic URL works
        problem_url_result = next((r for r in self.test_results if "spinquanta" in r["url"]), None)
        if problem_url_result:
            if problem_url_result["success"]:
                logger.info(f"\n🎉 PROBLEM URL NOW WORKS: {problem_url_result['content_length']} chars extracted")
            else:
                logger.error(f"\n🚨 PROBLEM URL STILL FAILS: {problem_url_result.get('error')}")
                
                # Suggest improvements
                logger.info("\n💡 SUGGESTED IMPROVEMENTS:")
                logger.info("   1. Increase timeout values further (300s+)")
                logger.info("   2. Add JavaScript rendering support (Selenium/Playwright)")
                logger.info("   3. Implement more sophisticated retry logic")
                logger.info("   4. Add proxy rotation")
                logger.info("   5. Implement CAPTCHA detection and handling")
        
        return len(successful_tests) == len(self.test_results)

async def main():
    """Main test execution"""
    tester = WebExtractionTester()
    success = await tester.run_comprehensive_test()
    
    if success:
        logger.info("\n🎉 ALL TESTS PASSED!")
        sys.exit(0)
    else:
        logger.error("\n💥 SOME TESTS FAILED!")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())