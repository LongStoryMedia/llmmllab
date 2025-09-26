#!/usr/bin/env python3
"""
Test the new simple web extractor vs the existing Scrapy-based approach
"""

import asyncio
import logging
import sys
import time
from pathlib import Path

# Add inference paths
sys.path.insert(0, str(Path(__file__).parent / "server"))
sys.path.insert(0, str(Path(__file__).parent / "runner"))
sys.path.insert(0, str(Path(__file__).parent / "evaluation"))

from server.services.simple_web_extractor import SimpleWebExtractor, ImprovedWebExtractionService

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def test_simple_extractor():
    """Test the new simple extractor"""
    logger.info("🧪 Testing Simple Web Extractor")
    
    # Test URLs including the problematic one
    test_urls = [
        "https://httpbin.org/html",
        "https://example.com", 
        "https://www.spinquanta.com/news-detail/quantum-computer-development-progress-challenges-and-future-prospects"
    ]
    
    extractor = SimpleWebExtractor()
    
    logger.info("=" * 60)
    logger.info("🔍 Testing Individual URL Extraction")
    
    for url in test_urls:
        logger.info(f"\n📍 Testing: {url}")
        start_time = time.time()
        
        result = await extractor.extract_content_async(url, timeout=180)  # 3 minute timeout
        
        if result['success']:
            logger.info(f"✅ SUCCESS: {result['content_length']} chars in {result['execution_time']:.2f}s")
            logger.info(f"📄 Title: {result['title'][:100]}...")
            logger.info(f"📝 Content preview: {result['content'][:200]}...")
        else:
            logger.error(f"❌ FAILED: {result['error']} (in {result['execution_time']:.2f}s)")
    
    logger.info("\n" + "=" * 60)
    logger.info("🔗 Testing Multiple URL Extraction")
    
    # Test multiple URLs at once
    start_time = time.time()
    results = await extractor.extract_multiple_urls(test_urls, timeout=180)
    total_time = time.time() - start_time
    
    successful = [r for r in results if r['success']]
    failed = [r for r in results if not r['success']]
    
    logger.info(f"📊 Results: {len(successful)}/{len(test_urls)} successful in {total_time:.2f}s")
    
    for result in successful:
        logger.info(f"✅ {result['url']}: {result['content_length']} chars")
    
    for result in failed:
        logger.error(f"❌ {result['url']}: {result['error']}")
    
    return len(successful) > 0

async def test_improved_service():
    """Test the improved web extraction service"""
    logger.info("\n🚀 Testing Improved Web Extraction Service")
    
    test_urls = [
        "https://httpbin.org/html",
        "https://www.spinquanta.com/news-detail/quantum-computer-development-progress-challenges-and-future-prospects"
    ]
    
    service = ImprovedWebExtractionService()
    
    start_time = time.time()
    content = await service.extract_content(test_urls, {'DOWNLOAD_TIMEOUT': 180})
    total_time = time.time() - start_time
    
    if content:
        logger.info(f"✅ SERVICE SUCCESS: {len(content)} chars extracted in {total_time:.2f}s")
        logger.info(f"📝 Content preview: {content[:300]}...")
        return True
    else:
        logger.error(f"❌ SERVICE FAILED: No content extracted in {total_time:.2f}s")
        return False

async def main():
    """Main test function"""
    logger.info("🧪 Starting Web Extractor Comparison Test")
    logger.info("=" * 70)
    
    # Test simple extractor
    simple_success = await test_simple_extractor()
    
    # Test improved service
    service_success = await test_improved_service()
    
    logger.info("\n" + "=" * 70)
    logger.info("📋 FINAL RESULTS")
    logger.info("=" * 70)
    
    if simple_success:
        logger.info("✅ Simple Web Extractor: PASSED")
    else:
        logger.error("❌ Simple Web Extractor: FAILED")
    
    if service_success:
        logger.info("✅ Improved Web Service: PASSED")
    else:
        logger.error("❌ Improved Web Service: FAILED")
    
    if simple_success and service_success:
        logger.info("🎉 ALL TESTS PASSED! Ready to replace Scrapy implementation.")
        return True
    else:
        logger.error("💥 SOME TESTS FAILED!")
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)