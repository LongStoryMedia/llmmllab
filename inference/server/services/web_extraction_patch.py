"""
Monkey patch for WebExtractionService to use simple web extractor
This replaces the problematic Scrapy implementation with requests + BeautifulSoup
"""

import logging
import asyncio
from typing import List, Dict, Optional
from datetime import datetime
import re

# Import the simple extractor
from server.services.simple_web_extractor import SimpleWebExtractor

logger = logging.getLogger(__name__)


def patch_web_extraction_service():
    """Monkey patch the WebExtractionService to use SimpleWebExtractor"""
    
    from server.services import web_extraction_service
    
    # Create a simple extractor instance
    simple_extractor = SimpleWebExtractor()
    
    async def improved_extract_content(self, urls: List[str], custom_settings: Optional[Dict] = None) -> str:
        """
        Improved extract_content method using SimpleWebExtractor
        
        Args:
            urls: List of URLs to extract content from  
            custom_settings: Custom settings including timeout
            
        Returns:
            Combined content from all URLs
        """
        if not urls:
            return ""
        
        # Get timeout from custom settings 
        timeout = 120
        if custom_settings:
            timeout = custom_settings.get('DOWNLOAD_TIMEOUT', 120)
            # Also check for CLOSESPIDER_TIMEOUT as fallback
            timeout = max(timeout, custom_settings.get('CLOSESPIDER_TIMEOUT', 120))
        
        logger.info(f"🔍 [IMPROVED] Extracting content from {len(urls)} URLs with timeout {timeout}s")
        
        # Extract content from all URLs
        results = await simple_extractor.extract_multiple_urls(urls, timeout)
        
        # Combine successful extractions
        all_content = []
        successful_count = 0
        
        for result in results:
            if result['success'] and result['content']:
                # Format content with source URL
                formatted_content = f"Source: {result['url']}\nTitle: {result.get('title', 'N/A')}\nContent: {result['content']}"
                all_content.append(formatted_content)
                successful_count += 1
                logger.info(f"✅ [IMPROVED] Successfully extracted {len(result['content'])} chars from {result['url']}")
            else:
                error = result.get('error', 'Unknown error')
                logger.warning(f"❌ [IMPROVED] Failed to extract from {result['url']}: {error}")
        
        if not all_content:
            logger.error("🚨 [IMPROVED] No content extracted from any URL!")
            return ""
        
        # Combine all content
        combined_content = "\n\n".join(all_content)
        
        logger.info(f"🎯 [IMPROVED] Successfully extracted content from {successful_count}/{len(urls)} URLs")
        logger.info(f"📝 [IMPROVED] Total content length: {len(combined_content)} characters")
        
        return combined_content
    
    async def improved_run_spider(self, url: str, topics: List[str], query: str, max_depth: int) -> List[Dict]:
        """
        Improved _run_spider method using SimpleWebExtractor
        
        Returns content items in the expected format for compatibility
        """
        logger.info(f"🕷️  [IMPROVED] Running improved spider for: {url}")
        
        # Use the simple extractor
        result = await simple_extractor.extract_content_async(url, timeout=180)
        
        if result['success']:
            # Convert to expected format
            content_items = [{
                "url": result['url'],
                "title": result.get('title', ''),
                "meta_description": result.get('meta_description', ''),
                "content": result['content']
            }]
            
            logger.info(f"✅ [IMPROVED] Spider extracted {len(result['content'])} chars from {url}")
            return content_items
        else:
            logger.error(f"❌ [IMPROVED] Spider failed for {url}: {result.get('error')}")
            return []
    
    # Apply the monkey patches
    logger.info("🔧 Applying WebExtractionService monkey patches...")
    
    # Patch the _run_spider method (this is the core method that needs fixing)
    if hasattr(web_extraction_service.WebExtractionService, '_run_spider'):
        setattr(web_extraction_service.WebExtractionService, '_run_spider', improved_run_spider)
        logger.info("✅ Patched WebExtractionService._run_spider")
    
    logger.info("🎉 WebExtractionService monkey patching complete!")


# Apply the patch when this module is imported
patch_web_extraction_service()