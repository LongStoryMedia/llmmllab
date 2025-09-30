# Web Search Configuration Guide

This guide explains how to configure the comprehensive SearxNG web search system in LLM ML Lab.

## Overview

The web search system uses SearxNG as the backend and provides a comprehensive configuration system through `WebSearchConfig`. Users can customize search engines, categories, timeouts, caching, and more.

## Configuration Schema

The `WebSearchConfig` supports the following options:

```python
class WebSearchConfig:
    enabled: bool = True                    # Enable/disable web search
    auto_detect: bool = True               # Auto-detect search needs
    max_results: int = 5                   # Number of results (1-20)
    include_results: bool = True           # Include full results in response
    max_urls_deep: int = 3                 # Deep crawl limit (0-10)
    
    # SearxNG Configuration
    engines: List[str]                     # Search engines to use
    categories: List[str]                  # Search categories
    language: str = "en"                   # Result language
    safesearch: int = 1                    # Safe search (0=off, 1=moderate, 2=strict)
    time_range: str = ""                   # Time filter ("", "day", "week", "month", "year")
    
    # Infrastructure
    searx_host: str = ""                   # SearxNG host (defaults to SEARX_HOST env var)
    timeout: float = 30.0                  # Search timeout (5-120 seconds)
    user_agent: str = "LLMMLLab-WebSearch/1.0"
    
    # Performance
    enable_caching: bool = True            # Enable result caching
    cache_ttl: int = 300                   # Cache time-to-live (60-3600 seconds)
```

## Default Configuration

The system provides sane defaults optimized for general web search:

```python
DEFAULT_WEB_SEARCH_CONFIG = WebSearchConfig(
    enabled=True,
    auto_detect=True,
    max_results=5,
    include_results=True,
    max_urls_deep=3,
    engines=["google", "bing", "duckduckgo", "startpage"],
    categories=["general"],
    language="en",
    safesearch=1,
    time_range="",
    timeout=30.0,
    user_agent="LLMMLLab-WebSearch/1.0",
    enable_caching=True,
    cache_ttl=300,
)
```

## Usage Examples

### Basic Web Search Tool

```python
from composer.tools.static.web_search_tool import WebSearchTool

# Use default configuration
tool = WebSearchTool()
result = await tool._arun("machine learning trends 2025")
```

### Custom Configuration

```python
from models import WebSearchConfig
from composer.tools.static.web_search_tool import WebSearchTool

# Custom configuration
custom_config = WebSearchConfig(
    engines=["google", "duckduckgo"],
    max_results=10,
    timeout=60.0,
    safesearch=0,
    categories=["general", "science"],
)

tool = WebSearchTool(web_config=custom_config)
result = await tool._arun("quantum computing research")
```

### Specialized Search Tools

The system provides pre-configured specialized search tools:

```python
from composer.tools.static.web_search_tool import (
    create_academic_search_tool,
    create_news_search_tool,
    create_technical_search_tool,
    create_shopping_search_tool
)

# Academic research tool (Google Scholar, ArXiv, CrossRef)
academic_tool = create_academic_search_tool()
papers = await academic_tool._arun("neural networks deep learning")

# News and current events (Google News, Bing News, Reddit)
news_tool = create_news_search_tool()
news = await news_tool._arun("artificial intelligence regulations")

# Technical documentation (GitHub, Stack Overflow)
tech_tool = create_technical_search_tool()
code = await tech_tool._arun("python asyncio best practices")

# Shopping and products (Google Shopping, Amazon, eBay)
shopping_tool = create_shopping_search_tool()
products = await shopping_tool._arun("gaming laptops 2025")
```

## Available Search Engines

SearxNG supports many search engines. Here are the most useful categories:

### General Web Search

- `google` - Most comprehensive results
- `bing` - Microsoft's search engine
- `duckduckgo` - Privacy-focused search
- `startpage` - Google results without tracking
- `yahoo` - Yahoo search results
- `yandex` - Russian search engine

### Academic & Research

- `google_scholar` - Academic papers and citations
- `arxiv` - Pre-print research papers
- `crossref` - Academic publication metadata
- `semantic_scholar` - AI-powered academic search

### News & Media

- `google_news` - Google News aggregation
- `bing_news` - Microsoft News
- `yahoo_news` - Yahoo News
- `reddit` - Community discussions

### Technical & Development

- `github` - Code repositories and issues
- `stackoverflow` - Programming Q&A
- `gitlab` - GitLab repositories

### Shopping & Commerce

- `google_shopping` - Google Shopping results
- `bing_shopping` - Microsoft Shopping
- `amazon` - Amazon product search  
- `ebay` - eBay auction listings

### Media & Content

- `youtube` - Video search
- `vimeo` - Video platform
- `soundcloud` - Music and audio
- `flickr` - Photo search

## Search Categories

Categories help filter results by content type:

- `general` - General web search
- `news` - News and current events
- `science` - Scientific and academic content
- `it` - Information technology and programming
- `shopping` - Products and e-commerce
- `images` - Image search results
- `videos` - Video content
- `music` - Audio and music
- `files` - File downloads
- `social` - Social media content

## Performance Optimization

### Caching

Enable caching for frequently searched terms:

```python
config = WebSearchConfig(
    enable_caching=True,
    cache_ttl=600,  # 10 minute cache
)
```

### Timeouts

Configure timeouts based on your needs:

```python
config = WebSearchConfig(
    timeout=45.0,  # 45 second timeout
    max_results=3,  # Fewer results for faster response
)
```

### Engine Selection

Choose engines based on your use case:

```python
# Fast, privacy-focused
fast_config = WebSearchConfig(
    engines=["duckduckgo", "startpage"]
)

# Comprehensive coverage
comprehensive_config = WebSearchConfig(
    engines=["google", "bing", "duckduckgo", "yahoo"]
)

# Specialized academic search
academic_config = WebSearchConfig(
    engines=["google_scholar", "arxiv", "crossref"],
    categories=["science"]
)
```

## Environment Variables

The system respects these environment variables:

- `SEARX_HOST` - SearxNG server URL (required)
- Search configuration can override this via `searx_host` parameter

## Integration with User Configuration

The `WebSearchConfig` integrates with the user configuration system:

```python
from db import storage
from models import WebSearchConfig

# Get user's search configuration
user_config = await storage.user_config.get_user_config(user_id)
search_config = user_config.web_search

# Create tool with user's preferences
tool = WebSearchTool(web_config=search_config)
```

This allows users to customize their search preferences through the UI, which will be automatically applied to all web search operations.

## Best Practices

1. **Engine Selection**: Use 2-4 engines for good coverage without excessive latency
2. **Timeouts**: Set realistic timeouts (30-60 seconds) based on your infrastructure
3. **Caching**: Enable caching for production environments to improve performance  
4. **Categories**: Use specific categories when you know the content type
5. **Safe Search**: Configure based on your content policies and user needs
6. **Results Limit**: Balance comprehensiveness with response time (3-10 results typical)

## Troubleshooting

### No Results Returned

1. Check that SearxNG is running and accessible
2. Verify `SEARX_HOST` environment variable
3. Test with different engines
4. Check search query formatting

### Slow Performance

1. Reduce `max_results` limit
2. Lower `timeout` value  
3. Use fewer search engines
4. Enable caching with appropriate TTL

### Engine Errors

1. Some engines may be unavailable or rate-limited
2. Use fallback engines in your configuration
3. Check SearxNG server logs for engine-specific issues

## Migration from Legacy Configuration

If you're upgrading from the legacy system:

```python
# Old way
tool = WebSearchTool(
    engines=["google", "bing"],
    config={"params": {"safesearch": 0}}
)

# New way  
config = WebSearchConfig(
    engines=["google", "bing"],
    safesearch=0
)
tool = WebSearchTool(web_config=config)
```

The new system provides better type safety, validation, and integration with user preferences while maintaining backward compatibility.