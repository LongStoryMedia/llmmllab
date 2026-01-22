# URL Deduplication in Web Search and Content Extraction

This document describes the URL deduplication implementation in the search and web extraction services.

## Overview

URL deduplication is implemented at two levels:

1. In the `SearchService` to ensure that search results contain unique URLs
2. In the `WebExtractionService` to prevent duplicate URLs from being added to the synthesis object

## SearchService Deduplication

In the `search` method of `SearchService`, we implement URL deduplication using a set to track seen URLs:

```python
# Filter contents to ensure unique URLs
unique_contents = []
seen_urls = set()

for content in contents:
    if content.url not in seen_urls:
        seen_urls.add(content.url)
        unique_contents.append(content)
    else:
        logger.debug(f"Skipping duplicate URL: {content.url}")

# Replace contents with deduplicated list
contents = unique_contents
```

This ensures that each URL appears only once in the search results, even if multiple search providers return the same URL or if a single provider returns duplicate URLs.

## WebExtractionService Deduplication

The `WebExtractionService` implements deduplication in two ways:

1. It tracks visited URLs in a set to avoid re-visiting the same page:

```python
# Check if we've reached the maximum depth or already visited this URL
if depth >= self.user_config.web_search.max_urls_deep or url in self.visited_urls:
    return

# Mark URL as visited
self.visited_urls.add(url)
```

1. It ensures URLs are not duplicated in the synthesis object:

```python
# Only add the URL if it's not already in the synthesis urls list
if url not in synthesis.urls:
    synthesis.urls.append(url)
```

This prevents duplicates when recursively crawling web pages and ensures that the final synthesis object contains only unique URLs.

## Testing

Unit tests have been created for both deduplication implementations:

1. `test_search_deduplication.py` - Tests URL deduplication in SearchService
2. `test_web_extraction_deduplication.py` - Tests URL deduplication in WebExtractionService

These tests verify that:

- Search results contain only unique URLs
- The synthesis object contains only unique URLs
- Revisiting a URL doesn't add it again to the synthesis

## Benefits

URL deduplication provides several benefits:

1. Cleaner search results without redundancy
2. More efficient processing (no wasted effort on duplicate content)
3. Better representation of unique sources in the final synthesis
4. Prevents circular references when crawling linked pages

## Implementation Details

- Both implementations use simple but effective set-based tracking of URLs
- The SearchService approach focuses on filtering search results
- The WebExtractionService approach prevents both redundant crawling and duplicate storage
