# Deep Crawling and Synthesis

This document describes the deep crawling algorithm implemented in the search service and web extraction service.

## Overview

The deep crawling algorithm is now applied to each unique URL found in the search results, creating a separate `SearchTopicSynthesis` for each URL and storing a memory for each synthesized content.

## Implementation

The algorithm works as follows:

1. Search for relevant results using configured search providers
2. Deduplicate the search results to ensure unique URLs
3. For each unique URL (up to the configured maximum):
   - Perform deep crawling starting from that URL
   - Create a `SearchTopicSynthesis` object for the content
   - Store the synthesis as a memory in the database
   - Add a synthesized content entry to the search results

## SearchService Deep Crawling

The `search` method in `SearchService` implements the deep crawling for each URL:

```python
# Check if we should perform deep crawling
if self.user_config.web_search.max_urls_deep > 0 and contents:
    # Process each unique URL for deep crawling
    synthesized_results = []
    
    # Determine how many URLs to process (up to the max_results limit)
    urls_to_process = min(len(contents), self.user_config.web_search.max_results)
    
    for i in range(urls_to_process):
        result = contents[i]
        logger.info(f"Performing deep crawling for URL: {result.url}")
        
        # Create synthesis for this URL
        synthesis = await self.web_extraction_service.extract_content_from_url(
            result.url, 
            formatted_query,
            conversation_id
        )
        
        if synthesis:
            # Add the synthesis to our collection
            synthesized_results.append({
                "url": result.url,
                "synthesis": synthesis
            })
            
            # Add the synthesis to the results if requested
            if self.user_config.web_search.include_results:
                # Create a special search result for the synthesized content
                synthesized_content = SearchResultContent(
                    title=f"Synthesized Information from {result.url} ({len(synthesis.urls)} sources)",
                    url=result.url,
                    content=synthesis.synthesis[:200] + "...",  # Truncate for snippet
                    relevance=1.0 / (i + 1)  # Gradually decreasing relevance
                )
                # Insert at the corresponding position
                contents.insert(i, synthesized_content)
```

## WebExtractionService Processing

For each URL, the `WebExtractionService` performs the following steps:

1. Extracts topics from the query using the key points model
2. Recursively follows links from the starting URL
   - Fetches and parses content from each URL
   - Filters links based on relevance to the topics
   - Follows relevant links up to the configured depth
3. Generates a synthesis of all collected content
4. Stores the synthesis in the database
5. Creates embeddings for the synthesis
6. Stores the synthesis as a memory

## Benefits

Processing each URL separately provides several benefits:

1. More comprehensive coverage of available information
2. Independent synthesis for each starting point
3. More granular memories that can be retrieved independently
4. Better context preservation for each source
5. More relevant search results with synthesized content interspersed

## Configuration

The depth and breadth of crawling can be configured through the `WebSearchConfig`:

- `max_results`: Maximum number of search results to process
- `max_urls_deep`: Maximum depth to follow links from each starting URL
- `include_results`: Whether to include synthesized content in search results
