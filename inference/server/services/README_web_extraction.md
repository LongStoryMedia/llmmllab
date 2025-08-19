# Web Extraction Service

This service provides web content extraction capabilities using BeautifulSoup. It extracts meaningful content from URLs, follows relevant links, and synthesizes the information into a concise summary.

## Features

- Extract content from web pages using BeautifulSoup
- Recursively follow and extract content from relevant links
- Generate topic tags from user queries
- Filter links based on relevance to the original query
- Synthesize extracted content into a concise summary
- Store synthesized content in the database with embeddings for semantic search

## Architecture

The web extraction service is integrated with the search service to provide deep crawling capabilities:

1. The search service performs a web search to get initial results
2. For the most relevant result, the web extraction service is used to:
   - Extract content from the URL
   - Follow and extract content from relevant links
   - Synthesize all the extracted content

## Usage

```python
from models import UserConfig
from server.services.web_extraction_service import WebExtractionService

# Initialize the service with user configuration
user_config = UserConfig(...)
web_extraction_service = WebExtractionService(user_config)

# Extract content from a URL
synthesis = await web_extraction_service.extract_content_from_url(
    url="https://example.com",
    query="How does example.com work?",
    conversation_id=123  # Required integer conversation ID
)

# The synthesis object contains:
# - urls: List of URLs that were crawled
# - topics: List of extracted topics/tags
# - synthesis: The synthesized text from all sources
```

## Algorithm

1. Create a `SearchTopicSynthesis` object to store the results
2. Generate labels/tags based on the user's query using an LLM
3. Collect content from the starting URL and add it to a list of messages
4. Find links in the content and use an LLM to determine if they're relevant
5. For relevant links, recursively extract content (up to max_urls_deep)
6. Send all collected messages to the summarization pipeline
7. Store the synthesis in the database and create a memory with embeddings
8. Return the completed synthesis object

## Dependencies

- BeautifulSoup for HTML parsing
- aiohttp for asynchronous HTTP requests
- re for extracting links using regular expressions
- Pydantic models for structured data
- The system's LLM pipeline factory for AI-powered content analysis
