#!/usr/bin/env python3
"""
Simple utility script to test the search service and providers.
This can be run from the command line to verify that the search providers
are working correctly.

Usage:
    python test_search_manual.py "your search query"
"""

import os
import sys
import asyncio
from typing import List, Dict
import json

# Add the parent directory to path so we can import modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models import Message, UserConfig, SearchResult, SearchResultContent
from models.message_role import MessageRole
from models.message_content import MessageContent
from models.message_content_type import MessageContentType
from models.web_search_providers import WebSearchProviders
from server.services.search_providers import (
    SearchProviderFactory,
    StandardSearchProvider,
    BraveSearchProviderWrapper,
    DuckDuckGoSearchProviderWrapper,
    SearxSearchProviderWrapper,
    GoogleSerperSearchProviderWrapper,
    GoogleSearchProviderWrapper,
)


async def test_provider(
    provider_enum: WebSearchProviders, query: str, max_results: int = 3
):
    """Test a specific search provider."""
    try:
        # Create the provider
        provider = SearchProviderFactory.create_provider(provider_enum, max_results)

        print(f"\nTesting {provider_enum.value} search provider...")
        results = await provider.search(query, max_results)

        assert isinstance(results, SearchResult)
        assert results.contents is not None

        if results:
            print(f"Found {len(results.contents)} results:")
            for i, result in enumerate(results.contents):
                print(f"\nResult {i+1}:")
                print(f"  Title: {result.title}")
                print(f"  URL: {result.url}")
                print(f"  Relevance: {result.relevance}")
                print(f"  Content snippet: {result.content[:100]}...")
        else:
            print("No results found.")

        return results
    except Exception as e:
        print(f"Error with {provider_enum.value}: {str(e)}")
        return []


async def main():
    """Run the test."""
    if len(sys.argv) < 2:
        print("Please provide a search query as a command-line argument.")
        print('Usage: python test_search_manual.py "your search query"')
        return

    query = sys.argv[1]
    print(f"Testing search providers with query: '{query}'")

    # Test available providers
    available_providers = []
    for provider in WebSearchProviders:
        provider_name = provider.value.lower()

        # Test for required environment variables
        if provider == WebSearchProviders.BRAVE:
            if os.getenv("BRAVE_SEARCH_API_KEY"):
                available_providers.append(provider)
            else:
                print(f"Skipping {provider_name} - BRAVE_SEARCH_API_KEY not set")

        elif provider == WebSearchProviders.SEARX:
            if os.getenv("SEARX_HOST"):
                available_providers.append(provider)
            else:
                print(f"Skipping {provider_name} - SEARX_HOST not set")

        elif provider == WebSearchProviders.SERPER:
            if os.getenv("SERPER_API_KEY"):
                available_providers.append(provider)
            else:
                print(f"Skipping {provider_name} - SERPER_API_KEY not set")

        elif provider == WebSearchProviders.GOOGLE:
            if os.getenv("GOOGLE_SEARCH_API_KEY") and os.getenv("GOOGLE_SEARCH_CX"):
                available_providers.append(provider)
            else:
                print(
                    f"Skipping {provider_name} - GOOGLE_SEARCH_API_KEY or GOOGLE_SEARCH_CX not set"
                )

        elif provider == WebSearchProviders.DDG:
            # DDG doesn't require any API keys
            available_providers.append(provider)

    if not available_providers:
        print("No search providers available with the current environment variables.")
        return

    print(
        f"Testing {len(available_providers)} available providers: {', '.join([p.value for p in available_providers])}"
    )

    # Run tests in parallel
    tasks = [test_provider(provider, query) for provider in available_providers]
    await asyncio.gather(*tasks)


if __name__ == "__main__":
    asyncio.run(main())
