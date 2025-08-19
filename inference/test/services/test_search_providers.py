"""
Test the search providers functionality.
"""

import os
import pytest
from unittest import mock
from unittest.mock import MagicMock, patch, AsyncMock

from server.services.search_providers import (
    SearchProviderFactory,
    BraveSearchProviderWrapper,
    DuckDuckGoSearchProviderWrapper,
    SearxSearchProviderWrapper,
    GoogleSerperSearchProviderWrapper,
    GoogleSearchProviderWrapper,
)
from models.web_search_providers import WebSearchProviders


class TestSearchProviderFactory:
    """Test the search provider factory."""

    def test_create_provider(self):
        """Test provider creation with different types."""
        # Test BRAVE provider
        with patch.dict(os.environ, {"BRAVE_SEARCH_API_KEY": "test_key"}):
            provider = SearchProviderFactory.create_provider(
                WebSearchProviders.BRAVE, 5
            )
            assert isinstance(provider, BraveSearchProviderWrapper)

        # Test DDG provider
        provider = SearchProviderFactory.create_provider(WebSearchProviders.DDG, 5)
        assert isinstance(provider, DuckDuckGoSearchProviderWrapper)

        # Test SEARX provider
        with patch.dict(os.environ, {"SEARX_HOST": "http://test.searx.org"}):
            provider = SearchProviderFactory.create_provider(
                WebSearchProviders.SEARX, 5
            )
            assert isinstance(provider, SearxSearchProviderWrapper)

        # Test SERPER provider
        with patch.dict(os.environ, {"SERPER_API_KEY": "test_key"}):
            provider = SearchProviderFactory.create_provider(
                WebSearchProviders.SERPER, 5
            )
            assert isinstance(provider, GoogleSerperSearchProviderWrapper)

        # Test GOOGLE provider
        with patch.dict(
            os.environ,
            {"GOOGLE_SEARCH_API_KEY": "test_key", "GOOGLE_SEARCH_CX": "test_cx"},
        ):
            provider = SearchProviderFactory.create_provider(
                WebSearchProviders.GOOGLE, 5
            )
            assert isinstance(provider, GoogleSearchProviderWrapper)
