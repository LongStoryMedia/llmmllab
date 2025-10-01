#!/usr/bin/env python3
"""
Example usage of the refactored WebSearchTool with required WebSearchConfig.

This demonstrates how the simplified API works with the user configuration system
where defaults are merged at the data layer, ensuring WebSearchConfig is always complete.
"""

import asyncio
from models import WebSearchConfig
from models.default_configs import DEFAULT_WEB_SEARCH_CONFIG
from composer.tools.static.web_search_tool import (
    WebSearchTool,
    create_web_search_tool,
    create_academic_search_tool,
    create_news_search_tool,
    create_technical_search_tool,
)


async def demo_web_search_usage():
    """Demonstrate the various ways to use the WebSearchTool."""

    print("🔍 WebSearchTool Usage Examples\n")

    # 1. Default configuration using factory function
    print("1. Default Configuration:")
    default_tool = create_web_search_tool()
    print(f"   Engines: {default_tool.web_config.engines}")
    print(f"   Max Results: {default_tool.web_config.max_results}")
    print(f"   Language: {default_tool.web_config.language}\n")

    # 2. Custom configuration
    print("2. Custom Configuration:")
    custom_config = WebSearchConfig(
        **DEFAULT_WEB_SEARCH_CONFIG.model_dump(),
        engines=["google", "duckduckgo"],
        max_results=8,
        timeout=45.0,
        safesearch=0,
    )
    custom_tool = WebSearchTool(web_config=custom_config)
    print(f"   Engines: {custom_tool.web_config.engines}")
    print(f"   Max Results: {custom_tool.web_config.max_results}")
    print(f"   Timeout: {custom_tool.web_config.timeout}s")
    print(f"   Safe Search: {custom_tool.web_config.safesearch}\n")

    # 3. Specialized search tools
    print("3. Specialized Search Tools:")

    academic_tool = create_academic_search_tool()
    print(f"   Academic - Engines: {academic_tool.web_config.engines}")
    print(f"   Academic - Categories: {academic_tool.web_config.categories}")

    news_tool = create_news_search_tool()
    print(f"   News - Engines: {news_tool.web_config.engines}")
    print(f"   News - Time Range: {news_tool.web_config.time_range}")

    tech_tool = create_technical_search_tool()
    print(f"   Technical - Engines: {tech_tool.web_config.engines}")
    print(f"   Technical - Categories: {tech_tool.web_config.categories}\n")

    # 4. User configuration integration (simulated)
    print("4. User Configuration Integration:")

    # Simulate getting user config from database (with defaults already merged)
    def get_user_web_config(user_id: str) -> WebSearchConfig:
        """Simulate getting user's web search config with defaults merged."""
        # In real implementation, this would come from:
        # user_config = await storage.user_config.get_user_config(user_id)
        # return user_config.web_search

        # Simulate user preferences merged with defaults
        user_preferences = WebSearchConfig(
            **DEFAULT_WEB_SEARCH_CONFIG.model_dump(),
            # User's custom preferences
            engines=["google", "bing", "startpage"],  # User prefers privacy
            max_results=7,  # User wants more results
            language="en",  # User's language preference
            timeout=60.0,  # User has slower connection
            enable_caching=True,  # User enables caching
            cache_ttl=600,  # User wants longer cache
        )
        return user_preferences

    # Create tool with user's configuration
    user_config = get_user_web_config("user_123")
    user_tool = WebSearchTool(web_config=user_config)

    print(f"   User's Engines: {user_tool.web_config.engines}")
    print(f"   User's Max Results: {user_tool.web_config.max_results}")
    print(f"   User's Timeout: {user_tool.web_config.timeout}s")
    print(f"   User's Cache TTL: {user_tool.web_config.cache_ttl}s\n")

    # 5. Type safety demonstration
    print("5. Type Safety Benefits:")
    print("   ✅ All properties are guaranteed to exist (merged with defaults)")
    print("   ✅ Proper type checking with Pydantic validation")
    print("   ✅ No optional properties or None checks needed")
    print("   ✅ IDE autocompletion and type hints work perfectly")
    print("   ✅ Runtime validation of configuration values\n")

    # 6. Configuration validation example
    print("6. Configuration Validation:")
    try:
        invalid_config = WebSearchConfig(
            **DEFAULT_WEB_SEARCH_CONFIG.model_dump(),
            max_results=25,  # Invalid: exceeds maximum of 20
        )
        print("   ❌ This shouldn't print - validation should fail")
    except Exception as e:
        print(f"   ✅ Validation caught invalid config: {type(e).__name__}")

    try:
        invalid_safesearch = WebSearchConfig(
            **DEFAULT_WEB_SEARCH_CONFIG.model_dump(),
            safesearch=5,  # Invalid: must be 0, 1, or 2
        )
        print("   ❌ This shouldn't print - validation should fail")
    except Exception as e:
        print(f"   ✅ Validation caught invalid safesearch: {type(e).__name__}")

    print("\n✨ All examples completed successfully!")


def demonstrate_api_simplification():
    """Show how the API was simplified."""

    print("\n📋 API Simplification Summary:\n")

    print("Before (complex with optional parameters):")
    print("```python")
    print("# Multiple ways to create, confusing API")
    print("tool1 = WebSearchTool()  # Uses defaults")
    print("tool2 = WebSearchTool(engines=['google'])  # Legacy")
    print("tool3 = WebSearchTool(config={'params': {...}})  # Dictionary")
    print("tool4 = WebSearchTool(web_config=config)  # New way")
    print("```\n")

    print("After (clean, type-safe):")
    print("```python")
    print("# Single, clear way to create tools")
    print("tool = create_web_search_tool()  # Default config")
    print("tool = WebSearchTool(web_config=config)  # Custom config")
    print("```\n")

    print("Benefits:")
    print("✅ Required WebSearchConfig ensures type safety")
    print("✅ No optional parameters or complex fallback logic")
    print("✅ Direct Pydantic property access instead of dictionaries")
    print("✅ Defaults handled at data layer, not in tool logic")
    print("✅ Cleaner, more maintainable code")
    print("✅ Better IDE support and autocompletion")


if __name__ == "__main__":
    print("WebSearchTool Configuration Demo")
    print("=" * 50)

    asyncio.run(demo_web_search_usage())
    demonstrate_api_simplification()
