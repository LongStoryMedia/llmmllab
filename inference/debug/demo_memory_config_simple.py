#!/usr/bin/env python3
"""
Simplified demo showing MemoryConfig patterns without dependencies.

This script demonstrates the configuration patterns and validation
without importing the full composer/runner stack.
"""

import sys
import os

# Add inference path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models.memory_config import MemoryConfig
from models.default_configs import DEFAULT_MEMORY_CONFIG


def demo_memory_config_patterns():
    """Demonstrate MemoryConfig usage patterns."""

    print("🧠 MemoryConfig Usage Patterns Demo\n")

    # 1. Default configuration
    print("1. Default Configuration:")
    default_config = DEFAULT_MEMORY_CONFIG
    print(f"   Enabled: {default_config.enabled}")
    print(f"   Limit: {default_config.limit}")
    print(f"   Similarity Threshold: {default_config.similarity_threshold}")
    print(f"   Cross-Conversation: {default_config.enable_cross_conversation}")
    print(f"   Cross-User: {default_config.enable_cross_user}")
    print(f"   Embedding Model: {default_config.embedding_model_name}")
    print(f"   Timeout: {default_config.timeout}s\n")

    # 2. Custom configuration
    print("2. Custom Configuration:")
    custom_config = MemoryConfig(
        enabled=True,
        limit=10,
        enable_cross_user=False,
        enable_cross_conversation=True,
        similarity_threshold=0.85,
        always_retrieve=False,
        embedding_model_name="custom-embedding-model",
        timeout=20.0,
    )
    print(f"   Custom Limit: {custom_config.limit}")
    print(f"   Custom Threshold: {custom_config.similarity_threshold}")
    print(f"   Custom Model: {custom_config.embedding_model_name}")
    print(f"   Custom Timeout: {custom_config.timeout}s\n")

    # 3. Configuration inheritance/merging pattern
    print("3. Configuration Override Pattern:")

    # Create base dict and modify specific fields
    focused_dict = DEFAULT_MEMORY_CONFIG.model_dump()
    focused_dict.update(
        {
            "similarity_threshold": 0.9,  # Higher precision
            "limit": 3,  # Fewer results
            "enable_cross_conversation": False,  # Stay focused
        }
    )
    focused_config = MemoryConfig(**focused_dict)
    print(
        f"   Focused - Threshold: {focused_config.similarity_threshold}, Limit: {focused_config.limit}"
    )

    broad_dict = DEFAULT_MEMORY_CONFIG.model_dump()
    broad_dict.update(
        {
            "similarity_threshold": 0.6,  # Lower threshold
            "limit": 15,  # More results
            "enable_cross_conversation": True,  # Broader search
        }
    )
    broad_config = MemoryConfig(**broad_dict)
    print(
        f"   Broad - Threshold: {broad_config.similarity_threshold}, Limit: {broad_config.limit}"
    )

    research_dict = DEFAULT_MEMORY_CONFIG.model_dump()
    research_dict.update(
        {
            "always_retrieve": True,  # Always provide context
            "timeout": 30.0,  # Longer timeout for thorough search
        }
    )
    research_config = MemoryConfig(**research_dict)
    print(
        f"   Research - Always Retrieve: {research_config.always_retrieve}, Timeout: {research_config.timeout}s\n"
    )


def demo_validation():
    """Demonstrate configuration validation."""

    print("4. Configuration Validation:\n")

    # Test valid configuration
    try:
        valid_config = MemoryConfig(
            enabled=True,
            limit=10,
            enable_cross_user=False,
            enable_cross_conversation=True,
            similarity_threshold=0.8,
            always_retrieve=False,
            embedding_model_name="test-model",
            timeout=15.0,
        )
        print("   ✅ Valid configuration created successfully")
    except Exception as e:
        print(f"   ❌ Valid config failed: {e}")

    # Test validation constraints
    validation_tests = [
        ("Limit too high", {"limit": 100}),
        ("Limit too low", {"limit": 0}),
        ("Threshold too high", {"similarity_threshold": 1.5}),
        ("Threshold too low", {"similarity_threshold": -0.1}),
        ("Timeout too high", {"timeout": 120.0}),
        ("Timeout too low", {"timeout": 0.5}),
    ]

    for test_name, override in validation_tests:
        try:
            test_dict = DEFAULT_MEMORY_CONFIG.model_dump()
            test_dict.update(override)
            test_config = MemoryConfig(**test_dict)
            print(f"   ❌ {test_name}: Should have failed validation")
        except Exception as e:
            print(f"   ✅ {test_name}: Validation working correctly")


def demo_api_improvements():
    """Show API design improvements."""

    print("\n5. API Design Improvements:\n")

    print("   BEFORE (Optional Configuration with Hardcoded Fallbacks):")
    print(
        """
   class MemoryRetrievalTool:
       def __init__(self):
           # Hardcoded defaults scattered throughout code
           self.limit = 5  
           self.threshold = 0.7
           self.cross_conversation = True
           
       async def _arun(self, query: str):
           # Complex fallback logic
           limit = getattr(self, 'limit', 5)
           threshold = self.config.get('threshold', 0.7) if hasattr(self, 'config') else 0.7
           cross_conv = self.cross_conversation if hasattr(self, 'cross_conversation') else True
   """
    )

    print("   AFTER (Required Configuration):")
    print(
        """
   class MemoryRetrievalTool:
       def __init__(self, memory_config: MemoryConfig):
           self.memory_config = memory_config  # Always complete, validated
           
       async def _arun(self, query: str):
           # Direct property access, no fallbacks needed
           limit = self.memory_config.limit
           threshold = self.memory_config.similarity_threshold
           cross_conv = self.memory_config.enable_cross_conversation
   """
    )

    print("   Benefits:")
    print("   • Eliminated hardcoded defaults and fallback chains")
    print("   • Type safety with Pydantic validation at construction")
    print("   • User preferences always respected via data layer defaults")
    print("   • Consistent behavior across all tool instances")
    print("   • Clear configuration source and ownership")


def demo_user_integration():
    """Show user configuration integration."""

    print("\n6. User Configuration Integration:\n")

    print("   Pattern in Composer Nodes:")
    print(
        """
   # Memory config retrieved with user preferences + defaults
   user_config = await storage.user_config.get_user_config(user_id) 
   memory_config = user_config.memory  # Guaranteed complete MemoryConfig
   
   # Create tool with user's preferences
   memory_tool = MemoryRetrievalTool(memory_config=memory_config)
   
   # Tool uses user's similarity threshold, limits, etc.
   results = await memory_tool._arun("machine learning concepts")
   """
    )

    print("   Data Layer Responsibilities:")
    print("   • Merge user preferences with system defaults")
    print("   • Ensure all required fields are populated")
    print("   • Validate configuration constraints")
    print("   • Provide consistent MemoryConfig objects")


def main():
    """Run all demonstrations."""
    print("🔧 MemoryConfig Pattern Demonstration")
    print("=" * 50)

    demo_memory_config_patterns()
    demo_validation()
    demo_api_improvements()
    demo_user_integration()

    print("\n" + "=" * 50)
    print("✅ MemoryConfig pattern demonstration complete!")
    print("\nKey Benefits:")
    print("• Required configuration eliminates optional parameter complexity")
    print("• Type safety prevents runtime configuration errors")
    print("• User preferences respected through data layer integration")
    print("• Consistent API pattern across all configurable tools")
    print("• Validation ensures configuration constraints are enforced")


if __name__ == "__main__":
    main()
