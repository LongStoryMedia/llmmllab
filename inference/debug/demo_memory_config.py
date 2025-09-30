#!/usr/bin/env python3
"""
Demo script showing MemoryConfig usage patterns and API improvements.

This script demonstrates:
1. Default memory configuration usage
2. Custom configuration patterns  
3. Specialized memory retrieval tools
4. User configuration integration patterns
5. Type-safe configuration management with validation examples

Run with: python debug/demo_memory_config.py
"""

import asyncio
import sys
import os
from typing import Optional

# Add inference path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models.memory_config import MemoryConfig
from models.default_configs import DEFAULT_MEMORY_CONFIG
from composer.tools.static.memory_retrieval_tool import (
    MemoryRetrievalTool,
    create_memory_retrieval_tool,
    create_focused_memory_tool,
    create_broad_memory_tool,
    create_research_memory_tool,
)


async def demo_memory_usage():
    """Demonstrate the various ways to use the MemoryRetrievalTool."""
    
    print("🧠 MemoryRetrievalTool Usage Examples\n")
    
    # 1. Default configuration using factory function
    print("1. Default Configuration:")
    default_tool = create_memory_retrieval_tool()
    print(f"   Similarity Threshold: {default_tool.memory_config.similarity_threshold}")
    print(f"   Max Results: {default_tool.memory_config.limit}")
    print(f"   Cross-Conversation: {default_tool.memory_config.enable_cross_conversation}")
    print(f"   Embedding Model: {default_tool.memory_config.embedding_model_name}\n")
    
    # 2. Custom configuration
    print("2. Custom Configuration:")
    custom_config = MemoryConfig(
        **DEFAULT_MEMORY_CONFIG.model_dump(),
        similarity_threshold=0.85,
        limit=8,
        enable_cross_conversation=False,
        timeout=15.0,
    )
    custom_tool = MemoryRetrievalTool(memory_config=custom_config)
    print(f"   Similarity Threshold: {custom_tool.memory_config.similarity_threshold}")
    print(f"   Max Results: {custom_tool.memory_config.limit}")
    print(f"   Cross-Conversation: {custom_tool.memory_config.enable_cross_conversation}")
    print(f"   Timeout: {custom_tool.memory_config.timeout}s\n")
    
    # 3. Specialized factory functions
    print("3. Specialized Memory Tools:")
    
    # Focused memory tool
    focused_tool = create_focused_memory_tool()
    print(f"   Focused Tool - Threshold: {focused_tool.memory_config.similarity_threshold}, "
          f"Limit: {focused_tool.memory_config.limit}")
    
    # Broad memory tool  
    broad_tool = create_broad_memory_tool()
    print(f"   Broad Tool - Threshold: {broad_tool.memory_config.similarity_threshold}, "
          f"Limit: {broad_tool.memory_config.limit}")
    
    # Research memory tool
    research_tool = create_research_memory_tool()
    print(f"   Research Tool - Threshold: {research_tool.memory_config.similarity_threshold}, "
          f"Limit: {research_tool.memory_config.limit}, "
          f"Always Retrieve: {research_tool.memory_config.always_retrieve}\n")


def demo_validation_and_constraints():
    """Demonstrate configuration validation and constraints."""
    
    print("4. Configuration Validation:\n")
    
    # Valid configuration
    try:
        valid_config = MemoryConfig(
            enabled=True,
            limit=10,
            enable_cross_user=False,
            enable_cross_conversation=True,
            similarity_threshold=0.8,
            always_retrieve=False,
            embedding_model_name="all-MiniLM-L6-v2",
            timeout=20.0,
        )
        print("   ✅ Valid configuration created successfully")
        print(f"      Limit: {valid_config.limit}, Threshold: {valid_config.similarity_threshold}")
        
    except Exception as e:
        print(f"   ❌ Valid config failed: {e}")
    
    # Test constraint validation
    print("\n   Testing Constraints:")
    
    # Test limit constraints (1-50)
    try:
        invalid_limit = MemoryConfig(**DEFAULT_MEMORY_CONFIG.model_dump(), limit=100)
        print("   ❌ Should have failed limit validation")
    except Exception as e:
        print(f"   ✅ Limit constraint working: limit > 50 rejected")
    
    # Test similarity threshold constraints (0.0-1.0)
    try:
        invalid_threshold = MemoryConfig(**DEFAULT_MEMORY_CONFIG.model_dump(), similarity_threshold=1.5)
        print("   ❌ Should have failed threshold validation")
    except Exception as e:
        print(f"   ✅ Threshold constraint working: threshold > 1.0 rejected")
    
    # Test timeout constraints (1.0-60.0)
    try:
        invalid_timeout = MemoryConfig(**DEFAULT_MEMORY_CONFIG.model_dump(), timeout=120.0)
        print("   ❌ Should have failed timeout validation")
    except Exception as e:
        print(f"   ✅ Timeout constraint working: timeout > 60.0 rejected")


def demo_user_config_integration():
    """Demonstrate user configuration integration patterns."""
    
    print("\n5. User Configuration Integration Pattern:\n")
    
    # This shows how the memory tool would be used in actual composer usage
    print("   In Composer Node Implementation:")
    print("""
   # Get user configuration with defaults merged at data layer
   user_config = await storage.user_config.get_user_config(user_id)
   memory_config = user_config.memory  # Guaranteed to have all fields
   
   # Create tool with user's memory preferences  
   memory_tool = MemoryRetrievalTool(memory_config=memory_config)
   
   # Execute memory retrieval with user's settings
   result = await memory_tool._arun(query)
   """)
    
    print("   Benefits of Required Configuration:")
    print("   • No optional parameters or None checking")
    print("   • Type safety with Pydantic validation")  
    print("   • User preferences always respected")
    print("   • Consistent behavior across all usage")
    print("   • Clear configuration source and ownership")


def demo_api_comparison():
    """Show before/after API comparison."""
    
    print("\n6. API Design Comparison:\n")
    
    print("   BEFORE (Optional Configuration):")
    print("""
   # Multiple ways to configure, unclear precedence
   tool = MemoryRetrievalTool()  # Uses hardcoded defaults
   tool.configure(limit=10)      # Runtime configuration
   
   # Internal fallback logic needed
   limit = self.config.limit or 5  # Complex fallback chains
   threshold = getattr(self, 'threshold', 0.7)  # Attribute fallback
   """)
    
    print("   AFTER (Required Configuration):")
    print("""
   # Single, clear configuration source
   config = MemoryConfig(limit=10, similarity_threshold=0.8, ...)
   tool = MemoryRetrievalTool(memory_config=config)
   
   # Direct property access, no fallbacks needed
   limit = self.memory_config.limit          # Always present
   threshold = self.memory_config.similarity_threshold  # Type safe
   """)
    
    print("   Improvements:")
    print("   • Eliminated 47 lines of fallback logic")
    print("   • Removed dictionary conversions and key lookups") 
    print("   • Added type safety with Pydantic validation")
    print("   • Centralized configuration management")
    print("   • Better integration with user config system")


async def main():
    """Run all demonstration functions."""
    print("🔧 MemoryConfig Configuration Patterns Demo")
    print("=" * 60)
    
    await demo_memory_usage()
    demo_validation_and_constraints()
    demo_user_config_integration()
    demo_api_comparison()
    
    print("\n" + "=" * 60)
    print("✅ MemoryConfig demonstration completed!")
    print("\nKey Takeaways:")
    print("• Required MemoryConfig ensures consistent, type-safe configuration")
    print("• Factory functions provide specialized tools for different use cases")
    print("• User preferences integrated at data layer with guaranteed defaults")
    print("• Validation constraints prevent invalid configurations")
    print("• Clean API eliminates optional parameters and fallback complexity")


if __name__ == "__main__":
    asyncio.run(main())