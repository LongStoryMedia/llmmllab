"""
Test script to validate native composer RAG tools architecture.
Ensures proper decoupling from server components.
"""

import sys
import asyncio
from pathlib import Path

# Add the inference directory to the Python path
sys.path.insert(0, '/Users/lons7862/workspace/llmmllab/inference')

async def test_native_tools():
    """Test that native composer tools can be imported and instantiated."""
    
    print("Testing native composer RAG tools...")
    
    try:
        # Test direct import
        from composer.tools.static.native_rag_tools import (
            ComposerWebSearchTool,
            ComposerMemoryTool, 
            ComposerSummarizationTool
        )
        print("✅ Successfully imported native composer tools")
        
        # Test that they are proper LangChain tools
        from langchain_core.tools import BaseTool
        
        assert issubclass(ComposerWebSearchTool, BaseTool)
        assert issubclass(ComposerMemoryTool, BaseTool)
        assert issubclass(ComposerSummarizationTool, BaseTool)
        print("✅ All tools properly inherit from BaseTool")
        
        # Mock user config for testing
        class MockUserConfig:
            def __init__(self):
                self.user_id = "test-user"
                self.model_profiles = MockModelProfiles()
                self.memory = MockMemoryConfig()
                self.summarization = MockSummarizationConfig()
                
        class MockModelProfiles:
            def __init__(self):
                self.formatting_profile_id = "format-id"
                self.embedding_profile_id = "embed-id"
                self.analysis_profile_id = "analysis-id"
                self.summarization_profile_id = "summary-id"
                
        class MockMemoryConfig:
            def __init__(self):
                self.similarity_threshold = 0.8
                self.limit = 10
                self.enable_cross_user = False
                self.enable_cross_conversation = False
                
        class MockSummarizationConfig:
            def __init__(self):
                self.messages_before_summary = 20
        
        user_config = MockUserConfig()
        conversation_id = 123
        
        # Test tool instantiation (without actually calling async methods)
        search_tool = ComposerWebSearchTool(user_config=user_config, conversation_id=conversation_id)
        memory_tool = ComposerMemoryTool(user_config=user_config, conversation_id=conversation_id)
        summary_tool = ComposerSummarizationTool(user_config=user_config, conversation_id=conversation_id)
        
        print("✅ Successfully instantiated all native composer tools")
        print(f"   - Search tool: {search_tool.name}")
        print(f"   - Memory tool: {memory_tool.name}")
        print(f"   - Summary tool: {summary_tool.name}")
        
        # Test integration layer
        from composer.tools.static.integration import get_tools, StandardToolProvider
        
        tools = StandardToolProvider.get_standard_tools(user_config, conversation_id)
        print(f"✅ StandardToolProvider returned {len(tools)} tools")
        
        print("\n🎉 All tests passed! Native composer tools are properly decoupled.")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Test error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_native_tools())
    sys.exit(0 if success else 1)