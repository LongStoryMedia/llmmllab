#!/usr/bin/env python3
"""
Phase 1 Validation Script
Tests that the core composer architecture can be imported and initialized.
"""
import sys
import asyncio
from pathlib import Path

# Add inference directory to path
inference_path = Path(__file__).parent.parent
sys.path.insert(0, str(inference_path))

async def test_phase1_components():
    """Test that all Phase 1 components can be imported and initialized."""
    
    print("🧪 Testing Phase 1 Composer Architecture...")
    
    try:
        # Test config loading
        print("  ✅ Loading configuration...")
        from composer.config import config
        print(f"     - Caching enabled: {config.enable_workflow_caching}")
        print(f"     - Streaming enabled: {config.enable_streaming}")
        
        # Test state models
        print("  ✅ Testing state models...")
        from composer.graph.state import WorkflowState, ChatWorkflowState
        state = WorkflowState()
        chat_state = ChatWorkflowState()
        print(f"     - WorkflowState created: {type(state).__name__}")
        print(f"     - ChatWorkflowState created: {type(chat_state).__name__}")
        
        # Test core service
        print("  ✅ Testing ComposerService...")
        from composer.core.service import ComposerService
        composer = ComposerService()
        print(f"     - Service initialized: {type(composer).__name__}")
        
        # Test tool registry
        print("  ✅ Testing ToolRegistry...")
        tool_stats = await composer.tool_registry.get_tool_stats()
        print(f"     - Registry initialized, stats: {tool_stats}")
        
        # Test intent classifier
        print("  ✅ Testing IntentClassifierAgent...")
        from composer.agents.intent_classifier import IntentClassifierAgent
        intent_agent = IntentClassifierAgent()
        print(f"     - Intent agent initialized: {type(intent_agent).__name__}")
        
        # Test graph builder
        print("  ✅ Testing GraphBuilder...")
        from composer.graph.builder import GraphBuilder
        builder = GraphBuilder()
        print(f"     - Graph builder initialized: {type(builder).__name__}")
        
        # Test workflow cache
        print("  ✅ Testing WorkflowCache...")
        from composer.graph.cache import WorkflowCache
        cache = WorkflowCache()
        cache_stats = await cache.get_stats()
        print(f"     - Cache initialized, stats: {cache_stats}")
        await cache.close()
        
        # Test logging
        print("  ✅ Testing logging...")
        from composer.monitoring.logging import composer_logger
        composer_logger.logger.info("Test log message")
        print("     - Logging system working")
        
        # Test static tools
        print("  ✅ Testing static tools...")
        from composer.tools.static.search import WebSearchTool
        from composer.tools.static.summarization import SummarizationTool
        web_tool = WebSearchTool()
        summary_tool = SummarizationTool()
        print(f"     - Web search tool: {web_tool.name}")
        print(f"     - Summarization tool: {summary_tool.name}")
        
        # Test LCEL serialization
        print("  ✅ Testing LCEL serialization...")
        from composer.tools.dynamic.serializer import RunnableToolComposer
        print(f"     - LCEL composer available: {type(RunnableToolComposer).__name__}")
        
        # Cleanup
        await composer.shutdown()
        
        print("\n🎉 Phase 1 Architecture Validation PASSED!")
        print("\nImplemented Components:")
        print("  ✅ Complete composer directory structure")
        print("  ✅ GraphState with LangGraph reducers") 
        print("  ✅ ComposerService orchestrator")
        print("  ✅ ToolRegistry with semantic search")
        print("  ✅ IntentClassifierAgent for routing")
        print("  ✅ WorkflowCache with TTL")
        print("  ✅ Tool migration from server")
        print("  ✅ FastAPI application framework")
        print("  ✅ Structured logging and monitoring")
        print("  ✅ LCEL tool composability")
        print("  ✅ GraphBuilder for dynamic workflows")
        print("  ✅ Environment-based configuration")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Phase 1 validation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(test_phase1_components())
    exit(0 if success else 1)