"""Debug tool schema sizes to find the massive token explosion source."""

import json
from composer.graph.state import WorkflowState 
from runner.pipelines.llamacpp.base_llamacpp import BaseLlamaCppPipeline
from models.model_profile import ModelProfile, ModelParameters
from models.complexity_level import ComplexityLevel


async def debug_tool_schemas():
    """Debug tool schema sizes to find source of token explosion."""
    print("🔍 Debugging tool schema sizes...")
    
    # Create a basic model profile (similar to E2E test)
    profile = ModelProfile(
        model_name="qwen3-30b-a3b-q4-k-m",
        provider="ollama",
        parameters=ModelParameters(
            temperature=0.2,
            max_tokens=8192,
            top_p=0.9,
            repeat_penalty=1.1,
            num_ctx=100000  # 100K context like E2E test
        ),
        complexity_level=ComplexityLevel.SPECIALIZED,
        context_window=100000,
        supports_tools=True,
        supports_streaming=True
    )
    
    # Create pipeline
    pipeline = BaseLlamaCppPipeline(profile=profile)
    
    # Get some example tools (whatever composer would normally bind)
    try:
        # Create a mock state with tools
        state = WorkflowState(
            request_id="debug-test",
            user_id="test-user",
            conversation_id="test-conv",
            static_tools=[],
            dynamic_tools=[]
        )
        
        # Try to get actual tools from composer if possible
        from composer.nodes.static_tools import StaticToolsNode
        static_node = StaticToolsNode()
        
        # Mock minimal intent classification for static tools
        from models.intent_classification import IntentClassification
        from models.analysis_depth import AnalysisDepth
        
        state.intent_classification = IntentClassification(
            intent="general_query",
            confidence=0.95,
            analysis_depth=AnalysisDepth.MINIMAL,
            requires_web_search=False,
            requires_deep_analysis=False,
            requires_code_analysis=False
        )
        
        # Get static tools
        state = await static_node(state)
        print(f"📊 Got {len(state.static_tools or [])} static tools")
        
        if state.static_tools:
            # Convert tools to llama format
            converted_tools = pipeline._convert_tools_to_openai_format(state.static_tools)
            
            if converted_tools:
                total_tool_schema_size = 0
                
                for i, tool in enumerate(converted_tools):
                    if 'function' in tool and 'parameters' in tool['function']:
                        schema_json = json.dumps(tool['function']['parameters'])
                        schema_size = len(schema_json)
                        total_tool_schema_size += schema_size
                        
                        print(f"🔧 Tool {i+1}: {tool['function'].get('name', 'Unknown')}")
                        print(f"   Schema size: {schema_size:,} characters")
                        if schema_size > 10000:  # Flag large schemas
                            print(f"   ⚠️  LARGE SCHEMA! First 500 chars:")
                            print(f"   {schema_json[:500]}...")
                
                print(f"\n📈 Total tool schema size: {total_tool_schema_size:,} characters")
                
                # Estimate tokens (using same logic as pipeline)
                estimated_tokens = max(1, total_tool_schema_size // 3)
                print(f"🎯 Estimated tokens for tool schemas: {estimated_tokens:,}")
                
                if estimated_tokens > 35000:
                    print("🚨 FOUND THE CULPRIT! Tool schemas are massive!")
                    print("   This explains the 41K token explosion!")
                
    except Exception as e:
        print(f"❌ Error getting actual tools: {e}")
        print("📝 Creating mock tools for testing...")
        
        # Create mock tools to test schema sizes
        from langchain_core.tools import tool
        
        @tool
        def simple_tool(query: str) -> str:
            """A simple tool."""
            return f"Result for {query}"
        
        mock_tools = [simple_tool]
        converted_tools = pipeline._convert_tools_to_openai_format(mock_tools)
        
        if converted_tools:
            for tool in converted_tools:
                if 'function' in tool and 'parameters' in tool['function']:
                    schema_json = json.dumps(tool['function']['parameters'])
                    print(f"🔧 Mock tool schema size: {len(schema_json)} characters")
                    print(f"   Schema: {schema_json}")


if __name__ == "__main__":
    import asyncio
    asyncio.run(debug_tool_schemas())