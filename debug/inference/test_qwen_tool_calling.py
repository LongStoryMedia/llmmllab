#!/usr/bin/env python3
"""
QwenMoE Tool Calling Test
Tests the tool calling functionality specifically for QwenMoE models.
"""

import asyncio
import json
import logging
import sys
import os
from pathlib import Path

# Add paths for modules
sys.path.append('/app/server')
sys.path.append('/app/runner')
sys.path.append('/app')

from runner.pipeline_factory import PipelineFactory
from models import Model, ModelProfile, ConversationCtx
from server.tools.integration import get_tools

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_qwen_tool_calling():
    """Test QwenMoE tool calling functionality."""
    logger.info("🧪 Starting QwenMoE Tool Calling Test")
    
    try:
        
        # Get QwenMoE model
        model = Model(
            id="qwen3-30b-a3b-q4-k-m",
            name="qwen3-30b-a3b-q4-k-m", 
            model="/models/qwen3-30b-a3b/qwen3-30b-a3b-iq4-xs-abliterated.gguf",
            task="TextToText",
            modified_at="2025-07-20",
            size=16557092832,
            digest="qwen3-30b-a3b-20250720",
            details={
                "parent_model": "Qwen/Qwen3-30B-A3B",
                "format": "gguf",
                "gguf_file": "/models/qwen3-30b-a3b/qwen3-30b-a3b-iq4-xs-abliterated.gguf",
                "family": "qwen",
                "families": ["Qwen", "MoE"],
                "parameter_size": "30.5B",
                "quantization_level": "Q4_K_M",
                "dtype": "BF16",
                "precision": "fp16",
                "specialization": "TextToText",
                "weight": 1.0
            },
            pipeline="Qwen3Pipe",
            lora_weights=[],
            provider="llama_cpp"
        )
        
        # Create model profile for tool calling
        profile = ModelProfile(
            id="test-qwen-tool-profile",
            user_id="test",
            name="QwenMoE Tool Test Profile",
            description="Profile for testing QwenMoE tool calling",
            model_name=model.id,
            parameters={
                "num_ctx": 100000,
                "temperature": 0.7,
                "top_p": 0.9,
                "max_tokens": 4000,
                "flash_attention": True
            },
            system_prompt="""You are a helpful AI assistant with access to tools.

When you need to use tools, respond with JSON in the following format:

```json
{
    "tool_calls": [
        {
            "name": "tool_name",
            "arguments": {
                "param": "value"
            }
        }
    ]
}
```

Available tools:
- web_search: Search for current information on the web
- memory_retrieval: Retrieve relevant memories
- summarization: Summarize content

CRITICAL: When the user asks for current information, recent developments, or web searches, you MUST use the web_search tool.""",
            created_at="2025-09-26T00:00:00Z",
            updated_at="2025-09-26T00:00:00Z",
            model_version="1.0",
            type=1
        )
        
        # Initialize pipeline factory
        factory = PipelineFactory()
        logger.info("✅ Pipeline factory initialized")
        
        # Create pipeline
        pipeline = await factory.create_pipeline(
            model=model,
            profile=profile,
            task="TextToText",
            expected_return_type=None
        )
        logger.info(f"✅ Pipeline created: {type(pipeline).__name__}")
        
        # Get tools
        conversation_ctx = ConversationCtx(
            user_id="test",
            conversation_id="test-conv",
            model_profile=profile
        )
        
        tools = []
        async for tool_result in get_tools(conversation_ctx):
            if isinstance(tool_result, list):
                tools.extend(tool_result)
                break
        
        logger.info(f"✅ Available tools: {[tool.name for tool in tools]}")
        
        # Test cases with increasing complexity
        test_cases = [
            {
                "name": "Simple Tool Request",
                "query": "Search for recent developments in quantum computing",
                "expected_tool": "web_search",
                "expected_patterns": ['"tool_calls"', '"name"', '"web_search"', '"arguments"']
            },
            {
                "name": "Complex Tool Request", 
                "query": "Find the latest breakthroughs in AI and machine learning from 2025",
                "expected_tool": "web_search",
                "expected_patterns": ['"tool_calls"', '"name"', '"web_search"', '"arguments"', "AI", "2025"]
            },
            {
                "name": "Direct Tool Instructions",
                "query": "Use web_search to find information about SpaceX recent launches",
                "expected_tool": "web_search", 
                "expected_patterns": ['"tool_calls"', '"name"', '"web_search"', '"arguments"', "SpaceX"]
            }
        ]
        
        results = []
        
        for i, test_case in enumerate(test_cases, 1):
            logger.info(f"\n🧪 Test Case {i}: {test_case['name']}")
            logger.info(f"📝 Query: {test_case['query']}")
            
            try:
                # Stream the response
                response_chunks = []
                full_response = ""
                
                async for chunk in pipeline.stream([{"role": "user", "content": test_case['query']}], tools):
                    response_chunks.append(chunk)
                    
                    # Extract text content
                    chunk_text = ""
                    if hasattr(chunk, "message") and chunk.message and hasattr(chunk.message, "content") and chunk.message.content:
                        if isinstance(chunk.message.content, list) and len(chunk.message.content) > 0:
                            message_content = chunk.message.content[0]
                            if hasattr(message_content, "text") and message_content.text:
                                chunk_text = str(message_content.text)
                    elif hasattr(chunk, "content") and chunk.content:
                        chunk_text = str(chunk.content)
                    
                    if chunk_text:
                        full_response += chunk_text
                
                logger.info(f"📊 Total chunks: {len(response_chunks)}")
                logger.info(f"📄 Response length: {len(full_response)} characters")
                
                # Analyze response for tool calling
                has_json_block = "```json" in full_response
                has_tool_calls_array = '"tool_calls"' in full_response
                has_expected_tool = test_case['expected_tool'] in full_response
                
                # Check for all expected patterns
                pattern_matches = {}
                for pattern in test_case['expected_patterns']:
                    pattern_matches[pattern] = pattern in full_response
                
                logger.info(f"🔍 Analysis:")
                logger.info(f"   JSON block: {has_json_block}")
                logger.info(f"   Tool calls array: {has_tool_calls_array}")
                logger.info(f"   Expected tool ({test_case['expected_tool']}): {has_expected_tool}")
                logger.info(f"   Pattern matches: {sum(pattern_matches.values())}/{len(pattern_matches)}")
                
                # Try to extract and validate JSON
                json_blocks = []
                import re
                json_pattern = r'```json\s*(\{.*?\})\s*```'
                matches = re.findall(json_pattern, full_response, re.DOTALL)
                
                valid_json_count = 0
                for match in matches:
                    try:
                        json_data = json.loads(match)
                        json_blocks.append(json_data)
                        valid_json_count += 1
                        logger.info(f"   ✅ Valid JSON found: {json_data}")
                    except json.JSONDecodeError as e:
                        logger.warning(f"   ❌ Invalid JSON: {e}")
                
                # Determine test success
                tool_calling_success = (
                    has_json_block and 
                    has_tool_calls_array and 
                    has_expected_tool and
                    valid_json_count > 0 and
                    sum(pattern_matches.values()) >= len(test_case['expected_patterns']) * 0.7  # 70% of patterns must match
                )
                
                result = {
                    "test_case": test_case['name'],
                    "success": tool_calling_success,
                    "response_length": len(full_response),
                    "has_json_block": has_json_block,
                    "has_tool_calls": has_tool_calls_array,
                    "has_expected_tool": has_expected_tool,
                    "valid_json_blocks": valid_json_count,
                    "pattern_matches": pattern_matches,
                    "response_preview": full_response[:200] + "..." if len(full_response) > 200 else full_response
                }
                results.append(result)
                
                if tool_calling_success:
                    logger.info(f"   ✅ SUCCESS: Tool calling worked correctly")
                else:
                    logger.warning(f"   ❌ FAILURE: Tool calling did not work as expected")
                    logger.info(f"   📄 Response preview: {full_response[:300]}...")
                
            except Exception as e:
                logger.error(f"   ❌ ERROR during test: {e}")
                results.append({
                    "test_case": test_case['name'],
                    "success": False,
                    "error": str(e)
                })
        
        # Final analysis
        successful_tests = sum(1 for r in results if r.get('success', False))
        total_tests = len(results)
        success_rate = (successful_tests / total_tests) * 100 if total_tests > 0 else 0
        
        logger.info(f"\n📊 FINAL RESULTS:")
        logger.info(f"   Tests passed: {successful_tests}/{total_tests}")
        logger.info(f"   Success rate: {success_rate:.1f}%")
        
        if success_rate >= 70:
            logger.info("   🎉 OVERALL SUCCESS: QwenMoE tool calling is working!")
        else:
            logger.error("   ❌ OVERALL FAILURE: QwenMoE tool calling needs improvement")
        
        # Print detailed results
        for i, result in enumerate(results, 1):
            logger.info(f"\n   Test {i} ({result['test_case']}):")
            logger.info(f"     Success: {'✅' if result.get('success', False) else '❌'}")
            if result.get('success', False):
                logger.info(f"     Response length: {result.get('response_length', 0)} chars")
                logger.info(f"     JSON blocks: {result.get('valid_json_blocks', 0)}")
                logger.info(f"     Has tool calls: {result.get('has_tool_calls', False)}")
            elif 'error' in result:
                logger.info(f"     Error: {result['error']}")
            else:
                logger.info(f"     Issues: JSON={result.get('has_json_block', False)}, ToolCalls={result.get('has_tool_calls', False)}")
        
        return success_rate >= 70
        
    except Exception as e:
        logger.error(f"❌ Test failed with error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False
    
    finally:
        # Cleanup
        logger.info("🧹 Cleanup completed")


if __name__ == "__main__":
    success = asyncio.run(test_qwen_tool_calling())
    sys.exit(0 if success else 1)