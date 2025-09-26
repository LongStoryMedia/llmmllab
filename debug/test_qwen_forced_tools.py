#!/usr/bin/env python3
"""Test script to force Qwen3 to use tools with very explicit instructions."""

import asyncio
from inference.runner.pipelines.txt2txt.qwen3moe import QwenLangGraphPipe
from inference.models.request_models import GenerateRequest, AvailableTool, DynamicTool
from inference.models.model_profile import ModelProfile
from inference.models.llm_model import LLMModel

async def test_forced_qwen_tools():
    """Test with extremely explicit tool calling instructions."""
    
    # Model configuration
    model = LLMModel(
        id="qwen3-30b-a3b-q4-k-m",
        name="qwen3-30b-a3b-q4-k-m", 
        model="/models/qwen3-30b-a3b/qwen3-30b-a3b-iq4-xs-abliterated.gguf",
        provider="llama_cpp",
        pipeline="Qwen3Pipe",
        modified_at="2025-07-20",
        size=16557092832,
        digest="qwen3-30b-a3b-20250720",
        details={
            "parent_model": "Qwen/Qwen3-30B-A3B",
            "gguf_file": "/models/qwen3-30b-a3b/qwen3-30b-a3b-iq4-xs-abliterated.gguf",
            "format": "gguf",
            "family": "qwen",
            "families": ["Qwen", "MoE"],
            "parameter_size": "30.5B",
            "quantization_level": "Q4_K_M",
            "dtype": "BF16",
            "specialization": "TextToText"
        },
        task="TextToText"
    )
    
    # Profile configuration
    profile = ModelProfile(
        model_name="qwen3-30b-a3b-q4-k-m",
        max_tokens=16384,
        temperature=0.1,
        top_p=0.8,
        dtype="float16",
        model_path="/models/qwen3-30b-a3b/qwen3-30b-a3b-iq4-xs-abliterated.gguf"
    )
    
    # Web search tool
    web_search_tool = AvailableTool(
        name="web_search",
        description="Search the web for current information about any topic. Use this tool whenever you need up-to-date information that might not be in your training data.",
        parameters={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query to look up current information"
                }
            },
            "required": ["query"]
        }
    )
    
    # Create pipeline
    pipeline = QwenLangGraphPipe(model, profile)
    
    # Request with very explicit instructions
    request = GenerateRequest(
        messages=[{
            "role": "user",
            "content": """IMPORTANT: You MUST use tools when asked for current information.

Question: What are the latest developments in quantum computing in 2024?

INSTRUCTIONS:
1. This question requires current information from 2024
2. You MUST use the web_search tool to get current information
3. Do NOT provide an answer without using the web_search tool first
4. Use the exact format: <tool_call>{"name": "web_search", "arguments": {"query": "latest quantum computing developments 2024"}}</tool_call>
5. This is MANDATORY - you cannot answer without using the tool"""
        }],
        model_name="qwen3-30b-a3b-q4-k-m",
        max_tokens=8192,
        temperature=0.1,
        top_p=0.8,
        tools=[web_search_tool],
        tool_choice="required"  # Force tool usage
    )
    
    print("Testing Qwen3 with forced tool calling...")
    print("="*60)
    
    try:
        response = await pipeline.generate(request)
        
        print(f"Status: {response.status}")
        print(f"Content length: {len(response.choices[0].message.content)}")
        print("\nContent:")
        print(response.choices[0].message.content)
        
        # Check for tool calls
        has_tool_calls = "<tool_call>" in response.choices[0].message.content
        print(f"\nHas <tool_call> tags: {has_tool_calls}")
        
        if response.choices[0].message.tool_calls:
            print(f"Tool calls detected: {len(response.choices[0].message.tool_calls)}")
            for i, tool_call in enumerate(response.choices[0].message.tool_calls):
                print(f"  Tool {i+1}: {tool_call.function.name}")
                print(f"    Arguments: {tool_call.function.arguments}")
        else:
            print("No tool calls detected")
            
        # Check for thinking
        has_thinking = "<think>" in response.choices[0].message.content
        print(f"Has <think> tags: {has_thinking}")
        
        return response
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    asyncio.run(test_forced_qwen_tools())