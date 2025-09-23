#!/usr/bin/env python3
"""
Test script to verify harmony channel routing behavior.
"""

import sys
import os

# Add inference directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'inference'))

from inference.runner.pipelines.run import EventStreamProcessor
from inference.utils.response import create_streaming_chunk_with_thinking

def test_harmony_channel_routing():
    """Test that harmony channels are properly routed to thinking and content fields."""
    
    # Create processor with minimal harmony filtering
    processor = EventStreamProcessor()
    processor.enable_minimal_harmony_filtering()
    
    print("Testing harmony channel routing...")
    
    # Test 1: Analysis channel content should go to thinking field
    print("\n=== Test 1: Analysis Channel Routing ===")
    
    test_content = """<|channel|>analysis<|message|>Let me think about this step by step.
    
First, I need to understand the user's request.
Then I'll formulate a response.<|end|><|start|>assistant<|channel|>final<|message|>Here is my response to you."""
    
    results = []
    for char in test_content:
        result = processor._filter_analysis_channel(char)
        if result:
            results.append(result)
    
    print(f"Number of responses: {len(results)}")
    for i, result in enumerate(results):
        print(f"Response {i+1}:")
        if result.thinking:
            print(f"  Thinking: {repr(result.thinking[:100])}...")
        if result.message and result.message.content:
            print(f"  Content: {repr(result.message.content[0].text[:100])}...")
        print()
    
    # Test 2: Test the helper function directly
    print("\n=== Test 2: Helper Function Test ===")
    
    # Test thinking only
    thinking_response = create_streaming_chunk_with_thinking(thinking="This is thinking content")
    print(f"Thinking response - thinking: {thinking_response.thinking}")
    print(f"Thinking response - message thinking: {thinking_response.message.thinking}")
    
    # Test content only
    content_response = create_streaming_chunk_with_thinking(text="This is final content")
    print(f"Content response - text: {thinking_response.message.content[0].text if thinking_response.message.content else 'None'}")
    
    # Test both
    both_response = create_streaming_chunk_with_thinking(
        text="Final response", 
        thinking="My analysis"
    )
    print(f"Both response - thinking: {both_response.thinking}")
    print(f"Both response - content: {both_response.message.content[0].text if both_response.message.content else 'None'}")

if __name__ == "__main__":
    test_harmony_channel_routing()