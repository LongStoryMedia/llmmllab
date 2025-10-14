#!/usr/bin/env python3
"""
Debug script to test streaming with metadata functionality.
"""

import asyncio
import os
import sys

# Add the inference directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models import NodeMetadata, ModelProfile, LangChainMessage
from composer.agents.chat_agent import ChatAgent
from runner import PipelineFactory


async def test_streaming_with_metadata():
    """Test the new streaming with metadata functionality."""
    print("🧪 Testing streaming with metadata functionality...")
    
    # Create mock dependencies
    pipeline_factory = PipelineFactory()
    
    # Create mock model profile
    profile = ModelProfile(
        model_name="test_model",
        model_id="test_model_001",
        provider="test_provider",
        # Add other required fields based on your ModelProfile schema
    )
    
    # Create node metadata
    node_metadata = NodeMetadata(
        node_name="Test Chat Agent",
        node_id="test_chat_001",
        node_type="ChatAgent",
        user_id="test_user_123",
        conversation_id=456
    )
    
    # Create chat agent
    chat_agent = ChatAgent(
        pipeline_factory=pipeline_factory,
        profile=profile,
        node_metadata=node_metadata
    )
    
    # Create test messages
    test_messages = [
        LangChainMessage(
            content="Hello, can you help me understand streaming with metadata?",
            additional_kwargs={}
        )
    ]
    
    print("\n📡 Starting streaming test...")
    
    try:
        chunk_count = 0
        boundary_chunks = 0
        content_chunks = 0
        
        async for chunk in chat_agent.stream_chat_completion(
            messages=test_messages,
            user_id="test_user_123"
        ):
            chunk_count += 1
            
            # Check if chunk has metadata
            if chunk.channels:
                node_meta = chunk.channels.get("node_metadata", {})
                stream_meta = chunk.channels.get("stream_metadata", {})
                chunk_meta = chunk.channels.get("chunk_metadata", {})
                
                if stream_meta.get("is_boundary"):
                    boundary_chunks += 1
                    if stream_meta.get("is_start"):
                        print(f"🚀 START: {node_meta.get('node_name')} ({node_meta.get('node_type')})")
                        print(f"   User: {node_meta.get('user_id')}, Conversation: {node_meta.get('conversation_id')}")
                    elif stream_meta.get("content_type") == "stream_end":
                        print(f"✅ END: {node_meta.get('node_name')}")
                        if "total_chunks" in stream_meta:
                            print(f"   Total chunks processed: {stream_meta['total_chunks']}")
                    elif stream_meta.get("content_type") == "stream_error":
                        print(f"❌ ERROR: {node_meta.get('node_name')}")
                        print(f"   Error: {stream_meta.get('error')}")
                else:
                    content_chunks += 1
                    chunk_idx = chunk_meta.get("chunk_index", "?")
                    print(f"📝 Chunk {chunk_idx}: {node_meta.get('node_name')}")
                    if chunk.message and chunk.message.content:
                        content_preview = str(chunk.message.content)[:50] + "..." if len(str(chunk.message.content)) > 50 else str(chunk.message.content)
                        print(f"   Content: {content_preview}")
            else:
                print(f"⚠️  Chunk {chunk_count} has no metadata!")
                
        print(f"\n📊 Summary:")
        print(f"   Total chunks: {chunk_count}")
        print(f"   Boundary chunks: {boundary_chunks}")
        print(f"   Content chunks: {content_chunks}")
        print(f"   ✅ Test completed successfully!")
                
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()


async def test_non_streaming_with_metadata():
    """Test non-streaming execution with metadata."""
    print("\n🔧 Testing non-streaming with metadata...")
    
    # This would use the same setup as above but call run_pipeline_with_metadata
    # Implementation would depend on having a working pipeline factory
    print("⏭️  Non-streaming test skipped (requires full pipeline setup)")


if __name__ == "__main__":
    print("🎯 BaseAgent Streaming Metadata Test")
    print("=" * 50)
    
    asyncio.run(test_streaming_with_metadata())
    asyncio.run(test_non_streaming_with_metadata())
    
    print("\n🏁 All tests completed!")