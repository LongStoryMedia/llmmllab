#!/usr/bin/env python3
"""
Quick Content Filtering Validation Test

This script tests the recent content filtering fixes by making a simple HTTP request
to the chat completion endpoint and validating that:
1. Intent analysis JSON doesn't leak into message content  
2. Thoughts don't appear in main message content
3. Tool calls show proper names (not "unknown_tool")
4. Thoughts are clean text (not serialized Pydantic objects)
5. System is aware of correct date (2025, not 2023)

This is a quick validation of our enhanced filtering and debugging.
"""

import asyncio
import json
import sys
import uuid
from datetime import datetime, timezone
import httpx
import re

async def quick_validation_test(server_url="http://localhost:8000"):
    """Run a quick validation test of content filtering fixes."""
    print("🧪 Quick Content Filtering Validation Test")
    print("=" * 60)
    
    # Create HTTP client
    client = httpx.AsyncClient(timeout=30.0)
    
    try:
        # Prepare a test message that should trigger tools and content generation
        test_query = "What are the latest developments in AI for October 2025? Please search for recent news and analyze the current trends."
        
        # Create test message data
        message_data = {
            "role": "user",
            "content": [{"type": "text", "text": test_query}],
            "conversation_id": 1,  # Use existing conversation
        }
        
        # Headers
        headers = {
            "Content-Type": "application/json",
            "User-ID": "test_validation_user",
            "X-Request-ID": f"validation_{uuid.uuid4().hex[:8]}",
        }
        
        print(f"📤 Sending request to {server_url}/chat/completions")
        print(f"💬 Query: {test_query[:100]}...")
        
        # Make streaming request
        content_issues = []
        streaming_chunks = 0
        full_content = ""
        tool_calls = []
        
        async with client.stream(
            "POST",
            f"{server_url}/chat/completions",
            json=message_data,
            headers=headers
        ) as response:
            
            if response.status_code != 200:
                print(f"❌ HTTP Error: {response.status_code}")
                response_text = await response.aread()
                print(f"Response: {response_text.decode()[:500]}")
                return False
            
            print("📡 Receiving streaming response...")
            
            async for line in response.aiter_lines():
                if not line.strip():
                    continue
                    
                try:
                    chunk_data = json.loads(line)
                    streaming_chunks += 1
                    
                    # Validate each chunk
                    if "message" in chunk_data and chunk_data["message"]:
                        message = chunk_data["message"]
                        
                        # Check main content
                        if "content" in message and message["content"]:
                            for content_item in message["content"]:
                                if content_item.get("type") == "text":
                                    text = content_item.get("text", "")
                                    full_content += text
                                    
                                    # VALIDATION 1: Intent analysis JSON leak
                                    if re.search(r'"intent":|"confidence":|IntentAnalysis\(', text, re.IGNORECASE):
                                        content_issues.append("Intent analysis JSON leaked into content")
                                    
                                    # VALIDATION 2: Thoughts leak  
                                    if re.search(r'<think>|</think>|Thought\(', text, re.IGNORECASE):
                                        content_issues.append("Thoughts leaked into main content")
                                    
                                    # VALIDATION 3: Wrong date (2023 instead of 2025)
                                    if re.search(r'\b2023\b', text):
                                        content_issues.append("System thinks it's 2023 instead of 2025")
                        
                        # Check tool calls
                        if "tool_calls" in message and message["tool_calls"]:
                            for tool_call in message["tool_calls"]:
                                tool_calls.append(tool_call)
                                
                                # VALIDATION 4: Unknown tool names
                                if tool_call.get("name") == "unknown_tool":
                                    content_issues.append("Tool call shows 'unknown_tool' name")
                        
                        # Check thoughts format
                        if "thoughts" in message and message["thoughts"]:
                            for thought in message["thoughts"]:
                                # VALIDATION 5: Serialized Pydantic objects
                                if isinstance(thought, dict) and any(key in str(thought) for key in ["__dict__", "__class__", "model_fields"]):
                                    content_issues.append("Thoughts contain serialized Pydantic objects")
                
                except json.JSONDecodeError:
                    content_issues.append("Invalid JSON in streaming response")
        
        # Results
        print(f"\n📊 Validation Results:")
        print(f"   Streaming chunks received: {streaming_chunks}")
        print(f"   Total content length: {len(full_content)} characters")
        print(f"   Tool calls detected: {len(tool_calls)}")
        
        # Check for content filtering issues
        if content_issues:
            print(f"\n❌ Content Filtering Issues Found ({len(content_issues)}):")
            for issue in content_issues:
                print(f"   • {issue}")
            return False
        else:
            print(f"\n✅ Content Filtering Validation PASSED!")
            print(f"   • No intent analysis JSON leaked")
            print(f"   • No thoughts leaked into main content") 
            print(f"   • No unknown tool names")
            print(f"   • No serialized Pydantic objects")
            print(f"   • Correct date context (2025)")
            
            # Additional validations
            if tool_calls:
                proper_tool_names = [tc.get("name") for tc in tool_calls if tc.get("name") != "unknown_tool"]
                print(f"   • Tool calls with proper names: {proper_tool_names}")
            
            if len(full_content) > 100:
                print(f"   • Generated substantial content ({len(full_content)} chars)")
            
            return True
            
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        return False
        
    finally:
        await client.aclose()


async def main():
    """Main test function."""
    success = await quick_validation_test()
    
    print(f"\n{'='*60}")
    if success:
        print("🎉 CONTENT FILTERING VALIDATION PASSED!")
        print("Recent fixes are working correctly.")
    else:
        print("⚠️  CONTENT FILTERING VALIDATION FAILED!")
        print("Issues detected - fixes need investigation.")
    print(f"{'='*60}")
    
    return 0 if success else 1


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n🛑 Test interrupted")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)