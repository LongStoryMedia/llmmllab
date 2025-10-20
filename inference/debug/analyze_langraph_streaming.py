#!/usr/bin/env python3
"""Debug script to analyze LangGraph streaming events and identify duplication sources."""

import asyncio
import sys
from typing import Dict, List

# Add inference directory to path for imports
sys.path.append('/Users/lons7862/workspace/llmmllab/inference')

async def debug_langraph_events():
    """Debug LangGraph streaming to identify event duplication patterns."""
    print("🔍 Debugging LangGraph Streaming Events")
    print("=" * 60)
    
    try:
        # Import composer service
        import composer
        
        # Initialize composer
        print("Initializing composer...")
        await composer.initialize_composer()
        
        # Test parameters
        user_id = "CgNsc20SBGxkYXA"
        conversation_id = 717
        
        # Create workflow and state
        print(f"Creating workflow for user: {user_id}")
        workflow = await composer.compose_workflow(user_id)
        
        print(f"Creating initial state for conversation: {conversation_id}")
        initial_state = await composer.create_initial_state(user_id, conversation_id)
        
        # Track events by type
        event_counts: Dict[str, int] = {}
        event_contents: Dict[str, List[str]] = {}
        
        print("\n📡 Streaming workflow events...")
        print("-" * 60)
        
        async for event in composer.execute_workflow(workflow, initial_state, stream=True):
            if isinstance(event, dict):
                event_type = event.get("event", "unknown")
                event_data = event.get("data", {})
                
                # Count events
                event_counts[event_type] = event_counts.get(event_type, 0) + 1
                
                # Extract content for analysis
                content = None
                
                if event_type == "on_chat_model_stream":
                    chunk = event_data.get("chunk", {})
                    if isinstance(chunk, dict):
                        content = chunk.get("content", "")
                    elif hasattr(chunk, "content"):
                        content = str(chunk.content)
                        
                elif event_type in ["on_chain_stream", "on_tool_end", "on_chain_start"]:
                    output = event_data.get("output", {})
                    if isinstance(output, dict):
                        messages = output.get("messages", [])
                        if messages and isinstance(messages, list):
                            for msg in messages:
                                if isinstance(msg, dict) and msg.get("type") == "ai":
                                    content = msg.get("content", "")
                                    break
                                    
                elif event_type == "on_chain_end":
                    output = event_data.get("output", {})
                    if isinstance(output, dict):
                        messages = output.get("messages", [])
                        if messages and isinstance(messages, list):
                            for msg in reversed(messages):
                                if isinstance(msg, dict) and msg.get("type") == "ai":
                                    content = msg.get("content", "")
                                    break
                
                # Store content for analysis
                if content:
                    if event_type not in event_contents:
                        event_contents[event_type] = []
                    event_contents[event_type].append(content)
                    
                    # Show first few content samples
                    if len(event_contents[event_type]) <= 5:
                        content_preview = content[:100] + "..." if len(content) > 100 else content
                        print(f"[{event_type}] Content: {repr(content_preview)}")
        
        print("\n📊 Event Analysis Summary")
        print("=" * 60)
        
        print("Event Counts:")
        for event_type, count in sorted(event_counts.items()):
            print(f"  {event_type}: {count}")
            
        print("\nContent Duplication Analysis:")
        for event_type, contents in event_contents.items():
            print(f"\n{event_type}:")
            print(f"  - Total content chunks: {len(contents)}")
            if contents:
                first_content = contents[0]
                print(f"  - First content length: {len(first_content)}")
                print(f"  - First content preview: {repr(first_content[:150])}")
                
                # Check for duplicates
                unique_contents = set(contents)
                if len(unique_contents) < len(contents):
                    print(f"  - ⚠️  Contains duplicates! {len(contents)} total, {len(unique_contents)} unique")
                else:
                    print("  - ✅ No duplicates detected")
                    
        # Analyze cross-event duplication
        print("\nCross-Event Content Overlap:")
        all_contents = []
        for event_type, contents in event_contents.items():
            for content in contents:
                all_contents.append((event_type, content))
                
        content_to_events = {}
        for event_type, content in all_contents:
            if content not in content_to_events:
                content_to_events[content] = []
            content_to_events[content].append(event_type)
            
        duplicated_content = {content: events for content, events in content_to_events.items() if len(events) > 1}
        
        if duplicated_content:
            print(f"⚠️  Found {len(duplicated_content)} pieces of content duplicated across events:")
            for i, (content, events) in enumerate(list(duplicated_content.items())[:3]):
                content_preview = content[:100] + "..." if len(content) > 100 else content
                print(f"  {i+1}. Content: {repr(content_preview)}")
                print(f"     Events: {events}")
        else:
            print("✅ No content duplication across different event types")
        
    except Exception as e:
        print(f"❌ Error debugging LangGraph events: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(debug_langraph_events())