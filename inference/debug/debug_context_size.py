#!/usr/bin/env python3
"""
Debug script to examine context size issues in the chat node.
"""

import sys
import os
import asyncio
import logging
from typing import Dict, Any, List

# Setup environment
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from composer.utils.state import assemble_context_messages, _estimate_tokens, _count_message_tokens
from models import Message, MessageContent, MessageContentType, MessageRole
from models.langgraph_state import WorkflowState


def analyze_context_components(state: WorkflowState) -> Dict[str, Any]:
    """
    Analyze the components of a WorkflowState to understand token usage.
    """
    analysis = {
        "total_estimated_tokens": 0,
        "components": {}
    }
    
    # Analyze state.messages
    if state.messages:
        messages_as_message_objects = []
        for msg in state.messages:
            # Convert LangChain message to Message for token counting
            if hasattr(msg, 'content') and hasattr(msg, 'type'):
                content_text = str(msg.content) if msg.content else ""
                role = MessageRole.USER if msg.type == 'human' else MessageRole.ASSISTANT
                message = Message(
                    content=[MessageContent(type=MessageContentType.TEXT, text=content_text)],
                    role=role,
                    conversation_id=state.conversation_id
                )
                messages_as_message_objects.append(message)
        
        messages_tokens = _count_message_tokens(messages_as_message_objects)
        analysis["components"]["state_messages"] = {
            "count": len(state.messages),
            "tokens": messages_tokens,
            "content_preview": [str(msg.content)[:200] + "..." if len(str(msg.content)) > 200 else str(msg.content) 
                              for msg in state.messages[:3]]
        }
        analysis["total_estimated_tokens"] += messages_tokens
    
    # Analyze retrieved_memories
    if state.retrieved_memories:
        memory_tokens = 0
        memory_count = 0
        memory_preview = []
        
        for memory in state.retrieved_memories:
            if hasattr(memory, "fragments") and memory.fragments:
                for fragment in memory.fragments:
                    if hasattr(fragment, "content"):
                        content = str(fragment.content)
                        memory_tokens += _estimate_tokens(content)
                        memory_count += 1
                        if len(memory_preview) < 3:
                            preview = content[:200] + "..." if len(content) > 200 else content
                            memory_preview.append(preview)
        
        analysis["components"]["retrieved_memories"] = {
            "memory_objects": len(state.retrieved_memories),
            "fragment_count": memory_count,
            "tokens": memory_tokens,
            "content_preview": memory_preview
        }
        analysis["total_estimated_tokens"] += memory_tokens
    
    # Analyze summaries
    if state.summaries:
        summary_tokens = 0
        summary_preview = []
        
        for summary in state.summaries:
            if hasattr(summary, 'content'):
                content = str(summary.content)
                summary_tokens += _estimate_tokens(content)
                if len(summary_preview) < 3:
                    preview = content[:200] + "..." if len(content) > 200 else content
                    summary_preview.append(preview)
        
        analysis["components"]["summaries"] = {
            "count": len(state.summaries),
            "tokens": summary_tokens,
            "content_preview": summary_preview
        }
        analysis["total_estimated_tokens"] += summary_tokens
    
    return analysis


def print_analysis(analysis: Dict[str, Any]):
    """Print the context analysis in a readable format."""
    print("=" * 80)
    print("CONTEXT SIZE ANALYSIS")
    print("=" * 80)
    print(f"TOTAL ESTIMATED TOKENS: {analysis['total_estimated_tokens']:,}")
    print()
    
    for component_name, component_data in analysis["components"].items():
        print(f"{component_name.upper()}:")
        print(f"  Token Count: {component_data['tokens']:,}")
        
        if "count" in component_data:
            print(f"  Object Count: {component_data['count']}")
        if "fragment_count" in component_data:
            print(f"  Fragment Count: {component_data['fragment_count']}")
        if "memory_objects" in component_data:
            print(f"  Memory Objects: {component_data['memory_objects']}")
        
        print(f"  Content Preview:")
        for i, preview in enumerate(component_data.get("content_preview", []), 1):
            print(f"    {i}. {preview}")
        print()
    
    print("=" * 80)


if __name__ == "__main__":
    print("🔍 Context Size Debug Tool")
    print("This tool can be imported and used to analyze WorkflowState context size.")
    print("Example usage:")
    print("  from debug.debug_context_size import analyze_context_components, print_analysis")
    print("  analysis = analyze_context_components(state)")
    print("  print_analysis(analysis)")