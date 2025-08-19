"""
Utility functions for preparing chat context and RAG data.
"""

from typing import List, Dict, Any, Optional
from models.message import Message
from models.message_content import MessageContent
from models.message_content_type import MessageContentType
from models.message_role import MessageRole
from models.model_profile import ModelProfile
from server.context.conversation import ConversationContext


def prepare_enhanced_messages(
    messages: List[Message],
    summary: Optional[Any],
    rag_data: Dict[str, Any],
    conversation_ctx: ConversationContext,
    model_profile: Optional[ModelProfile],
) -> List[Message]:
    """
    Prepare enhanced messages with context from RAG results and conversation summaries.

    Args:
        messages: Original message list
        summary: Conversation summary if available
        rag_data: RAG results (memories, web_results, etc.)
        conversation_ctx: Conversation context
        model_profile: Full model profile

    Returns:
        Enhanced message list with system messages for context
    """
    enhanced_messages = []

    # If we have a model profile with system prompt, use it
    if model_profile and model_profile.system_prompt:
        enhanced_messages.append(
            Message(
                role=MessageRole.SYSTEM,
                content=[
                    MessageContent(
                        type=MessageContentType.TEXT,
                        text=model_profile.system_prompt,
                        url=None,
                    )
                ],
                conversation_id=messages[0].conversation_id if messages else -1,
            )
        )

    # Add RAG context as a system message if we have any
    rag_contexts = []

    # Add conversation summary if available
    if summary:
        rag_contexts.append(f"# Conversation Summary\n{summary}")

    # Add memories if available
    if "memories" in rag_data and rag_data["memories"]:
        memory_context = "# Related Memories\n"
        for i, memory in enumerate(rag_data["memories"]):
            if hasattr(memory, "fragments") and memory.fragments:
                memory_context += f"\n## Memory {i+1}\n"
                for fragment in memory.fragments:
                    if hasattr(fragment, "role") and hasattr(fragment, "content"):
                        memory_context += f"{fragment.role}: {fragment.content}\n"
        rag_contexts.append(memory_context)

    # Add web search results if available
    if "web_results" in rag_data and rag_data["web_results"]:
        web_context = "# Web Search Results\n"
        for i, result in enumerate(rag_data["web_results"]):
            if hasattr(result, "title") and hasattr(result, "content"):
                web_context += f"\n## Result {i+1}: {result.title}\n{result.content}\n"
            elif isinstance(result, dict):
                title = result.get("title", f"Result {i+1}")
                snippet = result.get("snippet", "")
                web_context += f"\n## {title}\n{snippet}\n"
        rag_contexts.append(web_context)

    # Add URL content if available
    if "url_content" in rag_data and rag_data["url_content"]:
        url_context = "# URL Content\n"
        for i, (url, content) in enumerate(rag_data["url_content"].items()):
            url_context += f"\n## Content from {url}\n{content[:500]}...\n"
        rag_contexts.append(url_context)

    # If we have any RAG contexts, add them as a system message
    if rag_contexts:
        rag_system_message = Message(
            role=MessageRole.SYSTEM,
            content=[
                MessageContent(
                    type=MessageContentType.TEXT,
                    text="\n\n".join(rag_contexts),
                    url=None,
                )
            ],
            conversation_id=messages[0].conversation_id if messages else -1,
        )
        enhanced_messages.append(rag_system_message)

    # Add all the original messages
    enhanced_messages.extend(messages)

    return enhanced_messages


def parse_ddg_results(ddg_results: str) -> List[Dict[str, str]]:
    """
    Parse DuckDuckGo search results string into structured data.

    Args:
        ddg_results: String output from DuckDuckGo search

    Returns:
        List of dictionaries with title, link, and snippet
    """
    results = []

    # Simple parsing for the form that DDG usually returns
    try:
        lines = ddg_results.strip().split("\n")
        i = 0
        while i < len(lines):
            if lines[i].startswith("[") and "]" in lines[i]:
                title_parts = lines[i].split("]", 1)
                title = title_parts[1].strip() if len(title_parts) > 1 else ""
                i += 1
                if i < len(lines) and lines[i].startswith("(") and ")" in lines[i]:
                    link_parts = lines[i].split(")", 1)
                    link = link_parts[0].replace("(", "").strip()
                    i += 1
                    snippet = ""
                    while i < len(lines) and not (
                        lines[i].startswith("[") and "]" in lines[i]
                    ):
                        snippet += lines[i] + " "
                        i += 1
                    results.append(
                        {"title": title, "link": link, "snippet": snippet.strip()}
                    )
                else:
                    i += 1
            else:
                i += 1
    except Exception:
        # If parsing fails, return empty results
        pass

    return results
