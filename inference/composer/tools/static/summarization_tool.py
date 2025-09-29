"""
Static summarization tool using simple text processing.

This tool performs content summarization with consistent behavior
using basic text processing for reliable static operation.
"""

import asyncio
import json

from langchain_core.tools import BaseTool

from models.message import Message
from models.message_role import MessageRole
from models.message_content import MessageContent
from models.message_content_type import MessageContentType


class SummarizationTool(BaseTool):
    """Static tool for summarizing content using simple text processing."""
    name: str = "summarization"
    description: str = "Summarize content using basic text processing. Takes text content and returns a concise summary."

    async def _arun(self, content: str) -> str:
        """Async implementation of content summarization."""
        try:
            if not content.strip():
                return json.dumps({
                    "status": "error",
                    "error": "No content provided for summarization",
                    "content": content
                }, indent=2)
            
            # Simple static summarization approach
            # Take the first few sentences up to a reasonable length
            sentences = content.split('. ')
            summary_sentences = []
            current_length = 0
            max_length = 300  # Target summary length
            
            for sentence in sentences:
                if current_length + len(sentence) > max_length and summary_sentences:
                    break
                summary_sentences.append(sentence.strip())
                current_length += len(sentence) + 2  # +2 for '. '
            
            if not summary_sentences:
                # If content is very short or no sentences found, use first part
                summary_text = content[:max_length] + "..." if len(content) > max_length else content
            else:
                summary_text = '. '.join(summary_sentences)
                if not summary_text.endswith('.'):
                    summary_text += '.'
            
            return json.dumps({
                "status": "success",
                "summary": summary_text,
                "original_length": len(content),
                "summary_length": len(summary_text),
                "compression_ratio": round(len(summary_text) / len(content), 2)
            }, indent=2)
                
        except Exception as e:
            return json.dumps({
                "status": "error",
                "error": str(e),
                "content": content[:100] + "..." if len(content) > 100 else content
            }, indent=2)

    def _run(self, content: str) -> str:
        """Sync implementation using async."""
        return asyncio.run(self._arun(content))