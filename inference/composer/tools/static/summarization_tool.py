"""
Static summarization tool using simple text processing.

This tool performs content summarization with consistent behavior
using basic text processing for reliable static operation.
"""

import asyncio
import json

from langchain_core.tools import BaseTool
from runner import run_pipeline, pipeline_factory
from models import ModelProfileType, PipelinePriority
from utils.message import extract_message_text
from utils.model_profile import get_model_profile


class SummarizationTool(BaseTool):
    """Static tool for summarizing content using simple text processing."""

    name: str = "summarization"
    description: str = (
        "Summarize content using basic text processing. Takes text content and returns a concise summary."
    )

    # Declare user_id as a proper Pydantic field
    user_id: str

    def __init__(self, user_id: str, **kwargs):
        super().__init__(user_id=user_id, **kwargs)

    async def _arun(self, content: str) -> str:
        """Async implementation of content summarization."""
        try:
            if not content.strip():
                return json.dumps(
                    {
                        "status": "error",
                        "error": "No content provided for summarization",
                        "content": content,
                    },
                    indent=2,
                )

            # Use LLM pipeline for proper summarization
            try:
                mp = await get_model_profile(
                    self.user_id,
                    ModelProfileType.PrimarySummary,
                )

                # Create summarization prompt
                summary_prompt = f"Please provide a concise summary of the following content:\n\n{content}"

                with pipeline_factory.pipeline(
                    mp, str, PipelinePriority.NORMAL, mp.circuit_breaker
                ) as pipeline:
                    result = await run_pipeline(summary_prompt, pipeline)
                    return (
                        extract_message_text(result.message)
                        if result and result.message
                        else ""
                    )

            except Exception as llm_error:
                # Final fallback to simple processing
                max_length = 300
                summary_text = (
                    content[:max_length] + "..."
                    if len(content) > max_length
                    else content
                )

                return json.dumps(
                    {
                        "status": "partial_success",
                        "summary": summary_text,
                        "original_length": len(content),
                        "summary_length": len(summary_text),
                        "note": f"Used fallback method due to: {str(llm_error)}",
                    },
                    indent=2,
                )

        except Exception as e:
            return json.dumps(
                {
                    "status": "error",
                    "error": str(e),
                    "content": content[:100] + "..." if len(content) > 100 else content,
                },
                indent=2,
            )

    def _run(self, content: str) -> str:
        """Sync implementation using async."""
        return asyncio.run(self._arun(content))
