"""
Static summarization tool using simple text processing.

This tool performs content summarization with consistent behavior
using basic text processing for reliable static operation.
"""

import asyncio
import json

from langchain_core.tools import BaseTool
from runner import run_pipeline, pipeline_factory
from runner.pipeline_factory import PipelinePriority
from models.chat_response import ChatResponse
from models.message import Message
from models.message_role import MessageRole
from models.message_content import MessageContent
from models.message_content_type import MessageContentType


class SummarizationTool(BaseTool):
    """Static tool for summarizing content using simple text processing."""

    name: str = "summarization"
    description: str = (
        "Summarize content using basic text processing. Takes text content and returns a concise summary."
    )

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

                # Create summarization prompt
                summary_prompt = f"Please provide a concise summary of the following content:\n\n{content}"
                summary_message = Message(
                    role=MessageRole.USER,
                    content=[
                        MessageContent(
                            type=MessageContentType.TEXT, text=summary_prompt
                        )
                    ],
                )

                # For static tool, we need to use a default configuration approach
                # Since static tools don't have access to user config, we'll use first available model
                try:
                    # Get first available model that can handle text generation
                    available_models = pipeline_factory.list_available_models()
                    if not available_models:
                        raise Exception("No models available for summarization")

                    # Use pipeline factory to get a pipeline
                    # This is a simplified approach for static tool
                    from models.model_profile import ModelProfile

                    # Try to create a basic model profile for summarization
                    # In a real implementation, this would use utils.get_model_profile_for_task()
                    model_name = available_models[0]  # Use first available model
                    model = pipeline_factory.get_model(model_name)

                    if model:
                        # Create a basic profile for summarization
                        profile = ModelProfile(
                            id="static-summarization",
                            name="Static Summarization Profile",
                            user_id="static-tool",
                            model_id=model.id,
                            temperature=0.7,
                            max_tokens=200,
                            top_p=0.9,
                        )

                        # Get pipeline and run summarization
                        pipeline = pipeline_factory.get_pipeline(
                            profile=profile,
                            expected_type=ChatResponse,
                            priority=PipelinePriority.NORMAL,
                        )

                        result = await run_pipeline(
                            messages=[summary_message], pipeline=pipeline
                        )

                        if result and result.message:
                            from utils.message import extract_message_text

                            summary_text = extract_message_text(result.message)
                        else:
                            raise Exception("Pipeline returned no result")
                    else:
                        raise Exception("Could not get model from factory")

                except Exception as pipeline_error:
                    # Fallback to simple text processing if pipeline fails
                    sentences = content.split(". ")
                    summary_sentences = []
                    current_length = 0
                    max_length = 300

                    for sentence in sentences:
                        if (
                            current_length + len(sentence) > max_length
                            and summary_sentences
                        ):
                            break
                        summary_sentences.append(sentence.strip())
                        current_length += len(sentence) + 2

                    if not summary_sentences:
                        summary_text = (
                            content[:max_length] + "..."
                            if len(content) > max_length
                            else content
                        )
                    else:
                        summary_text = ". ".join(summary_sentences)
                        if not summary_text.endswith("."):
                            summary_text += "."

                    summary_text += f" (Fallback: {str(pipeline_error)})"

                return json.dumps(
                    {
                        "status": "success",
                        "summary": summary_text,
                        "original_length": len(content),
                        "summary_length": len(summary_text),
                        "compression_ratio": round(len(summary_text) / len(content), 2),
                    },
                    indent=2,
                )

            except Exception as llm_error:
                # Final fallback to simple processing
                sentences = content.split(". ")
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
