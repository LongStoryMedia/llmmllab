"""
Static summarization tool using LangGraph Command pattern.

This tool performs content summarization with efficient state access and
proper LangGraph integration following established architectural patterns.

Features:
- Single function-based tool using @tool decorator
- Strong typing with WorkflowState instead of generic parameters
- Efficient user_config access from injected state (no database calls)
- Command pattern for proper state updates
- LLM-based summarization with fallback handling

Usage in LangGraph workflows:
    # Tool is automatically available when registered in tool registry
    # LangGraph handles injection of tool_call_id and WorkflowState
"""

import json
from typing import Annotated, cast

from langchain_core.tools import tool, InjectedToolCallId
from langchain_core.messages import ToolMessage
from langchain.chat_models import BaseChatModel
from langchain.tools.tool_node import InjectedState
from langgraph.types import Command

from composer.graph.state import WorkflowState
from composer.utils.extraction import extract_content_from_base_langchain_message
from runner import pipeline_factory
from models import ModelProfileType, PipelinePriority
from utils.model_profile import get_model_profile
from utils.logging import llmmllogger


# Single summarization tool using Command pattern with strong typing
@tool
async def summarization(
    content: str,
    tool_call_id: Annotated[str, InjectedToolCallId],
    state: Annotated[WorkflowState, InjectedState],
) -> Command:
    """
    Summarize content using LLM and automatically add results to workflow state.

    This tool creates concise summaries of long text content while preserving
    key information and main points. Use this tool when you need to condense
    lengthy text, articles, or documents into digestible summaries.

    Args:
        content: The text content to summarize

    Returns:
        Summarized content with key points and main themes preserved
    """
    logger = llmmllogger.logger.bind(component="Summarization")

    try:
        # Validate input content
        if not content.strip():
            error_message = json.dumps(
                {
                    "status": "error",
                    "error": "No content provided for summarization",
                    "content": content,
                },
                indent=2,
            )
            return Command(
                update={
                    "messages": [ToolMessage(error_message, tool_call_id=tool_call_id)]
                }
            )

        # Ensure we have required state
        user_id = state.user_id
        if not user_id:
            error_message = json.dumps(
                {
                    "status": "error",
                    "error": "Missing user_id in state",
                    "content": content[:100],
                },
                indent=2,
            )
            return Command(
                update={
                    "messages": [ToolMessage(error_message, tool_call_id=tool_call_id)]
                }
            )

        # Use LLM pipeline for proper summarization
        try:
            # Get model profile for summarization
            model_profile = await get_model_profile(
                user_id,
                ModelProfileType.PrimarySummary,
            )

            # Create summarization prompt
            summary_prompt = f"Please provide a concise summary of the following content:\n\n{content}"

            # Get pipeline and generate summary
            with pipeline_factory.pipeline(
                model_profile, PipelinePriority.NORMAL
            ) as pipeline:
                llm = cast(BaseChatModel, pipeline)
                # Since run_pipeline is not available, use the pipeline directly
                # This is a simplified approach that should work with the pipeline
                result = await llm.ainvoke(summary_prompt)
                summary_text = (
                    extract_content_from_base_langchain_message(result)
                    if result
                    else ""
                )

                if summary_text:
                    # Create response message for the conversation
                    response_message = json.dumps(
                        {
                            "status": "success",
                            "summary": summary_text,
                            "original_length": len(content),
                            "summary_length": len(summary_text),
                        },
                        indent=2,
                    )

                    logger.info(
                        "Summarization completed successfully",
                        original_length=len(content),
                        summary_length=len(summary_text),
                    )

                    # Return Command that updates state with summary results
                    return Command(
                        update={
                            "summary_content": summary_text,
                            "original_content": content,
                            "messages": [
                                ToolMessage(response_message, tool_call_id=tool_call_id)
                            ],
                        }
                    )

        except Exception as llm_error:
            logger.warning(
                f"LLM summarization failed: {llm_error}, using fallback method"
            )

        # Fallback to simple processing if LLM fails
        max_length = 300
        summary_text = (
            content[:max_length] + "..." if len(content) > max_length else content
        )

        response_message = json.dumps(
            {
                "status": "partial_success",
                "summary": summary_text,
                "original_length": len(content),
                "summary_length": len(summary_text),
                "note": "Used fallback method due to LLM unavailability",
            },
            indent=2,
        )

        logger.info(
            "Summarization completed with fallback method",
            original_length=len(content),
            summary_length=len(summary_text),
        )

        return Command(
            update={
                "summary_content": summary_text,
                "original_content": content,
                "messages": [ToolMessage(response_message, tool_call_id=tool_call_id)],
            }
        )

    except Exception as e:
        # Log the full exception for debugging
        logger.error(
            f"Summarization failed for content: {e}",
            exc_info=True,
            content_length=len(content) if content else 0,
        )

        error_message = json.dumps(
            {
                "status": "error",
                "error": str(e),
                "content": content[:100] + "..." if len(content) > 100 else content,
            },
            indent=2,
        )

        return Command(
            update={
                "original_content": content,
                "messages": [ToolMessage(error_message, tool_call_id=tool_call_id)],
            }
        )
