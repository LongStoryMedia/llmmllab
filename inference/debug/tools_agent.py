"""Test ToolsAgentSubgraph as main graph with LlamaCpp pipeline."""

import datetime
import os
import sys
from typing import List, Optional

from langchain_core.messages import (
    HumanMessage,
    SystemMessage,
    AIMessage,
    BaseMessage,
    ToolMessage,
)
from models import (
    Message,
    MessageContent,
    MessageContentType,
    MessageRole,
    NodeMetadata,
    ModelProfile,
)
from models.default_model_profiles import DEFAULT_TEXT_TO_TEXT_MODEL, DEFAULT_PROFILES
from models.default_configs import create_default_user_config
from runner import pipeline_factory, ReasoningAwareAIMessageChunk
from composer.agents import ChatAgent
from composer.tools.registry import ToolRegistry
from composer.graph.subgraphs import ToolsAgentSubgraph
from composer.graph.state import WorkflowState
from composer import execute_workflow
from utils.message_conversion import messages_to_lc_messages
from utils.logging import llmmllogger, serialize_event_data

logger = llmmllogger.bind(component="test_tools_agent_subgraph")

TEST_IMAGE_URL = (
    "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"
)


def get_profile(model_id: str) -> ModelProfile:
    """
    Get the correct model profile for the specified model ID.

    This ensures we use the complete profile with compatible mmproj and other settings,
    rather than just changing the model_name on a mismatched profile.
    """
    # First try to find an exact profile match for the requested model
    # for profile_name, profile in DEFAULT_PROFILES.items():
    #     if getattr(profile, "model_name", None) == model_id:
    #         logger.info(
    #             f"📋 Found exact profile match '{profile_name}' for model {model_id}"
    #         )
    #         return profile

    # If no exact match, use primary profile with original model for safety
    # This ensures compatible mmproj and other model-specific settings
    profile = DEFAULT_PROFILES.get("primary")
    if profile is None:
        print(f"[error] No primary profile found")
        sys.exit(1)
    # if profile.model_name != model_id:
    #     logger.warning(
    #         f"⚠️ No exact profile match for {model_id}, using primary profile with original model {profile.model_name}"
    #     )
    return profile


async def wrapper(model_id: str, query: str = "", image_url: str = "") -> None:
    # Ensure verbose logging minimal unless TRACE requested
    os.environ.setdefault("LOG_LEVEL", "INFO")
    timestamp = datetime.datetime.now(datetime.timezone.utc)

    logger.info("🚀 Starting ToolsAgentSubgraph test")

    # Instantiate pipeline factory (local cache mode)
    # Retrieve a default profile (assumes DEFAULT_MODEL_PROFILES contains profile for model)
    profile = get_profile(model_id)
    logger.info(f"📊 Using model profile: {profile.model_name}")

    agent = ChatAgent(
        pipeline_factory,
        profile,
        NodeMetadata(node_name="debug_node", node_id="debug_001", node_type="debug"),
    )

    registry = ToolRegistry(pipeline_factory)
    logger.info(
        f"🛠️ Tool registry initialized with {len(registry.get_all_executable_tools())} tools"
    )

    content = []

    if image_url or (not query and not image_url):
        if not image_url:
            image_url = TEST_IMAGE_URL
        content.append(MessageContent(type=MessageContentType.IMAGE, url=image_url))
    content.append(
        MessageContent(
            type=MessageContentType.TEXT,
            text=query
            or "What do you see in this image? Please describe it briefly, then search the web for similar images and provide links to at least two relevant sources. Also, find 3 recent news articles related to the content of the image and provide a brief summary of each along with their links, titles, and publication dates.",
        )
    )

    # Create test messages
    test_messages = [
        Message(
            role=MessageRole.USER,
            content=content,
        )
    ]

    # Create ToolsAgentSubgraph
    tools_agent_subgraph = ToolsAgentSubgraph(
        registry,
        agent,
    )
    logger.info("🤖 ToolsAgentSubgraph initialized")

    user_config = create_default_user_config(user_id="test_user")
    # Create initial WorkflowState
    workflow_state = WorkflowState(
        messages=test_messages,
        user_id="test_user",
        conversation_id=717,
        user_config=user_config,
    )

    logger.info("🎯 Starting ToolsAgentSubgraph streaming execution...")

    # Execute the subgraph with streaming
    try:
        if not tools_agent_subgraph.graph:
            logger.error("❌ ToolsAgentSubgraph graph not initialized")
            return

        print("\n" + "=" * 80)
        print("STREAMING TOOLS AGENT SUBGRAPH EXECUTION")
        print("=" * 80)
        print()

        # Stream the graph execution
        async for res in execute_workflow(
            initial_state=workflow_state,
            workflow=tools_agent_subgraph.graph,
        ):
            if res.message is None:
                logger.warning("Received empty message in stream event")
                continue

            for c in res.message.content:
                if (
                    c.type == MessageContentType.THINKING
                    or c.type == MessageContentType.TEXT
                ):
                    print(c.text, end="", flush=True)

            if res.message.tool_calls:
                for t in res.message.tool_calls:
                    print("\n" + "-" * 40)
                    print(f"Tool Call: {t.name}")
                    print(f"Arguments: {serialize_event_data(t.args)}")
                    print(
                        f"RESULTS: {t.result_data.get('output', '') if t.result_data else ''}"
                    )
                    print("-" * 40)

        logger.info("🎉 Streaming test completed successfully")

    except Exception as e:
        logger.error(f"❌ ToolsAgentSubgraph execution failed: {e}")
        import traceback

        traceback.print_exc()
        raise

    finally:
        # CRITICAL: Clean up agent to prevent memory leaks
        try:
            agent.cleanup()
            logger.info("✅ Agent cleanup completed - pipelines unlocked")
        except Exception as e:
            logger.error(f"❌ Agent cleanup failed: {e}")

    # print("\n[debug] Completed ToolsAgentSubgraph test without immediate crash")


if __name__ == "__main__":
    import asyncio
    import argparse

    parser = argparse.ArgumentParser(description="Run a single LLaMA CPP chat.")
    parser.add_argument(
        "--model", type=str, default=DEFAULT_TEXT_TO_TEXT_MODEL, help="Model ID to use"
    )
    parser.add_argument(
        "--query", type=str, default="", help="Query to send to the model"
    )
    parser.add_argument(
        "--image", type=str, default="", help="Image URL to include in the query"
    )
    args = parser.parse_args()

    asyncio.run(wrapper(model_id=args.model, query=args.query, image_url=args.image))
