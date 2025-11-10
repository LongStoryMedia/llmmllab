"""Test ToolsAgentSubgraph as main graph with LlamaCpp pipeline."""

import os
import sys
from typing import List, Optional

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, BaseMessage
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
from runner.pipeline_factory import pipeline_factory
from composer.agents import ChatAgent
from composer.tools.registry import ToolRegistry
from composer.graph.subgraphs import ToolsAgentSubgraph
from composer.graph.state import ToolsState
from utils.message_conversion import messages_to_lc_messages
from utils.logging import llmmllogger

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
    for profile_name, profile in DEFAULT_PROFILES.items():
        if getattr(profile, "model_name", None) == model_id:
            logger.info(
                f"📋 Found exact profile match '{profile_name}' for model {model_id}"
            )
            return profile

    # If no exact match, use primary profile with original model for safety
    # This ensures compatible mmproj and other model-specific settings
    profile = DEFAULT_PROFILES.get("primary")
    if profile is None:
        print(f"[error] No primary profile found")
        sys.exit(1)
    if profile.model_name != model_id:
        logger.warning(
            f"⚠️ No exact profile match for {model_id}, using primary profile with original model {profile.model_name}"
        )
    return profile


async def wrapper(model_id: str, query: str = "", image_url: str = "") -> None:
    # Ensure verbose logging minimal unless TRACE requested
    os.environ.setdefault("LOG_LEVEL", "INFO")

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
            or "What do you see in this image? Please describe it briefly, then search the web for similar images and provide links to at least two relevant sources.",
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
    # Create initial ToolsState
    tools_state = ToolsState(
        messages=messages_to_lc_messages(test_messages),
        user_id="test_user",
        conversation_id=717,
        user_config=user_config,
        tool_call_count=0,
        shared_pipeline=None,  # Will be set by chat agent during execution
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

        message_count = 0

        # Stream the graph execution
        async for chunk in tools_agent_subgraph.graph.astream(tools_state):
            print(f"📦 Received chunk: {chunk}")

            # Handle different chunk formats from LangGraph
            for node_name, node_output in chunk.items():
                print(
                    f"🔄 Processing node '{node_name}' with output keys: {list(node_output.keys())}"
                )

                if node_name in ["chat_agent", "tools"]:
                    messages = node_output.get("messages", [])
                    print(f"📬 Node '{node_name}' has {len(messages)} messages")

                    for msg_idx, msg in enumerate(messages):
                        message_count += 1
                        msg_type = getattr(msg, "type", type(msg).__name__)
                        content = getattr(msg, "content", "")
                        tool_calls = getattr(msg, "tool_calls", [])

                        print(
                            f"\n🔄 [{node_name}] Message {message_count} (idx {msg_idx}) [{msg_type}]:"
                        )
                        print(
                            f"   Raw message: {type(msg)} with content length: {len(str(content))}"
                        )

                        # Always show content if it exists, regardless of previous content
                        if content:
                            # Handle different content formats
                            if isinstance(content, str):
                                print(f"📝 Content ({len(content)} chars):")
                                print(f"   {content}")
                            elif isinstance(content, list):
                                print(f"📝 Content List ({len(content)} items):")
                                for i, item in enumerate(content):
                                    if isinstance(item, dict) and "text" in item:
                                        text = item["text"]
                                        print(f"   Item {i+1}: {text}")
                                    else:
                                        print(f"   Item {i+1}: {item}")
                            else:
                                print(f"📝 Content ({len(str(content))} chars):")
                                print(f"   {content}")
                            print("   " + "-" * 50)
                        else:
                            print(f"📝 No content in this message")

                        # Handle tool calls
                        if tool_calls:
                            print(f"🛠️ Tool calls: {len(tool_calls)} calls")
                            for j, tc in enumerate(tool_calls):
                                if isinstance(tc, dict):
                                    tool_name = tc.get("name", "unknown")
                                    tool_args = tc.get("args", {})
                                    print(
                                        f"  {j+1}. Calling {tool_name} with args: {tool_args}"
                                    )
                                else:
                                    tool_name = getattr(tc, "name", "unknown")
                                    tool_args = getattr(tc, "args", {})
                                    print(
                                        f"  {j+1}. Calling {tool_name} with args: {tool_args}"
                                    )
                        else:
                            print(f"🛠️ No tool calls in this message")

                        # Show additional message attributes for debugging
                        print(f"🔍 Debug - Message attributes:")
                        for attr in ["id", "additional_kwargs", "response_metadata"]:
                            if hasattr(msg, attr):
                                attr_value = getattr(msg, attr)
                                print(f"   {attr}: {attr_value}")

                        print(flush=True)

        print("\n" + "=" * 80)
        print(f"✅ STREAMING COMPLETE - Total message events: {message_count}")
        print("=" * 80)

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

    print("\n[debug] Completed ToolsAgentSubgraph test without immediate crash")


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
