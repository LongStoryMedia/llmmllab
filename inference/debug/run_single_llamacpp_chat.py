"""Execute ToolsAgentSubgraph with a single multimodal user message for debugging."""

import os
import sys

from models import (
    Message,
    MessageContent,
    MessageContentType,
    MessageRole,
    NodeMetadata,
)
from models.default_model_profiles import DEFAULT_TEXT_TO_TEXT_MODEL, DEFAULT_PROFILES
from runner.pipeline_factory import pipeline_factory
from composer.agents import ChatAgent
from composer.tools.registry import ToolRegistry
from composer.graph.subgraphs import ToolsAgentSubgraph
from composer.graph.state import ToolsState
from composer.utils.conversion import convert_messages_to_base_langchain

TEST_IMAGE_URL = (
    "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"
)


def get_profile():
    model_name = DEFAULT_TEXT_TO_TEXT_MODEL
    profile = DEFAULT_PROFILES.get("primary")
    if getattr(profile, "model_name", None) != model_name:
        for _, p in DEFAULT_PROFILES.items():
            if getattr(p, "model_name", None) == model_name:
                profile = p
                break
    if profile is None:
        print(f"[error] No default model profile found for {model_name}")
        sys.exit(1)
    return profile


async def wrapper() -> None:
    # Ensure verbose logging minimal unless TRACE requested
    os.environ.setdefault("LOG_LEVEL", "INFO")

    # Instantiate pipeline factory (local cache mode)
    # Retrieve a default profile (assumes DEFAULT_MODEL_PROFILES contains profile for model)
    profile = get_profile()

    agent = ChatAgent(
        pipeline_factory,
        profile,
        NodeMetadata(node_name="debug_node", node_id="debug_001", node_type="debug"),
    )

    registry = ToolRegistry(pipeline_factory)

    messages = [
        Message(
            role=MessageRole.USER,
            content=[
                MessageContent(type=MessageContentType.IMAGE, url=TEST_IMAGE_URL),
                MessageContent(
                    type=MessageContentType.TEXT,
                    text="What do you see in this image? Please describe it briefly, then search the web for similar images and provide links to at least two relevant sources.",
                ),
            ],
        )
    ]

    tools_subgraph = ToolsAgentSubgraph(registry, agent)

    # Convert internal model messages to LangChain BaseMessage objects expected by ToolsState
    lc_messages = convert_messages_to_base_langchain(messages)

    # Minimal required ToolsState fields (fill with safe defaults)
    state: ToolsState = {
        "messages": lc_messages,  # type: ignore
        "user_id": "debug_user",
        "conversation_id": 1,
        "user_config": None,
        "system_config": None,
        "current_date": __import__("datetime").datetime.now().isoformat(),
        "tool_call_count": 0,
    }

    print("[debug] Executing ToolsAgentSubgraph graph (single invoke)...")

    try:
        result_state = await tools_subgraph.graph.ainvoke(state)  # type: ignore
    except Exception as e:
        print(f"[error] Subgraph execution failed: {e}")
        raise

    # Extract final assistant message(s)
    final_messages = (
        result_state.get("messages", []) if isinstance(result_state, dict) else []
    )
    last_ai = None
    for m in reversed(final_messages):
        if getattr(m, "type", "") == "ai":
            last_ai = m
            break
    if last_ai is None:
        print("[warn] No AI message produced")
    else:
        content = getattr(last_ai, "content", "")
        if isinstance(content, str):
            print("\n[assistant]", content)
        elif isinstance(content, list):  # LangChain content parts
            for part in content:
                if isinstance(part, dict) and part.get("type") == "output_text":
                    print(part.get("text", ""), end="")
        # Tool calls (structured)
        tool_calls = getattr(last_ai, "tool_calls", None)
        if tool_calls:
            print("\n[assistant tool_calls]:", tool_calls)

    print("\n[debug] ToolsAgentSubgraph execution complete")


if __name__ == "__main__":
    import asyncio

    asyncio.run(wrapper())
