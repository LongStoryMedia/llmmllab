"""Test PlanningIntentSubgraph as main graph with LlamaCpp pipeline."""

import datetime
import os
import sys
from typing import List, Optional

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
from composer.agents import ClassifierAgent
from composer.graph.subgraphs import PlanningIntentSubgraph
from composer.graph.state import WorkflowState
from composer import execute_workflow
from utils.message_conversion import messages_to_lc_messages
from utils.logging import llmmllogger, serialize_event_data

logger = llmmllogger.bind(component="test_planning_intent_subgraph")

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

    logger.info("🚀 Starting PlanningIntentSubgraph test")

    # Instantiate pipeline factory (local cache mode)
    # Retrieve a default profile (assumes DEFAULT_MODEL_PROFILES contains profile for model)
    profile = get_profile(model_id)
    logger.info(f"📊 Using model profile: {profile.model_name}")

    # Use ClassifierAgent for intent analysis
    classifier_agent = ClassifierAgent(
        pipeline_factory,
        profile,
        NodeMetadata(
            node_name="debug_classifier",
            node_id="debug_classifier_001",
            node_type="debug",
        ),
    )

    logger.info("🤖 ClassifierAgent initialized for planning intent")

    content = []

    if image_url or (not query and not image_url):
        if not image_url:
            image_url = TEST_IMAGE_URL
        content.append(MessageContent(type=MessageContentType.IMAGE, url=image_url))
    content.append(
        MessageContent(
            type=MessageContentType.TEXT,
            text=query
            or "Can you help me understand how neural networks work? I need to research the latest developments in transformer architectures and analyze their performance on different tasks.",
        )
    )

    # Create test messages
    test_messages = [
        Message(
            role=MessageRole.USER,
            content=content,
        )
    ]

    # Create PlanningIntentSubgraph
    planning_intent_subgraph = PlanningIntentSubgraph(classifier_agent)
    logger.info("🔍 PlanningIntentSubgraph initialized")

    # Initialize database for intent analysis storage
    try:
        from db import storage

        await storage.initialize(
            "postgresql://lsm:@psql.psql.svc.cluster.local:5432/llmmll"
        )
        logger.info("💾 Database initialized for intent analysis storage")
    except Exception as e:
        logger.warning(
            f"Database initialization failed: {e} - continuing without storage"
        )

    user_config = create_default_user_config(user_id="test_user")
    # Create initial WorkflowState
    workflow_state = WorkflowState(
        messages=test_messages,
        user_id="test_user",
        conversation_id=717,
        user_config=user_config,
    )

    logger.info("🎯 Starting PlanningIntentSubgraph streaming execution...")

    # Execute the subgraph with streaming
    try:
        if not planning_intent_subgraph.graph:
            logger.error("❌ PlanningIntentSubgraph graph not initialized")
            return

        print("\n" + "=" * 80)
        print("STREAMING PLANNING INTENT SUBGRAPH EXECUTION")
        print("=" * 80)
        print()

        # Execute the graph directly using ainvoke since this subgraph doesn't stream
        final_state = await planning_intent_subgraph.graph.ainvoke(workflow_state)

        print("🔍 INTENT ANALYSIS RESULTS:")
        print("-" * 40)

        # Check both the final_state object and its attributes
        if (
            hasattr(final_state, "intent_classification")
            and final_state.intent_classification
        ):
            for i, intent in enumerate(final_state.intent_classification):
                print(f"\nIntent Analysis {i+1}:")
                print(f"  Workflow Type: {intent.workflow_type}")
                print(f"  Complexity Level: {intent.complexity_level}")
                print(f"  Confidence: {intent.confidence}")
                print(f"  Technical Domain: {intent.technical_domain}")
                print(f"  Requires Tools: {intent.requires_tools}")
                print(f"  Requires Custom Tools: {intent.requires_custom_tools}")
                print(f"  Tool Complexity Score: {intent.tool_complexity_score}")
                print(f"  Domain Specificity: {intent.domain_specificity}")
                print(f"  Reusability Potential: {intent.reusability_potential}")
                print(f"  Response Format: {intent.response_format}")
                print(
                    f"  Required Capabilities: {[cap.value for cap in intent.required_capabilities]}"
                )
                print(
                    f"  Computational Requirements: {intent.computational_requirements.value}"
                )
        elif isinstance(final_state, dict) and "intent_classification" in final_state:
            # Handle case where final_state is a dict
            intents = final_state["intent_classification"]
            if intents:
                for i, intent in enumerate(intents):
                    print(f"\nIntent Analysis {i+1}:")
                    print(f"  Workflow Type: {intent.workflow_type}")
                    print(f"  Complexity Level: {intent.complexity_level}")
                    print(f"  Confidence: {intent.confidence}")
                    print(f"  Technical Domain: {intent.technical_domain}")
                    print(f"  Requires Tools: {intent.requires_tools}")
                    print(f"  Requires Custom Tools: {intent.requires_custom_tools}")
                    print(f"  Tool Complexity Score: {intent.tool_complexity_score}")
                    print(f"  Domain Specificity: {intent.domain_specificity}")
                    print(f"  Reusability Potential: {intent.reusability_potential}")
                    print(f"  Response Format: {intent.response_format}")
                    print(
                        f"  Required Capabilities: {[cap.value for cap in intent.required_capabilities]}"
                    )
                    print(
                        f"  Computational Requirements: {intent.computational_requirements.value}"
                    )
            else:
                print("No intent classification results found in dict")
        else:
            print(
                f"No intent classification results found. State type: {type(final_state)}"
            )
            if hasattr(final_state, "__dict__"):
                print(f"Available attributes: {list(final_state.__dict__.keys())}")
            elif isinstance(final_state, dict):
                print(f"Available keys: {list(final_state.keys())}")

        print("\n📝 TODO GENERATION RESULTS:")
        print("-" * 40)

        # Check todos in both formats
        todos = None
        if hasattr(final_state, "generated_todos"):
            todos = final_state.generated_todos
        elif isinstance(final_state, dict) and "generated_todos" in final_state:
            todos = final_state["generated_todos"]

        if todos:
            for i, todo in enumerate(todos):
                print(f"\nTodo {i+1}:")
                print(f"  Title: {todo.title}")
                print(f"  Description: {todo.description}")
                print(f"  Priority: {todo.priority}")
                print(f"  Status: {todo.status}")
        else:
            print("No todos generated")

        print("\n" + "=" * 80)
        print("PLANNING INTENT SUBGRAPH EXECUTION COMPLETED")
        print("=" * 80 + "\n")

        logger.info("🎉 Planning intent test completed successfully")

    except Exception as e:
        logger.error(f"❌ PlanningIntentSubgraph execution failed: {e}")
        import traceback

        traceback.print_exc()
        raise

    finally:
        # CRITICAL: Clean up agent to prevent memory leaks
        try:
            classifier_agent.cleanup()
            logger.info("✅ ClassifierAgent cleanup completed - pipelines unlocked")
        except Exception as e:
            logger.error(f"❌ ClassifierAgent cleanup failed: {e}")

    # print("\n[debug] Completed PlanningIntentSubgraph test without immediate crash")


if __name__ == "__main__":
    import asyncio
    import argparse

    parser = argparse.ArgumentParser(description="Run a planning intent subgraph test.")
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
