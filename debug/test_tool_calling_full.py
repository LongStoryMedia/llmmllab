#!/usr/bin/env python3
"""
Test OpenAI GPT OSS tool calling with proper model configuration
"""

import sys
import os
import asyncio
import json
import logging

# Add the paths for imports
sys.path.append("/app")
sys.path.append("/app/runner")
sys.path.append("/app/server")


async def test_real_tool_calling():
    """Test tool calling with proper model and pipeline setup"""

    try:
        # Set up logging to see our debugging output
        logging.basicConfig(level=logging.DEBUG)
        logger = logging.getLogger("tool-test")

        # Use the pipeline factory to get a properly configured pipeline
        from runner.pipeline_factory import pipeline_factory, PipelinePriority
        from models.chat_response import ChatResponse
        from models.message import Message
        from models.message_content import MessageContent
        from models.message_content_type import MessageContentType
        from models.message_role import MessageRole

        # Import will be done inline for mock tool

        print("✅ All imports successful")

        # Load an existing model profile
        from server.db import storage

        print("✅ Got pipeline factory")

        # Find a model that uses OpenAI GPT OSS pipeline
        models_config_path = "/app/.models.json"
        if os.path.exists(models_config_path):
            with open(models_config_path, "r") as f:
                models_data = json.load(f)

            # Look for an OpenAI GPT OSS model that's enabled
            gpt_oss_model = None
            for model_data in models_data:
                if model_data.get(
                    "pipeline"
                ) == "OpenAiGptOssPipe" and "disable_reason" not in model_data.get(
                    "details", {}
                ):
                    gpt_oss_model = model_data
                    break

            if not gpt_oss_model:
                # Use the first OpenAI GPT OSS model regardless of disable status
                for model_data in models_data:
                    if model_data.get("pipeline") == "OpenAiGptOssPipe":
                        gpt_oss_model = model_data
                        break

        if gpt_oss_model:
            model_id = gpt_oss_model["id"]
            print(f"✅ Using model: {model_id}")
        else:
            # Fallback to hardcoded model
            model_id = "openai-gpt-oss-20b-uncensored-q5_1"
            print(f"✅ Using fallback model: {model_id}")

        # Create a simple mock tool for testing
        from langchain.tools import BaseTool
        from typing import Type
        from pydantic import BaseModel, Field

        class MockSearchInput(BaseModel):
            query: str = Field(description="Search query")

        class MockWebSearchTool(BaseTool):
            name: str = "web_search"
            description: str = "Mock web search tool for testing"
            args_schema: Type[BaseModel] = MockSearchInput

            def _run(self, query: str) -> str:
                return f"Mock search results for: {query}"

            async def _arun(self, query: str) -> str:
                return f"Mock search results for: {query}"

        tools = [MockWebSearchTool()]
        print(f"✅ Created {len(tools)} mock tools")

        # Create test messages
        messages = [
            Message(
                role=MessageRole.USER,
                content=[
                    MessageContent(
                        type=MessageContentType.TEXT,
                        text="Please search for iPhone 16 models on Amazon using the web_search tool. I need to see real product links.",
                    )
                ],
            )
        ]

        # Test using the stream_pipeline interface
        from runner.pipelines.run import stream_pipeline
        from models.model import Model
        from models.model_profile import ModelProfile, ModelParameters
        from datetime import datetime

        # Create a proper model object based on the config
        if gpt_oss_model:
            model = Model(
                id=gpt_oss_model["id"],
                name=gpt_oss_model["name"],
                model=gpt_oss_model["model"],
                provider=gpt_oss_model["provider"],
                task=gpt_oss_model["task"],
                modified_at=gpt_oss_model["modified_at"],
                size=gpt_oss_model["size"],
                digest=gpt_oss_model["digest"],
                details=gpt_oss_model["details"],
            )
        else:
            # Fallback model
            model = Model(
                id=model_id,
                name=model_id,
                model="test-model-path",
                provider="llama_cpp",
                task="TextToText",
                modified_at=str(datetime.now().date()),
                size=1000000,
                digest="test-digest",
                details={},
            )

        # Create a test profile
        params = ModelParameters(temperature=0.7, top_p=0.9, max_tokens=1000)

        profile = ModelProfile(
            id="test-profile",
            name="Test Profile",
            model_id=model.id,
            parameters=params,
            user_id="test-user",
        )

        print("✅ Model and profile created, getting pipeline from factory...")

        # Use the factory to get a pipeline instance with proper priority
        try:
            with pipeline_factory.pipeline(
                profile, ChatResponse, PipelinePriority.HIGH
            ) as pipeline:
                print("✅ Got pipeline from factory, testing stream_pipeline...")

                # Now test the streaming
                response_chunks = []
                async for chunk in stream_pipeline(messages, pipeline, tools):
                    print(f"📦 Received chunk: {chunk}")
                    response_chunks.append(chunk)

                    # Print any debug info from the chunk
                    if hasattr(chunk, "message") and chunk.message:
                        msg_text = ""
                        if hasattr(chunk.message, "content"):
                            if isinstance(chunk.message.content, list):
                                for content in chunk.message.content:
                                    if hasattr(content, "text"):
                                        msg_text += content.text
                            else:
                                msg_text = str(chunk.message.content)
                        print(f"📝 Message text: {msg_text[:200]}...")

                print(f"✅ Streaming completed! Got {len(response_chunks)} chunks")
                return response_chunks

        except Exception as pipeline_error:
            print(f"❌ Pipeline error: {pipeline_error}")
            import traceback

            traceback.print_exc()
            return None

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()
        return None


if __name__ == "__main__":
    result = asyncio.run(test_real_tool_calling())
    if result:
        print("\n🎉 Tool calling test completed successfully!")
        print(f"Total response chunks: {len(result)}")
    else:
        print("\n💥 Tool calling test failed!")
