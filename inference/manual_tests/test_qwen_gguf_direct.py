#!/usr/bin/env python3
"""
Test script for the Qwen GGUF pipeline.
Uses the quantized Q4_K_M version of the Qwen3-30B-A3B model.
"""
import datetime
import logging
from typing import Optional, List

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_qwen_gguf_pipeline():
    """
    Test function for the Qwen GGUF pipeline.
    Uses the quantized Q4_K_M version of the Qwen3-30B-A3B model.
    """
    from pipelines.factory import pipeline_factory
    from models import MessageContent, Message, MessageRole, ChatReq, ModelParameters, MessageContentType
    from pipelines.txt2txt.qwen30a3b_q4km import QwenGGUFPipe
    from models import Model, ModelDetails

    print("Testing GGUF model: qwen3-30b-a3b-q4-k-m")

    # Create a model definition directly
    model_def = Model(
        id="qwen3-30b-a3b-q4-k-m",
        name="unsloth/Qwen3-30B-A3B-GGUF",
        model="unsloth/Qwen3-30B-A3B-GGUF",
        pipeline="Qwen30A3BQ4KMPipe",
        modified_at="2025-07-20",
        size=30500000000,
        digest="qwen3-30b-a3b-20250720",
        details=ModelDetails(
            parent_model="Qwen/Qwen3-30B-A3B",
            gguf_file="/models/Qwen3-30B-A3B-GGUF/Qwen3-30B-A3B-Q4_K_M.gguf",
            format="gguf",
            family="qwen",
            families=["Qwen", "MoE"],
            parameter_size="30.5B",
            quantization_level="Q4_K_M",
            dtype="BF16",
            specialization="TextToText"
        ),
        task="TextToText"
    )

    try:
        # Create pipeline directly
        pipeline = QwenGGUFPipe(model_def)

        # Create a simple test message
        messages = [
            Message(
                role=MessageRole.USER,
                content=[
                    MessageContent(
                        type=MessageContentType.TEXT,
                        text="Explain the benefits of using GGUF quantized models for large language model inference."
                    )
                ],
                id=999,
                conversation_id=999,
                created_at=datetime.datetime.now(tz=datetime.timezone.utc)
            ),
        ]

        # Create a request
        req = ChatReq(
            messages=messages,
            model="qwen3-30b-a3b-q4-k-m",
            stream=True,  # Add stream parameter
            options=ModelParameters(
                temperature=0.7,
                top_p=0.95,
                top_k=40,
                num_predict=100  # Keep it short for testing
            )
        )

        # Test the pipeline
        print("Running test inference...")
        start_time = datetime.datetime.now()
        generator = pipeline.run(req, load_time=(datetime.datetime.now() - start_time).total_seconds())

        # Print the first few tokens
        for i, response in enumerate(generator):
            if i < 5:  # Just show the first 5 chunks
                # Handle different response structures
                try:
                    if response.message and response.message.content and len(response.message.content) > 0:
                        print(f"Chunk {i}: {response.message.content[0].text}")
                    else:
                        print(f"Chunk {i}: {response}")
                except Exception as e:
                    print(f"Error processing chunk {i}: {e}")
                    print(f"Raw response: {response}")
            else:
                # Skip remaining chunks but still process them
                pass

        print("Test completed successfully!")

    except Exception as e:
        print(f"Test failed with error: {str(e)}")


if __name__ == "__main__":
    # Fix imports
    from models import MessageContentType
    test_qwen_gguf_pipeline()
