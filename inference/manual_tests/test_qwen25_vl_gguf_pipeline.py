#!/usr/bin/env python3
import argparse
import datetime
import logging
from time import sleep


def test_qwen25_vl_gguf_pipeline():
    """
    Test function for the Qwen 2.5 VL GGUF pipeline.
    Uses the quantized Q4_K_M version of the Qwen2.5-VL-72B-Instruct model.
    """
    from pipelines.factory import pipeline_factory
    from models import MessageContent, Message, MessageRole, ChatReq, ModelParameters, MessageContentType

    # Use the GGUF model ID from the model configuration
    MODEL_PATH = "qwen2.5-vl-32b-instruct-q4-k-m"
    logger = logging.getLogger(__name__)

    # Test 1: Image description task
    print(f"\nTest 1: Image description with Qwen 2.5 VL GGUF model")

    messages = [
        Message(
            role=MessageRole.USER,
            content=[
                MessageContent(
                    type=MessageContentType.IMAGE,
                    url="https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"
                ),
                MessageContent(
                    type=MessageContentType.TEXT,
                    text="Describe this image in detail."
                )
            ],
            id=999,
            conversation_id=999,
            created_at=datetime.datetime.now(tz=datetime.timezone.utc)
        ),
    ]

    req = ChatReq(
        messages=messages,
        model=MODEL_PATH,
        stream=True,
        options=ModelParameters(
            temperature=0.7,
            top_p=0.95,
            top_k=40,
            num_predict=256
        )
    )

    # Get the pipeline from the factory and run it
    pipe, load_time = pipeline_factory.get_pipeline(MODEL_PATH)

    # Run the pipeline
    print("Running image description inference...")
    result_generator = pipe.run(req, load_time)
    for response in result_generator:
        print(
            response.message.content[0].text if response.message.content and response.message.content[0].text else "", end="", flush=True)
        if response.done:
            print("\n\nFinal response metrics:")
            print(f"Total duration: {response.total_duration:.2f}ms")
            print(f"Load duration: {response.load_duration:.2f}ms")
            print(
                f"Prompt eval duration: {response.prompt_eval_duration:.2f}ms")
            print(f"Eval duration: {response.eval_duration:.2f}ms")
            print(f"Prompt tokens: {response.prompt_eval_count}")
            print(f"Completion tokens: {response.eval_count}")
            break

    # Test 2: Follow-up question
    sleep(1)
    print("\n\nTest 2: Follow-up question (text only)")

    messages = [
        Message(
            role=MessageRole.USER,
            content=[
                MessageContent(
                    type=MessageContentType.TEXT,
                    text="What are grayscale images used for in photography?"
                )
            ],
            id=1001,
            conversation_id=999,
            created_at=datetime.datetime.now(tz=datetime.timezone.utc)
        ),
    ]

    req = ChatReq(
        messages=messages,
        model=MODEL_PATH,
        stream=True,
        options=ModelParameters(
            temperature=0.7,
            top_p=0.95,
            top_k=40,
            num_predict=256
        )
    )

    # Use the cached pipeline
    pipe, load_time = pipeline_factory.get_pipeline(MODEL_PATH)

    # Run the pipeline
    print("Running follow-up question inference...")
    result_generator = pipe.run(req, load_time)
    for response in result_generator:
        print(
            response.message.content[0].text if response.message.content and response.message.content[0].text else "", end="", flush=True)
        if response.done:
            print("\n\nFinal response metrics:")
            print(f"Total duration: {response.total_duration:.2f}ms")
            print(f"Load duration: {response.load_duration:.2f}ms")
            print(
                f"Prompt eval duration: {response.prompt_eval_duration:.2f}ms")
            print(f"Eval duration: {response.eval_duration:.2f}ms")
            print(f"Prompt tokens: {response.prompt_eval_count}")
            print(f"Completion tokens: {response.eval_count}")
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test the Qwen 2.5 VL GGUF pipeline")
    args = parser.parse_args()

    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)

    try:
        logger.info("Starting Qwen 2.5 VL GGUF pipeline test")
        test_qwen25_vl_gguf_pipeline()
        logger.info("Test completed successfully")
        print("\nTest completed successfully")
        exit(0)
    except Exception as e:
        logger.error(f"Test failed with error: {str(e)}", exc_info=True)
        print(f"\nTest failed with error: {str(e)}")
        exit(1)
