import argparse
import datetime
import logging
from time import sleep


def test_mixtral_gguf_pipeline():
    """
    Test function for the Mixtral-8x7B-Instruct-v0.1 GGUF pipeline.
    Uses the quantized Q4_K_M version of the Mixtral-8x7B-Instruct-v0.1 model.
    """
    from pipelines.factory import pipeline_factory
    from pipelines.helpers import get_content
    from models import MessageContent, Message, MessageRole, ChatReq, ModelParameters, MessageContentType

    # Use the GGUF model ID from the model configuration
    MODEL_PATH = "mixtral-8x7b-instruct-v0.1-q4-k-m"

    logger = logging.getLogger(__name__)
    logger.info(f"Testing Mixtral GGUF model pipeline with model ID: {MODEL_PATH}")

    # Test 1: General explanation about GGUF models
    print(f"Testing Mixtral GGUF model: {MODEL_PATH}")

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

    req = ChatReq(
        messages=messages,
        model=MODEL_PATH,
        stream=True,
        options=ModelParameters(
            temperature=0.7,
            top_p=0.9,
            top_k=40,
            repeat_penalty=1.1,
            num_predict=1024
        )
    )

    pipe, load_time = pipeline_factory.get_pipeline(MODEL_PATH)
    logger.info(f"Pipeline load time: {load_time:.2f} ms")

    print("Running test request...")
    for response in pipe.run(req, load_time):
        if not response.done:
            print(get_content(response.message), end="", flush=True)
        else:
            print("\n\nGeneration complete!")
            print(f"Total tokens: {response.eval_count}")
            print(f"Time: {response.total_duration / 1000:.2f} seconds")
            break

    # Test 2: Creative writing prompt
    print("\n\n" + "="*50)
    print("Testing creative writing prompt...")

    creative_messages = [
        Message(
            role=MessageRole.USER,
            content=[
                MessageContent(
                    type=MessageContentType.TEXT,
                    text="Write a short science fiction story about AI and human cooperation in the year 2150."
                )
            ],
            id=1000,
            conversation_id=1000,
            created_at=datetime.datetime.now(tz=datetime.timezone.utc)
        ),
    ]

    creative_req = ChatReq(
        messages=creative_messages,
        model=MODEL_PATH,
        stream=True,
        options=ModelParameters(
            temperature=0.9,  # Higher temperature for more creative output
            top_p=0.92,
            top_k=60,
            repeat_penalty=1.15,
            num_predict=2048  # Longer output for story
        )
    )

    print("Running creative writing test...")
    for response in pipe.run(creative_req, 0):  # Load time already accounted for
        if not response.done:
            print(get_content(response.message), end="", flush=True)
        else:
            print("\n\nGeneration complete!")
            print(f"Total tokens: {response.eval_count}")
            print(f"Time: {response.total_duration / 1000:.2f} seconds")
            break

    # Optional cleanup to free GPU memory
    try:
        pipeline_factory.clear_cache(MODEL_PATH)
        print(f"Pipeline cache for {MODEL_PATH} cleared")
    except Exception as e:
        print(f"Error clearing pipeline cache: {e}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    parser = argparse.ArgumentParser(description="Test the Mixtral GGUF Pipeline")
    args = parser.parse_args()
    test_mixtral_gguf_pipeline()
