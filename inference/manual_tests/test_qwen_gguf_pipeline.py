import argparse
import datetime
import logging
from time import sleep


def test_qwen_gguf_pipeline():
    """
    Test function for the Qwen GGUF pipeline.
    Uses the quantized Q4_K_M version of the Qwen3-30B-A3B model.
    """
    from pipelines.factory import pipeline_factory
    from models import MessageContent, Message, MessageRole, ChatReq, ModelParameters, MessageContentType

    # Use the GGUF model ID from the model configuration
    MODEL_PATH = "qwen3-30b-a3b-q4-k-m"

    logger = logging.getLogger(__name__)
    logger.info(f"Testing GGUF model pipeline with model ID: {MODEL_PATH}")

    # Test 1: General explanation about GGUF models
    print(f"Testing GGUF model: {MODEL_PATH}")

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
            top_p=0.95,
            top_k=40,
            num_predict=100  # Keep it short for testing
        )
    )

    # Get the pipeline from the factory
    pipe, load_time = pipeline_factory.get_pipeline(MODEL_PATH)

    # Run the pipeline and consume the generator
    print("Running test inference...")
    result_generator = pipe.run(req, load_time)
    for response in result_generator:
        print(response.message.content[0].text if response.message.content and len(response.message.content) > 0 and response.message.content[0].text else "", end="", flush=True)
        if response.done:
            # Display thinking content if available
            if hasattr(response.message, 'metadata') and response.message.metadata and 'thinking' in response.message.metadata:
                print("\n\nThinking content:\n")
                print(response.message.metadata['thinking'])
            print("\n\nFinal response metrics:")
            print(f"Total duration: {response.total_duration:.2f}ms")
            print(f"Load duration: {response.load_duration:.2f}ms")
            print(f"Prompt eval duration: {response.prompt_eval_duration:.2f}s")
            print(f"Eval duration: {response.eval_duration:.2f}ms")
            print(f"Prompt tokens: {response.prompt_eval_count}")
            print(f"Completion tokens: {response.eval_count}")
            break

    # Test 2: Mathematical reasoning
    sleep(1)  # Short pause between requests
    print("\n\n--- Testing mathematical reasoning with GGUF model ---\n")

    messages = [
        Message(
            role=MessageRole.USER,
            content=[
                MessageContent(
                    type=MessageContentType.TEXT,
                    text="If I have a rectangle with length 12.5 meters and width 7.8 meters, what is its area? Also calculate the perimeter."
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
            temperature=0.2,  # Lower temperature for more deterministic reasoning
            top_p=0.95,
            top_k=40,
            num_ctx=1024,
            num_predict=512,  # Shorter response expected for this type of question
        )
    )

    # Get the pipeline again - it should use the cached version
    pipe, load_time = pipeline_factory.get_pipeline(MODEL_PATH)

    # Run the pipeline
    result_generator = pipe.run(req, load_time)
    for response in result_generator:
        print(response.message.content[0].text if response.message.content and len(response.message.content) > 0 and response.message.content[0].text else "", end="", flush=True)
        if response.done:
            # Display thinking content if available
            if hasattr(response.message, 'metadata') and response.message.metadata and 'thinking' in response.message.metadata:
                print("\n\nThinking content:\n")
                print(response.message.metadata['thinking'])
            print("\n\nFinal response metrics:")
            print(f"Total duration: {response.total_duration:.2f}ms")
            print(f"Load duration: {response.load_duration:.2f}ms")
            print(f"Prompt eval duration: {response.prompt_eval_duration:.2f}s")
            print(f"Eval duration: {response.eval_duration:.2f}ms")
            print(f"Prompt tokens: {response.prompt_eval_count}")
            print(f"Completion tokens: {response.eval_count}")
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test the Qwen GGUF pipeline")
    args = parser.parse_args()

    # Set up logging
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)

    try:
        logger.info("Starting Qwen GGUF pipeline test")
        test_qwen_gguf_pipeline()
        logger.info("Test completed successfully")
        print("\nTest completed successfully")
        exit(0)
    except Exception as e:
        logger.error(f"Test failed with error: {str(e)}", exc_info=True)
        print(f"\nTest failed with error: {str(e)}")
        exit(1)
