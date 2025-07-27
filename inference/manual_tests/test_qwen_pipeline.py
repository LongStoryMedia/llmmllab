import argparse
import datetime
from time import sleep


def test_qwen_pipeline():
    """
    Test function for the Qwen3-30B-A3B pipeline.
    """
    from pipelines.factory import pipeline_factory
    from models import MessageContent, Message, MessageContentType, MessageRole, ChatReq, ModelParameters

    MODEL_PATH = "qwen-qwen3-30b-a3b"
    messages = [
        Message(
            role=MessageRole.USER,
            content=[
                MessageContent(
                    type=MessageContentType.TEXT,
                    text="Explain the concept of mixture of experts in large language models."
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
            temperature=0.6,  # Higher temperature for more diverse outputs with thinking mode
            top_p=0.95,
            top_k=20,
            num_ctx=2048,
            # num_predict=2048,
        )
    )

    pipe, t = pipeline_factory.get_pipeline(MODEL_PATH)

    # Consume the generator returned by pipe.run()
    result_generator = pipe.run(req, t)
    for response in result_generator:
        print(response.message.content[0].text if response.message.content and len(response.message.content) > 0 and response.message.content[0].text else "", end="", flush=True)
        if response.done:
            # Display thinking content if available
            if hasattr(response.message, 'metadata') and response.message.metadata and 'thinking' in response.message.metadata:
                print("\nThinking content:\n")
                print(response.message.metadata['thinking'])
            print(response)
            break

    sleep(1)  # for testing the pipeline cache
    messages = [
        Message(
            role=MessageRole.USER,
            content=[
                MessageContent(
                    type=MessageContentType.TEXT,
                    text="Can you solve a complex math problem? What is 7863 * 9412?"
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
            temperature=0.1,  # Lower temperature for more deterministic outputs
            top_p=0.95,
            top_k=40,
            num_ctx=1024,
            num_predict=1024,
        )
    )

    pipe, t = pipeline_factory.get_pipeline(MODEL_PATH)

    result_generator = pipe.run(req, t)
    for response in result_generator:
        print(response.message.content[0].text if response.message.content and len(response.message.content) > 0 and response.message.content[0].text else "")
        if response.done:
            # Display thinking content if available
            if hasattr(response.message, 'metadata') and response.message.metadata and 'thinking' in response.message.metadata:
                print("\nThinking content:\n")
                print(response.message.metadata['thinking'])
            print(response)
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test the Qwen3-30B-A3B pipeline")
    args = parser.parse_args()

    try:
        test_qwen_pipeline()
        print("Test completed successfully")
        exit(0)
    except Exception as e:
        print(f"Test failed with error: {e}")
        exit(1)
