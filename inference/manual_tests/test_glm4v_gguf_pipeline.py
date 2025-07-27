import argparse
import datetime
from time import sleep
from pipelines.factory import pipeline_factory
from models import MessageContent, Message, MessageContentType, MessageRole, ChatReq, ModelParameters


def test_glm4v_gguf_pipeline():
    """
    Test function for the GLM4V GGUF pipeline.
    """

    MODEL_PATH = "thudm-glm-4.1v-9b-thinking-bf16"
    messages = [
        Message(
            role=MessageRole.USER,
            content=[
                MessageContent(
                    type=MessageContentType.IMAGE,
                    url="https://upload.wikimedia.org/wikipedia/commons/f/fa/Grayscale_8bits_palette_sample_image.png"
                ),
                MessageContent(
                    type=MessageContentType.TEXT,
                    text="describe this image"
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
            temperature=0.1,
            top_p=0.95,
            top_k=40,
            num_ctx=8192,
        )
    )

    pipe, t = pipeline_factory.get_pipeline(MODEL_PATH)

    # Consume the generator returned by pipe.run()
    result_generator = pipe.run(req, t)
    for response in result_generator:
        print(response.message.content[0].text if response.message.content and len(response.message.content) > 0 and response.message.content[0].text else "",
              end="", flush=True)
        if response.done:
            print(response)
            break

    sleep(1)  # for testing the pipeline cache
    messages = [
        Message(
            role=MessageRole.USER,
            content=[
                MessageContent(
                    type=MessageContentType.TEXT,
                    text="What is the difference between your GGUF version and the regular version?"
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
            temperature=0.1,
            top_p=0.95,
            top_k=40,
            num_ctx=8192,
        )
    )

    pipe, t = pipeline_factory.get_pipeline(MODEL_PATH)

    result_generator = pipe.run(req, t)
    for response in result_generator:
        print(response.message.content[0].text if response.message.content and len(response.message.content) > 0 and response.message.content[0].text else "",
              end="", flush=True)
        if response.done:
            print(response)
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test the GLM4V GGUF pipeline")
    args = parser.parse_args()

    try:
        test_glm4v_gguf_pipeline()
        print("Test completed successfully")
        exit(0)
    except Exception as e:
        print(f"Test failed with error: {e}")
        exit(1)
