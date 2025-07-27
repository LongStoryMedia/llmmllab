import argparse
import datetime
from time import sleep


def test_mixtral_pipeline():
    """
    Test function for the Mixtral-8x7B-Instruct pipeline.
    """
    from pipelines.factory import pipeline_factory
    from models import MessageContent, Message, MessageContentType, MessageRole, ChatReq, ModelParameters

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.utils.quantization_config import BitsAndBytesConfig

    model_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype="float16",
        bnb_4bit_quant_type="nf4",
        llm_int8_threshold=6.0,
        llm_int8_skip_modules=None,
        llm_int8_enable_fp32_cpu_offload=True,
        llm_int8_has_fp16_weight=False
    )

    model = AutoModelForCausalLM.from_pretrained(model_id,
                                                 device_map="auto",
                                                 quantization_config=quant)

    messages = [
        {"role": "user", "content": "What is your favourite condiment?"},
        {"role": "assistant", "content": "Well, I'm quite partial to a good squeeze of fresh lemon juice. It adds just the right amount of zesty flavour to whatever I'm cooking up in the kitchen!"},
        {"role": "user", "content": "Do you have mayonnaise recipes?"}
    ]

    input_ids = tokenizer.apply_chat_template(messages, return_tensors="pt").to(model.device)

    outputs = model.generate(input_ids, max_new_tokens=20)
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))

    # MODEL_PATH = "mistralai-mixtral-8x7b-instruct-v0.1"
    # messages = [
    #     Message(
    #         role=MessageRole.USER,
    #         content=[
    #             MessageContent(
    #                 type=MessageContentType.TEXT,
    #                 text="What are the main advantages of mixture of experts models like Mixtral-8x7B?"
    #             )
    #         ],
    #         id=999,
    #         conversation_id=999,
    #         created_at=datetime.datetime.now(tz=datetime.timezone.utc)
    #     ),
    # ]
    # req = ChatReq(
    #     messages=messages,
    #     model=MODEL_PATH,
    #     stream=True,
    #     options=ModelParameters(
    #         temperature=0.7,
    #         top_p=0.9,
    #         top_k=50,
    #         num_ctx=2048,
    #     )
    # )

    # pipe, t = pipeline_factory.get_pipeline(MODEL_PATH)

    # # Consume the generator returned by pipe.run()
    # result_generator = pipe.run(req, t)
    # for response in result_generator:
    #     print(response.message.content[0].text if response.message.content and len(response.message.content) > 0 and response.message.content[0].text else "")
    #     if response.done:
    #         print(response)
    #         break

    # sleep(1)  # for testing the pipeline cache
    # messages = [
    #     Message(
    #         role=MessageRole.USER,
    #         content=[
    #             MessageContent(
    #                 type=MessageContentType.TEXT,
    #                 text="Write a short poem about artificial intelligence"
    #             )
    #         ],
    #         id=999,
    #         conversation_id=999,
    #         created_at=datetime.datetime.now(tz=datetime.timezone.utc)
    #     ),
    # ]

    # req = ChatReq(
    #     messages=messages,
    #     model=MODEL_PATH,
    #     stream=True,
    #     options=ModelParameters(
    #         temperature=0.8,  # Slightly more creative for poem generation
    #         top_p=0.95,
    #         top_k=40,
    #         num_ctx=2048,
    #     )
    # )

    # pipe, t = pipeline_factory.get_pipeline(MODEL_PATH)

    # result_generator = pipe.run(req, t)
    # for response in result_generator:
    #     print(response.message.content[0].text if response.message.content and len(response.message.content) > 0 and response.message.content[0].text else "")
    #     if response.done:
    #         print(response)
    #         break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test the Mixtral-8x7B-Instruct pipeline")
    args = parser.parse_args()

    try:
        test_mixtral_pipeline()
        print("Test completed successfully")
        exit(0)
    except Exception as e:
        print(f"Test failed with error: {e}")
        exit(1)
