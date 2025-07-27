from json import load
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
import datetime
from time import sleep


def test_mixtral_pipeline():
    """
    Test function for the Mixtral-8x7B-Instruct pipeline.
    """
    from pipelines.factory import pipeline_factory
    from models import MessageContent, Message, MessageContentType, MessageRole, ChatReq, ModelParameters

    from transformers import Qwen3MoeForCausalLM, AutoTokenizer
    from transformers.utils.quantization_config import BitsAndBytesConfig

    model_name = "Qwen/Qwen3-30B-A3B"

    # load the tokenizer and the model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    quant = BitsAndBytesConfig(
        load_in_4bit=False,
        load_in_8bit=False,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="fp4",
        llm_int8_threshold=6.0,
        llm_int8_skip_modules=None,
        llm_int8_enable_fp32_cpu_offload=True,
        llm_int8_has_fp16_weight=True
    )

    model = Qwen3MoeForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        quantization_config=quant
    )
    print(f"Device map: {model.hf_device_map}")

    # prepare the model input
    prompt = "Give me a short introduction to large language model."
    messages = [
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True  # Switches between thinking and non-thinking modes. Default is True.
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    # conduct text completion
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=32768
    )
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()

    # parsing thinking content
    try:
        # rindex finding 151668 (</think>)
        index = len(output_ids) - output_ids[::-1].index(151668)
    except ValueError:
        index = 0

    thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
    content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")

    print("thinking content:", thinking_content)
    print("content:", content)

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
