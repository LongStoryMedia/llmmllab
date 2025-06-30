from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils.quantization_config import BitsAndBytesConfig
import torch
import gc
import os
import contextlib
import time
from config import logger  # Import logger from config

from fastapi import APIRouter, HTTPException
from typing import Optional, List, Dict, Any

router = APIRouter(
    prefix="/chat",
    tags=["chat"]
)


@router.get("/")
async def list_loras():
    """Test output."""
    start_time = time.time()  # Start time tracking
    logger.info("Starting chat endpoint execution")

    model_name = "Qwen/Qwen3-30B-A3B"
    logger.info(f"Loading model: {model_name}")

    # Clear CUDA cache to free up memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

    # Configure 4-bit quantization for memory efficiency
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    # load the tokenizer and the model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model_load_start = time.time()  # Track model loading time
    logger.info("Starting tokenizer and model loading")

    # Set environment variables to control memory usage
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

    try:
        # Load model with appropriate settings based on GPU capabilities
        load_kwargs = {
            "model_name": model_name,
            "torch_dtype": torch.bfloat16,  # Use float16 instead of auto
            "device_map": "auto",  # Let the library decide the optimal device mapping
            "quantization_config": quantization_config,  # Apply 4-bit quantization
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,  # Optimize CPU memory usage during loading
            "offload_folder": "offload_folder",  # Folder for offloading weights
            "attn_implementation": "eager"
        }

        # AutoModelForCausalLM.from_pretrained expects the model name as first arg, not in kwargs
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=load_kwargs["torch_dtype"],
            device_map=load_kwargs["device_map"],
            quantization_config=load_kwargs["quantization_config"],
            trust_remote_code=load_kwargs["trust_remote_code"],
            low_cpu_mem_usage=load_kwargs["low_cpu_mem_usage"],
            offload_folder=load_kwargs["offload_folder"],
            **({"attn_implementation": load_kwargs["attn_implementation"]} if "attn_implementation" in load_kwargs else {})
        )

    except (RuntimeError, ValueError, OSError) as e:
        logger.error(f"Error loading model: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")

    # Log model loading time
    model_load_time = time.time() - model_load_start
    logger.info(f"Model and tokenizer loaded in {model_load_time:.2f} seconds")

    # prepare the model input
    prompt = "tell me a long boring story."
    messages = [
        {"role": "user", "content": prompt}
    ]

    logger.info("Starting prompt processing")
    tokenization_start = time.time()

    try:
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True  # Switches between thinking and non-thinking modes. Default is True.
        )

        tokenization_time = time.time() - tokenization_start
        logger.info(f"Chat template applied in {tokenization_time:.2f} seconds")

        # Move inputs to the same device as model, chunk by chunk if needed
        model_inputs = tokenizer([text], return_tensors="pt")

        # Move to device safely
        for key in model_inputs:
            if isinstance(model_inputs[key], torch.Tensor):
                model_inputs[key] = model_inputs[key].to(model.device)

        # Reduce max_new_tokens to prevent OOM
        # conduct text completion with memory-efficient settings
        generation_start = time.time()
        logger.info("Starting text generation")

        with torch.inference_mode(), contextlib.nullcontext():
            # Removing autocast since it's causing issues
            generate_kwargs = {
                'max_new_tokens': 32768,  # Reduced from 32768 to prevent OOM
                'do_sample': False,      # Use greedy decoding to save memory
                'pad_token_id': tokenizer.eos_token_id,
                # Specific parameters that are valid for this model
                'repetition_penalty': 1.1
            }

            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=generate_kwargs['max_new_tokens'],
                do_sample=generate_kwargs['do_sample'],
                pad_token_id=generate_kwargs['pad_token_id'],
                repetition_penalty=generate_kwargs['repetition_penalty'],
            )

        generation_time = time.time() - generation_start
        logger.info(f"Text generation completed in {generation_time:.2f} seconds")

        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()

        # parsing thinking content
        try:
            # rindex finding 151668 (</think>)
            index = len(output_ids) - output_ids[::-1].index(151668)
        except ValueError:
            index = 0

        thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
        content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")

        logger.info("Thinking content and main content separated successfully")

        # Clean up to free memory
        del generated_ids, model_inputs, output_ids
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

        end_time = time.time()
        execution_time = end_time - start_time
        logger.info(f"Chat endpoint executed in {execution_time:.2f} seconds")

        return {
            "thinking_content": thinking_content,
            "content": content
        }

    except Exception as e:
        # Clean up resources in case of error
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
        error_time = time.time()
        execution_time = error_time - start_time
        logger.error(f"Error during generation: {e}, execution time until error: {execution_time:.2f} seconds")
        raise HTTPException(status_code=500, detail=f"Error generating response: {str(e)}")
