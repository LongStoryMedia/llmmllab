#!/usr/bin/env python3
import os
import argparse
import subprocess
import shutil
from pathlib import Path
import sys
import logging
import tempfile

# Add the current directory to the Python path so we can import from sibling scripts
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from download import download_model

# Configure logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def convert_to_gguf(model_path, output_path=None, output_dir=None, output_name=None):
    """
    Convert a Hugging Face model to GGUF format using llama.cpp's convert_hf_to_gguf.py script

    Args:
        model_path (str): Path to the model directory
        output_path (str, optional): Complete output path for the GGUF file
        output_dir (str, optional): Directory to save the output file (overridden by output_path)
        output_name (str, optional): Name for the output file without extension (overridden by output_path)

    Returns:
        str: Path to the generated GGUF file
    """
    if output_path is None:
        # Use model name as default if output_name not provided
        model_name = output_name or os.path.basename(os.path.normpath(model_path))
        
        # Determine output directory
        if output_dir is None:
            output_dir = model_path
        else:
            os.makedirs(output_dir, exist_ok=True)
            
        # Build the complete output path
        output_path = os.path.join(output_dir, f"{model_name}.gguf")

    logger.info(f"Output GGUF will be saved to: {output_path}")
    
    llama_cpp_path = "/llama.cpp"  # Base directory for llama.cpp in the k8s pod

    # Check for convert script in the main directory
    convert_script = os.path.join(llama_cpp_path, "convert_hf_to_gguf.py")
    if not os.path.isfile(convert_script):
        # Check in examples directory as alternative
        convert_script = os.path.join(llama_cpp_path, "examples", "convert_hf_to_gguf.py")
        if not os.path.isfile(convert_script):
            raise FileNotFoundError(
                f"Could not find the 'convert_hf_to_gguf.py' script at '{convert_script}'. "
                f"Please ensure llama.cpp is properly installed in the k8s pod."
            )

    logger.info(f"Using conversion script at: {convert_script}")
    logger.info(f"Converting model at {model_path} to GGUF format...")

    # Run the conversion script
    cmd = [
        sys.executable, convert_script,
        model_path,
        "--outfile", output_path
    ]

    try:
        subprocess.run(cmd, check=True)
        logger.info(f"Successfully converted model to GGUF: {output_path}")
        return output_path
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to convert model to GGUF: {e}")
        raise


def quantize_model(input_gguf, output_path=None, output_dir=None, output_name=None, quant_type=None):
    """
    Quantize a GGUF model using llama.cpp's quantize tool

    Args:
        input_gguf (str): Path to the input GGUF file
        output_path (str, optional): Complete output path for the quantized file
        output_dir (str, optional): Directory to save the output file (overridden by output_path)
        output_name (str, optional): Name for the output file without extension (overridden by output_path)
        quant_type (str): Quantization type (e.g., q4_0, q4_1, q5_0, q5_1, q8_0, etc.)

    Returns:
        str: Path to the quantized GGUF file
    """
    if output_path is None:
        # Use input filename as basis if output_name not provided
        if output_name is None:
            base_name = os.path.basename(input_gguf)
            name_without_ext = os.path.splitext(base_name)[0]
            output_name = f"{name_without_ext}_{quant_type}"
        else:
            output_name = f"{output_name}_{quant_type}"
        
        # Determine output directory
        if output_dir is None:
            output_dir = os.path.dirname(input_gguf)
        else:
            os.makedirs(output_dir, exist_ok=True)
            
        # Build the complete output path
        output_path = os.path.join(output_dir, f"{output_name}.gguf")

    logger.info(f"Quantized output will be saved to: {output_path}")
    
    llama_cpp_path = "/llama.cpp"  # Base directory for llama.cpp in the k8s pod

    # Check for llama-quantize in the build directory first (newer llama.cpp versions)
    quantize_tool = os.path.join(llama_cpp_path, "build", "bin", "llama-quantize")
    if not os.path.isfile(quantize_tool) or not os.access(quantize_tool, os.X_OK):
        # Check for quantize in build/bin (older versions)
        quantize_tool = os.path.join(llama_cpp_path, "build", "bin", "quantize")
        if not os.path.isfile(quantize_tool) or not os.access(quantize_tool, os.X_OK):
            # Fall back to the root directory
            quantize_tool = os.path.join(llama_cpp_path, "quantize")
            if not os.path.isfile(quantize_tool) or not os.access(quantize_tool, os.X_OK):
                raise FileNotFoundError(
                    f"Could not find quantize tool at '{quantize_tool}' or 'llama-quantize'. "
                    f"Please ensure llama.cpp is properly installed and compiled in the k8s pod."
                )

    logger.info(f"Using quantize tool at: {quantize_tool}")
    logger.info(f"Quantizing model {input_gguf} using {quant_type}...")

    # Ensure the quantization type is uppercase for the tool
    quant_type_upper = quant_type.upper()

    # Run the quantization tool
    cmd = [
        quantize_tool,
        input_gguf,
        output_path,
        quant_type_upper
    ]

    try:
        subprocess.run(cmd, check=True)
        logger.info(f"Successfully quantized model to {output_path}")
        return output_path
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to quantize model: {e}")
        raise


def split_gguf_file(input_gguf, output_prefix=None, output_dir=None, output_name=None, max_size=None, max_tensors=None):
    """
    Split a GGUF file into multiple smaller files using llama.cpp's gguf-split tool

    Args:
        input_gguf (str): Path to the input GGUF file
        output_prefix (str, optional): Prefix for the output files
        output_dir (str, optional): Directory to save the output files
        output_name (str, optional): Base name for the output files
        max_size (str, optional): Maximum size per split, e.g. "500M" or "2G"
        max_tensors (int, optional): Maximum number of tensors per split

    Returns:
        str: Path to the first split file
    """
    if output_prefix is None:
        # Use input filename as basis if output_name not provided
        if output_name is None:
            base_name = os.path.basename(input_gguf)
            name_without_ext = os.path.splitext(base_name)[0]
            output_name = name_without_ext
        
        # Determine output directory
        if output_dir is None:
            output_dir = os.path.dirname(input_gguf)
        else:
            os.makedirs(output_dir, exist_ok=True)
            
        # Build the complete output prefix
        output_prefix = os.path.join(output_dir, output_name)

    logger.info(f"Split files will be saved with prefix: {output_prefix}")
    
    llama_cpp_path = "/llama.cpp"  # Base directory for llama.cpp in the k8s pod

    # Check for llama-gguf-split in the build directory first (newer llama.cpp versions)
    split_tool = os.path.join(llama_cpp_path, "build", "bin", "llama-gguf-split")
    if not os.path.isfile(split_tool) or not os.access(split_tool, os.X_OK):
        # Check for gguf-split in build/bin (older versions or different naming)
        split_tool = os.path.join(llama_cpp_path, "build", "bin", "gguf-split")
        if not os.path.isfile(split_tool) or not os.access(split_tool, os.X_OK):
            # Fall back to the root directory
            split_tool = os.path.join(llama_cpp_path, "gguf-split")
            if not os.path.isfile(split_tool) or not os.access(split_tool, os.X_OK):
                raise FileNotFoundError(
                    f"Could not find split tool at '{split_tool}' or 'llama-gguf-split'. "
                    f"Please ensure llama.cpp is properly installed and compiled in the k8s pod."
                )

    logger.info(f"Using gguf-split tool at: {split_tool}")

    # Start building the command with the tool
    cmd = [split_tool]

    # Add options BEFORE the input and output paths
    if max_size:
        cmd.extend(["--split-max-size", max_size])

    if max_tensors:
        cmd.extend(["--split-max-tensors", str(max_tensors)])
    elif not max_size:
        # Only add default if neither max_size nor max_tensors is specified
        cmd.extend(["--split-max-tensors", "128"])  # Default value

    # Add input and output paths LAST
    cmd.extend([input_gguf, output_prefix])

    logger.info(f"Splitting GGUF file {input_gguf} into multiple files with prefix {output_prefix}...")

    try:
        subprocess.run(cmd, check=True)
        # The output files will be named <output_prefix>-00001-of-XXXXX.gguf, <output_prefix>-00002-of-XXXXX.gguf, etc.
        # Return the first split file
        directory = os.path.dirname(output_prefix)
        basename = os.path.basename(output_prefix)

        # Look for files matching the pattern
        split_files = [f for f in os.listdir(directory or '.') if f.startswith(f"{basename}-") and f.endswith(".gguf")]
        if not split_files:
            raise FileNotFoundError(f"No split files found with prefix {output_prefix}")

        # Sort files and get the first one (should be -00001-)
        split_files.sort()
        first_file = os.path.join(directory, split_files[0]) if directory else split_files[0]

        logger.info(f"Successfully split model into {len(split_files)} files. First file: {first_file}")
        return first_file
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to split GGUF file: {e}")
        raise


def merge_lora_weights(base_model_dir, lora_model_id, output_dir=None, revision="main"):
    """
    Merge LoRA weights with a base model using transformers and PEFT

    Args:
        base_model_dir (str): Path to the base model directory
        lora_model_id (str): The LoRA adapter ID on Hugging Face Hub (e.g., 'Heartsync/Flux-NSFW-uncensored')
        output_dir (str, optional): Path to save the merged model. If None, creates a temp directory.
        revision (str): The revision of the LoRA model to download

    Returns:
        str: Path to the directory containing the merged model
    """
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoModelForImageGeneration, AutoModelForVision2Seq
        from peft import PeftModel
    except ImportError:
        logger.error("Required packages not found. Installing transformers, peft, and accelerate...")
        subprocess.run([sys.executable, "-m", "pip", "install", "transformers", "peft", "accelerate"], check=True)
        import torch
        from transformers import AutoModelForCausalLM, AutoModelForImageGeneration, AutoModelForVision2Seq
        from peft import PeftModel

    if output_dir is None:
        output_dir = os.path.join(tempfile.mkdtemp(), "merged_model")
        os.makedirs(output_dir, exist_ok=True)
    else:
        os.makedirs(output_dir, exist_ok=True)

    logger.info(f"Downloading LoRA model from {lora_model_id}...")
    
    # Attempt to detect model type (image generation vs text generation)
    try:
        # First try to load as image generation model
        logger.info("Attempting to load as image generation model...")
        base_model = AutoModelForImageGeneration.from_pretrained(
            base_model_dir,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        logger.info("Successfully loaded as image generation model")
    except Exception as img_err:
        logger.warning(f"Failed to load as image generation model: {img_err}")
        try:
            # Try as vision-language model
            logger.info("Attempting to load as vision-language model...")
            base_model = AutoModelForVision2Seq.from_pretrained(
                base_model_dir,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
            logger.info("Successfully loaded as vision-language model")
        except Exception as vl_err:
            logger.warning(f"Failed to load as vision-language model: {vl_err}")
            # Fall back to text generation model
            logger.info("Attempting to load as text generation model...")
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_dir,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
            logger.info("Successfully loaded as text generation model")

    # Load the LoRA model
    logger.info(f"Loading LoRA adapter: {lora_model_id}")
    lora_model = PeftModel.from_pretrained(
        base_model, 
        lora_model_id,
        revision=revision
    )

    # Merge the LoRA weights into the base model
    logger.info("Merging LoRA weights...")
    merged_model = lora_model.merge_and_unload()

    # Save the merged model
    logger.info(f"Saving merged model to: {output_dir}")
    merged_model.save_pretrained(output_dir)
    
    # Try to save tokenizer if available
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(base_model_dir, trust_remote_code=True)
        tokenizer.save_pretrained(output_dir)
        logger.info("Tokenizer saved successfully")
    except Exception as e:
        logger.warning(f"Could not save tokenizer: {e}")
        
    # Try to save image processor or feature extractor if available
    try:
        from transformers import AutoImageProcessor, AutoFeatureExtractor, AutoProcessor
        
        try:
            processor = AutoProcessor.from_pretrained(base_model_dir, trust_remote_code=True)
            processor.save_pretrained(output_dir)
            logger.info("Processor saved successfully")
        except Exception:
            try:
                image_processor = AutoImageProcessor.from_pretrained(base_model_dir, trust_remote_code=True)
                image_processor.save_pretrained(output_dir)
                logger.info("Image processor saved successfully")
            except Exception:
                try:
                    feature_extractor = AutoFeatureExtractor.from_pretrained(base_model_dir, trust_remote_code=True)
                    feature_extractor.save_pretrained(output_dir)
                    logger.info("Feature extractor saved successfully")
                except Exception:
                    logger.warning("Could not save processor, image processor or feature extractor")
    except Exception as e:
        logger.warning(f"Could not save image processor or feature extractor: {e}")
    
    # Try to save configuration files
    try:
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(base_model_dir, trust_remote_code=True)
        config.save_pretrained(output_dir)
        logger.info("Configuration saved successfully")
    except Exception as e:
        logger.warning(f"Could not save configuration: {e}")

    logger.info("LoRA merged and saved successfully as a full model")
    return output_dir


def cleanup_model_files(model_dir, keep_file, remove_hf_cache=True):
    """
    Delete all files in the model directory except for the specified file to keep
    Optionally delete the HF cache for the model as well

    Args:
        model_dir (str): Path to the model directory
        keep_file (str): Path to the file to keep
        remove_hf_cache (bool): Whether to remove HF cache files
    """
    logger.info(f"Cleaning up original model files in {model_dir}...")
    keep_file_abs = os.path.abspath(keep_file)

    for root, dirs, files in os.walk(model_dir, topdown=False):
        # Remove files except the one to keep
        for file in files:
            file_path = os.path.join(root, file)
            file_abs = os.path.abspath(file_path)
            if file_abs != keep_file_abs:
                try:
                    os.remove(file_path)
                    logger.debug(f"Removed file: {file_path}")
                except Exception as e:
                    logger.warning(f"Failed to remove file {file_path}: {e}")

        # Remove empty directories (except the root directory)
        if root != model_dir:
            try:
                os.rmdir(root)
                logger.debug(f"Removed empty directory: {root}")
            except Exception as e:
                logger.debug(f"Could not remove directory {root}, might not be empty: {e}")

    logger.info(f"Cleanup completed. Kept only: {keep_file}")
    
    # Clean up Hugging Face cache if requested
    if remove_hf_cache:
        try:
            from huggingface_hub.constants import HF_HUB_CACHE
            
            # Get the model name from the directory
            model_name = os.path.basename(os.path.normpath(model_dir))
            
            # Locate the cache directory
            hf_cache_dir = os.environ.get("HF_HOME", os.path.expanduser(HF_HUB_CACHE))
            models_cache = os.path.join(hf_cache_dir, "models--" + model_name.replace("/", "--"))
            
            if os.path.exists(models_cache):
                logger.info(f"Removing Hugging Face cache for model {model_name}...")
                shutil.rmtree(models_cache, ignore_errors=True)
                logger.info(f"Successfully removed HF cache at: {models_cache}")
        except Exception as e:
            logger.warning(f"Failed to clean up Hugging Face cache: {e}")


def download_and_quantize(model_id=None, local_dir=None, use_symlinks=False, revision="main",
                          quant_type=None, cleanup=False, split=False, split_max_size=None, split_max_tensors=None,
                          skip_download=False, lora_model_id=None, lora_revision="main", 
                          model_name=None, output_dir=None, output_file_name=None):
    """
    Download a model from Hugging Face Hub, convert it to GGUF, and optionally quantize and split it

    Args:
        model_id (str, optional): The model ID on Hugging Face Hub (e.g., 'lmsys/vicuna-13b-v1.5'). 
                                  Required unless skip_download is True.
        local_dir (str): Local directory to save the model to or where the model is already located. 
                        If None and model_id is provided, uses the model name.
        use_symlinks (bool): Whether to use symlinks for the downloaded files
        revision (str): The revision to download
        quant_type (str): Quantization type. If None, no quantization is performed
        cleanup (bool): If True, delete original model files after conversion, keeping only the GGUF file
        split (bool): If True, split the GGUF file into multiple smaller files
        split_max_size (str, optional): Maximum size per split, e.g. "500M" or "2G"
        split_max_tensors (int, optional): Maximum number of tensors per split
        skip_download (bool): If True, skip the download step and use model files already at local_dir
        lora_model_id (str, optional): The LoRA adapter ID on Hugging Face Hub to merge with the base model
        lora_revision (str): The revision of the LoRA model to download
        model_name (str, optional): Custom name for the output files (overrides default naming)
        output_dir (str, optional): Directory to save the output GGUF file(s)
        output_file_name (str, optional): Custom name for the output file without extension

    Returns:
        tuple: (Path to the original model directory, Path to the final GGUF file or first split file)
    """
    # Validate arguments
    if skip_download:
        if not local_dir:
            raise ValueError("local_dir must be provided when skip_download is True")
        model_directory = local_dir
        # Try to determine model name from directory name if not provided
        model_base_name = os.path.basename(os.path.normpath(local_dir))
    else:
        if not model_id:
            raise ValueError("model_id must be provided when skip_download is False")
        # Download the model
        download_model(
            model_id=model_id,
            local_dir=local_dir,
            use_symlinks=use_symlinks,
            revision=revision
        )
        # Extract the model name for naming the output files
        model_base_name = model_id.split('/')[-1] if '/' in model_id else model_id
        # Determine the local directory path
        model_directory = local_dir if local_dir is not None else model_base_name

    # Apply custom model name if provided
    if output_file_name:
        output_model_name = output_file_name
    elif model_name:
        output_model_name = model_name
    else:
        output_model_name = model_base_name

    # If LoRA model ID is provided, merge the weights
    if lora_model_id:
        logger.info(f"LoRA model ID provided: {lora_model_id}. Merging weights...")
        # Use original model directory as base
        base_model_dir = model_directory
        # Create merged model directory
        merged_model_dir = os.path.join(os.path.dirname(model_directory), f"{output_model_name}_merged_lora")
        
        try:
            # Merge LoRA weights with base model
            model_directory = merge_lora_weights(
                base_model_dir=base_model_dir,
                lora_model_id=lora_model_id,
                output_dir=merged_model_dir,
                revision=lora_revision
            )
            # Update model name to reflect LoRA merge if output_file_name wasn't explicitly set
            if not output_file_name:
                lora_base_name = lora_model_id.split('/')[-1] if '/' in lora_model_id else lora_model_id
                output_model_name = f"{output_model_name}-with-{lora_base_name}"
            logger.info(f"Successfully merged LoRA weights. New model directory: {model_directory}")
        except Exception as e:
            logger.error(f"Failed to merge LoRA weights: {e}")
            logger.warning("Continuing with base model only")

    # Create output directory if specified and it doesn't exist
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Output files will be saved to directory: {output_dir}")

    # Convert to GGUF
    gguf_file = convert_to_gguf(
        model_directory,
        output_dir=output_dir,
        output_name=output_model_name
    )

    # Quantize if requested
    final_output_path = gguf_file
    if quant_type:
        final_output_path = quantize_model(
            gguf_file,
            output_dir=output_dir,
            output_name=output_model_name,
            quant_type=quant_type
        )
        # Remove the unquantized GGUF file to save space if quantization was successful
        if os.path.exists(final_output_path):
            os.remove(gguf_file)
            logger.info(f"Removed unquantized GGUF file: {gguf_file}")

    # Split the GGUF file if requested
    if split:
        split_output_name = os.path.splitext(os.path.basename(final_output_path))[0]
        split_first_file = split_gguf_file(
            final_output_path,
            output_dir=output_dir,
            output_name=split_output_name,
            max_size=split_max_size,
            max_tensors=split_max_tensors
        )

        # Remove the original GGUF file if splitting was successful
        if os.path.exists(split_first_file):
            os.remove(final_output_path)
            logger.info(f"Removed original GGUF file after splitting: {final_output_path}")
            final_output_path = split_first_file

    # Clean up original model files if requested
    if cleanup:
        cleanup_model_files(model_directory, final_output_path, remove_hf_cache=True)

    return model_directory, final_output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download models from Hugging Face Hub, convert to GGUF, optionally merge LoRA weights, quantize and split")

    # Model source options - either download from HF or use local directory
    model_source_group = parser.add_mutually_exclusive_group(required=True)
    model_source_group.add_argument("--model-id", help="The model ID on Hugging Face Hub (e.g., 'lmsys/vicuna-13b-v1.5')")
    model_source_group.add_argument("--skip-download", action="store_true", help="Skip downloading and use model files at --dir")

    # Other options
    parser.add_argument("--dir", help="Local directory to save the model to or where the model is already located (required with --skip-download)")
    parser.add_argument("--use-symlinks", action="store_true", help="Enable use of symlinks for downloaded files")
    parser.add_argument("--revision", default="main", help="The revision of the base model to download (default: 'main')")
    parser.add_argument("--model-name", help="Custom name for the output files (overrides default naming from model ID or directory)")
    parser.add_argument("--quantize", help=("Quantization method to apply. Available options: "
                                            "q4_0, q4_1, q5_0, q5_1, q8_0, q2_k, q3_k_s, q3_k_m, q3_k_l, "
                                            "q4_k_s, q4_k_m, q5_k_s, q5_k_m, q6_k"))
    parser.add_argument("--cleanup", action="store_true", help="Delete original model files after conversion, keeping only the GGUF file")

    # New output options
    parser.add_argument("--output-dir", help="Directory to save the output GGUF file(s)")
    parser.add_argument("--output-file-name", help="Custom name for the output file without extension")

    # LoRA options
    parser.add_argument("--lora-model-id", help="The LoRA adapter ID on Hugging Face Hub to merge with the base model")
    parser.add_argument("--lora-revision", default="main", help="The revision of the LoRA model to download (default: 'main')")

    # Split options
    parser.add_argument("--split", action="store_true", help="Split the GGUF file into multiple smaller files")
    parser.add_argument("--split-max-size", help="Maximum size per split, e.g. '500M' or '2G'")
    parser.add_argument("--split-max-tensors", type=int, help="Maximum number of tensors per split (default: 128)")

    args = parser.parse_args()

    # Validate arguments
    if args.skip_download and not args.dir:
        parser.error("--dir must be provided when using --skip-download")

    # Download or use local model, convert, and optionally quantize and split
    model_dir, final_path = download_and_quantize(
        model_id=None if args.skip_download else args.model_id,
        local_dir=args.dir,
        use_symlinks=args.use_symlinks,
        revision=args.revision,
        quant_type=args.quantize,
        cleanup=args.cleanup,
        split=args.split,
        split_max_size=args.split_max_size,
        split_max_tensors=args.split_max_tensors,
        skip_download=args.skip_download,
        lora_model_id=args.lora_model_id,
        lora_revision=args.lora_revision,
        model_name=args.model_name,
        output_dir=args.output_dir,
        output_file_name=args.output_file_name
    )

    print("\nProcess completed successfully!")
    print(f"Model directory: {model_dir}")
    print(f"Final GGUF file: {final_path}")
