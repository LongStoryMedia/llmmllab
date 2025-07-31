#!/usr/bin/env python3
"""
This example script demonstrates how to download, quantize and upload a GGUF model
using download_and_quantize.py and upload_gguf.py
"""
import os
import subprocess
import argparse
import logging
import sys

# Add the current directory to the Python path so we can import from sibling scripts
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Configure logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Download, quantize, and upload a GGUF model")
    
    # Model selection options
    parser.add_argument("--model-id", required=True, 
                        help="The model ID on Hugging Face Hub (e.g., 'lmsys/vicuna-13b-v1.5')")
    parser.add_argument("--output-dir", 
                        help="Directory to save the output GGUF file(s)")
    parser.add_argument("--output-file-name", 
                        help="Custom name for the output file without extension")
    
    # Quantization options
    parser.add_argument("--quantize", default="q4_k_m",
                        help="Quantization method to apply (default: q4_k_m)")
    
    # Upload options
    parser.add_argument("--upload-repo", 
                        help="Name of the repository to create or update on Hugging Face Hub")
    parser.add_argument("--organization", 
                        help="Organization name for the repository")
    parser.add_argument("--private", action="store_true", 
                        help="Make the repository private")
    parser.add_argument("--description", 
                        help="Description of the model")
    
    # HF authentication options
    parser.add_argument("--token", 
                        help="Hugging Face token")
    parser.add_argument("--token-file", 
                        help="Path to file containing Hugging Face token")
    
    # Additional options
    parser.add_argument("--cleanup", action="store_true", 
                        help="Delete original model files after conversion")
    parser.add_argument("--no-upload", action="store_true", 
                        help="Skip uploading to Hugging Face Hub")
    
    args = parser.parse_args()
    
    # Create output directory if not specified
    if not args.output_dir:
        model_name = args.model_id.split('/')[-1] if '/' in args.model_id else args.model_id
        args.output_dir = os.path.join(os.getcwd(), f"{model_name}_gguf")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Step 1: Download and quantize the model
    logger.info(f"Starting download and quantization of model: {args.model_id}")
    
    script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "download_and_quantize.py")
    download_cmd = [
        sys.executable, script_path,
        "--model-id", args.model_id,
        "--output-dir", args.output_dir,
        "--quantize", args.quantize
    ]
    
    if args.output_file_name:
        download_cmd.extend(["--output-file-name", args.output_file_name])
    
    if args.cleanup:
        download_cmd.append("--cleanup")
    
    logger.info(f"Running command: {' '.join(download_cmd)}")
    subprocess.run(download_cmd, check=True)
    
    # Find the generated GGUF file(s)
    gguf_files = [os.path.join(args.output_dir, f) for f in os.listdir(args.output_dir) if f.endswith('.gguf')]
    
    if not gguf_files:
        logger.error(f"No GGUF files found in {args.output_dir}")
        return 1
    
    logger.info(f"Found {len(gguf_files)} GGUF file(s): {', '.join(os.path.basename(f) for f in gguf_files)}")
    
    # Step 2: Upload the model (if requested)
    if not args.no_upload:
        if not args.upload_repo:
            # Use model name as repository name if not specified
            model_name = args.model_id.split('/')[-1] if '/' in args.model_id else args.model_id
            quant_suffix = f"-{args.quantize}" if args.quantize else ""
            args.upload_repo = f"{model_name}{quant_suffix}-gguf"
            logger.info(f"No repository name specified, using: {args.upload_repo}")
        
        logger.info(f"Starting upload of GGUF file(s) to repository: {args.upload_repo}")
        
        script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "upload_gguf.py")
        upload_cmd = [
            sys.executable, script_path,
            "--files"
        ] + gguf_files + [
            "--repo-name", args.upload_repo
        ]
        
        if args.organization:
            upload_cmd.extend(["--organization", args.organization])
        
        if args.private:
            upload_cmd.append("--private")
        
        if args.description:
            upload_cmd.extend(["--description", args.description])
        
        if args.token:
            upload_cmd.extend(["--token", args.token])
        
        if args.token_file:
            upload_cmd.extend(["--token-file", args.token_file])
        
        # Use custom model card template if it exists
        template_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_card_template.md")
        if os.path.exists(template_path):
            upload_cmd.extend(["--model-card-template", template_path])
        
        logger.info(f"Running command: {' '.join(upload_cmd)}")
        subprocess.run(upload_cmd, check=True)
    
    logger.info("Process completed successfully!")
    
    return 0

if __name__ == "__main__":
    exit(main())
