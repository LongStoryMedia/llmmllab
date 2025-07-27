#!/usr/bin/env python3
import os
import sys
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
import json
import re

# Configure logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def setup_huggingface_auth(token: Optional[str] = None, token_file: Optional[str] = None) -> str:
    """
    Set up authentication with Hugging Face Hub using either a token or token file

    Args:
        token (str, optional): Hugging Face token string
        token_file (str, optional): Path to file containing Hugging Face token

    Returns:
        str: The Hugging Face token
    """
    try:
        from huggingface_hub import login
        from huggingface_hub.constants import HF_TOKEN_PATH
    except ImportError:
        logger.error("huggingface_hub package not found. Installing...")
        import subprocess
        subprocess.run([sys.executable, "-m", "pip", "install", "huggingface_hub"], check=True)
        from huggingface_hub import login
        from huggingface_hub.constants import HF_TOKEN_PATH
    
    # Determine the token to use
    if token:
        # Use the provided token directly
        logger.info("Using provided token for Hugging Face Hub authentication")
        hf_token = token
    elif token_file:
        # Read token from the specified file
        logger.info(f"Reading Hugging Face token from file: {token_file}")
        try:
            with open(token_file, 'r') as f:
                hf_token = f.read().strip()
        except Exception as e:
            raise ValueError(f"Failed to read token from file {token_file}: {e}")
    elif os.path.exists(HF_TOKEN_PATH):
        # Use existing token in default location
        logger.info(f"Using existing Hugging Face token from: {HF_TOKEN_PATH}")
        with open(HF_TOKEN_PATH, 'r') as f:
            hf_token = f.read().strip()
    else:
        raise ValueError(
            "No Hugging Face token provided. Please provide a token using --token, --token-file, "
            "or login with 'huggingface-cli login' before running this script."
        )
    
    # Login to Hugging Face Hub
    login(token=hf_token)
    
    return hf_token


def create_or_update_repo(
    repo_name: str,
    organization: Optional[str] = None,
    private: bool = False,
    description: Optional[str] = None,
    token: Optional[str] = None
) -> str:
    """
    Create a new repository on Hugging Face Hub or update an existing one

    Args:
        repo_name (str): Name of the repository
        organization (str, optional): Organization name if the repo belongs to an organization
        private (bool): Whether the repository should be private
        description (str, optional): Description of the repository
        token (str, optional): Hugging Face token

    Returns:
        str: Full repository ID (e.g., "username/repo-name" or "organization/repo-name")
    """
    try:
        from huggingface_hub import create_repo, HfApi
    except ImportError:
        logger.error("huggingface_hub package not found. Installing...")
        import subprocess
        subprocess.run([sys.executable, "-m", "pip", "install", "huggingface_hub"], check=True)
        from huggingface_hub import create_repo, HfApi
    
    api = HfApi(token=token)
    
    # Get the current user if no organization is specified
    if not organization:
        organization = api.whoami()["name"]
    
    # Full repository ID
    repo_id = f"{organization}/{repo_name}"
    
    # Check if the repository already exists
    try:
        api.repo_info(repo_id=repo_id)
        logger.info(f"Repository {repo_id} already exists")
    except Exception:
        # Repository doesn't exist, create it
        logger.info(f"Creating new repository: {repo_id}")
        create_repo(
            repo_id=repo_id,
            token=token,
            private=private,
            exist_ok=True,
            repo_type="model"
        )
        
        # If a description is provided, update the repository README
        if description:
            try:
                # Create a simple README with the description
                readme_content = f"# {repo_name}\n\n{description}\n"
                api.upload_file(
                    path_or_fileobj=readme_content.encode(),
                    path_in_repo="README.md",
                    repo_id=repo_id,
                    token=token
                )
                logger.info(f"Created README.md for repository {repo_id}")
            except Exception as e:
                logger.warning(f"Failed to create README.md: {e}")
    
    return repo_id


def get_model_info(
    model_file: str, 
    model_name: Optional[str] = None, 
    model_description: Optional[str] = None
) -> Dict[str, Any]:
    """
    Generate metadata for the model

    Args:
        model_file (str): Path to the model file
        model_name (str, optional): Name for the model
        model_description (str, optional): Description for the model

    Returns:
        Dict[str, Any]: Model metadata dictionary
    """
    import hashlib
    import datetime
    
    # Get model file info
    file_path = Path(model_file)
    file_name = file_path.name
    file_size = file_path.stat().st_size
    file_size_mb = file_size / (1024 * 1024)
    
    # Generate file hash
    md5_hash = hashlib.md5()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            md5_hash.update(byte_block)
    file_hash = md5_hash.hexdigest()
    
    # Determine model name and quantization level if not provided
    if not model_name:
        # Extract from filename: remove .gguf extension and any split part
        base_name = file_name
        if base_name.endswith('.gguf'):
            base_name = base_name[:-5]
        
        # Remove any split identifiers like -00001-of-00002
        split_pattern = r'-\d{5}-of-\d{5}$'
        base_name = re.sub(split_pattern, '', base_name)
        
        model_name = base_name
    
    # Try to identify quantization level from filename
    quant_levels = [
        'q2_k', 'q3_k_s', 'q3_k_m', 'q3_k_l', 'q4_0', 'q4_1', 'q4_k_s', 'q4_k_m',
        'q5_0', 'q5_1', 'q5_k_s', 'q5_k_m', 'q6_k', 'q8_0'
    ]
    
    detected_quant = None
    for level in quant_levels:
        if level in file_name.lower():
            detected_quant = level
            break
    
    # Create metadata dictionary
    metadata = {
        "name": model_name,
        "file": {
            "name": file_name,
            "size_bytes": file_size,
            "size_mb": round(file_size_mb, 2),
            "md5_hash": file_hash
        },
        "format": "GGUF",
        "upload_date": datetime.datetime.now().isoformat(),
    }
    
    if detected_quant:
        metadata["quantization"] = detected_quant
    
    if model_description:
        metadata["description"] = model_description
    
    return metadata


def upload_model_file(
    file_path: str,
    repo_id: str,
    path_in_repo: Optional[str] = None,
    token: Optional[str] = None
) -> str:
    """
    Upload a model file to Hugging Face Hub

    Args:
        file_path (str): Path to the model file to upload
        repo_id (str): Repository ID (e.g., "username/repo-name")
        path_in_repo (str, optional): Path in the repository to store the file (default: file name)
        token (str, optional): Hugging Face token

    Returns:
        str: URL to the uploaded file
    """
    try:
        from huggingface_hub import HfApi
    except ImportError:
        logger.error("huggingface_hub package not found. Installing...")
        import subprocess
        subprocess.run([sys.executable, "-m", "pip", "install", "huggingface_hub"], check=True)
        from huggingface_hub import HfApi
    
    api = HfApi(token=token)
    
    # Determine the path in the repo
    if path_in_repo is None:
        path_in_repo = os.path.basename(file_path)
    
    # Get file size for logging
    file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
    logger.info(f"Starting upload of {file_path} ({file_size_mb:.2f} MB) to {repo_id}/{path_in_repo}")
    
    # Upload the file
    result = api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=path_in_repo,
        repo_id=repo_id,
        token=token
    )
    
    logger.info(f"Successfully uploaded file to {result}")
    
    return result


def upload_model_card(
    repo_id: str,
    metadata: Dict[str, Any],
    token: Optional[str] = None,
    template_path: Optional[str] = None
) -> str:
    """
    Generate and upload a model card (README.md) to the repository

    Args:
        repo_id (str): Repository ID (e.g., "username/repo-name")
        metadata (Dict[str, Any]): Model metadata
        token (str, optional): Hugging Face token
        template_path (str, optional): Path to a model card template file

    Returns:
        str: URL to the uploaded model card
    """
    try:
        from huggingface_hub import HfApi
    except ImportError:
        logger.error("huggingface_hub package not found. Installing...")
        import subprocess
        subprocess.run([sys.executable, "-m", "pip", "install", "huggingface_hub"], check=True)
        from huggingface_hub import HfApi
    
    api = HfApi(token=token)
    
    # Generate the model card content
    if template_path and os.path.exists(template_path):
        # Use custom template
        with open(template_path, 'r') as f:
            template = f.read()
        
        # Simple templating
        for key, value in metadata.items():
            if isinstance(value, dict):
                for subkey, subvalue in value.items():
                    template = template.replace(f"{{{{{key}.{subkey}}}}}", str(subvalue))
            else:
                template = template.replace(f"{{{{{key}}}}}", str(value))
        
        model_card = template
    else:
        # Use default template
        model_card = f"# {metadata['name']}\n\n"
        
        if 'description' in metadata:
            model_card += f"{metadata['description']}\n\n"
        
        model_card += "## Model Details\n\n"
        model_card += f"- **Format:** {metadata['format']}\n"
        if 'quantization' in metadata:
            model_card += f"- **Quantization:** {metadata['quantization']}\n"
        
        model_card += f"- **File:** {metadata['file']['name']}\n"
        model_card += f"- **Size:** {metadata['file']['size_mb']} MB\n"
        model_card += f"- **MD5 Hash:** {metadata['file']['md5_hash']}\n"
        model_card += f"- **Upload Date:** {metadata['upload_date']}\n"
    
    # Upload the model card
    logger.info(f"Uploading model card (README.md) to {repo_id}")
    result = api.upload_file(
        path_or_fileobj=model_card.encode(),
        path_in_repo="README.md",
        repo_id=repo_id,
        token=token
    )
    
    logger.info(f"Successfully uploaded model card to {result}")
    
    return result


def upload_metadata(
    repo_id: str,
    metadata: Dict[str, Any],
    token: Optional[str] = None
) -> str:
    """
    Upload metadata JSON file to the repository

    Args:
        repo_id (str): Repository ID (e.g., "username/repo-name")
        metadata (Dict[str, Any]): Model metadata
        token (str, optional): Hugging Face token

    Returns:
        str: URL to the uploaded metadata file
    """
    try:
        from huggingface_hub import HfApi
    except ImportError:
        logger.error("huggingface_hub package not found. Installing...")
        import subprocess
        subprocess.run([sys.executable, "-m", "pip", "install", "huggingface_hub"], check=True)
        from huggingface_hub import HfApi
    
    api = HfApi(token=token)
    
    # Convert metadata to JSON
    metadata_json = json.dumps(metadata, indent=2)
    
    # Upload the metadata file
    logger.info(f"Uploading metadata.json to {repo_id}")
    result = api.upload_file(
        path_or_fileobj=metadata_json.encode(),
        path_in_repo="metadata.json",
        repo_id=repo_id,
        token=token
    )
    
    logger.info(f"Successfully uploaded metadata to {result}")
    
    return result


def upload_gguf_model(
    model_files: List[str],
    repo_name: str,
    organization: Optional[str] = None,
    private: bool = False,
    description: Optional[str] = None,
    model_name: Optional[str] = None,
    token: Optional[str] = None,
    token_file: Optional[str] = None,
    model_card_template: Optional[str] = None
) -> str:
    """
    Upload a GGUF model file to Hugging Face Hub

    Args:
        model_files (List[str]): Paths to the model files to upload
        repo_name (str): Name of the repository
        organization (str, optional): Organization name if the repo belongs to an organization
        private (bool): Whether the repository should be private
        description (str, optional): Description of the repository
        model_name (str, optional): Name for the model (defaults to repo name)
        token (str, optional): Hugging Face token
        token_file (str, optional): Path to file containing Hugging Face token
        model_card_template (str, optional): Path to a model card template file

    Returns:
        str: URL to the repository
    """
    # Setup authentication
    hf_token = setup_huggingface_auth(token, token_file)
    
    # Create or update repository
    repo_id = create_or_update_repo(
        repo_name=repo_name,
        organization=organization,
        private=private,
        description=description,
        token=hf_token
    )
    
    # Use the first file as the primary file for metadata
    primary_file = model_files[0]
    
    # Generate model information for metadata
    metadata = get_model_info(
        model_file=primary_file,
        model_name=model_name or repo_name,
        model_description=description
    )
    
    # If multiple files, add the information to metadata
    if len(model_files) > 1:
        metadata["split_files"] = []
        for file_path in model_files:
            file_info = {
                "name": os.path.basename(file_path),
                "size_bytes": os.path.getsize(file_path),
                "size_mb": round(os.path.getsize(file_path) / (1024 * 1024), 2)
            }
            metadata["split_files"].append(file_info)
    
    # Upload each model file
    uploaded_files = []
    for file_path in model_files:
        file_name = os.path.basename(file_path)
        url = upload_model_file(
            file_path=file_path,
            repo_id=repo_id,
            path_in_repo=file_name,
            token=hf_token
        )
        uploaded_files.append(url)
    
    # Upload metadata
    metadata_url = upload_metadata(
        repo_id=repo_id,
        metadata=metadata,
        token=hf_token
    )
    
    # Upload model card
    model_card_url = upload_model_card(
        repo_id=repo_id,
        metadata=metadata,
        token=hf_token,
        template_path=model_card_template
    )
    
    # Generate repository URL
    repo_url = f"https://huggingface.co/{repo_id}"
    
    logger.info(f"Model successfully uploaded to {repo_url}")
    
    return repo_url


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload GGUF models to Hugging Face Hub")
    
    # Required arguments
    parser.add_argument("--files", nargs='+', required=True,
                        help="Path(s) to GGUF model file(s) to upload. For split files, provide all parts.")
    parser.add_argument("--repo-name", required=True,
                        help="Name of the repository to create or update on Hugging Face Hub")
    
    # Optional arguments
    parser.add_argument("--organization", help="Organization name for the repository")
    parser.add_argument("--private", action="store_true", help="Make the repository private")
    parser.add_argument("--description", help="Description of the model")
    parser.add_argument("--model-name", help="Name for the model (defaults to repo name)")
    parser.add_argument("--token", help="Hugging Face token")
    parser.add_argument("--token-file", help="Path to file containing Hugging Face token")
    parser.add_argument("--model-card-template", help="Path to a custom model card template file")
    
    args = parser.parse_args()
    
    # Validate model files
    for file_path in args.files:
        if not os.path.exists(file_path):
            parser.error(f"Model file not found: {file_path}")
        
        if not file_path.lower().endswith('.gguf'):
            parser.error(f"File doesn't have .gguf extension: {file_path}")
    
    # Upload the model
    repo_url = upload_gguf_model(
        model_files=args.files,
        repo_name=args.repo_name,
        organization=args.organization,
        private=args.private,
        description=args.description,
        model_name=args.model_name,
        token=args.token,
        token_file=args.token_file,
        model_card_template=args.model_card_template
    )
    
    print("\nUpload completed successfully!")
    print(f"Repository URL: {repo_url}")
    print("You can now use this model in applications that support Hugging Face Hub models.")
