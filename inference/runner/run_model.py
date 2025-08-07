#!/usr/bin/env python3
"""
Main entry point for the model runner package.
"""
import argparse
import logging
import os
import sys
from typing import Optional, Dict, Any, Union

from dotenv import load_dotenv
from models import ChatReq, Message, MessageContent, MessageContentType, MessageRole
from .pipelines.factory import PipelineFactory

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("runner")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="LLM ML Lab Model Runner")

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model name to run (must be in models.json config)",
    )

    parser.add_argument(
        "--prompt", type=str, required=True, help="Prompt to send to the model"
    )

    parser.add_argument(
        "--task",
        type=str,
        choices=["text", "image", "embedding", "vision"],
        default="text",
        help="Task type: text, image, embedding, or vision (default: text)",
    )

    parser.add_argument(
        "--output", type=str, help="Output file path (for image generation)"
    )

    parser.add_argument(
        "--params",
        type=str,
        help="JSON string of additional parameters to pass to the model",
    )

    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    return parser.parse_args()


def run_pipeline(
    model_name: str,
    prompt: str,
    task: str,
    output_path: Optional[str] = None,
    params: Optional[Dict[str, Any]] = None,
) -> Union[str, Dict[str, Any]]:
    """
    Run the appropriate pipeline for the specified model and task.

    Args:
        model_name: Name of the model to use
        prompt: Input prompt
        task: Task type (text, image, embedding, vision)
        output_path: Path to save output (for images)
        params: Additional parameters to pass to the pipeline

    Returns:
        Pipeline output (varies by task)
    """
    # Create the appropriate pipeline
    pipeline_factory = PipelineFactory()
    pipeline, t = pipeline_factory.get_pipeline(model_id=model_name)
    if not pipeline:
        raise ValueError(
            f"Could not create pipeline for model '{model_name}' and task '{task}'"
        )

    # Run the pipeline
    result = pipeline.run(
        ChatReq(
            model=model_name,
            messages=[
                Message(
                    role=MessageRole.USER,
                    content=[MessageContent(text=prompt, type=MessageContentType.TEXT)],
                    conversation_id=-1,
                )
            ],
            stream=True,
        ),
        t,
    )

    # Save output for image generation
    if task == "image" and output_path and hasattr(result, "save"):
        result.save(output_path)
        logger.info(f"Image saved to {output_path}")

    return result


def main():
    """Main function to run models."""
    args = parse_args()
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    logger.info(f"Running model '{args.model}' with task '{args.task}'")

    # Parse additional parameters if provided
    params = None
    if args.params:
        import json

        try:
            params = json.loads(args.params)
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON in --params: {args.params}")
            sys.exit(1)

    try:
        result = run_pipeline(args.model, args.prompt, args.task, args.output, params)

        # Print the result
        if args.task != "image" or not args.output:
            if isinstance(result, dict):
                import json

                print(json.dumps(result, indent=2))
            else:
                print(result)

    except Exception as e:
        logger.error(f"Error running pipeline: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
