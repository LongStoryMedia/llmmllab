"""
Script to create an embedding from a string
"""

import argparse
import asyncio
from typing import cast

from langchain.embeddings.base import Embeddings

from composer.agents.embed import EmbeddingAgent

from runner import pipeline_factory

from models import ModelProfileType
from models.default_configs import create_default_user_config
from models.default_model_profiles import DEFAULT_EMBEDDING_PROFILE

from utils.model_profile import get_model_profile_for_task


async def create_embedding(text: str):
    """Create an embedding for the given text"""

    user_config = create_default_user_config("debug_embedding_user")

    embedding_model = pipeline_factory.get_pipeline(profile=DEFAULT_EMBEDDING_PROFILE)

    # Create embedding agent
    embedding_agent = EmbeddingAgent(
        model=cast(Embeddings, embedding_model),
        profile=DEFAULT_EMBEDDING_PROFILE,
    )

    # Generate embedding
    embeddings = await embedding_agent.embed([text])
    return embeddings


def main():
    """Main function to create embedding"""
    parser = argparse.ArgumentParser(description="Create embedding from input text")
    parser.add_argument(
        "--text", type=str, help="Text to create embedding for", required=False
    )
    args = parser.parse_args()

    test_text = (
        args.text if args.text else "This is a test string to create an embedding for."
    )
    embeddings = asyncio.run(create_embedding(test_text))
    print(f"Embedding for text: {embeddings}")


if __name__ == "__main__":
    main()
