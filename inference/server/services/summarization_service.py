"""
Summarization service for generating summaries of text.
"""

from typing import List, Optional
import os
import asyncio
from pydantic import SecretStr

from langchain_openai import OpenAI
from langchain.chains.summarize import load_summarize_chain
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter

from ..config import logger


class SummarizationService:
    """
    Service for generating summaries of text.
    """

    # Default prompts for summarization
    DEFAULT_SUMMARIZATION_TEMPLATE = """
    Please provide a concise summary of the following conversation excerpt:
    
    {text}
    
    SUMMARY:
    """

    DEFAULT_COMBINE_TEMPLATE = """
    The following is a set of summaries from a conversation:
    
    {text}
    
    Please synthesize these summaries into a single coherent summary that captures the main points and progression of the conversation.
    
    SYNTHESIS:
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        summarization_template: Optional[str] = None,
        combine_template: Optional[str] = None,
    ):
        """
        Initialize the summarization service.

        Args:
            model_name: Name of the model to use for summarization.
            api_key: API key for the service (if needed).
            summarization_template: Custom template for summarization.
            combine_template: Custom template for combining summaries.
        """
        self.model_name = model_name or os.environ.get(
            "SUMMARIZATION_MODEL", "gpt-3.5-turbo"
        )
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")

        # Initialize summarization prompts
        self.summarization_template = (
            summarization_template or self.DEFAULT_SUMMARIZATION_TEMPLATE
        )
        self.combine_template = combine_template or self.DEFAULT_COMBINE_TEMPLATE

        # Initialize the LLM
        self._initialize_llm()

        # Initialize text splitter for chunking long texts
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=4000, chunk_overlap=200
        )

    def _initialize_llm(self) -> None:
        """Initialize the LLM model for summarization."""
        try:
            # Convert api_key to SecretStr if it exists
            api_key_secret = SecretStr(self.api_key) if self.api_key else None

            # Create OpenAI client with proper parameters for the newer LangChain version
            self.llm = OpenAI(
                model=self.model_name, temperature=0.3, api_key=api_key_secret
            )
            logger.info(f"Initialized summarization model: {self.model_name}")
        except ValueError as e:
            logger.error(f"Failed to initialize summarization model: {str(e)}")
            # Fall back to a default configuration
            self.llm = OpenAI(temperature=0.3)
            logger.info("Initialized fallback summarization model")

    async def summarize_text(self, text_items: List[str]) -> str:
        """
        Generate a summary of the provided text items.

        Args:
            text_items: List of text items to summarize.

        Returns:
            A summary of the text.
        """
        if not text_items:
            logger.warning("No text provided for summarization")
            return ""

        try:
            # Join text items with line breaks
            combined_text = "\n\n".join(text_items)

            # Split text into chunks if it's too long
            if len(combined_text) > 4000:
                return await self._summarize_long_text(combined_text)
            else:
                # Summarize directly
                return await self._summarize_single_text(combined_text)
        except ValueError as e:
            logger.error(f"Error in summarization: {str(e)}")
            # Return a simplified summary as fallback
            return self._fallback_summary(text_items)

    async def _summarize_single_text(self, text: str) -> str:
        """
        Summarize a single piece of text.

        Args:
            text: The text to summarize.

        Returns:
            A summary of the text.
        """
        prompt = PromptTemplate(
            template=self.summarization_template, input_variables=["text"]
        )

        # Execute in a separate thread to avoid blocking the event loop
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None, lambda: self.llm(prompt.format(text=text))
        )

        return result.strip()

    async def _summarize_long_text(self, text: str) -> str:
        """
        Summarize a long text by splitting it into chunks.

        Args:
            text: The long text to summarize.

        Returns:
            A summary of the text.
        """
        # Split text into chunks
        docs = self.text_splitter.create_documents([text])

        # Create summarization chain
        summarize_chain = load_summarize_chain(
            self.llm,
            chain_type="map_reduce",
            map_prompt=PromptTemplate(
                template=self.summarization_template, input_variables=["text"]
            ),
            combine_prompt=PromptTemplate(
                template=self.combine_template, input_variables=["text"]
            ),
        )

        # Execute in a separate thread to avoid blocking the event loop
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, lambda: summarize_chain.run(docs))

        return result.strip()

    def _fallback_summary(self, text_items: List[str]) -> str:
        """
        Create a simple fallback summary when the main summarization fails.

        Args:
            text_items: List of text items to summarize.

        Returns:
            A basic summary of the text.
        """
        # Take the first 2-3 items and truncate them
        summary_items = []
        for i, item in enumerate(text_items):
            if i >= 3:
                break
            # Truncate long items
            if len(item) > 100:
                summary_items.append(f"{item[:100]}...")
            else:
                summary_items.append(item)

        if len(text_items) > 3:
            summary_items.append(f"... (and {len(text_items) - 3} more items)")

        return "\n".join(summary_items)
