"""
Summarization pipeline for BART-large-CNN model in GGUF format.

This pipeline implements the facebook/bart-large-cnn model for text summarization
using llama-cpp-python with GGUF format. BART is particularly effective for
abstractive summarization tasks, generating concise summaries that capture
the key information from longer texts.

Requirements:
- Input: Long text documents to summarize
- Output: Concise abstractive summaries
- Model optimized for news articles and similar content
- Supports customizable summary length and parameters
- Context length: up to 1024 tokens for input

For more details see: https://huggingface.co/facebook/bart-large-cnn
"""

import datetime
import logging
import os
import torch
from typing import Any, Dict, List, Generator, Optional, Union
from llama_cpp import Llama  # type: ignore # pylint: disable=E0401
from transformers import AutoTokenizer

from models import (
    Model,
    Message,
    ChatResponse,
    ModelParameters,
    MessageContent,
    MessageContentType,
    ChatReq,
)
from models.message_role import MessageRole
from models.message_content_type import MessageContentType
from ..base_pipeline import BasePipeline
from ..helpers import get_role


class BARTSummarizationPipe(BasePipeline):
    """
    Pipeline for text summarization using BART-large-CNN model in GGUF format.

    This pipeline supports the facebook/bart-large-cnn model converted to GGUF format
    for efficient inference using llama-cpp-python.

    Key features of the model:
    - BART (Bidirectional and Auto-Regressive Transformers) architecture
    - Fine-tuned specifically for CNN/DailyMail summarization dataset
    - Generates abstractive summaries (not just extractive)
    - Optimal for news articles, documents, and similar content
    - Input length: up to 1024 tokens
    - Output: typically 56-142 tokens (configurable)
    """

    # Class-level attributes
    model: Any = None
    tokenizer: Any = None

    def __init__(self, model_definition: Model):
        """Initialize the BART Summarization pipeline."""
        # Call base class initialization first
        super().__init__(model_definition)

        # Set up logger
        self._logger = logging.getLogger(__name__)
        self._logger.info("Initializing BARTSummarizationPipe")
        self._logger.info(f"Model definition: {self.model_def.json()}")

        # Ensure model details for GGUF are provided
        if not (
            model_definition.details
            and model_definition.model
            and model_definition.details.parent_model
        ):
            raise ValueError(
                "Model definition for BARTSummarizationPipe must include details for 'gguf_file' and 'parent_model'."
            )

        # Log model info for debugging
        self._logger.info(f"Model ID: {self.model_def.id}")

        gguf = (
            model_definition.details.gguf_file
            if model_definition.details.gguf_file
            else model_definition.model
        )

        # Check file size
        file_size = os.path.getsize(gguf)
        if file_size < 1_000_000:  # Less than 1MB is suspicious for a model
            raise ValueError(
                f"GGUF file is too small ({file_size} bytes), likely a placeholder: {gguf}"
            )

        # Log the file path we're actually using
        self._logger.info(
            f"Using GGUF file path: {gguf} (size: {file_size/1_000_000:.2f} MB)"
        )

        try:
            # Load the GGUF model using llama-cpp-python
            self.model = Llama(
                model_path=gguf,
                n_ctx=2048,  # Context length for summarization (input + output)
                n_gpu_layers=-1,  # Offload all layers to GPU
                n_threads=4,
                use_mlock=True,
                verbose=True,
                n_batch=512,
                offload_kqv=True,
                flash_attn=True,
            )

            # Load the original HuggingFace tokenizer for proper text handling
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_def.details.parent_model
            )

            self._logger.info(
                f"BART Summarization model '{self.model_def.name}' loaded successfully."
            )

        except Exception as e:
            self._logger.error(
                f"Error initializing {self.__class__.__name__}: {str(e)}"
            )
            raise

    def run(self, req: ChatReq) -> Generator[ChatResponse, Any, None]:
        """
        Process input messages to generate summaries.

        Args:
            req (ChatReq): The chat request containing messages, model parameters, and other settings.

        Yields:
            Generator[ChatResponse, Any, None]: Yields ChatResponse objects with summaries.
        """
        messages = req.messages
        params = req.options
        start_time = datetime.datetime.now(tz=datetime.timezone.utc)
        message_dicts = []

        self._logger.info(
            f"Running BART Summarization model with {len(messages)} messages"
        )

        # Extract text content from messages
        texts_to_summarize = []
        for m in messages:
            cntnt = []
            for c in m.content:
                if c.text is not None:
                    cntnt.append(c.text)
            if cntnt:
                texts_to_summarize.append("\n".join(cntnt))

        if not texts_to_summarize:
            self._logger.warning("No text inputs found in messages")
            texts_to_summarize = [""]

        most_recent_message = messages[-1] if messages else None

        # Get generation parameters
        max_tokens = (
            params.num_predict if params and params.num_predict is not None else 142
        )
        temperature = float(
            params.temperature if params and params.temperature is not None else 0.3
        )
        top_p = float(params.top_p if params and params.top_p is not None else 0.95)
        top_k = int(params.top_k if params and params.top_k is not None else 40)
        repeat_penalty = float(
            params.repeat_penalty
            if params and params.repeat_penalty is not None
            else 1.1
        )

        # BART-specific stop tokens for summarization
        stop_tokens = (
            params.stop
            if params and params.stop is not None and len(params.stop) > 0
            else ["</s>", "<|endoftext|>", "\n\n"]
        )

        summaries = []
        total_input_tokens = 0
        total_output_tokens = 0

        try:
            for i, text in enumerate(texts_to_summarize):
                self._logger.info(
                    f"Summarizing text {i+1} of {len(texts_to_summarize)}"
                )

                # Create summarization prompt
                # Format the input for BART summarization
                prompt = f"Summarize the following text:\n\n{text}\n\nSummary:"

                # Check input length and truncate if necessary
                input_tokens = self.tokenizer.encode(prompt, add_special_tokens=False)
                if len(input_tokens) > 1000:  # Leave room for output
                    # Truncate the original text, not the prompt structure
                    available_tokens = 1000 - len(
                        self.tokenizer.encode(
                            "Summarize the following text:\n\n\n\nSummary:",
                            add_special_tokens=False,
                        )
                    )
                    text_tokens = self.tokenizer.encode(text, add_special_tokens=False)[
                        :available_tokens
                    ]
                    truncated_text = self.tokenizer.decode(
                        text_tokens, skip_special_tokens=True
                    )
                    prompt = (
                        f"Summarize the following text:\n\n{truncated_text}\n\nSummary:"
                    )
                    self._logger.info(
                        f"Truncated input text from {len(input_tokens)} to ~1000 tokens"
                    )

                total_input_tokens += len(
                    self.tokenizer.encode(prompt, add_special_tokens=False)
                )

                # Generate summary using llama-cpp
                response = self.model.create_completion(
                    prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    repeat_penalty=repeat_penalty,
                    stop=stop_tokens,
                    stream=False,  # Get complete response
                )

                # Extract summary from response
                if response and "choices" in response and len(response["choices"]) > 0:
                    summary_text = response["choices"][0]["text"].strip()

                    # Clean up the summary (remove any prompt artifacts)
                    summary_text = self._clean_summary_text(summary_text)
                    summaries.append(summary_text)

                    # Count output tokens
                    total_output_tokens += len(
                        self.tokenizer.encode(summary_text, add_special_tokens=False)
                    )

                    self._logger.debug(
                        f"Generated summary {i+1}: {len(summary_text)} characters"
                    )
                else:
                    self._logger.warning(f"No summary generated for text {i+1}")
                    summaries.append("Unable to generate summary.")

        except (RuntimeError, ValueError, KeyError) as e:
            self._logger.error(f"Error running BART Summarization model: {str(e)}")
            raise
        except (ImportError, AttributeError, TypeError, IndexError) as e:
            self._logger.error(
                f"Unexpected error running BART Summarization model: {str(e)}"
            )
            raise RuntimeError("Unexpected error in BART Summarization model") from e

        finally:
            end_time = datetime.datetime.now(tz=datetime.timezone.utc)
            total_duration = (end_time - start_time).total_seconds() * 1000

            # Combine summaries
            if len(summaries) == 1:
                final_summary = summaries[0]
            else:
                final_summary = "\n\n".join(
                    [
                        f"Summary {i+1}:\n{summary}"
                        for i, summary in enumerate(summaries)
                    ]
                )

            # Stream the summary in chunks for consistency with other pipelines
            summary_words = final_summary.split()
            chunk_size = 10  # Words per chunk

            for i in range(0, len(summary_words), chunk_size):
                chunk_text = " ".join(summary_words[i : i + chunk_size])
                if i + chunk_size < len(summary_words):
                    chunk_text += " "

                yield ChatResponse(
                    done=False,
                    message=Message(
                        role=MessageRole.ASSISTANT,
                        content=[
                            MessageContent(
                                type=MessageContentType.TEXT, text=chunk_text, url=None
                            )
                        ],
                        tool_calls=None,
                        thinking=None,
                        id=most_recent_message.id if most_recent_message else -1,
                        created_at=datetime.datetime.now(tz=datetime.timezone.utc),
                    ),
                    model=self.model_def.model,
                    created_at=datetime.datetime.now(tz=datetime.timezone.utc),
                    context=None,
                    finish_reason=None,
                    total_duration=None,
                    load_duration=None,
                    prompt_eval_count=None,
                    prompt_eval_duration=None,
                    eval_count=None,
                    eval_duration=None,
                )

            # Create the final response
            res = ChatResponse(
                done=True,
                finish_reason="stop",
                message=Message(
                    role=MessageRole.ASSISTANT,
                    content=[
                        MessageContent(
                            type=MessageContentType.TEXT, text=final_summary, url=None
                        )
                    ],
                    tool_calls=None,
                    thinking=None,
                    id=most_recent_message.id if most_recent_message else -1,
                    created_at=datetime.datetime.now(tz=datetime.timezone.utc),
                ),
                model=self.model_def.model,
                created_at=datetime.datetime.now(tz=datetime.timezone.utc),
                context=summaries,  # Return individual summaries in context
                total_duration=total_duration,
                load_duration=0.0,
                prompt_eval_count=total_input_tokens,
                prompt_eval_duration=0.0,
                eval_count=total_output_tokens,
                eval_duration=total_duration,
            )
            yield res

    def _clean_summary_text(self, summary: str) -> str:
        """
        Clean up generated summary text by removing artifacts and formatting issues.

        Args:
            summary: Raw summary text from the model

        Returns:
            Cleaned summary text
        """
        # Remove common artifacts
        summary = summary.strip()

        # Remove any remaining prompt text that might have leaked through
        summary = summary.replace("Summarize the following text:", "")
        summary = summary.replace("Summary:", "")

        # Remove excessive whitespace
        import re

        summary = re.sub(r"\s+", " ", summary)

        # Remove incomplete sentences at the end
        sentences = summary.split(".")
        if len(sentences) > 1 and len(sentences[-1].strip()) < 10:
            summary = ".".join(sentences[:-1]) + "."

        return summary.strip()

    async def summarize(
        self,
        text: str,
        max_length: Optional[int] = None,
        min_length: Optional[int] = None,
        temperature: float = 0.3,
    ) -> str:
        """
        Generate a summary for a single text.

        Args:
            text: The text to summarize
            max_length: Maximum length of the summary in tokens (default: 142)
            min_length: Minimum length (not enforced in this implementation)
            temperature: Temperature for generation (default: 0.3)

        Returns:
            The generated summary text
        """
        try:
            # Create message
            message = Message(
                role=MessageRole.USER,
                content=[MessageContent(type=MessageContentType.TEXT, text=text)],
                id=None,
                created_at=datetime.datetime.now(tz=datetime.timezone.utc),
            )

            # Create parameters
            params = ModelParameters(
                num_predict=max_length or 142,
                temperature=temperature,
                top_p=0.95,
                top_k=40,
                repeat_penalty=1.1,
            )

            # Create request
            req = ChatReq(
                messages=[message],
                conversation_id=999,
                stream=True,
                options=params,
            )

            # Execute the request
            responses = list(self.run(req))

            # Extract summary from final response
            for response in reversed(responses):  # Start from the end
                if response.done and response.message and response.message.content:
                    for content in response.message.content:
                        if content.text:
                            return content.text

            return "Unable to generate summary."

        except Exception as e:
            self._logger.error(f"Error in summarize method: {e}")
            return f"Error generating summary: {str(e)}"

    async def batch_summarize(
        self,
        texts: List[str],
        max_length: Optional[int] = None,
        temperature: float = 0.3,
        batch_size: int = 4,
    ) -> List[str]:
        """
        Generate summaries for multiple texts.

        Args:
            texts: List of texts to summarize
            max_length: Maximum length of each summary
            temperature: Temperature for generation
            batch_size: Number of texts to process in each batch (for memory management)

        Returns:
            List of summary texts
        """
        all_summaries = []

        try:
            # Process texts in batches to manage memory
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i : i + batch_size]
                self._logger.info(
                    f"Processing summarization batch {i//batch_size + 1}: {len(batch_texts)} texts"
                )

                batch_summaries = []
                for text in batch_texts:
                    try:
                        summary = await self.summarize(
                            text, max_length=max_length, temperature=temperature
                        )
                        batch_summaries.append(summary)
                    except Exception as e:
                        self._logger.error(f"Error summarizing text in batch: {e}")
                        batch_summaries.append("Error generating summary.")

                all_summaries.extend(batch_summaries)

            return all_summaries

        except Exception as e:
            self._logger.error(f"Error in batch_summarize: {e}")
            return ["Error generating summary."] * len(texts)

    def extract_key_points(self, text: str, num_points: int = 5) -> List[str]:
        """
        Extract key points from a text by generating a structured summary.

        Args:
            text: The text to extract key points from
            num_points: Target number of key points to extract

        Returns:
            List of key point strings
        """
        try:
            # Create a prompt specifically for key point extraction
            prompt = f"Extract {num_points} key points from the following text:\n\n{text}\n\nKey points:\n1."

            # Generate response
            response = self.model.create_completion(
                prompt,
                max_tokens=200,  # Enough for several key points
                temperature=0.3,
                top_p=0.95,
                stop=["</s>", "<|endoftext|>", "\n\nText:", "\n\nKey points:"],
                stream=False,
            )

            if response and "choices" in response and len(response["choices"]) > 0:
                key_points_text = "1." + response["choices"][0]["text"].strip()

                # Parse numbered points
                import re

                points = re.findall(r"\d+\.\s*([^0-9]+?)(?=\d+\.|$)", key_points_text)

                # Clean and return points
                cleaned_points = [
                    point.strip().rstrip(".") for point in points if point.strip()
                ]
                return (
                    cleaned_points[:num_points]
                    if len(cleaned_points) >= num_points
                    else cleaned_points
                )
            else:
                return ["Unable to extract key points."]

        except Exception as e:
            self._logger.error(f"Error extracting key points: {e}")
            return ["Error extracting key points."]

    def __del__(self) -> None:
        """Clean up resources used by the BARTSummarizationPipe."""
        try:
            self._logger.info(
                f"BARTSummarizationPipe for {self.model_def.name if hasattr(self, 'model_def') else 'unknown'}: Cleanup initiated"
            )
            if hasattr(self, "model") and self.model is not None:
                del self.model
            if hasattr(self, "tokenizer") and self.tokenizer is not None:
                del self.tokenizer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except (RuntimeError, AttributeError, ValueError) as e:
            logger = logging.getLogger(__name__)
            logger.error(f"Error cleaning up BARTSummarizationPipe resources: {str(e)}")
