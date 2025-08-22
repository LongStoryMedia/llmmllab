"""
Re-ranking pipeline for Qwen3-Reranker-0.6B model.

This pipeline implements the Qwen3-Reranker-0.6B model for ranking text passages
based on their relevance to a query. It's designed for RAG and search applications
where you need to reorder retrieved documents by relevance.

Requirements:
- Input format: query and list of passages to rank
- Returns relevance scores for each passage
- Supports batch processing for efficiency
- Model size: 0.6B parameters
- Context length: up to 8192 tokens

For more details see: https://huggingface.co/Qwen/Qwen3-Reranker-0.6B
"""

import datetime
import logging
import os
from typing import Any, Dict, Iterator, List, Generator, Optional, Tuple
import numpy as np
import torch
from llama_cpp import CreateCompletionResponse, Llama

from models import (
    Model,
    Message,
    ChatResponse,
    MessageRole,
    MessageContent,
    MessageContentType,
    ChatReq,
)
from ..base_pipeline import BasePipeline


class Qwen3RerankerPipe(BasePipeline):
    """
    Pipeline for running text re-ranking with Qwen3-Reranker-0.6B model.

    This pipeline supports the Qwen/Qwen3-Reranker-0.6B model in GGUF format.

    Key features of the model:
    - Specialized for text re-ranking tasks
    - 0.6B parameters for efficient inference
    - Context length up to 8192 tokens
    - Optimized for RAG and search applications
    - Returns relevance scores between query and passages
    """

    # Class-level attributes
    model: Llama

    def __init__(self, model_definition: Model):
        """Initialize the Qwen3 Reranker pipeline."""
        # Call base class initialization first
        super().__init__(model_definition)

        # Set up logger
        self._logger = logging.getLogger(__name__)
        self._logger.info("Initializing Qwen3RerankerPipe")
        self._logger.info(f"Model definition: {self.model_def.json()}")

        # Ensure model details for GGUF are provided
        if not (self.model_def.details and self.model_def.model):
            raise ValueError(
                "Model definition for Qwen3RerankerPipe must include model path details."
            )

        # Log model info for debugging
        self._logger.info(f"Model ID: {self.model_def.id}")

        # Get the GGUF file path
        gguf = (
            model_definition.details.gguf_file
            if hasattr(model_definition.details, "gguf_file")
            and model_definition.details.gguf_file
            else model_definition.model
        )

        # Check file size
        file_size = os.path.getsize(gguf)
        if file_size < 1_000_000:  # Less than 1MB is suspicious
            raise ValueError(
                f"GGUF file is too small ({file_size} bytes), likely a placeholder: {gguf}"
            )

        # Log the file path we're actually using
        self._logger.info(
            f"Using GGUF file path: {gguf} (size: {file_size/1_000_000:.2f} MB)"
        )

        # Load the GGUF model using llama-cpp-python
        try:
            self.model = Llama(
                model_path=gguf,
                n_ctx=8192,  # Maximum context length for Qwen3-Reranker
                n_gpu_layers=-1,  # Offload all layers to GPU
                n_threads=4,
                use_mlock=True,
                verbose=True,
                n_batch=512,
                offload_kqv=True,
                flash_attn=True,
            )

            self._logger.info(
                f"Qwen3 Reranker model '{self.model_def.name}' loaded successfully."
            )
        except Exception as e:
            self._logger.error(
                f"Error initializing {self.__class__.__name__}: {str(e)}"
            )
            raise

    def run(self, req: ChatReq) -> Generator[ChatResponse, Any, None]:
        """
        Process input messages to generate re-ranking scores.

        Expected input format:
        - First message should contain the query
        - Subsequent messages should contain passages to rank
        - Or single message with query and passages separated by special tokens

        Args:
            req (ChatReq): The chat request containing messages, model parameters, and other settings.

        Yields:
            Generator[ChatResponse, Any, None]: Yields ChatResponse objects with ranking scores.
        """
        start_time = datetime.datetime.now(tz=datetime.timezone.utc)
        load_time = 0.0

        try:
            # Extract query and passages from messages
            query = ""
            passages = []

            if len(req.messages) >= 2:
                # First message is query, rest are passages
                query_message = req.messages[0]
                if query_message.content:
                    for content in query_message.content:
                        if hasattr(content, "text") and content.text:
                            query = content.text
                            break

                # Remaining messages are passages
                for message in req.messages[1:]:
                    if message.content:
                        for content in message.content:
                            if hasattr(content, "text") and content.text:
                                passages.append(content.text)
            elif len(req.messages) == 1:
                # Single message - try to parse query and passages
                message = req.messages[0]
                if message.content and message.content[0].text:
                    text = message.content[0].text
                    # Look for common separators
                    if "QUERY:" in text and "PASSAGES:" in text:
                        parts = text.split("PASSAGES:", 1)
                        query = parts[0].replace("QUERY:", "").strip()
                        passage_text = parts[1].strip()
                        # Split passages by newlines or numbered format
                        passages = [
                            p.strip() for p in passage_text.split("\n") if p.strip()
                        ]
                    else:
                        # Fallback: treat as single passage with empty query
                        query = ""
                        passages = [text]

            if not query and not passages:
                self._logger.warning("No query or passages found in messages")
                query = ""
                passages = [""]

            self._logger.info(
                f"Re-ranking {len(passages)} passages for query: {query[:100]}..."
            )

            # Generate ranking scores
            scores = []
            for i, passage in enumerate(passages):
                try:
                    # Create prompt for re-ranking
                    # Format: "Query: {query}\nPassage: {passage}\nRelevant:"
                    prompt = f"Query: {query}\nPassage: {passage}\nRelevant:"

                    # Get logits for "Yes" vs "No" tokens to determine relevance
                    # This is a simplified approach - in practice you might want to use
                    # proper classification head or embedding similarity
                    output = self.model.create_completion(
                        prompt,
                        max_tokens=1,
                        temperature=0.0,
                        logprobs=10,  # Get top 10 logprobs to find Yes/No tokens
                    )

                    # Extract relevance score (simplified approach)
                    # In a real implementation, you'd want to use proper logits for Yes/No tokens
                    # For now, we'll use a placeholder scoring mechanism
                    score = self._extract_relevance_score(output, prompt)
                    scores.append((i, score))

                    self._logger.debug(f"Passage {i}: score={score:.4f}")

                except Exception as e:
                    self._logger.error(f"Error processing passage {i}: {str(e)}")
                    scores.append((i, 0.0))  # Default low score for errors

            # Sort by score (descending)
            scores.sort(key=lambda x: x[1], reverse=True)

            end_time = datetime.datetime.now(tz=datetime.timezone.utc)
            total_duration = (end_time - start_time).total_seconds() * 1000

            # Create response with ranking results
            ranking_text = f"Re-ranked {len(passages)} passages.\n"
            ranking_text += "Rankings (index: score):\n"
            for idx, score in scores:
                ranking_text += f"{idx}: {score:.4f}\n"

            response = ChatResponse(
                message=Message(
                    id=None,
                    role=MessageRole.ASSISTANT,
                    content=[
                        MessageContent(
                            type=MessageContentType.TEXT,
                            text=ranking_text,
                            url=None,
                        )
                    ],
                    tool_calls=None,
                    thinking=None,
                    created_at=start_time,
                ),
                done=True,
                finish_reason="stop",
                context=scores,  # Return scores in context for programmatic access
                total_duration=total_duration,
                load_duration=load_time,
                prompt_eval_count=len(passages),
                prompt_eval_duration=0,
                eval_count=len(passages),
                eval_duration=total_duration,
                created_at=start_time,
                model=str(self.model_def.id),
            )
            yield response

        except (RuntimeError, ValueError) as e:
            self._logger.error(f"Error running Qwen3 Reranker model: {str(e)}")
            error_response = ChatResponse(
                message=Message(
                    id=None,
                    role=MessageRole.ASSISTANT,
                    content=[
                        MessageContent(
                            type=MessageContentType.TEXT,
                            text=f"Error generating rankings: {str(e)}",
                            url=None,
                        )
                    ],
                    tool_calls=None,
                    thinking=None,
                    created_at=start_time,
                ),
                done=True,
                finish_reason="error",
                total_duration=(
                    datetime.datetime.now(tz=datetime.timezone.utc) - start_time
                ).total_seconds()
                * 1000,
                load_duration=load_time,
                prompt_eval_count=0,
                prompt_eval_duration=0,
                eval_count=0,
                eval_duration=0,
                created_at=start_time,
                model=str(self.model_def.id),
            )
            yield error_response
            raise

    def _extract_relevance_score(
        self,
        output: CreateCompletionResponse | Iterator[CreateCompletionResponse],
        prompt: str,
    ) -> float:
        """
        Extract relevance score from model output.

        This is a simplified implementation. In practice, you'd want to:
        1. Use the model's classification head if available
        2. Look for specific token logits (Yes/No, Relevant/Irrelevant)
        3. Use embedding similarity as a fallback

        Args:
            output: Model output dictionary
            prompt: Original prompt used

        Returns:
            float: Relevance score between 0 and 1
        """
        try:
            # Check if we have logprobs in the output
            if "choices" in output and len(output["choices"]) > 0:
                choice = output["choices"][0]
                if "logprobs" in choice and choice["logprobs"]:
                    # Look for tokens that indicate relevance
                    logprobs = choice["logprobs"]
                    if "top_logprobs" in logprobs and len(logprobs["top_logprobs"]) > 0:
                        top_logprobs = logprobs["top_logprobs"][
                            0
                        ]  # First token logprobs

                        # Look for relevant tokens and their probabilities
                        relevant_tokens = {
                            "Yes",
                            "yes",
                            "TRUE",
                            "true",
                            "1",
                            "relevant",
                            "Relevant",
                        }
                        irrelevant_tokens = {
                            "No",
                            "no",
                            "FALSE",
                            "false",
                            "0",
                            "irrelevant",
                            "Irrelevant",
                        }

                        relevant_prob = 0.0
                        irrelevant_prob = 0.0

                        for token, logprob in top_logprobs.items():
                            prob = np.exp(logprob)
                            if token in relevant_tokens:
                                relevant_prob += prob
                            elif token in irrelevant_tokens:
                                irrelevant_prob += prob

                        # Normalize and return relevance score
                        total_prob = relevant_prob + irrelevant_prob
                        if total_prob > 0:
                            return relevant_prob / total_prob

            # Fallback: use text length similarity as a rough proxy
            # This is very basic and should be replaced with proper scoring
            query_in_prompt = (
                prompt.split("Query: ")[1].split("\nPassage:")[0]
                if "Query: " in prompt
                else ""
            )
            passage_in_prompt = (
                prompt.split("Passage: ")[1].split("\nRelevant:")[0]
                if "Passage: " in prompt
                else ""
            )

            # Simple word overlap scoring as fallback
            query_words = set(query_in_prompt.lower().split())
            passage_words = set(passage_in_prompt.lower().split())

            if len(query_words) == 0:
                return 0.5  # Neutral score if no query

            overlap = len(query_words.intersection(passage_words))
            return min(overlap / len(query_words), 1.0)

        except Exception as e:
            self._logger.warning(f"Error extracting relevance score: {e}")
            return 0.5  # Default neutral score

    async def rank_passages(
        self, query: str, passages: List[str], top_k: Optional[int] = None
    ) -> List[Tuple[int, float, str]]:
        """
        Rank passages by relevance to query.

        Args:
            query: The search query
            passages: List of text passages to rank
            top_k: Optional limit on number of results to return

        Returns:
            List of tuples: (original_index, relevance_score, passage_text)
            Sorted by relevance score (descending)
        """
        try:
            # Create messages for the ranking request
            messages = [
                Message(
                    role=MessageRole.USER,
                    content=[MessageContent(type=MessageContentType.TEXT, text=query)],
                    id=None,
                    created_at=datetime.datetime.now(tz=datetime.timezone.utc),
                )
            ]

            # Add passage messages
            for passage in passages:
                messages.append(
                    Message(
                        role=MessageRole.USER,
                        content=[
                            MessageContent(type=MessageContentType.TEXT, text=passage)
                        ],
                        id=None,
                        created_at=datetime.datetime.now(tz=datetime.timezone.utc),
                    )
                )

            # Create request
            req = ChatReq(
                messages=messages,
                conversation_id=999,
                stream=True,
            )

            # Execute the request
            responses = list(self.run(req))

            # Extract rankings from response context
            rankings = []
            for response in responses:
                if hasattr(response, "context") and response.context:
                    # response.context contains list of (index, score) tuples
                    for idx, score in response.context:
                        if idx < len(passages):
                            rankings.append((idx, score, passages[idx]))
                    break

            # Apply top_k limit if specified
            if top_k and len(rankings) > top_k:
                rankings = rankings[:top_k]

            return rankings

        except Exception as e:
            self._logger.error(f"Error ranking passages: {e}")
            # Return passages with neutral scores as fallback
            return [
                (i, 0.5, passage)
                for i, passage in enumerate(passages[:top_k] if top_k else passages)
            ]

    def __del__(self) -> None:
        """Clean up resources used by the Qwen3RerankerPipe."""
        try:
            self._logger.info(
                f"Qwen3RerankerPipe for {self.model_def.name if hasattr(self, 'model_def') else 'unknown'}: Cleanup initiated"
            )
            if hasattr(self, "model") and self.model is not None:
                del self.model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except (RuntimeError, AttributeError, ValueError) as e:
            logger = logging.getLogger(__name__)
            logger.error(f"Error cleaning up Qwen3RerankerPipe resources: {str(e)}")
