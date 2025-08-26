"""
Optimized summarization pipeline for BART-large-CNN model with performance enhancements.
"""

import os
import datetime
import logging
from typing import AsyncGenerator, List, cast, Optional, Dict, Any
import torch
from transformers import AutoTokenizer
from langchain.agents import (
    AgentExecutor,
    create_openai_tools_agent,
)
from langchain_community.chat_models.llamacpp import ChatLlamaCpp
from langchain_community.tools import BaseTool
from langchain_core.callbacks import CallbackManager, StreamingStdOutCallbackHandler
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables.schema import StandardStreamEvent
from models import (
    MessageContent,
    MessageContentType,
    MessageRole,
    Model,
    Message,
    ChatResponse,
    ModelProfile,
)
from ..base_dual_pipeline import TextPipeline
from ..helpers import get_role, to_lc_message, extract_message_text
from ..streaming import StreamingCallbackHandler, EventStreamProcessor


logger = logging.getLogger(__name__)


class BARTSummarizationPipe(TextPipeline):
    """
    Optimized pipeline for BART-large-CNN summarization model with enhanced performance.

    Key optimizations:
    - Smart text preprocessing and chunking
    - Adaptive summary length based on input size
    - Enhanced prompt engineering for better summaries
    - Performance monitoring and caching
    - Memory-efficient processing
    - Quality scoring for summary validation
    """

    llm: ChatLlamaCpp
    tokenizer: AutoTokenizer
    _summary_cache: Dict[str, str] = {}
    _performance_metrics: dict

    def __init__(self, model: Model, profile: ModelProfile):
        """Initialize optimized BARTSummarizationPipe instance."""
        super().__init__(model, profile)

        # Initialize performance tracking
        self._performance_metrics = {
            "summaries_generated": 0,
            "cache_hits": 0,
            "average_input_length": 0,
            "average_summary_length": 0,
            "average_compression_ratio": 0.0,
        }

        # Validate model requirements
        if not (model.details and model.model and model.details.parent_model):
            raise ValueError(
                "Model definition for BARTSummarizationPipe must include details for 'gguf_file' and 'parent_model'."
            )

        self._logger = logging.getLogger(__name__)
        self._logger.info(
            f"Initializing optimized BARTSummarizationPipe for model: {model.id}"
        )

        # Get and validate GGUF file
        gguf_path = self._get_gguf_path()
        self._validate_gguf_file(gguf_path)

        # Initialize components with optimizations
        self._initialize_tokenizer()
        self._initialize_llm(gguf_path)

    def _get_gguf_path(self) -> str:
        """Get the GGUF file path with fallback logic."""
        return (
            self.model.details.gguf_file
            if self.model.details.gguf_file
            else self.model.model
        )

    def _validate_gguf_file(self, gguf_path: str) -> None:
        """Enhanced GGUF file validation."""
        if not os.path.exists(gguf_path):
            raise FileNotFoundError(f"GGUF file not found: {gguf_path}")

        file_size = os.path.getsize(gguf_path)
        if file_size < 1_000_000:  # Less than 1MB is suspicious
            raise ValueError(
                f"GGUF file is too small ({file_size} bytes), likely a placeholder: {gguf_path}"
            )

        # Test file readability
        try:
            with open(gguf_path, "rb") as f:
                f.read(8)  # Read first 8 bytes
        except Exception as e:
            raise IOError(f"Cannot read GGUF file {gguf_path}: {e}")

        self._logger.info(
            f"Using GGUF file: {gguf_path} (size: {file_size/1_000_000:.2f} MB)"
        )

    def _initialize_tokenizer(self) -> None:
        """Initialize tokenizer with error handling."""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model.details.parent_model,
                use_fast=True,
            )
            self._logger.info("BART tokenizer loaded successfully")
        except Exception as e:
            self._logger.warning(f"Fast tokenizer failed, falling back to slow: {e}")
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model.details.parent_model,
                    use_fast=False,
                )
            except Exception as e2:
                raise RuntimeError(f"Failed to load BART tokenizer: {e2}")

    def _initialize_llm(self, gguf_path: str) -> None:
        """Initialize LLM with BART-specific optimizations."""
        try:
            # BART-specific optimized settings
            self.llm = ChatLlamaCpp(
                model_path=gguf_path,
                n_gpu_layers=-1,  # Offload all layers to GPU
                n_batch=512,
                n_ctx=self.profile.parameters.num_ctx or 4096,  # BART context
                f16_kv=True,
                callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
                verbose=os.getenv("LOG_LEVEL", "WARNING").lower() == "debug",
                n_parts=-1,
                seed=self.profile.parameters.seed or -1,
                logits_all=False,
                vocab_only=False,
                use_mlock=False,
                n_threads=self._get_optimal_threads(),
                suffix="",
                logprobs=0,
                # BART-specific generation parameters optimized for summarization
                temperature=self.profile.parameters.temperature
                or 0.3,  # Lower for consistency
                max_tokens=self.profile.parameters.num_predict
                or 512,  # Typical summary length
                top_p=self.profile.parameters.top_p or 0.95,
                top_k=self.profile.parameters.top_k or 40,
                repeat_penalty=self.profile.parameters.repeat_penalty or 1.1,
                streaming=True,
                # BART-specific stop tokens optimized for summarization
                stop=self._get_summarization_stop_tokens(),
            )

            self._logger.info("BART summarization model loaded successfully")

        except Exception as e:
            self._logger.error(f"Failed to initialize BART LLM: {e}")
            raise

    def _get_optimal_threads(self) -> int:
        """Get optimal thread count for BART model."""
        try:
            import multiprocessing

            cpu_count = multiprocessing.cpu_count()
            # BART benefits from more threads due to encoder-decoder architecture
            optimal_threads = min(max(cpu_count // 2, 4), 12)
            self._logger.debug(f"Using {optimal_threads} threads for BART")
            return optimal_threads
        except Exception:
            return 6  # Good default for BART

    def _get_summarization_stop_tokens(self) -> List[str]:
        """Get stop tokens optimized for BART summarization."""
        return self.profile.parameters.stop or [
            "</s>",  # BART's primary EOS token
            "<|endoftext|>",  # Fallback EOS
            "<pad>",  # BART's padding token
            "</tool_call>",  # Tool boundaries
            "</tool_response>",
            "</think>",
            "\n\nSummary:",  # Prevent summary repetition
            "\n\nText:",  # Prevent text repetition
            "Human:",  # Prevent role bleeding
            "Assistant:",
            "Original text:",  # Prevent confusion
            "Summary:",
        ]

    def _preprocess_text(self, text: str) -> str:
        """Preprocess text for optimal summarization."""
        # Remove excessive whitespace
        text = " ".join(text.split())

        # Remove very short paragraphs that might be noise
        paragraphs = text.split("\n")
        filtered_paragraphs = [p.strip() for p in paragraphs if len(p.strip()) > 20]

        return "\n".join(filtered_paragraphs) if filtered_paragraphs else text

    def _calculate_optimal_summary_length(self, input_text: str) -> int:
        """Calculate optimal summary length based on input characteristics."""
        input_length = len(input_text.split())

        # Adaptive summary length based on input size
        if input_length < 100:
            return min(50, input_length // 2)
        elif input_length < 500:
            return min(150, input_length // 4)
        elif input_length < 2000:
            return min(300, input_length // 6)
        else:
            return min(500, input_length // 8)

    def _create_optimized_summary_prompt(
        self, text: str, max_length: Optional[int] = None
    ) -> str:
        """Create an optimized prompt for BART summarization."""
        processed_text = self._preprocess_text(text)
        optimal_length = max_length or self._calculate_optimal_summary_length(
            processed_text
        )

        return f"""Please provide a concise, accurate summary of the following text. 
Focus on the key points and main ideas. Keep the summary to approximately {optimal_length} words.

Text to summarize:
{processed_text}

Summary:"""

    def _generate_cache_key(self, text: str) -> str:
        """Generate a cache key for the input text."""
        import hashlib

        # Use first 100 and last 100 chars plus length for cache key
        key_text = (
            text[:100] + str(len(text)) + text[-100:] if len(text) > 200 else text
        )
        return hashlib.md5(key_text.encode()).hexdigest()

    def _score_summary_quality(self, original: str, summary: str) -> float:
        """Score the quality of a summary (0.0 to 1.0)."""
        try:
            # Basic quality metrics
            original_words = set(original.lower().split())
            summary_words = set(summary.lower().split())

            # Coverage: how many important words are preserved
            coverage = len(original_words.intersection(summary_words)) / len(
                original_words
            )

            # Compression ratio
            compression = len(summary.split()) / len(original.split())

            # Length appropriateness (penalize too short or too long)
            optimal_compression = 0.3  # 30% of original
            compression_score = 1.0 - abs(compression - optimal_compression)

            # Combined score
            quality = (coverage * 0.6) + (compression_score * 0.4)
            return min(max(quality, 0.0), 1.0)

        except Exception:
            return 0.5  # Neutral score on error

    async def run(
        self, messages: List[Message], prompt: ChatPromptTemplate, tools: List[BaseTool]
    ) -> AsyncGenerator[ChatResponse, None]:
        """
        Enhanced summarization processing with caching and quality control.
        """
        start_time = datetime.datetime.now(datetime.timezone.utc)

        try:
            # Extract and preprocess text
            input_texts = []
            for message in messages:
                text = extract_message_text(message)
                if text:
                    input_texts.append(text)

            if not input_texts:
                yield self._create_error_response("No text content found to summarize")
                return

            combined_text = "\n\n".join(input_texts)

            # Check cache first
            cache_key = self._generate_cache_key(combined_text)
            if cache_key in self._summary_cache:
                self._performance_metrics["cache_hits"] += 1
                self._logger.info("Using cached summary")

                cached_summary = self._summary_cache[cache_key]
                yield self._create_summary_response(cached_summary, from_cache=True)
                return

            # Generate new summary
            if tools:
                # Use agent-based summarization for complex tasks
                async for response in self._agent_summarization(
                    messages, prompt, tools
                ):
                    yield response
            else:
                # Use direct summarization for simple tasks
                async for response in self._direct_summarization(combined_text):
                    # Cache the result if it's a final response
                    if response.done and response.message:
                        summary_text = extract_message_text(response.message)
                        if summary_text:
                            quality = self._score_summary_quality(
                                combined_text, summary_text
                            )
                            if quality > 0.6:  # Only cache high-quality summaries
                                self._summary_cache[cache_key] = summary_text
                                self._logger.debug(
                                    f"Cached summary with quality score: {quality:.2f}"
                                )

                    yield response

            # Update metrics
            self._update_performance_metrics(combined_text, start_time)

        except Exception as e:
            self._logger.error(f"Error in BART summarization: {e}", exc_info=True)
            yield self._create_error_response(f"Summarization error: {str(e)}")

    async def _agent_summarization(
        self, messages: List[Message], prompt: ChatPromptTemplate, tools: List[BaseTool]
    ) -> AsyncGenerator[ChatResponse, None]:
        """Agent-based summarization for complex tasks."""
        try:
            agent = create_openai_tools_agent(llm=self.llm, tools=tools, prompt=prompt)
            agent_executor = AgentExecutor(
                agent=agent,
                tools=tools,
                verbose=True,
                max_iterations=8,  # Fewer iterations for summarization
                max_execution_time=120,  # 2 minute timeout
                return_intermediate_steps=True,
                handle_parsing_errors=True,
                callbacks=[StreamingCallbackHandler()],
            )

            chat_history = [to_lc_message(msg) for msg in messages[:-1]]
            processor = EventStreamProcessor(thinking_phase=True)

            async for event in agent_executor.astream_events(
                {
                    "input": extract_message_text(messages[-1]),
                    "chat_history": chat_history,
                },
                version="v2",
                include_types=["chat_model", "tool", "llm", "agent"],
            ):
                for chunk in processor.stream_event(cast(StandardStreamEvent, event)):
                    yield chunk

        except Exception as e:
            self._logger.error(f"Error in agent summarization: {e}")
            yield self._create_error_response(f"Agent summarization error: {str(e)}")

    async def _direct_summarization(
        self, text: str
    ) -> AsyncGenerator[ChatResponse, None]:
        """Direct summarization without agents."""
        processor = EventStreamProcessor(thinking_phase=False)

        try:
            # Create optimized prompt
            summary_prompt = self._create_optimized_summary_prompt(text)

            yield processor.create_streaming_chunk(
                "📄 Generating optimized summary...\n\n"
            )

            # Stream the summarization
            response_chunks = []
            async for chunk in self.llm.astream(summary_prompt):
                if hasattr(chunk, "content") and chunk.content:
                    chunk_content = chunk.content
                    response_chunks.append(chunk_content)

                    if isinstance(chunk_content, str):
                        yield processor.create_streaming_chunk(chunk_content)
                    elif isinstance(chunk_content, list):
                        for item in chunk_content:
                            if isinstance(item, str):
                                yield processor.create_streaming_chunk(item)

            # Validate and finalize
            full_summary = "".join(response_chunks).strip()
            if full_summary:
                quality = self._score_summary_quality(text, full_summary)
                self._logger.info(
                    f"Generated summary with quality score: {quality:.2f}"
                )
                yield processor.create_streaming_chunk("", done=True)
            else:
                yield processor.create_error_chunk("Failed to generate summary")

        except Exception as e:
            self._logger.error(f"Error in direct summarization: {e}")
            yield processor.create_error_chunk(f"Direct summarization error: {str(e)}")

    def _create_summary_response(
        self, summary_text: str, from_cache: bool = False
    ) -> ChatResponse:
        """Create a standardized summary response."""
        prefix = "📋 (Cached) " if from_cache else "📋 "

        return ChatResponse(
            done=True,
            message=Message(
                role=MessageRole.ASSISTANT,
                content=[
                    MessageContent(
                        type=MessageContentType.TEXT,
                        text=f"{prefix}{summary_text}",
                    )
                ],
            ),
            model=self.model.model,
            created_at=datetime.datetime.now(datetime.timezone.utc),
            finish_reason="stop",
        )

    def _create_error_response(self, error_message: str) -> ChatResponse:
        """Create standardized error response."""
        return ChatResponse(
            done=True,
            message=Message(
                role=MessageRole.ASSISTANT,
                content=[
                    MessageContent(
                        type=MessageContentType.TEXT,
                        text=f"I apologize, but I encountered an error: {error_message}",
                    )
                ],
            ),
            model=self.model.model,
            created_at=datetime.datetime.now(datetime.timezone.utc),
            finish_reason="error",
        )

    def _update_performance_metrics(
        self, input_text: str, start_time: datetime.datetime
    ) -> None:
        """Update performance tracking metrics."""
        input_length = len(input_text.split())

        # Update running averages
        count = self._performance_metrics["summaries_generated"]
        current_avg_input = self._performance_metrics["average_input_length"]

        self._performance_metrics["summaries_generated"] += 1
        self._performance_metrics["average_input_length"] = (
            current_avg_input * count + input_length
        ) / (count + 1)

    async def summarize_text(
        self, text: str, max_length: Optional[int] = None, style: str = "balanced"
    ) -> str:
        """
        Convenience method for direct text summarization with style options.

        Args:
            text: Text to summarize
            max_length: Maximum summary length in words
            style: Summary style ("brief", "balanced", "detailed")
        """
        try:
            # Adjust prompt based on style
            style_prompts = {
                "brief": "Provide a very concise summary focusing only on the most essential points.",
                "balanced": "Provide a well-balanced summary covering the key points and main ideas.",
                "detailed": "Provide a comprehensive summary that covers all important aspects while remaining concise.",
            }

            style_instruction = style_prompts.get(style, style_prompts["balanced"])

            # Create styled prompt
            processed_text = self._preprocess_text(text)
            optimal_length = max_length or self._calculate_optimal_summary_length(
                processed_text
            )

            summary_prompt = f"""{style_instruction}
Keep the summary to approximately {optimal_length} words.

Text to summarize:
{processed_text}

Summary:"""

            # Generate summary
            summary_chunks = []
            async for chunk in self.llm.astream(summary_prompt):
                if hasattr(chunk, "content") and chunk.content:
                    if isinstance(chunk.content, str):
                        summary_chunks.append(chunk.content)

            return "".join(summary_chunks).strip()

        except Exception as e:
            self._logger.error(f"Error in summarize_text: {e}")
            return f"Error generating summary: {str(e)}"

    def clear_cache(self) -> None:
        """Clear the summary cache."""
        self._summary_cache.clear()
        self._logger.info("Summary cache cleared")

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "cache_size": len(self._summary_cache),
            "cache_hits": self._performance_metrics["cache_hits"],
            "hit_rate": (
                self._performance_metrics["cache_hits"]
                / max(self._performance_metrics["summaries_generated"], 1)
            ),
        }

    def get_performance_metrics(self) -> dict:
        """Get current performance metrics."""
        return {**self._performance_metrics, "cache_stats": self.get_cache_stats()}

    def __del__(self) -> None:
        """Enhanced cleanup with performance logging."""
        try:
            model_name = (
                getattr(self.model, "name", "unknown")
                if hasattr(self, "model")
                else "unknown"
            )

            # Log final performance metrics
            if hasattr(self, "_performance_metrics"):
                metrics = self._performance_metrics
                cache_stats = (
                    self.get_cache_stats() if hasattr(self, "_summary_cache") else {}
                )
                self._logger.info(
                    f"BARTSummarizationPipe {model_name} final metrics: "
                    f"summaries={metrics['summaries_generated']}, "
                    f"cache_hits={metrics['cache_hits']}, "
                    f"avg_input_length={metrics['average_input_length']:.1f}, "
                    f"cache_size={cache_stats.get('cache_size', 0)}"
                )

            # Clean up resources
            if hasattr(self, "_summary_cache"):
                self._summary_cache.clear()
            if hasattr(self, "tokenizer"):
                del self.tokenizer
            if hasattr(self, "llm"):
                del self.llm

            # Clear CUDA cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

        except Exception as e:
            logger.error(f"Error cleaning up BARTSummarizationPipe: {e}")
