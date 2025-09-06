"""
Optimized summarization pipeline for Llama-Chat-Summary-3.2-3B model with performance enhancements.
"""

import os
import datetime
import logging
import hashlib
import multiprocessing
from typing import AsyncGenerator, List, cast, Optional, Dict, Any, TypeVar, Union
import torch
from transformers import AutoTokenizer
from langchain_community.chat_models.llamacpp import ChatLlamaCpp
from langchain_core.callbacks import CallbackManager, StreamingStdOutCallbackHandler
from langchain_core.prompts import ChatPromptTemplate

from langgraph.graph import StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph
from langchain_core.tools import BaseTool

from models import (
    MessageContent,
    MessageContentType,
    MessageRole,
    Model,
    Message,
    ChatResponse,
    ModelProfile,
)
from utils.message import extract_message_text
from utils.response import create_streaming_chunk, create_error_response
from ..base import BasePipelineCore

# Define generic return type for text pipelines
T = TypeVar("T", bound=Union[str, ChatResponse])

# Type hints for factory usage
LlamaChatSummarizationChatPipe = "LlamaChatSummPipe[ChatResponse]"
LlamaChatSummarizationTextPipe = "LlamaChatSummPipe[str]"


logger = logging.getLogger(__name__)


class LlamaChatSummPipe(BasePipelineCore[T]):
    """
    Optimized pipeline for Llama-Chat-Summary-3.2-3B model with enhanced performance.

    Key optimizations:
    - Smart text preprocessing and chunking
    - Adaptive summary length based on input size
    - Enhanced prompt engineering optimized for Llama
    - Performance monitoring and caching
    - Memory-efficient processing
    - Quality scoring for summary validation
    """

    llm: ChatLlamaCpp
    tokenizer: Optional[AutoTokenizer]
    _summary_cache: Dict[str, str] = {}
    _performance_metrics: dict
    # Only str or ChatResponse supported
    allowed_return_types = (str, ChatResponse)

    def __init__(
        self, model: Model, profile: ModelProfile, return_type: type = ChatResponse
    ):
        """Initialize optimized LlamaChatSummPipe instance."""
        # Enforce expected return type (str or ChatResponse)
        super().__init__(model, profile, expected_return_type=return_type)
        self._return_type = return_type

        # Initialize performance tracking
        self._performance_metrics = {
            "summaries_generated": 0,
            "cache_hits": 0,
            "average_input_length": 0,
            "average_summary_length": 0,
            "average_compression_ratio": 0.0,
        }

        # Validate model requirements - Llama models need GGUF file
        if not (model.details and model.model):
            raise ValueError(
                "Model definition for LlamaChatSummPipe must include model path or details.gguf_file"
            )

        self._logger = logging.getLogger(__name__)
        self._logger.info(
            f"Initializing optimized LlamaChatSummPipe for model: {model.id}"
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
        """Initialize tokenizer with error handling for Llama model."""
        try:
            # For Llama Chat Summary, try to load tokenizer from parent model if available
            if (
                self.model.details
                and hasattr(self.model.details, "parent_model")
                and self.model.details.parent_model
            ):
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model.details.parent_model,
                    use_fast=True,
                )
                self._logger.info(
                    "Llama tokenizer loaded successfully from parent model"
                )
            else:
                # Fallback to meta-llama/Llama-2-7b-chat-hf for Llama-based models
                self.tokenizer = AutoTokenizer.from_pretrained(
                    "meta-llama/Llama-2-7b-chat-hf",
                    use_fast=True,
                )
                self._logger.info("Llama tokenizer loaded successfully from fallback")
        except Exception as e:
            self._logger.warning(
                f"Tokenizer initialization failed, will use GGUF-based tokenization: {e}"
            )
            self.tokenizer = None

    def _initialize_llm(self, gguf_path: str) -> None:
        """Initialize LLM with Llama-Chat-Summary-specific optimizations."""
        try:
            # Llama-Chat-Summary-3.2-3B optimized settings
            self.llm = ChatLlamaCpp(
                model_path=gguf_path,
                n_gpu_layers=-1,  # Offload all layers to GPU
                n_batch=512,
                n_ctx=self.profile.parameters.num_ctx
                or 8192,  # Llama context (increased from BART)
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
                # Llama-specific generation parameters optimized for summarization
                temperature=self.profile.parameters.temperature
                or 0.1,  # Very low temperature to reduce repetition
                max_tokens=self.profile.parameters.num_predict
                or 512,  # Reduced from 1024 to prevent verbose output
                top_p=self.profile.parameters.top_p or 0.85,  # Reduced from 0.9
                top_k=self.profile.parameters.top_k or 30,  # Reduced from 50
                repeat_penalty=self.profile.parameters.repeat_penalty
                or 1.15,  # Increased from 1.05
                streaming=True,
                # Llama-specific stop tokens optimized for summarization
                stop=self.profile.parameters.stop
                or self._get_summarization_stop_tokens(),
            )

            self._logger.info("Llama Chat Summary model loaded successfully")

        except Exception as e:
            self._logger.error(f"Failed to initialize Llama LLM: {e}")
            raise

    def _get_optimal_threads(self) -> int:
        """Get optimal thread count for Llama model."""
        try:
            cpu_count = multiprocessing.cpu_count()
            # Llama models are efficient with moderate threading
            optimal_threads = min(max(cpu_count // 2, 4), 8)
            self._logger.debug(f"Using {optimal_threads} threads for Llama")
            return optimal_threads
        except Exception:
            return 4  # Conservative default for Llama

    def _get_summarization_stop_tokens(self) -> List[str]:
        """Get stop tokens optimized for Llama summarization."""
        return self.profile.parameters.stop or [
            "</s>",  # Llama's primary EOS token
            "<|endoftext|>",  # Alternative EOS
            "<|end|>",  # Llama-2 style EOS
            "[/INST]",  # Llama instruction format
            "</tool_call>",  # Tool boundaries
            "</tool_response>",
            "</think>",
            "\n\nSummary:",  # Prevent summary repetition
            "\n\nText:",  # Prevent text repetition
            "Human:",  # Prevent role bleeding
            "Assistant:",
            "User:",  # Llama chat format
            "Original text:",  # Prevent confusion
            "Summary:",
            "SUMMARY:",
            "\n\n\n",  # Multiple newlines often indicate repetition
            "The user",  # Prevent meta-commentary
            "In this",  # Common repetitive phrase starter
            "Additionally,",  # Common repetitive connector
            "Furthermore,",  # Common repetitive connector
            "Moreover,",  # Common repetitive connector
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
        """Create an optimized prompt for Llama-Chat-Summary."""
        processed_text = self._preprocess_text(text)
        optimal_length = max_length or self._calculate_optimal_summary_length(
            processed_text
        )

        # Use Llama-style instruction format for better performance
        return f"""<s>[INST] Please provide a concise and accurate summary of the following text. Focus on the key points and main ideas. Keep the summary to approximately {optimal_length} words.

Text to summarize:
{processed_text}

Summary: [/INST]"""

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

    async def process_messages(
        self, messages: List[Message], tools: Optional[List[BaseTool]] = None
    ) -> T:
        """Process messages and return appropriate response type."""

        try:
            # Extract and preprocess text
            input_texts = []
            for message in messages:
                text = extract_message_text(message)
                if text:
                    input_texts.append(text)

            if not input_texts:
                error_msg = "No text content found to summarize"
                if self._return_type == str:
                    return cast(T, error_msg)

                return cast(
                    T,
                    ChatResponse(
                        done=True,
                        message=Message(
                            role=MessageRole.ASSISTANT,
                            content=[
                                MessageContent(
                                    type=MessageContentType.TEXT,
                                    text=error_msg,
                                )
                            ],
                        ),
                        created_at=datetime.datetime.now(datetime.timezone.utc),
                        finish_reason="error",
                    ),
                )

            combined_text = "\n\n".join(input_texts)

            # Check cache first
            cache_key = self._generate_cache_key(combined_text)
            if cache_key in self._summary_cache:
                self._performance_metrics["cache_hits"] += 1
                self._logger.info("Using cached summary")
                cached_summary = self._summary_cache[cache_key]

                if self._return_type == str:
                    return cast(T, cached_summary)

                return cast(
                    T,
                    ChatResponse(
                        done=True,
                        message=Message(
                            role=MessageRole.ASSISTANT,
                            content=[
                                MessageContent(
                                    type=MessageContentType.TEXT,
                                    text=cached_summary,
                                )
                            ],
                        ),
                        created_at=datetime.datetime.now(datetime.timezone.utc),
                        finish_reason="stop",
                    ),
                )

            # Generate summary using direct approach (simpler than agent)
            summary = await self._direct_summarization_simple(combined_text)

            # Cache the result
            self._summary_cache[cache_key] = summary

            if self._return_type == str:
                return cast(T, summary)

            return cast(
                T,
                ChatResponse(
                    done=True,
                    message=Message(
                        role=MessageRole.ASSISTANT,
                        content=[
                            MessageContent(
                                type=MessageContentType.TEXT,
                                text=summary,
                            )
                        ],
                    ),
                    created_at=datetime.datetime.now(datetime.timezone.utc),
                    finish_reason="stop",
                ),
            )

        except Exception as e:
            self._logger.error(f"Error in process_messages: {e}")
            error_msg = f"Error: {str(e)[:100]}..."

            if self._return_type == str:
                return cast(T, error_msg)

            return cast(
                T,
                ChatResponse(
                    done=True,
                    message=Message(
                        role=MessageRole.ASSISTANT,
                        content=[
                            MessageContent(
                                type=MessageContentType.TEXT,
                                text=error_msg,
                            )
                        ],
                    ),
                    created_at=datetime.datetime.now(datetime.timezone.utc),
                    finish_reason="error",
                ),
            )

    async def prompt(self, text: str | List[str]) -> T:
        """Process a single message and return appropriate response type."""
        if isinstance(text, list):
            text = " ".join(text)

        # Create a simple user message
        message = Message(
            role=MessageRole.USER,
            content=[
                MessageContent(
                    type=MessageContentType.TEXT,
                    text=text,
                )
            ],
        )

        return await self.process_messages([message])

    def create_graph(
        self, tools: Optional[List[BaseTool]] = None
    ) -> CompiledStateGraph:
        """Create a simple LangGraph for BART summarization."""
        from models import LangGraphState

        # Simple state graph that just processes the input
        workflow = StateGraph(LangGraphState)

        async def summarize_node(state: LangGraphState) -> Dict[str, Any]:
            """Node that performs summarization."""
            try:
                # Extract text from messages
                combined_text = ""
                for msg in state.messages:
                    if hasattr(msg, "content") and msg.content:
                        combined_text += str(msg.content) + "\n"

                # Generate summary
                summary = await self._direct_summarization_simple(combined_text)

                # Return as a message
                from langchain_core.messages import AIMessage

                return {"messages": [AIMessage(content=summary)]}
            except Exception as e:
                from langchain_core.messages import AIMessage

                return {"messages": [AIMessage(content=f"Error: {str(e)}")]}

        workflow.add_node("summarize", summarize_node)
        workflow.add_edge(START, "summarize")
        workflow.add_edge("summarize", END)

        return workflow.compile()

    async def _direct_summarization_simple(self, text: str) -> str:
        """Simple direct summarization optimized for Llama-Chat-Summary."""
        try:
            # Preprocess text
            cleaned_text = self._preprocess_text(text)
            summary_length = self._calculate_optimal_summary_length(cleaned_text)

            # Create Llama-optimized prompt with instruction format
            prompt = f"""<s>[INST] Summarize the following text in about {summary_length} words. Be concise and capture the key points:

{cleaned_text[:6000]}

Provide only the summary without any additional text. [/INST]

Summary: """

            # Direct LLM call
            from langchain_core.messages import HumanMessage

            response = await self.llm.ainvoke([HumanMessage(content=prompt)])

            summary = ""
            if response.content:
                if isinstance(response.content, str):
                    summary = response.content
                elif isinstance(response.content, list):
                    for content in response.content:
                        if isinstance(content, str):
                            summary += content + " "
                        elif hasattr(content, "text"):
                            summary += content.text + " "
                        elif isinstance(content, dict) and "text" in content:
                            summary += content["text"] + " "

            # Clean up Llama response artifacts
            summary = summary.strip()
            if summary.startswith("Summary:"):
                summary = summary[8:].strip()

            return summary

        except Exception as e:
            self._logger.error(f"Error in direct summarization: {e}")
            return f"Error generating summary: {str(e)[:100]}..."

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
                yield create_error_response("No text content found to summarize")
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
            yield create_error_response(f"Summarization error: {str(e)}")

    async def _agent_summarization(
        self,
        messages: List[Message],
        _prompt: ChatPromptTemplate,
        _tools: List[BaseTool],
    ) -> AsyncGenerator[ChatResponse, None]:
        """Simple agent-based summarization replacement."""
        try:
            # Convert to simple text processing since we removed AgentExecutor
            combined_text = ""
            for message in messages:
                text = extract_message_text(message)
                if text:
                    combined_text += text + "\n"

            # Generate summary using direct method
            summary = await self._direct_summarization_simple(combined_text)

            # Yield as ChatResponse
            yield ChatResponse(
                done=True,
                message=Message(
                    role=MessageRole.ASSISTANT,
                    content=[
                        MessageContent(
                            type=MessageContentType.TEXT,
                            text=summary,
                        )
                    ],
                ),
                created_at=datetime.datetime.now(datetime.timezone.utc),
                finish_reason="stop",
            )

        except Exception as e:
            self._logger.error(f"Error in agent summarization: {e}")
            yield create_error_response(f"Agent summarization error: {str(e)}")

    async def _direct_summarization(
        self, text: str
    ) -> AsyncGenerator[ChatResponse, None]:
        """Direct summarization without agents."""
        try:
            # Create optimized prompt
            summary_prompt = self._create_optimized_summary_prompt(text)

            yield create_streaming_chunk("📄 Generating optimized summary...\n\n")

            # Stream the summarization
            response_chunks = []
            async for chunk in self.llm.astream(summary_prompt):
                if hasattr(chunk, "content") and chunk.content:
                    chunk_content = chunk.content
                    response_chunks.append(chunk_content)

                    if isinstance(chunk_content, str):
                        yield create_streaming_chunk(chunk_content)
                    elif isinstance(chunk_content, list):
                        for item in chunk_content:
                            if isinstance(item, str):
                                yield create_streaming_chunk(item)

            # Validate and finalize
            full_summary = "".join(response_chunks).strip()
            if full_summary:
                quality = self._score_summary_quality(text, full_summary)
                self._logger.info(
                    f"Generated summary with quality score: {quality:.2f}"
                )
                yield create_streaming_chunk("", done=True)
            else:
                yield create_streaming_chunk("Failed to generate summary", done=True)

        except Exception as e:
            self._logger.error(f"Error in direct summarization: {e}")
            yield create_streaming_chunk(
                f"Direct summarization error: {str(e)}", done=True
            )

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
            created_at=datetime.datetime.now(datetime.timezone.utc),
            finish_reason="stop",
        )

    def _update_performance_metrics(
        self, input_text: str, _start_time: datetime.datetime
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

            # Create styled prompt with Llama instruction format
            processed_text = self._preprocess_text(text)
            optimal_length = max_length or self._calculate_optimal_summary_length(
                processed_text
            )

            summary_prompt = f"""<s>[INST] {style_instruction}
Keep the summary to approximately {optimal_length} words.

Text to summarize:
{processed_text}

Provide only the summary without any additional text. [/INST]

Summary: """

            # Generate summary
            summary_chunks = []
            async for chunk in self.llm.astream(summary_prompt):
                if hasattr(chunk, "content") and chunk.content:
                    if isinstance(chunk.content, str):
                        summary_chunks.append(chunk.content)

            return "".join(summary_chunks).strip()

        except Exception as e:
            self._logger.error(f"Error generating summary: {e}")
            return f"Error: {str(e)}"

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
                    f"LlamaChatSummPipe {model_name} final metrics: "
                    f"summaries={metrics['summaries_generated']}, "
                    f"cache_hits={metrics['cache_hits']}, "
                    f"avg_input_length={metrics['average_input_length']:.1f}, "
                    f"cache_size={cache_stats.get('cache_size', 0)}"
                )

            # Clean up resources
            if hasattr(self, "_summary_cache"):
                self._summary_cache.clear()
            if hasattr(self, "tokenizer") and self.tokenizer:
                del self.tokenizer
            if hasattr(self, "llm"):
                del self.llm

            # Clear CUDA cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

        except Exception as e:
            logger.error(f"Error cleaning up LlamaChatSummPipe: {e}")
