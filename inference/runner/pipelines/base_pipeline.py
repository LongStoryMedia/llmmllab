from abc import ABC, abstractmethod
from typing import (
    List,
    Any,
    Union,
    AsyncGenerator,
    Iterator,
    Optional,
    Type,
    Generator,
    TypeVar,
)
import logging
import asyncio
from datetime import datetime, UTC

import torch
from .helpers import get_dtype
from models import (
    Message,
    MessageContent,
    MessageRole,
    MessageContentType,
    ChatResponse,
    ModelParameters,
    ChatReq,
    Model,
)
from langchain_core.language_models.llms import LLM
from langchain_core.callbacks.manager import CallbackManagerForLLMRun

# Configure logging
logger = logging.getLogger(__name__)
type Embeddings = List[List[float]]


class BasePipeline(LLM, ABC):
    """
    Base abstract class for all pipeline implementations.

    Inherits from LangChain's LLM class to make pipelines directly compatible
    with LangChain's agents and chains.

    All concrete pipeline classes should inherit from this class
    and implement the required methods:
    - run: Process messages and generate responses
    - __del__: Clean up resources when the pipeline is destroyed
    """

    # Class-level attributes
    model_def: Model = None  # type: ignore
    device: str = "cpu"
    _is_loaded: bool = False

    def __init__(self, model_definition: Model):
        """Initialize the pipeline with default attributes."""
        # Initialize LLM
        super().__init__()

        # Set model definition first since other initializations might need it
        self.model_def = model_definition

        # Initialize attributes
        self.device = getattr(self, "device", "cpu")
        self._is_loaded = getattr(self, "_is_loaded", False)

    @abstractmethod
    def run(self, req: ChatReq) -> Generator[ChatResponse, Any, None]:
        """
        Process the chat request and generate a response using the loaded model.

        Args:
            req (ChatReq): The chat request containing messages, model parameters, and other settings.

        Returns:
            Generator[ChatResponse, Any, None]: A generator yielding chat response chunks.
        """
        raise NotImplementedError("Subclasses must implement the run method.")

    def get(
        self, messages: List[Message], params: Optional[ModelParameters] = None
    ) -> str:
        """
        Get a response for the given messages using the model.

        Args:
            messages (List[Message]): The list of messages to process.
            params (Optional[ModelParameters]): The model parameters to use for generation.

        Returns:
            str: The generated text response.
        """
        # Create a ChatReq object from the messages and params
        req = ChatReq(
            messages=messages,
            stream=False,
            options=params,
            conversation_id=0,
        )

        # Process each chunk from the generator
        return "".join(
            content_item.text
            for chunk in self.run(req)
            if chunk.message and chunk.message.content
            for content_item in chunk.message.content
            if content_item.text
        )

    @property
    def _llm_type(self) -> str:
        """Return the type of LLM"""
        return "base-pipeline"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs,
    ) -> str:
        """
        Process a single text prompt and return a response.
        This implements the required LLM interface method.

        Args:
            prompt (str): The text prompt to process.
            stop (Optional[List[str]]): Stop sequences. Defaults to None.
            run_manager (Optional[CallbackManagerForLLMRun]): Callback manager.

        Returns:
            str: The generated text response.
        """
        # Convert prompt to a Message and call run
        message = Message(
            role=MessageRole.USER,
            content=[
                MessageContent(type=MessageContentType.TEXT, text=prompt, url=None)
            ],
            tool_calls=None,
            thinking=None,
            id=None,
            created_at=datetime.now(),
        )

        # Create parameters with stop sequences if provided
        params = None
        if stop:
            # Create ModelParameters with just the stop sequences
            params = ModelParameters(
                num_ctx=None,
                repeat_last_n=None,
                repeat_penalty=None,
                temperature=None,
                seed=None,
                stop=stop,
                num_predict=None,
                top_k=None,
                top_p=None,
                min_p=None,
            )

        return self.get([message], params)

    async def _agenerate(
        self,
        prompts: List[str],
        stop: Optional[List[str]] = None,
        run_manager=None,
        **kwargs,
    ):
        """
        Async implementation to integrate with LLM interface.

        Args:
            prompts: List of text prompts
            stop: Optional stop sequences
            run_manager: Optional callback manager

        Returns:
            LLMResult: Object with generated text
        """
        from langchain_core.outputs import LLMResult, Generation

        # Create parameters with stop sequences if provided
        params = None
        if stop:
            # Create ModelParameters with just the stop sequences
            params = ModelParameters(
                num_ctx=None,
                repeat_last_n=None,
                repeat_penalty=None,
                temperature=None,
                seed=None,
                stop=stop,
                num_predict=None,
                top_k=None,
                top_p=None,
                min_p=None,
            )

        generations = []
        for prompt in prompts:
            message = Message(
                role=MessageRole.USER,
                content=[
                    MessageContent(type=MessageContentType.TEXT, text=prompt, url=None)
                ],
                tool_calls=None,
                thinking=None,
                id=None,
                created_at=datetime.now(),
            )
            try:
                # Create a ChatReq object and call run
                req = ChatReq(
                    conversation_id=0,
                    messages=[message],
                    stream=False,
                    options=params,
                )
                response_generator = self.run(req)
                full_text = ""

                for chunk in response_generator:
                    if chunk.message and chunk.message.content:
                        for content_item in chunk.message.content:
                            if content_item.text:
                                full_text += content_item.text

                generations.append([Generation(text=full_text)])
            except (ValueError, RuntimeError, KeyError) as e:
                logger.error(f"Error generating response: {e}")
                generations.append([Generation(text=f"Error: {str(e)}")])

        return LLMResult(generations=generations)

    async def generate_stream(
        self,
        prompt: Union[str, List[Message]],
        params: Optional[ModelParameters] = None,
        **kwargs,
    ) -> AsyncGenerator[str, None]:
        """
        Generate a streaming response for a prompt.

        This method provides backward compatibility with older code.
        New code should use the run method which returns ChatResponse objects.

        Args:
            prompt: Either a string prompt or a list of Message objects
            params: Optional model parameters
            **kwargs: Additional arguments to pass to the model

        Yields:
            str: Chunks of the generated response as they become available
        """
        # Convert string prompt to Message if needed
        messages = []
        if isinstance(prompt, str):
            messages = [
                Message(
                    role=MessageRole.USER,
                    content=[
                        MessageContent(
                            type=MessageContentType.TEXT, text=prompt, url=None
                        )
                    ],
                    tool_calls=None,
                    thinking=None,
                    id=None,
                    created_at=datetime.now(),
                )
            ]
        else:
            messages = prompt

        # Create a ChatReq object and call run
        req = ChatReq(
            conversation_id=0,
            messages=messages,
            stream=True,
            options=params,
        )

        # Process each chunk from the generator
        response_generator = self.run(req)

        for chunk in response_generator:
            if chunk.message and chunk.message.content:
                for content_item in chunk.message.content:
                    if content_item.text:
                        yield content_item.text

    async def emb(
        self,
        texts: Union[str, List[str]],
        is_query: Optional[bool] = None,
        matryoshka_dim: Optional[int] = None,
    ) -> List[List[float]]:
        """
        Generate embeddings for one or more texts using the runner.

        Args:
            texts: The text or list of texts to embed
            model_path: Path or ID of the embedding model
            is_query: Whether the text is a query (True), document (False), or auto-detect (None)
            matryoshka_dim: Optional dimension for Matryoshka embedding truncation (256, 512, or 768)

        Returns:
            A list of embeddings for each input text
        """
        return []

    def _setup_quantization_config(self) -> Any:
        """
        Set up the quantization configuration based on the model details.

        Returns:
            BitsAndBytesConfig: The quantization configuration parameters.
        """
        from transformers.utils.quantization_config import BitsAndBytesConfig

        if not self.model_def or not hasattr(self.model_def, "details"):
            return None

        quantization_config = {
            "load_in_8bit": False,
            "load_in_4bit": False,
            "llm_int8_threshold": 6.0,
            "llm_int8_skip_modules": None,
            "llm_int8_enable_fp32_cpu_offload": False,
            "llm_int8_has_fp16_weight": False,
            "bnb_4bit_compute_dtype": None,
            "bnb_4bit_quant_type": "fp4",
            "bnb_4bit_use_double_quant": False,
            "bnb_4bit_quant_storage": torch.uint8,
        }

        if (
            hasattr(self.model_def.details, "quantization_level")
            and self.model_def.details.quantization_level is not None
        ):
            quant_level = self.model_def.details.quantization_level.lower()

            if quant_level.startswith(("q4", "int4", "nf4", "fp4", "q2", "int2")):
                quantization_config.update(
                    {
                        "load_in_4bit": True,
                        "bnb_4bit_quant_type": "fp4" if quant_level == "fp4" else "nf4",
                        "bnb_4bit_compute_dtype": get_dtype(self.model_def),
                        "bnb_4bit_use_double_quant": quant_level.startswith(
                            ("q2", "int2")
                        ),
                    }
                )
            elif quant_level.startswith(("q8", "int8")):
                quantization_config.update(
                    {
                        "load_in_8bit": True,
                    }
                )

        return BitsAndBytesConfig(**quantization_config)

    def _process_response(self, response: Any) -> str:
        """
        Extract text from various response formats.

        Args:
            response: The response object from the run method

        Returns:
            str: The extracted text
        """
        # Handle simple string response
        if isinstance(response, str):
            return response

        # Handle generator/streaming response
        if hasattr(response, "__iter__") and hasattr(response, "__next__"):
            return self._process_streaming_response(response)

        # Try various attribute access patterns for structured responses
        try:
            # Case: response has direct text attribute
            if hasattr(response, "text"):
                return response.text

            # Case: response has message with content
            if hasattr(response, "message") and hasattr(response.message, "content"):
                for content in response.message.content:
                    if hasattr(content, "text") and content.text:
                        return content.text

            # Case: response has content list directly
            if hasattr(response, "content") and isinstance(response.content, list):
                for content in response.content:
                    if hasattr(content, "text") and content.text:
                        return content.text
        except (AttributeError, TypeError, ValueError) as e:
            logger.warning(f"Error extracting text from response: {e}")

        # Return string representation as fallback
        return str(response)

    def _process_streaming_response(self, response_iter: Iterator) -> str:
        """
        Process a streaming/iterator response.

        Args:
            response_iter: Iterator or generator returning response chunks

        Returns:
            str: The complete concatenated text
        """
        full_text = ""
        try:
            for chunk in response_iter:
                # Handle ChatResponse objects
                if hasattr(chunk, "message") and not isinstance(chunk, str):
                    message = getattr(chunk, "message")
                    if hasattr(message, "content") and getattr(message, "content"):
                        content_list = getattr(message, "content")
                        for content_item in content_list:
                            if (
                                hasattr(content_item, "text")
                                and getattr(content_item, "text") is not None
                            ):
                                full_text += getattr(content_item, "text")
                # Handle simple string chunks
                elif isinstance(chunk, str):
                    full_text += chunk
            return full_text
        except (StopIteration, RuntimeError, ValueError) as e:
            logger.warning(f"Error consuming generator: {e}")
            return full_text  # Return whatever we've collected so far

    def _extract_embedding_from_response(self, responses) -> Optional[List[float]]:
        """
        Extract embedding from model responses.

        Args:
            responses: List of responses from the model

        Returns:
            Embedding vector as list of floats or None if not found
        """
        # Extract embedding from the context field of ChatResponse
        for response in responses:
            if hasattr(response, "context") and isinstance(response.context, list):
                return response.context

        return None
