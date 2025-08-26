from abc import ABC, abstractmethod
from typing import (
    List,
    Any,
    Union,
    AsyncGenerator,
    Iterator,
    Optional,
    Generator,
    Dict,
)
import logging
from datetime import datetime

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
from langchain_core.callbacks.manager import (
    CallbackManagerForLLMRun,
    AsyncCallbackManagerForLLMRun,
)
from langchain_core.language_models.llms import LLM
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.outputs import (
    GenerationChunk,
    LLMResult,
    Generation,
    ChatResult,
    ChatGeneration,
    ChatGenerationChunk,
)
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    FunctionMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
    ChatMessage,
)

# Configure logging
logger = logging.getLogger(__name__)
type Embeddings = List[List[float]]


class BasePipeline(BaseChatModel, ABC):
    """
    Base abstract class for all pipeline implementations.

    This class inherits from LangChain's BaseChatModel class to enable direct integration with
    LangChain's components and pipelines. It implements the required LangChain chat model
    interface methods while maintaining backward compatibility with existing code.

    All concrete pipeline classes should inherit from this class
    and implement the required methods:
    - run: Process messages and generate responses
    - __del__: Clean up resources when the pipeline is destroyed

    Example usage with LangChain:
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import StrOutputParser

        # Create your pipeline instance
        pipeline = YourPipelineClass(model_definition)

        # Create a LangChain chain
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful AI assistant."),
            ("human", "Tell me about {topic}")
        ])
        chain = prompt | pipeline | StrOutputParser()

        # Run the chain
        result = chain.invoke({"topic": "artificial intelligence"})

        # Or stream responses
        for chunk in chain.stream({"topic": "quantum computing"}):
            print(chunk, end="")
    """

    # Class-level attributes
    model_def: Model = None  # type: ignore
    model_parameters: ModelParameters = None  # type: ignore
    device: str = "cpu"
    _is_loaded: bool = False

    def __init__(self, model_definition: Model, model_parameters: ModelParameters):
        """Initialize the pipeline with default attributes."""
        # Set model definition first since other initializations might need it
        self.model_def = model_definition
        self.model_parameters = model_parameters

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

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs,
    ) -> ChatResult:
        """
        Generate a chat result from a list of messages.
        Required method for BaseChatModel compatibility.

        Args:
            messages (List[BaseMessage]): LangChain messages to process
            stop (Optional[List[str]]): Optional stop sequences
            run_manager (Optional[CallbackManagerForLLMRun]): Callback manager
            **kwargs: Additional arguments

        Returns:
            ChatResult: The generated chat result
        """
        # Convert LangChain messages to our Message format
        converted_messages = self._convert_langchain_to_native_messages(messages)

        # Create parameters with stop sequences if provided
        params = self.model_parameters
        if stop:
            # Create a copy of model parameters with stop sequences
            params = ModelParameters(
                num_ctx=(
                    self.model_parameters.num_ctx if self.model_parameters else None
                ),
                repeat_last_n=(
                    self.model_parameters.repeat_last_n
                    if self.model_parameters
                    else None
                ),
                repeat_penalty=(
                    self.model_parameters.repeat_penalty
                    if self.model_parameters
                    else None
                ),
                temperature=(
                    self.model_parameters.temperature if self.model_parameters else None
                ),
                seed=self.model_parameters.seed if self.model_parameters else None,
                stop=stop,
                num_predict=(
                    self.model_parameters.num_predict if self.model_parameters else None
                ),
                top_k=self.model_parameters.top_k if self.model_parameters else None,
                top_p=self.model_parameters.top_p if self.model_parameters else None,
                min_p=self.model_parameters.min_p if self.model_parameters else None,
            )

        # Create a ChatReq object and call run
        req = ChatReq(
            messages=converted_messages,
            stream=False,
            options=params,
            conversation_id=0,
        )

        # Process the response
        response_generator = self.run(req)
        full_text = ""

        # Collect all text from chunks
        for chunk in response_generator:
            if chunk.message and chunk.message.content:
                for content_item in chunk.message.content:
                    if content_item.text:
                        # If run_manager is provided, report new token
                        if run_manager:
                            run_manager.on_llm_new_token(content_item.text)
                        full_text += content_item.text

        # Create AI message from generated text
        ai_message = AIMessage(content=full_text)

        # Return the chat result
        return ChatResult(
            generations=[ChatGeneration(message=ai_message)],
            llm_output={
                "model_name": (
                    getattr(self.model_def, "name", "unknown")
                    if self.model_def
                    else "unknown"
                )
            },
        )

    def _convert_langchain_to_native_messages(
        self, messages: List[BaseMessage]
    ) -> List[Message]:
        """
        Convert LangChain messages to native Message format.

        Args:
            messages (List[BaseMessage]): LangChain messages

        Returns:
            List[Message]: Converted messages in our format
        """
        native_messages = []

        for msg in messages:
            # Determine the role
            role = MessageRole.USER
            if isinstance(msg, SystemMessage):
                role = MessageRole.SYSTEM
            elif isinstance(msg, AIMessage):
                role = MessageRole.ASSISTANT
            elif isinstance(msg, HumanMessage):
                role = MessageRole.USER
            elif isinstance(msg, FunctionMessage) or isinstance(msg, ToolMessage):
                # Use TOOL role as a fallback if FUNCTION is not available
                role = (
                    MessageRole.TOOL
                    if hasattr(MessageRole, "TOOL")
                    else MessageRole.ASSISTANT
                )
            elif isinstance(msg, ChatMessage):
                if msg.role == "system":
                    role = MessageRole.SYSTEM
                elif msg.role == "assistant":
                    role = MessageRole.ASSISTANT
                elif msg.role == "function" or msg.role == "tool":
                    # Use TOOL role as a fallback if FUNCTION is not available
                    role = (
                        MessageRole.TOOL
                        if hasattr(MessageRole, "TOOL")
                        else MessageRole.ASSISTANT
                    )

            # Create content
            content = []
            if msg.content:
                if isinstance(msg.content, str):
                    content = [
                        MessageContent(
                            type=MessageContentType.TEXT, text=msg.content, url=None
                        )
                    ]
                elif isinstance(msg.content, list):
                    # Handle multimodal content
                    for item in msg.content:
                        if isinstance(item, dict) and "type" in item:
                            if item["type"] == "text":
                                content.append(
                                    MessageContent(
                                        type=MessageContentType.TEXT,
                                        text=item.get("text", ""),
                                        url=None,
                                    )
                                )
                            elif item["type"] == "image_url":
                                content.append(
                                    MessageContent(
                                        type=MessageContentType.IMAGE,
                                        text=None,
                                        url=item.get("image_url", {}).get("url", ""),
                                    )
                                )

            # Create native message
            native_messages.append(
                Message(
                    role=role,
                    content=content,
                    tool_calls=None,  # Would need additional conversion for tool calls
                    thinking=None,
                    id=None,
                    created_at=datetime.now(),
                )
            )

        return native_messages

    @property
    def _llm_type(self) -> str:
        """Return the type of LLM for logging purposes."""
        return "base-pipeline"

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Return a dictionary of identifying parameters."""
        model_name = (
            getattr(self.model_def, "name", "unknown") if self.model_def else "unknown"
        )
        model_id = getattr(self.model_def, "id", None) if self.model_def else None

        return {
            "model_name": model_name,
            "model_id": model_id,
            "device": self.device,
            "is_loaded": self._is_loaded,
        }

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

        # Process the request and collect the response
        result = self.get([message], params)

        # If run_manager is provided, send the token callback
        if run_manager is not None and result:
            run_manager.on_llm_new_token(result)

        return result

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs,
    ) -> Iterator[ChatGenerationChunk]:
        """
        Stream the chat model on the given messages.

        Args:
            messages (List[BaseMessage]): The list of messages to process
            stop (Optional[List[str]]): Stop sequences. Defaults to None.
            run_manager (Optional[CallbackManagerForLLMRun]): Callback manager.

        Yields:
            Iterator[ChatGenerationChunk]: An iterator of ChatGenerationChunks
        """
        # Convert LangChain messages to our Message format
        converted_messages = self._convert_langchain_to_native_messages(messages)

        # Create parameters with stop sequences if provided
        params = self.model_parameters
        if stop:
            # Create a copy of model parameters with stop sequences
            params = ModelParameters(
                num_ctx=(
                    self.model_parameters.num_ctx if self.model_parameters else None
                ),
                repeat_last_n=(
                    self.model_parameters.repeat_last_n
                    if self.model_parameters
                    else None
                ),
                repeat_penalty=(
                    self.model_parameters.repeat_penalty
                    if self.model_parameters
                    else None
                ),
                temperature=(
                    self.model_parameters.temperature if self.model_parameters else None
                ),
                seed=self.model_parameters.seed if self.model_parameters else None,
                stop=stop,
                num_predict=(
                    self.model_parameters.num_predict if self.model_parameters else None
                ),
                top_k=self.model_parameters.top_k if self.model_parameters else None,
                top_p=self.model_parameters.top_p if self.model_parameters else None,
                min_p=self.model_parameters.min_p if self.model_parameters else None,
            )

        # Create a ChatReq object and call run
        req = ChatReq(
            conversation_id=0,
            messages=converted_messages,
            stream=True,  # Important: set streaming to true
            options=params,
        )

        # Process each chunk from the generator
        for chunk in self.run(req):
            if chunk.message and chunk.message.content:
                for content_item in chunk.message.content:
                    if content_item.text:
                        # Create an AI message chunk
                        ai_message_chunk = AIMessageChunk(content=content_item.text)
                        chat_chunk = ChatGenerationChunk(message=ai_message_chunk)

                        # If run_manager is provided, send the token callback
                        # Note: This should happen BEFORE yielding the chunk
                        if run_manager is not None:
                            run_manager.on_llm_new_token(
                                content_item.text, chunk=chat_chunk
                            )

                        yield chat_chunk

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager=None,
        **kwargs,
    ) -> ChatResult:
        """
        Async implementation of _generate for BaseChatModel.

        Args:
            messages: List of LangChain messages
            stop: Optional stop sequences
            run_manager: Optional callback manager

        Returns:
            ChatResult: Object with generated chat message
        """
        # Convert LangChain messages to our Message format
        converted_messages = self._convert_langchain_to_native_messages(messages)

        # Create parameters with stop sequences if provided
        params = self.model_parameters
        if stop:
            params = ModelParameters(
                num_ctx=(
                    self.model_parameters.num_ctx if self.model_parameters else None
                ),
                repeat_last_n=(
                    self.model_parameters.repeat_last_n
                    if self.model_parameters
                    else None
                ),
                repeat_penalty=(
                    self.model_parameters.repeat_penalty
                    if self.model_parameters
                    else None
                ),
                temperature=(
                    self.model_parameters.temperature if self.model_parameters else None
                ),
                seed=self.model_parameters.seed if self.model_parameters else None,
                stop=stop,
                num_predict=(
                    self.model_parameters.num_predict if self.model_parameters else None
                ),
                top_k=self.model_parameters.top_k if self.model_parameters else None,
                top_p=self.model_parameters.top_p if self.model_parameters else None,
                min_p=self.model_parameters.min_p if self.model_parameters else None,
            )

        # Create a ChatReq object and call run
        req = ChatReq(
            messages=converted_messages,
            stream=False,
            options=params,
            conversation_id=0,
        )

        # Process the response
        response_generator = self.run(req)
        full_text = ""

        # Collect all text from chunks
        for chunk in response_generator:
            if chunk.message and chunk.message.content:
                for content_item in chunk.message.content:
                    if content_item.text:
                        # If run_manager is provided, report new token
                        if run_manager:
                            await run_manager.on_llm_new_token(content_item.text)
                        full_text += content_item.text

        # Create AI message from generated text
        ai_message = AIMessage(content=full_text)

        # Return the chat result
        return ChatResult(
            generations=[ChatGeneration(message=ai_message)],
            llm_output={
                "model_name": (
                    getattr(self.model_def, "name", "unknown")
                    if self.model_def
                    else "unknown"
                )
            },
        )

    async def _astream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager=None,
        **kwargs,
    ) -> AsyncGenerator[ChatGenerationChunk, None]:
        """
        Async version of chat stream functionality.

        Args:
            messages (List[BaseMessage]): The messages to process
            stop (Optional[List[str]]): Stop sequences.
            run_manager: Async callback manager.

        Yields:
            AsyncGenerator[ChatGenerationChunk, None]: Async generator of chunks.
        """
        # Convert LangChain messages to our Message format
        converted_messages = self._convert_langchain_to_native_messages(messages)

        # Create parameters with stop sequences if provided
        params = self.model_parameters
        if stop:
            params = ModelParameters(
                num_ctx=(
                    self.model_parameters.num_ctx if self.model_parameters else None
                ),
                repeat_last_n=(
                    self.model_parameters.repeat_last_n
                    if self.model_parameters
                    else None
                ),
                repeat_penalty=(
                    self.model_parameters.repeat_penalty
                    if self.model_parameters
                    else None
                ),
                temperature=(
                    self.model_parameters.temperature if self.model_parameters else None
                ),
                seed=self.model_parameters.seed if self.model_parameters else None,
                stop=stop,
                num_predict=(
                    self.model_parameters.num_predict if self.model_parameters else None
                ),
                top_k=self.model_parameters.top_k if self.model_parameters else None,
                top_p=self.model_parameters.top_p if self.model_parameters else None,
                min_p=self.model_parameters.min_p if self.model_parameters else None,
            )

        # Create a ChatReq object and call run
        req = ChatReq(
            conversation_id=0,
            messages=converted_messages,
            stream=True,
            options=params,
        )

        # Process each chunk from the generator
        for chunk in self.run(req):
            if chunk.message and chunk.message.content:
                for content_item in chunk.message.content:
                    if content_item.text:
                        # Create an AI message chunk
                        ai_message_chunk = AIMessageChunk(content=content_item.text)
                        chat_chunk = ChatGenerationChunk(message=ai_message_chunk)

                        # If run_manager is provided, send the token callback
                        if run_manager is not None:
                            await run_manager.on_llm_new_token(
                                content_item.text, chunk=chat_chunk
                            )

                        yield chat_chunk

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
