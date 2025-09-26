"""
Pipeline for Qwen 2.5 Vision Language GGUF models.
Simple implementation inheriting from BaseLlamaCppPipeline.
"""

import os
import logging
import re
import json
from typing import List, Optional, Dict, Any, AsyncGenerator
from llama_cpp import Llama
from llama_cpp.llama_chat_format import Qwen25VLChatHandler

from models import (
    ChatResponse,
    Message,
    MessageContent,
    MessageContentType,
    MessageRole,
    Model,
    ModelProfile,
)
from ..llamacpp.base_llamacpp import BaseLlamaCppPipeline

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"


class Qwen25VLPipeline(BaseLlamaCppPipeline):
    """
    Pipeline class for Qwen 2.5 Vision Language GGUF models using llama-cpp-python.
    Uses the Qwen25VLChatHandler for proper multimodal support.
    Inherits from BaseLlamaCppPipeline for consistency.
    """

    def __init__(
        self,
        model: Model,
        profile: ModelProfile,
        expected_return_type: Optional[type] = None,
    ):
        """Initialize a Qwen25VLPipeline instance."""
        super().__init__(model, profile, expected_return_type)
        self._logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self._logger.info("Qwen25VLPipeline initialized")

    def _get_gguf_path(self) -> str:
        """Get the GGUF file path from the profile."""
        model_path = self.profile.model_path
        if not model_path:
            raise ValueError(f"Model path not configured for {self.profile.name}")
        
        # Ensure it's an absolute path
        if not os.path.isabs(model_path):
            model_path = os.path.join("/models", model_path)
        
        self._logger.info(f"Using model path: {model_path}")
        return model_path

    def _create_llm(self, gguf_path: str) -> Llama:
        """Initialize the LlamaCpp model with vision support."""
        try:
            # Initialize with vision chat handler
            chat_handler = Qwen25VLChatHandler()
            
            # Get parameters from profile, with safe access
            n_ctx = getattr(self.profile.parameters, 'num_ctx', 8192)
            n_batch = getattr(self.profile.parameters, 'batch_size', 512)
            seed = getattr(self.profile.parameters, 'seed', -1)
            
            llm = Llama(
                model_path=gguf_path,
                chat_handler=chat_handler,
                n_ctx=n_ctx,
                n_gpu_layers=-1,  # Use all GPU layers
                n_batch=n_batch,
                seed=seed,
                logits_all=False,  # Set to False for better performance
                verbose=True,
            )
            
            self._logger.info(
                f"Initialized Qwen25VL model with ctx={n_ctx}, batch={n_batch}, seed={seed}"
            )
            return llm
            
        except Exception as e:
            self._logger.error(f"Failed to initialize Qwen25VL model: {e}")
            raise

    async def _create_system_prompt(self, tools: Optional[List] = None) -> str:
        """Create system prompt for Qwen2.5-VL - required by BaseLlamaCppPipeline."""
        return """You are Qwen2.5-VL, a helpful AI assistant with vision capabilities. You can:
- Analyze and describe images in detail
- Answer questions about visual content
- Help with text-based tasks
- Provide accurate, helpful responses

Always be helpful, accurate, and concise in your responses."""

    def _build_system_prompt(self) -> str:
        """Build the system prompt for Qwen2.5-VL."""
        return """You are Qwen2.5-VL, a helpful AI assistant with vision capabilities. You can:
- Analyze and describe images in detail
- Answer questions about visual content
- Help with text-based tasks
- Provide accurate, helpful responses

Always be helpful, accurate, and concise in your responses."""

    def _parse_qwen_tool_calls(self, content: str) -> List[Dict[str, Any]]:
        """Parse tool calls from Qwen response content."""
        tool_calls = []
        
        # Multiple patterns to catch different formats
        patterns = [
            r'<function_call>\s*(\{[^}]+\})\s*</function_call>',
            r'<function_call>([^<]+)</function_call>',
            r'```json\s*(\{[^}]+\})\s*```',
            r'\{[^}]*"name"[^}]*"arguments"[^}]*\}',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, content, re.DOTALL | re.IGNORECASE)
            for match in matches:
                try:
                    # Clean up the match
                    clean_match = match.strip()
                    if not clean_match.startswith('{'):
                        clean_match = '{' + clean_match
                    if not clean_match.endswith('}'):
                        clean_match = clean_match + '}'
                    
                    tool_data = json.loads(clean_match)
                    if 'name' in tool_data:
                        tool_calls.append(tool_data)
                        
                except json.JSONDecodeError:
                    continue
        
        return tool_calls

    async def _process_messages(self, messages: List[Message]) -> str:
        """Process messages and extract text content, preserving image data."""
        processed_parts = []
        
        for message in messages:
            if message.content:
                if isinstance(message.content, str):
                    processed_parts.append(message.content)
                elif isinstance(message.content, list):
                    # Handle multimodal content
                    for part in message.content:
                        if part.type == MessageContentType.TEXT:
                            processed_parts.append(part.text)
                        elif part.type == MessageContentType.IMAGE_URL:
                            # For vision models, we preserve the image reference
                            processed_parts.append(f"[Image: {part.image_url.url}]")
        
        return " ".join(processed_parts)

    async def _generate_response(self, messages: List[Message]) -> ChatResponse:
        """Generate response using the vision model."""
        try:
            # Get GGUF path and initialize model
            gguf_path = self._get_gguf_path()
            llm = self._create_llm(gguf_path)
            
            # Convert messages to the format expected by llama-cpp-python
            chat_messages = []
            for msg in messages:
                content_text = await self._process_messages([msg])
                
                chat_messages.append({
                    "role": msg.role.value,
                    "content": content_text
                })
            
            # Generate response
            self._logger.info(f"Generating response with {len(chat_messages)} messages")
            
            # Use chat completion
            response = llm.create_chat_completion(
                messages=chat_messages,
                max_tokens=getattr(self.profile.parameters, 'max_tokens', 2000),
                temperature=getattr(self.profile.parameters, 'temperature', 0.1),
                stream=False,
            )
            
            # Extract content
            if response and hasattr(response, 'get') and response.get('choices'):
                content = response['choices'][0]['message']['content']
                
                # Clean up any text corruption (remove unwanted .strip() effects)
                # Don't use .strip() to preserve spacing
                
                self._logger.info(f"Generated response: {len(content or '')} characters")
                
                return ChatResponse(
                    id=f"qwen25vl-{hash(content or '') % 10000}",
                    model=self.model.name,
                    choices=[{
                        "index": 0,
                        "message": Message(
                            role=MessageRole.ASSISTANT,
                            content=[MessageContent(
                                type=MessageContentType.TEXT,
                                text=content or ""
                            )],
                        ),
                        "finish_reason": "stop"
                    }],
                    done=True,
                    usage=response.get('usage', {}) if hasattr(response, 'get') else {}
                )
            else:
                raise ValueError("No response generated from model")
                
        except Exception as e:
            self._logger.error(f"Error generating response: {e}")
            raise

    async def _initialize_llm(self, gguf_path: str, tools = None) -> None:
        """Initialize the LLM - required by BaseLangGraphPipeline."""
        # For now, this is a placeholder since we initialize the LLM in _generate_response
        pass

    def create_graph(self, tools = None):
        """Create graph - simple implementation for BaseLangGraphPipeline compatibility."""
        # For now, return a simple placeholder since we don't use LangGraph
        class SimpleGraph:
            def invoke(self, state, **kwargs):
                return {"messages": []}
        
        return SimpleGraph()

    async def generate(
        self, 
        messages: List[Message], 
        stream: bool = False,
        **kwargs
    ) -> ChatResponse:
        """Main generation method."""
        if stream:
            raise NotImplementedError("Streaming not yet implemented for Qwen2.5VL")
        
        return await self._generate_response(messages)