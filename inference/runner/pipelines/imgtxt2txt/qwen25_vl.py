"""
Pipeline for Qwen 2.5 Vision Language GGUF models.
Enhanced with LangGraph integration and tool calling support.
"""

import os
import logging
import re
import json
import datetime
from typing import List, Optional, Dict, Any, cast, Type, Union, AsyncGenerator
from llama_cpp import Llama
from llama_cpp.llama_chat_format import Qwen25VLChatHandler
from langchain_core.tools import BaseTool
from langchain_core.messages import AIMessage
from langgraph.graph.state import CompiledStateGraph
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition

from models import (
    ChatResponse,
    Message,
    MessageContent,
    MessageContentType,
    MessageRole,
    Model,
    ModelProfile,
)
from utils.langgraph import (
    LangGraphState,
    build_lc_messages,
    coerce_to_langchain_message_dict,
    coerce_to_lc_message,
)
from models.lang_chain_message import LangChainMessage
from ..base_langgraph import CircuitBreakerConfig, BaseLangGraphPipeline
from ..llamacpp.base_llamacpp import BaseLlamaCppPipeline

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

ReturnType = Union[str, ChatResponse]


class Qwen25VLPipeline(BaseLangGraphPipeline):
    """
    Pipeline class for Qwen 2.5 Vision Language GGUF models using llama-cpp-python.
    Uses the Qwen25VLChatHandler for proper multimodal support.
    Clean implementation with only essential methods.
    """

    # Override allowed return types to include Type for compatibility with typing system
    allowed_return_types: tuple[type, ...] = (str, ChatResponse, list, Type)

    def __init__(
        self,
        model: Model,
        profile: ModelProfile,
        expected_return_type: Optional[type] = None,
        circuit_config: Optional[CircuitBreakerConfig] = None,
    ):
        """Initialize a Qwen25VLPipeline instance."""
        # Create logger early so we can use it
        self._logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        # Log the received circuit config for debugging
        if circuit_config is not None:
            self._logger.info(
                f"Qwen25VLPipeline: Received circuit_config with perplexity_guard={circuit_config.enable_perplexity_guard}"
            )
        else:
            self._logger.info(
                "Qwen25VLPipeline: No circuit_config provided, will use defaults from BaseLangGraphPipeline"
            )

        # Let the parent class handle circuit breaker configuration and defaults
        # Initialize with ChatResponse as the expected return type for multimodal
        super().__init__(
            model,
            profile,
            expected_return_type or ChatResponse,
            circuit_config,
        )
        self.model = model
        self.profile = profile
        self.llm: Optional[Llama] = None

        # Validate required model details
        if not (model.details and model.model):
            raise ValueError("Model definition requires model details.")

        # Validate GGUF file
        gguf_path = self._get_gguf_path()
        self._validate_gguf_file(gguf_path)

        self._logger.info(f"Initialized Qwen 2.5 VL GGUF pipeline: {model.name}")

    def _validate_gguf_file(self, gguf_path: str) -> None:
        """Validate that the GGUF file exists and is accessible."""
        # Allow bypassing validation in dev/test environments
        if os.environ.get("ALLOW_MISSING_GGUF", "false").lower() in (
            "1",
            "true",
            "yes",
        ):  # pragma: no cover
            self._logger.warning(
                f"Skipping GGUF validation for dev/test (ALLOW_MISSING_GGUF set). Expected at: {gguf_path}"
            )
            return

        if not os.path.exists(gguf_path):
            raise FileNotFoundError(f"GGUF file not found: {gguf_path}")

        if not os.access(gguf_path, os.R_OK):
            raise PermissionError(f"Cannot read GGUF file: {gguf_path}")

    def _get_gguf_path(self) -> str:
        """Get GGUF file path."""
        return (
            self.model.details.gguf_file
            if self.model.details.gguf_file
            else self.model.model
        )

    def _initialize_llama_cpp_direct(self) -> None:
        """Initialize the Llama model with multimodal support."""
        if self.llm is not None:
            return

        gguf_path = self._get_gguf_path()
        mmproj_path = "/models/qwen2.5-vl-32b-instruct/mmproj-Qwen_Qwen2.5-VL-32B-Instruct-bf16.gguf"

        # Validate file paths
        if not os.path.exists(gguf_path):
            raise FileNotFoundError(f"GGUF model file not found: {gguf_path}")
        if not os.path.exists(mmproj_path):
            raise FileNotFoundError(f"MMProj file not found: {mmproj_path}")

        self._logger.info(f"Loading GGUF model from: {gguf_path}")
        self._logger.info(f"Loading mmproj from: {mmproj_path}")

        # Get circuit breaker configuration for perplexity guard
        enable_perplexity = self.circuit_config.enable_perplexity_guard
        if enable_perplexity is None:
            enable_perplexity = True  # Default to enabled if not specified

        logits_all = enable_perplexity
        logprobs = 1 if enable_perplexity else 0

        self._logger.info(
            f"Perplexity guard {'enabled' if enable_perplexity else 'disabled'} - loading with logits_all={logits_all}, logprobs={logprobs}"
        )

        try:
            chat_handler = Qwen25VLChatHandler(clip_model_path=mmproj_path)
            self.llm = Llama(
                model_path=gguf_path,
                chat_handler=chat_handler,
                n_gpu_layers=-1,
                n_threads=4,
                verbose=True,
                logits_all=logits_all,  # Respect circuit breaker configuration
                logprobs=logprobs,  # Respect circuit breaker configuration
                embedding=False,
                n_ctx=96000,
                type_k=1,
                type_v=1,
                n_batch=256,
                n_ubatch=128,
                flash_attn=True,
                tensor_split=[0.5, 0.25, 0.25],
                f16_kv=True,
                use_mlock=False,
                use_mmap=True,
                numa=True,
            )
            self._logger.info("Successfully loaded Qwen 2.5 VL model")
        except Exception as e:
            self._logger.error(f"Failed to load model: {e}")
            raise RuntimeError(f"Failed to load Qwen2.5-VL model: {e}") from e

    def _format_messages(self, messages: List[Message]) -> List[Dict[str, Any]]:
        """Convert internal messages to OpenAI format."""
        formatted_messages = []
        for message in messages:
            role = message.role.value.lower()
            content_list = []

            for content_item in message.content:
                if content_item.type == MessageContentType.TEXT:
                    content_list.append({"type": "text", "text": content_item.text})
                elif content_item.type == MessageContentType.IMAGE:
                    if hasattr(content_item, "url") and content_item.url:
                        content_list.append(
                            {
                                "type": "image_url",
                                "image_url": {"url": content_item.url},
                            }
                        )

            formatted_messages.append({"role": role, "content": content_list})
        return formatted_messages

    def _parse_qwen_tool_calls(self, content: str) -> List[Dict[str, Any]]:
        """Parse Qwen function calls from generated content - supports multiple formats."""
        import json
        import re

        tool_calls = []
        
        # Pattern 1a: Look for proper Qwen function call format (arguments as string)
        function_call_pattern_str = r'"function_call":\s*\{\s*"name":\s*"([^"]+)",\s*"arguments":\s*"([^"]+)"\s*\}'
        function_matches_str = re.findall(function_call_pattern_str, content, re.DOTALL)
        
        for i, (name, args_str) in enumerate(function_matches_str):
            try:
                # Parse the arguments JSON string
                args = json.loads(args_str)
                formatted_call = {
                    "name": name,
                    "args": args,
                    "id": f"call_{i}_{name}",
                    "type": "tool_call"
                }
                tool_calls.append(formatted_call)
                self._logger.debug(f"Parsed Qwen function_call (string args): {formatted_call}")
            except (json.JSONDecodeError, KeyError) as e:
                self._logger.warning(f"Failed to parse function_call arguments '{args_str}': {e}")
                continue
        
        # Pattern 1b: Look for proper Qwen function call format (arguments as object)
        if not tool_calls:
            function_call_pattern_obj = r'"function_call":\s*\{\s*"name":\s*"([^"]+)",\s*"arguments":\s*(\{[^}]+\})\s*\}'
            function_matches_obj = re.findall(function_call_pattern_obj, content, re.DOTALL)
            
            for i, (name, args_str) in enumerate(function_matches_obj):
                try:
                    args = json.loads(args_str)
                    formatted_call = {
                        "name": name,
                        "args": args,
                        "id": f"call_{i}_{name}",
                        "type": "tool_call"
                    }
                    tool_calls.append(formatted_call)
                    self._logger.debug(f"Parsed Qwen function_call (object args): {formatted_call}")
                except (json.JSONDecodeError, KeyError) as e:
                    self._logger.warning(f"Failed to parse function_call arguments '{args_str}': {e}")
                    continue

        # Pattern 2: Look for mixed function_call tags (what we see in logs)
        if not tool_calls:
            mixed_pattern = r'<function_call>\s*(\{.*?\})\s*</(?:function_call|FunctionCall)>'
            mixed_matches = re.findall(mixed_pattern, content, re.DOTALL | re.IGNORECASE)
            
            for i, match in enumerate(mixed_matches):
                try:
                    tool_data = json.loads(match)
                    
                    if "name" in tool_data:
                        formatted_call = {
                            "name": tool_data["name"],
                            "args": tool_data.get("arguments", {}),
                            "id": f"call_{i}_{tool_data['name']}",
                            "type": "tool_call"
                        }
                        tool_calls.append(formatted_call)
                        self._logger.debug(f"Parsed mixed function_call: {formatted_call}")
                    else:
                        self._logger.warning(f"Mixed function call missing 'name' field: {match[:100]}...")
                        
                except (json.JSONDecodeError, KeyError) as e:
                    self._logger.warning(f"Failed to parse mixed function call from: {match[:100]}... Error: {e}")
                    continue

        if tool_calls:
            self._logger.info(f"Successfully parsed {len(tool_calls)} tool calls from content")
        else:
            self._logger.warning("No tool calls found in content")
            
        return tool_calls

    def _clean_tool_calls_from_content(self, content: str) -> str:
        """Remove tool call patterns from content to get clean user-facing text."""
        import re
        
        # Remove function_call JSON patterns (proper Qwen format)
        func_call_pattern = r'"function_call":\s*\{\s*"name":\s*"[^"]+",\s*"arguments":\s*"[^"]+"\s*\}'
        content = re.sub(func_call_pattern, '', content, flags=re.DOTALL)
        
        # Remove mixed function_call tags
        mixed_pattern = r'<function_call>\s*\{.*?\}\s*</(?:function_call|FunctionCall)>'
        content = re.sub(mixed_pattern, '', content, flags=re.DOTALL | re.IGNORECASE)
        
        # Clean up extra whitespace
        content = re.sub(r'\n\s*\n\s*\n', '\n\n', content)
        
        return content

    def _build_system_prompt(self, tools: Optional[List[BaseTool]] = None) -> str:
        """Build system prompt with multimodal tool calling support."""
        base_prompt = """You are Qwen 2.5 VL, a helpful multimodal AI assistant that can analyze images and text. 
You can see and understand images, answer questions about visual content, and use tools when needed.

When you need to use tools, format your response using proper function calls."""

        if tools and len(tools) > 0:
            # Build tool descriptions using proper Qwen format
            formatted_tools = []
            for tool in tools:
                tool_def = {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": {}
                }
                
                # Extract parameters from tool args
                if hasattr(tool, 'args_schema') and tool.args_schema:
                    try:
                        if hasattr(tool.args_schema, 'model_json_schema'):
                            schema = tool.args_schema.model_json_schema()  # type: ignore
                            tool_def["parameters"] = schema.get("properties", {})
                        elif hasattr(tool.args_schema, 'schema'):
                            schema = tool.args_schema.schema()  # type: ignore
                            tool_def["parameters"] = schema.get("properties", {})
                    except Exception as e:
                        self._logger.warning(f"Could not extract schema for tool {tool.name}: {e}")
                    
                formatted_tools.append(tool_def)
            
            tools_text = json.dumps(formatted_tools, indent=2)
            
            function_example = '{"function_call": {"name": "tool_name", "arguments": "{\\"param\\": \\"value\\"}"}}'
            
            tool_prompt = f"""

You have access to the following tools:
{tools_text}

To use a tool, respond with a function call in this format:
{function_example}

Always explain what you're doing before calling a tool and provide a clear response after."""

            return base_prompt + tool_prompt
        
        return base_prompt

    async def _agent_node(self, state: LangGraphState) -> Dict[str, Any]:
        """Agent node for multimodal processing with tool calling."""
        if state.current_iteration >= state.max_iterations:
            timeout_error = f"Maximum iterations ({state.max_iterations}) reached. Stopping to prevent infinite loops."
            lang_chain_message = LangChainMessage(
                content=timeout_error,
                type="ai",
                additional_kwargs={},
                response_metadata={},
            )
            return {
                "messages": [lang_chain_message],
                "current_iteration": state.current_iteration + 1,
            }

        try:
            # Initialize LLM if not done yet
            if self.llm is None:
                self._initialize_llama_cpp_direct()

            # Convert LangChain messages to our Message format for processing
            our_messages = []
            for lc_msg in state.messages:
                if hasattr(lc_msg, 'content') and hasattr(lc_msg, 'type'):
                    role_map = {
                        'human': MessageRole.USER,
                        'ai': MessageRole.ASSISTANT,
                        'system': MessageRole.SYSTEM,
                        'tool': MessageRole.TOOL
                    }
                    role = role_map.get(lc_msg.type, MessageRole.USER)
                    
                    # Handle content - can be string or list of content items
                    content_list = []
                    if isinstance(lc_msg.content, str):
                        content_list = [MessageContent(type=MessageContentType.TEXT, text=lc_msg.content)]
                    elif isinstance(lc_msg.content, list):
                        for item in lc_msg.content:
                            if isinstance(item, dict):
                                if item.get('type') == 'text':
                                    content_list.append(MessageContent(type=MessageContentType.TEXT, text=item.get('text', '')))
                                elif item.get('type') == 'image_url':
                                    url = item.get('image_url', {}).get('url', '')
                                    content_list.append(MessageContent(type=MessageContentType.IMAGE, url=url))
                    
                    if content_list:
                        our_messages.append(Message(role=role, content=content_list))

            # Build messages for LLM with multimodal support
            messages = self._format_messages(our_messages)

            # Generate response using llama.cpp chat completion
            if self.llm:
                response = self.llm.create_chat_completion(
                    messages=messages,  # type: ignore
                    max_tokens=4000,
                    temperature=0.7,
                    stream=False
                )
                
                # Extract content from response
                response_content = ""
                if isinstance(response, dict) and "choices" in response:
                    choice = response["choices"][0]
                    if "message" in choice and "content" in choice["message"]:
                        response_content = choice["message"]["content"] or ""
                
                # Parse tool calls from Qwen's format
                tool_calls = self._parse_qwen_tool_calls(response_content)

                # Create final response with tool calls if found
                if tool_calls:
                    # Remove tool call patterns from the visible content
                    clean_content = self._clean_tool_calls_from_content(response_content)
                    formatted_response = AIMessage(
                        content=clean_content if clean_content else "Let me process that for you.",
                        tool_calls=tool_calls
                    )
                    self._logger.info(f"Qwen2.5VL parsed {len(tool_calls)} tool calls")
                    for i, tool_call in enumerate(tool_calls):
                        self._logger.debug(f"Tool call {i}: {tool_call['name']} with args {tool_call['args']}")
                else:
                    # No tool calls, return regular response
                    formatted_response = AIMessage(content=response_content)
                    self._logger.info("No tool calls found in Qwen2.5VL response")
            else:
                formatted_response = AIMessage(content="LLM not initialized")

            # Convert to LangChain format
            coerced_message = coerce_to_langchain_message_dict(formatted_response)
            lang_chain_message = LangChainMessage(**coerced_message)

            return {
                "messages": [lang_chain_message],
                "current_iteration": state.current_iteration + 1,
            }

        except Exception as e:
            self._logger.error(f"Error in Qwen2.5VL agent node: {e}")
            error_response = LangChainMessage(
                content=f"I encountered an error while processing your request: {str(e)}",
                type="ai",
                additional_kwargs={},
                response_metadata={},
            )
            return {
                "messages": [error_response],
                "current_iteration": state.current_iteration + 1,
            }

    def create_graph(
        self, tools: Optional[List[BaseTool]] = None
    ) -> CompiledStateGraph:
        """Create LangGraph state graph for multimodal processing with tool calling."""
        workflow = StateGraph(LangGraphState)
        
        # Add nodes
        workflow.add_node("agent", self._agent_node)
        
        if tools and len(tools) > 0:
            workflow.add_node("tools", ToolNode(tools))
        
        # Add edges
        workflow.add_edge(START, "agent")
        
        if tools and len(tools) > 0:
            workflow.add_conditional_edges("agent", tools_condition)
            workflow.add_edge("tools", "agent")
        else:
            workflow.add_edge("agent", END)
        
        return workflow.compile()

    def _create_system_prompt(self) -> str:
        """Create system prompt for multimodal processing."""
        return self._build_system_prompt()
