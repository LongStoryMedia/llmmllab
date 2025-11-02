"""
Tools Agent Subgraph - Simple LangChain Agent Pattern

Following the exact pattern from LangChain documentation:
https://docs.langchain.com/oss/python/langgraph/workflows-agents#agents

Simple architecture:
1. chat_agent: LLM node that can call tools
2. tool_executor: ToolNode that executes tools
3. Built-in tools_condition for routing
4. No custom logic - let LangChain handle everything
"""

from typing import Dict, Any, List

from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, BaseMessage
from langchain.tools import BaseTool
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.types import Command

from models import LangChainMessage, NodeMetadata
from composer.graph.state import WorkflowState, ToolsState
from composer.agents.chat_agent import ChatAgent
from composer.tools.registry import ToolRegistry
from composer.middleware import VisionSummarizationMiddleware
from composer.utils.conversion import (
    convert_base_langchain_to_messages,
    convert_messages_to_langchain,
)
from composer.utils.tool_call_types import (
    LangChainToolCall,
    extract_tool_call_requests,
    has_tool_calls,
)
from utils.logging import llmmllogger

logger = llmmllogger.bind(component="ToolsAgentSubgraph")


class ToolsAgentSubgraph:
    """
    Simple agent subgraph following LangChain quickstart pattern.
    """

    def __init__(
        self,
        tool_registry: ToolRegistry,
        chat_agent: ChatAgent,
        enable_vision_optimization: bool = True,
    ):
        """Initialize subgraph with dependency injection."""
        self.tool_registry = tool_registry
        self.chat_agent = chat_agent
        self.enable_vision_optimization = enable_vision_optimization
        self.graph = None
        
        # Initialize vision cache for optimization
        self.vision_cache = {}
        logger.info("🖼️ Vision optimization enabled")
        
        self._build_graph()

    def _create_tool_node(self) -> ToolNode:
        """
        Create LangGraph ToolNode with proper tools list and ToolRuntime injection.

        LangChain will automatically inject ToolRuntime for tools with `runtime: ToolRuntime` parameter.
        This is the correct pattern - no manual ToolRuntime creation needed.
        """
        try:
            # Get executable tools from registry
            executable_tools = self.tool_registry.get_all_executable_tools()
            tools_dict: dict[str, BaseTool] = (
                executable_tools if executable_tools else {}
            )

            if not tools_dict:
                logger.warning("No tools available for ToolNode creation")
                return ToolNode([])  # Empty tool node

            # Convert to list of tool functions for ToolNode
            tools_list = list(tools_dict.values())

            logger.info(
                f"🛠️ Creating ToolNode with {len(tools_list)} tools: {list(tools_dict.keys())}"
            )

            # Create ToolNode - LangChain will handle ToolRuntime injection automatically
            return ToolNode(tools_list)

        except Exception as e:
            logger.error(f"Failed to create ToolNode: {e}")
            return ToolNode([])  # Return empty tool node on error

    def _build_graph(self) -> None:
        """Build simple agent following LangChain quickstart pattern."""
        try:
            # Simple StateGraph following LangChain docs exactly
            builder = StateGraph(ToolsState)

            # Add chat agent node
            builder.add_node("chat_agent", self._chat_agent_node)

            # Add tool executor node - must be named "tools" for tools_condition
            tool_node = self._create_tool_node()
            builder.add_node("tools", tool_node)

            # EXACTLY like the LangChain quickstart - use built-in tools_condition
            builder.add_conditional_edges(
                "chat_agent",
                tools_condition,  # Use built-in routing - expects "tools" node
                {
                    "tools": "tools",
                    "__end__": END,
                },
            )

            # Simple continuation after tools
            builder.add_edge("tools", "chat_agent")

            # Start with chat agent
            builder.add_edge(START, "chat_agent")

            # Compile with reasonable recursion limit
            self.graph = builder.compile()

            logger.info("Simple tools agent subgraph built following LangChain pattern")

        except Exception as e:
            logger.error(f"Failed to build agent subgraph: {e}")
            raise

    def _extract_tool_call_requests_from_message(
        self, msg: BaseMessage | LangChainMessage
    ) -> List[LangChainToolCall]:
        """
        Extract tool call requests from a message with strong typing.

        Returns:
            List of LangChain tool call requests (what AI wants to call)
        """
        if isinstance(msg, BaseMessage):
            return extract_tool_call_requests(msg)

        # Handle our LangChainMessage format
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            validated_calls = []
            for tc in msg.tool_calls:
                if isinstance(tc, dict) and "name" in tc and "args" in tc:
                    validated_calls.append(
                        LangChainToolCall(
                            name=tc["name"], args=tc["args"], id=tc.get("id")
                        )
                    )
            return validated_calls

        return []

    def _extract_previous_tool_call_requests(
        self, messages: List[BaseMessage]
    ) -> List[LangChainToolCall]:
        """Extract all previous tool call requests from conversation history."""
        previous_requests = []
        for msg in messages:
            tool_call_requests = self._extract_tool_call_requests_from_message(msg)
            previous_requests.extend(tool_call_requests)

        logger.debug(
            f"Extracted {len(previous_requests)} previous tool call requests: {[req['name'] for req in previous_requests]}"
        )
        return previous_requests

    def _is_duplicate_tool_call_request(
        self,
        current_request: LangChainToolCall,
        previous_requests: List[LangChainToolCall],
    ) -> bool:
        """
        Check if a tool call request is a duplicate of a previous one.

        Only considers exact duplicates (same tool name AND same arguments).
        Different arguments to the same tool are allowed for legitimate use cases like:
        - Multiple web searches with different queries
        - Reading multiple URLs with read_web_content
        - Multiple API calls with different parameters
        """
        duplicate_count = 0
        for prev_request in previous_requests:
            if (
                prev_request["name"] == current_request["name"]
                and prev_request["args"] == current_request["args"]
            ):
                duplicate_count += 1

        # Allow 1 duplicate (so 2 total calls with same args), block after that
        # This handles cases where the AI might legitimately retry a failed call
        return duplicate_count >= 2

    def _optimize_vision_content(self, messages: List) -> List:
        """Simple vision optimization using standard LangChain patterns."""
        import re
        import hashlib
        import json
        
        optimized_messages = []
        for msg in messages:
            has_vision = False
            content_hash = None
            
            # Check for different vision content formats
            content = getattr(msg, 'content', '')
            
            if isinstance(content, str):
                # Check for vision tokens (our format)
                vision_pattern = r'<\|vision_start\|>.*?<\|vision_end\|>'
                if re.search(vision_pattern, content):
                    has_vision = True
                    content_hash = hashlib.md5(content.encode()).hexdigest()[:8]
            
            elif isinstance(content, list):
                # Check for LangChain content blocks with images
                for block in content:
                    if isinstance(block, dict):
                        if block.get('type') == 'image_url' or block.get('type') == 'image':
                            has_vision = True
                            # Create hash from image URL or data
                            image_content = json.dumps(block, sort_keys=True)
                            content_hash = hashlib.md5(image_content.encode()).hexdigest()[:8]
                            break
            
            if has_vision and content_hash:
                # Check if we've processed this before
                if content_hash in self.vision_cache:
                    # Replace with cached summary
                    cached_summary = self.vision_cache[content_hash]
                    logger.info(f"🖼️ Using cached analysis for hash {content_hash}: {cached_summary[:50]}...")
                    
                    if isinstance(content, str):
                        # Replace vision tokens with summary
                        optimized_content = re.sub(
                            r'<\|vision_start\|>.*?<\|vision_end\|>', 
                            f"[Previous image analysis: {cached_summary}]", 
                            content
                        )
                        new_msg = HumanMessage(content=optimized_content)
                    else:
                        # Replace ALL image content blocks with text summary - this is key!
                        new_content = []
                        has_images = False
                        for block in content:
                            if isinstance(block, dict) and (block.get('type') == 'image_url' or block.get('type') == 'image'):
                                has_images = True
                                # Don't add image blocks - replace with text only
                            else:
                                new_content.append(block)
                        
                        # Add the cached summary as a text block
                        if has_images:
                            new_content.insert(0, {
                                'type': 'text', 
                                'text': f"[Previous image analysis: {cached_summary}]"
                            })
                        
                        new_msg = HumanMessage(content=new_content if new_content else f"[Previous image analysis: {cached_summary}]")
                    
                    optimized_messages.append(new_msg)
                    logger.info(f"🖼️ Using cached vision analysis (hash: {content_hash})")
                else:
                    # First time seeing this image - store hash for later caching
                    setattr(msg, '_vision_hash', content_hash)
                    optimized_messages.append(msg)
                    logger.debug(f"🖼️ New vision content detected (hash: {content_hash})")
            else:
                optimized_messages.append(msg)
                
        return optimized_messages
    
    def _cache_vision_analysis(self, messages: List, response_content: str):
        """Extract and cache vision analysis from AI response."""
        import re
        
        # Look for messages that had vision content
        for msg in messages:
            if hasattr(msg, '_vision_hash'):
                content_hash = msg._vision_hash
                
                # Extract analysis from response (simple heuristic)
                if response_content and len(response_content) > 50:
                    # Use first meaningful sentence as summary  
                    sentences = response_content.split('.')
                    for sentence in sentences:
                        clean_sentence = sentence.strip()
                        if len(clean_sentence) > 30 and 'image' in clean_sentence.lower():
                            summary = clean_sentence + '.'
                            break
                    else:
                        # Fallback to first 100 characters
                        summary = response_content[:100].strip()
                    
                    if summary:
                        self.vision_cache[content_hash] = summary
                        logger.info(f"🖼️ Cached vision analysis (hash: {content_hash})")

    async def _chat_agent_node(self, state: ToolsState) -> ToolsState:
        """LangChain agent node with vision optimization preprocessing."""
        from langchain_core.messages import AIMessage, HumanMessage
        from composer.utils.conversion import convert_langchain_messages_to_messages, message_to_langchain_message
        
        # Apply vision optimization to messages before sending to model  
        messages = state["messages"]
        optimized_messages = self._optimize_vision_content(messages)
        
        # Log optimization activity
        vision_optimized = any(hasattr(msg, '_vision_hash') for msg in optimized_messages)
        if vision_optimized:
            logger.info("🖼️ Vision optimization applied to messages")
        
        try:
            # Get available tools
            tools_dict = self.tool_registry.get_all_executable_tools()
            tools_list = list(tools_dict.values()) if tools_dict else None
            
            # Convert optimized LangChain messages to Message objects for direct pipeline use
            # This bypasses chat_completion_with_conversion and its internal conversions
            optimized_core_messages = convert_langchain_messages_to_messages(optimized_messages)
            
            # Use the ChatAgent's base chat completion method directly
            response = await self.chat_agent.chat_completion(
                messages=optimized_messages,  # Pass LangChain messages directly
                tools=tools_list,
                stream=False
            )
            
            # Cache vision analysis from the response
            if response and hasattr(response, 'message') and response.message:
                content = str(response.message.content[0].text if response.message.content else "")
                self._cache_vision_analysis(optimized_messages, content)
            
            # Convert response message to LangChain format
            langchain_response = message_to_langchain_message(response.message) if response.message else AIMessage(content="No response generated")
            
            # Return updated state following LangChain agent pattern
            return {
                **state,
                "messages": optimized_messages + [langchain_response]
            }
            
        except Exception as e:
            logger.error(f"Error in chat agent node: {e}")
            # Fallback: return state unchanged
            return state

    # Removed _should_continue - using LangGraph's built-in tools_condition instead

    def transform_to_tools_state(self, main_state: WorkflowState) -> ToolsState:
        """Transform main WorkflowState to minimal ToolsState for agent subgraph."""
        # Get recent messages for agent context and convert to LangChain core messages
        recent_messages = getattr(main_state, "messages", [])[-10:]
        langchain_messages = []

        for msg in recent_messages:
            if hasattr(msg, "type") and hasattr(msg, "content"):
                # Convert custom LangChainMessage to proper LangChain core message
                if msg.type == "human":
                    langchain_messages.append(HumanMessage(content=msg.content))
                elif msg.type == "ai":
                    # Check if this AI message has tool calls and convert properly
                    tool_calls = []
                    if hasattr(msg, "tool_calls") and msg.tool_calls:
                        for tc in msg.tool_calls:
                            if isinstance(tc, dict):
                                tool_calls.append(
                                    {
                                        "name": tc.get("name", ""),
                                        "args": tc.get("args", {}),
                                        "id": tc.get("id", "unknown"),
                                        "type": "tool_call",
                                    }
                                )
                            else:
                                # Handle other tool call formats
                                tool_calls.append(
                                    {
                                        "name": getattr(tc, "name", ""),
                                        "args": getattr(tc, "args", {}),
                                        "id": getattr(tc, "id", "unknown"),
                                        "type": "tool_call",
                                    }
                                )

                    langchain_messages.append(
                        AIMessage(
                            content=msg.content,
                            tool_calls=tool_calls if tool_calls else [],
                        )
                    )
                elif msg.type == "tool":
                    langchain_messages.append(
                        ToolMessage(
                            content=msg.content,
                            tool_call_id=getattr(msg, "id", None) or "unknown",
                        )
                    )
                else:
                    # Default to human message for unknown types
                    langchain_messages.append(HumanMessage(content=str(msg.content)))
            else:
                # Already a proper LangChain message, use as-is
                langchain_messages.append(msg)

        # Pass full user_config object for tool access (tools need full config objects)
        user_config = getattr(main_state, "user_config", None)

        return {
            "messages": langchain_messages,
            "user_id": getattr(main_state, "user_id", ""),
            "conversation_id": getattr(main_state, "conversation_id", 0),
            "user_config": user_config,
            "system_config": None,  # Not available in WorkflowState
            "current_date": getattr(main_state, "current_date", ""),
            "tool_call_count": 0,
        }

    def transform_to_main_state(
        self, agent_result: Dict[str, Any], main_state: WorkflowState
    ) -> Dict[str, Any]:
        """Transform agent subgraph results back to main WorkflowState updates."""
        updates = {}

        # Add new messages from agent execution
        if agent_result.get("messages"):
            main_messages = getattr(main_state, "messages", [])
            agent_messages = agent_result["messages"]

            # Find messages that weren't in the original main state
            original_count = len(main_messages)
            new_messages = []

            for i, msg in enumerate(agent_messages):
                if i >= original_count:  # This is a new message from agent
                    if isinstance(msg, (AIMessage, ToolMessage)):
                        # Convert to LangChainMessage format for main state
                        logger.info(
                            f"🔄 transform_to_main_state: Converting {type(msg).__name__} with type='{msg.type}' to LangChainMessage"
                        )
                        lang_chain_msg = LangChainMessage(
                            content=msg.content,
                            type=msg.type,
                            name=getattr(msg, "name", None),
                            id=getattr(msg, "id", None)
                            or getattr(msg, "tool_call_id", None),
                            additional_kwargs=getattr(msg, "additional_kwargs", {}),
                            response_metadata=getattr(msg, "response_metadata", {}),
                        )
                        logger.info(
                            f"🔄 transform_to_main_state: Created LangChainMessage with type='{lang_chain_msg.type}'"
                        )
                        new_messages.append(lang_chain_msg)

            if new_messages:
                updates["messages"] = main_messages + new_messages

        return updates

    async def execute(self, main_state: WorkflowState) -> Command:
        """Execute the agent subgraph and return Command with state updates."""
        try:
            if not self.graph:
                logger.error("Agent subgraph not initialized")
                return Command(update={})

            # Transform to agent state
            tools_state = self.transform_to_tools_state(main_state)

            # Execute the agent subgraph with LangChain defaults
            result = await self.graph.ainvoke(tools_state)

            # Transform results back to main state updates
            logger.info(
                f"🔄 ToolsAgentSubgraph: Calling transform_to_main_state with result containing {len(result.get('messages', []))} messages"
            )
            updates = self.transform_to_main_state(result, main_state)

            logger.info(
                f"🔄 ToolsAgentSubgraph: Agent subgraph completed with {len(updates)} state updates"
            )
            if "messages" in updates:
                logger.info(
                    f"🔄 ToolsAgentSubgraph: Returning {len(updates['messages']) - len(main_state.messages)} new messages"
                )
            return Command(update=updates)

        except Exception as e:
            logger.error(f"Agent subgraph execution failed: {e}", exc_info=True)
            return Command(update={})
