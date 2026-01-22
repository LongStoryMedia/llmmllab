"""
LCEL serialization helpers for tool composability.
Implements abstraction through LCEL RunnableSequence composition.
"""

from typing import Any, Callable, Dict, List, Optional
from langchain_core.tools import BaseTool

from models import Tool
from utils.logging import llmmllogger
from composer.tools.registry import ToolRegistry


class RunnableToolComposer:
    """
    Composes tools using LangChain Expression Language (LCEL).

    Implements the abstraction mandate by creating composable RunnableSequences
    that can be chained together and wrapped as single tools using .as_tool().
    """

    @staticmethod
    def chain_tools(tools: List[BaseTool], name: str, description: str) -> BaseTool:
        """
        Chain multiple tools together using LCEL pipe operator.

        Creates a RunnableSequence from tools and wraps it with .as_tool()
        to provide abstraction for the main agent LLM.
        """
        try:
            if not tools:
                raise ValueError("Cannot chain empty tool list")

            if len(tools) == 1:
                # Single tool - just return it
                return tools[0]

            # Create runnable sequence by chaining tools
            sequence = tools[0]
            for tool in tools[1:]:
                sequence = sequence | tool

            # Wrap sequence as a single tool for abstraction
            composed_tool = sequence.as_tool(name=name, description=description)

            llmmllogger.log_tool_generation(
                tool_spec=f"chain:{name}",
                method="composed",
                success=True,
                additional_context={"tool_count": len(tools)},
            )

            return composed_tool

        except Exception as e:
            llmmllogger.log_error(
                e, {"context": "tool_chaining", "tool_count": len(tools)}
            )
            # Return first tool as fallback
            if not tools:
                raise ValueError("Cannot return tool from empty list")
            return tools[0]

    @staticmethod
    def parallel_tools(tools: List[BaseTool], name: str, description: str) -> BaseTool:
        """
        Compose tools to run in parallel and combine results.

        Creates a parallel execution pattern for tools that can run independently.
        """
        try:
            if not tools:
                raise ValueError("Cannot compose empty tool list")

            if len(tools) == 1:
                return tools[0]

            # Create parallel execution using RunnableParallel
            from langchain_core.runnables import RunnableParallel

            parallel_dict = {f"tool_{i}": tool for i, tool in enumerate(tools)}
            parallel_runner = RunnableParallel(parallel_dict)

            # Add result aggregation
            def aggregate_results(results: Dict[str, Any]) -> str:
                """Aggregate parallel tool results into single output."""
                aggregated = []
                for key, result in results.items():
                    aggregated.append(f"{key}: {result}")
                return "\n".join(aggregated)

            # Chain parallel execution with aggregation
            composed_sequence = parallel_runner | aggregate_results

            # Wrap as tool
            composed_tool = composed_sequence.as_tool(
                name=name, description=description
            )

            llmmllogger.log_tool_generation(
                tool_spec=f"parallel:{name}",
                method="composed",
                success=True,
                additional_context={"tool_count": len(tools), "execution": "parallel"},
            )

            return composed_tool

        except Exception as e:
            llmmllogger.log_error(e, {"context": "parallel_tool_composition"})
            # Return first tool as fallback
            if not tools:
                raise ValueError("Cannot return tool from empty list")
            return tools[0]

    @staticmethod
    def conditional_tool(
        condition_fn: Callable,
        true_tool: BaseTool,
        false_tool: BaseTool,
        name: str,
        description: str,
    ) -> BaseTool:
        """
        Create a conditional tool that selects between two tools based on input.

        Implements conditional logic using LCEL RunnableBranch.
        """
        try:
            from langchain_core.runnables import RunnableBranch

            # Create conditional branch
            conditional_runner = RunnableBranch(
                (condition_fn, true_tool), false_tool  # Default case
            )

            # Wrap as tool
            conditional_tool = conditional_runner.as_tool(
                name=name, description=description
            )

            llmmllogger.log_tool_generation(
                tool_spec=f"conditional:{name}",
                method="composed",
                success=True,
                additional_context={"type": "conditional"},
            )

            return conditional_tool

        except Exception as e:
            llmmllogger.log_error(e, {"context": "conditional_tool_composition"})
            return true_tool  # Fallback to true tool

    @staticmethod
    def serialize_tool_chain(tools: List[BaseTool]) -> Dict[str, Any]:
        """
        Serialize a tool chain for persistence and reuse.

        Enables dynamic tool chains to be stored and retrieved from the tool registry.
        """
        try:
            serialized = {
                "type": "tool_chain",
                "tools": [],
                "composition_type": "sequential",
            }

            for tool in tools:
                tool_data = {
                    "name": tool.name,
                    "description": tool.description,
                    "class": tool.__class__.__name__,
                }
                serialized["tools"].append(tool_data)

            return serialized

        except Exception as e:
            llmmllogger.log_error(e, {"context": "tool_serialization"})
            return {"type": "tool_chain", "tools": [], "error": str(e)}

    @staticmethod
    def deserialize_tool_chain(
        serialized_data: Dict[str, Any], tool_registry: ToolRegistry
    ) -> Optional[BaseTool]:
        """
        Deserialize a tool chain from stored data.

        Reconstructs tool chains from serialized representations.
        """
        try:
            if serialized_data.get("type") != "tool_chain":
                return None

            tools = []
            for tool_data in serialized_data.get("tools", []):
                # This would need to reconstruct tools from the registry
                # Implementation depends on tool registry interface
                pass  # Placeholder for now

            if tools:
                composition_type = serialized_data.get("composition_type", "sequential")
                if composition_type == "sequential":
                    return RunnableToolComposer.chain_tools(
                        tools,
                        name="deserialized_chain",
                        description="Reconstructed tool chain",
                    )
                elif composition_type == "parallel":
                    return RunnableToolComposer.parallel_tools(
                        tools,
                        name="deserialized_parallel",
                        description="Reconstructed parallel tools",
                    )

            return None

        except Exception as e:
            llmmllogger.log_error(e, {"context": "tool_deserialization"})
            return None
