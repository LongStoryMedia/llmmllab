"""
Example usage of the generic WorkflowExecutor with different state types.

This demonstrates how the executor can work with any CompiledStateGraph
and state type, as long as the state can be converted to dict format.
"""

import asyncio
from typing import Dict, Any, List
from datetime import datetime

# Example: Using with a simple dict state
async def example_dict_state():
    """Example using executor with plain dictionary state."""
    from composer.graph.executor import WorkflowExecutor
    
    # Mock a simple workflow (in real usage, this would be a CompiledStateGraph)
    # This is just for demonstration - normally you'd have a real LangGraph workflow
    
    simple_state = {
        "messages": ["Hello", "World"],
        "user_id": "test_user",
        "step_count": 0
    }
    
    executor = WorkflowExecutor(default_context="dict_example")
    
    print("=== Dict State Example ===")
    print(f"Initial state: {simple_state}")
    
    # In real usage, you'd stream from a real workflow:
    # async for event in executor.stream_workflow(workflow, simple_state, thread_id="test_thread"):
    #     print(f"Event: {event}")


# Example: Using with a Pydantic model state
from pydantic import BaseModel
from typing import Optional

class CustomWorkflowState(BaseModel):
    """Example custom state model."""
    messages: List[str]
    user_id: str
    conversation_id: int
    metadata: Dict[str, Any] = {}
    processing_step: str = "initial"
    completed: bool = False
    
    def model_dump(self) -> Dict[str, Any]:
        """Convert to dict format for LangGraph execution."""
        return super().model_dump()


async def example_pydantic_state():
    """Example using executor with Pydantic model state."""
    from composer.graph.executor import WorkflowExecutor
    
    # Create a custom state instance
    custom_state = CustomWorkflowState(
        messages=["User query", "Processing..."],
        user_id="user_123",
        conversation_id=456,
        metadata={"timestamp": datetime.now().isoformat()},
        processing_step="analysis"
    )
    
    executor = WorkflowExecutor(default_context="pydantic_example")
    
    print("=== Pydantic State Example ===")
    print(f"Initial state type: {type(custom_state)}")
    print(f"State dict: {custom_state.model_dump()}")
    
    # Create a thread config
    config = executor.create_thread_config("custom_thread_123")
    print(f"Thread config: {config}")


# Example: Using convenience functions
async def example_convenience_functions():
    """Example using the convenience stream_workflow function."""
    from composer.graph.executor import stream_workflow, execute_workflow
    
    # Simple state for demonstration
    state = {
        "query": "What is the weather like?",
        "user_id": "demo_user",
        "tools_available": ["weather_api", "location_service"]
    }
    
    print("=== Convenience Functions Example ===")
    print("Available convenience functions:")
    print("- stream_workflow: For streaming execution")
    print("- execute_workflow: For batch execution")
    print("- create_executor: For creating executor instances")
    
    # In real usage, you would pass a real CompiledStateGraph:
    # async for event in stream_workflow(real_workflow, state, thread_id="demo"):
    #     handle_event(event)


# Example: Custom executor with different enrichment
from composer.graph.executor import WorkflowExecutor as BaseWorkflowExecutor

class CustomWorkflowExecutor(BaseWorkflowExecutor):
    """Example of extending WorkflowExecutor with custom behavior."""
    
    def _enrich_event(self, event: Dict[str, Any], context_name: str) -> Dict[str, Any]:
        """Custom event enrichment with additional metadata."""
        # Call parent enrichment first
        enriched_event = super()._enrich_event(event, context_name)
        
        # Add custom metadata
        if "metadata" not in enriched_event:
            enriched_event["metadata"] = {}
        
        enriched_event["metadata"].update({
            "custom_executor": True,
            "enrichment_version": "1.0",
            "processor": "CustomWorkflowExecutor"
        })
        
        return enriched_event


async def example_custom_executor():
    """Example using a custom executor with extended functionality."""
    
    custom_executor = CustomWorkflowExecutor(default_context="custom_demo")
    
    print("=== Custom Executor Example ===")
    print("Custom executor with extended enrichment capabilities")
    
    # Mock event for demonstration
    mock_event = {
        "event": "on_tool_start", 
        "data": {"tool_name": "example_tool"}
    }
    
    enriched = custom_executor._enrich_event(mock_event, "test_context")
    print(f"Enriched event: {enriched}")


# Main demonstration runner
async def main():
    """Run all examples to demonstrate WorkflowExecutor usage."""
    print("🚀 WorkflowExecutor Examples\n")
    
    await example_dict_state()
    print()
    
    await example_pydantic_state()
    print()
    
    await example_convenience_functions()
    print()
    
    await example_custom_executor()
    print()
    
    print("✅ All examples completed!")
    print("\nKey Benefits of Generic WorkflowExecutor:")
    print("- Works with any CompiledStateGraph")
    print("- Supports multiple state formats (dict, Pydantic, etc.)")
    print("- Consistent event enrichment and error handling")
    print("- Reusable across different workflow types")
    print("- Easy to extend with custom behavior")


if __name__ == "__main__":
    asyncio.run(main())