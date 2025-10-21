"""
Tool schema filtering utilities to prevent InjectedState from being included in LLM schemas.

This module provides utilities to create clean tool schemas for LLMs by filtering out
InjectedState and InjectedToolCallId parameters that should not be visible to the model.
"""

from typing import Any, Dict, Type, get_type_hints, get_origin, get_args
from pydantic import BaseModel, create_model
from langchain_core.tools import BaseTool
import inspect


def create_filtered_args_schema(tool_func) -> Type[BaseModel]:
    """
    Create a filtered args schema that excludes InjectedState and InjectedToolCallId parameters.
    
    Args:
        tool_func: The tool function decorated with @tool
        
    Returns:
        A Pydantic model class with only the parameters that should be visible to the LLM
    """
    # Get the function signature
    sig = inspect.signature(tool_func)
    
    # Fields to include in the schema (exclude injected parameters)
    schema_fields = {}
    
    for param_name, param in sig.parameters.items():
        # Use raw annotation to preserve Annotated types
        param_type = param.annotation
        
        # Check if this is an Annotated type with injection markers
        if hasattr(param_type, '__origin__') and get_origin(param_type) is not None:
            # This is likely Annotated[Type, InjectedState] or similar
            origin = get_origin(param_type)
            if origin is not None:
                args = get_args(param_type)
                if len(args) >= 2:
                    # Check if any of the annotation args indicate injection
                    has_injection = any(
                        'Injected' in str(arg) for arg in args[1:]
                    )
                    if has_injection:
                        continue  # Skip this parameter
                    else:
                        # Use the first type arg (the actual type)
                        param_type = args[0]
        
        # Skip if parameter type is still an injection type after resolution
        if 'Injected' in str(param_type):
            continue
            
        # Handle case where param_type is still annotated but should be included
        if param_type == inspect.Parameter.empty:
            param_type = Any
            
        # Add to schema if not injected
        default_value = param.default if param.default != inspect.Parameter.empty else ...
        schema_fields[param_name] = (param_type, default_value)
    
    # Create a dynamic Pydantic model with only the non-injected fields
    filtered_model = create_model(
        f"{tool_func.__name__}_FilteredSchema",
        **schema_fields
    )
    
    return filtered_model


def patch_tool_schema(tool: BaseTool) -> BaseTool:
    """
    Patch a LangChain tool to use a filtered args schema that excludes injected parameters.
    
    This creates both a filtered schema for the LLM and a wrapper function that provides
    the injection parameters when LangChain calls the tool.
    
    Args:
        tool: The LangChain tool to patch
        
    Returns:
        The same tool with a modified args_schema and wrapper function
    """
    # Find the original function - could be in 'func', 'coroutine', or other attributes
    original_func = None
    if hasattr(tool, 'func') and callable(getattr(tool, 'func', None)):
        original_func = getattr(tool, 'func')
    elif hasattr(tool, 'coroutine') and callable(getattr(tool, 'coroutine', None)):
        original_func = getattr(tool, 'coroutine')
    
    if original_func:
        # Create filtered schema
        filtered_schema = create_filtered_args_schema(original_func)
        
        # Create wrapper function that handles injection parameters
        async def wrapper_func(**kwargs):
            """Wrapper that provides dummy injection parameters."""
            import inspect  # Import here to avoid circular imports
            
            # Simple debug log
            print(f"� WRAPPER CALLED with {len(kwargs)} args")
            
            # Get the parameters from the original function signature
            original_sig = inspect.signature(original_func)
            filtered_kwargs = {}
            
            # Always ensure we have a query parameter if it's expected
            for param_name, param in original_sig.parameters.items():
                if param_name == 'query' and param_name not in kwargs:
                    # This shouldn't happen, but just in case
                    print(f"❌ DEBUG: Expected parameter '{param_name}' missing from {kwargs}")
                    raise ValueError(f"Required parameter '{param_name}' not provided to tool")
                elif param_name in kwargs:
                    # Include parameter from kwargs
                    filtered_kwargs[param_name] = kwargs[param_name]
            
            # Add tool_call_id if missing but expected by original function
            if 'tool_call_id' in original_sig.parameters and 'tool_call_id' not in filtered_kwargs:
                filtered_kwargs['tool_call_id'] = 'langchain_call'
                
            # Add state if missing but expected by original function
            if 'state' in original_sig.parameters and 'state' not in filtered_kwargs:
                from composer.graph.state import WorkflowState
                from models.default_configs import create_default_user_config
                
                # Create minimal state for LangChain tool calls
                minimal_state = WorkflowState(
                    user_id='langchain_user',
                    conversation_id=0,
                    user_config=create_default_user_config('langchain_user'),
                    messages=[],
                    things_to_remember=[],
                )
                filtered_kwargs['state'] = minimal_state
            
            # Call original function with only the parameters it accepts
            return await original_func(**filtered_kwargs)
        
        # Replace the args_schema
        tool.args_schema = filtered_schema
        
        # Replace the coroutine with our wrapper (this is the async function)
        if hasattr(tool, 'coroutine'):
            setattr(tool, 'coroutine', wrapper_func)
        
        # Store original function for debugging 
        if not hasattr(tool, '_original_func'):
            setattr(tool, '_original_func', original_func)
    
    return tool