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
    # Check if already patched
    if hasattr(tool, '_original_func'):
        # Already patched, return as-is
        return tool
    
    # Find the original function - could be in 'func', 'coroutine', or other attributes
    original_func = None
    if hasattr(tool, 'func') and callable(getattr(tool, 'func', None)):
        original_func = getattr(tool, 'func')
    elif hasattr(tool, 'coroutine') and callable(getattr(tool, 'coroutine', None)):
        original_func = getattr(tool, 'coroutine')
    
    if original_func:
        # Get the original function signature ONCE
        import inspect
        original_sig = inspect.signature(original_func)
        
        # Pre-compute required parameters (excluding injected ones) from the ORIGINAL signature
        required_params = []
        for name, param in original_sig.parameters.items():
            # Skip parameters with defaults
            if param.default != param.empty:
                continue
                
            # Skip injection parameters by checking the annotation
            annotation = param.annotation
            is_injected = False
            
            # Handle Annotated types
            if hasattr(annotation, '__origin__') and hasattr(annotation, '__metadata__'):
                # This is an Annotated type, check metadata for injection markers
                metadata = getattr(annotation, '__metadata__', ())
                for meta in metadata:
                    if hasattr(meta, '__name__'):
                        if meta.__name__ in ['InjectedState', 'InjectedToolCallId']:
                            is_injected = True
                            break
            
            # Handle string annotations
            elif isinstance(annotation, str):
                if 'InjectedState' in annotation or 'InjectedToolCallId' in annotation:
                    is_injected = True
            
            if not is_injected:
                required_params.append(name)
        
        print(f"🔍 DEBUG: Tool '{tool.name}' original signature: {original_sig}")
        print(f"🔍 DEBUG: Identified required parameters: {required_params}")
        
        # Create filtered schema
        filtered_schema = create_filtered_args_schema(original_func)
        
        # Create wrapper function that handles injection parameters
        async def wrapper_func(**kwargs):
            """Wrapper that provides dummy injection parameters."""
            # Simple debug log
            print(f"🔧 WRAPPER CALLED with {len(kwargs)} args")
            print(f"🔍 Raw kwargs: {kwargs}")
            
            # Handle the case where LLM wraps arguments in 'kwargs'
            actual_kwargs = kwargs
            if len(kwargs) == 1 and 'kwargs' in kwargs:
                print("🔄 Unwrapping nested kwargs structure")
                actual_kwargs = kwargs['kwargs']
                print(f"🔍 Unwrapped kwargs: {actual_kwargs}")
            
            # Use the pre-computed required_params instead of computing them again
            print(f"🎯 Required parameters: {required_params}")
            print(f"🎯 Available parameters: {list(actual_kwargs.keys())}")
            
            # Build filtered kwargs
            filtered_kwargs = {}
            for param_name, param in original_sig.parameters.items():
                if param_name in actual_kwargs:
                    # Include parameter from actual_kwargs
                    filtered_kwargs[param_name] = actual_kwargs[param_name]
                elif param_name in required_params:
                    # Missing required parameter - this is a problem
                    print(f"❌ DEBUG: Required parameter '{param_name}' missing from {actual_kwargs}")
                    # Try to provide a helpful error message
                    if param_name == 'query' and not actual_kwargs:
                        raise ValueError(f"Tool called without parameters. Expected 'query' parameter for web search.")
                    else:
                        raise ValueError(f"Required parameter '{param_name}' not provided to tool")
                    
            print(f"🎯 Final filtered kwargs: {filtered_kwargs}")
            
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