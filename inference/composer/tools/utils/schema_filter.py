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
        
        # Skip parameters that are clearly injection parameters by name
        if param_name in ['tool_call_id', 'state']:
            print(f"    🚫 Skipping injection parameter: {param_name}")
            continue
            
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
                        print(f"    🚫 Skipping injection parameter: {param_name} (has injection annotation)")
                        continue  # Skip injected parameters entirely
                    else:
                        # Use the first type arg (the actual type)
                        param_type = args[0]
        
        # Skip if parameter type is still an injection type after resolution
        if 'Injected' in str(param_type):
            print(f"    🚫 Skipping injection parameter: {param_name} (injection type)")
            continue
            
        # Handle case where param_type is still annotated but should be included
        if param_type == inspect.Parameter.empty:
            param_type = Any
            
        # Add to schema - this parameter should be visible to LLM
        default_value = param.default if param.default != inspect.Parameter.empty else ...
        schema_fields[param_name] = (param_type, default_value)
        print(f"    ✅ Including parameter: {param_name} ({param_type})")
    
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
    # Check if tool is already patched to avoid double-wrapping
    if hasattr(tool, '_schema_filter_patched'):
        print(f"🔄 Tool {tool.name} already patched, skipping")
        return tool
    
    # Find the original function - could be in 'func', 'coroutine', or other attributes
    original_func = None
    
    # Debug - print all callable attributes
    print(f"🔍 Tool attributes: {[attr for attr in dir(tool) if not attr.startswith('_')]}")
    print(f"🔍 Tool type: {type(tool)}")
    
    if hasattr(tool, 'func') and callable(getattr(tool, 'func', None)):
        original_func = getattr(tool, 'func')
        print(f"🔍 Found func: {original_func}")
    elif hasattr(tool, 'coroutine') and callable(getattr(tool, 'coroutine', None)):
        original_func = getattr(tool, 'coroutine')
        print(f"🔍 Found coroutine: {original_func}")
    elif hasattr(tool, '_run') and callable(getattr(tool, '_run', None)):
        original_func = getattr(tool, '_run')
        print(f"🔍 Found _run: {original_func}")
    
    print(f"🔍 Final original_func: {original_func}")
    
    if not original_func:
        print(f"❌ Could not find callable function in tool: {tool}")
        return tool
        
    # ENHANCED: Check if this tool uses LangGraph's injection system
    # If it does, we need to create a clean schema for the LLM while preserving injection
    import inspect
    try:
        sig = inspect.signature(original_func)
        injection_params = set()
        
        # Check if the function uses InjectedToolCallId or InjectedState annotations
        for param_name, param in sig.parameters.items():
            if hasattr(param, 'annotation') and param.annotation != inspect.Parameter.empty:
                annotation_str = str(param.annotation)
                if 'InjectedToolCallId' in annotation_str or 'InjectedState' in annotation_str:
                    injection_params.add(param_name)
        
        if injection_params:
            print(f"🔧 Tool {tool.name} uses LangGraph injection - creating clean schema for LLM")
            print(f"🔧 Hiding injection parameters: {injection_params}")
            # Continue with filtering to create clean schema that hides injection params
        else:
            print(f"✅ Tool {tool.name} has no injection parameters - applying standard filtering")
            
    except Exception as e:
        print(f"⚠️  Could not check injection annotations for {tool.name}: {e}")
        # Continue with filtering as fallback
        
    print(f"🔧 Creating filtered schema...")
    # Create filtered schema (this will hide injection parameters)
    filtered_schema = create_filtered_args_schema(original_func)
    
    # Create wrapper function that handles injection parameters
    async def wrapper_func(**kwargs):
        """Wrapper that provides dummy injection parameters."""
        import inspect  # Import here to avoid circular imports
        
        # Enhanced debug log
        print(f"🔧 WRAPPER CALLED with {len(kwargs)} kwargs: {list(kwargs.keys())}")
        print(f"🔧 ALL KWARGS: {kwargs}")
        print(f"🔧 WRAPPER: original_func is: {original_func}")
        print(f"🔧 WRAPPER: original_func type: {type(original_func)}")
        
        # Handle the case where LLM wraps arguments in 'kwargs'
        actual_kwargs = kwargs
        if len(kwargs) == 1 and 'kwargs' in kwargs:
            print("🔄 Unwrapping nested kwargs structure")
            actual_kwargs = kwargs['kwargs']
            print(f"🔧 After unwrapping: {len(actual_kwargs)} args: {list(actual_kwargs.keys())}")
        else:
            print(f"🔧 Direct args: {list(actual_kwargs.keys())}")

        # Get the parameters from the original function signature
        try:
            original_sig = inspect.signature(original_func)
        except Exception as e:
            print(f"❌ Could not get signature of original_func: {e}")
            print(f"❌ original_func = {original_func}")
            raise
            
        filtered_kwargs = {}
        
        # Extract tool_call_id from the kwargs if provided (LangGraph passes this)
        tool_call_id = actual_kwargs.get('tool_call_id')
        print(f"🔍 Extracted tool_call_id from kwargs: {tool_call_id}")
        
        # Always ensure we have a query parameter if it's expected
        for param_name, param in original_sig.parameters.items():
            if param_name == 'query' and param_name not in actual_kwargs:
                # This shouldn't happen, but just in case
                print(f"❌ DEBUG: Expected parameter '{param_name}' missing from {actual_kwargs}")
                raise ValueError(f"Required parameter '{param_name}' not provided to tool")
            elif param_name in actual_kwargs:
                # Include parameter from actual_kwargs
                filtered_kwargs[param_name] = actual_kwargs[param_name]
                print(f"✅ DEBUG: Added parameter '{param_name}': {repr(actual_kwargs[param_name])[:50]}")
        
        print(f"🎯 Calling original function with: {list(filtered_kwargs.keys())}")
        
        # Add tool_call_id if missing but expected by original function
        if 'tool_call_id' in original_sig.parameters:
            if tool_call_id:
                filtered_kwargs['tool_call_id'] = tool_call_id
                print(f"🎯 Using provided tool_call_id: {tool_call_id}")
            else:
                # Fallback if no tool_call_id provided
                filtered_kwargs['tool_call_id'] = 'langchain_call'
                print(f"🎯 Injected fallback tool_call_id: langchain_call")
            
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
        print(f"📞 About to call original function: {original_func}")
        print(f"📞 Original function type: {type(original_func)}")
        print(f"📞 Is callable: {callable(original_func)}")
        result = await original_func(**filtered_kwargs)
        print(f"✅ Function call completed successfully")
        
        # CRITICAL FIX: Handle Command returns from LangGraph tools
        # LangGraph's ToolNode expects simple return values, not Command objects
        if hasattr(result, 'update') and hasattr(result.update, 'get'):
            # This is a Command object with messages - extract the ToolMessage content
            messages = result.update.get('messages', [])
            if messages and hasattr(messages[0], 'content'):
                tool_message_content = messages[0].content
                print(f"🔧 Extracted ToolMessage content from Command: {len(tool_message_content)} chars")
                return tool_message_content
            elif 'messages' in result.update and isinstance(result.update['messages'], list):
                # Handle case where messages is in the update dict
                for msg in result.update['messages']:
                    if hasattr(msg, 'content'):
                        print(f"🔧 Extracted ToolMessage content from Command update: {len(msg.content)} chars")
                        return msg.content
        
        # Handle simple Command objects that just have content
        if hasattr(result, 'content'):
            print(f"🔧 Extracted content from result: {len(result.content)} chars")
            return result.content
            
        # Return result as-is if it's not a Command
        return result
    
    # Replace the args_schema
    tool.args_schema = filtered_schema
    
    # Replace the coroutine with our wrapper (this is the async function)
    print(f"🔧 Checking coroutine attribute...")
    print(f"🔧 hasattr(tool, 'coroutine'): {hasattr(tool, 'coroutine')}")
    print(f"🔧 tool.coroutine: {getattr(tool, 'coroutine', 'NOT_FOUND')}")
    
    if hasattr(tool, 'coroutine'):
        print(f"🔧 Setting coroutine from {getattr(tool, 'coroutine', None)} to {wrapper_func}")
        try:
            setattr(tool, 'coroutine', wrapper_func)
            print(f"🔧 After setting: {getattr(tool, 'coroutine', None)}")
        except Exception as e:
            print(f"❌ Failed to set coroutine: {e}")
            # Try alternative approach - maybe coroutine is readonly
            print("🔧 Attempting to use func instead...")
            if hasattr(tool, 'func'):
                print(f"🔧 Setting func from {getattr(tool, 'func', 'NOT_FOUND')} to {wrapper_func}")
                setattr(tool, 'func', wrapper_func)
    else:
        print("🔧 No coroutine attribute, trying func...")
        if hasattr(tool, 'func'):
            print(f"🔧 Setting func from {getattr(tool, 'func', 'NOT_FOUND')} to {wrapper_func}")
            setattr(tool, 'func', wrapper_func)
    
    # Store original function for debugging 
    if not hasattr(tool, '_original_func'):
        setattr(tool, '_original_func', original_func)
    
    # Mark tool as patched to prevent double-wrapping
    setattr(tool, '_schema_filter_patched', True)
    
    return tool