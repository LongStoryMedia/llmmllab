"""
Integration helpers for using the dynamic tool system with chat completions
"""
import logging
import re
from typing import List, Optional

from .workflow import AgenticWorkflow
from .production import ProductionDynamicToolSystem

logger = logging.getLogger(__name__)


def should_use_agentic_workflow(user_message: str) -> bool:
    """
    Determine if a user message would benefit from agentic processing with tools
    
    Args:
        user_message: The user's message text
        
    Returns:
        bool: True if agentic workflow should be used
    """
    # Keywords that suggest need for tools/computation
    tool_indicators = [
        # Calculation keywords
        'calculate', 'compute', 'add', 'subtract', 'multiply', 'divide',
        'sum', 'average', 'mean', 'median', 'standard deviation',
        'percentage', 'percent', 'ratio', 'proportion',
        
        # Data processing keywords
        'analyze', 'process', 'transform', 'convert', 'parse',
        'filter', 'sort', 'group', 'aggregate', 'summarize',
        
        # Programming/algorithm keywords
        'algorithm', 'function', 'code', 'script', 'program',
        'logic', 'formula', 'equation', 'solve',
        
        # Complex task indicators
        'step by step', 'break down', 'systematic', 'methodical',
        'optimize', 'find the best', 'compare options'
    ]
    
    # Check for mathematical expressions
    math_patterns = [
        r'\d+\s*[+\-*/]\s*\d+',  # Basic math operations
        r'\d+\s*%',               # Percentages
        r'\$\d+',                 # Currency
        r'\d+\.\d+',              # Decimals
    ]
    
    message_lower = user_message.lower()
    
    # Check for tool indicator keywords
    for indicator in tool_indicators:
        if indicator in message_lower:
            return True
    
    # Check for mathematical patterns
    for pattern in math_patterns:
        if re.search(pattern, user_message):
            return True
    
    # Check for question words that might need computation
    computation_questions = [
        'how many', 'how much', 'what is the', 'calculate the',
        'find the', 'determine the', 'compute the'
    ]
    
    for question in computation_questions:
        if question in message_lower:
            return True
    
    return False


def extract_parameters_from_message(message: str) -> dict:
    """
    Extract parameters from a user message for tool execution
    
    Args:
        message: User message text
        
    Returns:
        dict: Extracted parameters
    """
    # This is a simple implementation - a more robust version might use
    # the LLM to extract structured parameters from natural language
    
    params = {}
    
    # Look for numbers
    number_pattern = r'(\d+(\.\d+)?)'
    numbers = re.findall(number_pattern, message)
    if numbers:
        for i, (num, _) in enumerate(numbers[:2]):  # Limit to first two numbers
            if '.' in num:
                params[f'number_{i+1}'] = float(num)
            else:
                params[f'number_{i+1}'] = int(num)
    
    # Look for operation type
    if 'add' in message.lower() or '+' in message or 'sum' in message.lower() or 'plus' in message.lower():
        params['operation'] = 'add'
    elif 'subtract' in message.lower() or '-' in message or 'minus' in message.lower() or 'difference' in message.lower():
        params['operation'] = 'subtract'
    elif 'multiply' in message.lower() or '*' in message or 'times' in message.lower() or 'product' in message.lower():
        params['operation'] = 'multiply'
    elif 'divide' in message.lower() or '/' in message:
        params['operation'] = 'divide'
    
    return params


async def create_agentic_chat_completion(
    pipeline_factory, conversation_ctx, user_message: str, model_id: str
) -> str:
    """
    Create an agentic chat completion with dynamic tool generation
    
    Args:
        pipeline_factory: Factory for creating pipelines
        conversation_ctx: Conversation context
        user_message: The user's message
        model_id: ID of the model to use
        
    Returns:
        str: The response from the agentic workflow
    """
    try:
        # Create and initialize the agentic workflow
        workflow = AgenticWorkflow(pipeline_factory, conversation_ctx)
        await workflow.initialize(model_id)

        # Process the request
        response = await workflow.process_request(user_message)

        return response

    except Exception as e:
        logger.error(f"Error in agentic chat completion: {e}")
        return f"I apologize, but I encountered an error: {str(e)}"


async def create_production_agentic_completion(
    pipeline_factory, conversation_ctx, user_message: str, model_id: str, user_id: Optional[str] = None
) -> str:
    """
    Create an agentic completion using the production system with all advanced features
    
    Args:
        pipeline_factory: Factory for creating pipelines
        conversation_ctx: Conversation context
        user_message: The user's message
        model_id: ID of the model to use
        user_id: Optional user ID for marketplace integration
        
    Returns:
        str: The response from the production system
    """
    try:
        # Get the primary pipeline
        primary_pipeline, _ = pipeline_factory.get_pipeline(model_id)
        
        # Initialize the production system with marketplace enabled if user_id provided
        tool_system = ProductionDynamicToolSystem(
            llm=primary_pipeline,
            enable_marketplace=user_id is not None
        )
        
        # Create tool for the task
        tool = await tool_system.create_and_validate_tool(user_message, user_id)
        
        if not tool:
            # Fall back to standard agentic workflow if tool creation fails
            return await create_agentic_chat_completion(
                pipeline_factory, conversation_ctx, user_message, model_id
            )
        
        # Extract parameters from message
        params = extract_parameters_from_message(user_message)
        
        # Execute the tool
        result = await tool_system.execute_tool_with_monitoring(tool, **params)
        
        # Format the response
        response = (
            f"I created a tool to help with your request: '{tool.name}'\n\n"
            f"Here's the result: {result}"
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Error in production agentic completion: {e}")
        return f"I apologize, but I encountered an error: {str(e)}"
