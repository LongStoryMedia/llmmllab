"""
Dynamic Tool Generation System for LangChain Integration
Allows LLMs to generate and execute custom tools at runtime
"""

import logging
from typing import Optional

# Export all components needed for integration
from .security import ToolSecurityValidator
from .dynamic_tool import DynamicToolRunner
from .generator import DynamicToolGenerator
from .caching import ToolCache
from .learning import ToolLearner
from .composition import ToolComposer
from .validation import ToolValidator
from .marketplace import ToolMarketplace
from .enhanced_generator import EnhancedDynamicToolGenerator
from .production import ProductionDynamicToolSystem
from .workflow import AgenticWorkflow

# Configure logging
logger = logging.getLogger(__name__)

# For backward compatibility
__all__ = [
    "ToolSecurityValidator",
    "DynamicToolRunner",
    "DynamicToolGenerator",
    "ToolCache",
    "ToolLearner",
    "ToolComposer",
    "ToolValidator",
    "ToolMarketplace",
    "EnhancedDynamicToolGenerator",
    "ProductionDynamicToolSystem",
    "AgenticWorkflow",
]

# Legacy code has been moved to inference/server/tools/legacy/original_tools.py
# Import from there if needed for backward compatibility


# Example usage in your chat_new.py

# To integrate this into your existing chat system, you could modify the
# chat_completion function in chat_new.py like this:

# # In chat_completion function, after determining the model_id:

# # Check if the request would benefit from agentic processing
# user_text = ""
# for content in user_message.content:
#     if content.type == MessageContentType.TEXT and content.text:
#         user_text += content.text + " "

# # Use agentic workflow for complex requests
# if should_use_agentic_workflow(user_text):
#     response_content = await create_agentic_chat_completion(
#         pipeline_factory,
#         conversation_ctx,
#         user_text.strip(),
#         model_id
#     )
# else:
#     # Use existing pipeline workflow
#     response_content = await pipeline.generate(enhanced_messages, options)
