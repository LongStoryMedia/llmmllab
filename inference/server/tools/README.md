# Dynamic Tools System

This module provides a modular dynamic tool generation system for the LangChain integration. It allows LLMs to generate and execute custom tools at runtime.

## Module Structure

```bash
server/tools/
├── __init__.py              # Main exports and module initialization
├── security.py              # Tool security validation
├── dynamic_tool.py          # Base dynamic tool class
├── generator.py             # Tool code generation logic
├── caching.py               # Caching for generated tools
├── learning.py              # Tool improvement through feedback
├── composition.py           # Tool composition for complex workflows
├── validation.py            # Tool validation and testing
├── marketplace.py           # Sharing and discovery of tools
├── enhanced_generator.py    # Advanced tool generation techniques
├── production.py            # Production-ready tool system
├── workflow.py              # Agentic workflow implementation
├── integration.py           # Integration helpers for chat system
├── errors.py                # Centralized error handling system
└── legacy/                  # Legacy code for backward compatibility
    └── original_tools.py    # Original monolithic implementation
```

## Core Components

### Error Handling (errors.py)

The error handling system provides:

- Consistent error classes for different failure modes (`ToolError`, `ToolExecutionError`, etc.)
- Centralized logging with context information
- Standardized error handling with the `handle_error` utility
- Traceback capture for debugging

### Security (security.py)

The `ToolSecurityValidator` class ensures tools are safe by:

- Validating code against forbidden patterns
- Blocking dangerous imports and function calls
- Preventing code injection

### Dynamic Tool (dynamic_tool.py)

The `DynamicTool` class provides:

- A LangChain-compatible tool interface
- Sandboxed code execution environment
- Error handling and result formatting

### Generator (generator.py)

The `DynamicToolGenerator` class handles:

- Creating prompts for tool generation
- Parsing LLM responses into executable code
- Extracting metadata and function signatures

### Caching (caching.py)

The `ToolCache` class provides:

- Persistent storage of generated tools
- Tool retrieval by name or functionality
- Versioning and usage tracking

### Learning (learning.py)

The `ToolLearner` class enables:

- Learning from user feedback
- Refining tools based on usage patterns
- Evolving tool capabilities over time

### Composition (composition.py)

The `ToolComposer` class allows:

- Combining multiple tools into workflows
- Sequential or parallel execution
- Creating meta-tools from simpler components

### Validation (validation.py)

The `ToolValidator` class ensures:

- Tools meet quality standards
- Generated code passes tests
- Results match expected outputs

### Marketplace (marketplace.py)

The `ToolMarketplace` class provides:

- Sharing tools between users
- Rating and reviewing tools
- Discovering useful tools created by others

### Enhanced Generator (enhanced_generator.py)

The `EnhancedDynamicToolGenerator` class adds:

- More sophisticated code generation
- Better handling of complex requirements
- Context-aware tool creation

### Production (production.py)

The `ProductionDynamicToolSystem` class offers:

- Enterprise-grade tool management
- Monitoring and logging
- Access control and governance

### Workflow (workflow.py)

The `AgenticWorkflow` class implements:

- Dynamic tool generation based on user needs
- Integration with conversation context
- Agent-based tool execution

### Integration (integration.py)

Integration helpers that provide:

- Detection of when to use agentic workflows
- Parameter extraction from user messages
- Functions for chat completions with tool support

## Usage Example

```python
from server.tools import (
    DynamicToolGenerator, 
    AgenticWorkflow, 
    create_agentic_chat_completion
)

# Initialize the workflow
workflow = AgenticWorkflow(pipeline_factory, conversation_context)
await workflow.initialize(model_id)

# Process a request with dynamic tools
response = await workflow.process_request("Calculate the compound interest on $1000 over 5 years at 5% annual interest")

# Or use the simplified integration function
response = await create_agentic_chat_completion(
    pipeline_factory, 
    conversation_context, 
    "Calculate the compound interest on $1000 over 5 years at 5% annual interest",
    model_id
)
```

## Legacy Support

The original monolithic implementation is preserved in `legacy/original_tools.py` for backward compatibility. New code should use the modular implementation.

## Best Practices

1. Always initialize the agentic workflow before processing requests
2. Use the integration helpers to determine when dynamic tools are needed
3. Implement proper error handling for tool execution
4. Consider caching frequently used tools for better performance
5. Use the composition system for complex multi-step workflows
