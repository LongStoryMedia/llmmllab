# Refined Tool Collection Architecture

## Overview

The tool collection system has been refined to provide more intelligent and accurate tool selection through enhanced intent-based filtering. The system now uses sophisticated decision logic based on expanded IntentAnalysis properties to determine when dynamic tools are needed and which static tools should be included.

## Enhanced IntentAnalysis Schema

### New Properties

The IntentAnalysis model now includes additional properties for better tool decision making:

```yaml
requires_tools: boolean
  description: "Whether this intent specifically requires tool execution"
  default: false

requires_custom_tools: boolean  
  description: "Whether this intent requires custom/dynamic tool generation"
  default: false

tool_complexity_score: number (0.0-1.0)
  description: "Tool complexity score (0-1, higher = more complex tools needed)"

computational_requirements: ComputationalRequirement
  description: "Computational requirements assessment for the request"
```

### ComputationalRequirement Enum

New enum for granular computational assessment:

```yaml
- MINIMAL: Basic operations requiring minimal resources
- LOW: Simple operations with low computational overhead  
- MODERATE: Standard operations with moderate resource needs
- HIGH: Complex operations requiring significant resources
- INTENSIVE: Very complex operations requiring maximum resources
```

## Refined Tool Collection Logic

### Dynamic Tool Generation Decision

The `_should_generate_dynamic_tools` method now uses a sophisticated decision matrix:

1. **Explicit Custom Tool Requirement**: If `requires_custom_tools` is true, dynamic tools are generated
2. **High Complexity + Tool Requirement**: If request requires tools, has COMPLEX/SPECIALIZED complexity, and tool_complexity_score > 0.7
3. **Domain Specificity + Computational Requirements**: If domain_specificity > 0.8 and computational requirements are HIGH/INTENSIVE

### Static Tool Filtering

The `_collect_static_tools` method now implements intelligent filtering:

1. **Capability-Based Matching**: Tools are matched to required capabilities
   - Web search tools for `web_search` or `information_retrieval` capabilities
   - Memory tools for `conversation_memory` capabilities  
   - Processing tools for `data_processing`, `file_manipulation`, `text_processing`
   - API tools for `api_integration` capabilities
   - Math tools for `basic_math` capabilities

2. **Complexity-Based Inclusion**: Basic tools (web_search, memory_search, summarization) are included for MODERATE+ complexity

3. **Fallback for Simple Requests**: If no tools match but basic capabilities are needed, essential tools are included

## Updated Classification Prompt

The classifier agent prompt now provides detailed guidance for the new properties:

### Tool Assessment Guidelines

- **requires_tools**: Set to true if the request needs external tools/APIs
- **requires_custom_tools**: Set to true if existing tools won't suffice
- **tool_complexity_score**: Rate 0.0-1.0 based on tooling complexity
  - 0.0-0.3: Basic tools (search, simple calculations)
  - 0.4-0.6: Moderate tools (data processing, API calls)  
  - 0.7-1.0: Complex tools (custom integrations, specialized processing)

### Scoring Guidelines

- **domain_specificity**: 0.0-1.0 (0.0=general, 1.0=highly domain-specific)
- **reusability_potential**: 0.0-1.0 (0.0=one-time use, 1.0=highly reusable)
- **confidence**: 0.0-1.0 (confidence in analysis)

## Benefits

### Improved Accuracy

- More precise tool selection based on actual request requirements
- Reduced over-provisioning of unnecessary tools
- Better matching of tool capabilities to user needs

### Enhanced Performance

- Faster tool collection through targeted filtering
- Reduced computational overhead from unused tools
- More efficient workflow execution

### Better Maintainability

- Clean separation of decision logic
- Clear property-based decision making
- Easier to extend and modify filtering rules

## Usage Examples

### Example 1: Simple Information Request

```python
intent = IntentAnalysis(
    workflow_type=WorkflowType.GENERAL,
    complexity_level=ComplexityLevel.SIMPLE,
    required_capabilities=[RequiredCapability.INFORMATION_RETRIEVAL],
    computational_requirements=ComputationalRequirement.LOW,
    requires_tools=True,
    requires_custom_tools=False,
    tool_complexity_score=0.2
)
# Result: Basic search tools only, no dynamic tool generation
```

### Example 2: Complex Engineering Task

```python  
intent = IntentAnalysis(
    workflow_type=WorkflowType.ENGINEERING,
    complexity_level=ComplexityLevel.COMPLEX,
    required_capabilities=[RequiredCapability.API_INTEGRATION, RequiredCapability.DATA_PROCESSING],
    computational_requirements=ComputationalRequirement.HIGH,
    domain_specificity=0.9,
    requires_tools=True,
    requires_custom_tools=True,
    tool_complexity_score=0.8
)
# Result: Specialized static tools + dynamic tool generation
```

## Testing

Comprehensive tests validate the refined logic:

- **Model Property Tests**: Verify new IntentAnalysis properties work correctly
- **Decision Logic Tests**: Validate dynamic tool generation decisions
- **Static Tool Filtering Tests**: Ensure proper capability-based tool matching  
- **Integration Tests**: Test end-to-end tool collection workflow

## Migration Impact

This refinement maintains backward compatibility while adding enhanced capabilities:

- Existing IntentAnalysis instances continue to work (new properties have defaults)
- Previous tool collection behavior is preserved for simple cases
- Enhanced accuracy for complex requests requiring sophisticated tool selection

The refined system provides more intelligent, accurate, and performant tool collection while maintaining the simplicity and maintainability of the unified architecture.