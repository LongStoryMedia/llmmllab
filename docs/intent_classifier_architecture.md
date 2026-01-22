# ClassifierAgent Architecture

## Overview

The ClassifierAgent is an LLM-driven graph node that uses the "analysis" model profile for sophisticated intent classification. It provides comprehensive intent analysis as part of the workflow orchestration system.

## Current Architecture

### LLM-Driven Classification
The ClassifierAgent uses the analysis model profile to perform comprehensive intent analysis via LLM. This provides sophisticated and context-aware classification that understands nuanced user requests beyond simple keyword matching.

### Graph Node Architecture
The agent functions as a proper graph node that uses the `current_user_message` field with assertion validation. It is designed for seamless integration into LangGraph workflow orchestration.

### Analysis Model Profile Integration
The agent accesses `conversation_ctx.user_config.model_profiles.analysis_profile_id` through the pipeline factory infrastructure. This leverages actual LLM capabilities for classification decisions rather than rule-based approaches.

### Structured LLM Prompting
The agent uses comprehensive JSON-structured prompts covering all IntentAnalysis fields. This ensures consistent, structured output with proper enum validation.

## Implementation Details

### Core Methods Transformed

1. **`analyze()` method**: 
   - Added current_user_message and user_config assertions
   - Integrated analysis model profile retrieval
   - Implemented LLM-based classification pipeline
   - Added statistical augmentation of results

2. **`_llm_analyze_intent()`**: NEW
   - Constructs comprehensive analysis prompt
   - Executes LLM pipeline with structured JSON request
   - Returns structured intent analysis

3. **`_parse_llm_response()`**: NEW
   - Parses JSON response from LLM
   - Converts strings to proper enum objects
   - Handles parsing errors with fallback

4. **`_augment_with_statistics()`**: NEW
   - Supplements LLM analysis with statistical insights
   - Adjusts confidence based on query characteristics
   - Maintains hybrid LLM+statistics approach

5. **`_fallback_heuristic_analysis()`**: NEW
   - Provides error recovery when LLM parsing fails
   - Maintains system reliability
   - Uses simplified heuristic classification



## LLM Prompt Engineering

### Structured Analysis Request
The LLM receives a comprehensive prompt requesting:
- Primary intent classification (chat, research, creative, technical, etc.)
- Complexity assessment (TRIVIAL → SPECIALIZED)  
- Required capability identification
- Computational requirement analysis
- Domain specificity scoring
- Reusability potential assessment
- Confidence estimation

### JSON Response Format
```json
{
    "workflow_type": "<intent_type>",
    "complexity_level": "<complexity_level>",
    "required_capabilities": ["<capability1>", "<capability2>"],
    "computational_requirements": ["<requirement1>", "<requirement2>"],
    "domain_specificity": 0.0,
    "reusability_potential": 0.0,
    "confidence": 0.0,
    "reasoning": "Brief explanation"
}
```

## Error Handling & Reliability

### Multi-Layer Approach
1. **Primary**: LLM analysis with structured JSON parsing
2. **Secondary**: Statistical augmentation for confidence adjustment
3. **Fallback**: Heuristic analysis if LLM parsing fails
4. **Validation**: Enum validation and constraint checking

### Assertion Validation
- `current_user_message is not None` - ensures proper graph node input
- `user_config is not None` - ensures model profile access
- Proper error messages for debugging

## Integration Points

### Graph Node Ready
- Uses `current_user_message` field as expected by graph architecture
- Outputs structured `IntentAnalysis` object for downstream processing
- Proper assertions ensure reliable graph execution

### Model Profile Integration  
- Accesses analysis model profile via user configuration
- Uses pipeline factory with HIGH priority for intent classification
- Integrates with existing model management infrastructure

### RAG Depth Compatibility
- `determine_rag_depth()` method maps complexity to retrieval depth
- Supports "shallow", "moderate", "deep" RAG strategies
- Enables intelligent retrieval based on analysis complexity

## Performance Characteristics

### Efficiency Improvements
- **High Priority Pipeline**: Intent classification gets priority access to models
- **Structured Output**: Eliminates post-processing ambiguity
- **Cached Results**: Pipeline factory handles model caching automatically

### Quality Improvements
- **Context Awareness**: LLM understands nuanced intent beyond keywords
- **Domain Intelligence**: Sophisticated domain specificity assessment
- **Capability Mapping**: Intelligent mapping of requirements to capabilities

## Testing & Validation

### Architectural Validation
- ✅ All key architectural changes verified
- ✅ Proper import structure and dependencies  
- ✅ Assertion validation working
- ✅ Schema model compatibility confirmed
- ✅ Enum validation functional

### Integration Testing
Created comprehensive test suites:
- `test_intent_architecture.py` - Validates architectural changes
- `test_llm_classifier_agent.py` - Tests full integration (requires infrastructure)

## System Requirements

### Interface Compatibility
The agent maintains the same `analyze()` method signature and returns the same `IntentAnalysis` object structure. It uses the standard `ConversationCtx` input pattern for seamless integration.

### Configuration Requirements
- Requires `analysis_profile_id` in user model profiles
- Needs pipeline factory infrastructure
- Depends on storage service for model profile retrieval

### Performance Characteristics
- Uses HIGH priority pipeline for responsive intent classification
- Provides improved classification accuracy through LLM analysis
- Enhanced reliability with multi-layer fallback system

## Future Enhancements

### Potential Improvements
1. **Model Optimization**: Fine-tune analysis model for intent classification
2. **Caching Strategy**: Cache common intent patterns for faster response
3. **Adaptive Learning**: Learn from user feedback to improve classification
4. **Batch Processing**: Support multiple message analysis in single call

### Graph Integration
- Ready for LangGraph node registration
- Supports streaming output for real-time analysis  
- Compatible with conditional graph routing based on intent

## Summary

The ClassifierAgent is an LLM-driven graph node that provides AI-powered intent analysis. The agent:

1. **Leverages LLM Intelligence**: Uses analysis model profile for sophisticated classification
2. **Integrates with Graph Architecture**: Proper current_user_message usage and assertions
3. **Provides Structured Output**: JSON-based LLM prompting with enum validation
4. **Maintains Reliability**: Multi-layer fallback and error handling
5. **Enables Advanced Workflows**: RAG depth determination and capability mapping

This architecture enables sophisticated workflow orchestration based on intelligent intent understanding.

---

*Documentation updated: September 29, 2025*  
*Architecture: LLM-driven intent analysis graph node*