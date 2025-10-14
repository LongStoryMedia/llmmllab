# Structured Output Requirements - Grammar-Based System

## Overview

This document outlines the requirements for implementing a comprehensive grammar-based structured output system across the LLM ML Lab platform. The system will use YAML schemas to define output structure, generate Pydantic models, and leverage llamacpp's grammar features to ensure reliable, structured responses from language models.

## Architecture Overview

```plaintext
YAML Schema Definition
    ↓
Pydantic Model Generation (via schema2code)
    ↓
Grammar Generation (Pydantic → JSON Schema → llamacpp grammar)
    ↓
LLM Pipeline Integration
    ↓
Structured Output Validation & Parsing
```

## Core Concept: Grammar-Driven Output

### Grammar System Benefits

1. **Guaranteed Structure**: LLM output is constrained to valid JSON matching the schema
2. **Type Safety**: Automatic validation and parsing into Pydantic models
3. **Performance**: Faster parsing and reduced error handling
4. **Consistency**: Unified approach across all structured output needs
5. **Schema-Driven**: Single source of truth for output structure

### Grammar Generation Flow

```python
# 1. YAML Schema defines structure
# schemas/intent_analysis.yaml

# 2. Pydantic model generated
# models/intent_analysis.py (via schema2code)

# 3. Grammar generated from Pydantic model
grammar = generate_grammar_from_pydantic(IntentAnalysis)

# 4. Grammar used in llamacpp pipeline
pipeline = create_grammar_constrained_pipeline(model_profile, grammar)

# 5. Structured output guaranteed
result: IntentAnalysis = await run_structured_pipeline(pipeline, prompt)
```

## System Components

### 1. Grammar Generation Utility

Create `utils/grammar_generator.py`:

```python
"""
Grammar generation utility for structured output using llamacpp grammars.
"""
from typing import Type, Dict, Any, Optional
from pydantic import BaseModel
import json
from runner.pipelines.grammar import GrammarConstrainedPipeline

class GrammarGenerator:
    """Generate llamacpp grammars from Pydantic models."""
    
    @staticmethod
    def from_pydantic_model(model_class: Type[BaseModel]) -> str:
        """
        Generate llamacpp grammar from Pydantic model.
        
        Args:
            model_class: Pydantic model class to generate grammar for
            
        Returns:
            Grammar string compatible with llamacpp
        """
        # Get JSON schema from Pydantic model
        schema = model_class.model_json_schema()
        
        # Convert JSON schema to llamacpp grammar format
        return GrammarGenerator._json_schema_to_grammar(schema)
    
    @staticmethod
    def _json_schema_to_grammar(schema: Dict[str, Any]) -> str:
        """Convert JSON schema to llamacpp grammar format."""
        # Implementation based on llamacpp grammar syntax
        # This would follow the grammar rules from:
        # https://github.com/ggml-org/llama.cpp/blob/master/grammars/README.md
        
        grammar_rules = []
        
        # Root rule
        grammar_rules.append("root ::= object")
        
        # Generate object rules from schema properties
        if "properties" in schema:
            object_rule = "object ::= \"{\" ws "
            property_rules = []
            
            for prop_name, prop_schema in schema["properties"].items():
                property_rules.append(f'"\"{prop_name}\"" ws ":" ws {GrammarGenerator._get_type_rule(prop_schema)}')
            
            object_rule += " \",\" ws ".join(property_rules)
            object_rule += " ws \"}\""
            grammar_rules.append(object_rule)
        
        # Add common rules
        grammar_rules.extend([
            "ws ::= [ \\t\\n]*",
            "string ::= \"\\\"\" ([^\\\\\\\"] | \"\\\\\" ([\"\\\\/bfnrt] | \"u\" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F]))* \"\\\"\"",
            "number ::= \"-\"? ([0-9] | [1-9] [0-9]*) (\".\" [0-9]+)? ([eE] [-+]? [0-9]+)?",
            "boolean ::= \"true\" | \"false\"",
            "null ::= \"null\"",
        ])
        
        return "\n".join(grammar_rules)
    
    @staticmethod
    def _get_type_rule(prop_schema: Dict[str, Any]) -> str:
        """Get grammar rule for a property type."""
        prop_type = prop_schema.get("type", "string")
        
        if prop_type == "string":
            if "enum" in prop_schema:
                enum_values = " | ".join(f'"{value}"' for value in prop_schema["enum"])
                return f"({enum_values})"
            return "string"
        elif prop_type == "number" or prop_type == "integer":
            return "number"
        elif prop_type == "boolean":
            return "boolean"
        elif prop_type == "array":
            item_rule = GrammarGenerator._get_type_rule(prop_schema.get("items", {"type": "string"}))
            return f"\"[\" ws ({item_rule} (ws \",\" ws {item_rule})*)? ws \"]\""
        elif prop_type == "object":
            return "object"
        else:
            return "string"  # fallback

# Grammar-constrained pipeline integration
class StructuredOutputPipeline:
    """Pipeline wrapper for grammar-constrained structured output."""
    
    def __init__(self, model_class: Type[BaseModel]):
        self.model_class = model_class
        self.grammar = GrammarGenerator.from_pydantic_model(model_class)
    
    async def run_with_grammar(self, pipeline, prompt: str) -> BaseModel:
        """
        Run pipeline with grammar constraints and parse result.
        
        Args:
            pipeline: LLM pipeline to use
            prompt: Prompt to send to LLM
            
        Returns:
            Parsed Pydantic model instance
        """
        # Set grammar constraint on pipeline
        pipeline.set_grammar_constraint(self.grammar)
        
        # Run pipeline
        result = await pipeline.run(prompt)
        
        # Parse result into Pydantic model
        try:
            parsed_data = json.loads(result.text)
            return self.model_class.model_validate(parsed_data)
        except (json.JSONDecodeError, ValueError) as e:
            raise StructuredOutputError(f"Failed to parse structured output: {e}")

class StructuredOutputError(Exception):
    """Exception raised when structured output parsing fails."""
    pass
```

### 2. Pipeline Integration

Update `runner/pipelines/` to support grammar constraints:

```python
# runner/pipelines/grammar.py
"""
Grammar-constrained pipeline implementation.
"""
from typing import Optional, Type
from pydantic import BaseModel
from .base import BasePipeline
from utils.grammar_generator import GrammarGenerator, StructuredOutputPipeline

class GrammarConstrainedPipeline(BasePipeline):
    """Pipeline with grammar constraints for structured output."""
    
    def __init__(self, model_profile, expected_model: Type[BaseModel]):
        super().__init__(model_profile)
        self.expected_model = expected_model
        self.structured_pipeline = StructuredOutputPipeline(expected_model)
        self.grammar = self.structured_pipeline.grammar
    
    async def run_structured(self, prompt: str) -> BaseModel:
        """Run with grammar constraints and return parsed model."""
        return await self.structured_pipeline.run_with_grammar(self, prompt)
    
    def set_grammar_constraint(self, grammar: str):
        """Set grammar constraint for llamacpp."""
        # This would integrate with the actual llamacpp binding
        # to set the grammar parameter
        self._grammar_constraint = grammar
```

### 3. Agent Integration Pattern

Update agent base class for structured output:

```python
# composer/agents/analysis/base_analysis_agent.py (updated)
"""
Base analysis agent with structured output support.
"""
from abc import ABC, abstractmethod
from typing import Any, TypeVar, Generic, Type
from pydantic import BaseModel
from models.conversation_ctx import ConversationCtx
from runner.pipelines.grammar import GrammarConstrainedPipeline
from utils.model_profile import get_model_profile_for_task

T = TypeVar('T', bound=BaseModel)

class BaseAnalysisAgent(ABC, Generic[T]):
    """Base class for analysis agents with structured output."""
    
    def __init__(self, model_profile_type: ModelProfileType, output_model: Type[T]):
        self.model_profile_type = model_profile_type
        self.output_model = output_model
    
    async def get_structured_pipeline(self, conversation_ctx: ConversationCtx) -> GrammarConstrainedPipeline:
        """Get grammar-constrained pipeline for structured output."""
        mp = await get_model_profile_for_task(
            config=conversation_ctx.user_config.model_profiles,
            task=self.model_profile_type,
            user_id=conversation_ctx.user_config.user_id
        )
        
        return GrammarConstrainedPipeline(
            model_profile=mp,
            expected_model=self.output_model
        )
    
    async def analyze(self, conversation_ctx: ConversationCtx) -> T:
        """Perform structured analysis with guaranteed output format."""
        try:
            user_query = extract_message_text(conversation_ctx.current_user_message)
            
            pipeline = await self.get_structured_pipeline(conversation_ctx)
            prompt = self.create_analysis_prompt(user_query)
            
            # Grammar-constrained execution guarantees valid structure
            result = await pipeline.run_structured(prompt)
            return result
            
        except Exception as e:
            # Fallback to manual parsing if grammar constraints fail
            return await self._fallback_parsing(user_query, conversation_ctx, str(e))
    
    @abstractmethod
    def create_analysis_prompt(self, user_query: str) -> str:
        """Create prompt for structured analysis."""
        pass
    
    @abstractmethod
    async def _fallback_parsing(self, user_query: str, conversation_ctx: ConversationCtx, error: str) -> T:
        """Fallback parsing when structured output fails."""
        pass
```

## Current System Analysis

### Locations Requiring Structured Output

#### 1. Intent Analysis Agent

**File**: `composer/agents/analysis/intent_analysis_agent.py`
**Current State**: Manual JSON parsing with fallbacks
**Required Schema**: `schemas/intent_analysis.yaml` (already exists)
**Updates Needed**:

- Convert to grammar-constrained pipeline
- Remove manual JSON parsing logic
- Add structured prompt generation

#### 2. Dynamic Tool Generation

**Files**: `composer/tools/dynamic/tool_generator.py`
**Current State**: Free-form LLM output with parsing
**Required Schema**: Create `schemas/dynamic_tool_spec.yaml`
**Updates Needed**:

- Define tool specification schema
- Implement grammar-constrained generation
- Ensure valid Python code generation

#### 3. Research Task Planning

**Files**: Research task agent (from deep research requirements)
**Current State**: JSON parsing with manual validation
**Required Schema**: `schemas/research_plan.yaml`
**Updates Needed**:

- Grammar-constrained research plan generation
- Guaranteed valid question structure
- Proper node type validation

#### 4. Memory Storage and Synthesis

**Files**: `composer/services/memory_service.py`
**Current State**: Basic memory storage
**Required Schema**: Use existing `schemas/memory.yaml`
**Updates Needed**:

- Grammar-constrained memory fragment creation
- Structured memory consolidation using existing schema
- Type-safe memory source classification

#### 5. Conversation Summarization

**Files**: Summary generation services
**Current State**: Free-form text output
**Required Schema**: Create `schemas/conversation_summary.yaml`
**Updates Needed**:

- Structured summaries with metadata
- Key topics extraction
- Action items identification

#### 6. Error Analysis & Reporting

**Files**: Error handling across services
**Current State**: String-based error messages
**Required Schema**: Create `schemas/error_analysis.yaml`
**Updates Needed**:

- Structured error categorization
- Solution suggestions
- Severity classification

## Implementation Plan

### Phase 1: Core Infrastructure

#### Step 1: Grammar Generation System

1. **Create Grammar Generator**: Implement `utils/grammar_generator.py`
2. **Pipeline Integration**: Update pipeline factory for grammar constraints
3. **Schema Validation**: Ensure all existing schemas work with grammar generation
4. **Testing Framework**: Create tests for grammar generation accuracy

#### Step 2: Base Agent Refactoring

1. **Update Base Agent**: Modify `BaseAnalysisAgent` for structured output
2. **Migration Utilities**: Create tools for converting existing agents
3. **Error Handling**: Implement robust fallback mechanisms
4. **Performance Testing**: Validate grammar constraint performance

### Phase 2: Agent Migration

#### Step 1: Intent Analysis Migration

```python
# composer/agents/analysis/intent_analysis_agent.py (updated)
class IntentAnalysisAgent(BaseAnalysisAgent[IntentAnalysis]):
    def __init__(self):
        super().__init__(
            ModelProfileType.Analysis,
            IntentAnalysis  # Pydantic model for structured output
        )
    
    def create_analysis_prompt(self, user_query: str) -> str:
        return f"""
Analyze the user request and provide a structured response.

User Request: {user_query}

Classify the intent, complexity, capabilities, and requirements.
Respond with a valid JSON object matching the IntentAnalysis schema.
All fields are required and must match the specified types and constraints.
"""
    
    async def _fallback_parsing(self, user_query: str, conversation_ctx: ConversationCtx, error: str) -> IntentAnalysis:
        # Heuristic analysis as fallback
        return IntentAnalysis(
            workflow_type="tool_use",  # Conservative fallback
            complexity_level=ComplexityLevel.MODERATE,
            required_capabilities=["general_knowledge"],
            computational_requirements=[],
            domain_specificity=0.5,
            reusability_potential=0.5,
            confidence=0.3,  # Low confidence for fallback
            reasoning=f"Fallback analysis due to parsing error: {error}"
        )
```

#### Step 2: Research Task Agent Migration

```python
# composer/agents/analysis/research_task_agent.py (updated)
class ResearchTaskAgent(BaseAnalysisAgent[ResearchPlan]):
    def __init__(self):
        super().__init__(
            ModelProfileType.ResearchTask,
            ResearchPlan  # New structured output model
        )
    
    def create_analysis_prompt(self, user_query: str) -> str:
        return f"""
Create a comprehensive research plan for the given query.

Research Query: {user_query}

Generate a structured research plan with:
1. Plan overview (approach and methodology)
2. Specific research questions (3-8 questions)
3. Execution strategy and timing

Each question must specify:
- Clear, specific question text
- Relevant search keywords
- Appropriate node type (web_search, memory_search, static_analysis, agent_analysis, dynamic_tool)
- Node-specific configuration
- Execution priority (1=highest)

Respond with valid JSON matching the ResearchPlan schema.
"""
```

### Phase 3: Dynamic Tool Generation

#### Step 1: Tool Specification Schema

Create structured output for dynamic tool generation:

```python
# composer/tools/dynamic/structured_tool_generator.py
class StructuredToolGenerator(BaseAnalysisAgent[DynamicToolSpec]):
    def __init__(self):
        super().__init__(
            ModelProfileType.ToolGeneration,
            DynamicToolSpec
        )
    
    def create_analysis_prompt(self, tool_description: str) -> str:
        return f"""
Generate a complete Python tool implementation for the following requirement:

Tool Requirement: {tool_description}

Create a structured tool specification with:
1. Valid Python function name (snake_case)
2. Clear description for LangChain integration
3. Complete parameter schema (JSON Schema format)
4. Full Python implementation with proper error handling
5. Required dependencies and imports
6. Input validation rules

The implementation must be complete, executable Python code.
Respond with valid JSON matching the DynamicToolSpec schema.
"""
    
    async def generate_structured_tool(self, tool_description: str, conversation_ctx: ConversationCtx) -> DynamicToolSpec:
        """Generate a structured tool specification."""
        pipeline = await self.get_structured_pipeline(conversation_ctx)
        prompt = self.create_analysis_prompt(tool_description)
        
        tool_spec = await pipeline.run_structured(prompt)
        
        # Validate generated code syntax
        self._validate_generated_code(tool_spec.implementation)
        
        return tool_spec
    
    def _validate_generated_code(self, code: str):
        """Validate that generated code is syntactically correct."""
        try:
            compile(code, '<generated>', 'exec')
        except SyntaxError as e:
            raise StructuredOutputError(f"Generated code has syntax errors: {e}")
```

### Phase 4: Memory & Summarization

#### Step 1: Memory Service Update

```python
# composer/services/structured_memory_service.py
from models.memory import Memory
from models.memory_fragment import MemoryFragment
from models.memory_source import MemorySource

class StructuredMemoryService(BaseAnalysisAgent[Memory]):
    def __init__(self):
        super().__init__(
            ModelProfileType.MemoryStorage,
            Memory
        )
    
    async def create_memory_from_conversation(
        self, 
        content: str, 
        source_type: MemorySource, 
        conversation_ctx: ConversationCtx
    ) -> Memory:
        """Create structured memory from conversation content."""
        prompt = self.create_memory_prompt(content, source_type)
        pipeline = await self.get_structured_pipeline(conversation_ctx)
        return await pipeline.run_structured(prompt)
    
    def create_memory_prompt(self, content: str, source_type: MemorySource) -> str:
        return f"""
Analyze the conversation content and create structured memory fragments.

Content:
{content}

Source Type: {source_type.value}

Create structured memory with:
1. Meaningful memory fragments with role and content
2. Appropriate timestamps and similarity scoring
3. Source identification and conversation linking

Respond with valid JSON matching the Memory schema.
"""
```

### Phase 5: Integration & Testing

#### Step 1: Service Integration

1. **Update Pipeline Factory**: Support grammar constraints across all pipelines
2. **Error Handling**: Robust fallbacks when grammar constraints fail
3. **Performance Monitoring**: Track grammar constraint impact on performance
4. **Configuration**: User-configurable grammar strictness levels

#### Step 2: Comprehensive Testing

1. **Unit Tests**: Test grammar generation for all schemas
2. **Integration Tests**: Validate end-to-end structured output flows
3. **Performance Tests**: Benchmark grammar-constrained vs free-form output
4. **Error Recovery Tests**: Validate fallback mechanisms

## Migration Strategy

### Backward Compatibility

During migration, maintain backward compatibility:

```python
class CompatibilityAgent:
    def __init__(self, use_structured_output: bool = True):
        self.use_structured_output = use_structured_output
    
    async def analyze(self, conversation_ctx: ConversationCtx):
        if self.use_structured_output:
            # New grammar-constrained approach
            return await self._structured_analysis(conversation_ctx)
        else:
            # Legacy parsing approach
            return await self._legacy_analysis(conversation_ctx)
```

### Feature Flags

Implement feature flags for gradual rollout:

```yaml
# User configuration for structured output
structured_output:
  enabled: true
  fallback_on_error: true
  strictness_level: "standard"  # relaxed, standard, strict
  timeout_ms: 5000
```

## Expected Outcomes

### 1. Reliability Improvements

- **Guaranteed Structure**: Eliminate JSON parsing errors
- **Type Safety**: Automatic validation and type checking
- **Consistent Format**: Uniform output structure across all agents

### 2. Performance Benefits

- **Faster Parsing**: Direct Pydantic model instantiation
- **Reduced Tokens**: More efficient prompts with structure constraints
- **Better Caching**: Structured outputs enable better result caching

### 3. Developer Experience

- **Schema-Driven**: Single source of truth for output structure
- **Type Hints**: Full IDE support with Pydantic models
- **Easy Testing**: Structured outputs are easier to test and validate

### 4. Extensibility

- **New Schemas**: Easy to add new structured output types
- **Grammar Evolution**: Support for complex nested structures
- **Multi-Model Support**: Grammar constraints work across different model types

The structured output system will provide a robust foundation for reliable LLM interactions while maintaining the flexibility needed for complex AI workflows.
