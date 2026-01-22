# Intent Analysis and Capability Architecture

## Schema Relationships

The LLM ML Lab uses a hierarchical capability-driven architecture for dynamic workflow construction:

### 1. Intent Analysis Pipeline

```text
User Request → IntentAnalysis → RequiredCapabilities → ModelProfileType → ModelTask
```

### 2. Schema Hierarchy

#### IntentAnalysis (Primary Analysis)

- **Purpose**: Comprehensive analysis of user intent with structured scoring
- **Key Fields**:
  - `workflow_type`: Enum of intent categories (chat, research, creative, etc.)
  - `required_capabilities`: List of `RequiredCapability` enums needed
  - `computational_requirements`: Structured list of compute needs
  - `complexity_level`: TRIVIAL → SIMPLE → MODERATE → COMPLEX → SPECIALIZED

#### RequiredCapability (Functional Requirements)

- **Purpose**: What functional capabilities the system needs to provide
- **Examples**: `web_search`, `text_processing`, `reasoning`, `image_processing`
- **Usage**: Drives selection of appropriate model profiles

#### ModelProfileType (Behavioral Profiles)

- **Purpose**: How the model should behave for specific tasks
- **Examples**: `Primary`, `Analysis`, `Summarization`, `Research`
- **Usage**: Selected based on required capabilities and complexity

#### ModelTask (I/O Modalities)

- **Purpose**: Input/output type constraints for model selection
- **Examples**: `TextToText`, `ImageToText`, `TextToImage`
- **Usage**: Final filter for model compatibility

### 3. Capability → Profile Selection Logic

Each `RequiredCapability` maps to compatible `ModelProfileType` instances:

```yaml
# Example mapping
web_search:
  compatible_profiles: [Research, Analysis, Primary]
  preferred_profile: Research
  minimum_complexity: SIMPLE
  
text_processing:
  compatible_profiles: [Primary, Analysis, Summarization]  
  preferred_profile: Primary
  minimum_complexity: TRIVIAL

reasoning:
  compatible_profiles: [Analysis, Primary, Research]
  preferred_profile: Analysis
  minimum_complexity: MODERATE
```

### 4. Dynamic Profile Selection Algorithm

1. **Analyze Intent**: Extract `RequiredCapability` list from user request
2. **Map Capabilities**: For each capability, get compatible `ModelProfileType` options
3. **Apply Complexity Filter**: Remove profiles below minimum complexity threshold
4. **Select Optimal Profile**: Choose based on:
   - Capability coverage (can handle all required capabilities)
   - Complexity match (appropriate for assessed complexity level)  
   - Resource availability (computational requirements)
   - User preferences (from user config)

### 5. Schema Design Principles

#### Shared Schema Pattern

To avoid duplication and ensure consistency, shared enums and structures are extracted to separate schemas:

- **ComputationalRequirement**: Referenced by both `IntentAnalysis` and `CapabilityProfileMapping`
- **RequiredCapability**: Used across multiple workflow analysis components
- **ComplexityLevel**: Shared complexity assessment scale

#### Design Rules

- **Avoid Duplication**: Extract shared enums/structures to separate schema files
- **Use $ref**: Reference shared schemas instead of copying definitions
- **Single Source of Truth**: Each data structure defined exactly once
- **Consistent Naming**: Use consistent enum values across related schemas

### 6. Deprecated Schemas

#### Intent (Simple Boolean Classification)

- **Status**: DEPRECATED - should be removed
- **Reason**: Replaced by comprehensive `IntentAnalysis`
- **Migration**: All usage should migrate to `IntentAnalysis`

## Implementation Notes

### Smart Intent Analyzer

The `SmartIntentAnalyzer` class implements the capability analysis logic:

- Maps user text to `RequiredCapability` enums
- Assesses complexity and computational needs
- Provides structured `IntentAnalysis` output

### Dynamic Tool Generation

The capability system drives dynamic tool generation decisions:

- Low complexity + single capability → Use existing system
- High complexity + multiple capabilities → Generate custom tool
- Specialized domain + novel requirements → Create new tool

### Profile Selection Service

The composer service uses capability mappings to:

- Select appropriate model profiles for each workflow node
- Ensure computational requirements are met
- Optimize for user preferences and system resources

## Example Flow

```python
# User: "Research the latest developments in quantum computing and create a comprehensive analysis"

# 1. Intent Analysis
intent_analysis = IntentAnalysis(
    workflow_type="research",
    required_capabilities=[RequiredCapability.WEB_SEARCH, RequiredCapability.REASONING, RequiredCapability.SUMMARIZATION],
    computational_requirements=["complex_reasoning", "large_data_handling"],
    complexity_level=ComplexityLevel.COMPLEX,
    confidence=0.9
)

# 2. Capability Mapping
web_search → ResearchProfile
reasoning → AnalysisProfile  
summarization → SummarizationProfile

# 3. Workflow Construction
workflow_nodes = [
    ResearchNode(profile=ResearchProfile),
    AnalysisNode(profile=AnalysisProfile),
    SummarizationNode(profile=SummarizationProfile)
]
```

This architecture ensures type safety, clear separation of concerns, and extensible capability-driven workflow construction.