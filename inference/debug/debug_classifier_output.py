"""
Debug the actual model output for intent analysis to see what it's returning.
"""
import asyncio
from runner.pipeline_factory import pipeline_factory
from composer.agents.classifier_agent import ClassifierAgent
from models import Message, ModelProfile, NodeMetadata, ModelParameters
from models import MessageContent, MessageContentType, MessageRole
from datetime import datetime, timezone
import logging
import uuid
import json

logger = logging.getLogger("debug_intent_output")
logging.basicConfig(level=logging.INFO)

async def debug_classifier_output():
    """Debug what the classifier model is actually outputting."""
    try:
        # Use the global pipeline factory instance
        factory = pipeline_factory
        
        # Create classifier agent with minimal profile
        profile = ModelProfile(
            id=str(uuid.uuid4()),
            user_id="test-user",
            name="analysis_profile", 
            model_name="qwen3-30b-a3b-q4-k-m",
            parameters=ModelParameters(temperature=0.3, num_predict=1024),
            system_prompt="You are an intent analysis system.",
            type=1,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )
        
        node_metadata = NodeMetadata(
            node_id="test_classifier",
            node_name="test_classifier",
            node_type="classifier",
            execution_order=1
        )
        
        classifier = ClassifierAgent(factory, profile, node_metadata)
        
        # Test engineering-focused query
        engineering_query = "I need engineering guidance to design a scalable microservices architecture for a high-traffic e-commerce platform. Please provide technical analysis and system architecture recommendations."
        
        messages = [
            Message(
                role=MessageRole.USER,
                content=[MessageContent(type=MessageContentType.TEXT, text=engineering_query)],
                message_type="text"
            )
        ]
        
        # Call the agent's run method directly to capture raw output
        logger.info("Getting raw model output...")
        
        # Build the prompt manually to see what it looks like
        from models import IntentAnalysis
        from pydantic import BaseModel
        from typing import List
        
        class _Intnts(BaseModel):
            intents: List[IntentAnalysis]
            
        intnt_schema = _Intnts.model_json_schema()
        user_query = messages[-1].content[0].text
        
        analysis_prompt = f"""
You are an expert intent classification system. Analyze the user request and output ONLY JSON.

Valid enumerations ONLY:
workflow_type (choose one per intent): [ {" | ".join(intnt_schema['$defs']['WorkflowType']['enum'])} ]
complexity_level (choose one per intent): [ {" | ".join(intnt_schema['$defs']['ComplexityLevel']['enum'])} ]
computational_requirements (choose one per intent): [ {" | ".join(intnt_schema['$defs']['ComputationalRequirement']['enum'])} ]
technical_domain (set for ENGINEERING workflows): [ {" | ".join(intnt_schema['$defs']['TechnicalDomain']['enum'])} ]
response_format (set for ENGINEERING workflows): [ {" | ".join(intnt_schema['$defs']['ResponseFormat']['enum'])} ]

required_capabilities (functionality needed - choose many, one, or none):
{", ".join(intnt_schema['$defs']['RequiredCapability']['enum'])}
required_capabilities can be empty if none apply. It is usually empty for simple queries.
DO NOT invent capabilities or requirements - only use those listed above.


Tool Assessment Guidelines:
- requires_tools: Set to true if the request needs external tools/APIs to be fulfilled (web search, file operations, calculations, etc.)
- requires_custom_tools: Set to true if existing tools won't suffice and custom tool creation is needed
- tool_complexity_score: Rate 0.0-1.0 based on how complex the required tooling would be
  * 0.0-0.3: Basic tools (search, simple calculations)  
  * 0.4-0.6: Moderate tools (data processing, API calls)
  * 0.7-1.0: Complex tools (custom integrations, specialized processing)

Scoring Guidelines:
- domain_specificity: 0.0-1.0 (0.0=general, 1.0=highly domain-specific)
- reusability_potential: 0.0-1.0 (0.0=one-time use, 1.0=highly reusable)
- confidence: 0.0-1.0 (confidence in your analysis)

Instructions:
1. Decompose only if there are clearly separable sub-tasks; else one intent in the intents array.
2. Each element in intents must follow the enumerations exactly.
3. For workflow_type=ENGINEERING, always set technical_domain and response_format to appropriate values.
4. For other workflow types, omit response_format / technical_domain unless clearly implied.
5. All boolean fields (requires_tools, requires_custom_tools) must be explicitly set.
6. All required numeric fields must be provided as numbers (not strings).
7. Output strictly valid JSON. No prose, no markdown, no comments.

User Request: {user_query}

IMPORTANT: Return JSON that is valid against this schema:
{json.dumps(intnt_schema)}

If multiple intents are needed, include additional objects in the intents array.
"""
        
        logger.info("Prompt being sent to model:")
        logger.info("="*80)
        logger.info(analysis_prompt)
        logger.info("="*80)
        
        # Get the raw response using BaseAgent run method
        result = await classifier.run(
            messages=[analysis_prompt],
            tools=None,
            priority=None,
            grammar=_Intnts,
        )
        
        # Extract and log raw response
        from utils.message import extract_message_text
        raw_response = extract_message_text(result.message) if result and result.message else ""
        
        logger.info("Raw model response:")
        logger.info("="*80)
        logger.info(raw_response)
        logger.info("="*80)
        
        # Now try to parse it
        from utils.grammar_generator import parse_structured_output
        try:
            intents = parse_structured_output(raw_response, _Intnts)
            logger.info(f"Parsed intents: {intents}")
            
            for i, intent in enumerate(intents.intents):
                logger.info(f"Intent {i+1}:")
                logger.info(f"  workflow_type: {intent.workflow_type}")
                logger.info(f"  technical_domain: {intent.technical_domain}")
                logger.info(f"  response_format: {intent.response_format}")
                logger.info(f"  complexity_level: {intent.complexity_level}")
        except Exception as e:
            logger.error(f"Failed to parse response: {e}")
        
        return raw_response
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None

if __name__ == "__main__":
    asyncio.run(debug_classifier_output())