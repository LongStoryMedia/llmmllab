"""
Demo service showing the refactored architecture with intelligent LangGraph routing.

This demonstrates how to eliminate redundant intent analysis and tool selection
by using LangGraph's native capabilities for intelligent routing.
"""

import asyncio
from typing import Dict, Any, Optional, List

from langgraph.graph import StateGraph, END
from langgraph.graph.state import CompiledStateGraph

from models import Message, LangChainMessage
from composer.graph.state import WorkflowState
from composer.nodes.intent_classifier import IntentClassifierNode
from composer.nodes.engineering_agent import EngineeringAgentNode
from composer.nodes.standard import PipelineNode
from composer.nodes.rag.router import ShallowRAGExecutor
from composer.monitoring.logging import composer_logger
from models.model_profile_type import ModelProfileType


class RefactoredComposerService:
    """
    Refactored composer service using LangGraph's native intelligent routing.
    
    Key improvements:
    1. Single intent analysis (not redundant)
    2. Dynamic tool selection within the graph
    3. Intelligent workflow routing based on analysis
    4. Eliminates pre-workflow decision making
    """

    def __init__(self):
        self.logger = composer_logger.logger
        
        # Import pipeline factory
        try:
            from runner import pipeline_factory
            self.pipeline_factory = pipeline_factory
        except ImportError as e:
            self.logger.warning(f"Could not import pipeline_factory: {e}")
            self.pipeline_factory = None

    async def compose_master_workflow(self, user_id: str) -> CompiledStateGraph:
        """
        Compose a single master workflow with intelligent routing.
        
        This workflow handles:
        1. Intent analysis (once)
        2. Tool selection (dynamic)
        3. Workflow routing (intelligent)
        4. Execution (appropriate subflow)
        """
        try:
            self.logger.info("Building master workflow", extra={"user_id": user_id})
            
            # Create master workflow graph
            workflow = StateGraph(WorkflowState)
            
            # Add core nodes
            workflow.add_node("intent_analysis", IntentClassifierNode())
            workflow.add_node("tool_selection", EngineeringAgentNode(self.pipeline_factory))
            workflow.add_node("execute_workflow", self._create_intelligent_executor(user_id))
            
            # Linear flow with intelligence embedded in nodes
            workflow.set_entry_point("intent_analysis")
            workflow.add_edge("intent_analysis", "tool_selection")
            workflow.add_edge("tool_selection", "execute_workflow")
            workflow.add_edge("execute_workflow", END)
            
            compiled_workflow = workflow.compile()
            
            self.logger.info(
                "Master workflow built successfully",
                extra={"user_id": user_id, "node_count": len(workflow.nodes)}
            )
            
            return compiled_workflow
            
        except Exception as e:
            self.logger.error(
                "Failed to build master workflow",
                extra={"user_id": user_id, "error": str(e)}
            )
            raise

    def _create_intelligent_executor(self, user_id: str):
        """Create executor that routes based on intent analysis."""
        
        async def intelligent_executor(state: WorkflowState) -> WorkflowState:
            """Route and execute based on intent classification."""
            try:
                # Get intent analysis from state (done once at start of graph)
                intent_analysis = getattr(state, "intent_classification", None)
                
                if not intent_analysis:
                    return await self._execute_default_flow(state, user_id)
                
                primary_intent = getattr(intent_analysis, "primary_intent", "").lower()
                
                # Intelligent routing based on intent
                if "research" in primary_intent or "analysis" in primary_intent:
                    self.logger.info("Routing to research flow", extra={"user_id": user_id})
                    return await self._execute_research_flow(state, user_id)
                elif "creative" in primary_intent or "generate" in primary_intent:
                    self.logger.info("Routing to creative flow", extra={"user_id": user_id})
                    return await self._execute_creative_flow(state, user_id)
                else:
                    self.logger.info("Routing to chat flow", extra={"user_id": user_id})
                    return await self._execute_chat_flow(state, user_id)
                    
            except Exception as e:
                self.logger.error(f"Intelligent executor failed: {e}")
                return await self._execute_default_flow(state, user_id)
        
        return intelligent_executor

    async def _execute_chat_flow(self, state: WorkflowState, user_id: str) -> WorkflowState:
        """Execute optimized chat flow."""
        # Lightweight RAG
        rag_executor = ShallowRAGExecutor(user_id)
        state = await rag_executor(state)
        
        # Chat response with streaming
        chat_agent = PipelineNode(
            self.pipeline_factory,
            ModelProfileType.Primary,
            stream=True
        )
        state = await chat_agent(state)
        
        return state

    async def _execute_research_flow(self, state: WorkflowState, user_id: str) -> WorkflowState:
        """Execute research-focused flow."""
        # Enhanced RAG for research
        from composer.nodes.rag.executor import EnhancedRAGExecutor
        enhanced_rag = EnhancedRAGExecutor(user_id)
        state = await enhanced_rag(state)
        
        # Research synthesis
        synthesizer = PipelineNode(
            self.pipeline_factory,
            ModelProfileType.Primary,
            stream=True
        )
        state = await synthesizer(state)
        
        return state

    async def _execute_creative_flow(self, state: WorkflowState, user_id: str) -> WorkflowState:
        """Execute creative generation flow."""
        # Creative generation
        generator = PipelineNode(
            self.pipeline_factory,
            ModelProfileType.Primary,
            stream=True
        )
        state = await generator(state)
        
        return state

    async def _execute_default_flow(self, state: WorkflowState, user_id: str) -> WorkflowState:
        """Fallback execution flow."""
        return await self._execute_chat_flow(state, user_id)

    async def create_initial_state(
        self,
        user_id: str,
        messages: List[Message],
        additional_context: Optional[Dict[str, Any]] = None,
    ) -> WorkflowState:
        """Create initial workflow state."""
        
        # Get user configuration
        from db import storage
        user_config = await storage.get_service(storage.user_config).get_user_config(user_id)
        
        # Convert messages to LangChain format
        langchain_messages = []
        for msg in messages:
            content_text = ""
            if isinstance(msg.content, list):
                content_parts = []
                for content_part in msg.content:
                    if hasattr(content_part, "text"):
                        content_parts.append(content_part.text)
                    elif isinstance(content_part, str):
                        content_parts.append(content_part)
                content_text = "\n".join(content_parts)
            else:
                content_text = str(msg.content)

            langchain_messages.append(
                LangChainMessage(
                    content=content_text,
                    type="human" if msg.role.value == "user" else "ai",
                )
            )

        state = WorkflowState(
            messages=langchain_messages,
            user_id=user_id,
            execution_metadata={
                "created_at": asyncio.get_event_loop().time(),
                "composer_version": "0.2.0",
                "streaming_enabled": user_config.workflow.enable_streaming,
                "workflow_timeout": user_config.workflow.default_timeout,
            },
        )

        # Add additional context
        if additional_context:
            for key, value in additional_context.items():
                state.execution_metadata[key] = value

        return state

    async def execute_workflow(
        self,
        workflow: CompiledStateGraph,
        initial_state: WorkflowState,
        stream: bool = True,
    ):
        """Execute the master workflow with streaming support."""
        try:
            streaming_enabled = initial_state.execution_metadata.get("streaming_enabled", True)
            
            if stream and streaming_enabled:
                async for event in workflow.astream_events(
                    initial_state.model_dump(), version="v2"
                ):
                    yield event
            else:
                result = await workflow.ainvoke(initial_state.model_dump())
                yield {"event": "workflow_complete", "data": result}

        except Exception as e:
            self.logger.error("Workflow execution failed", extra={"error": str(e)})
            yield {"event": "workflow_error", "data": {"error": str(e)}}


# Example usage demonstrating the refactored architecture
async def demo_refactored_service():
    """Demonstrate the refactored service with intelligent routing."""
    
    service = RefactoredComposerService()
    
    # Sample messages
    messages = [
        Message(
            role="user", 
            content="Can you research the latest developments in quantum computing?"
        )
    ]
    
    # Create master workflow (one workflow handles all types)
    master_workflow = await service.compose_master_workflow("demo_user")
    
    # Create initial state
    initial_state = await service.create_initial_state("demo_user", messages)
    
    # Execute with intelligent routing
    async for event in service.execute_workflow(master_workflow, initial_state):
        print(f"Event: {event}")

# This demonstrates the architectural improvement:
# 1. No redundant intent analysis 
# 2. No pre-workflow tool selection
# 3. Single workflow with intelligent internal routing
# 4. LangGraph handles the complexity natively