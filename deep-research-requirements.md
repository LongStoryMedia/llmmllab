# Deep Research Static Tool - Requirements Document

## Overview

This document outlines the requirements for implementing a deep research static tool for the composer system. The tool will provide comprehensive research capabilities using LangGraph subgraphs, database persistence with TimescaleDB, and intelligent caching based on semantic similarity and temporal relevance.

## Architecture Overview

```plaintext
Static Research Tool (LangChain BaseTool)
    ↓
LangGraph Subgraph
    ↓
┌─────────────────────────────────────────────────┐
│ Research Orchestrator Node                      │
│ - Check existing research cache                 │
│ - Create/update research task                   │
│ - Generate research plan                        │
└─────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────┐
│ Subtask Execution Nodes (Parallel)              │
│ - Static tools (web_search, memory, etc.)      │
│ - Agent nodes (analysis, specialized agents)   │
│ - Dynamic tools (generated per subtask)        │
│ - Information gathering & synthesis             │
└─────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────┐
│ Research Consolidation Node                     │
│ - Combine all subtask results                   │
│ - Generate final research synthesis             │
│ - Update task completion status                 │
└─────────────────────────────────────────────────┘
```

## Database Schema Requirements

### 1. Research Task Schema (`research_task.yaml`)

**Already exists** - The schema is properly defined with the following key properties:

```yaml
# schemas/research_task.yaml
properties:
  id: integer (Primary Key)
  user_id: string (User identifier)
  query: string (Original research query)
  conversation_id: integer (Optional conversation context)
  status: ResearchTaskStatus enum
  error_message: string (Optional error details)
  plan: string (LLM-generated research plan)
  embedding: array of numbers (Semantic search vector)
  subtasks: array of ResearchSubtask objects
  created_at: datetime (TimescaleDB hypertable partition key)
  updated_at: datetime
  completed_at: datetime (Optional)
```

### 2. Research Subtask Schema (`research_subtask.yaml`)

**Already exists** - Defines individual research questions within a task:

```yaml
# schemas/research_subtask.yaml
properties:
  id: integer (Primary Key)
  task_id: integer (Foreign Key to research_task)
  question: ResearchQuestion object
  result: ResearchQuestionResult object (Optional)
  status: ResearchTaskStatus enum
  error_message: string (Optional)
  created_at: datetime (TimescaleDB compatible)
  updated_at: datetime
```

### 3. Research Question Schema (`research_question.yaml`)

**Already exists** - Defines the structure of individual research questions:

```yaml
# schemas/research_question.yaml
properties:
  id: integer
  question: string (The question to investigate)
  keywords: array of strings (Search keywords)
  node_type: string (Type of node: "static_tool", "agent", "dynamic_tool")
  node_config: object (Configuration specific to the node type)
  execution_priority: integer (Execution order priority)
```

### 4. Research Question Result Schema (`research_question_result.yaml`)

**Already exists** - Stores the synthesized results:

```yaml
# schemas/research_question_result.yaml
properties:
  id: integer
  sources: array of strings (URLs, references)
  synthesized_answer: string (LLM-generated answer)
  error_message: string (Optional error details)
```

### 5. Research Task Status Enum (`research_task_status.yaml`)

**Already exists** - Comprehensive status tracking:

```yaml
# schemas/research_task_status.yaml
enum:
  - PENDING      # Initial state
  - PLANNING     # Generating research plan
  - GATHERING    # Executing subtasks
  - PROCESSING   # Processing gathered information
  - SYNTHESIZING # Creating final synthesis
  - COMPLETED    # Research complete
  - FAILED       # Error occurred
  - CANCELED     # User canceled
```

## Implementation Requirements

### Phase 1: Agent Architecture Refactoring

#### 1.1 Create Base Analysis Agent

Create `composer/agents/analysis/` directory structure:

```
composer/agents/analysis/
├── __init__.py
├── base_analysis_agent.py
├── intent_analysis_agent.py
└── research_task_agent.py
```

#### 1.2 Base Analysis Agent (`base_analysis_agent.py`)

```python
"""
Base analysis agent providing common LLM pipeline functionality.
"""
from abc import ABC, abstractmethod
from typing import Any, TypeVar, Generic
from models.conversation_ctx import ConversationCtx
from runner import pipeline_factory
from runner.pipeline_factory import PipelinePriority
from runner.pipelines.run import run_pipeline
from db import storage
from utils.model_profile import get_model_profile_for_task
from models.model_profile_type import ModelProfileType

T = TypeVar('T')

class BaseAnalysisAgent(ABC, Generic[T]):
    """
    Base class for analysis agents that use LLM pipelines.
    Provides common functionality for model profile resolution and pipeline execution.
    """
    
    def __init__(self, model_profile_type: ModelProfileType):
        """
        Initialize with the specific model profile type for this agent.
        
        Args:
            model_profile_type: The type of model profile to use for analysis
        """
        self.model_profile_type = model_profile_type
    
    async def get_analysis_pipeline(self, conversation_ctx: ConversationCtx):
        """Get the appropriate pipeline for analysis tasks."""
        mp = await get_model_profile_for_task(
            config=conversation_ctx.user_config.model_profiles,
            task=self.model_profile_type,
            user_id=conversation_ctx.user_config.user_id
        )
        
        return pipeline_factory.pipeline(
            profile=mp,
            expected_type=str,  # Most analysis returns text
            priority=PipelinePriority.NORMAL
        )
    
    @abstractmethod
    async def analyze(self, conversation_ctx: ConversationCtx) -> T:
        """
        Perform analysis and return the appropriate result type.
        
        Args:
            conversation_ctx: The conversation context to analyze
            
        Returns:
            Analysis result of type T
        """
        pass
    
    @abstractmethod
    def create_analysis_prompt(self, user_query: str) -> str:
        """
        Create the LLM prompt for analysis.
        
        Args:
            user_query: The user's query to analyze
            
        Returns:
            Formatted prompt string
        """
        pass
```

#### 1.3 Refactored Intent Analysis Agent (`intent_analysis_agent.py`)

```python
"""
Intent analysis agent - refactored to use base analysis functionality.
"""
from models.intent_analysis import IntentAnalysis
from models.model_profile_type import ModelProfileType
from .base_analysis_agent import BaseAnalysisAgent
from utils.message import extract_message_text
import json

class IntentAnalysisAgent(BaseAnalysisAgent[IntentAnalysis]):
    """
    Analyzes user intent and classifies complexity, capabilities, and requirements.
    """
    
    def __init__(self):
        """Initialize with Analysis model profile type."""
        super().__init__(ModelProfileType.Analysis)
    
    async def analyze(self, conversation_ctx: ConversationCtx) -> IntentAnalysis:
        """
        Analyze conversation context to determine intent and requirements.
        """
        try:
            user_query = extract_message_text(conversation_ctx.current_user_message)
            
            async with await self.get_analysis_pipeline(conversation_ctx) as pipeline:
                prompt = self.create_analysis_prompt(user_query)
                result = await run_pipeline(
                    messages=prompt,
                    pipeline=pipeline,
                    tools=None
                )
                
                llm_response = extract_message_text(result.message) if result.message else ""
                return self._parse_llm_response(llm_response)
                
        except Exception as e:
            # Fallback to heuristic analysis
            return self._fallback_heuristic_analysis(user_query)
    
    def create_analysis_prompt(self, user_query: str) -> str:
        """Create intent analysis prompt."""
        return f"""
You are an expert intent classification system. Analyze the user request and provide a structured JSON response.

Available intent types: chat, research, creative, technical, summarization, analysis, tool_use
Available complexity levels: TRIVIAL, SIMPLE, MODERATE, COMPLEX, SPECIALIZED
Available capabilities: basic_math, text_processing, information_retrieval, conversation_memory, web_search, summarization, reasoning, general_knowledge, api_integration, async_processing, file_manipulation, data_processing, image_processing, audio_processing, database_access, network_communication
Available computational requirements: high_memory, gpu_acceleration, parallel_processing, real_time_processing, large_data_handling, complex_reasoning, multi_modal_processing, external_api_calls, file_operations, database_operations

User Request: {user_query}

Respond with ONLY a JSON object in this exact format:
{{
    "primary_intent": "<intent_type>",
    "complexity_level": "<complexity_level>",
    "required_capabilities": ["<capability1>", "<capability2>"],
    "computational_requirements": ["<requirement1>", "<requirement2>"],
    "domain_specificity": 0.0,
    "reusability_potential": 0.0,
    "confidence": 0.0,
    "reasoning": "Brief explanation of classification decisions"
}}
"""
    
    # ... (existing parsing and fallback methods remain the same)
```

#### 1.4 Research Task Agent (`research_task_agent.py`)

```python
"""
Research task planning agent - creates comprehensive research plans.
"""
from models.research_task import ResearchTask
from models.research_question import ResearchQuestion
from models.research_subtask import ResearchSubtask
from models.research_task_status import ResearchTaskStatus
from models.model_profile_type import ModelProfileType
from .base_analysis_agent import BaseAnalysisAgent
from utils.message import extract_message_text
from datetime import datetime
import json

class ResearchTaskAgent(BaseAnalysisAgent[ResearchTask]):
    """
    Creates detailed research plans with subtasks for complex queries.
    """
    
    def __init__(self):
        """Initialize with Research Task model profile type."""
        super().__init__(ModelProfileType.ResearchTask)
    
    async def analyze(self, conversation_ctx: ConversationCtx) -> ResearchTask:
        """
        Create a comprehensive research task with plan and subtasks.
        """
        try:
            user_query = extract_message_text(conversation_ctx.current_user_message)
            
            async with await self.get_analysis_pipeline(conversation_ctx) as pipeline:
                prompt = self.create_analysis_prompt(user_query)
                result = await run_pipeline(
                    messages=prompt,
                    pipeline=pipeline,
                    tools=None
                )
                
                llm_response = extract_message_text(result.message) if result.message else ""
                return await self._create_research_task(llm_response, user_query, conversation_ctx)
                
        except Exception as e:
            # Create fallback research task
            return await self._create_fallback_task(user_query, conversation_ctx, str(e))
    
    def create_analysis_prompt(self, user_query: str) -> str:
        """Create research planning prompt."""
        return f"""
You are an expert research planner. Create a comprehensive research plan for the given query.

User Research Query: {user_query}

Create a detailed research plan with the following structure:

1. **Research Plan Overview**: A 2-3 sentence summary of the research approach
2. **Key Research Questions**: 3-8 specific questions that need to be investigated
3. **Execution Strategy**: For each question, determine the best node type and configuration

Available node types:
- "web_search": For web-based information gathering
- "memory_search": For searching conversation/user memory
- "static_analysis": For text analysis and summarization
- "agent_analysis": For complex reasoning tasks
- "dynamic_tool": For specialized tools generated per task

Respond with ONLY a JSON object in this exact format:
{{
    "plan": "Detailed research plan overview describing the approach and methodology",
    "questions": [
        {{
            "question": "Specific research question 1",
            "keywords": ["keyword1", "keyword2", "keyword3"],
            "node_type": "web_search",
            "node_config": {{"max_results": 5, "search_depth": "standard"}},
            "execution_priority": 1
        }},
        {{
            "question": "Specific research question 2", 
            "keywords": ["keyword1", "keyword2", "keyword3"],
            "node_type": "agent_analysis",
            "node_config": {{"analysis_type": "comparative", "depth": "detailed"}},
            "execution_priority": 2
        }}
    ]
}}

Guidelines:
- Questions should be specific and answerable through the chosen method
- Select the most appropriate node type for each question
- Keywords should be optimized for the chosen execution method
- Plan should be comprehensive but focused
- Limit to maximum 8 questions to ensure quality over quantity
- Consider execution dependencies when setting priorities
"""
    
    async def _create_research_task(
        self, 
        llm_response: str, 
        user_query: str, 
        conversation_ctx: ConversationCtx
    ) -> ResearchTask:
        """Create ResearchTask from LLM response."""
        try:
            data = json.loads(llm_response.strip())
            
            # Create subtasks from questions
            subtasks = []
            for i, q_data in enumerate(data.get("questions", [])):
                question = ResearchQuestion(
                    id=i + 1,
                    question=q_data["question"],
                    keywords=q_data["keywords"]
                )
                
                subtask = ResearchSubtask(
                    id=i + 1,
                    task_id=0,  # Will be set after task creation
                    question=question,
                    result=None,
                    status=ResearchTaskStatus.PENDING,
                    error_message=None,
                    created_at=datetime.now(),
                    updated_at=datetime.now()
                )
                subtasks.append(subtask)
            
            # Create the research task
            task = ResearchTask(
                id=0,  # Will be set by database
                user_id=conversation_ctx.user_config.user_id,
                query=user_query,
                conversation_id=conversation_ctx.conversation.id,
                status=ResearchTaskStatus.PLANNING,
                error_message=None,
                plan=data.get("plan", ""),
                embedding=None,  # Will be generated later
                subtasks=subtasks,
                created_at=datetime.now(),
                updated_at=datetime.now(),
                completed_at=None
            )
            
            return task
            
        except Exception as e:
            return await self._create_fallback_task(user_query, conversation_ctx, str(e))
    
    async def _create_fallback_task(
        self, 
        user_query: str, 
        conversation_ctx: ConversationCtx, 
        error: str
    ) -> ResearchTask:
        """Create a basic research task when LLM planning fails."""
        fallback_question = ResearchQuestion(
            id=1,
            question=f"Research and analyze: {user_query}",
            keywords=user_query.split()[:5]  # Use first 5 words as keywords
        )
        
        fallback_subtask = ResearchSubtask(
            id=1,
            task_id=0,
            question=fallback_question,
            result=None,
            status=ResearchTaskStatus.PENDING,
            error_message=None,
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        
        return ResearchTask(
            id=0,
            user_id=conversation_ctx.user_config.user_id,
            query=user_query,
            conversation_id=conversation_ctx.conversation.id,
            status=ResearchTaskStatus.PLANNING,
            error_message=f"Fallback task created due to: {error}",
            plan=f"Conduct comprehensive research on: {user_query}",
            embedding=None,
            subtasks=[fallback_subtask],
            created_at=datetime.now(),
            updated_at=datetime.now(),
            completed_at=None
        )
```

### Phase 2: Database Layer Implementation

#### 2.1 Database Service Interface

Create `db/research_task_service.py`:

```python
"""
Database service for research task management with TimescaleDB integration.
"""
from typing import List, Optional
from datetime import datetime, timedelta
from db.base_service import BaseService
from models.research_task import ResearchTask
from models.research_subtask import ResearchSubtask
from models.research_task_status import ResearchTaskStatus

class ResearchTaskService(BaseService):
    """Service for managing research tasks in TimescaleDB."""
    
    async def create_task(self, task: ResearchTask) -> int:
        """
        Create a new research task with subtasks.
        
        Args:
            task: ResearchTask object to create
            
        Returns:
            ID of the created task
        """
        query = """
            INSERT INTO research_tasks (
                user_id, query, conversation_id, status, plan, 
                embedding, created_at, updated_at
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            RETURNING id
        """
        
        task_id = await self.fetch_val(
            query,
            task.user_id,
            task.query,
            task.conversation_id,
            task.status.value,
            task.plan,
            task.embedding,
            task.created_at,
            task.updated_at
        )
        
        # Create subtasks
        for subtask in task.subtasks or []:
            subtask.task_id = task_id
            await self.create_subtask(subtask)
        
        return task_id
    
    async def create_subtask(self, subtask: ResearchSubtask) -> int:
        """Create a research subtask."""
        query = """
            INSERT INTO research_subtasks (
                task_id, question_id, question_text, question_keywords,
                status, created_at, updated_at
            ) VALUES ($1, $2, $3, $4, $5, $6, $7)
            RETURNING id
        """
        
        return await self.fetch_val(
            query,
            subtask.task_id,
            subtask.question.id if subtask.question else None,
            subtask.question.question if subtask.question else None,
            subtask.question.keywords if subtask.question else [],
            subtask.status.value,
            subtask.created_at,
            subtask.updated_at
        )
    
    async def find_similar_tasks(
        self,
        embedding: List[float],
        user_id: str,
        similarity_threshold: float = 0.8,
        max_age_days: int = 30,
        limit: int = 5
    ) -> List[ResearchTask]:
        """
        Find similar research tasks based on embedding similarity and recency.
        
        Args:
            embedding: Query embedding vector
            user_id: User ID to search within
            similarity_threshold: Minimum cosine similarity (0.0-1.0)
            max_age_days: Maximum age of tasks to consider
            limit: Maximum number of results
            
        Returns:
            List of similar research tasks
        """
        cutoff_date = datetime.now() - timedelta(days=max_age_days)
        
        query = """
            SELECT rt.*, 
                   1 - (rt.embedding <=> $1::vector) as similarity,
                   -- Age factor: newer tasks get higher scores
                   EXTRACT(EPOCH FROM (NOW() - rt.created_at)) / (86400.0 * $4) as age_factor
            FROM research_tasks rt
            WHERE rt.user_id = $2
            AND rt.status = 'COMPLETED'
            AND rt.created_at >= $3
            AND rt.embedding IS NOT NULL
            AND 1 - (rt.embedding <=> $1::vector) >= $5
            ORDER BY (
                (1 - (rt.embedding <=> $1::vector)) * 
                (1 - (EXTRACT(EPOCH FROM (NOW() - rt.created_at)) / (86400.0 * $4)))
            ) DESC
            LIMIT $6
        """
        
        rows = await self.fetch(
            query,
            embedding,
            user_id,
            cutoff_date,
            max_age_days,
            similarity_threshold,
            limit
        )
        
        return [self._row_to_research_task(row) for row in rows]
    
    async def update_task_status(self, task_id: int, status: ResearchTaskStatus, error_message: str = None):
        """Update research task status."""
        query = """
            UPDATE research_tasks 
            SET status = $2, error_message = $3, updated_at = NOW()
            WHERE id = $1
        """
        
        await self.execute(query, task_id, status.value, error_message)
        
        if status == ResearchTaskStatus.COMPLETED:
            await self.execute(
                "UPDATE research_tasks SET completed_at = NOW() WHERE id = $1",
                task_id
            )
    
    async def update_subtask_result(
        self, 
        subtask_id: int, 
        result: dict, 
        status: ResearchTaskStatus = ResearchTaskStatus.COMPLETED
    ):
        """Update subtask with results."""
        query = """
            UPDATE research_subtasks 
            SET result_sources = $2, result_answer = $3, status = $4, updated_at = NOW()
            WHERE id = $1
        """
        
        await self.execute(
            query,
            subtask_id,
            result.get('sources', []),
            result.get('synthesized_answer', ''),
            status.value
        )
    
    def _row_to_research_task(self, row) -> ResearchTask:
        """Convert database row to ResearchTask object."""
        # Implementation details for row conversion
        pass
```

#### 2.2 SQL Schema Migration

Create `db/sql/research_tables.sql`:

```sql
-- Research Tasks table with TimescaleDB hypertable
CREATE TABLE research_tasks (
    id SERIAL PRIMARY KEY,
    user_id VARCHAR(255) NOT NULL,
    query TEXT NOT NULL,
    conversation_id INTEGER,
    status VARCHAR(50) NOT NULL DEFAULT 'PENDING',
    error_message TEXT,
    plan TEXT,
    embedding VECTOR(768),  -- Requires pgvector extension
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    completed_at TIMESTAMP WITH TIME ZONE,
    
    -- Indexes for performance
    INDEX idx_research_tasks_user_id (user_id),
    INDEX idx_research_tasks_status (status),
    INDEX idx_research_tasks_created_at (created_at)
);

-- Convert to TimescaleDB hypertable partitioned by time
SELECT create_hypertable('research_tasks', 'created_at');

-- Create vector similarity index
CREATE INDEX idx_research_tasks_embedding 
ON research_tasks 
USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);

-- Research Subtasks table
CREATE TABLE research_subtasks (
    id SERIAL PRIMARY KEY,
    task_id INTEGER NOT NULL REFERENCES research_tasks(id) ON DELETE CASCADE,
    question_id INTEGER,
    question_text TEXT NOT NULL,
    question_keywords TEXT[] DEFAULT '{}',
    result_sources TEXT[] DEFAULT '{}',
    result_answer TEXT,
    status VARCHAR(50) NOT NULL DEFAULT 'PENDING',
    error_message TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    INDEX idx_research_subtasks_task_id (task_id),
    INDEX idx_research_subtasks_status (status)
);

-- Convert subtasks to hypertable as well (optional, for large datasets)
SELECT create_hypertable('research_subtasks', 'created_at');
```

### Phase 3: Static Research Tool Implementation

#### 3.1 Deep Research Tool (`composer/tools/static/deep_research_tool.py`)

```python
"""
Static deep research tool using LangGraph subgraphs.

This tool performs comprehensive research using a multi-stage process:
1. Check for existing similar research (semantic + temporal)
2. Create research plan with subtasks
3. Execute subtasks in parallel
4. Synthesize final results
"""

import asyncio
import json
from typing import Dict, Any, List
from datetime import datetime

from langchain_core.tools import BaseTool
from langgraph import StateGraph, END
from langgraph.graph import Graph

from models.research_task import ResearchTask
from models.research_task_status import ResearchTaskStatus
from models.conversation_ctx import ConversationCtx
from composer.agents.analysis.research_task_agent import ResearchTaskAgent
from composer.tools.static.web_search_tool import WebSearchTool
from composer.tools.static.summarization_tool import SummarizationTool
from db import storage
from runner import embed_pipeline
from utils.model_profile import get_model_profile_for_task
from models.model_profile_type import ModelProfileType


class DeepResearchState:
    """State object for the research subgraph."""
    
    def __init__(self):
        self.query: str = ""
        self.conversation_ctx: ConversationCtx = None
        self.research_task: ResearchTask = None
        self.existing_results: List[Dict] = []
        self.subtask_results: Dict[int, Dict] = {}
        self.final_synthesis: str = ""
        self.error_message: str = ""


class DeepResearchTool(BaseTool):
    """Static tool for comprehensive research using LangGraph subgraphs."""
    
    name: str = "deep_research"
    description: str = """
    Perform comprehensive research on complex topics using a structured approach.
    Creates research plans, executes subtasks, and provides synthesized results.
    Uses caching to avoid duplicate research and provides detailed citations.
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.research_agent = ResearchTaskAgent()
        self.web_search_tool = WebSearchTool()
        self.summarization_tool = SummarizationTool()
    
    async def _arun(self, query: str, **kwargs) -> str:
        """Execute deep research using LangGraph subgraph."""
        try:
            # Create conversation context (simplified for static tool)
            conversation_ctx = self._create_minimal_context(query)
            
            # Build and execute research subgraph
            research_graph = self._build_research_graph()
            
            # Initialize state
            initial_state = DeepResearchState()
            initial_state.query = query
            initial_state.conversation_ctx = conversation_ctx
            
            # Execute the graph
            final_state = await research_graph.ainvoke(initial_state)
            
            # Format final results
            return self._format_research_results(final_state)
            
        except Exception as e:
            return json.dumps({
                "status": "error",
                "error": str(e),
                "query": query
            }, indent=2)
    
    def _build_research_graph(self) -> Graph:
        """Build the LangGraph subgraph for research execution."""
        
        # Define the state graph
        workflow = StateGraph(DeepResearchState)
        
        # Add nodes
        workflow.add_node("check_existing", self._check_existing_research)
        workflow.add_node("create_plan", self._create_research_plan)
        workflow.add_node("execute_subtasks", self._execute_subtasks)
        workflow.add_node("synthesize_results", self._synthesize_results)
        workflow.add_node("store_results", self._store_results)
        
        # Define the flow
        workflow.set_entry_point("check_existing")
        
        workflow.add_conditional_edges(
            "check_existing",
            self._should_use_existing,
            {
                "use_existing": "synthesize_results",
                "create_new": "create_plan"
            }
        )
        
        workflow.add_edge("create_plan", "execute_subtasks")
        workflow.add_edge("execute_subtasks", "synthesize_results")
        workflow.add_edge("synthesize_results", "store_results")
        workflow.add_edge("store_results", END)
        
        return workflow.compile()
    
    async def _check_existing_research(self, state: DeepResearchState) -> DeepResearchState:
        """Check for existing similar research in the database."""
        try:
            # Generate embedding for the query
            embedding_mp = await get_model_profile_for_task(
                config=state.conversation_ctx.user_config.model_profiles,
                task=ModelProfileType.Embedding,
                user_id=state.conversation_ctx.user_config.user_id
            )
            
            # Get query embedding
            query_embedding = await embed_pipeline(
                text=state.query,
                pipeline=embedding_mp
            )
            
            # Search for similar completed research
            research_service = storage.get_service(storage.research_task)
            similar_tasks = await research_service.find_similar_tasks(
                embedding=query_embedding[0] if query_embedding else [],
                user_id=state.conversation_ctx.user_config.user_id,
                similarity_threshold=0.8,  # User configurable
                max_age_days=30,  # User configurable
                limit=3
            )
            
            # Extract results from similar tasks
            state.existing_results = []
            for task in similar_tasks:
                if task.status == ResearchTaskStatus.COMPLETED:
                    task_results = {
                        "task_id": task.id,
                        "query": task.query,
                        "plan": task.plan,
                        "subtask_results": [],
                        "created_at": task.created_at.isoformat()
                    }
                    
                    # Get subtask results
                    for subtask in task.subtasks or []:
                        if subtask.result:
                            task_results["subtask_results"].append({
                                "question": subtask.question.question,
                                "answer": subtask.result.synthesized_answer,
                                "sources": subtask.result.sources or []
                            })
                    
                    state.existing_results.append(task_results)
            
            return state
            
        except Exception as e:
            state.error_message = f"Error checking existing research: {str(e)}"
            return state
    
    def _should_use_existing(self, state: DeepResearchState) -> str:
        """Determine whether to use existing results or create new research."""
        if state.existing_results and len(state.existing_results) > 0:
            # Check if existing results are comprehensive enough
            total_subtasks = sum(
                len(result.get("subtask_results", [])) 
                for result in state.existing_results
            )
            if total_subtasks >= 3:  # User configurable threshold
                return "use_existing"
        
        return "create_new"
    
    async def _create_research_plan(self, state: DeepResearchState) -> DeepResearchState:
        """Create a new research plan using the research agent."""
        try:
            # Use research agent to create plan
            research_task = await self.research_agent.analyze(state.conversation_ctx)
            
            # Generate embedding for the plan
            embedding_mp = await get_model_profile_for_task(
                config=state.conversation_ctx.user_config.model_profiles,
                task=ModelProfileType.Embedding,
                user_id=state.conversation_ctx.user_config.user_id
            )
            
            plan_embedding = await embed_pipeline(
                text=research_task.plan,
                pipeline=embedding_mp
            )
            research_task.embedding = plan_embedding[0] if plan_embedding else None
            
            # Store in database
            research_service = storage.get_service(storage.research_task)
            task_id = await research_service.create_task(research_task)
            research_task.id = task_id
            
            state.research_task = research_task
            return state
            
        except Exception as e:
            state.error_message = f"Error creating research plan: {str(e)}"
            return state
    
    async def _execute_subtasks(self, state: DeepResearchState) -> DeepResearchState:
        """Execute research subtasks in parallel using appropriate node types."""
        try:
            if not state.research_task or not state.research_task.subtasks:
                state.error_message = "No subtasks to execute"
                return state
            
            # Group subtasks by priority for ordered execution
            priority_groups = self._group_subtasks_by_priority(state.research_task.subtasks)
            
            # Execute subtasks in parallel (with concurrency limit)
            semaphore = asyncio.Semaphore(3)  # User configurable
            
            async def execute_single_subtask(subtask):
                async with semaphore:
                    return await self._execute_subtask_by_node_type(
                        subtask.question,
                        state.conversation_ctx
                    )
            
            # Execute priority groups in sequence, but subtasks within groups in parallel
            all_results = {}
            for priority, subtasks_group in sorted(priority_groups.items()):
                tasks = [
                    execute_single_subtask(subtask) 
                    for subtask in subtasks_group
                ]
                
                group_results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Store results with subtask IDs
                for i, result in enumerate(group_results):
                    subtask = subtasks_group[i]
                    all_results[subtask.id] = result
            
            # Store results
            research_service = storage.get_service(storage.research_task)
            for i, result in enumerate(results):
                subtask = state.research_task.subtasks[i]
                
                if isinstance(result, Exception):
                    state.subtask_results[subtask.id] = {
                        "error": str(result),
                        "sources": [],
                        "synthesized_answer": ""
                    }
                    await research_service.update_subtask_result(
                        subtask.id,
                        {"error": str(result)},
                        ResearchTaskStatus.FAILED
                    )
                else:
                    state.subtask_results[subtask.id] = result
                    await research_service.update_subtask_result(
                        subtask.id,
                        result,
                        ResearchTaskStatus.COMPLETED
                    )
            
            return state
            
        except Exception as e:
            state.error_message = f"Error executing subtasks: {str(e)}"
            return state
    
    def _group_subtasks_by_priority(self, subtasks: List[ResearchSubtask]) -> Dict[int, List[ResearchSubtask]]:
        """Group subtasks by execution priority."""
        priority_groups = {}
        for subtask in subtasks:
            priority = getattr(subtask.question, 'execution_priority', 1)
            if priority not in priority_groups:
                priority_groups[priority] = []
            priority_groups[priority].append(subtask)
        return priority_groups
    
    async def _execute_subtask_by_node_type(self, question: ResearchQuestion, conversation_ctx: ConversationCtx) -> Dict:
        """Execute a subtask using the appropriate node type."""
        try:
            node_type = getattr(question, 'node_type', 'web_search')
            node_config = getattr(question, 'node_config', {})
            
            if node_type == "web_search":
                return await self._execute_web_search_subtask(question, node_config)
            elif node_type == "memory_search":
                return await self._execute_memory_search_subtask(question, node_config)
            elif node_type == "static_analysis":
                return await self._execute_static_analysis_subtask(question, node_config)
            elif node_type == "agent_analysis":
                return await self._execute_agent_analysis_subtask(question, node_config, conversation_ctx)
            elif node_type == "dynamic_tool":
                return await self._execute_dynamic_tool_subtask(question, node_config, conversation_ctx)
            else:
                # Fallback to web search for unknown node types
                return await self._execute_web_search_subtask(question, {})
                
        except Exception as e:
            return {
                "sources": [],
                "synthesized_answer": f"Error executing subtask: {str(e)}",
                "error": str(e),
                "question": question.question,
                "node_type": getattr(question, 'node_type', 'unknown')
            }
    
    async def _execute_web_search_subtask(self, question: ResearchQuestion, config: Dict) -> Dict:
        """Execute web search subtask."""
        try:
            max_results = config.get('max_results', 5)
            search_query = f"{question.question} {' '.join(question.keywords[:3])}"
            
            search_results = await self.web_search_tool._arun(search_query)
            search_data = json.loads(search_results)
            
            if search_data.get("status") != "success" or not search_data.get("results"):
                return {
                    "sources": [],
                    "synthesized_answer": f"No search results found for: {question.question}",
                    "error": "No search results",
                    "node_type": "web_search"
                }
            
            # Combine and summarize search content
            combined_content = ""
            sources = []
            
            for result in search_data["results"][:max_results]:
                combined_content += f"Source: {result['title']}\nURL: {result['url']}\nContent: {result['content']}\n\n"
                sources.append(result["url"])
            
            summary_prompt = f"Question: {question.question}\n\nBased on the following information, provide a comprehensive answer:\n\n{combined_content}"
            summary_result = await self.summarization_tool._arun(summary_prompt)
            summary_data = json.loads(summary_result)
            
            return {
                "sources": sources,
                "synthesized_answer": summary_data.get("summary", ""),
                "question": question.question,
                "node_type": "web_search"
            }
            
        except Exception as e:
            return {
                "sources": [],
                "synthesized_answer": f"Error in web search: {str(e)}",
                "error": str(e),
                "question": question.question,
                "node_type": "web_search"
            }
    
    async def _execute_memory_search_subtask(self, question: ResearchQuestion, config: Dict) -> Dict:
        """Execute memory search subtask."""
        try:
            from composer.tools.static.memory_retrieval_tool import MemoryRetrievalTool
            
            memory_tool = MemoryRetrievalTool()
            search_query = f"{question.question} {' '.join(question.keywords)}"
            
            memory_results = await memory_tool._arun(search_query)
            memory_data = json.loads(memory_results)
            
            if memory_data.get("status") != "success" or not memory_data.get("memories"):
                return {
                    "sources": ["conversation_memory"],
                    "synthesized_answer": f"No relevant memories found for: {question.question}",
                    "question": question.question,
                    "node_type": "memory_search"
                }
            
            # Combine memory content
            combined_memories = ""
            for memory in memory_data["memories"]:
                combined_memories += f"Memory: {memory.get('content', '')}\nTimestamp: {memory.get('timestamp', '')}\n\n"
            
            # Summarize memories to answer question
            summary_prompt = f"Question: {question.question}\n\nBased on the following memories, provide a comprehensive answer:\n\n{combined_memories}"
            summary_result = await self.summarization_tool._arun(summary_prompt)
            summary_data = json.loads(summary_result)
            
            return {
                "sources": ["conversation_memory"],
                "synthesized_answer": summary_data.get("summary", ""),
                "question": question.question,
                "node_type": "memory_search"
            }
            
        except Exception as e:
            return {
                "sources": [],
                "synthesized_answer": f"Error in memory search: {str(e)}",
                "error": str(e),
                "question": question.question,
                "node_type": "memory_search"
            }
    
    async def _execute_static_analysis_subtask(self, question: ResearchQuestion, config: Dict) -> Dict:
        """Execute static analysis subtask (text processing, summarization)."""
        try:
            analysis_type = config.get('analysis_type', 'summarization')
            
            if analysis_type == 'summarization':
                # Use summarization tool directly on the question context
                summary_result = await self.summarization_tool._arun(question.question)
                summary_data = json.loads(summary_result)
                
                return {
                    "sources": ["static_analysis"],
                    "synthesized_answer": summary_data.get("summary", ""),
                    "question": question.question,
                    "node_type": "static_analysis"
                }
            else:
                # Other analysis types can be implemented here
                return {
                    "sources": ["static_analysis"],
                    "synthesized_answer": f"Static analysis completed for: {question.question}",
                    "question": question.question,
                    "node_type": "static_analysis"
                }
                
        except Exception as e:
            return {
                "sources": [],
                "synthesized_answer": f"Error in static analysis: {str(e)}",
                "error": str(e),
                "question": question.question,
                "node_type": "static_analysis"
            }
    
    async def _execute_agent_analysis_subtask(self, question: ResearchQuestion, config: Dict, conversation_ctx: ConversationCtx) -> Dict:
        """Execute agent-based analysis subtask."""
        try:
            from composer.agents.analysis.intent_analysis_agent import IntentAnalysisAgent
            
            # Use intent analysis agent for complex reasoning tasks
            analysis_agent = IntentAnalysisAgent()
            
            # Create a temporary conversation context for the question
            temp_ctx = conversation_ctx  # Could be modified for the specific question
            
            # This would need to be adapted based on the specific agent capabilities
            analysis_result = await analysis_agent.analyze(temp_ctx)
            
            return {
                "sources": ["agent_analysis"],
                "synthesized_answer": f"Agent analysis completed for: {question.question}. Reasoning: {analysis_result.reasoning}",
                "question": question.question,
                "node_type": "agent_analysis",
                "analysis_confidence": analysis_result.confidence
            }
            
        except Exception as e:
            return {
                "sources": [],
                "synthesized_answer": f"Error in agent analysis: {str(e)}",
                "error": str(e),
                "question": question.question,
                "node_type": "agent_analysis"
            }
    
    async def _execute_dynamic_tool_subtask(self, question: ResearchQuestion, config: Dict, conversation_ctx: ConversationCtx) -> Dict:
        """Execute dynamic tool subtask (tools generated per specific need)."""
        try:
            # This would integrate with the dynamic tool generation system
            # For now, return a placeholder implementation
            tool_type = config.get('tool_type', 'general')
            
            # Dynamic tool generation would happen here
            # This could involve creating specialized tools based on the question domain
            
            return {
                "sources": ["dynamic_tool"],
                "synthesized_answer": f"Dynamic tool analysis completed for: {question.question} (Tool type: {tool_type})",
                "question": question.question,
                "node_type": "dynamic_tool",
                "tool_config": config
            }
            
        except Exception as e:
            return {
                "sources": [],
                "synthesized_answer": f"Error in dynamic tool: {str(e)}",
                "error": str(e),
                "question": question.question,
                "node_type": "dynamic_tool"
            }
    
    async def _synthesize_results(self, state: DeepResearchState) -> DeepResearchState:
        """Synthesize final research results."""
        try:
            # Combine all results (existing + new)
            all_results = []
            
            # Add existing results
            for existing in state.existing_results:
                all_results.extend(existing.get("subtask_results", []))
            
            # Add new subtask results
            for result in state.subtask_results.values():
                if "synthesized_answer" in result and result["synthesized_answer"]:
                    all_results.append(result)
            
            # Create final synthesis
            if all_results:
                synthesis_content = f"Research Query: {state.query}\n\n"
                synthesis_content += "Comprehensive Research Results:\n\n"
                
                for i, result in enumerate(all_results, 1):
                    synthesis_content += f"{i}. {result.get('question', 'Research Finding')}\n"
                    synthesis_content += f"Answer: {result.get('synthesized_answer', '')}\n"
                    if result.get('sources'):
                        synthesis_content += f"Sources: {', '.join(result['sources'][:3])}\n"
                    synthesis_content += "\n"
                
                # Use summarization tool for final synthesis
                final_summary = await self.summarization_tool._arun(synthesis_content)
                summary_data = json.loads(final_summary)
                
                state.final_synthesis = summary_data.get("summary", synthesis_content)
            else:
                state.final_synthesis = f"No comprehensive results found for: {state.query}"
            
            return state
            
        except Exception as e:
            state.error_message = f"Error synthesizing results: {str(e)}"
            state.final_synthesis = f"Research completed with errors for: {state.query}"
            return state
    
    async def _store_results(self, state: DeepResearchState) -> DeepResearchState:
        """Store final results and update task status."""
        try:
            if state.research_task:
                research_service = storage.get_service(storage.research_task)
                await research_service.update_task_status(
                    state.research_task.id,
                    ResearchTaskStatus.COMPLETED if not state.error_message else ResearchTaskStatus.FAILED,
                    state.error_message
                )
            
            return state
            
        except Exception as e:
            state.error_message = f"Error storing results: {str(e)}"
            return state
    
    def _format_research_results(self, state: DeepResearchState) -> str:
        """Format the final research results for return."""
        # Count total sources
        all_sources = set()
        for result in state.subtask_results.values():
            all_sources.update(result.get("sources", []))
        
        for existing in state.existing_results:
            for subtask in existing.get("subtask_results", []):
                all_sources.update(subtask.get("sources", []))
        
        return json.dumps({
            "status": "success" if not state.error_message else "partial_success",
            "query": state.query,
            "research_synthesis": state.final_synthesis,
            "total_sources": len(all_sources),
            "source_urls": list(all_sources)[:10],  # Limit displayed sources
            "subtask_count": len(state.subtask_results),
            "used_existing_research": len(state.existing_results) > 0,
            "error": state.error_message or None,
            "task_id": state.research_task.id if state.research_task else None
        }, indent=2)
    
    def _create_minimal_context(self, query: str) -> ConversationCtx:
        """Create minimal conversation context for static tool use."""
        # This would need to be implemented based on your ConversationCtx structure
        # For static tools, you might need default configurations
        pass
    
    def _run(self, query: str, **kwargs) -> str:
        """Sync implementation using async."""
        return asyncio.run(self._arun(query, **kwargs))
```

#### 3.2 Update Static Tools Init (`composer/tools/static/__init__.py`)

```python
"""
Static tools for LangChain integration.
"""

from .web_search_tool import WebSearchTool
from .memory_retrieval_tool import MemoryRetrievalTool  
from .summarization_tool import SummarizationTool
from .deep_research_tool import DeepResearchTool

__all__ = [
    "WebSearchTool",
    "MemoryRetrievalTool", 
    "SummarizationTool",
    "DeepResearchTool",
]
```

### Phase 4: Configuration Requirements

#### 4.1 User Configuration Schema Updates

Add to `schemas/user_config.yaml`:

```yaml
research:
  type: object
  description: Deep research tool configuration
  properties:
    max_subtasks:
      type: integer
      minimum: 1
      maximum: 10
      default: 6
      description: Maximum number of research subtasks to create
    similarity_threshold:
      type: number
      minimum: 0.0
      maximum: 1.0
      default: 0.8
      description: Minimum similarity threshold for reusing existing research
    max_age_days:
      type: integer
      minimum: 1
      maximum: 365
      default: 30
      description: Maximum age of existing research to consider (days)
    max_concurrent_subtasks:
      type: integer
      minimum: 1
      maximum: 10
      default: 3
      description: Maximum number of subtasks to execute concurrently
    enable_caching:
      type: boolean
      default: true
      description: Whether to cache and reuse research results
    node_type_preferences:
      type: object
      description: Preferences for different node types
      properties:
        web_search:
          type: object
          properties:
            max_results: {type: integer, default: 5}
            search_depth: {type: string, enum: ["standard", "deep"], default: "standard"}
        memory_search:
          type: object
          properties:
            max_memories: {type: integer, default: 10}
            similarity_threshold: {type: number, default: 0.7}
        agent_analysis:
          type: object
          properties:
            analysis_depth: {type: string, enum: ["basic", "detailed", "comprehensive"], default: "detailed"}
            enable_reasoning_chains: {type: boolean, default: true}
        dynamic_tools:
          type: object
          properties:
            enable_generation: {type: boolean, default: true}
            max_generation_time: {type: integer, default: 60}
    execution_strategy:
      type: string
      enum: ["parallel", "sequential", "priority_based"]
      default: "priority_based"
      description: How to execute subtasks
```

### Phase 5: Implementation Steps

#### Step 1: Database Setup
1. Run database migration to create research tables
2. Install pgvector extension for embedding similarity
3. Configure TimescaleDB hypertables for time-series optimization

#### Step 2: Agent Refactoring
1. Create `composer/agents/analysis/` directory
2. Implement `BaseAnalysisAgent` class
3. Refactor existing `IntentClassifierAgent` to extend base class
4. Implement new `ResearchTaskAgent`
5. Update imports and dependencies

#### Step 3: Database Services
1. Implement `ResearchTaskService` with full CRUD operations
2. Add semantic similarity search methods
3. Integrate with existing storage service registry
4. Add proper error handling and connection management

#### Step 4: Research Tool Implementation
1. Implement `DeepResearchTool` with LangGraph subgraph
2. Define proper state management between nodes
3. Implement parallel subtask execution with concurrency control
4. Add comprehensive error handling and fallback mechanisms
5. Integrate with existing static tools for web search and summarization

#### Step 5: Configuration Integration
1. Update user configuration schema for research settings
2. Add configuration validation and defaults
3. Integrate research configuration with existing config system
4. Add user-configurable thresholds and limits

#### Step 6: Testing and Validation
1. Create unit tests for all agent classes
2. Test database operations with TimescaleDB
3. Test LangGraph subgraph execution
4. Validate semantic similarity and caching behavior
5. Test error handling and fallback scenarios

## User Configuration Options

The following user-configurable options should be available:

```python
class ResearchConfig(BaseModel):
    max_subtasks: int = 6                    # Maximum research questions per task
    similarity_threshold: float = 0.8        # Semantic similarity threshold (0.0-1.0) 
    max_age_days: int = 30                   # Cache expiration in days
    max_concurrent_subtasks: int = 3         # Parallel execution limit
    enable_caching: bool = True              # Enable result caching
    search_results_per_question: int = 5     # Web search results per subtask
    synthesis_max_length: int = 2000         # Maximum synthesis length
```

## Expected Outcomes

Upon completion, the system will provide:

1. **Comprehensive Research Capability**: Multi-stage research with plan generation, parallel execution, and synthesis
2. **Flexible Node Execution**: Support for static tools, agents, and dynamically generated tools per subtask
3. **Intelligent Caching**: Semantic and temporal similarity matching to avoid duplicate work
4. **Scalable Architecture**: TimescaleDB integration for large-scale research data storage
5. **Adaptive Execution Strategy**: Priority-based, parallel, or sequential execution modes
6. **Multi-Modal Research**: Web search, memory retrieval, static analysis, agent reasoning, and dynamic tool generation
7. **Flexible Configuration**: User-configurable thresholds, limits, and node-specific behavior
8. **Robust Error Handling**: Graceful degradation and fallback mechanisms across all node types
9. **Performance Optimization**: Intelligent subtask routing and efficient database queries
10. **Extensible Architecture**: Easy addition of new node types and execution strategies

The deep research tool will integrate seamlessly with existing static tools, agents, and dynamic tool generation while providing advanced research capabilities and maintaining the static, predictable behavior required for LangChain integration.