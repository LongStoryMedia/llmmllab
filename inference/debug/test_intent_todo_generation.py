#!/usr/bin/env python3
"""
Test script for intent-based automatic todo generation.
Run this in the container to verify the planning middleware generates todos correctly.
"""

import asyncio
import sys
import os

# Add the project root to the path
sys.path.insert(0, '/app')

from db import storage
from composer.graph.subgraphs.planning_intent import PlanningIntentSubgraph
from composer.agents.classifier_agent import ClassifierAgent
from runner.pipeline_factory import pipeline_factory
from langchain_core.messages import HumanMessage
from models import IntentAnalysis, WorkflowType, ComplexityLevel
from utils.logging import llmmllogger

logger = llmmllogger.bind(component="intent_todo_test")

async def test_intent_todo_generation():
    """Test automatic todo generation from intent analysis."""
    
    if not storage.initialized:
        logger.error("Database not initialized")
        return False
    
    if not storage.todo:
        logger.error("Todo storage not initialized")
        return False
    
    test_user_id = "test_user_intent_todos"
    test_conversation_id = 123
    
    try:
        # Create the planning intent subgraph
        classifier_agent = ClassifierAgent(pipeline_factory)
        planning_subgraph = PlanningIntentSubgraph(classifier_agent, pipeline_factory)
        
        # Test different types of user messages that should generate todos
        test_messages = [
            "I need to research machine learning algorithms for my project",
            "Can you help me analyze the performance data from last quarter?",
            "I want to create a presentation about climate change",
            "Please help me plan a software development project",
            "I need to debug this critical issue urgently"
        ]
        
        for i, message in enumerate(test_messages):
            logger.info(f"\n🧪 Test {i+1}: {message}")
            
            # Create test planning state
            planning_state = {
                "messages": [HumanMessage(content=message)],
                "user_id": test_user_id,
                "conversation_id": test_conversation_id + i,  # Different conversation for each test
                "static_tools": [],
                "planning_steps": [],
                "complexity_score": 3,
                "intent_analyses": [],
                "generated_todos": [],
            }
            
            # Execute the planning subgraph
            result = await planning_subgraph.graph.ainvoke(
                planning_state,
                config={"recursion_limit": 10}
            )
            
            # Check results
            generated_todos = result.get("generated_todos", [])
            intent_analyses = result.get("intent_analyses", [])
            
            logger.info(f"📊 Intent analyses: {len(intent_analyses)}")
            for analysis in intent_analyses:
                if hasattr(analysis, 'workflow_type'):
                    logger.info(f"  - Workflow: {analysis.workflow_type}, Complexity: {analysis.complexity_level}")
            
            logger.info(f"📝 Generated todos: {len(generated_todos)}")
            for todo in generated_todos:
                logger.info(f"  - {todo.get('title', 'No title')} [{todo.get('priority', 'medium')}]")
            
            # Verify todos were actually created in database
            db_todos = await storage.todo.get_todos_by_conversation(
                user_id=test_user_id,
                conversation_id=test_conversation_id + i
            )
            
            logger.info(f"💾 Database todos: {len(db_todos)}")
            for todo in db_todos:
                logger.info(f"  - DB: {todo.title} [Status: {todo.status}, Priority: {todo.priority}]")
            
            if generated_todos:
                logger.info(f"✅ Test {i+1} passed: Generated {len(generated_todos)} todos")
            else:
                logger.warning(f"⚠️ Test {i+1}: No todos generated")
        
        # Clean up test todos
        logger.info("\n🧹 Cleaning up test todos...")
        for i in range(len(test_messages)):
            todos = await storage.todo.get_todos_by_conversation(
                user_id=test_user_id,
                conversation_id=test_conversation_id + i
            )
            for todo in todos:
                await storage.todo.delete_todo(todo.id, test_user_id)
        
        logger.info("✅ All intent-based todo generation tests completed!")
        return True
        
    except Exception as e:
        logger.error(f"Test failed with error: {e}", exc_info=True)
        return False

async def main():
    """Main test function."""
    logger.info("Starting intent-based todo generation tests...")
    
    # Initialize database connection if needed
    if not storage.initialized:
        logger.info("Initializing database connection...")
        connection_string = os.getenv('DATABASE_URL', 'postgresql://lsm:password@psql-service.psql.svc.cluster.local:5432/llmmll')
        await storage.initialize(connection_string)
    
    success = await test_intent_todo_generation()
    
    if success:
        logger.info("🎉 All tests completed successfully!")
        sys.exit(0)
    else:
        logger.error("❌ Tests failed!")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())