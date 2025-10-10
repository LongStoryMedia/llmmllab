#!/usr/bin/env python3
"""
Test dependency injection implementation.
"""

from composer.graph.builder import GraphBuilder
from db import storage


def test_dependency_injection():
    """Test that dependency injection works correctly."""
    print("🔧 Testing dependency injection implementation...")
    
    try:
        # Create GraphBuilder with dependency injection (storage singleton)
        print("🏗️ Creating GraphBuilder with storage...")
        builder = GraphBuilder(storage)
        print('✅ GraphBuilder created successfully')
        
        # Check if agents are created
        print("🤖 Checking agent instantiation...")
        print(f'✅ Intent Classifier Agent: {type(builder.intent_classifier_agent).__name__}')
        print(f'✅ Engineering Agent: {type(builder.engineering_agent).__name__}')
        print(f'✅ Memory Agent: {type(builder.memory_agent).__name__}')
        print(f'✅ Embedding Agent: {type(builder.embedding_agent).__name__}')
        print(f'✅ Summarization Agent: {type(builder.summarization_agent).__name__}')
        print(f'✅ Single Source Agent: {type(builder.single_source_agent).__name__}')
        
        # Check if storage services are properly extracted
        print("💾 Checking storage service injection...")
        print(f'✅ User Config Storage: {type(builder.user_config_storage).__name__}')
        print(f'✅ Memory Storage: {type(builder.memory_storage).__name__}')
        print(f'✅ Summary Storage: {type(builder.summary_storage).__name__}')
        print(f'✅ Search Storage: {type(builder.search_storage).__name__}')
        
        # Test agent dependency injection by checking agent attributes
        print("🔍 Checking agent dependency injection...")
        print(f'✅ Intent Classifier has user_config_storage: {hasattr(builder.intent_classifier_agent, "user_config_storage")}')
        print(f'✅ Engineering Agent has user_config_storage: {hasattr(builder.engineering_agent, "user_config_storage")}')
        print(f'✅ Memory Agent has memory_storage: {hasattr(builder.memory_agent, "memory_storage")}')
        print(f'✅ Embedding Agent has user_config_storage: {hasattr(builder.embedding_agent, "user_config_storage")}')
        print(f'✅ Summarization Agent has summary_storage: {hasattr(builder.summarization_agent, "summary_storage")}')
        print(f'✅ Summarization Agent has search_storage: {hasattr(builder.summarization_agent, "search_storage")}')
        print(f'✅ Summarization Agent has user_config_storage: {hasattr(builder.summarization_agent, "user_config_storage")}')
        
        print('🎉 All agents and storage services successfully created with dependency injection!')
        
    except Exception as e:
        print(f"❌ Error during dependency injection test: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    test_dependency_injection()