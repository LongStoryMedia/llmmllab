#!/usr/bin/env python3
"""
Test script to verify that the refactored storage architecture 
loads correctly without database connectivity.
"""

import sys
sys.path.append('/app')

from utils.logging import llmmllogger

logger = llmmllogger.bind(component="test_storage_refactor")

def test_storage_imports():
    """Test that all storage modules import correctly."""
    logger.info("Testing storage module imports...")
    
    try:
        # Test message content storage
        from db.message_content_storage import MessageContentStorage
        logger.info("✅ MessageContentStorage import successful")
        
        # Test updated thought storage
        from db.thought_storage import ThoughtStorage
        logger.info("✅ ThoughtStorage import successful")
        
        # Test updated tool call storage
        from db.tool_call_storage import ToolCallStorage
        logger.info("✅ ToolCallStorage import successful")
        
        # Test updated message storage
        from db.message_storage import MessageStorage
        logger.info("✅ MessageStorage import successful")
        
        # Test database module
        from db import storage
        logger.info("✅ Database storage singleton import successful")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Import failed: {e}")
        return False

def test_storage_class_structure():
    """Test that storage classes have expected method signatures."""
    logger.info("Testing storage class method signatures...")
    
    try:
        from db.message_content_storage import MessageContentStorage
        from db.thought_storage import ThoughtStorage
        from db.tool_call_storage import ToolCallStorage
        from db.message_storage import MessageStorage
        
        # Create mock instances for testing (without actual DB connection)
        class MockPool:
            pass
            
        def mock_get_query(query_name):
            return f"SELECT * FROM {query_name}"
        
        # Test MessageContentStorage
        content_storage = MessageContentStorage(MockPool(), mock_get_query)
        assert hasattr(content_storage, 'add_content'), "MessageContentStorage missing add_content method"
        assert hasattr(content_storage, 'get_contents_by_message'), "MessageContentStorage missing get_contents_by_message method"
        logger.info("✅ MessageContentStorage methods present")
        
        # Test ThoughtStorage
        thought_storage = ThoughtStorage(MockPool(), mock_get_query)
        assert hasattr(thought_storage, 'add_thought'), "ThoughtStorage missing add_thought method"
        logger.info("✅ ThoughtStorage methods present")
        
        # Test ToolCallStorage
        tool_call_storage = ToolCallStorage(MockPool(), mock_get_query)
        assert hasattr(tool_call_storage, 'add_tool_call'), "ToolCallStorage missing add_tool_call method"
        logger.info("✅ ToolCallStorage methods present")
        
        # Test MessageStorage
        message_storage = MessageStorage(MockPool(), mock_get_query)
        assert hasattr(message_storage, 'set_storage_dependencies'), "MessageStorage missing set_storage_dependencies method"
        assert hasattr(message_storage, 'add_message'), "MessageStorage missing add_message method"
        logger.info("✅ MessageStorage methods present")
        
        # Test dependency setting
        message_storage.set_storage_dependencies(
            thought_storage,
            tool_call_storage, 
            content_storage,
            None  # analysis_storage
        )
        assert message_storage.thought_storage is thought_storage, "ThoughtStorage dependency not set correctly"
        assert message_storage.tool_call_storage is tool_call_storage, "ToolCallStorage dependency not set correctly"
        assert message_storage.message_content_storage is content_storage, "MessageContentStorage dependency not set correctly"
        logger.info("✅ Storage dependencies set correctly")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Method signature test failed: {e}")
        return False

def main():
    """Run all tests."""
    logger.info("🚀 Starting storage refactoring tests...")
    
    all_tests_passed = True
    
    # Test imports
    if not test_storage_imports():
        all_tests_passed = False
    
    # Test class structure 
    if not test_storage_class_structure():
        all_tests_passed = False
    
    if all_tests_passed:
        logger.info("🎉 All storage refactoring tests passed!")
        print("✅ Storage architecture refactoring successful")
    else:
        logger.error("❌ Some tests failed")
        print("❌ Storage architecture has issues")
        sys.exit(1)

if __name__ == "__main__":
    main()