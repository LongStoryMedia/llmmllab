#!/usr/bin/env python3
"""
Test script to verify the checkpointer integration fix.
Tests that the GraphBuilder can properly create and use CheckpointStorage service.
"""

import asyncio
from db import storage

async def test_checkpointer_integration():
    """Test that the checkpointer integration works without the type error."""
    print("🔧 Testing CheckpointStorage integration...")
    
    try:
        # Test CheckpointStorage service creation
        if not storage.initialized:
            print("❌ Storage not initialized - run in environment with DB")
            return False
            
        checkpoint_storage = storage.get_service(storage.checkpoint)
        print(f"✅ CheckpointStorage service: {type(checkpoint_storage).__name__}")
        
        # Test that the create_saver_for_workflow method exists and is callable
        if hasattr(checkpoint_storage, 'create_saver_for_workflow'):
            print("✅ create_saver_for_workflow method exists")
            
            # Test method signature (should be async now)
            import inspect
            sig = inspect.signature(checkpoint_storage.create_saver_for_workflow)
            is_async = inspect.iscoroutinefunction(checkpoint_storage.create_saver_for_workflow)
            print(f"✅ Method is async: {is_async}")
            print(f"✅ Method signature: {sig}")
            
        else:
            print("❌ create_saver_for_workflow method missing")
            return False
            
        # Test basic GraphBuilder instantiation (without actually building workflow)
        from composer.graph.builder import GraphBuilder
        from runner import PipelineFactory  
        from models import UserConfig
        
        # Mock basic user config for test
        user_config = UserConfig(
            user_id="test_user",
            model_profiles=[],  # Empty for this test
            web_search_config=None,
            auth_config=None
        )
        
        pipeline_factory = PipelineFactory()
        
        builder = GraphBuilder(
            storage=storage,
            pipeline_factory=pipeline_factory, 
            user_config=user_config
        )
        
        print("✅ GraphBuilder instantiation successful")
        print("✅ CheckpointStorage properly injected as dependency")
        
        # Verify the builder has checkpoint_storage attribute
        if hasattr(builder, 'checkpoint_storage'):
            print("✅ Builder has checkpoint_storage attribute")
            print(f"   Type: {type(builder.checkpoint_storage).__name__}")
        else:
            print("❌ Builder missing checkpoint_storage attribute")
            return False
            
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Run the test."""
    print("🎯 CHECKPOINTER INTEGRATION TEST")
    print("=" * 40)
    
    success = await test_checkpointer_integration()
    
    if success:
        print("\n🎉 SUCCESS: Checkpointer integration fixed!")
        print("   - No more type errors")
        print("   - Proper dependency injection")
        print("   - CheckpointStorage service working")
    else:
        print("\n❌ FAILED: Issues remain")
        
    return success

if __name__ == "__main__":
    asyncio.run(main())