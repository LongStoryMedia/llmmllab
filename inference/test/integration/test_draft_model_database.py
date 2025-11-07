"""
Integration test to verify that draft_model field is properly saved to and retrieved from database.
"""

import asyncio
import uuid
from datetime import datetime
from models.model_profile import ModelProfile, ModelParameters
from models.model_profile_type import ModelProfileType
from db import storage


async def test_draft_model_database_integration():
    """Test that draft_model field is properly handled in database operations."""
    
    test_user_id = "test_draft_model_user"
    
    try:
        # Create a test profile with draft model
        test_profile = ModelProfile(
            id=uuid.uuid4(),
            user_id=test_user_id,
            name="Test Profile with Draft Model",
            description="Testing draft model database functionality",
            model_name="qwen3-vl-32b-thinking",
            draft_model="qwen3-4b",  # This is what we're testing
            parameters=ModelParameters(),
            system_prompt="Test system prompt",
            type=ModelProfileType.Primary,
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        
        print(f"📝 Creating test profile with draft_model: '{test_profile.draft_model}'")
        
        # Test CREATE operation
        saved_profile = await storage.get_service(storage.model_profile).create_model_profile(test_profile)
        print(f"✅ Successfully created profile: {saved_profile.name}")
        print(f"✅ Created draft_model: '{saved_profile.draft_model}'")
        
        # Test READ operation
        retrieved_profile = await storage.get_service(storage.model_profile).get_model_profile_by_id(
            saved_profile.id, test_user_id
        )
        
        if not retrieved_profile:
            print("❌ FAILED: Could not retrieve saved profile")
            return False
            
        print(f"✅ Successfully retrieved profile: {retrieved_profile.name}")
        print(f"✅ Retrieved draft_model: '{retrieved_profile.draft_model}'")
        
        # Verify the draft_model was saved correctly
        if retrieved_profile.draft_model != "qwen3-4b":
            print(f"❌ FAILED: Expected 'qwen3-4b', got '{retrieved_profile.draft_model}'")
            return False
            
        # Test UPDATE operation
        retrieved_profile.draft_model = "nomic-embed-text-v2"  # Change to a different model
        updated_profile = await storage.get_service(storage.model_profile).update_model_profile(retrieved_profile)
        print(f"✅ Successfully updated draft_model to: '{updated_profile.draft_model}'")
        
        # Test READ after UPDATE
        final_profile = await storage.get_service(storage.model_profile).get_model_profile_by_id(
            updated_profile.id, test_user_id
        )
        
        if final_profile.draft_model != "nomic-embed-text-v2":
            print(f"❌ FAILED: Update test - Expected 'nomic-embed-text-v2', got '{final_profile.draft_model}'")
            return False
            
        print(f"✅ Update verification successful: '{final_profile.draft_model}'")
        
        # Test LIST operation (ensure draft_model is included in list results)
        all_profiles = await storage.get_service(storage.model_profile).list_model_profiles_by_user(test_user_id)
        test_profile_in_list = None
        
        for profile in all_profiles:
            if profile.id == final_profile.id:
                test_profile_in_list = profile
                break
                
        if not test_profile_in_list:
            print("❌ FAILED: Test profile not found in list")
            return False
            
        if test_profile_in_list.draft_model != "nomic-embed-text-v2":
            print(f"❌ FAILED: List test - Expected 'nomic-embed-text-v2', got '{test_profile_in_list.draft_model}'")
            return False
            
        print(f"✅ List verification successful: '{test_profile_in_list.draft_model}'")
        
        # Cleanup - delete test profile
        await storage.get_service(storage.model_profile).delete_model_profile(
            final_profile.id, test_user_id
        )
        print("✅ Test profile cleaned up successfully")
        
        print("\n🎉 ALL TESTS PASSED! draft_model field is working correctly in database operations")
        return True
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("Testing draft_model database integration...")
    result = asyncio.run(test_draft_model_database_integration())
    print(f"\nTest result: {'✅ PASSED' if result else '❌ FAILED'}")