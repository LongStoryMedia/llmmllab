"""
Integration test to verify that the server models API endpoint properly loads models
without validation errors.

This test validates the fix for UI model selector loading errors.
"""

import pytest
from unittest.mock import MagicMock
from server.routers.model import list_models


@pytest.mark.asyncio
async def test_models_api_loads_without_validation_errors():
    """Test that the /models/ API endpoint loads all models without Pydantic validation errors."""
    
    # Mock request object
    request = MagicMock()
    request.headers = {'Authorization': 'Bearer test-token'}
    
    # Call the API endpoint
    models = await list_models(request)
    
    # Verify models were loaded
    assert len(models) > 0, "No models were loaded from API"
    
    # Verify all models have required fields that were causing validation errors
    for model in models:
        assert hasattr(model, 'details'), f"Model {model.id} missing details"
        
        # These were the specific fields causing validation errors
        assert model.details.size is not None, f"Model {model.id} missing details.size"
        assert model.details.original_ctx is not None, f"Model {model.id} missing details.original_ctx"
        assert model.details.size >= 0, f"Model {model.id} has negative size"
        assert model.details.original_ctx > 0, f"Model {model.id} has invalid original_ctx"
        
        # Verify other critical fields are present
        assert model.id, f"Model missing id"
        assert model.name, f"Model missing name"
        assert model.provider, f"Model missing provider"
        
    print(f"✅ Successfully loaded {len(models)} models through API without validation errors")


@pytest.mark.asyncio 
async def test_models_api_handles_missing_fields_gracefully():
    """Test that the API handles models with missing fields gracefully using ModelLoader defaults."""
    
    # Mock request object
    request = MagicMock()
    request.headers = {'Authorization': 'Bearer test-token'}
    
    # Call the API endpoint
    models = await list_models(request)
    
    # Look for models that would have had missing fields (like FLUX models)
    flux_models = [m for m in models if 'flux' in m.id.lower()]
    
    if flux_models:
        for flux_model in flux_models:
            # These models had missing size/original_ctx in config
            # ModelLoader should provide defaults
            assert flux_model.details.size == 0, f"FLUX model {flux_model.id} should have default size of 0"
            assert flux_model.details.original_ctx == 4096, f"FLUX model {flux_model.id} should have default original_ctx of 4096"
            
            print(f"✅ FLUX model {flux_model.id} loaded with appropriate defaults")
    
    print(f"✅ All models with missing fields handled gracefully")


if __name__ == "__main__":
    import asyncio
    
    async def run_tests():
        print("Running integration tests for models API endpoint...")
        
        try:
            await test_models_api_loads_without_validation_errors()
            await test_models_api_handles_missing_fields_gracefully()
            print("\n🎉 All integration tests passed!")
        except Exception as e:
            print(f"❌ Test failed: {e}")
            import traceback
            traceback.print_exc()
    
    asyncio.run(run_tests())