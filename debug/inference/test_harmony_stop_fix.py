#!/usr/bin/env python3
"""
Quick test to verify the GPT-OSS harmony format stop sequence fix.
This test checks that stop sequences are properly configured to allow <|end|> markers.
"""

import asyncio
import uuid
from runner.pipelines.txt2txt.openai_gpt_oss import OpenAiGptOssPipe
from models import ModelProfile, ModelParameters

async def test_stop_sequence_fix():
    """Test that GPT-OSS pipeline configures stop sequences correctly for harmony format."""
    print("🧪 Testing GPT-OSS stop sequence configuration...")
    
    # Create a test model profile
    profile = ModelProfile(
        id=uuid.uuid4(),
        user_id=str(uuid.uuid4()),
        name="Test GPT-OSS",
        description="Test profile",
        model_name="test-model",
        parameters=ModelParameters(
            temperature=0.7,
            max_tokens=1000
        ),
        system_prompt="Test",
        type=1
    )
    
    # Create GPT-OSS pipeline instance
    pipeline = OpenAiGptOssPipe(profile, None, None, None)
    
    # Test the optimal parameters method
    optimal_params = pipeline._get_optimal_gpt_oss_parameters()
    
    print(f"✅ Optimal parameters retrieved: {optimal_params}")
    print(f"📝 Stop sequences: {optimal_params.get('stop', 'Not found')}")
    
    # Verify stop sequences
    stop_sequences = optimal_params.get('stop', [])
    
    # Check that <|end|> is NOT in the stop sequences
    if '<|end|>' not in stop_sequences:
        print("✅ SUCCESS: <|end|> is correctly removed from stop sequences")
        print(f"   Current stop sequences: {stop_sequences}")
        print("   This allows harmony format channel transitions to work properly!")
        return True
    else:
        print("❌ FAIL: <|end|> is still in stop sequences")
        print(f"   Current stop sequences: {stop_sequences}")
        print("   This will prevent harmony format from working!")
        return False

async def main():
    """Run the stop sequence test."""
    print("🚀 GPT-OSS Harmony Format Stop Sequence Fix Test")
    print("=" * 60)
    
    try:
        success = await test_stop_sequence_fix()
        
        if success:
            print("\n✅ OVERALL SUCCESS: Stop sequence fix is working correctly!")
            print("The GPT-OSS model should now be able to generate <|end|> markers")
            print("and properly transition between harmony format channels.")
        else:
            print("\n❌ OVERALL FAILURE: Stop sequence fix is not working!")
            
    except Exception as e:
        print(f"\n💥 Test failed with error: {e}")
        success = False
    
    print("\n" + "=" * 60)
    return success

if __name__ == "__main__":
    asyncio.run(main())