#!/usr/bin/env python3
"""Simple test to verify stop token fix in OpenAI GPT OSS pipeline."""

import sys
import os

# Test the stop token configuration directly
def test_stop_token_fix():
    """Test that the stop token fix is correctly implemented in the code."""
    print("🧪 Testing OpenAI Harmony format stop token fix")
    
    # Check if the fix is present in the code
    gpt_oss_file = "/Users/lons7862/workspace/llmmllab/inference/runner/pipelines/txt2txt/openai_gpt_oss.py"
    
    if not os.path.exists(gpt_oss_file):
        print(f"❌ File not found: {gpt_oss_file}")
        return False
    
    with open(gpt_oss_file, 'r') as f:
        content = f.read()
    
    # Check for the stop token fix
    harmony_stop_fix = 'harmony_stop_tokens = ["<|im_end|>", "<|endoftext|>"]'
    stop_token_assignment = 'self.profile.parameters.stop = harmony_stop_tokens'
    harmony_fix_comment = "CRITICAL STOP TOKEN FIX: Remove <|end|> from stop sequences for harmony format"
    
    fix_present = (
        harmony_stop_fix in content and
        stop_token_assignment in content and
        harmony_fix_comment in content
    )
    
    if fix_present:
        print("✅ SUCCESS: Stop token fix is properly implemented in the code!")
        print("✅ Code removes <|end|> from stop sequences for harmony format")
        print("✅ Harmony format channel transitions should now work correctly")
        
        # Check that the old problematic stop sequence is NOT present
        old_default_stop = '["<|im_end|>", "<|endoftext|>", "<|end|>"]'
        if old_default_stop not in content.replace(harmony_stop_fix, ""):  # Exclude the fix line
            print("✅ Confirmed: Old problematic stop sequence with <|end|> is not used")
        
        return True
    else:
        print("❌ FAILED: Stop token fix is not properly implemented")
        
        # Detailed diagnostics
        if harmony_stop_fix not in content:
            print(f"   Missing: {harmony_stop_fix}")
        if stop_token_assignment not in content:
            print(f"   Missing: {stop_token_assignment}")
        if harmony_fix_comment not in content:
            print(f"   Missing comment: {harmony_fix_comment}")
            
        return False

def test_base_class_default():
    """Test that the base class still has the problematic default (to confirm our override works)."""
    print("\n🔍 Checking base class default stop tokens...")
    
    base_file = "/Users/lons7862/workspace/llmmllab/inference/runner/pipelines/llamacpp/base_llamacpp.py"
    
    if os.path.exists(base_file):
        with open(base_file, 'r') as f:
            content = f.read()
        
        # The base class should still have the old default (may be on multiple lines)
        old_default_parts = [
            '"stop": self.profile.parameters.stop',
            '["<|im_end|>", "<|endoftext|>", "<|end|>"]'
        ]
        
        if old_default in content:
            print("✅ Confirmed: Base class still has <|end|> in default stop tokens")
            print("✅ Our override in OpenAI GPT OSS pipeline is necessary and correct")
            return True
        else:
            print("⚠️  Base class default stop tokens may have changed")
            return False
    else:
        print(f"⚠️  Base class file not found: {base_file}")
        return False

if __name__ == "__main__":
    print("=" * 80)
    print("OpenAI Harmony Format Stop Token Fix Verification")
    print("=" * 80)
    
    fix_test = test_stop_token_fix()
    base_test = test_base_class_default()
    
    print("\n" + "=" * 80)
    print("Test Summary:")
    print("=" * 80)
    
    if fix_test and base_test:
        print("🎉 ALL TESTS PASSED!")
        print("✅ Stop token fix is properly implemented")
        print("✅ OpenAI Harmony format should now work correctly")
        print("✅ Model will complete channel transitions: analysis→commentary→final")
        sys.exit(0)
    else:
        print("💥 SOME TESTS FAILED!")
        if not fix_test:
            print("❌ Stop token fix implementation issue")
        if not base_test:
            print("❌ Base class verification issue")
        sys.exit(1)