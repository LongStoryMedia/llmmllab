"""Test if the tool schema filtering fix works."""

import json
from runner.pipelines.llamacpp.base_llamacpp import BaseLlamaCppPipeline
from models.model_profile import ModelProfile, ModelParameters
from models.complexity_level import ComplexityLevel
from composer.tools.static.web_search_tool import web_search


def test_schema_fix():
    """Test if the schema filtering fix reduces token count dramatically."""
    print("🔍 Testing tool schema filtering fix...")
    
    # Create a pipeline to test the conversion
    profile = ModelProfile(
        user_id="test-user",
        name="test-profile", 
        system_prompt="Test prompt",
        type=0,  # Use integer for type
        model_name="qwen3-30b-a3b-q4-k-m",
        provider="ollama",
        parameters=ModelParameters(
            temperature=0.2,
            max_tokens=8192,
            top_p=0.9,
            repeat_penalty=1.1,
            num_ctx=100000
        ),
        complexity_level=ComplexityLevel.SPECIALIZED,
        context_window=100000,
        supports_tools=True,
        supports_streaming=True
    )
    
    pipeline = BaseLlamaCppPipeline(profile=profile)
    
    # Test with the problematic tools
    tools = [web_search]
    
    print(f"🔧 Testing conversion of {len(tools)} tools...")
    
    # Convert tools using the fixed method
    converted_tools = pipeline._convert_tools_to_openai_format(tools)
    
    if converted_tools:
        total_schema_size = 0
        
        for i, tool in enumerate(converted_tools):
            if 'function' in tool and 'parameters' in tool['function']:
                schema_json = json.dumps(tool['function']['parameters'])
                schema_size = len(schema_json)
                total_schema_size += schema_size
                
                print(f"   Tool {i+1}: {tool['function'].get('name', 'Unknown')}")
                print(f"     Schema size: {schema_size:,} characters")
                print(f"     Estimated tokens: {max(1, schema_size // 3):,}")
                
                if schema_size < 500:
                    print(f"     ✅ FIXED! Schema: {schema_json}")
                elif schema_size < 2000:
                    print(f"     ⚠️  Reasonable size: {schema_json[:200]}...")
                else:
                    print(f"     ❌ Still too large: {schema_json[:200]}...")
        
        total_tokens = max(1, total_schema_size // 3)
        print(f"\n📊 RESULTS:")
        print(f"   Total schema size: {total_schema_size:,} characters")
        print(f"   Estimated tokens: {total_tokens:,}")
        
        if total_tokens < 500:
            print(f"   🎉 SUCCESS! Schema filtering fix works!")
            print(f"   Reduced from ~79K tokens to {total_tokens} tokens!")
        elif total_tokens < 5000:
            print(f"   ✅ Much better but could be optimized further")
        else:
            print(f"   ❌ Still too large - fix didn't work")
    else:
        print("❌ No converted tools returned")


if __name__ == "__main__":
    test_schema_fix()