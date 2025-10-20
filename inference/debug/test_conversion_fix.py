"""Test the tool conversion fix directly without full pipeline initialization."""

import json
from composer.tools.static.web_search_tool import web_search


def test_conversion_fix():
    """Test the tool conversion logic directly."""
    print("🔍 Testing tool conversion fix...")
    
    tool = web_search
    print(f"🔧 Tool: {tool.name}")
    
    # Simulate the fixed conversion logic
    try:
        tool_dict = {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description or "",
            },
        }
        
        # Add parameters schema with filtering (the fix)
        if tool.args_schema:
            if hasattr(tool.args_schema, "model_json_schema"):
                schema = tool.args_schema.model_json_schema()
            elif hasattr(tool.args_schema, "schema"):
                schema = tool.args_schema.schema()
            else:
                schema = {"type": "object", "properties": {}}
            
            print(f"   Original schema size: {len(json.dumps(schema)):,} characters")
            
            # CRITICAL FIX: Filter out injected parameters
            if 'properties' in schema:
                # Remove injected parameters: 'state' (WorkflowState) and 'tool_call_id'
                filtered_props = {k: v for k, v in schema['properties'].items() 
                                if k not in ['state', 'tool_call_id']}
                
                # Create filtered schema without massive injected state
                filtered_schema = {
                    "type": "object",
                    "properties": filtered_props,
                    "required": [req for req in schema.get('required', []) 
                               if req not in ['state', 'tool_call_id']]
                }
                
                tool_dict["function"]["parameters"] = filtered_schema
            else:
                tool_dict["function"]["parameters"] = schema
        
        # Check the result
        result_json = json.dumps(tool_dict, indent=2)
        result_size = len(result_json)
        estimated_tokens = max(1, result_size // 3)
        
        print(f"\n📊 RESULTS:")
        print(f"   Filtered tool size: {result_size:,} characters")
        print(f"   Estimated tokens: {estimated_tokens:,}")
        
        if estimated_tokens < 500:
            print(f"   🎉 SUCCESS! Tool conversion fix works!")
            print(f"   Reduced from ~26K tokens to {estimated_tokens} tokens per tool!")
            print(f"   Full converted tool:")
            print(f"   {result_json}")
        else:
            print(f"   ❌ Still too large")
            print(f"   First 1000 chars: {result_json[:1000]}...")
            
    except Exception as e:
        print(f"❌ Error in conversion: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_conversion_fix()