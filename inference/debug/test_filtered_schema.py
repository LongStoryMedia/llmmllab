"""Test filtered tool schema extraction to fix the massive token explosion."""

import json
from composer.tools.static.web_search_tool import web_search


def test_filtered_schema():
    """Test extracting filtered schema without injected parameters."""
    print("🔍 Testing filtered schema extraction...")
    
    tool = web_search
    print(f"🔧 Tool: {tool.name}")
    
    # Check if LangChain has a method to get schema without injected parameters
    print(f"   Tool attributes: {[attr for attr in dir(tool) if not attr.startswith('_')]}")
    
    # Try different schema extraction methods
    try:
        # Check if there's a way to get the "input schema" without injected params
        if hasattr(tool, 'get_input_schema'):
            input_schema = tool.get_input_schema()
            print(f"   get_input_schema(): {type(input_schema)}")
            if hasattr(input_schema, 'model_json_schema'):
                filtered_schema = input_schema.model_json_schema()
                filtered_json = json.dumps(filtered_schema, indent=2)
                print(f"   Filtered schema size: {len(filtered_json):,} characters")
                print(f"   Filtered tokens: {max(1, len(filtered_json) // 3):,}")
                
                if len(filtered_json) < 1000:
                    print(f"   ✅ Filtered schema (reasonable size):")
                    print(f"   {filtered_json}")
                else:
                    print(f"   ⚠️  Still large - first 500 chars:")
                    print(f"   {filtered_json[:500]}...")
        
        # Check args_schema properties
        if hasattr(tool, 'args_schema'):
            print(f"   args_schema type: {type(tool.args_schema)}")
            print(f"   args_schema attributes: {[attr for attr in dir(tool.args_schema) if not attr.startswith('_')]}")
            
            # Check if we can inspect the fields
            if hasattr(tool.args_schema, '__fields__'):
                fields = tool.args_schema.__fields__
                print(f"   Fields: {list(fields.keys())}")
                
                # Look for injected fields
                for field_name, field_info in fields.items():
                    print(f"     {field_name}: {type(field_info)}")
                    if hasattr(field_info, 'annotation'):
                        print(f"       annotation: {field_info.annotation}")
                        
        # Try to manually create a filtered schema
        if hasattr(tool.args_schema, 'model_json_schema'):
            full_schema = tool.args_schema.model_json_schema()
            
            # Remove the massive WorkflowState and tool_call_id from required and properties
            if 'properties' in full_schema:
                filtered_props = {k: v for k, v in full_schema['properties'].items() 
                                if k not in ['state', 'tool_call_id']}
                
                filtered_schema = {
                    "type": "object",
                    "properties": filtered_props,
                    "required": [req for req in full_schema.get('required', []) 
                               if req not in ['state', 'tool_call_id']]
                }
                
                filtered_json = json.dumps(filtered_schema, indent=2)
                print(f"\n   🎯 MANUALLY FILTERED SCHEMA:")
                print(f"   Size: {len(filtered_json):,} characters")
                print(f"   Tokens: {max(1, len(filtered_json) // 3):,}")
                print(f"   Schema: {filtered_json}")
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_filtered_schema()