"""Debug specific tool schemas to find the massive one."""

import json
from composer.tools.static.web_search_tool import web_search
from composer.tools.static import memory_retrieval, summarization


def test_specific_tool_schemas():
    """Test the specific tools we know are being loaded."""
    print("🔍 Testing specific static tool schemas...")
    
    tools_to_test = [
        ("web_search", web_search),
        ("memory_retrieval", memory_retrieval), 
        ("summarization", summarization),
    ]
    
    total_schema_size = 0
    
    for tool_name, tool in tools_to_test:
        try:
            print(f"\n🔧 Testing tool: {tool_name}")
            print(f"   Name: {getattr(tool, 'name', 'Unknown')}")
            print(f"   Description length: {len(getattr(tool, 'description', ''))}")
            
            # Check if it has args_schema
            if hasattr(tool, 'args_schema') and tool.args_schema:
                try:
                    if hasattr(tool.args_schema, 'model_json_schema'):
                        schema = tool.args_schema.model_json_schema()
                    elif hasattr(tool.args_schema, 'schema'):
                        schema = tool.args_schema.schema()
                    else:
                        schema = {"type": "object", "properties": {}}
                    
                    schema_json = json.dumps(schema, indent=2)
                    schema_size = len(schema_json)
                    total_schema_size += schema_size
                    
                    print(f"   Schema size: {schema_size:,} characters")
                    print(f"   Estimated tokens: {max(1, schema_size // 3):,}")
                    
                    if schema_size > 10000:
                        print(f"   🚨 MASSIVE SCHEMA FOUND!")
                        print(f"   First 1000 characters:")
                        print(f"   {schema_json[:1000]}...")
                        print(f"   ...")
                        print(f"   Last 1000 characters:")
                        print(f"   ...{schema_json[-1000:]}")
                    elif schema_size > 2000:
                        print(f"   ⚠️  Large schema")
                        print(f"   First 500 characters:")
                        print(f"   {schema_json[:500]}...")
                    else:
                        print(f"   ✅ Reasonable schema size")
                        print(f"   Full schema: {schema_json}")
                        
                except Exception as e:
                    print(f"   ❌ Error extracting schema: {e}")
                    # Try to see what type of args_schema it is
                    print(f"   args_schema type: {type(tool.args_schema)}")
                    print(f"   args_schema dir: {dir(tool.args_schema)}")
            else:
                print(f"   ℹ️  No args_schema found")
                
        except Exception as e:
            print(f"❌ Error testing tool {tool_name}: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n📊 FINAL SUMMARY:")
    print(f"   Total schema size: {total_schema_size:,} characters")
    print(f"   Estimated total tokens: {max(1, total_schema_size // 3):,}")
    
    if total_schema_size > 100000:
        print(f"   🚨 MASSIVE SCHEMAS! This is the source of the 41K+ token explosion!")
    elif total_schema_size > 30000:
        print(f"   ⚠️  Significant schema overhead")
    else:
        print(f"   ✅ Schema sizes seem reasonable")


if __name__ == "__main__":
    test_specific_tool_schemas()