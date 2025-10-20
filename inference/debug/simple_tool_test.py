"""Simple test to check tool schema sizes in actual llama.cpp prompt generation."""

import json
from langchain_core.tools import tool


# Create a simple tool to test schema size
@tool
def simple_calculator(operation: str, a: float, b: float) -> float:
    """Perform basic mathematical operations.
    
    Args:
        operation: The operation to perform (add, subtract, multiply, divide)
        a: First number
        b: Second number
        
    Returns:
        The result of the calculation
    """
    if operation == "add":
        return a + b
    elif operation == "subtract":
        return a - b
    elif operation == "multiply":
        return a * b
    elif operation == "divide":
        return a / b if b != 0 else 0
    else:
        return 0


@tool 
def web_search(query: str, max_results: int = 10, language: str = "en") -> str:
    """Search the web for information.
    
    Args:
        query: The search query
        max_results: Maximum number of results to return (default: 10)
        language: Language for search results (default: "en")
        
    Returns:
        Search results as formatted text
    """
    return f"Search results for: {query}"


def test_tool_schema_sizes():
    """Test tool schema sizes that could cause token explosion."""
    print("🔍 Testing tool schema sizes...")
    
    # Mock the tool conversion logic from BaseLlamaCppPipeline
    tools = [simple_calculator, web_search]
    converted_tools = []
    
    for tool in tools:
        try:
            tool_dict = {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description or "",
                }
            }
            
            # Extract schema (this is the potential culprit)
            if hasattr(tool, "args_schema") and tool.args_schema:
                try:
                    if hasattr(tool.args_schema, "model_json_schema"):
                        schema = tool.args_schema.model_json_schema()
                    elif hasattr(tool.args_schema, "schema"):
                        schema = tool.args_schema.schema()
                    else:
                        schema = {"type": "object", "properties": {}}
                    
                    tool_dict["function"]["parameters"] = schema
                    
                    # Check schema size
                    schema_json = json.dumps(schema, indent=2)
                    schema_size = len(schema_json)
                    
                    print(f"\n🔧 Tool: {tool.name}")
                    print(f"   Description length: {len(tool.description or '')}")
                    print(f"   Schema size: {schema_size:,} characters")
                    print(f"   Estimated tokens: {max(1, schema_size // 3):,}")
                    
                    if schema_size > 5000:
                        print(f"   ⚠️  LARGE SCHEMA!")
                        print(f"   Schema preview:")
                        print(f"   {schema_json[:1000]}...")
                    
                except Exception as e:
                    print(f"   ❌ Error extracting schema: {e}")
                    tool_dict["function"]["parameters"] = {"type": "object", "properties": {}}
            
            converted_tools.append(tool_dict)
            
        except Exception as e:
            print(f"❌ Error converting tool {tool.name}: {e}")
    
    # Calculate total schema size
    total_schema_size = 0
    for tool in converted_tools:
        if 'function' in tool and 'parameters' in tool['function']:
            schema_json = json.dumps(tool['function']['parameters'])
            total_schema_size += len(schema_json)
    
    total_tokens = max(1, total_schema_size // 3)
    print(f"\n📊 SUMMARY:")
    print(f"   Total tools: {len(converted_tools)}")
    print(f"   Total schema size: {total_schema_size:,} characters")
    print(f"   Estimated tokens: {total_tokens:,}")
    
    if total_tokens > 30000:
        print(f"   🚨 POTENTIAL CULPRIT! This could explain the 41K token explosion!")
    elif total_tokens > 10000:
        print(f"   ⚠️  Significant token usage from tool schemas")
    else:
        print(f"   ✅ Tool schemas seem reasonable")


if __name__ == "__main__":
    test_tool_schema_sizes()