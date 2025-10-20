"""Test to capture what tools are actually being bound in composer workflow."""

import json
import asyncio
from composer.graph.core import ComposerCore  
from models.chat_req import ChatReq
from models.conversation_ctx import ConversationCtx


async def test_composer_tool_binding():
    """Test what tools actually get bound in composer workflow."""
    print("🔍 Testing composer tool binding...")
    
    try:
        # Create a real composer instance
        composer = ComposerCore()
        
        # Create a minimal chat request
        conversation = ConversationCtx(
            id="test-conv",
            user_id="test-user",
            messages=[],
            metadata={}
        )
        
        request = ChatReq(
            user_id="test-user",
            conversation=conversation,
            stream=False,
            model="qwen3-30b-a3b-q4-k-m"  # The model that's failing
        )
        
        # Create initial state  
        state = composer._create_initial_state(request)
        print(f"📋 Created initial state for user: {state.user_id}")
        
        # Run through the workflow to see what tools get bound
        print("🔄 Running composer workflow...")
        
        # Execute the workflow
        async for event in composer.astream(request):
            if hasattr(event, 'event') and event.event == 'chunk':
                continue  # Skip streaming chunks
                
            print(f"📊 Event type: {type(event)}")
            
            # Check if we have tools in the event
            if hasattr(event, 'tools') and event.tools:
                print(f"🔧 Found {len(event.tools)} tools!")
                
                total_schema_size = 0
                for i, tool in enumerate(event.tools):
                    # Try to extract schema like BaseLlamaCppPipeline does
                    if hasattr(tool, 'args_schema') and tool.args_schema:
                        try:
                            if hasattr(tool.args_schema, 'model_json_schema'):
                                schema = tool.args_schema.model_json_schema()
                            elif hasattr(tool.args_schema, 'schema'):
                                schema = tool.args_schema.schema()
                            else:
                                schema = {}
                                
                            schema_json = json.dumps(schema)
                            schema_size = len(schema_json)
                            total_schema_size += schema_size
                            
                            print(f"   Tool {i+1}: {getattr(tool, 'name', 'Unknown')}")
                            print(f"     Schema size: {schema_size:,} characters")
                            print(f"     Estimated tokens: {max(1, schema_size // 3):,}")
                            
                            if schema_size > 10000:
                                print(f"     ⚠️  LARGE SCHEMA! First 200 chars:")
                                print(f"     {schema_json[:200]}...")
                                
                        except Exception as e:
                            print(f"   Tool {i+1}: {getattr(tool, 'name', 'Unknown')} - Error: {e}")
                    else:
                        print(f"   Tool {i+1}: {getattr(tool, 'name', 'Unknown')} - No schema")
                
                total_tokens = max(1, total_schema_size // 3)
                print(f"\n📈 TOTAL TOOL SCHEMA SIZE: {total_schema_size:,} characters")
                print(f"🎯 ESTIMATED TOOL TOKENS: {total_tokens:,}")
                
                if total_tokens > 35000:
                    print("🚨 FOUND THE CULPRIT! Tool schemas are massive!")
                    print("   This explains the 41K+ token explosion!")
                    break
                elif total_tokens > 10000:
                    print("⚠️  Significant tool token usage detected")
                else:
                    print("✅ Tool token usage seems reasonable")
                    
    except Exception as e:
        print(f"❌ Error in composer test: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test_composer_tool_binding())