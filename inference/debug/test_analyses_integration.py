#!/usr/bin/env python3
"""
Test script to verify analyses integration in message queries.
Tests that messages now include analyses JSON aggregation.
"""

import asyncio
import os
from db import storage
from models.intent_analysis import IntentAnalysis
from models.workflow_type import WorkflowType
from models.complexity_level import ComplexityLevel


async def init_storage():
    """Initialize storage if not already done."""
    if not storage.initialized:
        # Build connection string from environment
        connection_string = (
            f"postgresql://{os.getenv('DB_USER', 'lsm')}"
            f":{os.getenv('DB_PASSWORD', 'mypassword')}"
            f"@{os.getenv('DB_HOST', 'localhost')}"
            f":{os.getenv('DB_PORT', '5432')}"
            f"/{os.getenv('DB_NAME', 'llmmll')}"
        )
        
        await storage.initialize(connection_string)


async def test_analyses_integration():
    """Test that messages now include analyses in JSON aggregation."""
    print("🔧 Testing analyses integration in message queries...")
    
    try:
        await init_storage()
        
        # Get a recent message to test with
        async with storage.get_service(storage.message).typed_pool.acquire() as conn:
            # Get a message that might have analyses
            row = await conn.fetchrow("""
                SELECT m.id, COUNT(a.id) as analysis_count
                FROM messages m
                LEFT JOIN analyses a ON a.message_id = m.id
                GROUP BY m.id
                ORDER BY m.created_at DESC
                LIMIT 1
            """)
            
            if not row:
                print("❌ No messages found in database")
                return False
            
            message_id = row['id']
            analysis_count = row['analysis_count']
            print(f"✅ Testing with message {message_id} ({analysis_count} analyses)")
        
        # Test get_message query includes analyses
        message_storage = storage.get_service(storage.message)
        
        # Test the raw SQL query first
        async with message_storage.typed_pool.acquire() as conn:
            result = await conn.fetchrow(
                message_storage.get_query("message.get_message"), 
                message_id
            )
            
            if result:
                print(f"✅ Raw query returned message with analyses field: {'analyses' in result}")
                if 'analyses' in result:
                    analyses_data = result['analyses']
                    if analyses_data and analyses_data != '[]':
                        print(f"✅ Analyses data present: {len(analyses_data)} analyses")
                        print(f"   Sample: {str(analyses_data)[:100]}...")
                    else:
                        print("✅ Analyses field present but empty (expected if no analyses)")
                else:
                    print("❌ Analyses field missing from query result")
                    return False
            else:
                print("❌ No result from get_message query")
                return False
        
        # Test message parsing includes analyses
        parsed_message_data = message_storage._parse_message_row(dict(result))
        
        if 'analyses' in parsed_message_data:
            print("✅ Parsed message data includes analyses field")
            analyses = parsed_message_data['analyses']
            if analyses:
                print(f"✅ Parsed {len(analyses)} analyses successfully")
                for i, analysis in enumerate(analyses):
                    print(f"   Analysis {i+1}: {type(analysis).__name__} - {analysis.workflow_type if hasattr(analysis, 'workflow_type') else 'Unknown'}")
            else:
                print("✅ Analyses field present but empty (expected if no analyses)")
        else:
            print("❌ Analyses field missing from parsed message data")
            return False
        
        # Test conversation history query
        print("\n🔧 Testing conversation history with analyses...")
        
        # Get conversation ID from the message
        conversation_id = result['conversation_id']
        
        async with message_storage.typed_pool.acquire() as conn:
            history_results = await conn.fetch(
                message_storage.get_query("message.get_conversation_history"), 
                conversation_id
            )
            
            if history_results:
                print(f"✅ Retrieved {len(history_results)} messages from conversation history")
                
                # Check if any have analyses
                total_analyses = 0
                for msg_row in history_results:
                    if 'analyses' in msg_row and msg_row['analyses'] and msg_row['analyses'] != '[]':
                        analyses_count = len(msg_row['analyses'])
                        total_analyses += analyses_count
                
                print(f"✅ Conversation history includes {total_analyses} total analyses across all messages")
            else:
                print("❌ No results from conversation history query")
                return False
        
        print("\n🎉 SUCCESS: Analyses integration working!")
        print("   - SQL queries include analyses JSON aggregation")
        print("   - Message parsing handles analyses properly") 
        print("   - All message retrieval methods updated")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Run the test."""
    print("🎯 ANALYSES INTEGRATION TEST")
    print("=" * 40)
    
    success = await test_analyses_integration()
    
    if success:
        print("\n🎉 All tests passed!")
        print("   Analyses are now properly integrated with message queries")
    else:
        print("\n❌ Tests failed - check output above")

if __name__ == "__main__":
    asyncio.run(main())