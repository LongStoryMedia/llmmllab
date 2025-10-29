#!/usr/bin/env python3
"""
Test tool call persistence to debug schema alignment and storage issues.
"""

import asyncio
import json
from datetime import datetime
from db import storage
from models import ToolCall, ResourceUsage
from utils.logging import llmmllogger

logger = llmmllogger.bind(component="tool_call_test")


async def test_tool_call_crud():
    """Test tool call create, read, update, delete operations."""
    
    try:
        # Initialize storage
        await storage.initialize("postgresql://lsm:7cb9c812e384e16c911a72f1066517d205e8641b78edb3b1b3c78d0c351b1885@192.168.0.71:32345/llmmll")
        
        logger.info("🔧 Testing tool call CRUD operations")
        
        # Create test tool call - use message from conversation 717
        test_tool_call = ToolCall(
            message_id=3626,  # Use existing message ID from conversation 717
            name="test_tool",
            execution_id="test_exec_123",
            success=True,
            args={"query": "test query", "max_results": 5},
            result_data={"result": "test result", "count": 1},
            error_message=None,
            execution_time_ms=150.5,
            resource_usage=ResourceUsage(
                tokens_used=100,
                memory_mb=50.0,
                cpu_time_ms=75.25
            )
        )
        
        print("\n" + "="*80)
        print("🔧 TOOL CALL PERSISTENCE TEST")
        print("="*80)
        
        # Test 1: Add tool call
        logger.info("📝 Adding test tool call...")
        tool_call_id = await storage.get_service(storage.tool_call).add_tool_call(test_tool_call)
        
        if tool_call_id:
            print(f"✅ Tool call created with ID: {tool_call_id}")
        else:
            print("❌ Failed to create tool call")
            return {"success": False, "error": "Failed to create tool call"}
        
        # Test 2: Retrieve tool calls by message
        logger.info("📖 Retrieving tool calls by message...")
        retrieved_calls = await storage.get_service(storage.tool_call).get_tool_calls_by_message(3626)
        
        print(f"📋 Retrieved {len(retrieved_calls)} tool calls")
        
        if retrieved_calls:
            call = retrieved_calls[0]
            print(f"   Name: {call.name}")
            print(f"   Execution ID: {call.execution_id}")
            print(f"   Success: {call.success}")
            print(f"   Args: {call.args}")
            print(f"   Result: {call.result_data}")
            print(f"   Resource Usage: {call.resource_usage}")
        
        # Test 3: Schema validation
        schema_issues = []
        
        if retrieved_calls:
            call = retrieved_calls[0]
            
            # Check field mapping
            if not hasattr(call, 'name'):
                schema_issues.append("Missing 'name' field")
            
            if not isinstance(call.args, dict):
                schema_issues.append(f"args is {type(call.args)}, expected dict")
                
            if call.result_data and not isinstance(call.result_data, dict):
                schema_issues.append(f"result_data is {type(call.result_data)}, expected dict")
                
            if call.resource_usage and not isinstance(call.resource_usage, ResourceUsage):
                schema_issues.append(f"resource_usage is {type(call.resource_usage)}, expected ResourceUsage")
        
        if schema_issues:
            print("\n⚠️ SCHEMA ISSUES:")
            for issue in schema_issues:
                print(f"   {issue}")
        else:
            print("\n✅ Schema validation passed")
        
        # Test 4: Clean up
        logger.info("🧹 Cleaning up test data...")
        # Note: We'd normally clean up, but for debugging let's leave it
        
        print("="*80)
        
        return {
            "success": True,
            "tool_call_id": tool_call_id,
            "retrieved_count": len(retrieved_calls),
            "schema_issues": schema_issues
        }
        
    except Exception as e:
        logger.error(f"❌ Tool call test failed: {e}")
        print(f"\n❌ ERROR: {e}")
        return {"success": False, "error": str(e)}


async def main():
    """Main function to run the test."""
    result = await test_tool_call_crud()
    print(f"\n🎯 Test Result: {json.dumps(result, indent=2, default=str)}")


if __name__ == "__main__":
    asyncio.run(main())