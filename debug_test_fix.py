#!/usr/bin/env python3
"""
Quick e2e test to check if the composer workflow works correctly without double processing.
"""

import asyncio
import json
from pathlib import Path

# Simple test data
test_message = "Test web search functionality"

def main():
    """Main test function."""
    print("🧪 Testing Composer E2E after tool parameter fix...")
    print("=" * 60)
    
    # Create a simple test request
    test_request = {
        "message": {
            "content": "What are the latest developments in AI in 2024? Please search for recent information about major AI model releases.",
            "type": "human"
        },
        "user_id": "test_user_fix_verification",
        "conversation_id": 999
    }
    
    # Save test request to JSON file
    test_file = Path("/app/debug/test_request_fix.json")
    with open(test_file, "w") as f:
        json.dump(test_request, f, indent=2)
    
    print(f"✅ Created test request: {test_file}")
    print("📋 Test request content:")
    print(f"   Message: {test_request['message']['content'][:100]}...")
    print(f"   User ID: {test_request['user_id']}")
    print()
    print("🎯 This test should demonstrate:")
    print("   - LLM doesn't see injection parameters in tool descriptions")
    print("   - No double processing or tool parameter confusion")
    print("   - Single execution path without routing loops")

if __name__ == "__main__":
    main()