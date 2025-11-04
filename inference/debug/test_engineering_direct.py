#!/usr/bin/env python3
"""
Direct test of the engineering agent to verify its behavior.
"""

import asyncio
import sys
import os

# Add the inference directory to the Python path
sys.path.insert(0, '/app')

from composer.agents.engineering_agent import EngineeringAgent
from models import TechnicalDomain, ResponseFormat

async def test_engineering_agent():
    """Test the engineering agent directly."""
    print("🔧 Testing Engineering Agent Directly")
    
    try:
        # Initialize the engineering agent
        agent = EngineeringAgent()
        
        # Test query
        query = "Help me create a FastAPI application with authentication"
        user_id = "test_user"
        domain = TechnicalDomain.SOFTWARE_DEVELOPMENT
        response_format = ResponseFormat.CODE_SOLUTION
        
        print(f"Query: {query}")
        print(f"Domain: {domain}")
        print(f"Format: {response_format}")
        print("\n" + "="*50)
        
        # Generate response
        response = await agent.generate_technical_response(
            query=query,
            user_id=user_id,
            domain=domain,
            response_format=response_format
        )
        
        print("🤖 Engineering Agent Response:")
        print(response)
        print("\n" + "="*50)
        
        # Check if response is generic or specific
        if "Hello! I'm here to help with engineering tasks" in response:
            print("❌ ISSUE: Agent gave generic greeting instead of FastAPI response")
        elif "FastAPI" in response and "authentication" in response:
            print("✅ SUCCESS: Agent gave specific FastAPI authentication response")
        else:
            print("⚠️  UNCLEAR: Response doesn't match expected patterns")
            
    except Exception as e:
        print(f"❌ Error testing engineering agent: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_engineering_agent())