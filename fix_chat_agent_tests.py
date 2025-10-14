#!/usr/bin/env python3
"""
Script to bulk fix ChatAgent test instantiations to use new constructor signature.
"""

import re
import os

def fix_chat_agent_tests():
    test_file = "/Users/lons7862/workspace/llmmllab/inference/test/unit/test_chat_agent.py"
    
    with open(test_file, 'r') as f:
        content = f.read()
    
    # Pattern 1: ChatAgent(pipeline_factory, profile, stream=True)
    content = re.sub(
        r'agent = ChatAgent\(pipeline_factory, profile, stream=True\)',
        'node_metadata = create_test_node_metadata()\n        agent = ChatAgent(pipeline_factory, profile, node_metadata, stream=True)',
        content
    )
    
    # Pattern 2: ChatAgent(pipeline_factory, profile)
    content = re.sub(
        r'(\s+)agent = ChatAgent\(pipeline_factory, profile\)',
        r'\1agent = create_test_chat_agent(pipeline_factory, profile)',
        content
    )
    
    # Pattern 3: ChatAgent(pipeline_factory, profile, stream=False) that we haven't fixed yet
    content = re.sub(
        r'agent = ChatAgent\(pipeline_factory, profile, stream=False\)',
        'agent = create_test_chat_agent(pipeline_factory, profile, stream=False)',
        content
    )
    
    with open(test_file, 'w') as f:
        f.write(content)
    
    print("Fixed ChatAgent test instantiations")

if __name__ == "__main__":
    fix_chat_agent_tests()