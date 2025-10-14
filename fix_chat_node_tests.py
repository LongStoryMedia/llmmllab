#!/usr/bin/env python3
"""
Script to bulk fix ChatNode test instantiations to use new constructor signature.
"""

import re

def fix_chat_node_tests():
    test_file = "/Users/lons7862/workspace/llmmllab/inference/test/unit/test_chat_node.py"
    
    with open(test_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Replace all instances of ChatNode(mock_agent) with create_test_chat_node()
    content = re.sub(
        r'node = ChatNode\(mock_agent\)',
        'node = create_test_chat_node()',
        content
    )
    
    # Replace instances with node_name parameter
    content = re.sub(
        r'node = ChatNode\(mock_agent, node_name="([^"]+)"\)',
        r'node = create_test_chat_node(node_name="\1")',
        content
    )
    
    with open(test_file, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("Fixed ChatNode test instantiations")

if __name__ == "__main__":
    fix_chat_node_tests()