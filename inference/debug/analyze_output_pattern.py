#!/usr/bin/env python3
"""
Analyze the composer output file to understand the cycling pattern.
"""

import re

def analyze_model_outputs(file_path):
    """Analyze MODEL_OUTPUT sections to understand the cycling pattern."""
    print(f"Analyzing file: {file_path}")
    print("=" * 60)
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Find all MODEL_OUTPUT sections
    pattern = r'SECTION: MODEL_OUTPUT.*?TIMESTAMP: ([0-9T:-]+).*?(?=SECTION:|$)'
    matches = list(re.finditer(pattern, content, re.DOTALL))
    
    print(f"Found {len(matches)} MODEL_OUTPUT sections:")
    print("-" * 40)
    
    for i, match in enumerate(matches, 1):
        timestamp = match.group(1)
        section_content = match.group(0)
        
        # Look for tool calls in this section
        has_tool_calls = "tool_call" in section_content.lower()
        
        # Look for the actual response content (rough estimation)
        lines = section_content.split('\n')
        content_lines = [line for line in lines if line.strip() and not line.startswith('=') and not line.startswith('SECTION:') and not line.startswith('TITLE:') and not line.startswith('TIMESTAMP:') and not line.startswith('DESCRIPTION:')]
        
        print(f"OUTPUT #{i}:")
        print(f"  Timestamp: {timestamp}")
        print(f"  Has tool calls: {has_tool_calls}")
        print(f"  Content lines: {len(content_lines)}")
        print()
    
    # Look at workflow events around each MODEL_OUTPUT
    print("\nWorkflow events around MODEL_OUTPUT sections:")
    print("-" * 40)
    
    # Find workflow events before each MODEL_OUTPUT
    for i, match in enumerate(matches, 1):
        start_pos = max(0, match.start() - 2000)  # Look 2000 chars before
        before_content = content[start_pos:match.start()]
        
        # Find recent workflow events
        event_pattern = r'WORKFLOW EVENT: ([^\\n]+)'
        events = re.findall(event_pattern, before_content)
        recent_events = events[-5:] if events else []  # Last 5 events
        
        print(f"OUTPUT #{i} - Recent workflow events:")
        for event in recent_events:
            print(f"  {event}")
        print()

if __name__ == "__main__":
    file_path = "/Users/lons7862/workspace/llmmllab/inference/debug/out/composer_llm_output_qwen3_30b_a3b_q4_k_m_20251022_144647.txt"
    analyze_model_outputs(file_path)