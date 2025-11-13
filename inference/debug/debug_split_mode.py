#!/usr/bin/env python3
"""
Debug script to trace split-mode flag parsing
"""
import re
from runner.server_manager.dynamic_flag_parser import DynamicFlagParser

def debug_split_mode():
    parser = DynamicFlagParser('/llama.cpp/build/bin/llama-server')
    help_output = parser.get_help_output()
    lines = help_output.split('\n')
    
    for line in lines:
        if 'split-mode' in line and not line.startswith(' '):
            print(f"=== Processing line: {repr(line)} ===")
            
            # Simulate the parsing logic from the actual parser
            for potential_split in re.finditer(r'\s{2,}', line):
                start = potential_split.start()
                potential_desc = line[potential_split.end():].strip()
                if potential_desc and not potential_desc.startswith('-'):
                    parts = [line[:start].strip(), potential_desc]
                    break
            else:
                parts = [line.strip(), '']
            
            print(f"Parts: {parts}")
            flag_spec = parts[0].strip()
            description = parts[1].strip() if len(parts) > 1 else ""
            print(f"Flag spec: {repr(flag_spec)}")
            print(f"Description: {repr(description)}")
            
            # Parse flag specification
            # Split on comma but respect braces - don't split commas inside {}
            flag_parts = []
            current_part = ""
            brace_depth = 0
            
            for char in flag_spec:
                if char == '{':
                    brace_depth += 1
                    current_part += char
                elif char == '}':
                    brace_depth -= 1
                    current_part += char
                elif char == ',' and brace_depth == 0:
                    # Only split on comma if we're not inside braces
                    if current_part.strip():
                        flag_parts.append(current_part.strip())
                    current_part = ""
                else:
                    current_part += char
            
            # Add the last part
            if current_part.strip():
                flag_parts.append(current_part.strip())
            print(f"Flag parts: {flag_parts}")
            
            for part in flag_parts:
                part = part.strip()
                print(f"  Processing part: {repr(part)}")
                
                tokens = part.split()
                print(f"    Tokens: {tokens}")
                
                if len(tokens) > 1:
                    potential_value_type = tokens[1]
                    print(f"    Potential value type: {repr(potential_value_type)}")
                    
                    # Check for choice pattern
                    if potential_value_type.startswith('{') and potential_value_type.endswith('}'):
                        print(f"    ✅ Found choice pattern: {potential_value_type}")
                    else:
                        print("    ❌ Not a choice pattern")

if __name__ == "__main__":
    debug_split_mode()