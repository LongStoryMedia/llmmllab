#!/usr/bin/env python3
"""
Debug QwenMoE Tool Calling - Explicit Test
Send very direct tool calling request to QwenMoE
"""

import re

def analyze_qwen_response():
    """Analyze the actual QwenMoE response from the test"""
    
    # This is the actual response from the test logs
    response = """As of now, there are no widely reported breakthroughs in quantum computing from2024, as the year is still ongoing (as of July2024). However, here's a summary of the most recent developments up to early2024, along with speculative trends that align with the field's trajectory:

---

### **Quantum Error Correction**
- **Advances**: Researchers have made progress in stabilizing qubits using **topological error correction codes** (e.g., Majorana zero modes) and hybrid quantum-classical approaches to mitigate decoherence. A 2023 paper by Google and UC Berkeley proposed a scalable method for fault-tolerant error correction, which is expected to be further refined in 2024.
- **Key Players**: IBM, Google, and startups like IonQ are exploring hardware designs that prioritize error resilience."""
    
    print("🔍 Analyzing QwenMoE Response for Tool Calling")
    print("=" * 60)
    
    # Check for tool calling patterns
    patterns = {
        'json_blocks': len(re.findall(r'```json.*?```', response, re.DOTALL)),
        'tool_calls_array': len(re.findall(r'"tool_calls":\s*\[', response)),
        'name_field': len(re.findall(r'"name":\s*"[\w_]+"', response)),
        'arguments_field': len(re.findall(r'"arguments":\s*{', response)),
        'web_search_call': 'web_search' in response,
        'search_mention': 'search' in response.lower(),
        'current_info_request': any(term in response.lower() for term in ['2024', 'recent', 'current', 'latest']),
        'speculative_language': any(term in response.lower() for term in ['speculative', 'expected', 'might', 'could'])
    }
    
    print("📊 Pattern Analysis:")
    for pattern, result in patterns.items():
        status = "✅" if (result > 0 if isinstance(result, int) else result) else "❌"
        print(f"   {status} {pattern}: {result}")
    
    print(f"\n📄 Response Length: {len(response)} characters")
    print(f"📄 First 200 chars: {response[:200]}...")
    
    # Key insights
    print("\n🎯 Key Insights:")
    if patterns['current_info_request']:
        print("   ✅ Request was for current/2024 information - SHOULD have triggered web_search")
    if patterns['speculative_language']:
        print("   ⚠️  Response contains speculative language - indicates lack of current data")
    if not patterns['json_blocks']:
        print("   ❌ No JSON blocks found - model did NOT use tool calling format")
    if not patterns['web_search_call']:
        print("   ❌ No web_search tool usage - model should have searched for 2024 info")
    
    # Conclusion
    print(f"\n🔍 DIAGNOSIS:")
    if not patterns['json_blocks'] and patterns['current_info_request']:
        print("   🚨 TOOL CALLING FAILURE: Model should have used web_search for 2024 info but didn't")
        print("   🔧 REQUIRED: JSON format with tool_calls array")
        print("   📝 EXPECTED: ```json { \"tool_calls\": [...] } ```")
    
    return patterns

if __name__ == "__main__":
    analyze_qwen_response()