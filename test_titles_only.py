#!/usr/bin/env python3
"""
Simple test to check for multiple title generation.
"""
import requests
import json

def test_single_title():
    url = "http://192.168.0.122:8000/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "X-User-ID": "CgNsc20SBGxkYXA"
    }
    
    payload = {
        "role": "user",
        "content": [{"type": "text", "text": "What is 2+2?"}],
        "conversation_id": 999  # Use different conversation ID
    }
    
    print("🧪 Testing for multiple titles...")
    print(f"Request: {payload['content'][0]['text']}")
    print("=" * 80)
    
    response = requests.post(url, headers=headers, json=payload, stream=True, timeout=30)
    
    full_content = ""
    title_count = 0
    
    for line in response.iter_lines():
        if line:
            try:
                data = json.loads(line.decode('utf-8'))
                if data.get("done") == False:
                    message = data.get("message", {})
                    content_items = message.get("content", [])
                    for item in content_items:
                        if item.get("type") == "text":
                            text = item.get("text", "")
                            full_content += text
                            
                            # Count title occurrences
                            if "**Title:**" in text or "Title:" in text:
                                title_count += 1
                                print(f"🎯 TITLE DETECTED #{title_count}: '{text.strip()}'")
                            
            except json.JSONDecodeError:
                pass
    
    print("=" * 80)
    print(f"📊 TITLE COUNT: {title_count}")
    
    # Extract any titles from full content
    lines = full_content.split('\n')
    for i, line in enumerate(lines):
        if "title" in line.lower() and ("**" in line or ":" in line):
            print(f"🔍 Potential title on line {i+1}: '{line.strip()}'")
    
    if title_count == 0:
        print("✅ No duplicate titles found!")
    elif title_count == 1:
        print("✅ Single title found - good!")
    else:
        print(f"❌ MULTIPLE TITLES FOUND: {title_count}")
    
    return title_count

if __name__ == "__main__":
    test_single_title()