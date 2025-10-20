#!/usr/bin/env python3
"""
Test script that exactly mimics what the UI sees when making a chat completion request.
This shows the raw HTTP response that the frontend would receive.
"""

import json
import sys
import urllib.request
import urllib.parse
from urllib.error import HTTPError


def test_ui_chat_request():
    """Test the exact request the UI makes and show the exact response."""

    # Exact same request format as UI
    url = "http://192.168.0.122:8000/chat/completions"  # Use external IP directly

    # Headers exactly as UI sends them
    headers = {"Content-Type": "application/json", "X-User-ID": "CgNsc20SBGxkYXA"}

    # Request body exactly as UI sends it
    payload = {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": "What are the most commonly used shallow learning algorithms in 2025? Provide examples.",
            }
        ],
        "conversation_id": 717,
    }

    print("=" * 80)
    print("🧪 UI CHAT COMPLETION TEST")
    print("=" * 80)
    print(f"URL: {url}")
    print(f"Headers: {json.dumps(headers, indent=2)}")
    print(f"Payload: {json.dumps(payload, indent=2)}")
    print("=" * 80)
    print("📡 STREAMING RESPONSE (exactly what UI sees):")
    print("=" * 80)

    try:
        # Prepare the request exactly like the UI
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers=headers)

        # Make the request and stream the response
        with urllib.request.urlopen(req, timeout=300) as response:
            print(f"Status: {response.getcode()}")
            print(f"Headers: {dict(response.headers)}")
            print("-" * 80)
            print("RAW RESPONSE STREAM:")
            print("-" * 80)

            line_count = 0
            total_content = ""

            # Read the streaming response line by line (exactly like UI would)
            for line in response:
                line_count += 1
                line_str = line.decode("utf-8").strip()

                if line_str:
                    print(f"[{line_count:03d}] {line_str}")

                    # Try to parse each line as JSON to extract content
                    try:
                        chunk = json.loads(line_str)
                        if "message" in chunk and "content" in chunk["message"]:
                            content_parts = chunk["message"]["content"]
                            if isinstance(content_parts, list):
                                for part in content_parts:
                                    if part.get("type") == "text":
                                        text = part.get("text", "")
                                        total_content += text

                    except json.JSONDecodeError:
                        pass

                # Stop if we've seen too many lines (prevent infinite scroll)
                if line_count > 1000:
                    print(f"\n⚠️  Truncated after {line_count} lines")
                    break

            print("-" * 80)
            print(f"📊 SUMMARY:")
            print(f"Total lines: {line_count}")
            print(f"Total content length: {len(total_content)} characters")
            print("-" * 80)
            print("📝 ASSEMBLED CONTENT:")
            print("-" * 80)
            print(total_content)
            print("-" * 80)

    except HTTPError as e:
        print(f"❌ HTTP Error: {e.code} - {e.reason}")
        print(f"Response: {e.read().decode('utf-8')}")
    except Exception as e:
        print(f"💥 Error: {str(e)}")


if __name__ == "__main__":
    print("Testing direct connection to Kubernetes service...")
    print("URL: http://192.168.0.122:8000/chat/completions")
    print()

    test_ui_chat_request()
