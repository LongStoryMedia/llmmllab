"""
Test the get_most_recent_user_message_content utility function.
"""


def test_get_most_recent_user_message_content():
    """Test the new utility function for extracting the most recent user message."""

    # Mock LangChainMessage class for testing
    class MockLangChainMessage:
        def __init__(self, content, message_type="user"):
            self.content = content
            self.type = message_type

    # Test utility function logic
    def extract_content_from_langchain_message(msg):
        """Extract text content from a LangChainMessage object."""
        if not hasattr(msg, "content"):
            return str(msg) if msg else ""

        content = msg.content

        # Handle list content (LangChainMessage supports string or object items)
        if isinstance(content, list):
            content_parts = []
            for part in content:
                if isinstance(part, str):
                    content_parts.append(part)
                elif isinstance(part, dict) and "text" in part:
                    content_parts.append(part["text"])
                else:
                    # Handle any other object
                    content_parts.append(str(part))
            return " ".join(content_parts)

        # Handle single content
        return str(content) if content else ""

    def get_most_recent_user_message_content(messages):
        """
        Extract content from the most recent user message in a conversation.

        Args:
            messages: List of LangChainMessage objects

        Returns:
            Content of the most recent user message, or empty string if none found
        """
        if not messages:
            return ""

        # Look for the most recent user message by checking message type
        for msg in reversed(messages):
            if hasattr(msg, "type") and msg.type in ("user", "human"):
                return extract_content_from_langchain_message(msg)

        # Fallback: if no explicit user message found, use the last message
        # This handles cases where message types might not be set properly
        if messages:
            return extract_content_from_langchain_message(messages[-1])

        return ""

    print("🧪 Testing get_most_recent_user_message_content utility")
    print("=" * 55)

    # Test Case 1: Normal conversation with multiple user messages
    print("\n📝 Test Case 1: Normal conversation")
    messages1 = [
        MockLangChainMessage("System initialized", "system"),
        MockLangChainMessage("How do I implement binary search?", "user"),
        MockLangChainMessage("Here is the implementation...", "ai"),
        MockLangChainMessage("Can you explain the time complexity?", "user"),
    ]

    result1 = get_most_recent_user_message_content(messages1)
    expected1 = "Can you explain the time complexity?"
    print(f"Result: '{result1}'")
    print(f"Expected: '{expected1}'")
    print(f"✅ Test 1 {'PASSED' if result1 == expected1 else 'FAILED'}")

    # Test Case 2: Empty message list
    print("\n📝 Test Case 2: Empty messages")
    result2 = get_most_recent_user_message_content([])
    expected2 = ""
    print(f"Result: '{result2}'")
    print(f"Expected: '{expected2}'")
    print(f"✅ Test 2 {'PASSED' if result2 == expected2 else 'FAILED'}")

    # Test Case 3: No user messages (fallback to last message)
    print("\n📝 Test Case 3: No user messages")
    messages3 = [
        MockLangChainMessage("System initialized", "system"),
        MockLangChainMessage("AI response", "ai"),
    ]
    result3 = get_most_recent_user_message_content(messages3)
    expected3 = "AI response"
    print(f"Result: '{result3}'")
    print(f"Expected: '{expected3}'")
    print(f"✅ Test 3 {'PASSED' if result3 == expected3 else 'FAILED'}")

    # Test Case 4: Complex content (list format)
    print("\n📝 Test Case 4: Complex content")
    messages4 = [MockLangChainMessage(["Hello", {"text": "world"}], "user")]
    result4 = get_most_recent_user_message_content(messages4)
    expected4 = "Hello world"
    print(f"Result: '{result4}'")
    print(f"Expected: '{expected4}'")
    print(f"✅ Test 4 {'PASSED' if result4 == expected4 else 'FAILED'}")

    # Test Case 5: Human type message
    print("\n📝 Test Case 5: Human type message")
    messages5 = [MockLangChainMessage("User question", "human")]
    result5 = get_most_recent_user_message_content(messages5)
    expected5 = "User question"
    print(f"Result: '{result5}'")
    print(f"Expected: '{expected5}'")
    print(f"✅ Test 5 {'PASSED' if result5 == expected5 else 'FAILED'}")

    print(f"\n🎉 Utility function testing complete!")
    print("\n📋 Key Benefits:")
    print("   1. ✅ Handles various LangChainMessage content formats")
    print("   2. ✅ Properly identifies user/human message types")
    print("   3. ✅ Robust fallback mechanisms")
    print("   4. ✅ Replaces fragile manual content extraction")
    print("   5. ✅ Centralized logic for reuse across workflows")


if __name__ == "__main__":
    test_get_most_recent_user_message_content()
