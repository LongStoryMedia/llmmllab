"""
Test for context assembly utility function.

This test verifies that the assemble_context_messages function correctly
constructs Message objects from WorkflowState following the context extension
architecture.
"""

from datetime import datetime

from composer.graph.state import WorkflowState, assemble_context_messages
from models import (
    Message,
    MessageRole,
    LangChainMessage,
    Memory,
    MemoryFragment,
    MemorySource,
    Summary,
)


def test_assemble_context_messages_basic():
    """Test basic context assembly with all components."""
    
    # Create test WorkflowState with various components
    state = WorkflowState(
        messages=[
            LangChainMessage(content="Hello", type="user"),
            LangChainMessage(content="Hi there!", type="ai"),
        ],
        user_id="test_user",
        conversation_id=123,
    )
    
    # Add test summaries
    state.summaries = [
        Summary(
            id=1,
            content="Previous conversation about greetings",
            level=1,
            conversation_id=123,
            source_ids=[1, 2],
            created_at=datetime.now(),
        )
    ]
    
    # Add test memories
    state.retrieved_memories = [
        Memory(
            fragments=[
                MemoryFragment(id=1, role=MessageRole.USER, content="How are you?"),
                MemoryFragment(id=2, role=MessageRole.ASSISTANT, content="I'm doing well!"),
            ],
            source=MemorySource.MESSAGE,
            created_at=datetime.now(),
            similarity=0.85,
            source_id=1,
            conversation_id=123,
        )
    ]
    
    # Add test search results
    state.search_results = "External information about greetings and conversations"
    
    # Assemble context
    context_messages = assemble_context_messages(state)
    
    # Verify structure
    assert len(context_messages) >= 4  # Summary + Memory + Search + 2 Messages
    assert all(isinstance(msg, Message) for msg in context_messages)
    
    # Check that system messages were added for context components
    system_messages = [msg for msg in context_messages if msg.role == MessageRole.SYSTEM]
    assert len(system_messages) >= 3  # Summary + Memory + Search
    
    # Check that conversation messages were converted
    user_messages = [msg for msg in context_messages if msg.role == MessageRole.USER]
    assistant_messages = [msg for msg in context_messages if msg.role == MessageRole.ASSISTANT]
    assert len(user_messages) >= 1
    assert len(assistant_messages) >= 1


def test_assemble_context_messages_selective():
    """Test selective inclusion of context components."""
    
    state = WorkflowState(
        messages=[LangChainMessage(content="Test message", type="user")],
        user_id="test_user",
        conversation_id=123,
    )
    
    state.summaries = [
        Summary(
            id=1,
            content="Test summary",
            level=1,
            conversation_id=123,
            source_ids=[],
            created_at=datetime.now(),
        )
    ]
    
    state.retrieved_memories = [
        Memory(
            fragments=[MemoryFragment(id=1, role=MessageRole.USER, content="Test memory")],
            source=MemorySource.MESSAGE,
            created_at=datetime.now(),
            similarity=0.9,
            source_id=1,
            conversation_id=123,
        )
    ]
    
    state.search_results = "Test search results"
    
    # Test with only memories
    memory_only = assemble_context_messages(
        state,
        include_summaries=False,
        include_search_results=False,
        include_memories=True
    )
    
    system_messages = [msg for msg in memory_only if msg.role == MessageRole.SYSTEM]
    assert len(system_messages) == 1  # Only memory
    assert system_messages[0].content[0].text and "Relevant Memory" in system_messages[0].content[0].text
    
    # Test with no context components
    messages_only = assemble_context_messages(
        state,
        include_summaries=False,
        include_search_results=False,
        include_memories=False
    )
    
    system_messages = [msg for msg in messages_only if msg.role == MessageRole.SYSTEM]
    assert len(system_messages) == 0  # No context components


def test_assemble_context_messages_token_budgeting():
    """Test token budgeting functionality."""
    
    state = WorkflowState(
        messages=[
            LangChainMessage(content="A" * 1000, type="user"),  # ~250 tokens
            LangChainMessage(content="B" * 1000, type="ai"),    # ~250 tokens
            LangChainMessage(content="C" * 1000, type="user"),  # ~250 tokens
        ],
        user_id="test_user",
        conversation_id=123,
    )
    
    # Test with tight token budget
    budgeted_messages = assemble_context_messages(
        state,
        max_context_tokens=400,  # Should fit ~2 messages
        include_summaries=False,
        include_memories=False,
        include_search_results=False
    )
    
    # Should prioritize recent messages and respect budget
    assert len(budgeted_messages) <= 2
    # Most recent messages should be included
    assert any(msg.content[0].text and "C" * 100 in msg.content[0].text for msg in budgeted_messages)


def test_memory_formatting():
    """Test memory fragment formatting."""
    
    from composer.graph.state import _format_memory_for_message
    
    # Test user-assistant pair
    pair_memory = Memory(
        fragments=[
            MemoryFragment(id=1, role=MessageRole.USER, content="What's the weather?"),
            MemoryFragment(id=2, role=MessageRole.ASSISTANT, content="It's sunny today."),
        ],
        source=MemorySource.MESSAGE,
        created_at=datetime.now(),
        similarity=0.9,
        source_id=1,
        conversation_id=123,
    )
    
    formatted = _format_memory_for_message(pair_memory)
    assert "User: What's the weather?" in formatted
    assert "Assistant: It's sunny today." in formatted
    
    # Test single fragment
    single_memory = Memory(
        fragments=[
            MemoryFragment(id=1, role=MessageRole.SYSTEM, content="System information"),
        ],
        source=MemorySource.SUMMARY,
        created_at=datetime.now(),
        similarity=0.8,
        source_id=1,
        conversation_id=123,
    )
    
    formatted_single = _format_memory_for_message(single_memory)
    assert "System: System information" in formatted_single


def test_langchain_type_mapping():
    """Test LangChain message type to MessageRole mapping."""
    
    from composer.graph.state import _map_langchain_type_to_message_role
    
    assert _map_langchain_type_to_message_role("user") == MessageRole.USER
    assert _map_langchain_type_to_message_role("human") == MessageRole.USER
    assert _map_langchain_type_to_message_role("ai") == MessageRole.ASSISTANT
    assert _map_langchain_type_to_message_role("assistant") == MessageRole.ASSISTANT
    assert _map_langchain_type_to_message_role("system") == MessageRole.SYSTEM
    assert _map_langchain_type_to_message_role("tool") == MessageRole.TOOL
    assert _map_langchain_type_to_message_role("unknown") == MessageRole.USER  # Default


if __name__ == "__main__":
    # Run basic test
    test_assemble_context_messages_basic()
    print("✅ Basic context assembly test passed")
    
    test_assemble_context_messages_selective()
    print("✅ Selective context assembly test passed")
    
    test_memory_formatting()
    print("✅ Memory formatting test passed")
    
    test_langchain_type_mapping()
    print("✅ LangChain type mapping test passed")
    
    print("\n🎉 All context assembly tests passed!")