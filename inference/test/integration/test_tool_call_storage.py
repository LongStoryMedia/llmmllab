"""Integration tests for ToolCallStorage — exercises real PostgreSQL."""

import pytest
from db.tool_call_storage import ToolCallStorage
from models.tool_call import ToolCall


@pytest.mark.asyncio
async def test_add_and_get_tool_calls(
    tool_call_storage: ToolCallStorage, seed_message: int
):
    tc_id = await tool_call_storage.add_tool_call(
        tool_call=ToolCall(
            message_id=seed_message,
            name="search",
            execution_id="exec-1",
            args={"query": "test"},
        )
    )
    assert tc_id is not None and tc_id > 0

    calls = await tool_call_storage.get_tool_calls_by_message(seed_message)
    assert len(calls) == 1
    assert calls[0].name == "search"


@pytest.mark.asyncio
async def test_get_tool_calls_empty(tool_call_storage: ToolCallStorage):
    calls = await tool_call_storage.get_tool_calls_by_message(999)
    assert calls == []


@pytest.mark.asyncio
async def test_update_tool_call_result(
    tool_call_storage: ToolCallStorage, seed_message: int
):
    await tool_call_storage.add_tool_call(
        tool_call=ToolCall(
            message_id=seed_message,
            name="calculator",
            execution_id="exec-2",
            args={"expr": "2+2"},
        )
    )

    updated = await tool_call_storage.update_tool_call_result(
        message_id=seed_message,
        execution_id="exec-2",
        result_data={"output": "4"},
        success=True,
        error_message=None,
        execution_time_ms=42,
    )
    assert updated is not None
    assert updated.result_data is not None

    calls = await tool_call_storage.get_tool_calls_by_message(seed_message)
    assert len(calls) >= 1
    assert any(c.result_data is not None for c in calls)


@pytest.mark.asyncio
async def test_delete_tool_calls_by_message(
    tool_call_storage: ToolCallStorage, seed_message: int
):
    await tool_call_storage.add_tool_call(
        tool_call=ToolCall(
            message_id=seed_message,
            name="web_search",
            execution_id="exec-3",
            args={},
        )
    )
    result = await tool_call_storage.delete_tool_calls_by_message(seed_message)
    assert result is True


@pytest.mark.asyncio
async def test_multiple_tool_calls(
    tool_call_storage: ToolCallStorage, seed_message: int
):
    await tool_call_storage.add_tool_call(
        tool_call=ToolCall(
            message_id=seed_message,
            name="tool_a",
            execution_id="exec-4a",
            args={},
        )
    )
    await tool_call_storage.add_tool_call(
        tool_call=ToolCall(
            message_id=seed_message,
            name="tool_b",
            execution_id="exec-4b",
            args={},
        )
    )

    calls = await tool_call_storage.get_tool_calls_by_message(seed_message)
    assert len(calls) >= 2
    names = {c.name for c in calls}
    assert "tool_a" in names
    assert "tool_b" in names
