"""Integration tests for ThoughtStorage — exercises real PostgreSQL."""

import pytest
from db.thought_storage import ThoughtStorage
from models.thought import Thought


@pytest.mark.asyncio
async def test_add_and_get_thoughts(
    thought_storage: ThoughtStorage, seed_message: int
):
    tid = await thought_storage.add_thought(
        thought=Thought(message_id=seed_message, text="First thought")
    )
    assert tid is not None and tid > 0

    await thought_storage.add_thought(
        thought=Thought(message_id=seed_message, text="Second thought")
    )

    thoughts = await thought_storage.get_thoughts_by_message(seed_message)
    assert len(thoughts) == 2
    assert thoughts[0].text == "First thought"
    assert thoughts[1].text == "Second thought"


@pytest.mark.asyncio
async def test_get_thoughts_empty(thought_storage: ThoughtStorage):
    thoughts = await thought_storage.get_thoughts_by_message(999)
    assert thoughts == []


@pytest.mark.asyncio
async def test_delete_thoughts_by_message(
    thought_storage: ThoughtStorage, seed_message: int
):
    await thought_storage.add_thought(
        thought=Thought(message_id=seed_message, text="Delete me")
    )
    await thought_storage.add_thought(
        thought=Thought(message_id=seed_message, text="And me")
    )

    result = await thought_storage.delete_thoughts_by_message(seed_message)
    assert result is True

    remaining = await thought_storage.get_thoughts_by_message(seed_message)
    assert remaining == []


@pytest.mark.asyncio
async def test_delete_thoughts_empty(thought_storage: ThoughtStorage):
    result = await thought_storage.delete_thoughts_by_message(999)
    assert result is False
