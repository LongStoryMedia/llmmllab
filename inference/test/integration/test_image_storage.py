"""Integration tests for ImageStorage — exercises real PostgreSQL."""

from datetime import datetime, timedelta

import pytest
from db.image_storage import ImageStorage
from models.image_metadata import ImageMetadata


def _make_image(
    user_id: str = "user-1",
    filename: str = "test.png",
    conversation_id: int | None = 1,
) -> ImageMetadata:
    return ImageMetadata(
        user_id=user_id,
        filename=filename,
        thumbnail="/thumb/test.png",
        format="png",
        width=1024,
        height=1024,
        conversation_id=conversation_id,
        created_at=datetime.now(),
    )


@pytest.mark.asyncio
async def test_store_and_get_image(image_storage: ImageStorage):
    img = _make_image(filename="output.png", conversation_id=42)
    img_id = await image_storage.store_image(img)
    assert img_id > 0

    fetched = await image_storage.get_image_by_id("user-1", img_id)
    assert fetched is not None
    assert fetched.filename == "output.png"
    assert fetched.format == "png"
    assert fetched.conversation_id == 42


@pytest.mark.asyncio
async def test_list_images(image_storage: ImageStorage):
    await image_storage.store_image(_make_image(filename="a.png", conversation_id=1))
    await image_storage.store_image(_make_image(filename="b.png", conversation_id=2))

    all_imgs = await image_storage.list_images("user-1")
    assert len(all_imgs) == 2

    filtered = await image_storage.list_images("user-1", conversation_id=1)
    assert len(filtered) == 1
    assert filtered[0].filename == "a.png"


@pytest.mark.asyncio
async def test_list_images_wrong_user(image_storage: ImageStorage):
    await image_storage.store_image(_make_image(user_id="owner", filename="x.png", conversation_id=99))
    result = await image_storage.list_images("other-user")
    assert len(result) == 0


@pytest.mark.asyncio
async def test_delete_image(image_storage: ImageStorage):
    img = _make_image(filename="del.png")
    img_id = await image_storage.store_image(img)

    await image_storage.delete_image(img_id)
    fetched = await image_storage.get_image_by_id("user-1", img_id)
    assert fetched is None


@pytest.mark.asyncio
async def test_delete_images_older_than(image_storage: ImageStorage):
    await image_storage.store_image(_make_image(filename="old.png"))
    await image_storage.store_image(_make_image(filename="new.png"))

    # Delete everything older than now (all images qualify)
    await image_storage.delete_images_older_than(datetime.now() + timedelta(hours=1))

    remaining = await image_storage.list_images("user-1")
    assert len(remaining) == 0


@pytest.mark.asyncio
async def test_list_images_pagination(image_storage: ImageStorage):
    for i in range(5):
        await image_storage.store_image(_make_image(filename=f"page-{i}.png"))

    page1 = await image_storage.list_images("user-1", limit=2, offset=0)
    assert len(page1) == 2

    page2 = await image_storage.list_images("user-1", limit=2, offset=2)
    assert len(page2) == 2

    page3 = await image_storage.list_images("user-1", limit=2, offset=4)
    assert len(page3) == 1
