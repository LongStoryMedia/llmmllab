"""initial schema - consolidated from db/sql/ DDL

Revision ID: 0001
Revises:
Create Date: 2026-04-25
"""

import logging
from alembic import op
from pathlib import Path

import sqlalchemy as sa  # noqa: F401

# revision identifiers, used by Alembic.
revision = "0001"
down_revision = None
branch_labels = None
depends_on = None

logger = logging.getLogger(__name__)

# SQL directory relative to this migration file's project root
SQL_DIR = Path(__file__).resolve().parent.parent.parent / "db" / "sql"


def _sql(filename: str) -> str:
    """Load a SQL file from the db/sql/ directory."""
    return (SQL_DIR / filename).read_text()


def _schema_already_exists() -> bool:
    """Check if the schema was already created by init_db.py."""
    conn = op.get_bind()
    try:
        result = conn.execute(
            sa.text("""
                SELECT EXISTS(
                    SELECT 1 FROM information_schema.tables
                    WHERE table_schema = 'public'
                    AND table_name = 'users'
                )
            """)
        )
        return bool(result.scalar())
    except Exception:
        return False


def upgrade() -> None:
    """Apply the full schema, matching init_db.py initialization order.

    If the schema already exists (created by init_db.py), this migration
    is a no-op — the alembic_version stamp ensures future migrations (0002+)
    will still run correctly.  All DDL statements use IF NOT EXISTS /
    OR REPLACE so they are safe to re-run on a fresh database.
    """
    if _schema_already_exists():
        logger.info("Schema already exists, skipping initial migration DDL")
        return

    # Extensions
    op.execute(_sql("init/create_extensions.sql"))

    # Step 1: Users
    op.execute(_sql("user/create_users_table.sql"))

    # Step 1b: API keys
    op.execute(_sql("api_key/create_api_keys_table.sql"))

    # Step 2: Conversations
    op.execute(_sql("conversation/create_conversations_table.sql"))
    op.execute(_sql("conversation/create_conversations_indexes.sql"))
    op.execute(_sql("conversation/create_conversations_hypertable.sql"))
    op.execute(_sql("conversation/create_cascade_delete_trigger.sql"))
    op.execute(_sql("conversation/enable_conversations_compression.sql"))
    op.execute(_sql("conversation/conversations_compression_policy.sql"))
    op.execute(_sql("conversation/conversations_retention_policy.sql"))

    # Step 2b: User triggers (reference conversations table, must come after)
    op.execute(_sql("user/create_user_check_trigger.sql"))
    op.execute(_sql("user/user_delete_cascade.sql"))

    # Step 4: Messages
    op.execute(_sql("message/create_messages_table.sql"))

    # Conversation update trigger (references messages table)
    op.execute(_sql("conversation/create_conversation_update_trigger.sql"))

    # Step 5: Message content
    op.execute(_sql("message_content/create_message_content_table.sql"))
    op.execute(_sql("message_content/create_message_contents_hypertable.sql"))
    op.execute(_sql("message_content/enable_message_contents_compression.sql"))
    op.execute(_sql("message_content/message_contents_compression_policy.sql"))
    op.execute(_sql("message_content/message_contents_retention_policy.sql"))

    # Step 6: Documents
    op.execute(_sql("document/create_documents_table.sql"))
    op.execute(_sql("document/create_documents_hypertable.sql"))

    # Step 7: Summaries
    op.execute(_sql("summary/create_summaries_table.sql"))
    op.execute(_sql("summary/create_summaries_hypertable.sql"))
    op.execute(_sql("summary/enable_summaries_compression.sql"))
    op.execute(_sql("summary/create_summaries_indexes.sql"))
    op.execute(_sql("summary/create_summary_cascade_delete_triggers.sql"))

    # Step 8: Search
    op.execute(_sql("search/create_search_topic_synthesis_table.sql"))
    op.execute(_sql("search/create_search_topic_synthesis_hypertable.sql"))
    op.execute(_sql("search/create_search_cascade_delete_triggers.sql"))

    # Step 9: Memory
    op.execute(_sql("memory/init_memory_schema.sql"))
    op.execute(_sql("memory/enable_memories_compression.sql"))
    op.execute(_sql("memory/memories_compression_policy.sql"))
    op.execute(_sql("memory/memories_retention_policy.sql"))
    op.execute(_sql("memory/create_memory_indexes.sql"))
    op.execute(_sql("memory/create_memory_cascade_delete_triggers.sql"))

    # Step 10: Images
    op.execute(_sql("images/create_images_schema.sql"))

    # Step 12: Research
    op.execute(_sql("research/create_research_tasks_table.sql"))
    op.execute(_sql("research/create_research_subtasks_table.sql"))

    # Step 13: Structured response tables
    op.execute(_sql("thought/create_table.sql"))
    op.execute(_sql("tool_call/create_table.sql"))
    op.execute(_sql("tool_call/migrate_to_tool_execution_result_schema.sql"))

    # Step 14: Message cascade delete triggers
    op.execute(_sql("message/create_message_cascade_delete_triggers.sql"))

    # Step 17: Todos
    op.execute(_sql("todo/create_table.sql"))


def downgrade() -> None:
    """Drop all tables in reverse order."""
    op.execute("""
        DROP TABLE IF EXISTS todos CASCADE;
        DROP TABLE IF EXISTS tool_calls CASCADE;
        DROP TABLE IF EXISTS thoughts CASCADE;
        DROP TABLE IF EXISTS research_subtasks CASCADE;
        DROP TABLE IF EXISTS research_tasks CASCADE;
        DROP TABLE IF EXISTS images CASCADE;
        DROP TABLE IF EXISTS memories CASCADE;
        DROP TABLE IF EXISTS search_topic_syntheses CASCADE;
        DROP TABLE IF EXISTS summaries CASCADE;
        DROP TABLE IF EXISTS documents CASCADE;
        DROP TABLE IF EXISTS message_contents CASCADE;
        DROP TABLE IF EXISTS messages CASCADE;
        DROP TABLE IF EXISTS conversations CASCADE;
        DROP TABLE IF EXISTS api_keys CASCADE;
        DROP TABLE IF EXISTS users CASCADE;
    """)
    op.execute("DROP EXTENSION IF EXISTS vector CASCADE")
    op.execute("DROP EXTENSION IF EXISTS timescaledb CASCADE")
