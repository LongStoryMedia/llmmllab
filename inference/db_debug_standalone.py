import os
import json
import asyncio
from pprint import pprint


async def debug_db():
    print("\n=== DATABASE DEBUGGING START ===\n")

    try:
        print("Loading modules...")
        import asyncpg
        from server.db import storage
        from server.config import DB_CONNECTION_STRING
        from server.db.queries import get_loader

        print("Modules loaded successfully")
    except Exception as e:
        print(f"Error importing modules: {e}")
        return

    print(f"\nEnvironment variables:")
    for key in ["DB_CONNECTION_STRING", "DB_HOST", "DB_PORT", "DB_NAME", "DB_USER"]:
        if key in os.environ:
            print(f"  {key} is set")
        else:
            print(f"  {key} is NOT set")

    if not DB_CONNECTION_STRING:
        print("\nERROR: DB_CONNECTION_STRING is not set or empty!")
        return

    # Check SQL loader
    print("\nChecking SQL loader...")
    try:
        loader = get_loader()
        print(f"SQL directory: {loader.sql_dir}")
        print(f"Total SQL queries loaded: {len(loader.queries)}")
        print("\nChecking for key SQL files:")
        key_queries = [
            "user.create_users_table",
            "user.get_config",
            "user.get_all_users",
            "conversation.create_conversations_table",
            "message.create_messages_table",
        ]
        for key in key_queries:
            if key in loader.queries:
                print(f"  ✓ {key} exists")
            else:
                print(f"  ✗ {key} MISSING")

        # Dump all query keys for reference
        print("\nAll query keys:")
        for key in sorted(loader.queries.keys()):
            if key.startswith("user."):
                print(f"  {key}")
    except Exception as e:
        print(f"Error checking SQL loader: {e}")

    # Check storage initialization
    print("\nChecking database storage...")
    print(f"Storage initialized: {storage.initialized}")
    print(f"Storage pool exists: {storage.pool is not None}")
    print(f"Storage user_config exists: {storage.user_config is not None}")

    # Try to initialize
    if not storage.initialized:
        print("\nAttempting to initialize storage...")
        try:
            await storage.initialize(DB_CONNECTION_STRING)
            print(f"Storage initialized after attempt: {storage.initialized}")
        except Exception as e:
            print(f"Error initializing storage: {e}")

    # Try to get users
    if storage.initialized and storage.user_config:
        print("\nAttempting to get users...")
        try:
            users = await storage.user_config.get_all_users()
            print(f"Successfully retrieved {len(users)} users")
            if users:
                print("Sample user data:")
                pprint(users[0])
        except Exception as e:
            print(f"Error getting users: {e}")

    print("\n=== DATABASE DEBUGGING END ===\n")


# Run the debug function
asyncio.run(debug_db())
