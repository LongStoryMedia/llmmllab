#!/usr/bin/env python3
"""
Debug script to check raw JSON output from message queries.
"""

import asyncio
import sys
import os

# Add the inference directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from db import storage

async def debug_json_output():
    """Debug the raw JSON output from message queries."""
    print("🔍 Debugging JSON aggregation output...")
    
    try:
        # Initialize database connection
        print("   💾 Initializing database...")
        db_host = os.environ.get("DB_HOST", "localhost")
        db_port = os.environ.get("DB_PORT", "5432")
        db_user = os.environ.get("DB_USER", "lsm")
        db_password = os.environ.get("DB_PASSWORD", "")
        db_name = os.environ.get("DB_NAME", "llmmll")

        connection_string = (
            f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
        )
        await storage.initialize(connection_string)
        print("   ✅ Database initialized")
        
        # Get direct database connection
        async with storage.pool.acquire() as conn:
            print("\n1️⃣ Testing raw get_message query...")
            
            # Get the raw row from the database
            row = await conn.fetchrow(storage.get_query("message.get_message"), 2545)
            if row:
                print(f"Raw row keys: {list(row.keys())}")
                print(f"Raw row values: {dict(row)}")
                print(f"Contents field type: {type(row.get('contents'))}")
                print(f"Contents field value: {row.get('contents')}")
            else:
                print("❌ No row found for message ID 2545")
                
            print("\n2️⃣ Testing message_contents table directly...")
            contents_rows = await conn.fetch(
                "SELECT * FROM message_contents WHERE message_id = $1", 2545
            )
            print(f"Found {len(contents_rows)} content rows:")
            for i, content_row in enumerate(contents_rows):
                print(f"  Content {i+1}: {dict(content_row)}")
                
            print("\n3️⃣ Testing JSON aggregation manually...")
            manual_json = await conn.fetchrow("""
                SELECT 
                    m.id,
                    m.role,
                    m.conversation_id,
                    m.created_at,
                    COALESCE(
                        JSON_AGG(
                            JSON_BUILD_OBJECT(
                                'content_type', mc.type,
                                'content', mc.text_content,
                                'content_url', mc.url
                            )
                        ) FILTER (WHERE mc.message_id IS NOT NULL),
                        '[]'::json
                    ) AS contents
                FROM messages m
                LEFT JOIN message_contents mc ON m.id = mc.message_id
                WHERE m.id = $1
                GROUP BY m.id, m.role, m.conversation_id, m.created_at
            """, 2545)
            
            if manual_json:
                print(f"Manual JSON result: {dict(manual_json)}")
                print(f"Manual contents: {manual_json['contents']}")
            else:
                print("❌ No manual JSON result")
        
    except Exception as e:
        print(f"❌ Failed to debug: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(debug_json_output())