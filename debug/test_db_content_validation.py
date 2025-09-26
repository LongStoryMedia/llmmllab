#!/usr/bin/env python3
"""
Test to validate that the database content storage fix is working.
"""

import asyncio
import sys
import os

# Add the project paths
sys.path.insert(0, '/app/server')
sys.path.insert(0, '/app/runner') 
sys.path.insert(0, '/app')

from server.storage.message import MessageStorage
from server.storage.config import DatabaseConfig
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def test_database_content():
    """Test that database content is properly stored and retrieved."""
    
    logger.info("🔍 Testing database content storage and retrieval")
    
    try:
        # Initialize database connection
        config = DatabaseConfig()
        storage = MessageStorage(config)
        await storage.initialize()
        
        logger.info("✅ Database connection established")
        
        # Query recent messages directly
        async with storage.pool.acquire() as conn:
            # Get recent messages with their content length
            result = await conn.fetch("""
                SELECT 
                    m.id,
                    m.role,
                    m.created_at,
                    (SELECT COUNT(*) FROM message_contents mc WHERE mc.message_id = m.id) as content_parts,
                    (SELECT char_length(string_agg(mc.text_content, '')) 
                     FROM message_contents mc WHERE mc.message_id = m.id) as total_length
                FROM messages m 
                WHERE m.created_at > NOW() - INTERVAL '30 minutes'
                ORDER BY m.created_at DESC
                LIMIT 5
            """)
            
            logger.info(f"📊 Found {len(result)} recent messages")
            
            for row in result:
                logger.info(f"Message {row['id']}: role={row['role']}, content_parts={row['content_parts']}, total_length={row['total_length']} chars")
                
                if row['role'] == 'assistant' and row['total_length'] and row['total_length'] > 1000:
                    # This looks like a substantial response, let's get a preview
                    content_result = await conn.fetch("""
                        SELECT text_content 
                        FROM message_contents 
                        WHERE message_id = $1 
                        ORDER BY sequence_number
                    """, row['id'])
                    
                    full_content = ''.join([c['text_content'] for c in content_result])
                    
                    logger.info(f"✅ Found substantial assistant response: {len(full_content)} characters")
                    logger.info(f"📝 Preview: {full_content[:200]}...")
                    
                    if len(full_content) > 8000:
                        logger.info("🎉 SUCCESS: Database is storing substantial content correctly!")
                        return True
            
            logger.warning("⚠️ No substantial assistant responses found in recent messages")
            return False
            
    except Exception as e:
        logger.error(f"❌ Database test failed: {e}")
        return False

if __name__ == "__main__":
    result = asyncio.run(test_database_content())
    if result:
        sys.exit(0)  # Success
    else:
        sys.exit(1)  # Failure