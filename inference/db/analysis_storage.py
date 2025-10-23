"""
Storage service for managing analysis entities in the database.
Analyses represent intent analyses associated with messages.
"""

import asyncpg
from typing import List, Optional
from datetime import datetime
from models.intent_analysis import IntentAnalysis
from db.db_utils import TypedConnection, typed_pool
from utils.logging import llmmllogger

logger = llmmllogger.bind(component="analysis_storage")


class AnalysisStorage:
    """Storage service for analysis entities with CRUD operations."""
    
    def __init__(self, pool: asyncpg.Pool, get_query):
        self.pool = pool
        self.typed_pool = typed_pool(pool)
        self.get_query = get_query
        self.logger = llmmllogger.bind(component="analysis_storage_instance")

    async def add_analysis(
        self,
        message_id: int,
        analysis_data: dict,
        created_at: Optional[datetime] = None,
    ) -> Optional[int]:
        """
        Add a new analysis to the database.
        
        Args:
            message_id: ID of the associated message
            analysis_data: The intent analysis data as JSON
            created_at: Optional timestamp (defaults to NOW())
            
        Returns:
            The ID of the created analysis, or None on failure
        """
        if created_at is None:
            created_at = datetime.utcnow()
            
        try:
            async with self.typed_pool.acquire() as conn:
                    import json
                    analysis_json = json.dumps(analysis_data) if isinstance(analysis_data, dict) else analysis_data
                    row = await conn.fetchrow(
                        self.get_query("analysis.add_analysis"),
                        message_id,
                        analysis_json,
                        created_at
                    )
                    
                    if row:
                        analysis_id = row["id"]
                        self.logger.info(f"Added analysis {analysis_id} for message {message_id}")
                        return analysis_id
                    else:
                        self.logger.error(f"Failed to add analysis for message {message_id}")
                        return None
                    
        except Exception as e:
            self.logger.error(f"Error adding analysis for message {message_id}: {e}")
            return None

    async def get_analyses_by_message(self, message_id: int) -> List[dict]:
        """
        Retrieve all analyses associated with a message.
        
        Args:
            message_id: ID of the message
            
        Returns:
            List of analysis dictionaries
        """
        try:
            async with self.typed_pool.acquire() as conn:
                rows = await conn.fetch(
                    self.get_query("analysis.get_by_message"),
                    message_id
                )
                
                analyses = []
                for row in rows:
                    analysis = {
                        "id": row["id"],
                        "message_id": row["message_id"],
                        "analysis_data": row["analysis_data"],
                        "created_at": row["created_at"]
                    }
                    analyses.append(analysis)
                    
                self.logger.debug(f"Retrieved {len(analyses)} analyses for message {message_id}")
                return analyses
                
        except Exception as e:
            self.logger.error(f"Error retrieving analyses for message {message_id}: {e}")
            return []

    async def delete_analyses_by_message(self, message_id: int) -> bool:
        """
        Delete all analyses associated with a message.
        
        Args:
            message_id: ID of the message
            
        Returns:
            True if deletion was successful, False otherwise
        """
        try:
            async with self.typed_pool.acquire() as conn:
                result = await conn.execute(
                    self.get_query("analysis.delete_by_message"),
                    message_id
                )
                
                self.logger.info(f"Deleted analyses for message {message_id}")
                return True
                
        except Exception as e:
            self.logger.error(f"Error deleting analyses for message {message_id}: {e}")
            return False