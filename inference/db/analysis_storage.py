"""
Storage service for managing analysis entities in the database.
Analyses represent intent analyses associated with messages.
"""

import asyncpg
from typing import List, Optional
from datetime import datetime
from models.intent_analysis import IntentAnalysis
from models.computational_requirement import ComputationalRequirement
from db.db_utils import typed_pool
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
        intent_analysis: IntentAnalysis,
        created_at: Optional[datetime] = None,
    ) -> Optional[int]:
        """
        Add a new analysis to the database.

        Args:
            message_id: ID of the associated message
            intent_analysis: The intent analysis data
            created_at: Optional timestamp (defaults to NOW())

        Returns:
            The ID of the created analysis, or None on failure
        """
        if created_at is None:
            created_at = datetime.utcnow()

        try:
            async with self.typed_pool.acquire() as conn:
                import json

                # Convert list and object fields to JSON strings
                required_capabilities_json = json.dumps(
                    [
                        cap.value if hasattr(cap, "value") else str(cap)
                        for cap in intent_analysis.required_capabilities
                    ]
                )
                # ComputationalRequirement is an enum, so we use .value to get the string
                computational_requirements_json = json.dumps(
                    intent_analysis.computational_requirements.value
                    if hasattr(intent_analysis.computational_requirements, "value")
                    else str(intent_analysis.computational_requirements)
                )

                row = await conn.fetchrow(
                    self.get_query("analysis.add_analysis"),
                    message_id,  # $1
                    (
                        intent_analysis.workflow_type.value
                        if hasattr(intent_analysis.workflow_type, "value")
                        else str(intent_analysis.workflow_type)
                    ),  # $2
                    (
                        intent_analysis.complexity_level.value
                        if hasattr(intent_analysis.complexity_level, "value")
                        else str(intent_analysis.complexity_level)
                    ),  # $3
                    required_capabilities_json,  # $4
                    intent_analysis.domain_specificity,  # $5
                    intent_analysis.reusability_potential,  # $6
                    intent_analysis.confidence,  # $7
                    (
                        intent_analysis.response_format.value
                        if intent_analysis.response_format
                        and hasattr(intent_analysis.response_format, "value")
                        else (
                            str(intent_analysis.response_format)
                            if intent_analysis.response_format
                            else None
                        )
                    ),  # $8
                    (
                        intent_analysis.technical_domain.value
                        if intent_analysis.technical_domain
                        and hasattr(intent_analysis.technical_domain, "value")
                        else (
                            str(intent_analysis.technical_domain)
                            if intent_analysis.technical_domain
                            else None
                        )
                    ),  # $9
                    intent_analysis.requires_tools,  # $10
                    intent_analysis.requires_custom_tools,  # $11
                    intent_analysis.tool_complexity_score,  # $12
                    computational_requirements_json,  # $13
                    created_at,  # $14
                )

                if row:
                    analysis_id = row["id"]
                    self.logger.info(
                        f"Added analysis {analysis_id} ({intent_analysis.workflow_type}) for message {message_id}"
                    )
                    return analysis_id
                else:
                    self.logger.error(
                        f"Failed to add analysis for message {message_id}"
                    )
                    return None

        except Exception as e:
            self.logger.error(f"Error adding analysis for message {message_id}: {e}")
            return None

    async def add_analysis_legacy(
        self,
        message_id: int,
        analysis_data: dict,
        created_at: Optional[datetime] = None,
    ) -> Optional[int]:
        """
        Add a new analysis to the database using legacy analysis_data format.
        This method converts the legacy format to IntentAnalysis.

        Args:
            message_id: ID of the associated message
            analysis_data: The analysis data as dict (legacy format)
            created_at: Optional timestamp (defaults to NOW())

        Returns:
            The ID of the created analysis, or None on failure
        """
        try:
            # Convert legacy format to IntentAnalysis
            intent_analysis = IntentAnalysis(
                workflow_type=analysis_data.get("workflow_type", "unknown"),
                complexity_level=analysis_data.get("complexity_level", "unknown"),
                required_capabilities=analysis_data.get("required_capabilities", []),
                domain_specificity=analysis_data.get("domain_specificity", 0.0),
                reusability_potential=analysis_data.get("reusability_potential", 0.0),
                confidence=analysis_data.get("confidence", 0.0),
                response_format=analysis_data.get("response_format"),
                technical_domain=analysis_data.get("technical_domain"),
                requires_tools=analysis_data.get("requires_tools", False),
                requires_custom_tools=analysis_data.get("requires_custom_tools", False),
                tool_complexity_score=analysis_data.get("tool_complexity_score", 0.0),
                computational_requirements=analysis_data.get(
                    "computational_requirements", {}
                ),
            )

            return await self.add_analysis(message_id, intent_analysis, created_at)

        except Exception as e:
            self.logger.error(
                f"Error converting legacy analysis for message {message_id}: {e}"
            )
            return None

    async def get_analyses_by_message(self, message_id: int) -> List[IntentAnalysis]:
        """
        Retrieve all analyses associated with a message.

        Args:
            message_id: ID of the message

        Returns:
            List of IntentAnalysis objects
        """
        try:
            async with self.typed_pool.acquire() as conn:
                rows = await conn.fetch(
                    self.get_query("analysis.get_by_message"), message_id
                )

                analyses = []
                for row in rows:
                    import json
                    
                    # Parse JSON fields back to objects
                    required_capabilities = (
                        json.loads(row["required_capabilities"])
                        if row["required_capabilities"]
                        else []
                    )
                    # ComputationalRequirement is stored as string value, convert back to enum
                    computational_req_str = (
                        json.loads(row["computational_requirements"])
                        if row["computational_requirements"]
                        else "MINIMAL"
                    )
                    computational_requirements = ComputationalRequirement(computational_req_str)

                    intent_analysis = IntentAnalysis(
                        workflow_type=row["workflow_type"],
                        complexity_level=row["complexity_level"],
                        required_capabilities=required_capabilities,
                        domain_specificity=float(row["domain_specificity"]),
                        reusability_potential=float(row["reusability_potential"]),
                        confidence=float(row["confidence"]),
                        response_format=row["response_format"],
                        technical_domain=row["technical_domain"],
                        requires_tools=row["requires_tools"],
                        requires_custom_tools=row["requires_custom_tools"],
                        tool_complexity_score=float(row["tool_complexity_score"]),
                        computational_requirements=computational_requirements,
                    )
                    analyses.append(intent_analysis)

                self.logger.debug(
                    f"Retrieved {len(analyses)} analyses for message {message_id}"
                )
                return analyses

        except Exception as e:
            self.logger.error(
                f"Error retrieving analyses for message {message_id}: {e}"
            )
            return []

    async def get_analyses_by_message_legacy(self, message_id: int) -> List[dict]:
        """
        Retrieve all analyses associated with a message in legacy dict format.

        Args:
            message_id: ID of the message

        Returns:
            List of analysis dictionaries (legacy format)
        """
        try:
            intent_analyses = await self.get_analyses_by_message(message_id)

            # Convert IntentAnalysis objects back to legacy dict format
            analyses = []
            for ia in intent_analyses:
                analysis = {
                    "workflow_type": (
                        ia.workflow_type.value
                        if hasattr(ia.workflow_type, "value")
                        else str(ia.workflow_type)
                    ),
                    "complexity_level": (
                        ia.complexity_level.value
                        if hasattr(ia.complexity_level, "value")
                        else str(ia.complexity_level)
                    ),
                    "required_capabilities": [
                        cap.value if hasattr(cap, "value") else str(cap)
                        for cap in ia.required_capabilities
                    ],
                    "domain_specificity": ia.domain_specificity,
                    "reusability_potential": ia.reusability_potential,
                    "confidence": ia.confidence,
                    "response_format": (
                        ia.response_format.value
                        if ia.response_format and hasattr(ia.response_format, "value")
                        else str(ia.response_format) if ia.response_format else None
                    ),
                    "technical_domain": (
                        ia.technical_domain.value
                        if ia.technical_domain and hasattr(ia.technical_domain, "value")
                        else str(ia.technical_domain) if ia.technical_domain else None
                    ),
                    "requires_tools": ia.requires_tools,
                    "requires_custom_tools": ia.requires_custom_tools,
                    "tool_complexity_score": ia.tool_complexity_score,
                    "computational_requirements": (
                        ia.computational_requirements.value
                        if hasattr(ia.computational_requirements, "value")
                        else str(ia.computational_requirements)
                    ),
                }
                analyses.append(analysis)

            return analyses

        except Exception as e:
            self.logger.error(
                f"Error retrieving legacy analyses for message {message_id}: {e}"
            )
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
                await conn.execute(
                    self.get_query("analysis.delete_by_message"), message_id
                )

                self.logger.info(f"Deleted analyses for message {message_id}")
                return True

        except Exception as e:
            self.logger.error(f"Error deleting analyses for message {message_id}: {e}")
            return False
