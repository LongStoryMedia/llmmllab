"""
Node for hierarchical summary consolidation.

Encapsulates the consolidation capability for Level 2+ summarization,
combining multiple summaries into higher-level abstractions.
"""

from typing import TYPE_CHECKING

from models import SummaryType
from composer.graph.state import WorkflowState
from composer.utils.conversion import convert_langchain_messages_to_messages
from utils.logging import llmmllogger

if TYPE_CHECKING:
    from composer.agents.primary_summary_agent import PrimarySummaryAgent
    from composer.agents.master_summary_agent import MasterSummaryAgent


class ConsolidationNode:
    """
    Node for hierarchical summary consolidation.

    Encapsulates the consolidation capability for Level 2+ summarization,
    combining multiple summaries into higher-level abstractions.
    """

    def __init__(
        self,
        primary_summary_agent: "PrimarySummaryAgent",
        master_summary_agent: "MasterSummaryAgent",
    ):
        self.primary_summary_agent = primary_summary_agent
        self.master_summary_agent = master_summary_agent
        self.logger = llmmllogger.logger.bind(component="ConsolidationNode")

    async def __call__(self, state: WorkflowState) -> WorkflowState:
        """
        Consolidate multiple summaries into higher-level summary.

        Args:
            state: WorkflowState with summaries to consolidate and level information

        Returns:
            Updated state with consolidated summary
        """
        try:
            assert state.user_id
            assert state.user_config
            assert state.user_config.summarization
            sc = state.user_config.summarization

            msg_ids = {m.id for m in state.messages if m.id is not None}
            unsum_msg_ids = msg_ids.difference(s.id for s in state.summaries)
            unsummarized_messages = [m for m in state.messages if m.id in unsum_msg_ids]

            if len(unsummarized_messages) < sc.messages_before_summary:
                self.logger.info(
                    "Not enough new messages for summarization",
                    user_id=state.user_id,
                    unsummarized_count=len(unsummarized_messages),
                    threshold=sc.messages_before_summary,
                )
                return state

            if len(unsummarized_messages) >= sc.messages_before_summary:
                self.logger.info(
                    "Performing primary summarization of new messages",
                    user_id=state.user_id,
                    unsummarized_count=len(unsummarized_messages),
                )
                # Use primary summary agent for conversation summarization
                primary_summary = (
                    await self.primary_summary_agent.summarize_conversation(
                        convert_langchain_messages_to_messages(unsummarized_messages),
                        state.conversation_id or 0,
                    )
                )
                state.summaries.append(primary_summary)

            for lvl in range(1, sc.max_summary_levels):
                lvl_summaries = [s for s in state.summaries if s.level == lvl]

                if len(lvl_summaries) >= sc.summaries_before_consolidation:
                    self.logger.info(
                        "Consolidating summaries",
                        user_id=state.user_id,
                        level=lvl,
                        summary_count=len(lvl_summaries),
                    )
                    # Use master summary agent for consolidation
                    master_summary = (
                        await self.master_summary_agent.consolidate_summaries(
                            lvl_summaries,
                            state.user_id,
                            state.conversation_id or 0,
                            lvl,
                            lvl + 1,
                        )
                    )
                    state.summaries.append(master_summary)

            state.summaries.sort(key=lambda s: s.level, reverse=True)

            return state

        except Exception as e:
            self.logger.error(f"Summary consolidation failed: {e}")
            return state
