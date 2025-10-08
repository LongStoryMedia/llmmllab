"""
Node for hierarchical summary consolidation.

Encapsulates the consolidation capability for Level 2+ summarization,
combining multiple summaries into higher-level abstractions.
"""

from typing import List

from models import SummaryType
from runner import PipelineFactory
from composer.graph.state import WorkflowState
from composer.utils.conversion import convert_langchain_messages_to_messages
from .base_summary_node import BaseSummaryNode


class ConsolidationNode(BaseSummaryNode):
    """
    Node for hierarchical summary consolidation.

    Encapsulates the consolidation capability for Level 2+ summarization,
    combining multiple summaries into higher-level abstractions.
    """

    def __init__(self, pipeline_factory: PipelineFactory):
        super().__init__("ConsolidationNode", pipeline_factory=pipeline_factory)

    async def __call__(self, state: WorkflowState) -> WorkflowState:
        """
        Consolidate multiple summaries into higher-level summary.

        Args:
            state: WorkflowState with summaries to consolidate and level information

        Returns:
            Updated state with consolidated summary
        """
        try:
            user_id = self._validate_user_id(state)

            # Extract consolidation parameters from user config or use defaults
            assert state.user_config
            assert state.user_config.summarization
            sc = state.user_config.summarization

            msg_ids = {m.id for m in state.messages if m.id is not None}
            unsum_msg_ids = msg_ids.difference(s.id for s in state.summaries)
            unsummarized_messages = [m for m in state.messages if m.id in unsum_msg_ids]

            if len(unsummarized_messages) < sc.messages_before_summary:
                self.logger.info(
                    "Not enough new messages for summarization",
                    user_id=user_id,
                    unsummarized_count=len(unsummarized_messages),
                    threshold=sc.messages_before_summary,
                )
                return state

            if len(unsummarized_messages) >= sc.messages_before_summary:
                self.logger.info(
                    "Performing primary summarization of new messages",
                    user_id=user_id,
                    unsummarized_count=len(unsummarized_messages),
                )
                state.summaries.append(
                    await self.agent.summarize_conversation(
                        convert_langchain_messages_to_messages(unsummarized_messages),
                        user_id,
                        state.conversation_id or 0,
                        SummaryType.PRIMARY,
                    )
                )

            for lvl in range(1, sc.max_summary_levels):
                lvl_summaries = [s for s in state.summaries if s.level == lvl]

                if len(lvl_summaries) >= sc.summaries_before_consolidation:
                    self.logger.info(
                        "Consolidating summaries",
                        user_id=user_id,
                        level=lvl,
                        summary_count=len(lvl_summaries),
                    )
                    state.summaries.append(
                        await self.agent.consolidate_summaries(
                            lvl_summaries,
                            user_id,
                            state.conversation_id or 0,
                            SummaryType.MASTER,
                            lvl,
                            lvl + 1,
                        )
                    )

            state.summaries.sort(key=lambda s: s.level, reverse=True)

            return state

        except Exception as e:
            return self._handle_error(state, e, "Summary consolidation")
