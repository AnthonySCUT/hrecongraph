"""Optional LLM evaluator for memory relationship classification."""

from typing import Any

from pydantic import BaseModel, Field

from graphiti_core.prompts.models import Message
from h_recongraph.schemas.conflict import ConflictAssessment, MemoryRelationType
from h_recongraph.schemas.memory_unit import UnifiedMemoryUnit


class MemoryRelationEvaluation(BaseModel):
    relation_type: MemoryRelationType
    confidence: float = Field(ge=0.0, le=1.0)
    contradiction_risk: float = Field(default=0.0, ge=0.0, le=1.0)
    rationale: str = ''


class LLMMemoryEvaluator:
    """LLM-backed evaluator that can refine rule-based conflict assessments."""

    def __init__(self, llm_client: Any | None = None):
        self.llm_client = llm_client

    async def evaluate_relation(
        self,
        new_memory: UnifiedMemoryUnit,
        existing_memory: UnifiedMemoryUnit | None,
        fallback: ConflictAssessment,
    ) -> ConflictAssessment:
        if self.llm_client is None or existing_memory is None:
            return fallback

        messages = [
            Message(
                role='system',
                content=(
                    'Classify the relationship between two evidence-grounded agent memories. '
                    'Prefer temporal_update or exception over contradiction when time or condition '
                    'scope explains the difference. Return only structured JSON.'
                ),
            ),
            Message(
                role='user',
                content=(
                    f'Existing memory: {existing_memory.raw_content}\n'
                    f'New memory: {new_memory.raw_content}\n'
                    'Allowed relation_type values: duplicate, elaboration, temporal_update, '
                    'contradiction, exception, new_fact.'
                ),
            ),
        ]
        response = await self.llm_client.generate_response(
            messages,
            response_model=MemoryRelationEvaluation,
        )
        evaluation = MemoryRelationEvaluation(**response)
        return ConflictAssessment(
            relation_type=evaluation.relation_type,
            confidence=evaluation.confidence,
            contradiction_risk=evaluation.contradiction_risk,
            matched_memory_id=fallback.matched_memory_id,
            rationale=evaluation.rationale,
            should_version=evaluation.relation_type
            in {'temporal_update', 'contradiction', 'exception'},
            should_merge=evaluation.relation_type in {'duplicate', 'elaboration'},
        )
