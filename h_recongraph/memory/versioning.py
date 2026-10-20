"""Long-term memory version management."""

from h_recongraph.schemas.conflict import ConflictAssessment
from h_recongraph.schemas.memory_unit import UnifiedMemoryUnit


class MemoryVersionManager:
    """Apply versioning rules without physically deleting long-term memories."""

    def apply(
        self,
        new_memory: UnifiedMemoryUnit,
        existing_memory: UnifiedMemoryUnit | None,
        assessment: ConflictAssessment,
    ) -> UnifiedMemoryUnit:
        new_memory.metadata = {
            **new_memory.metadata,
            'update_relation': assessment.relation_type,
            'update_rationale': assessment.rationale,
        }
        new_memory.scores.contradiction_risk = assessment.contradiction_risk

        if existing_memory is None:
            return new_memory

        if assessment.should_merge:
            new_memory.version = existing_memory.version
            new_memory.supersedes = list(dict.fromkeys(existing_memory.supersedes))
            new_memory.evidence_ids = list(
                dict.fromkeys([*existing_memory.evidence_ids, *new_memory.evidence_ids])
            )
            new_memory.metadata = {
                **new_memory.metadata,
                'merged_with': existing_memory.memory_id,
            }
            return new_memory

        if not assessment.should_version:
            return new_memory

        new_memory.version = existing_memory.version + 1
        new_memory.supersedes = list(
            dict.fromkeys([*new_memory.supersedes, existing_memory.memory_id])
        )
        new_memory.metadata = {
            **new_memory.metadata,
            'previous_version': existing_memory.memory_id,
        }

        if assessment.relation_type == 'contradiction':
            existing_memory.lifecycle_status = 'archived'
            existing_memory.metadata = {
                **existing_memory.metadata,
                'archived_by': new_memory.memory_id,
                'archive_reason': 'contradicted_by_new_evidence',
            }
        else:
            existing_memory.lifecycle_status = 'revised'
            existing_memory.metadata = {
                **existing_memory.metadata,
                'revised_by': new_memory.memory_id,
                'revision_reason': assessment.relation_type,
            }
        return new_memory
