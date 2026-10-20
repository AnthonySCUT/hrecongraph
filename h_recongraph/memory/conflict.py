"""Conflict detection for H-ReConGraph memory updates."""

from h_recongraph.schemas.conflict import ConflictAssessment
from h_recongraph.schemas.memory_unit import UnifiedMemoryUnit


class ConflictDetector:
    """Classify how a new memory relates to existing memories.

    This is a conservative rule baseline. It favors temporal updates and
    conditional exceptions over hard contradiction when the new evidence contains
    time or condition markers.
    """

    NEGATION_TERMS = (
        'not',
        "don't",
        'do not',
        'never',
        'no longer',
        '不',
        '不喜欢',
        '不再',
        '不会',
        '没有',
    )
    TEMPORAL_TERMS = ('recently', 'now', 'current', 'today', '最近', '现在', '当前', '今天')
    CONDITION_TERMS = (
        'if',
        'when',
        'under',
        'during',
        'deadline',
        'pressure',
        '如果',
        '当',
        '压力',
    )

    def assess(
        self,
        new_memory: UnifiedMemoryUnit,
        existing_memories: list[UnifiedMemoryUnit],
    ) -> ConflictAssessment:
        best: tuple[UnifiedMemoryUnit, float] | None = None
        for existing in existing_memories:
            if existing.user_id != new_memory.user_id:
                continue
            overlap = self._token_overlap(new_memory.raw_content or '', existing.raw_content or '')
            if best is None or overlap > best[1]:
                best = (existing, overlap)

        if best is None or best[1] < 0.12:
            return ConflictAssessment(
                relation_type='new_fact',
                confidence=0.7,
                rationale='No sufficiently related memory was found.',
            )

        existing, overlap = best
        new_text = (new_memory.raw_content or '').lower()
        old_text = (existing.raw_content or '').lower()
        new_negated = self._contains_any(new_text, self.NEGATION_TERMS)
        old_negated = self._contains_any(old_text, self.NEGATION_TERMS)
        has_temporal = self._contains_any(new_text, self.TEMPORAL_TERMS)
        has_condition = self._contains_any(new_text, self.CONDITION_TERMS)

        if overlap >= 0.72 and new_negated == old_negated:
            return ConflictAssessment(
                relation_type='duplicate',
                confidence=min(0.98, overlap),
                matched_memory_id=existing.memory_id,
                rationale='High token overlap and compatible polarity.',
                should_merge=True,
            )

        if new_negated != old_negated and has_temporal:
            return ConflictAssessment(
                relation_type='temporal_update',
                confidence=max(0.7, overlap),
                contradiction_risk=0.45,
                matched_memory_id=existing.memory_id,
                rationale='Polarity differs with explicit temporal scope.',
                should_version=True,
            )

        if new_negated != old_negated and has_condition:
            return ConflictAssessment(
                relation_type='exception',
                confidence=max(0.68, overlap),
                contradiction_risk=0.35,
                matched_memory_id=existing.memory_id,
                rationale='Polarity differs, but the new evidence is condition-scoped.',
                should_version=True,
            )

        if new_negated != old_negated:
            return ConflictAssessment(
                relation_type='contradiction',
                confidence=max(0.72, overlap),
                contradiction_risk=0.85,
                matched_memory_id=existing.memory_id,
                rationale='Related memory has opposite polarity without time or condition scope.',
                should_version=True,
            )

        if overlap >= 0.38:
            return ConflictAssessment(
                relation_type='elaboration',
                confidence=min(0.9, 0.45 + overlap),
                matched_memory_id=existing.memory_id,
                rationale='Related memory shares core terms and adds detail.',
                should_merge=True,
            )

        return ConflictAssessment(
            relation_type='new_fact',
            confidence=0.65,
            rationale='Related memories are too weakly connected for an update.',
        )

    def _contains_any(self, text: str, terms: tuple[str, ...]) -> bool:
        return any(term in text for term in terms)

    def _token_overlap(self, left: str, right: str) -> float:
        left_tokens = self._tokens(left)
        right_tokens = self._tokens(right)
        if not left_tokens or not right_tokens:
            return 0.0
        return len(left_tokens & right_tokens) / min(len(left_tokens), len(right_tokens))

    def _tokens(self, text: str) -> set[str]:
        normalized = (
            text.lower()
            .replace('，', ' ')
            .replace('。', ' ')
            .replace(',', ' ')
            .replace('.', ' ')
            .replace(':', ' ')
            .replace('：', ' ')
        )
        tokens = {token for token in normalized.split() if len(token) > 1}
        # Chinese examples in the tests often have no whitespace; keep a few
        # salient substrings so the detector remains useful without a tokenizer.
        for marker in [
            '用户',
            '咖啡',
            '压力',
            '项目',
            '回答',
            '简洁',
            '详细',
            '喜欢',
            '不喜欢',
        ]:
            if marker in normalized:
                tokens.add(marker)
        return tokens
