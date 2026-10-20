"""Hierarchical memory evolution rules."""

import math
from datetime import datetime

from graphiti_core.utils.datetime_utils import utc_now
from h_recongraph.config.memory_policy import MemoryPolicy
from h_recongraph.schemas.memory_unit import MemoryScores, UnifiedMemoryUnit


class MemoryEvolutionEngine:
    """Rule-based promote, consolidate, archive, and forget decisions."""

    PREFERENCE_TERMS = ('记住', '以后', '偏好', '喜欢', '不喜欢', 'prefer', 'remember', 'always')
    TASK_TERMS = ('任务', '项目', 'deadline', 'todo', '计划', '需要', 'project', 'task')
    STATE_CHANGE_TERMS = ('最近', '开始', '变得', '不再', 'changed', 'now', 'recently')
    PRIVATE_TERMS = ('密码', 'secret', 'token', 'api key', '身份证')
    CONFIRMATION_TERMS = ('记住这个', '请记住', 'remember this', 'save this')

    def __init__(self, policy: MemoryPolicy | None = None):
        self.policy = policy or MemoryPolicy()

    def score_m0(self, memory: UnifiedMemoryUnit, query: str | None = None) -> MemoryScores:
        text = memory.raw_content or ''
        lower = text.lower()
        salience = 0.18
        importance = 0.18
        task_relevance = 0.0
        confidence = 0.45
        user_confirmation = 0.0

        if self._contains_any(lower, self.PREFERENCE_TERMS):
            salience += 0.34
            importance += 0.3
        if self._contains_any(lower, self.TASK_TERMS):
            salience += 0.2
            importance += 0.18
            task_relevance += 0.35
        if self._contains_any(lower, self.STATE_CHANGE_TERMS):
            salience += 0.2
            confidence += 0.12
        if self._contains_any(lower, self.CONFIRMATION_TERMS):
            salience += 0.25
            importance += 0.25
            user_confirmation = 1.0
        if query:
            task_relevance = max(task_relevance, self._token_overlap(text, query))

        privacy_risk = 0.55 if self._contains_any(lower, self.PRIVATE_TERMS) else 0.0
        return MemoryScores(
            salience=round(min(salience, 1.0), 3),
            importance=round(min(importance, 1.0), 3),
            recency=1.0,
            confidence=round(min(confidence, 1.0), 3),
            task_relevance=round(min(task_relevance, 1.0), 3),
            evidence_strength=1.0 if memory.evidence_ids else 0.0,
            privacy_risk=privacy_risk,
            user_confirmation=user_confirmation,
        )

    def should_promote_to_m1(self, memory: UnifiedMemoryUnit) -> bool:
        scores = memory.scores
        text = (memory.raw_content or '').lower()
        return (
            scores.salience >= self.policy.promotion_salience_threshold
            or scores.task_relevance >= self.policy.promotion_task_relevance_threshold
            or self._contains_any(text, self.PREFERENCE_TERMS)
            or self._contains_any(text, self.STATE_CHANGE_TERMS)
        )

    def promote_to_m1(self, memory: UnifiedMemoryUnit) -> UnifiedMemoryUnit:
        data = memory.model_dump()
        promoted = UnifiedMemoryUnit(**data)
        promoted.memory_id = memory.memory_id.replace('im_', 'stm_', 1)
        promoted.layer = 'M1'
        promoted.lifecycle_status = 'active_short_term'
        promoted.ttl = self.policy.m1_ttl
        promoted.metadata = {**memory.metadata, 'promoted_from': memory.memory_id}
        return promoted

    def consolidation_score(self, memory: UnifiedMemoryUnit, now: datetime | None = None) -> float:
        now = now or utc_now()
        age_seconds = max((now - memory.created_at).total_seconds(), 0.0)
        recency = math.exp(-age_seconds / max(self.policy.m1_ttl.total_seconds(), 1.0))
        retrieval_usage = min(memory.access_count / 5, 1.0)
        scores = memory.scores
        score = (
            self.policy.frequency_weight * min(memory.access_count / 3, 1.0)
            + self.policy.recency_weight * recency
            + self.policy.importance_weight * scores.importance
            + self.policy.evidence_strength_weight * scores.evidence_strength
            + self.policy.retrieval_usage_weight * retrieval_usage
            + self.policy.user_confirmation_weight * scores.user_confirmation
            - self.policy.contradiction_risk_weight * scores.contradiction_risk
        )
        return round(max(0.0, min(score, 1.0)), 3)

    def should_consolidate_to_m2(self, memory: UnifiedMemoryUnit) -> bool:
        return self.consolidation_score(memory) >= self.policy.consolidation_threshold

    def consolidate_to_m2(self, memory: UnifiedMemoryUnit) -> UnifiedMemoryUnit:
        data = memory.model_dump()
        consolidated = UnifiedMemoryUnit(**data)
        consolidated.memory_id = memory.memory_id.replace('stm_', 'ltm_', 1)
        if consolidated.memory_id == memory.memory_id:
            consolidated.memory_id = f'ltm_{memory.memory_id}'
        consolidated.layer = 'M2'
        consolidated.lifecycle_status = 'consolidated'
        consolidated.ttl = None
        consolidated.metadata = {**memory.metadata, 'consolidated_from': memory.memory_id}
        return consolidated

    def forget_score(self, memory: UnifiedMemoryUnit, now: datetime | None = None) -> float:
        now = now or utc_now()
        ttl = memory.ttl or self.policy.m1_ttl
        age = min(max((now - memory.created_at).total_seconds(), 0.0) / ttl.total_seconds(), 1.0)
        low_usage = 1.0 - min(memory.access_count / 3, 1.0)
        low_confidence = 1.0 - memory.scores.confidence
        score = (
            self.policy.age_forgetting_weight * age
            + self.policy.low_usage_forgetting_weight * low_usage
            + self.policy.low_confidence_forgetting_weight * low_confidence
            + self.policy.contradiction_forgetting_weight * memory.scores.contradiction_risk
            + self.policy.privacy_forgetting_weight * memory.scores.privacy_risk
            - self.policy.importance_retention_weight * memory.scores.importance
            - self.policy.evidence_retention_weight * memory.scores.evidence_strength
            - self.policy.user_pin_retention_weight * memory.scores.user_confirmation
        )
        return round(max(0.0, min(score, 1.0)), 3)

    def archive_if_stale(self, memory: UnifiedMemoryUnit) -> bool:
        if memory.layer == 'M2':
            return False
        if self.forget_score(memory) < self.policy.archive_threshold:
            return False
        memory.lifecycle_status = 'archived'
        return True

    def _contains_any(self, text: str, terms: tuple[str, ...]) -> bool:
        return any(term.lower() in text for term in terms)

    def _token_overlap(self, left: str, right: str) -> float:
        left_tokens = {token for token in left.lower().split() if token}
        right_tokens = {token for token in right.lower().split() if token}
        if not left_tokens or not right_tokens:
            return 0.0
        return len(left_tokens & right_tokens) / len(right_tokens)
