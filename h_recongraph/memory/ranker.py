"""Memory utility ranker."""

import re
from typing import Any

from h_recongraph.schemas.memory_unit import UnifiedMemoryUnit


class MemoryUtilityRanker:
    """Rank memory units by reasoning utility rather than plain similarity."""

    LEXICAL_STOPWORDS = frozenset(
        'a an the and or but of to in on at for from with by as '  # noqa: SIM905
        'is are was were be been being do does did have has had '
        'i me my we our you your he him his she her it its they them their '
        'this that these those what which who whom whose when where why how '
        'can could would should will shall may might must '
        'please tell about just also 的 了 是 吗 呢 啊 和 与 或 在 有 我 你 他 她 它 '
        '我们 你们 他们 什么 怎么 如何 为什么 请 告诉 这个 那个'.split()
    )

    def lexical_relevance(self, text: str, query: str) -> float:
        """Require content-word overlap before admitting local fallback candidates."""
        return self._token_overlap(' '.join(self.tokenize(text)), ' '.join(self.tokenize(query)))

    def tokenize(self, text: str) -> set[str]:
        tokens: set[str] = set()
        for token in re.findall(r'[\u4e00-\u9fff]+|[a-z0-9]+(?:[-_][a-z0-9]+)*', text.lower()):
            if token in self.LEXICAL_STOPWORDS:
                continue
            if '\u4e00' <= token[0] <= '\u9fff':
                tokens.update(
                    token[i : i + 2]
                    for i in range(len(token) - 1)
                    if token[i : i + 2] not in self.LEXICAL_STOPWORDS
                    and any(char not in self.LEXICAL_STOPWORDS for char in token[i : i + 2])
                )
            elif len(token) > 1:
                tokens.add(token)
        return tokens

    def score_memory(self, memory: UnifiedMemoryUnit, query: str) -> float:
        semantic_relevance = self.lexical_relevance(memory.raw_content or '', query)
        scores = memory.scores
        score = (
            semantic_relevance
            + scores.retrieval_utility
            + scores.recency
            + scores.evidence_strength
            + scores.importance
            - scores.contradiction_risk
            - scores.privacy_risk
        )
        return round(max(score, 0.0), 3)

    def rank_memories(
        self, memories: list[UnifiedMemoryUnit], query: str
    ) -> list[UnifiedMemoryUnit]:
        return sorted(memories, key=lambda memory: self.score_memory(memory, query), reverse=True)

    def rank_graph_edges(self, edges: list[Any], query: str) -> list[Any]:
        return sorted(edges, key=lambda edge: self._edge_score(edge, query), reverse=True)

    def _edge_score(self, edge: Any, query: str) -> float:
        fact = getattr(edge, 'fact', None) or getattr(edge, 'name', None) or str(edge)
        return self._token_overlap(fact, query)

    def _token_overlap(self, left: str, right: str) -> float:
        left_tokens = {token for token in left.lower().split() if token}
        right_tokens = {token for token in right.lower().split() if token}
        if not left_tokens or not right_tokens:
            return 0.0
        return len(left_tokens & right_tokens) / len(right_tokens)
