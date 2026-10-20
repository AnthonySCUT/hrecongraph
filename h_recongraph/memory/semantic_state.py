"""Rule-based semantic state construction."""

from collections.abc import Iterable

from h_recongraph.schemas.memory_unit import SourceSpan, UnifiedMemoryUnit
from h_recongraph.schemas.semantic_state import MemorySemanticState


class SemanticStateConstructor:
    """Build evidence-bounded semantic states from raw memory text.

    This first implementation is intentionally conservative: it only fills slots
    when a source span can be located in the original evidence.
    """

    CAUSE_MARKERS = ('because', '由于', '因为', 'caused by', '导致')
    INTENT_MARKERS = ('为了', '以便', 'in order to', 'so that', '想要')
    CONDITION_MARKERS = ('如果', '当', 'when', 'if', 'deadline', '压力', '最近')
    CHANGE_MARKERS = ('开始', '变得', '不再', '最近', 'increase', 'decrease', 'more often')
    TEMPORAL_MARKERS = ('最近', '今天', '昨天', '明天', 'current', 'recent', 'now', '下午')

    def construct(
        self,
        memory: UnifiedMemoryUnit,
        *,
        edge_or_subgraph_id: str | None = None,
    ) -> MemorySemanticState:
        text = memory.raw_content or ''
        spans: list[SourceSpan] = []

        temporal_scope = self._span_for_any(
            text, self.TEMPORAL_MARKERS, memory.evidence_ids, 'temporal_scope', spans
        )
        cause = self._clause_after_any(
            text, self.CAUSE_MARKERS, memory.evidence_ids, 'cause', spans
        )
        intent = self._clause_after_any(
            text, self.INTENT_MARKERS, memory.evidence_ids, 'intent', spans
        )
        condition = self._span_for_any(
            text, self.CONDITION_MARKERS, memory.evidence_ids, 'condition', spans
        )
        state_change = self._span_for_any(
            text, self.CHANGE_MARKERS, memory.evidence_ids, 'state_change', spans
        )

        validity = temporal_scope or condition
        if validity is not None:
            validity_source = next(
                (span for span in spans if span.text == validity),
                None,
            )
            if validity_source is not None:
                spans.append(
                    SourceSpan(
                        evidence_id=validity_source.evidence_id,
                        text=validity_source.text,
                        start=validity_source.start,
                        end=validity_source.end,
                        field='validity',
                    )
                )

        confidence = max(memory.scores.confidence, memory.scores.salience, 0.1 if spans else 0.0)
        state_type = 'evidence_supported' if spans else 'unsupported'
        return MemorySemanticState(
            edge_or_subgraph_id=edge_or_subgraph_id or memory.memory_id,
            temporal_scope=temporal_scope,
            cause=cause,
            effect=None,
            intent=intent,
            condition=condition,
            state_change=state_change,
            validity=validity,
            confidence=round(min(confidence, 1.0), 3),
            evidence_ids=memory.evidence_ids,
            source_spans=spans,
            state_type=state_type,
        )

    def _span_for_any(
        self,
        text: str,
        markers: Iterable[str],
        evidence_ids: list[str],
        field: str,
        spans: list[SourceSpan],
    ) -> str | None:
        lower = text.lower()
        for marker in markers:
            index = lower.find(marker.lower())
            if index == -1:
                continue
            end = min(len(text), index + max(len(marker), 12))
            value = text[index:end].strip()
            spans.append(
                SourceSpan(
                    evidence_id=evidence_ids[0] if evidence_ids else '',
                    text=value,
                    start=index,
                    end=end,
                    field=field,
                )
            )
            return value
        return None

    def _clause_after_any(
        self,
        text: str,
        markers: Iterable[str],
        evidence_ids: list[str],
        field: str,
        spans: list[SourceSpan],
    ) -> str | None:
        lower = text.lower()
        for marker in markers:
            index = lower.find(marker.lower())
            if index == -1:
                continue
            start = index + len(marker)
            end_candidates = [
                pos for pos in [text.find('，', start), text.find(',', start)] if pos != -1
            ]
            end = min(end_candidates) if end_candidates else min(len(text), start + 48)
            value = text[start:end].strip(' ，,')
            if not value:
                value = text[index : min(len(text), index + len(marker) + 12)].strip()
                start = index
                end = start + len(value)
            spans.append(
                SourceSpan(
                    evidence_id=evidence_ids[0] if evidence_ids else '',
                    text=value,
                    start=start,
                    end=end,
                    field=field,
                )
            )
            return value
        return None
