"""Rule-based semantic state construction."""

import re
from collections.abc import Iterable

from h_recongraph.schemas.memory_unit import SourceSpan, UnifiedMemoryUnit
from h_recongraph.schemas.semantic_state import MemorySemanticState


class SemanticStateConstructor:
    """Build evidence-bounded semantic states from raw memory text.

    This first implementation is intentionally conservative: it only fills slots
    when a source span can be located in the original evidence.
    """

    CAUSE_MARKERS = ('because', '由于', '因为', 'caused by', '导致')
    PATH_MARKERS = (
        'via',
        'through',
        'depends on',
        'connects',
        'leads to',
        'routes',
        '通过',
        '依赖',
        '连接',
        '沿着',
    )
    INTENT_MARKERS = ('为了', '以便', 'in order to', 'so that', '想要')
    CONDITION_MARKERS = (
        '如果',
        '当',
        'when',
        'if',
        'only',
        'unless',
        'must',
        'deadline',
        'during',
        '压力',
        '最近',
        '仅',
        '只有',
        '必须',
        '期间',
    )
    CHANGE_MARKERS = ('开始', '变得', '不再', '最近', 'increase', 'decrease', 'more often')
    TEMPORAL_MARKERS = ('最近', '今天', '昨天', '明天', 'current', 'recent', 'now', '下午')

    def construct(
        self,
        memory: UnifiedMemoryUnit,
        *,
        edge_or_subgraph_id: str | None = None,
    ) -> MemorySemanticState:
        text = memory.raw_content or ''
        spans = [span.model_copy(deep=True) for span in memory.source_spans if span.field != 'raw']
        self._bind_missing_dimensions(text, memory, spans)

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

        # Explicit dimension evidence defines the current condition and cause.
        for dimension, field in (('constraint', 'condition'), ('causal', 'cause')):
            bound = [span for span in memory.source_spans if span.field == dimension]
            if bound:
                spans = [span for span in spans if span.field != field]
                spans.extend(span.model_copy(update={'field': field}) for span in bound)
                value = '; '.join(dict.fromkeys(span.text for span in bound))
                if field == 'condition':
                    condition = value
                else:
                    cause = value

        validity = (
            condition
            if any(span.field == 'constraint' for span in memory.source_spans)
            else temporal_scope or condition
        )
        if validity is not None:
            validity_sources = (
                [span for span in spans if span.field == 'condition']
                if validity == condition
                else [span for span in spans if span.text == validity]
            )
            spans.extend(span.model_copy(update={'field': 'validity'}) for span in validity_sources)

        spans = list(
            {
                (span.evidence_id, span.field, span.text, span.start, span.end): span
                for span in spans
            }.values()
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

    def _bind_missing_dimensions(
        self,
        text: str,
        memory: UnifiedMemoryUnit,
        spans: list[SourceSpan],
    ) -> None:
        """Bind missing dimensions to original paragraphs for later source recovery."""
        bound = {span.field for span in spans}
        paragraphs = self._paragraphs(text, memory.evidence_ids)
        if 'fact' not in bound:
            spans.extend(
                span.model_copy(update={'field': 'fact'}, deep=True) for span in paragraphs
            )
        if 'path' not in bound:
            spans.extend(
                span.model_copy(update={'field': 'path'}, deep=True)
                for span in paragraphs
                if self._contains_marker(span.text, self.PATH_MARKERS)
            )
        if 'constraint' not in bound:
            spans.extend(
                span.model_copy(update={'field': 'constraint'}, deep=True)
                for span in paragraphs
                if self._contains_marker(span.text, self.CONDITION_MARKERS)
            )
        if 'causal' not in bound:
            spans.extend(
                span.model_copy(update={'field': 'causal'}, deep=True)
                for span in paragraphs
                if self._contains_marker(span.text, self.CAUSE_MARKERS)
            )

    @staticmethod
    def _paragraphs(text: str, evidence_ids: list[str]) -> list[SourceSpan]:
        evidence_id = evidence_ids[0] if evidence_ids else ''
        paragraphs: list[SourceSpan] = []
        for match in re.finditer(r'(?s)(.*?)(?:\n\s*\n|\Z)', text):
            raw = match.group(1)
            value = raw.strip()
            if not value:
                continue
            start = match.start(1) + len(raw) - len(raw.lstrip())
            end = match.start(1) + len(raw.rstrip())
            paragraphs.append(
                SourceSpan(
                    evidence_id=evidence_id,
                    text=value,
                    start=start,
                    end=end,
                    field='raw',
                )
            )
        return paragraphs

    @staticmethod
    def _contains_marker(text: str, markers: Iterable[str]) -> bool:
        lower = text.lower()
        return any(marker.lower() in lower for marker in markers)

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
                pos + (delimiter in ('。', '.'))
                for delimiter in ('，', ',', '。', '.', '\n')
                if (pos := text.find(delimiter, start)) != -1
            ]
            end = min(end_candidates) if end_candidates else min(len(text), start + 48)
            value = text[start:end].strip(' ，,')
            start += len(text[start:end]) - len(text[start:end].lstrip(' ，,'))
            end = start + len(value)
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
