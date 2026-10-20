"""M0 instant memory buffer."""

from __future__ import annotations

from collections.abc import Iterable
from uuid import uuid4

from graphiti_core.utils.datetime_utils import utc_now
from h_recongraph.config.memory_policy import MemoryPolicy
from h_recongraph.schemas.memory_unit import SourceSpan, UnifiedMemoryUnit


class InstantMemoryBuffer:
    """Volatile M0 buffer that keeps raw interactions out of the long-term graph."""

    def __init__(self, policy: MemoryPolicy | None = None):
        self.policy = policy or MemoryPolicy()
        self._memories: dict[str, UnifiedMemoryUnit] = {}

    def add(
        self,
        text: str,
        user_id: str,
        session_id: str | None = None,
        metadata: dict | None = None,
    ) -> UnifiedMemoryUnit:
        evidence_id = f'im_{uuid4().hex}'
        now = utc_now()
        memory = UnifiedMemoryUnit(
            memory_id=evidence_id,
            user_id=user_id,
            session_id=session_id,
            layer='M0',
            content_type='episode',
            raw_content=text,
            evidence_ids=[evidence_id],
            source_spans=[
                SourceSpan(evidence_id=evidence_id, text=text, start=0, end=len(text), field='raw')
            ],
            lifecycle_status='volatile',
            created_at=now,
            ttl=self.policy.m0_ttl,
            metadata=metadata or {},
        )
        self._memories[memory.memory_id] = memory
        return memory

    def get(self, memory_id: str) -> UnifiedMemoryUnit | None:
        memory = self._memories.get(memory_id)
        if memory is not None:
            memory.touch()
        return memory

    def list(
        self,
        *,
        user_id: str | None = None,
        session_id: str | None = None,
        include_expired: bool = False,
    ) -> list[UnifiedMemoryUnit]:
        now = utc_now()
        memories = []
        for memory in self._memories.values():
            if user_id is not None and memory.user_id != user_id:
                continue
            if session_id is not None and memory.session_id != session_id:
                continue
            if not include_expired and memory.is_expired(now):
                continue
            memories.append(memory)
        return sorted(memories, key=lambda item: item.created_at, reverse=True)

    def prune_expired(self) -> list[UnifiedMemoryUnit]:
        now = utc_now()
        expired = [memory for memory in self._memories.values() if memory.is_expired(now)]
        for memory in expired:
            memory.lifecycle_status = 'expired'
            del self._memories[memory.memory_id]
        return expired

    def remove_many(self, memory_ids: Iterable[str]) -> None:
        for memory_id in memory_ids:
            self._memories.pop(memory_id, None)
