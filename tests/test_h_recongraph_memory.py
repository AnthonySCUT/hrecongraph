from datetime import timedelta

import pytest

from h_recongraph import HReConGraphMemory, MemoryPolicy
from h_recongraph.memory.conflict import ConflictDetector
from h_recongraph.memory.llm_evaluator import LLMMemoryEvaluator
from h_recongraph.memory.semantic_state import SemanticStateConstructor
from h_recongraph.schemas.memory_unit import UnifiedMemoryUnit


class FakeGraphiti:
    def __init__(self):
        self.added = []
        self.searches = []

    async def add_episode(self, **kwargs):
        self.added.append(kwargs)
        return kwargs

    async def search(self, **kwargs):
        self.searches.append(kwargs)
        return []


class FakeLLMClient:
    def __init__(self, response):
        self.response = response
        self.calls = []

    async def generate_response(self, messages, response_model=None):
        self.calls.append((messages, response_model))
        return self.response


@pytest.mark.asyncio
async def test_add_interaction_promotes_salient_memory_to_m1():
    fake_graphiti = FakeGraphiti()
    memory = HReConGraphMemory(fake_graphiti)

    result = await memory.add_interaction(
        '用户最近项目压力大，下午经常喝咖啡提神',
        user_id='u1',
        session_id='s1',
    )

    assert result.layer == 'M1'
    assert result.lifecycle_status == 'active_short_term'
    assert result.semantic_state_id is not None
    assert len(fake_graphiti.added) == 1
    assert 'layer=M1' in fake_graphiti.added[0]['source_description']
    assert fake_graphiti.added[0]['group_id'] == 'user:u1'


@pytest.mark.asyncio
async def test_low_salience_interaction_stays_in_m0():
    fake_graphiti = FakeGraphiti()
    memory = HReConGraphMemory(fake_graphiti)

    result = await memory.add_interaction('好的', user_id='u1', session_id='s1')

    assert result.layer == 'M0'
    assert result.lifecycle_status == 'volatile'
    assert fake_graphiti.added == []


@pytest.mark.asyncio
async def test_consolidate_promotes_eligible_m1_to_m2():
    fake_graphiti = FakeGraphiti()
    policy = MemoryPolicy(consolidation_threshold=0.7)
    memory = HReConGraphMemory(fake_graphiti, policy=policy)

    short_term = await memory.add_interaction(
        '请记住这个：以后回答尽量简洁',
        user_id='u1',
        session_id='s1',
    )
    short_term.access_count = 4

    consolidated = await memory.consolidate(user_id='u1', session_id='s1')

    assert len(consolidated) == 1
    assert consolidated[0].layer == 'M2'
    assert consolidated[0].lifecycle_status == 'consolidated'
    assert len(fake_graphiti.added) == 2
    assert 'layer=M2' in fake_graphiti.added[-1]['source_description']


@pytest.mark.asyncio
async def test_retrieve_builds_cross_layer_context_with_ephemeral_edges():
    fake_graphiti = FakeGraphiti()
    memory = HReConGraphMemory(fake_graphiti)
    await memory.add_interaction(
        'current project deadline causes afternoon coffee',
        user_id='u1',
        session_id='s1',
    )

    context = await memory.retrieve(
        'why current project coffee',
        user_id='u1',
        session_id='s1',
    )

    assert context.selected_layers == ['M0', 'M1', 'M2']
    assert context.instant_memories
    assert context.ephemeral_edges
    assert context.evidence_trace
    assert fake_graphiti.searches[0]['group_ids'] == ['user:u1']


def test_semantic_state_requires_evidence_bound_fields():
    constructor = SemanticStateConstructor()
    memory = UnifiedMemoryUnit(
        user_id='u1',
        session_id='s1',
        layer='M1',
        raw_content='因为项目压力大，所以最近下午经常喝咖啡提神',
        evidence_ids=['im_1'],
    )

    state = constructor.construct(memory)

    assert state.state_type == 'evidence_supported'
    assert state.cause is not None
    assert state.validity is not None
    assert any(span.field == 'cause' for span in state.source_spans)
    assert any(span.field == 'validity' for span in state.source_spans)


@pytest.mark.asyncio
async def test_maintain_expires_m0_buffer_items():
    memory = HReConGraphMemory(policy=MemoryPolicy(m0_ttl=timedelta(microseconds=1)))
    instant = await memory.add_interaction('好的', user_id='u1', session_id='s1')
    instant.created_at -= timedelta(seconds=1)

    result = await memory.maintain(user_id='u1')

    assert len(result['expired_m0']) == 1


def test_conflict_detector_prefers_temporal_update_for_recent_change():
    detector = ConflictDetector()
    old_memory = UnifiedMemoryUnit(
        user_id='u1',
        layer='M2',
        raw_content='user does not drink coffee',
        lifecycle_status='consolidated',
    )
    new_memory = UnifiedMemoryUnit(
        user_id='u1',
        layer='M1',
        raw_content='user recently drinks coffee during project pressure',
    )

    assessment = detector.assess(new_memory, [old_memory])

    assert assessment.relation_type == 'temporal_update'
    assert assessment.should_version
    assert assessment.matched_memory_id == old_memory.memory_id


@pytest.mark.asyncio
async def test_llm_evaluator_can_refine_rule_assessment():
    old_memory = UnifiedMemoryUnit(
        user_id='u1', layer='M2', raw_content='user likes concise answers'
    )
    new_memory = UnifiedMemoryUnit(
        user_id='u1',
        layer='M1',
        raw_content='user likes concise answers with examples',
    )
    evaluator = LLMMemoryEvaluator(
        FakeLLMClient(
            {
                'relation_type': 'elaboration',
                'confidence': 0.91,
                'contradiction_risk': 0.02,
                'rationale': 'Adds detail without conflict.',
            }
        )
    )
    fallback = ConflictDetector().assess(new_memory, [old_memory])

    assessment = await evaluator.evaluate_relation(new_memory, old_memory, fallback)

    assert assessment.relation_type == 'elaboration'
    assert assessment.should_merge
    assert assessment.contradiction_risk == 0.02


@pytest.mark.asyncio
async def test_consolidation_versions_long_term_temporal_update():
    fake_graphiti = FakeGraphiti()
    policy = MemoryPolicy(consolidation_threshold=0.5)
    memory = HReConGraphMemory(fake_graphiti, policy=policy)
    old_long_term = UnifiedMemoryUnit(
        memory_id='ltm_old',
        user_id='u1',
        session_id='s1',
        layer='M2',
        raw_content='user does not drink coffee',
        lifecycle_status='consolidated',
        version=1,
        evidence_ids=['old_ev'],
    )
    memory.long_term_memories[old_long_term.memory_id] = old_long_term

    short_term = await memory.add_interaction(
        'user recently drinks coffee during project pressure',
        user_id='u1',
        session_id='s1',
    )
    short_term.access_count = 4

    consolidated = await memory.consolidate(user_id='u1', session_id='s1')

    assert len(consolidated) == 1
    assert consolidated[0].version == 2
    assert consolidated[0].supersedes == ['ltm_old']
    assert old_long_term.lifecycle_status == 'revised'
    assert old_long_term.metadata['revised_by'] == consolidated[0].memory_id
