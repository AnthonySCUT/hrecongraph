from datetime import timedelta
from unittest.mock import AsyncMock

import pytest

from h_recongraph import HReConGraphMemory, MemoryPolicy
from h_recongraph.memory.conflict import ConflictDetector
from h_recongraph.memory.llm_evaluator import LLMMemoryEvaluator
from h_recongraph.memory.semantic_state import SemanticStateConstructor
from h_recongraph.schemas.memory_unit import SourceSpan, UnifiedMemoryUnit


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


def bundle(text='Project Atlas uses Redis.', memory_id='stm-atlas'):
    paragraphs = [
        text,
        'Only during October.',
        'The cache serves the dashboard.',
        'The reason is compatibility.',
    ]
    return UnifiedMemoryUnit(
        memory_id=memory_id,
        user_id='u1',
        layer='M1',
        lifecycle_status='active_short_term',
        raw_content='\n\n'.join(paragraphs),
        source_spans=[
            SourceSpan(evidence_id='ev-atlas', field=field, text=paragraph)
            for field, paragraph in zip(
                ('fact', 'constraint', 'path', 'causal'), paragraphs, strict=True
            )
        ],
    )


@pytest.mark.asyncio
async def test_write_bundle_keeps_four_dimensions_through_m0_m1_m2():
    controller = HReConGraphMemory(policy=MemoryPolicy(consolidation_threshold=2.0))
    supplied = bundle()
    short = await controller.add_interaction(
        'remember ' + supplied.raw_content, 'u1', source_spans=supplied.source_spans
    )
    long = await controller._consolidate_memory(short)

    for memory in [*controller.buffer.list(), short, long]:
        assert all(span in memory.source_spans for span in supplied.source_spans)
    for memory in (short, long):
        assert all(
            span.model_dump(mode='json') in memory.semantic_state['source_spans']
            for span in supplied.source_spans
        )


@pytest.mark.asyncio
async def test_modify_replaces_entire_bundle_in_both_layers_and_preserves_other_memory():
    controller = HReConGraphMemory()
    old = await controller.add_memory(bundle())
    old.metadata['promoted_from'] = 'im-atlas'
    long = await controller._consolidate_memory(old)
    unrelated = await controller.add_memory(bundle('Project Beta uses SQLite.', 'stm-beta'))
    kept = unrelated.model_dump()
    store = AsyncMock()
    controller.store = store
    replacement = bundle('Project Atlas uses PostgreSQL.', 'stm-replacement')
    replacement.source_spans[1].text = 'Only during November.'
    replacement.raw_content = replacement.raw_content.replace('October', 'November')

    result = await controller.update_memory(
        old.memory_id, operation='modify', replacement=replacement
    )

    for memory in (result, controller.long_term_memories[long.memory_id]):
        assert memory.version == 2
        assert 'Redis' not in memory.raw_content
        assert 'October' not in str(memory.semantic_state)
        assert memory.source_spans == replacement.source_spans
    assert unrelated.model_dump() == kept
    store.remove_memories.assert_awaited_once_with([old, long])
    assert store.add_memory.await_count == 2


@pytest.mark.asyncio
async def test_supersede_records_predecessor_and_reads_only_new_bundle():
    controller = HReConGraphMemory()
    old = await controller.add_memory(bundle())
    new = await controller.update_memory(
        old.memory_id,
        operation='supersede',
        replacement=bundle('Project Atlas uses PostgreSQL.', 'stm-new'),
    )
    context = await controller.retrieve('Project Atlas', 'u1')

    assert new.supersedes == [old.memory_id]
    assert new.version == 2
    assert old.lifecycle_status == 'revised'
    assert [state.edge_or_subgraph_id for state in context.semantic_states] == [new.memory_id]
    assert old.semantic_state_id not in controller.semantic_states


@pytest.mark.asyncio
async def test_delete_clears_bundle_lineage_and_keeps_unrelated_memory():
    controller = HReConGraphMemory(policy=MemoryPolicy(consolidation_threshold=2.0))
    old = await controller.add_interaction(
        'remember ' + bundle().raw_content, 'u1', source_spans=bundle().source_spans
    )
    long = await controller._consolidate_memory(old)
    unrelated = await controller.add_memory(bundle('Project Beta uses SQLite.', 'stm-beta'))
    kept = unrelated.model_dump()

    await controller.update_memory(long.memory_id, operation='delete')
    context = await controller.retrieve('Redis', 'u1')

    assert old.lifecycle_status == long.lifecycle_status == 'deleted'
    assert controller.buffer.list() == []
    assert old.semantic_state_id not in controller.semantic_states
    assert long.semantic_state_id not in controller.semantic_states
    assert context.instant_memories == context.semantic_states == []
    assert unrelated.model_dump() == kept
    assert await controller.consolidate('u1') == []


def test_span_delta_retains_other_dimensions_and_deduplicates_append():
    original = bundle()
    snapshot = original.model_dump()
    additional = SourceSpan(
        evidence_id='turn-2', text='Only on weekdays.', start=7, end=24, field='constraint'
    )

    revised = original.with_span_updates(append_spans=[additional, additional])

    assert original.model_dump() == snapshot
    assert revised.source_spans == [*original.source_spans, additional]
    assert revised.source_spans[-1].start == 7
    assert 'Only during October.' in revised.raw_content
    assert 'Only on weekdays.' in revised.raw_content
    assert revised.evidence_ids == ['ev-atlas', 'turn-2']


@pytest.mark.asyncio
async def test_append_constraints_updates_migrated_bundle_and_keeps_other_memory():
    controller = HReConGraphMemory()
    old = await controller.add_memory(bundle())
    long = await controller._consolidate_memory(old)
    unrelated = await controller.add_memory(bundle('Project Beta uses SQLite.', 'stm-beta'))
    kept = unrelated.model_dump()
    for memory in (old, long):
        memory.metadata['graphiti_episode_ids'] = ['old-episode']
        memory.metadata['graphiti_episode_id'] = 'old-episode'
    controller.store = AsyncMock()
    added = SourceSpan(evidence_id='turn-2', field='constraint', text='Only on weekdays.')

    await controller.update_memory(old.memory_id, operation='modify', append_spans=[added])

    for memory_id in (old.memory_id, long.memory_id):
        memory = controller._get_memory(memory_id)
        state = controller.semantic_states[memory.semantic_state_id]
        assert memory.version == 2
        assert 'Only during October.' in state.condition
        assert 'Only on weekdays.' in state.condition
        assert state.validity == state.condition
        assert state.cause == 'The reason is compatibility.'
        assert added in memory.source_spans
        assert 'graphiti_episode_ids' not in memory.metadata
        assert 'graphiti_episode_id' not in memory.metadata
    assert unrelated.model_dump() == kept


@pytest.mark.asyncio
@pytest.mark.parametrize('dimension', ['fact', 'constraint', 'path', 'causal'])
async def test_replace_one_dimension_preserves_other_source_spans_and_refreshes_state(dimension):
    controller = HReConGraphMemory()
    original = await controller.add_memory(bundle())
    previous = next(span for span in original.source_spans if span.field == dimension)
    changed = SourceSpan(
        evidence_id='turn-2', field=dimension, text=f'Updated {dimension} evidence.'
    )

    revised = await controller.update_memory(
        original.memory_id, operation='modify', span_updates={dimension: [changed]}
    )

    assert changed in revised.source_spans
    assert previous.text not in revised.raw_content
    assert previous.text not in str(revised.semantic_state)
    assert all(
        span in revised.source_spans for span in original.source_spans if span.field != dimension
    )
    assert set(revised.evidence_ids) == {'ev-atlas', 'turn-2'}
    if dimension == 'causal':
        assert revised.semantic_state['cause'] == changed.text
        cleared = await controller.update_memory(
            revised.memory_id, operation='modify', span_updates={'causal': []}
        )
        assert cleared.semantic_state['cause'] is None
        assert changed.text not in str(cleared.semantic_state)


def test_semantic_state_binds_natural_paragraphs_to_missing_dimensions():
    memory = UnifiedMemoryUnit(
        user_id='u1',
        layer='M1',
        raw_content=(
            'Atlas stores reports.\n\n'
            'The cache connects to the dashboard through Redis.\n\n'
            'Only during the October trial.\n\n'
            'Because the legacy client requires it.'
        ),
        evidence_ids=['ev-natural'],
    )

    state = SemanticStateConstructor().construct(memory)

    fields = {span.field for span in state.source_spans}
    assert {'fact', 'path', 'constraint', 'causal'} <= fields
    assert all(
        memory.raw_content[span.start : span.end] == span.text
        for span in state.source_spans
        if span.field in {'fact', 'path', 'constraint', 'causal'}
        if span.start is not None and span.end is not None
    )


@pytest.mark.asyncio
async def test_add_memory_persists_automatic_fact_and_path_bindings():
    controller = HReConGraphMemory()
    memory = UnifiedMemoryUnit(
        user_id='u1',
        layer='M1',
        raw_content='Atlas stores reports.\n\nReports travel through Redis to the dashboard.',
        evidence_ids=['ev-natural'],
    )

    written = await controller.add_memory(memory)

    assert any(span.field == 'fact' for span in written.source_spans)
    assert any(span.field == 'path' for span in written.source_spans)
    assert written.semantic_state['source_spans']


def test_word_boundary_binding_separates_markers_from_embedded_substrings():
    memory = UnifiedMemoryUnit(user_id='u1', layer='M1', raw_content='A specific viable design.')
    original = SemanticStateConstructor().construct(memory)
    boundary = SemanticStateConstructor('word_boundary').construct(memory)
    assert {'constraint', 'path'} <= {span.field for span in original.source_spans}
    assert not {'constraint', 'path'} & {span.field for span in boundary.source_spans}
    assert boundary.condition is None
    memory.raw_content = 'If Atlas runs via Redis, 因为兼容需求。'
    boundary = SemanticStateConstructor('word_boundary').construct(memory)
    assert {'constraint', 'path', 'causal'} <= {span.field for span in boundary.source_spans}


def test_fact_scope_binds_automatic_dependencies_by_original_paragraph_position():
    memory = UnifiedMemoryUnit(
        user_id='u1',
        layer='M1',
        raw_content=(
            'Atlas stores reports.\n\nAtlas works only during October.\n\n'
            'Weather alerts apply only during storms.'
        ),
    )
    state = SemanticStateConstructor('fact_scope').construct(memory)
    constraints = [span for span in state.source_spans if span.field == 'constraint']
    assert 0 in constraints[0].fact_starts
    assert 0 not in constraints[1].fact_starts
    assert all(memory.raw_content[span.start : span.end] == span.text for span in constraints)


def test_sentence_facts_keep_original_offsets_and_repeated_positions():
    text = 'user: Redis costs 3.5 dollars. Atlas runs.\nAtlas runs.\n\n缓存就绪。开始读取！'
    memory = UnifiedMemoryUnit(user_id='u1', layer='M1', raw_content=text)
    state = SemanticStateConstructor(fact_granularity='sentence').construct(memory)
    facts = [span for span in state.source_spans if span.field == 'fact']
    assert [span.text for span in facts] == [
        'user: Redis costs 3.5 dollars.',
        'Atlas runs.',
        'Atlas runs.',
        '缓存就绪。',
        '开始读取！',
    ]
    assert all(text[span.start : span.end] == span.text for span in facts)
    assert facts[1].start != facts[2].start


def test_word_forms_keep_temporal_variants_and_exclude_embedded_markers():
    constructor = SemanticStateConstructor('word_form_scope')
    assert constructor._marker_index('I recently visited.', 'recent') == 2
    assert constructor._marker_index('Atlas currently runs.', 'current') == 6
    assert constructor._marker_index('The service connected via Redis.', 'connects') == 12
    assert constructor._marker_index('Do you know a specific viable plan?', 'now') == -1
    assert constructor._marker_index('Do you know a specific viable plan?', 'if') == -1
    assert constructor._marker_index('Do you know a specific viable plan?', 'via') == -1
