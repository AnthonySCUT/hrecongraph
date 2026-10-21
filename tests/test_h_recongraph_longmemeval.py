import json

import pytest

from h_recongraph.examples.evaluate_longmemeval_read import (
    build_context,
    evaluate_cases,
    fact_turn_indexes,
    hit_session_indexes,
    load_cases,
)
from h_recongraph.examples.evaluate_read_context import reading_configurations


def _case():
    return {
        'question_id': 'case-1',
        'question_type': 'single-hop',
        'question': 'Where is the telescope stored?',
        'answer': 'observatory',
        'haystack_sessions': [
            [{'role': 'user', 'content': 'I bought a blue bicycle.', 'has_answer': False}],
            [
                {
                    'role': 'user',
                    'content': 'The telescope is stored at the observatory.',
                    'has_answer': True,
                },
                {
                    'role': 'assistant',
                    'content': 'I can also help compare telescope mounts and camera lenses.',
                    'has_answer': False,
                },
            ],
            [{'role': 'user', 'content': 'I prefer tea in the morning.', 'has_answer': False}],
        ],
    }


def test_query_selection_uses_content_overlap_and_keeps_order():
    indexes = hit_session_indexes(_case()['haystack_sessions'], _case()['question'])
    assert indexes == [1]


@pytest.mark.asyncio
async def test_evaluation_reports_turn_level_evidence_bundle():
    result = await evaluate_cases([_case()], repeats=1, window_sizes=(0, 1, 2))
    full = result['summary']['full_source']
    selected = result['summary']['evidence_bundle']
    assert selected['mean_context_tokens'] < full['mean_context_tokens']
    assert selected['supporting_turn_recall'] == 1.0
    assert selected['answer_visible_rate'] == 1.0
    assert selected['token_reduction_vs_full_source'] > 0
    assert len(result['rows']) == 4
    assert selected['overall_recall'] == {
        'micro': 1.0,
        'macro': 1.0,
        'retained': 1,
        'total': 1,
        'valid_cases': 1,
    }
    assert result['rows'][1]['supporting_turn_indices'] == [[1, 0]]
    assert result['rows'][1]['assembly_indexes'][0]['fact'] == [0]
    case = _case()
    del case['answer']
    for session in case['haystack_sessions']:
        for turn in session:
            del turn['has_answer']
    unlabeled_context = await build_context(case)
    assert len(unlabeled_context.render_text('evidence_bundle')) == result['rows'][1]['chars']


def test_load_cases_supports_offset_and_limit(tmp_path):
    path = tmp_path / 'cases.json'
    path.write_text(json.dumps([{'id': i} for i in range(4)]), encoding='utf-8')
    assert load_cases(path, offset=1, limit=2) == [{'id': 1}, {'id': 2}]


def test_conditional_second_anchor_requires_ratio_and_two_distinct_matches():
    query = 'Atlas launch date location budget weather organizer'
    session = [
        {'role': 'user', 'content': 'Atlas launch date location.'},
        {'role': 'user', 'content': 'Budget and weather.'},
        {'role': 'user', 'content': 'Tea and coffee.'},
    ]
    assert fact_turn_indexes(session, query) == [0]
    assert fact_turn_indexes(session, query, max_anchors=2) == [0, 1]
    session[0]['content'] += ' Budget weather.'
    assert fact_turn_indexes(session, query, max_anchors=2) == [0]
    session[0]['content'] = 'Atlas launch.'
    session[1]['content'] = 'Budget budget budget.'
    assert fact_turn_indexes(session, query, max_anchors=2) == [0]
    session[1]['content'] = 'Atlas.'
    assert fact_turn_indexes(session, 'user Atlas launch', max_anchors=2) == [0]


def test_second_anchor_ties_use_earliest_turn_and_render_in_source_order():
    session = [
        {'role': 'user', 'content': 'Atlas location.'},
        {'role': 'user', 'content': 'Atlas launch location.'},
        {'role': 'user', 'content': 'Atlas location.'},
    ]
    assert fact_turn_indexes(session, 'Atlas launch location', max_anchors=2) == [0, 1]


@pytest.mark.asyncio
async def test_conditional_anchors_keep_session_bindings_and_ignore_answer_labels():
    case = _case()
    case['question'] = 'Where is the telescope stored at the observatory?'
    case['haystack_sessions'][1][1]['content'] = 'Telescope observatory mounts are available.'
    context = await build_context(case, max_fact_anchors=2)
    assert list(context.source_fact_indices.values()) == [[0, 1]]
    assert len(context.graph_edges) == 2
    result = await evaluate_cases([case], repeats=1, window_sizes=(0, 1, 2), max_fact_anchors=2)
    binding = result['fixed_retrieval'][0]['source_bindings'][0]
    assert binding['session_index'] == 1
    assert len(binding['graph_facts']) == 2
    assert result['rows'][1]['assembly_indexes'][0]['fact'] == [0, 1]
    del case['answer']
    for session in case['haystack_sessions']:
        for turn in session:
            del turn['has_answer']
    unlabeled = await build_context(case, max_fact_anchors=2)
    assert [edge.fact for edge in unlabeled.graph_edges] == [
        edge.fact for edge in context.graph_edges
    ]
    assert unlabeled.source_fact_indices == context.source_fact_indices


@pytest.mark.asyncio
async def test_dependency_ablation_reports_shared_retrieval_and_support_diagnostics():
    configs = reading_configurations((0, 1, 2), ablation=True)
    result = await evaluate_cases([_case()], repeats=1, reading_configs=configs)
    assert len(result['fixed_retrieval']) == 1
    assert len(result['rows']) == len(configs) == 11
    assert (
        result['summary']['full_source_dedup']['total_tokens']
        <= result['summary']['full_source']['total_tokens']
    )
    assert result['summary']['fact_window_w1']['overall_recall']['retained'] == 1
    assert result['dependency_diagnostic']['retrieved_source_turns'] == 2
    assert result['dependency_diagnostic']['outside_fact_window_w1'] == 0
    for row in result['rows']:
        assert row['supporting_turn_indices'] == [[1, 0]]
