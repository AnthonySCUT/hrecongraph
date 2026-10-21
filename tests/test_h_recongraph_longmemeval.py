import json
from argparse import Namespace
from types import SimpleNamespace

import pytest

from h_recongraph.examples.evaluate_longmemeval_read import (
    build_context,
    evaluate_anchor_calibration,
    evaluate_binding_calibration,
    evaluate_cases,
    fact_turn_indexes,
    hit_session_indexes,
    load_cases,
    prepare_qa_contexts,
    qa_judge_payload,
    run_qa,
    summarize_qa,
    update_annotation,
)
from h_recongraph.examples.evaluate_read_context import reading_configurations


@pytest.mark.asyncio
async def test_frozen_qa_contexts_share_sessions_dates_and_ignore_answer_labels():
    case = _case()
    case['haystack_dates'] = ['2026/01/01', '2026/01/02', '2026/01/03']
    case['haystack_sessions'][1][1]['content'] = 'Telescope observatory mounts. Tea is ready.'
    case['question'] = 'Where is the telescope at the observatory?'
    contexts = await prepare_qa_contexts(case)
    original = await build_context(
        case,
        binding_rule='word_form_scope',
        slot_mode='all',
        sentence_locator=True,
        max_fact_anchors=1,
    )
    assert contexts['top1_w1']['text'].endswith(
        original.render_text('evidence_bundle', window_size=1)
    )
    assert all(value['hit_session_indices'] == [1] for value in contexts.values())
    assert all('date=2026/01/02' in value['text'] for value in contexts.values())
    del case['answer']
    for session in case['haystack_sessions']:
        for turn in session:
            turn.pop('has_answer', None)
    assert await prepare_qa_contexts(case) == contexts


def test_update_scoring_uses_cited_replacement_and_blinded_answer_ids():
    case = _case()
    case['haystack_sessions'][1] = [
        {'role': 'user', 'content': 'The telescope is in the garage.'},
        {'role': 'user', 'content': 'I moved the telescope to the observatory.'},
    ]
    raw = {
        'eligible': True,
        'old_value': 'garage',
        'current_value': 'observatory',
        'old_evidence': {'session_index': 1, 'turn_index': 0, 'quote': 'in the garage'},
        'current_evidence': {'session_index': 1, 'turn_index': 1, 'quote': 'to the observatory'},
    }
    annotation = update_annotation(case, raw)
    assert annotation['eligible'] and annotation['evidence_verified']
    answers = {name: {'text': 'observatory'} for name in ('full_source', 'top1_w1', 'top2_w1')}
    payload, mapping = qa_judge_payload(case, answers, annotation)
    assert set(mapping.values()) == set(answers)
    assert all(name not in json.dumps(payload) for name in answers)
    assert payload['verified_update']['old_value'] == 'garage'


def test_qa_summary_separates_accuracy_stale_errors_and_actual_phase_usage():
    rows, calls = [], []
    for name, correct, stale in [
        ('full_source', True, False),
        ('top1_w1', False, False),
        ('top2_w1', False, True),
    ]:
        rows.append(
            {
                'question_id': 'q',
                'configuration': name,
                'correct': correct,
                'stale_eligible': True,
                'stale_misuse': stale,
            }
        )
        calls.append(
            {
                'question_id': 'q',
                'phase': 'answer',
                'configuration': name,
                'seconds': 1,
                'usage': {'prompt_tokens': 100, 'completion_tokens': 10, 'total_tokens': 110},
            }
        )
    calls.append(
        {
            'question_id': 'q',
            'phase': 'judge',
            'configuration': 'shared',
            'seconds': 2,
            'usage': {'prompt_tokens': 20, 'completion_tokens': 5, 'total_tokens': 25},
        }
    )
    summary = summarize_qa(rows, calls, 1)
    assert summary['configurations']['top1_w1']['stale_misuse_rate'] == 0
    assert summary['configurations']['top2_w1']['stale_misuse_rate'] == 1
    assert summary['configurations']['full_source']['mean_total_tokens'] == 110
    assert summary['phase_costs']['judge']['total_tokens'] == 25
    assert summary['paired_vs_full_source']['top1_w1']['lost_correct'] == 1


@pytest.mark.asyncio
async def test_qa_checkpoints_answers_and_scores_for_resumption(tmp_path, monkeypatch):
    from h_recongraph.examples import evaluate_longmemeval_read as module

    case = _case()
    case.update(
        question_type='knowledge-update',
        question_date='2026/01/04',
        haystack_dates=['2026/01/01', '2026/01/02', '2026/01/03'],
    )
    data = tmp_path / 'data.json'
    data.write_text(json.dumps([case]), encoding='utf-8')
    args = Namespace(
        qa_dir=tmp_path / 'run',
        data=data,
        model='glm-5.1',
        base_url='http://localhost/v1',
        api_key_env='TEST_QA_KEY',
        concurrency=3,
        qa_prepare_only=False,
    )
    monkeypatch.setenv('TEST_QA_KEY', 'test-key')
    requests = []

    class FakeClient:
        def __init__(self, **kwargs):
            self.http = kwargs['http_client']
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

        async def create(self, **kwargs):
            requests.append(kwargs)
            system = kwargs['messages'][0]['content']
            payload = json.loads(kwargs['messages'][1]['content'])
            if system == module.UPDATE_SYSTEM:
                text = json.dumps({'eligible': False, 'reason': 'addition only'})
            elif system == module.QA_JUDGE_SYSTEM:
                text = json.dumps(
                    {
                        'scores': [
                            {
                                'id': item['id'],
                                'correct': True,
                                'stale_misuse': False,
                                'reason': 'matches',
                            }
                            for item in payload['answers']
                        ]
                    }
                )
            else:
                assert 'answer' not in payload and 'has_answer' not in json.dumps(payload)
                text = 'observatory'
            return SimpleNamespace(
                id='call',
                model='glm-5.1',
                choices=[
                    SimpleNamespace(message=SimpleNamespace(content=text), finish_reason='stop')
                ],
                usage=SimpleNamespace(
                    model_dump=lambda: {
                        'prompt_tokens': 20,
                        'completion_tokens': 5,
                        'total_tokens': 25,
                    }
                ),
            )

        async def close(self):
            await self.http.aclose()

    monkeypatch.setattr('openai.AsyncOpenAI', FakeClient)
    first = await run_qa([case], args)
    args.base_url = 'http://localhost-authorized-resume/v1'
    second = await run_qa([case], args)
    assert len(requests) == 5
    assert all(item['temperature'] == 0 for item in requests)
    assert first == second
    assert all(item['correct'] is True for item in first['rows'])
    assert first['summary']['phase_costs']['answer']['calls'] == 3
    segments = [
        json.loads(line)
        for line in (args.qa_dir / 'execution_segments.jsonl').read_text().splitlines()
    ]
    assert segments[-1]['base_url'] == args.base_url
    assert segments[-1]['reused_successful_calls'] == 5


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


@pytest.mark.asyncio
async def test_binding_calibration_freezes_retrieval_and_ignores_answer_labels():
    case = _case()
    result = await evaluate_binding_calibration([case], repeats=1)
    assert result['fixed_retrieval_equal']
    assert set(result['binding_calibration']) == {'substring', 'word_boundary', 'fact_scope'}
    for rule, value in result['binding_calibration'].items():
        assert value['summary']['evidence_bundle']['overall_recall']['retained'] == 1
        labeled = await build_context(case, binding_rule=rule)
        unlabeled_case = {key: value for key, value in case.items() if key != 'answer'}
        unlabeled_case['haystack_sessions'] = [
            [{key: value for key, value in turn.items() if key != 'has_answer'} for turn in session]
            for session in case['haystack_sessions']
        ]
        unlabeled = await build_context(unlabeled_case, binding_rule=rule)
        assert unlabeled.render_text('evidence_bundle') == labeled.render_text('evidence_bundle')


@pytest.mark.asyncio
async def test_binding_calibration_records_distant_support_removed_by_lexical_scope():
    case = _case()
    case['haystack_sessions'][1] = [
        case['haystack_sessions'][1][0],
        {'role': 'assistant', 'content': 'Tea is ready.'},
        {'role': 'user', 'content': 'The weather is pleasant.'},
        {'role': 'user', 'content': 'Only during winter.', 'has_answer': True},
    ]
    result = await evaluate_binding_calibration([case], repeats=1)
    original = result['comparison_w1']['substring']
    scoped = result['comparison_w1']['fact_scope']
    assert original['reference_far_supports'] == original['far_supports_retained'] == 1
    assert scoped['removed_annotated_support_turns'] == scoped['removed_supports_lost'] == 1
    assert scoped['far_supports_retained'] == 0


@pytest.mark.asyncio
async def test_sentence_anchor_keeps_turn_provenance_and_restricts_cross_topic_dependencies():
    case = _case()
    case['haystack_sessions'][1] = [
        {
            'role': 'user',
            'content': 'The telescope is stored at the observatory.\nWe discussed weather forecasts.',
            'has_answer': True,
        },
        {'role': 'assistant', 'content': 'Tea is ready.'},
        {'role': 'user', 'content': 'The concert was fun.'},
        {'role': 'user', 'content': 'Weather alerts apply only during storms.'},
    ]
    turn = await build_context(case, binding_rule='fact_scope')
    sentence = await build_context(case, binding_rule='fact_scope', fact_granularity='sentence')
    assert sentence.source_fact_indices == turn.source_fact_indices
    episode = sentence.source_episodes[0]
    anchor = sentence.source_fact_spans[episode.uuid][0]
    raw = sentence._source_text(episode.content)
    assert raw[anchor.start : anchor.end] == sentence.graph_edges[0].fact == anchor.text
    assert 'weather' not in anchor.text
    assert 3 in turn.source_indices(turn.source_episodes[0])['dependencies']
    assert 3 not in sentence.source_indices(episode)['dependencies']
    assert sentence.source_segments(episode)[0].count('\n') == 1
    unlabeled = {key: value for key, value in case.items() if key != 'answer'}
    unlabeled['haystack_sessions'] = [
        [
            {key: value for key, value in message.items() if key != 'has_answer'}
            for message in session
        ]
        for session in case['haystack_sessions']
    ]
    rebuilt = await build_context(unlabeled, binding_rule='fact_scope', fact_granularity='sentence')
    assert rebuilt.render_text('evidence_bundle') == sentence.render_text('evidence_bundle')


@pytest.mark.asyncio
async def test_anchor_calibration_separates_granularity_and_word_form_effects():
    case = _case()
    case['haystack_sessions'][1].extend(
        [
            {'role': 'assistant', 'content': 'Tea is ready.'},
            {
                'role': 'user',
                'content': 'I recently inspected the observatory.',
                'has_answer': True,
            },
        ]
    )
    result = await evaluate_anchor_calibration([case], repeats=1)
    assert result['fixed_retrieval_equal']
    assert len(result['anchor_calibration']) == 6
    assert len(result['summary']) == 24
    for granularity in ('paragraph', 'sentence'):
        scoped = result['anchor_calibration'][f'fact_scope/{granularity}']
        forms = result['anchor_calibration'][f'word_form_scope/{granularity}']
        assert scoped['summary']['evidence_bundle_w0']['overall_recall']['retained'] == 1
        assert forms['summary']['evidence_bundle_w0']['overall_recall']['retained'] == 2


@pytest.mark.asyncio
async def test_on_demand_sentences_preserve_paragraph_dependencies_and_original_turn_offsets():
    case = _case()
    case['haystack_sessions'][1] = [
        {'role': 'user', 'content': 'Tea is ready.'},
        {
            'role': 'user',
            'content': 'The telescope is stored at the observatory.\nWe discussed weather forecasts.',
            'has_answer': True,
        },
        {'role': 'assistant', 'content': 'Tea is ready.'},
        {'role': 'user', 'content': 'Weather alerts apply only during storms.'},
    ]
    original = await build_context(case, binding_rule='word_form_scope')
    located = await build_context(case, binding_rule='word_form_scope', sentence_locator=True)
    assert located.source_fact_indices == original.source_fact_indices
    assert not located.source_fact_spans
    for left, right in zip(original.source_episodes, located.source_episodes, strict=True):
        assert original.source_indices(left, 1) == located.source_indices(right, 1)
        assert located.source_indices(right)['dependencies'] == [3]
        anchors = located.locate_fact_sentences()[right.uuid]
        text = located._source_text(right.content)
        assert len(anchors) == 1
        assert text[anchors[0].start : anchors[0].end] == located.graph_edges[0].fact
        assert 'weather' not in anchors[0].text
    assert [span.model_dump() for span in original.semantic_states[0].source_spans] == [
        span.model_dump() for span in located.semantic_states[0].source_spans
    ]


@pytest.mark.asyncio
async def test_on_demand_top2_uses_fixed_anchors_and_ignores_answer_labels():
    case = _case()
    case['question'] = 'Where is the telescope stored at the observatory?'
    case['haystack_sessions'][1][1]['content'] = 'Telescope observatory mounts. Tea is ready.'
    result = await evaluate_cases(
        [case],
        repeats=1,
        window_sizes=(0, 1, 2),
        binding_rule='word_form_scope',
        max_fact_anchors=2,
        sentence_locator=True,
    )
    binding = result['fixed_retrieval'][0]['source_bindings'][0]
    assert binding['graph_fact_turn_indices'] == [0, 1]
    assert [anchor['turn_index'] for anchor in binding['fact_anchors']] == [0, 1]
    context = await build_context(
        case, binding_rule='word_form_scope', max_fact_anchors=2, sentence_locator=True
    )
    del case['answer']
    for session in case['haystack_sessions']:
        for turn in session:
            turn.pop('has_answer', None)
    unlabeled = await build_context(
        case, binding_rule='word_form_scope', max_fact_anchors=2, sentence_locator=True
    )
    assert unlabeled.render_text('evidence_bundle', window_size=1) == context.render_text(
        'evidence_bundle', window_size=1
    )
