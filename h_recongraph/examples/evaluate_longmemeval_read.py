"""Explore LongMemEval evidence-recall/token Pareto configurations on fixed retrieval."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
from collections import Counter
from hashlib import sha256
from pathlib import Path
from statistics import mean, median
from time import perf_counter
from typing import Any

from graphiti_core.edges import EntityEdge
from graphiti_core.nodes import EpisodeType, EpisodicNode
from graphiti_core.utils.content_chunking import estimate_tokens
from graphiti_core.utils.datetime_utils import utc_now
from h_recongraph.examples.evaluate_read_context import (
    assembly_stages,
    experiment_environment,
    finalize_summaries,
    measure_render,
    pareto_analysis,
    reading_configurations,
    select_fact_span,
    stage_source_texts,
    summarize_rows,
)
from h_recongraph.memory.graphiti_store import GraphitiMemoryStore
from h_recongraph.memory.manager import HReConGraphMemory
from h_recongraph.memory.ranker import MemoryUtilityRanker
from h_recongraph.memory.relinking import HierarchicalMemoryContext
from h_recongraph.memory.semantic_state import SemanticStateConstructor
from h_recongraph.schemas.memory_unit import UnifiedMemoryUnit

QA_CONFIGS = {
    'full_source': {'source_mode': 'full_source', 'max_fact_anchors': 2, 'window_size': 0},
    'top1_w1': {'source_mode': 'evidence_bundle', 'max_fact_anchors': 1, 'window_size': 1},
    'top2_w1': {'source_mode': 'evidence_bundle', 'max_fact_anchors': 2, 'window_size': 1},
}
QA_SYSTEM = """Answer the question using only the supplied memory. Use session dates and the
question date to resolve temporal references. Respect corrections, negation, uncertainty and
the time requested by the question. Give the required details concisely, within 100 words.
If the memory is insufficient, say so explicitly. Output only the final answer."""
UPDATE_SYSTEM = """Identify whether this knowledge-update question tests a superseded value.
Use the reference answer and dated original dialogue. Eligible means an earlier value was
explicitly replaced by a different value effective at the time requested by this question.
Historical questions, additions without replacement, and unconfirmed alternatives are ineligible.
Return JSON: {"eligible":bool,"old_value":str,"current_value":str,"reason":str,
"old_evidence":{"session_index":int,"turn_index":int,"quote":str},
"current_evidence":{"session_index":int,"turn_index":int,"quote":str}}.
For eligible cases cite short verbatim quotes from the indexed source turns. Otherwise use
empty values and null evidence. This annotation is completed before candidate answers exist."""
QA_JUDGE_SYSTEM = """Evaluate anonymized answers against the question and reference answer.
Accept semantically equivalent wording. Check all required items, quantities, dates, ordering,
negation and uncertainty. For multi-session questions require the requested combined facts;
for temporal-reasoning require the requested time/order; for preference questions require the
user-specific preference; for knowledge-update respect the value effective at the requested time.
For unanswerable questions, an explicit appropriate abstention is correct.
When a verified update is supplied, stale_misuse is true only if an answer presents its old_value
as effective at the requested time. Mentioning it explicitly as historical is acceptable.
Other wrong answers are incorrect but are not automatically stale misuse.
Return JSON: {"scores":[{"id":str,"correct":bool,"stale_misuse":bool,"reason":str}]}.
Score every supplied id independently. Return only JSON."""


def qa_manifest(cases: list[dict], data: Path, model: str, base_url: str) -> dict:
    """Freeze source, prompts and generation settings before answering."""
    root = Path(__file__).parents[2]
    files = [
        'memory/relinking.py',
        'memory/semantic_state.py',
        'memory/ranker.py',
        'examples/evaluate_longmemeval_read.py',
    ]
    return {
        'experiment_role': 'exploratory paired QA on previously explored oracle questions',
        'dataset_sha256': sha256(data.read_bytes()).hexdigest(),
        'question_ids': [case['question_id'] for case in cases],
        'configurations': QA_CONFIGS,
        'binding_rule': 'word_form_scope',
        'fact_granularity': 'paragraph',
        'slot_mode': 'all',
        'sentence_locator': True,
        'related_source_limit': 0,
        'model': model,
        'base_url': base_url,
        'temperature': 0,
        'max_tokens': {'answer': 512, 'annotation': 1536, 'judge': 2048},
        'thinking': {'type': 'disabled'},
        'prompts': {'answer': QA_SYSTEM, 'annotation': UPDATE_SYSTEM, 'judge': QA_JUDGE_SYSTEM},
        'code_sha256': {
            name: sha256((root / 'h_recongraph' / name).read_bytes()).hexdigest() for name in files
        },
    }


async def prepare_qa_contexts(case: dict) -> dict:
    """Reuse one write state, select frozen Top-1/Top-2 edges and retain source dates."""
    context = await build_context(
        case,
        max_fact_anchors=2,
        binding_rule='word_form_scope',
        slot_mode='all',
        sentence_locator=True,
    )
    hit = hit_session_indexes(case['haystack_sessions'], case['question'])
    dates = '\n'.join(
        f'[SESSION {episode.uuid} date={case["haystack_dates"][index]}]'
        for index, episode in zip(hit, context.source_episodes, strict=True)
    )
    texts = {
        'full_source': context.render_text('full_source'),
        'top2_w1': context.render_text('evidence_bundle', window_size=1),
    }
    single = context.model_copy(deep=True)
    single.graph_edges = []
    for session_index, episode in zip(hit, single.source_episodes, strict=True):
        wanted = fact_turn_indexes(case['haystack_sessions'][session_index], case['question'])
        edges = [edge for edge in context.graph_edges if episode.uuid in edge.episodes]
        single.graph_edges.extend(
            edge
            for index, edge in zip(context.source_fact_indices[episode.uuid], edges, strict=True)
            if index in wanted
        )
        single.source_fact_indices[episode.uuid] = wanted
    texts['top1_w1'] = single.render_text('evidence_bundle', window_size=1)
    return {
        name: {
            'text': dates + '\n' + texts[name],
            'sha256': sha256((dates + '\n' + texts[name]).encode()).hexdigest(),
            'estimated_tokens': estimate_tokens(texts[name]),
            'hit_session_indices': hit,
        }
        for name in QA_CONFIGS
    }


def update_annotation(case: dict, raw: dict) -> dict:
    """Verify the old/new source citations used by the stale-answer metric."""
    if not raw['eligible']:
        return {**raw, 'evidence_verified': True}
    verified = raw['old_value'] != raw['current_value'] and all(
        item['quote']
        and item['quote']
        in case['haystack_sessions'][item['session_index']][item['turn_index']]['content']
        for item in (raw['old_evidence'], raw['current_evidence'])
    )
    return {**raw, 'eligible': True if verified else None, 'evidence_verified': verified}


def parse_qa_json(text: str) -> dict:
    text = text.strip()
    if text.startswith('```'):
        text = '\n'.join(text.splitlines()[1:-1])
    result = json.loads(text)
    for score in result.get('scores', []):
        if 'stale_muse' in score and 'stale_misuse' not in score:
            score['stale_misuse'] = score.pop('stale_muse')
    return result


def qa_judge_payload(case: dict, answers: dict, annotation: dict) -> tuple[dict, dict]:
    """Rotate opaque answer ids and keep configuration names outside the judge prompt."""
    names = list(QA_CONFIGS)
    shift = int(sha256(case['question_id'].encode()).hexdigest(), 16) % len(names)
    names = names[shift:] + names[:shift]
    mapping = {f'answer_{index}': name for index, name in enumerate(names)}
    return {
        'question': case['question'],
        'question_date': case.get('question_date', ''),
        'question_type': case['question_type'],
        'reference_answer': case['answer'],
        'verified_update': annotation if annotation.get('eligible') is True else None,
        'answers': [{'id': key, 'answer': answers[name]['text']} for key, name in mapping.items()],
    }, mapping


def wilson_interval(successes: int, total: int) -> list[float] | None:
    if not total:
        return None
    z, proportion = 1.96, successes / total
    center = proportion + z * z / (2 * total)
    radius = z * math.sqrt(proportion * (1 - proportion) / total + z * z / (4 * total * total))
    return [(center - radius) / (1 + z * z / total), (center + radius) / (1 + z * z / total)]


def summarize_qa(rows: list[dict], calls: list[dict], case_count: int) -> dict:
    """Report answer quality, actual provider usage and paired correctness differences."""
    summary = {}
    for name in QA_CONFIGS:
        selected = [row for row in rows if row['configuration'] == name]
        judged = [row for row in selected if row.get('correct') is not None]
        eligible = [row for row in judged if row.get('stale_eligible') is True]
        stale_scored = [row for row in eligible if row.get('stale_misuse') is not None]
        answers = [
            call
            for call in calls
            if call['phase'] == 'answer' and call['configuration'] == name and 'error' not in call
        ]
        usage = [call['usage'] for call in answers if call['usage'] is not None]
        correct = sum(row['correct'] for row in judged)
        summary[name] = {
            'expected_cases': case_count,
            'answered_cases': len(answers),
            'judged_cases': len(judged),
            'correct': correct,
            'accuracy': correct / len(judged) if judged else None,
            'accuracy_wilson95': wilson_interval(correct, len(judged)),
            'stale_eligible_cases': len(eligible),
            'stale_scored_cases': len(stale_scored),
            'stale_misuse_count': sum(row['stale_misuse'] for row in stale_scored),
            'stale_misuse_rate': mean(row['stale_misuse'] for row in stale_scored)
            if stale_scored
            else None,
            'usage_observed_calls': len(usage),
            'mean_input_tokens': mean(item['prompt_tokens'] for item in usage) if usage else None,
            'mean_output_tokens': mean(item['completion_tokens'] for item in usage)
            if usage
            else None,
            'mean_total_tokens': mean(item['total_tokens'] for item in usage) if usage else None,
            'median_answer_seconds': median(call['seconds'] for call in answers)
            if answers
            else None,
        }
    paired = {}
    base = {
        row['question_id']: row
        for row in rows
        if row['configuration'] == 'full_source' and row.get('correct') is not None
    }
    for name in ('top1_w1', 'top2_w1'):
        comparable = [
            row
            for row in rows
            if row['configuration'] == name
            and row.get('correct') is not None
            and row['question_id'] in base
        ]
        differences = [
            int(row['correct']) - int(base[row['question_id']]['correct']) for row in comparable
        ]
        delta = mean(differences) if differences else None
        se = (
            math.sqrt(
                sum((value - delta) ** 2 for value in differences)
                / (len(differences) * (len(differences) - 1))
            )
            if len(differences) > 1
            else 0
        )
        paired[name] = {
            'valid_pairs': len(differences),
            'accuracy_delta': delta,
            'gained_correct': sum(value == 1 for value in differences),
            'lost_correct': sum(value == -1 for value in differences),
            'paired_normal95': [max(-1, delta - 1.96 * se), min(1, delta + 1.96 * se)]
            if delta is not None
            else None,
        }
    costs = {}
    for phase in ('answer', 'annotation', 'judge'):
        selected = [call for call in calls if call['phase'] == phase]
        usage = [call['usage'] for call in selected if call.get('usage') is not None]
        costs[phase] = {
            'calls': len(selected),
            'failed_calls': sum('error' in call for call in selected),
            'usage_observed_calls': len(usage),
            **{
                key: sum(item[key] for item in usage)
                for key in ('prompt_tokens', 'completion_tokens', 'total_tokens')
            },
        }
    return {'configurations': summary, 'paired_vs_full_source': paired, 'phase_costs': costs}


async def run_qa(cases: list[dict], args: argparse.Namespace) -> dict:
    """Checkpoint frozen contexts, source annotations, answers and blinded batch judgements."""
    import httpx
    from openai import APIError, AsyncOpenAI

    folder = args.qa_dir
    folder.mkdir(parents=True, exist_ok=True)
    manifest_path = folder / 'manifest.json'
    manifest = qa_manifest(cases, args.data, args.model, args.base_url)
    execution = manifest
    if manifest_path.exists():
        frozen = json.loads(manifest_path.read_text(encoding='utf-8'))
        fixed_keys = frozen.keys() - {'base_url', 'code_sha256'}
        assert all(manifest[key] == frozen[key] for key in fixed_keys)
        assert all(
            manifest['code_sha256'][name] == digest
            for name, digest in frozen['code_sha256'].items()
            if name != 'examples/evaluate_longmemeval_read.py'
        )
        manifest = frozen
    else:
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8'
        )

    def read_lines(path):
        return (
            [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
            if path.exists()
            else []
        )

    def append(path, item):
        with path.open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(item, ensure_ascii=False) + '\n')

    contexts_path, calls_path = folder / 'contexts.jsonl', folder / 'calls.jsonl'
    contexts = {row['question_id']: row['contexts'] for row in read_lines(contexts_path)}
    for index, case in enumerate(cases):
        if case['question_id'] not in contexts:
            contexts[case['question_id']] = await prepare_qa_contexts(case)
            append(
                contexts_path,
                {'question_id': case['question_id'], 'contexts': contexts[case['question_id']]},
            )
        if (index + 1) % 25 == 0:
            print(f'Prepared {index + 1}/{len(cases)} questions', flush=True)
    if args.qa_prepare_only:
        return {'prepared_cases': len(contexts), 'status': 'prepared'}
    calls = read_lines(calls_path)
    cache = {
        (row['question_id'], row['phase'], row['configuration']): row
        for row in calls
        if 'error' not in row
    }
    segments_path = folder / 'execution_segments.jsonl'
    append(
        segments_path,
        {
            'started_at': utc_now().isoformat(),
            'base_url': args.base_url,
            'model': args.model,
            'concurrency': args.concurrency,
            'reused_successful_calls': len(cache),
            'execution_code_sha256': execution['code_sha256'],
        },
    )
    semaphore = asyncio.Semaphore(args.concurrency)
    client = AsyncOpenAI(
        api_key=os.environ[args.api_key_env],
        base_url=args.base_url,
        timeout=1200,
        max_retries=2,
        http_client=httpx.AsyncClient(
            trust_env=False,
            timeout=httpx.Timeout(1200, connect=15),
            limits=httpx.Limits(
                max_connections=args.concurrency, max_keepalive_connections=args.concurrency
            ),
        ),
    )

    async def call(case, phase, configuration, system, payload):
        key = (case['question_id'], phase, configuration)
        payload_digest = sha256(json.dumps(payload, ensure_ascii=False).encode()).hexdigest()
        if key in cache and cache[key].get('payload_sha256', payload_digest) != payload_digest:
            cache.pop(key)
        if key in cache:
            return cache[key]
        async with semaphore:
            started = perf_counter()
            row = {
                'question_id': case['question_id'],
                'phase': phase,
                'configuration': configuration,
                'base_url': args.base_url,
                'requested_model': args.model,
                'payload_sha256': payload_digest,
            }
            try:
                response = await client.chat.completions.create(
                    model=args.model,
                    temperature=0,
                    max_tokens=manifest['max_tokens'][phase],
                    messages=[
                        {'role': 'system', 'content': system},
                        {'role': 'user', 'content': json.dumps(payload, ensure_ascii=False)},
                    ],
                    extra_body={'thinking': {'type': 'disabled'}},
                    **({'response_format': {'type': 'json_object'}} if phase != 'answer' else {}),
                )
                text = response.choices[0].message.content or ''
                row.update(
                    text=text,
                    seconds=perf_counter() - started,
                    usage=response.usage.model_dump() if response.usage else None,
                    response_id=response.id,
                    response_model=response.model,
                    finish_reason=response.choices[0].finish_reason,
                )
                if not text.strip() or row['finish_reason'] == 'length':
                    row['error'] = 'empty_or_truncated_response'
            except APIError as exc:
                row.update(
                    error=type(exc).__name__,
                    status_code=getattr(exc, 'status_code', None),
                    seconds=perf_counter() - started,
                )
            append(calls_path, row)
            calls.append(row)
            if 'error' not in row:
                cache[key] = row
            return row

    async def annotate(case):
        if case['question_type'] != 'knowledge-update':
            return {'eligible': False, 'reason': 'other_question_type'}
        payload = {key: case[key] for key in ('question', 'question_date', 'answer')}
        payload['sessions'] = [
            {
                'session_index': index,
                'date': case['haystack_dates'][index],
                'turns': [
                    {'turn_index': turn, 'role': message['role'], 'content': message['content']}
                    for turn, message in enumerate(session)
                ],
            }
            for index, session in enumerate(case['haystack_sessions'])
        ]
        row = await call(case, 'annotation', 'shared', UPDATE_SYSTEM, payload)
        if 'error' in row:
            return {'eligible': None, 'error': row['error']}
        try:
            return update_annotation(case, parse_qa_json(row['text']))
        except (ValueError, KeyError, TypeError, IndexError):
            return {'eligible': None, 'error': 'invalid_update_annotation'}

    try:
        annotations_path = folder / 'update_annotations.json'
        previous_annotations = (
            json.loads(annotations_path.read_text(encoding='utf-8'))
            if annotations_path.exists()
            else {}
        )
        annotations = dict(
            zip(
                (case['question_id'] for case in cases),
                await asyncio.gather(*(annotate(case) for case in cases)),
                strict=True,
            )
        )
        reviews_path = folder / 'update_annotation_reviews.json'
        if reviews_path.exists():
            reviews = json.loads(reviews_path.read_text(encoding='utf-8'))
            for qid, review in reviews.items():
                annotations[qid].update(review, eligibility_reviewed=True)
        annotations_path.write_text(
            json.dumps(annotations, ensure_ascii=False, indent=2), encoding='utf-8'
        )
        rows = []

        async def evaluate(case):
            qid = case['question_id']
            answers = dict(
                zip(
                    QA_CONFIGS,
                    await asyncio.gather(
                        *(
                            call(
                                case,
                                'answer',
                                name,
                                QA_SYSTEM,
                                {
                                    'question': case['question'],
                                    'question_date': case.get('question_date', ''),
                                    'memory': contexts[qid][name]['text'],
                                },
                            )
                            for name in QA_CONFIGS
                        )
                    ),
                    strict=True,
                )
            )
            annotation = annotations[qid]
            scores = {}
            judge_error = None
            if all('error' not in answer for answer in answers.values()):
                payload, mapping = qa_judge_payload(case, answers, annotation)
                judge_key = (qid, 'judge', 'shared')
                if judge_key in cache:
                    previous = previous_annotations.get(qid, {})
                    old_update = previous if previous.get('eligible') is True else None
                    try:
                        cached_scores = parse_qa_json(cache[judge_key]['text'])['scores']
                        complete = {item['id'] for item in cached_scores} == set(mapping) and all(
                            type(item.get('correct')) is bool
                            and (
                                annotation['eligible'] is not True
                                or type(item.get('stale_misuse')) is bool
                            )
                            for item in cached_scores
                        )
                    except (ValueError, KeyError, TypeError):
                        complete = False
                    if not complete or old_update != payload['verified_update']:
                        cache.pop(judge_key)
                judged = await call(case, 'judge', 'shared', QA_JUDGE_SYSTEM, payload)
                if 'error' not in judged:
                    try:
                        scores = {
                            mapping[item['id']]: item
                            for item in parse_qa_json(judged['text'])['scores']
                        }
                    except (ValueError, KeyError, TypeError):
                        judge_error = 'invalid_judge_json'
            for name in QA_CONFIGS:
                score = scores.get(name, {})
                rows.append(
                    {
                        'question_id': qid,
                        'question_type': case['question_type'],
                        'configuration': name,
                        'answer': answers[name].get('text', ''),
                        'correct': score.get('correct'),
                        'reason': score.get('reason', ''),
                        'stale_eligible': annotation['eligible'],
                        'stale_misuse': score.get('stale_misuse')
                        if annotation['eligible'] is True
                        else None,
                        'context_sha256': contexts[qid][name]['sha256'],
                        'answer_error': answers[name].get('error'),
                        'judge_error': judge_error,
                    }
                )
            if len(rows) % 30 == 0:
                print(f'Answered/scored {len(rows) // 3}/{len(cases)} questions', flush=True)

        await asyncio.gather(*(evaluate(case) for case in cases))
    finally:
        await client.close()
    rows.sort(key=lambda row: (row['question_id'], row['configuration']))
    result = {
        'protocol': manifest,
        'summary': summarize_qa(rows, calls, len(cases)),
        'annotation_counts': dict(Counter(str(item['eligible']) for item in annotations.values())),
        'by_question_type': {
            kind: summarize_qa(
                [row for row in rows if row['question_type'] == kind],
                [
                    call
                    for call in calls
                    if call['question_id']
                    in {case['question_id'] for case in cases if case['question_type'] == kind}
                ],
                sum(case['question_type'] == kind for case in cases),
            )
            for kind in sorted({case['question_type'] for case in cases})
        },
        'rows': rows,
    }
    (folder / 'results.json').write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8'
    )
    return result


def load_cases(path: str | Path, *, offset: int = 0, limit: int | None = None) -> list[dict]:
    cases = json.loads(Path(path).read_text(encoding='utf-8'))
    return cases[offset : None if limit is None else offset + limit]


def turn_text(turn: dict) -> str:
    return f'{turn["role"]}: {turn["content"]}'


def session_text(session: list[dict]) -> str:
    return '\n\n'.join(turn_text(turn) for turn in session)


def hit_session_indexes(sessions: list[list[dict]], query: str) -> list[int]:
    ranker = MemoryUtilityRanker()
    return [
        index
        for index, session in enumerate(sessions)
        if ranker.lexical_relevance(session_text(session), query) > 0
    ]


def fact_turn_indexes(session: list[dict], query: str, *, max_anchors: int = 1) -> list[int]:
    """Keep Top-1; admit Top-2 with half its score and two distinct content-word matches."""
    ranker = MemoryUtilityRanker()
    query_tokens = ranker.tokenize(query)
    overlaps = [len(ranker.tokenize(turn_text(turn)) & query_tokens) for turn in session]
    ranked = sorted(range(len(session)), key=lambda index: -overlaps[index])
    selected = ranked[:1]
    if max_anchors == 2 and len(ranked) > 1:
        first, second = ranked[:2]
        content_overlap = len(ranker.tokenize(session[second]['content']) & query_tokens)
        if content_overlap >= 2 and overlaps[second] >= 0.5 * overlaps[first]:
            selected.append(second)
    return sorted(selected)


async def build_context(
    case: dict,
    *,
    max_fact_anchors: int = 1,
    binding_rule: str = 'substring',
    fact_granularity: str = 'paragraph',
    slot_mode: str = 'all',
    sentence_locator: bool = False,
) -> HierarchicalMemoryContext:
    """Use only question/source text to build one fixed lexical graph adapter."""
    now = utc_now()
    writer = HReConGraphMemory(
        semantic_state_constructor=SemanticStateConstructor(
            binding_rule, fact_granularity=fact_granularity, slot_mode=slot_mode
        )
    )
    episodes, edges, states = [], [], []
    units, fact_indices, fact_spans = {}, {}, {}
    for session_index in hit_session_indexes(case['haystack_sessions'], case['question']):
        session = case['haystack_sessions'][session_index]
        memory = UnifiedMemoryUnit(
            memory_id=f'lm-{case["question_id"]}-{session_index}',
            user_id='eval',
            layer='M1',
            lifecycle_status='active_short_term',
            raw_content=session_text(session),
        )
        await writer.add_memory(memory)
        episode = EpisodicNode(
            uuid=f'episode-{case["question_id"]}-{session_index}',
            name=case['question_id'],
            group_id='user:eval',
            source=EpisodeType.text,
            source_description='LongMemEval dialogue session',
            content=GraphitiMemoryStore(None)._episode_body(memory),
            valid_at=now,
        )
        selected = fact_turn_indexes(session, case['question'], max_anchors=max_fact_anchors)
        units[episode.uuid] = [turn_text(message) for message in session]
        fact_indices[episode.uuid] = selected
        for anchor_index, turn_index in enumerate(selected):
            fact = turn_text(session[turn_index])
            if fact_granularity == 'sentence':
                start = sum(len(turn_text(turn)) + 2 for turn in session[:turn_index])
                candidates = [
                    span
                    for span in memory.source_spans
                    if span.field == 'fact' and start <= span.start < start + len(fact)
                ]
                anchor = select_fact_span(candidates, case['question'])
                fact_spans.setdefault(episode.uuid, []).append(anchor)
                fact = anchor.text
            edges.append(
                EntityEdge(
                    uuid=f'{episode.uuid}-edge' + (f'-{anchor_index}' if anchor_index else ''),
                    created_at=now,
                    group_id='user:eval',
                    source_node_uuid=f'{episode.uuid}-source',
                    target_node_uuid=f'{episode.uuid}-target',
                    name='RELEVANT_TURN',
                    fact=fact,
                    episodes=[episode.uuid],
                )
            )
        episodes.append(episode)
        states.append(writer.semantic_states[memory.semantic_state_id])
    context = HierarchicalMemoryContext(
        query=case['question'],
        selected_layers=['M1'],
        graph_edges=edges,
        semantic_states=states,
        source_episodes=episodes,
        source_units=units,
        source_fact_indices=fact_indices,
        source_fact_spans=fact_spans,
    )
    if sentence_locator:
        located = context.locate_fact_sentences()
        for episode in episodes:
            source_edges = [edge for edge in edges if episode.uuid in edge.episodes]
            for edge, anchor in zip(source_edges, located[episode.uuid], strict=True):
                edge.fact = anchor.text
    return context


def support_scores(case: dict, hit_indexes: list[int], stages: list[dict], mode: str) -> dict:
    """Read support annotations only after source assembly, using original turn identities."""
    supports = {
        (session_index, turn_index)
        for session_index, session in enumerate(case['haystack_sessions'])
        for turn_index, turn in enumerate(session)
        if turn.get('has_answer')
    }
    included = {ref for ref in supports if ref[0] in hit_indexes}
    sets = {
        key: {
            (session_index, index)
            for session_index, stage in zip(hit_indexes, stages, strict=True)
            for index in stage[key]
        }
        for key in ('fact', 'fact_plus_dependencies', 'dependencies_plus_window', 'final')
    }
    retained = included if mode == 'full_source' else supports & sets['final']
    missing = supports - retained
    return {
        'required_count': len(supports),
        'full_required_count': len(included),
        'retained_count': len(retained),
        'supporting_turn_indices': [list(ref) for ref in sorted(supports)],
        'retained_supporting_turn_indices': [list(ref) for ref in sorted(retained)],
        'stage_retained': {
            name: len(supports & sets[key])
            for name, key in (
                ('fact', 'fact'),
                ('dependencies', 'fact_plus_dependencies'),
                ('window', 'dependencies_plus_window'),
                ('final', 'final'),
            )
        },
        'stage_support_indices': {
            name: [list(ref) for ref in sorted(supports & sets[key])]
            for name, key in (
                ('fact', 'fact'),
                ('dependencies', 'fact_plus_dependencies'),
                ('window', 'dependencies_plus_window'),
                ('final', 'final'),
            )
        },
        'dependency_added_support_indices': [
            list(ref) for ref in sorted(supports & (sets['fact_plus_dependencies'] - sets['fact']))
        ],
        'window_added_support_indices': [
            list(ref)
            for ref in sorted(
                supports & (sets['dependencies_plus_window'] - sets['fact_plus_dependencies'])
            )
        ],
        'missing_supports': [
            {
                'session_index': ref[0],
                'turn_index': ref[1],
                'reason': 'session_not_retrieved'
                if ref not in included
                else 'outside_evidence_bundle',
                'distance_to_dependency_bundle': min(
                    (
                        abs(ref[1] - index)
                        for session_index, index in sets['fact_plus_dependencies']
                        if session_index == ref[0]
                    ),
                    default=None,
                ),
                'role': case['haystack_sessions'][ref[0]][ref[1]]['role'],
            }
            for ref in sorted(missing)
        ],
    }


def loss_summary(rows: list[dict]) -> dict:
    losses = [item for row in rows for item in row['missing_supports']]
    return {
        'affected_cases': sum(bool(row['missing_supports']) for row in rows),
        'missing_support_count': len(losses),
        'by_reason': dict(Counter(item['reason'] for item in losses)),
        'by_role': dict(Counter(item['role'] for item in losses)),
        'cropping_distance': dict(
            Counter(
                item['distance_to_dependency_bundle']
                for item in losses
                if item['reason'] == 'outside_evidence_bundle'
            )
        ),
    }


def dependency_diagnostic(rows: list[dict], fixed_retrieval: list[dict], names: list[str]) -> dict:
    """Trace the legacy dependency-added supports after each label-blind assembly."""
    groups = {name: [row for row in rows if row['configuration'] == name] for name in names}

    def positions(selected: list[dict], key: str) -> set[tuple]:
        return {(row['question_id'], *ref) for row in selected for ref in row[key]}

    added = positions(groups['legacy_w1'], 'dependency_added_support_indices')
    near = added & positions(groups['fact_window_w1'], 'retained_supporting_turn_indices')
    far = added - near
    turn_count = sum(
        source['turn_count'] for item in fixed_retrieval for source in item['source_bindings']
    )
    return {
        'legacy_dependency_added_supports': len(added),
        'inside_fact_window_w1': len(near),
        'outside_fact_window_w1': len(far),
        'far_support_indices': [list(ref) for ref in sorted(far)],
        'retrieved_source_turns': turn_count,
        'by_configuration': {
            name: {
                'far_supports_retained': len(
                    far & positions(selected, 'retained_supporting_turn_indices')
                ),
                'far_supports_lost': [
                    list(ref)
                    for ref in sorted(far - positions(selected, 'retained_supporting_turn_indices'))
                ],
                'dependency_added_turns': sum(
                    len(stage['dependency_added'])
                    for row in selected
                    for stage in row['assembly_indexes']
                ),
                'selected_source_turns': sum(
                    len(source['indices'])
                    for row in selected
                    for source in row['rendered_source_indices']
                ),
            }
            for name, selected in groups.items()
        },
    }


async def evaluate_cases(
    cases: list[dict[str, Any]],
    *,
    repeats: int = 3,
    window_sizes: tuple[int, ...] = (0,),
    max_fact_anchors: int = 1,
    reading_configs: dict | None = None,
    binding_rule: str = 'substring',
    fact_granularity: str = 'paragraph',
    slot_mode: str = 'all',
    sentence_locator: bool = False,
) -> dict:
    rows, fixed_retrieval = [], []
    configs = reading_configs or reading_configurations(window_sizes)
    names = list(configs)
    for case in cases:
        context = await build_context(
            case,
            max_fact_anchors=max_fact_anchors,
            binding_rule=binding_rule,
            fact_granularity=fact_granularity,
            slot_mode=slot_mode,
            sentence_locator=sentence_locator,
        )
        located, locator_times = {}, []
        if sentence_locator:
            context.locate_fact_sentences()
            for _ in range(repeats):
                start = perf_counter()
                located = context.locate_fact_sentences()
                locator_times.append((perf_counter() - start) * 1000)
        hit_indexes = hit_session_indexes(case['haystack_sessions'], case['question'])
        # All configurations render before answer/has_answer labels are read.
        rendered = {
            name: measure_render(
                context,
                config['mode'],
                repeats,
                **{key: value for key, value in config.items() if key != 'mode'},
            )
            for name, config in configs.items()
        }
        assemblies = {
            name: assembly_stages(
                context,
                config['window_size'],
                dependency_mode=config.get('dependency_mode', 'all'),
                window_scope=config.get('window_scope', 'bundle'),
            )
            for name, config in configs.items()
        }
        for stages in assemblies.values():
            for session_index, stage in zip(hit_indexes, stages, strict=True):
                stage['session_index'] = session_index
        fixed_retrieval.append(
            {
                'question_id': case['question_id'],
                'hit_session_indices': hit_indexes,
                'sentence_locator_median_ms': median(locator_times) if locator_times else 0,
                'source_bindings': [
                    {
                        'session_index': session_index,
                        'episode_id': episode.uuid,
                        'turn_count': len(context.source_units[episode.uuid]),
                        'graph_fact_turn_indices': context.source_fact_indices[episode.uuid],
                        'fact_anchors': [
                            {'turn_index': turn_index, **span.model_dump()}
                            for turn_index, span in zip(
                                context.source_fact_indices[episode.uuid],
                                located.get(
                                    episode.uuid, context.source_fact_spans.get(episode.uuid, [])
                                ),
                                strict=False,
                            )
                        ],
                        'graph_fact': next(
                            edge.fact
                            for edge in context.graph_edges
                            if episode.uuid in edge.episodes
                        ),
                        'graph_facts': [
                            edge.fact
                            for edge in context.graph_edges
                            if episode.uuid in edge.episodes
                        ],
                        'source_span_fields': dict(
                            Counter(span.field for span in state.source_spans)
                        ),
                        'source_envelope_tokens': estimate_tokens(episode.content),
                        'source_spans_sha256': sha256(
                            json.dumps(
                                [span.model_dump() for span in state.source_spans], sort_keys=True
                            ).encode()
                        ).hexdigest(),
                    }
                    for session_index, episode, state in zip(
                        hit_indexes,
                        context.source_episodes,
                        context.semantic_states,
                        strict=True,
                    )
                ],
            }
        )
        for name, config in configs.items():
            mode, size = config['mode'], config['window_size']
            text, latency = rendered[name]
            stages = assemblies[name]
            scores = support_scores(case, hit_indexes, stages, mode)
            stage_texts = stage_source_texts(context, stages)
            rows.append(
                {
                    'question_id': case['question_id'],
                    'question_type': case['question_type'],
                    'configuration': name,
                    'mode': mode,
                    'window_size': None if mode == 'full_source' else size,
                    'chars': len(text),
                    'tokens': estimate_tokens(text),
                    'answer_visible': str(case['answer']).lower() in text.lower(),
                    'render_median_ms': latency,
                    'assembly_indexes': stages,
                    'rendered_source_indices': [
                        {
                            'session_index': session_index,
                            'indices': list(range(len(context.source_units[episode.uuid])))
                            if mode == 'full_source'
                            else stages[index]['final'],
                        }
                        for index, (session_index, episode) in enumerate(
                            zip(hit_indexes, context.source_episodes, strict=True)
                        )
                    ],
                    'stage_source_tokens': {
                        stage: estimate_tokens(value) for stage, value in stage_texts.items()
                    },
                    **scores,
                }
            )
    summary = {
        name: summarize_rows([row for row in rows if row['configuration'] == name])
        for name in names
    }
    finalize_summaries(summary)
    for item in summary.values():
        item['supporting_turn_recall'] = item['overall_recall']['micro']
    for name in names:
        summary[name]['losses'] = loss_summary(
            [row for row in rows if row['configuration'] == name]
        )
    summary['evidence_bundle'] = summary['evidence_bundle_w0']
    by_type = {}
    for question_type in sorted({case['question_type'] for case in cases}):
        group = {
            name: summarize_rows(
                [
                    row
                    for row in rows
                    if row['question_type'] == question_type and row['configuration'] == name
                ]
            )
            for name in names
        }
        finalize_summaries(group)
        for name in names:
            group[name]['losses'] = loss_summary(
                [
                    row
                    for row in rows
                    if row['configuration'] == name and row['question_type'] == question_type
                ]
            )
        by_type[question_type] = group
    return {
        'protocol': 'fixed retrieval; original message-index evidence/window ablation',
        'binding_rule': binding_rule,
        'fact_granularity': fact_granularity,
        'slot_mode': slot_mode,
        'sentence_locator': sentence_locator,
        'experiment_role': 'configuration exploration on oracle 500',
        'selection': 'all positive lexical-overlap sessions; Top-1 per session; optional Top-2 with relative score >= 0.5 and >= 2 distinct content-word matches; ties use earliest turn',
        'anchor_policy': {
            'max_fact_anchors': max_fact_anchors,
            'second_min_score_ratio': 0.5,
            'second_min_overlap_words': 2,
        },
        'labels': 'answer and has_answer used after all configurations have been assembled',
        'index_unit': 'zero-based original session and message indices; internal newlines stay in the same turn',
        'stage_scope': 'assembly_indexes and stage costs describe evidence assembly; rendered_source_indices describe the selected configuration',
        'timing_repeats': repeats,
        'environment': experiment_environment(),
        'reading_configurations': configs,
        'dependency_diagnostic': dependency_diagnostic(rows, fixed_retrieval, names)
        if 'legacy_w1' in configs
        else {},
        'summary': summary,
        'by_question_type': by_type,
        'pareto': pareto_analysis(summary, names),
        'fixed_retrieval': fixed_retrieval,
        'rows': rows,
    }


def binding_diagnostic(results: dict) -> dict:
    """Score removed dependencies and remote supports after all rules have rendered."""
    baseline = results['substring']
    reference = {
        row['question_id']: row
        for row in baseline['rows']
        if row['configuration'] == 'evidence_bundle_w1'
    }

    def dependencies(row: dict) -> set[tuple[int, int]]:
        return {
            (stage['session_index'], index)
            for stage in row['assembly_indexes']
            for index in stage['dependency_added']
        }

    def distance(row: dict, ref: tuple[int, int]) -> int | None:
        return min(
            (
                abs(ref[1] - index)
                for stage in row['assembly_indexes']
                if stage['session_index'] == ref[0]
                for index in stage['fact']
            ),
            default=None,
        )

    diagnostic = {}
    for rule, result in results.items():
        removed, removed_supports, rescued, far_total, far_kept = 0, 0, 0, 0, 0
        baseline_lost, gained, window_lost = 0, 0, 0
        buckets, examples = {}, []
        for row in result['rows']:
            if row['configuration'] != 'evidence_bundle_w1':
                continue
            base = reference[row['question_id']]
            supports = {tuple(ref) for ref in row['supporting_turn_indices']}
            retained = {tuple(ref) for ref in row['retained_supporting_turn_indices']}
            deleted = dependencies(base) - dependencies(row)
            previous = {tuple(ref) for ref in base['retained_supporting_turn_indices']}
            baseline_lost += len(previous - retained)
            gained += len(retained - previous)
            window_lost += len((previous - retained) - deleted)
            removed += len(deleted)
            removed_supports += len(deleted & supports)
            rescued += len(deleted & supports & retained)
            far = {ref for ref in dependencies(base) & supports if distance(base, ref) > 1}
            far_total += len(far)
            far_kept += len(far & retained)
            for ref in supports:
                value = distance(base, ref)
                key = (
                    'unretrieved'
                    if value is None
                    else str(value)
                    if value <= 2
                    else '3-5'
                    if value <= 5
                    else '6+'
                )
                bucket = buckets.setdefault(key, {'total': 0, 'retained': 0})
                bucket['total'] += 1
                bucket['retained'] += int(ref in retained)
            if len(examples) < 12 and deleted & supports - retained:
                examples.append(
                    {
                        'question_id': row['question_id'],
                        'question_type': row['question_type'],
                        'lost_removed_supports': [
                            list(ref) for ref in sorted(deleted & supports - retained)
                        ],
                    }
                )
        for bucket in buckets.values():
            bucket['recall'] = bucket['retained'] / bucket['total']
        diagnostic[rule] = {
            'removed_dependency_turns': removed,
            'removed_annotated_support_turns': removed_supports,
            'removed_supports_still_retained': rescued,
            'removed_supports_lost': removed_supports - rescued,
            'baseline_retained_supports_lost': baseline_lost,
            'new_supports_gained': gained,
            'window_supports_lost_after_dependency_removal': window_lost,
            'reference_far_supports': far_total,
            'far_supports_retained': far_kept,
            'far_support_recall': far_kept / far_total if far_total else None,
            'by_support_distance': buckets,
            'loss_examples': examples,
            'source_envelope_tokens': sum(
                binding['source_envelope_tokens']
                for item in result['fixed_retrieval']
                for binding in item['source_bindings']
            ),
        }
    return diagnostic


async def evaluate_binding_calibration(
    cases: list[dict], *, repeats: int = 3, slot_mode: str = 'all'
) -> dict:
    results = {
        rule: await evaluate_cases(
            cases, repeats=repeats, window_sizes=(0, 1, 2), binding_rule=rule, slot_mode=slot_mode
        )
        for rule in ('substring', 'word_boundary', 'fact_scope')
    }
    retrieval = [
        [
            (
                item['question_id'],
                item['hit_session_indices'],
                [
                    (binding['graph_fact_turn_indices'], binding['graph_facts'])
                    for binding in item['source_bindings']
                ],
            )
            for item in result['fixed_retrieval']
        ]
        for result in results.values()
    ]
    summaries = {
        f'{rule}/{name}': summary
        for rule, result in results.items()
        for name, summary in result['summary'].items()
        if name != 'evidence_bundle'
    }
    return {
        'protocol': 'write-binding calibration; fixed lexical retrieval, Top-1, renderer and windows',
        'experiment_role': 'exploratory reuse of oracle 500',
        'fixed_retrieval_equal': all(item == retrieval[0] for item in retrieval),
        'binding_calibration': results,
        'comparison_w1': binding_diagnostic(results),
        'joint_pareto': pareto_analysis(summaries, list(summaries)),
    }


async def evaluate_anchor_calibration(
    cases: list[dict], *, repeats: int = 3, slot_mode: str = 'all'
) -> dict:
    results = {
        f'{rule}/{granularity}': await evaluate_cases(
            cases,
            repeats=repeats,
            window_sizes=(0, 1, 2),
            binding_rule=rule,
            fact_granularity=granularity,
            slot_mode=slot_mode,
        )
        for rule in ('substring', 'fact_scope', 'word_form_scope')
        for granularity in ('paragraph', 'sentence')
    }
    retrieval = [
        [
            (
                item['question_id'],
                item['hit_session_indices'],
                [binding['graph_fact_turn_indices'] for binding in item['source_bindings']],
            )
            for item in result['fixed_retrieval']
        ]
        for result in results.values()
    ]
    summaries = {
        f'{variant}/{name}': summary
        for variant, result in results.items()
        for name, summary in result['summary'].items()
        if name != 'evidence_bundle'
    }
    return {
        'protocol': 'sentence-anchor and word-form ablation; fixed Sessions and Top-1 turns',
        'experiment_role': 'exploratory reuse of oracle 500',
        'fixed_retrieval_equal': all(item == retrieval[0] for item in retrieval),
        'summary': summaries,
        'anchor_calibration': results,
        'comparison_w1': {
            granularity: binding_diagnostic(
                {
                    rule: results[f'{rule}/{granularity}']
                    for rule in ('substring', 'fact_scope', 'word_form_scope')
                }
            )
            for granularity in ('paragraph', 'sentence')
        },
        'joint_pareto': pareto_analysis(summaries, list(summaries)),
    }


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--data',
        type=Path,
        default=Path(__file__).parents[2]
        / 'tests/evals/data/longmemeval_data/longmemeval_oracle.json',
    )
    parser.add_argument('--offset', type=int, default=0)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--max-fact-anchors', type=int, choices=(1, 2), default=1)
    parser.add_argument('--dependency-ablation', action='store_true')
    parser.add_argument('--binding-calibration', action='store_true')
    parser.add_argument('--anchor-calibration', action='store_true')
    parser.add_argument('--slot-mode', choices=('first', 'all'), default='all')
    parser.add_argument(
        '--binding-rule',
        default='substring',
        choices=('substring', 'word_boundary', 'fact_scope', 'word_form_scope'),
    )
    parser.add_argument(
        '--fact-granularity', choices=('paragraph', 'sentence'), default='paragraph'
    )
    parser.add_argument('--sentence-locator', action='store_true')
    parser.add_argument('--output-json', type=Path)
    parser.add_argument('--qa-dir', type=Path)
    parser.add_argument('--qa-prepare-only', action='store_true')
    parser.add_argument('--model', default='glm-5.1')
    parser.add_argument('--base-url', default='http://118.89.124.131:3001/v1')
    parser.add_argument('--api-key-env', default='HRECONGRAPH_QA_API_KEY')
    parser.add_argument('--concurrency', type=int, default=100)
    args = parser.parse_args()
    cases = load_cases(args.data, offset=args.offset, limit=args.limit)
    if args.qa_dir:
        result = await run_qa(cases, args)
        print(json.dumps(result.get('summary', result), ensure_ascii=False, indent=2))
        return
    result = (
        await evaluate_anchor_calibration(cases, repeats=args.repeats, slot_mode=args.slot_mode)
        if args.anchor_calibration
        else await evaluate_binding_calibration(
            cases, repeats=args.repeats, slot_mode=args.slot_mode
        )
        if args.binding_calibration
        else await evaluate_cases(
            cases,
            repeats=args.repeats,
            window_sizes=(0, 1, 2),
            max_fact_anchors=args.max_fact_anchors,
            slot_mode=args.slot_mode,
            binding_rule=args.binding_rule,
            fact_granularity=args.fact_granularity,
            sentence_locator=args.sentence_locator,
            reading_configs=reading_configurations((0, 1, 2), ablation=args.dependency_ablation),
        )
    )
    result['dataset'] = {
        'path': str(args.data),
        'sha256': sha256(args.data.read_bytes()).hexdigest(),
        'offset': args.offset,
        'case_count': len(cases),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(
        json.dumps(result.get('summary', result.get('comparison_w1')), ensure_ascii=False, indent=2)
    )


if __name__ == '__main__':
    asyncio.run(main())
