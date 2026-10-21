"""Explore LongMemEval evidence-recall/token Pareto configurations on fixed retrieval."""

from __future__ import annotations

import argparse
import asyncio
import json
from collections import Counter
from hashlib import sha256
from pathlib import Path
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
    stage_source_texts,
    summarize_rows,
)
from h_recongraph.memory.graphiti_store import GraphitiMemoryStore
from h_recongraph.memory.manager import HReConGraphMemory
from h_recongraph.memory.ranker import MemoryUtilityRanker
from h_recongraph.memory.relinking import HierarchicalMemoryContext
from h_recongraph.schemas.memory_unit import UnifiedMemoryUnit


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


async def build_context(case: dict, *, max_fact_anchors: int = 1) -> HierarchicalMemoryContext:
    """Use only question/source text to build one fixed lexical graph adapter."""
    now = utc_now()
    writer = HReConGraphMemory()
    episodes, edges, states = [], [], []
    units, fact_indices = {}, {}
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
            edges.append(
                EntityEdge(
                    uuid=f'{episode.uuid}-edge' + (f'-{anchor_index}' if anchor_index else ''),
                    created_at=now,
                    group_id='user:eval',
                    source_node_uuid=f'{episode.uuid}-source',
                    target_node_uuid=f'{episode.uuid}-target',
                    name='RELEVANT_TURN',
                    fact=turn_text(session[turn_index]),
                    episodes=[episode.uuid],
                )
            )
        episodes.append(episode)
        states.append(writer.semantic_states[memory.semantic_state_id])
    return HierarchicalMemoryContext(
        query=case['question'],
        selected_layers=['M1'],
        graph_edges=edges,
        semantic_states=states,
        source_episodes=episodes,
        source_units=units,
        source_fact_indices=fact_indices,
    )


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
) -> dict:
    rows, fixed_retrieval = [], []
    configs = reading_configs or reading_configurations(window_sizes)
    names = list(configs)
    for case in cases:
        context = await build_context(case, max_fact_anchors=max_fact_anchors)
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
                'source_bindings': [
                    {
                        'session_index': session_index,
                        'episode_id': episode.uuid,
                        'turn_count': len(context.source_units[episode.uuid]),
                        'graph_fact_turn_indices': context.source_fact_indices[episode.uuid],
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
    parser.add_argument('--output-json', type=Path, required=True)
    args = parser.parse_args()
    result = await evaluate_cases(
        load_cases(args.data, offset=args.offset, limit=args.limit),
        repeats=args.repeats,
        window_sizes=(0, 1, 2),
        max_fact_anchors=args.max_fact_anchors,
        reading_configs=reading_configurations((0, 1, 2), ablation=args.dependency_ablation),
    )
    result['dataset'] = {
        'path': str(args.data),
        'sha256': sha256(args.data.read_bytes()).hexdigest(),
        'offset': args.offset,
        'case_count': len(result['fixed_retrieval']),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(result['summary'], ensure_ascii=False, indent=2))


if __name__ == '__main__':
    asyncio.run(main())
