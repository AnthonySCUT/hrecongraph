"""Paired evidence-recall and cost ablation on fixed reading contexts."""

import argparse
import asyncio
import json
import platform
import re
from pathlib import Path
from statistics import mean, median
from time import perf_counter

from graphiti_core.edges import EntityEdge
from graphiti_core.nodes import EpisodeType, EpisodicNode
from graphiti_core.utils.content_chunking import estimate_tokens
from graphiti_core.utils.datetime_utils import utc_now
from h_recongraph.memory.graphiti_store import GraphitiMemoryStore
from h_recongraph.memory.manager import HReConGraphMemory
from h_recongraph.memory.relinking import HierarchicalMemoryContext
from h_recongraph.schemas.memory_unit import SourceSpan, UnifiedMemoryUnit
from h_recongraph.schemas.semantic_state import MemorySemanticState


def build_context(case: dict) -> HierarchicalMemoryContext:
    now = utc_now()
    sources = [
        EpisodicNode(
            uuid=source['id'],
            name=case['id'],
            group_id='user:eval',
            source=EpisodeType.text,
            source_description='fixed reading case',
            content=source['content'],
            valid_at=now,
        )
        for source in case['sources']
    ]
    edges = [
        EntityEdge(
            uuid=fact['id'],
            created_at=now,
            group_id='user:eval',
            source_node_uuid=f'node-{i}',
            target_node_uuid=f'node-{i + 1}',
            name='RELATION',
            fact=fact['text'],
            episodes=[fact['source']],
        )
        for i, fact in enumerate(case['facts'])
    ]
    spans = [SourceSpan(evidence_id=case['id'], **span) for span in case['spans']]
    states = (
        [
            MemorySemanticState(
                edge_or_subgraph_id=case['id'],
                source_spans=spans,
                state_type='evidence_supported',
            )
        ]
        if spans
        else []
    )
    return HierarchicalMemoryContext(
        query=case['question'],
        selected_layers=['M1', 'M2'],
        graph_edges=edges,
        semantic_states=states,
        source_episodes=sources,
    )


async def build_written_contexts(cases: list[dict], bindings: dict) -> dict:
    """Apply source-only bindings, serialize, then recover for reading."""
    contexts = {}
    for case in cases:
        context = build_context(case)
        writer = HReConGraphMemory()
        for source in context.source_episodes:
            memory = UnifiedMemoryUnit(
                memory_id=f'stm-{source.uuid}',
                user_id='eval',
                layer='M1',
                lifecycle_status='active_short_term',
                raw_content=source.content,
                source_spans=[
                    SourceSpan(evidence_id=source.uuid, **span) for span in bindings[case['id']]
                ],
            )
            await writer.add_memory(memory)
            source.content = GraphitiMemoryStore(None)._episode_body(memory)
        reader = HReConGraphMemory(store=GraphitiMemoryStore(None))
        reader._restore_source_memories(
            context.source_episodes, context.graph_edges, 'eval', ['M1', 'M2']
        )
        context.semantic_states = list(reader.semantic_states.values())
        contexts[case['id']] = context
    return contexts


async def build_naturally_bound_contexts(cases: list[dict]) -> dict:
    """Write source text without external spans and recover automatic bindings."""
    contexts = {}
    for case in cases:
        context = build_context(case)
        writer = HReConGraphMemory()
        for source in context.source_episodes:
            memory = UnifiedMemoryUnit(
                memory_id=f'stm-{source.uuid}',
                user_id='eval',
                layer='M1',
                lifecycle_status='active_short_term',
                raw_content=source.content,
            )
            await writer.add_memory(memory)
            source.content = GraphitiMemoryStore(None)._episode_body(memory)
        reader = HReConGraphMemory(store=GraphitiMemoryStore(None))
        reader._restore_source_memories(
            context.source_episodes, context.graph_edges, 'eval', ['M1', 'M2']
        )
        context.semantic_states = list(reader.semantic_states.values())
        contexts[case['id']] = context
    return contexts


def reading_configurations(window_sizes: tuple[int, ...], *, ablation: bool = False) -> dict:
    """Separate repeated edges, window scope and dependency filtering in paired runs."""
    configs = {'full_source': {'mode': 'full_source', 'window_size': 0}}
    configs.update(
        {
            f'evidence_bundle_w{size}': {'mode': 'evidence_bundle', 'window_size': size}
            for size in window_sizes
        }
    )
    if ablation:
        configs['full_source']['deduplicate_edge_facts'] = False
        configs['full_source_dedup'] = {'mode': 'full_source', 'window_size': 0}
        configs['legacy_w1'] = {
            'mode': 'evidence_bundle',
            'window_size': 1,
            'deduplicate_edge_facts': False,
        }
        configs['fact_window_w1'] = {
            'mode': 'evidence_bundle',
            'window_size': 1,
            'dependency_mode': 'none',
            'window_scope': 'fact',
        }
        configs['all_dependencies_fact_window_w1'] = {
            'mode': 'evidence_bundle',
            'window_size': 1,
            'window_scope': 'fact',
        }
        configs.update(
            {
                f'lexical_fact_window_w{size}': {
                    'mode': 'evidence_bundle',
                    'window_size': size,
                    'dependency_mode': 'lexical',
                    'window_scope': 'fact',
                }
                for size in (0, 1, 2)
            }
        )
    return configs


def assembly_stages(
    context: HierarchicalMemoryContext,
    window_size: int,
    *,
    dependency_mode: str = 'all',
    window_scope: str = 'bundle',
) -> list[dict]:
    """Record cumulative source indexes and disjoint additions, before scoring labels."""
    stages = []
    for episode in context.source_episodes:
        indexes = context.source_indices(
            episode, window_size, dependency_mode=dependency_mode, window_scope=window_scope
        )
        fact = set(indexes['fact'])
        dependencies = fact | set(indexes['dependencies'])
        window = dependencies | set(indexes['window'])
        stages.append(
            {
                'episode_id': episode.uuid,
                'fact': sorted(fact),
                'fact_plus_dependencies': sorted(dependencies),
                'dependencies_plus_window': sorted(window),
                'dependency_added': sorted(dependencies - fact),
                'window_added': indexes['window'],
                'fallback_added': indexes['fallback'],
                'final': sorted(window | set(indexes['fallback'])),
            }
        )
    return stages


def stage_source_texts(context: HierarchicalMemoryContext, stages: list[dict]) -> dict[str, str]:
    sources = {episode.uuid: episode for episode in context.source_episodes}
    return {
        name: '\n\n'.join(
            context._render_source_units(sources[stage['episode_id']], set(stage[key]))
            for stage in stages
        )
        for name, key in (
            ('fact', 'fact'),
            ('dependencies', 'fact_plus_dependencies'),
            ('window', 'dependencies_plus_window'),
            ('final', 'final'),
        )
    }


def measure_render(
    context, mode: str, repeats: int, window_size: int = 0, **render_options
) -> tuple[str, float]:
    text = context.render_text(mode, window_size=window_size, **render_options)
    durations = []
    for _ in range(repeats):
        start = perf_counter()
        context.render_text(mode, window_size=window_size, **render_options)
        durations.append((perf_counter() - start) * 1000)
    return text, median(durations)


def summarize_rows(rows: list[dict]) -> dict:
    overall = [row for row in rows if row['required_count'] > 0]
    conditional = [row for row in rows if row['full_required_count'] > 0]

    def recall(selected: list[dict], denominator: str) -> dict:
        total = sum(row[denominator] for row in selected)
        retained = sum(row['retained_count'] for row in selected)
        return {
            'macro': mean(row['retained_count'] / row[denominator] for row in selected)
            if selected
            else None,
            'micro': retained / total if total else None,
            'retained': retained,
            'total': total,
            'valid_cases': len(selected),
        }

    stage_counts = {
        stage: sum(row['stage_retained'][stage] for row in overall)
        for stage in ('fact', 'dependencies', 'window', 'final')
    }
    return {
        'case_count': len(rows),
        'mean_context_chars': mean(row['chars'] for row in rows),
        'mean_context_tokens': mean(row['tokens'] for row in rows),
        'total_chars': sum(row['chars'] for row in rows),
        'total_tokens': sum(row['tokens'] for row in rows),
        'overall_recall': recall(overall, 'required_count'),
        'conditional_recall': recall(conditional, 'full_required_count'),
        'complete_cases': sum(row['retained_count'] == row['required_count'] for row in overall),
        'complete_rate': mean(row['retained_count'] == row['required_count'] for row in overall)
        if overall
        else None,
        'answer_visible_rate': mean(row['answer_visible'] for row in rows),
        'median_render_ms': median(row['render_median_ms'] for row in rows),
        'stage_counts': stage_counts,
        'dependency_added_supports': stage_counts['dependencies'] - stage_counts['fact'],
        'window_added_supports': stage_counts['window'] - stage_counts['dependencies'],
        'fallback_added_supports': stage_counts['final'] - stage_counts['window'],
        'stage_source_tokens': {
            stage: sum(row['stage_source_tokens'][stage] for row in rows)
            for stage in ('fact', 'dependencies', 'window', 'final')
        },
    }


def pareto_analysis(summary: dict, configurations: list[str]) -> dict:
    result = {}
    for metric in ('overall_recall', 'conditional_recall'):
        result[metric] = {}
        for average in ('micro', 'macro'):
            scored = [name for name in configurations if summary[name][metric][average] is not None]
            frontier = [
                name
                for name in scored
                if not any(
                    summary[other][metric][average] >= summary[name][metric][average]
                    and summary[other]['mean_context_tokens']
                    <= summary[name]['mean_context_tokens']
                    and (
                        summary[other][metric][average] > summary[name][metric][average]
                        or summary[other]['mean_context_tokens']
                        < summary[name]['mean_context_tokens']
                    )
                    for other in scored
                )
            ]
            result[metric][average] = {
                'frontier': frontier,
                'reference_lines': {
                    str(threshold): min(
                        (name for name in scored if summary[name][metric][average] >= threshold),
                        key=lambda name: summary[name]['mean_context_tokens'],
                        default=None,
                    )
                    for threshold in (0.90, 0.95)
                },
            }
    return result


def experiment_environment() -> dict:
    return {
        'python': platform.python_version(),
        'platform': platform.platform(),
        'token_estimator': 'graphiti_core.utils.content_chunking.estimate_tokens: len(text) // 4',
        'timing_scope': 'render_text only; warmup then repeated measurements',
    }


def finalize_summaries(summary: dict) -> None:
    full = summary['full_source']
    for item in summary.values():
        item['token_reduction_vs_full_source'] = 1 - item['total_tokens'] / full['total_tokens']
        item['char_reduction_vs_full_source'] = 1 - item['total_chars'] / full['total_chars']
        item['dependency_added_source_tokens'] = (
            item['stage_source_tokens']['dependencies'] - item['stage_source_tokens']['fact']
        )
        item['window_added_source_tokens'] = (
            item['stage_source_tokens']['window'] - item['stage_source_tokens']['dependencies']
        )


def evaluate_cases(
    cases: list[dict],
    repeats: int = 100,
    contexts: dict | None = None,
    window_sizes: tuple[int, ...] = (0,),
    reading_configs: dict | None = None,
) -> dict:
    rows = []
    configs = reading_configs or reading_configurations(window_sizes)
    for case in cases:
        context = contexts[case['id']] if contexts is not None else build_context(case)
        full_text = context.render_text('full_source')
        for name, config in configs.items():
            mode, size = config['mode'], config['window_size']
            options = {key: value for key, value in config.items() if key != 'mode'}
            text, latency = measure_render(context, mode, repeats, **options)
            stages = assembly_stages(
                context,
                size,
                dependency_mode=config.get('dependency_mode', 'all'),
                window_scope=config.get('window_scope', 'bundle'),
            )
            stage_texts = stage_source_texts(context, stages)
            missing = [item for item in case['required'] if item not in stage_texts['final']]
            retained = len(case['required']) - len(missing)
            if mode == 'full_source':
                retained = sum(item in full_text for item in case['required'])
                missing = [item for item in case['required'] if item not in full_text]
            match = re.search(case['answer_pattern'], text)
            answer = match.group('answer') if match else ''
            rows.append(
                {
                    'case_id': case['id'],
                    'dimension': case['dimension'],
                    'mode': mode,
                    'window_size': None if mode == 'full_source' else size,
                    'configuration': name,
                    'chars': len(text),
                    'tokens': estimate_tokens(text),
                    'required_count': len(case['required']),
                    'full_required_count': sum(item in full_text for item in case['required']),
                    'retained_count': retained,
                    'complete': not missing,
                    'missing': missing,
                    'probe_answer': answer,
                    'probe_correct': answer == case['expected_answer'],
                    'answer_visible': case['expected_answer'].lower() in text.lower(),
                    'render_median_ms': latency,
                    'assembly_indexes': stages,
                    'stage_retained': {
                        stage: sum(item in value for item in case['required'])
                        for stage, value in stage_texts.items()
                    },
                    'stage_source_tokens': {
                        stage: estimate_tokens(value) for stage, value in stage_texts.items()
                    },
                    'required_evidence': [
                        {
                            'text': item,
                            'locations': [
                                {'episode_id': episode.uuid, 'paragraph_index': index}
                                for episode in context.source_episodes
                                for index, paragraph in enumerate(context.source_segments(episode))
                                if item in paragraph
                            ],
                        }
                        for item in case['required']
                    ],
                }
            )
    configurations = list(configs)
    summary = {}
    for configuration in configurations:
        selected = [row for row in rows if row['configuration'] == configuration]
        summary[configuration] = {
            **summarize_rows(selected),
            'evidence_retention': sum(row['retained_count'] for row in selected)
            / sum(row['required_count'] for row in selected),
            'extractive_probe_correct': sum(row['probe_correct'] for row in selected),
            'dimensions': {
                dimension: {
                    'complete': sum(
                        row['complete'] for row in selected if row['dimension'] == dimension
                    ),
                    'count': sum(row['dimension'] == dimension for row in selected),
                }
                for dimension in ('fact', 'constraint', 'path', 'causal')
            },
        }
    finalize_summaries(summary)
    # Keep the existing w=0 summary key for the earlier paired experiments.
    summary['evidence_bundle'] = summary['evidence_bundle_w0']
    return {
        'protocol': 'fixed retrieval; paragraph-level evidence-window ablation and extractive probes',
        'length_unit': 'unicode_characters',
        'latency_scope': 'render_text only',
        'timing_repeats': repeats,
        'environment': experiment_environment(),
        'reading_configurations': configs,
        'summary': summary,
        'pareto': pareto_analysis(summary, configurations),
        'rows': rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--cases', type=Path, default=Path(__file__).with_name('read_context_cases.json')
    )
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--write-bindings', type=Path)
    parser.add_argument('--natural-bindings', action='store_true')
    parser.add_argument('--dependency-ablation', action='store_true')
    parser.add_argument('--output-json', type=Path, required=True)
    args = parser.parse_args()
    cases = json.loads(args.cases.read_text(encoding='utf-8'))
    sizes = (0, 1, 2)
    configs = reading_configurations(sizes, ablation=args.dependency_ablation)
    if args.write_bindings:
        bindings = json.loads(args.write_bindings.read_text(encoding='utf-8'))
        contexts = asyncio.run(build_written_contexts(cases, bindings))
        result = {
            'binding_protocol': 'separate source-only paragraph annotations; fixed graph edges',
            'original_bindings': evaluate_cases(
                cases, args.repeats, window_sizes=sizes, reading_configs=configs
            ),
            'complete_write_bindings': evaluate_cases(
                cases, args.repeats, contexts, sizes, reading_configs=configs
            ),
        }
    elif args.natural_bindings:
        contexts = asyncio.run(build_naturally_bound_contexts(cases))
        result = {
            'binding_protocol': 'SemanticStateConstructor automatic paragraph bindings',
            'natural_auto_bindings': evaluate_cases(
                cases, args.repeats, contexts, sizes, reading_configs=configs
            ),
        }
    else:
        result = evaluate_cases(cases, args.repeats, window_sizes=sizes, reading_configs=configs)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(
        json.dumps(
            {
                name: value['summary']
                for name, value in result.items()
                if isinstance(value, dict) and 'summary' in value
            }
            or result['summary'],
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == '__main__':
    main()
