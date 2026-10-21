"""Paired multi-turn replay of dimension deltas through the graph read/write interfaces."""

import argparse
import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

from graphiti_core.edges import EntityEdge
from graphiti_core.graphiti import AddEpisodeResults
from graphiti_core.nodes import EntityNode, EpisodeType, EpisodicNode
from graphiti_core.utils.datetime_utils import utc_now
from h_recongraph.memory.manager import HReConGraphMemory
from h_recongraph.memory.ranker import MemoryUtilityRanker
from h_recongraph.schemas.memory_unit import SourceSpan, UnifiedMemoryUnit

DIMENSIONS = ('fact', 'constraint', 'path', 'causal')


class ReplayGraphiti:
    """In-memory graph driver for explicit, annotated update sequences."""

    def __init__(self):
        self.episodes = {}
        self.edges = {}
        self.writes = []
        self.driver = SimpleNamespace(
            graph_operations_interface=SimpleNamespace(
                episodic_node_get_by_uuids=self.load_episodes,
                edge_get_by_uuids=self.load_edges,
                edge_save=self.save_edge,
                edge_delete=self.delete_edge,
                node_delete=self.delete_episode,
            )
        )

    async def add_episode(self, **kwargs):
        self.writes.append(kwargs['episode_body'])
        body = json.loads(kwargs['episode_body'])
        episode = EpisodicNode(
            uuid=f'episode-{len(self.writes)}',
            name=kwargs['name'],
            group_id=kwargs['group_id'],
            source=EpisodeType.text,
            source_description=kwargs['source_description'],
            content=kwargs['episode_body'],
            valid_at=kwargs['reference_time'],
        )
        spans = body['h_recongraph']['source_spans']
        facts = list(dict.fromkeys(s['text'] for s in spans if s['field'] in {'fact', 'path'}))
        facts = facts or [body['content'].split('\n\n')[0]]
        nodes = [
            EntityNode(uuid=f'{episode.uuid}-node-{i}', name=f'node-{i}', group_id=episode.group_id)
            for i in range(len(facts) + 1)
        ]
        edges = [
            EntityEdge(
                uuid=f'{episode.uuid}-edge-{i}',
                group_id=episode.group_id,
                source_node_uuid=nodes[i].uuid,
                target_node_uuid=nodes[i + 1].uuid,
                created_at=utc_now(),
                name='RELATION',
                fact=fact,
                episodes=[episode.uuid],
            )
            for i, fact in enumerate(facts)
        ]
        episode.entity_edges = [edge.uuid for edge in edges]
        self.episodes[episode.uuid] = episode
        self.edges.update({edge.uuid: edge for edge in edges})
        return AddEpisodeResults(
            episode=episode,
            nodes=nodes,
            edges=edges,
            episodic_edges=[],
            communities=[],
            community_edges=[],
        )

    async def search(self, query, group_ids, num_results):
        ranker = MemoryUtilityRanker()
        return [
            edge
            for edge in self.edges.values()
            if edge.group_id in group_ids
            and any(
                ranker.lexical_relevance(json.loads(self.episodes[eid].content)['content'], query)
                > 0
                for eid in edge.episodes
            )
        ][:num_results]

    async def load_episodes(self, node_type, driver, uuids):
        return [self.episodes[uuid] for uuid in uuids if uuid in self.episodes]

    async def load_edges(self, edge_type, driver, uuids):
        return [self.edges[uuid] for uuid in uuids if uuid in self.edges]

    async def save_edge(self, edge, driver):
        self.edges[edge.uuid] = edge

    async def delete_edge(self, edge, driver):
        del self.edges[edge.uuid]

    async def delete_episode(self, episode, driver):
        del self.episodes[episode.uuid]


def make_bundle(memory_id, spans):
    return UnifiedMemoryUnit(
        memory_id=memory_id,
        user_id='eval',
        layer='M1',
        lifecycle_status='active_short_term',
        raw_content='\n\n'.join(dict.fromkeys(span.text for span in spans)),
        source_spans=spans,
    )


def evidence_key(span):
    return (span.field, span.evidence_id, span.text, span.start, span.end)


def kept_snapshot(memory):
    return memory.model_dump(
        mode='json',
        include={
            'memory_id',
            'raw_content',
            'source_spans',
            'evidence_ids',
            'semantic_state',
            'semantic_state_id',
            'graph_binding',
            'lifecycle_status',
            'version',
            'supersedes',
        },
    )


def graph_snapshot(client, memory_id):
    episodes = [
        episode
        for episode in client.episodes.values()
        if json.loads(episode.content)['h_recongraph']['memory_id'] == memory_id
    ]
    edge_ids = {edge_id for episode in episodes for edge_id in episode.entity_edges}
    return {
        'sources': [episode.model_dump(mode='json') for episode in episodes],
        'edges': [
            edge.model_dump(mode='json') for edge in client.edges.values() if edge.uuid in edge_ids
        ],
    }


def score_checkpoint(turn, actual, context, expected_ids):
    expected = {
        dimension: [SourceSpan(**span) for span in turn['expected'][dimension]]
        for dimension in DIMENSIONS
    }
    actual_keys = [{evidence_key(span) for span in memory.source_spans} for memory in actual]
    identity_ok = {memory.memory_id for memory in actual} == expected_ids
    dimension_complete = {
        dimension: bool(actual)
        and identity_ok
        and all(all(evidence_key(span) in keys for span in required) for keys in actual_keys)
        and all(span.text in context.context_text for span in required)
        for dimension, required in expected.items()
        if required
    }
    changed = (
        identity_ok
        and all(
            all(
                {evidence_key(span) for span in memory.source_spans if span.field == dimension}
                == {evidence_key(span) for span in expected[dimension]}
                for dimension in turn['changed_dimensions']
            )
            for memory in actual
        )
        and all(
            span.text in context.context_text
            for dimension in turn['changed_dimensions']
            for span in expected[dimension]
        )
    )
    stale_hits = [text for text in turn['stale'] if text in context.context_text]
    return {
        'changed_correct': changed,
        'dimension_complete': dimension_complete,
        'stale_eligible': bool(turn['stale']),
        'stale_hits': stale_hits,
        'active_ids': [memory.memory_id for memory in actual],
        'actual_spans': {
            memory.memory_id: [span.model_dump() for span in memory.source_spans]
            for memory in actual
        },
        'expected': turn['expected'],
        'context_text': context.context_text,
    }


async def replay_scenario(case, condition):
    client = ReplayGraphiti()
    writer = HReConGraphMemory(client)
    kept = {}
    for item in case['kept']:
        spans = [
            SourceSpan(evidence_id=f'ev-{item["id"]}', field=dimension, text=text)
            for dimension, texts in item['spans'].items()
            for text in texts
        ]
        memory = await writer.add_memory(make_bundle(item['id'], spans))
        kept[memory.memory_id] = (kept_snapshot(memory), graph_snapshot(client, memory.memory_id))
    rows = []
    for turn in case['turns']:
        updates = {
            dimension: [SourceSpan(**span) for span in spans]
            for dimension, spans in turn['input'].items()
        }
        incoming = [span for spans in updates.values() for span in spans]
        if turn['action'] == 'add':
            current = await writer.add_interaction(turn['utterance'], 'eval', source_spans=incoming)
            long = await writer._consolidate_memory(current)
            expected_ids = {current.memory_id, long.memory_id}
        elif turn['action'] in {'append', 'modify'}:
            if condition == 'incremental_merge':
                delta = (
                    {'append_spans': incoming}
                    if turn['action'] == 'append'
                    else {'span_updates': updates}
                )
                current = await writer.update_memory(current.memory_id, operation='modify', **delta)
            else:
                current = await writer.update_memory(
                    current.memory_id,
                    operation='modify',
                    replacement=make_bundle('draft', incoming),
                )
        elif turn['action'] == 'supersede':
            current = await writer.update_memory(
                current.memory_id,
                operation='supersede',
                replacement=make_bundle(f'stm-{case["id"]}-new', incoming),
            )
            expected_ids = {current.memory_id}
        else:
            await writer.update_memory(current.memory_id, operation='delete')
            expected_ids = set()
        for cache in ('warm', 'restored'):
            for mode in ('full_source', 'evidence_bundle'):
                reader = writer if cache == 'warm' else HReConGraphMemory(client)
                context = await reader.retrieve(case['query'], 'eval', source_mode=mode)
                active = [
                    memory
                    for memory in [
                        *reader.short_term_memories.values(),
                        *reader.long_term_memories.values(),
                    ]
                    if memory.memory_id not in kept
                    and reader._is_readable_memory(memory, 'eval', ['M1', 'M2'])
                ]
                row = score_checkpoint(turn, active, context, expected_ids)
                kept_checks = {}
                for item in case['kept']:
                    memory_id = item['id']
                    await reader.retrieve(
                        item['spans']['fact'][0].split()[0], 'eval', source_mode=mode
                    )
                    memory = reader.short_term_memories[memory_id]
                    kept_checks[memory_id] = (
                        kept_snapshot(memory) == kept[memory_id][0]
                        and graph_snapshot(client, memory_id) == kept[memory_id][1]
                    )
                rows.append(
                    {
                        'case_id': case['id'],
                        'turn': turn['turn'],
                        'action': turn['action'],
                        'condition': condition,
                        'cache': cache,
                        'mode': mode,
                        'kept_checks': kept_checks,
                        **row,
                    }
                )
    return rows


def summarize(rows):
    dimensions = [value for row in rows for value in row['dimension_complete'].values()]
    kept = [value for row in rows for value in row['kept_checks'].values()]
    stale_rows = [row for row in rows if row['stale_eligible']]

    def fraction(correct, total):
        return {'correct': correct, 'total': total, 'rate': correct / total}

    return {
        'changed': fraction(sum(row['changed_correct'] for row in rows), len(rows)),
        'kept': fraction(sum(kept), len(kept)),
        'four_dim_complete': fraction(sum(dimensions), len(dimensions)),
        'stale_misuse': fraction(
            sum(bool(row['stale_hits']) for row in stale_rows), len(stale_rows)
        ),
        'dimensions': {
            dimension: fraction(
                sum(
                    row['dimension_complete'][dimension]
                    for row in rows
                    if dimension in row['dimension_complete']
                ),
                sum(dimension in row['dimension_complete'] for row in rows),
            )
            for dimension in DIMENSIONS
        },
    }


async def evaluate_sequences(cases):
    rows = []
    for condition in ('replace_fragment', 'incremental_merge'):
        for case in cases:
            rows.extend(await replay_scenario(case, condition))
    return {
        'protocol': 'annotated multi-turn graph-interface replay; paired dimension-merge ablation',
        'stale_definition': 'retired evidence text exposed in the current reader context',
        'case_count': len(cases),
        'turn_count': sum(len(case['turns']) for case in cases),
        'conditions': {
            condition: {
                f'{cache}/{mode}': summarize(
                    [
                        row
                        for row in rows
                        if row['condition'] == condition
                        and row['cache'] == cache
                        and row['mode'] == mode
                    ]
                )
                for cache in ('warm', 'restored')
                for mode in ('full_source', 'evidence_bundle')
            }
            for condition in ('replace_fragment', 'incremental_merge')
        },
        'rows': rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--cases', type=Path, default=Path(__file__).with_name('continuous_update_cases.json')
    )
    parser.add_argument('--output-json', type=Path, required=True)
    args = parser.parse_args()
    result = asyncio.run(evaluate_sequences(json.loads(args.cases.read_text(encoding='utf-8'))))
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(result['conditions'], ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
