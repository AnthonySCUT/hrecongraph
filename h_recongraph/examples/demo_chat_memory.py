"""Minimal H-ReConGraph usage example.

Run this with a configured Graphiti client when Neo4j and OpenAI credentials are available.
"""

from graphiti_core.graphiti import Graphiti
from h_recongraph import HReConGraphMemory


async def build_memory() -> HReConGraphMemory:
    graphiti = Graphiti(uri='bolt://localhost:7687', user='neo4j', password='password')
    return HReConGraphMemory(graphiti)
