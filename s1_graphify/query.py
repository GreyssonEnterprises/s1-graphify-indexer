from __future__ import annotations

import re
from pathlib import Path

from s1_graphify.budget import Budget
from s1_graphify.engine import DecisionsEngine, Hit, RankedHits
from s1_graphify.graph import GraphDocument

_TOKEN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def retrieve(graph: GraphDocument, question: str, *, k: int = 20) -> tuple[Hit, ...]:
    tokens = [t.lower() for t in _TOKEN.findall(question)]
    hits: list[Hit] = []
    for node in graph.nodes:
        score = 0.0
        name = node.name.lower()
        path = node.loc.path.lower()
        for t in tokens:
            if t == name:
                score += 3
            elif t in name:
                score += 2
            if t in path:
                score += 1
        if score:
            hits.append(Hit(node.id, node.name, node.loc, score))
    hits.sort(key=lambda h: (-h.score, h.name, h.loc.path, h.node_id.value))
    return tuple(hits[:k])


def query_graph(
    question: str,
    graph_path: Path,
    *,
    engine: DecisionsEngine | None = None,
    rerank: bool = False,
) -> RankedHits | tuple[Hit, ...]:
    graph = GraphDocument.load(graph_path)
    hits = retrieve(graph, question)
    if rerank and engine is not None:
        return engine.rerank(hits, question, budget=Budget.query())
    return hits
