from __future__ import annotations

from s1_graphify.extract import CandidateId, SourceLoc
from s1_graphify.graph import GraphDocument, Node, Provenance
from s1_graphify.query import retrieve


def _graph() -> GraphDocument:
    prov = Provenance(extractor="stdlib_regex", judged_by="typesafe/jev-1.13", endpoint="https://example")
    nodes = (
        Node(
            id=CandidateId("symbol:pkg/config.py:parse_config:1"),
            kind="symbol",
            name="parse_config",
            loc=SourceLoc("pkg/config.py", 1, 2),
            confidence=0.8,
            provenance=prov,
        ),
        Node(
            id=CandidateId("symbol:pkg/user.py:load_user:4"),
            kind="symbol",
            name="load_user",
            loc=SourceLoc("pkg/user.py", 4, 6),
            confidence=0.7,
            provenance=prov,
        ),
        Node(
            id=CandidateId("file:pkg/config.py"),
            kind="file",
            name="config.py",
            loc=SourceLoc("pkg/config.py", 1, 2),
            confidence=0.9,
            provenance=prov,
        ),
    )
    return GraphDocument(
        commit="abc",
        endpoint="https://example",
        model="typesafe/jev-1.13",
        status="complete",
        nodes=nodes,
        edges=(),
    )


def test_query_deterministic():
    graph = _graph()
    a = retrieve(graph, "where is parse_config defined?")
    b = retrieve(graph, "where is parse_config defined?")
    assert a == b
    assert a[0].name == "parse_config"
    assert [h.node_id for h in a] == [h.node_id for h in b]
