from __future__ import annotations

from s1_graphify.engine import CandidateJudgment, Judgments, ZERO_USAGE
from s1_graphify.extract import Candidate, CandidateId, CandidateSet, SourceLoc, SourceText, extract_candidates
from s1_graphify.graph import Edge, FileFacts, GraphDocument, Node, Provenance, integrity_problems


def _facts(candidates: tuple[Candidate, ...], keep: tuple[CandidateId, ...]) -> list[FileFacts]:
    items = tuple(
        CandidateJudgment(cid, True, 0.9, 1.0, 0.8, None) for cid in keep
    )
    path = candidates[0].loc.path
    return [FileFacts(path, CandidateSet(path, candidates), Judgments(items, ZERO_USAGE))]


def test_integrity_problems_reports_missing_and_duplicate_ids():
    file_n_id = CandidateId("file:a.py")
    call_id = CandidateId("call:a.py:foo:1:1")
    loc = SourceLoc("a.py", 1, 1)
    prov = Provenance("stdlib_regex", "typesafe/jev-1.13", "https://example")
    broken = GraphDocument(
        commit="abc",
        endpoint="https://example",
        model="typesafe/jev-1.13",
        status="complete",
        nodes=(Node(file_n_id, "file", "a.py", loc, 0.8, prov),),
        edges=(
            Edge(call_id, "call", file_n_id, CandidateId("name:foo"), loc, 0.8, prov),
            Edge(call_id, "call", file_n_id, CandidateId("name:foo"), loc, 0.8, prov),
        ),
    )
    problems = integrity_problems(broken)
    assert "duplicate_edge_ids" in problems
    assert "missing_dst" in problems


def test_from_facts_mints_endpoint_nodes_and_is_integral():
    loc = SourceLoc("a.py", 1, 1)
    file_id = CandidateId("file:a.py")
    call_id = CandidateId("call:a.py:foo:1:1")
    doc = GraphDocument.from_facts(
        _facts(
            (
                Candidate(file_id, "file", "a.py", loc, {}),
                Candidate(
                    call_id, "call", "foo", loc,
                    {"src": "file:a.py", "dst": "name:foo"},
                ),
            ),
            (file_id, call_id),
        ),
        commit="abc",
        endpoint="https://example",
        model="typesafe/jev-1.13",
    )
    ids = {n.id.value for n in doc.nodes}
    assert "file:a.py" in ids
    assert "name:foo" in ids
    assert integrity_problems(doc) == ()
    assert {e.src.value for e in doc.edges} <= ids
    assert {e.dst.value for e in doc.edges} <= ids


def test_from_facts_dedupes_identical_edge_ids():
    loc = SourceLoc("a.py", 1, 1)
    file_id = CandidateId("file:a.py")
    call_id = CandidateId("call:a.py:foo:1:1")
    call = Candidate(
        call_id, "call", "foo", loc, {"src": "file:a.py", "dst": "name:foo"}
    )
    doc = GraphDocument.from_facts(
        _facts(
            (Candidate(file_id, "file", "a.py", loc, {}), call, call),
            (file_id, call_id),
        ),
        commit="abc",
        endpoint="https://example",
        model="typesafe/jev-1.13",
    )
    edge_ids = [e.id.value for e in doc.edges]
    assert len(edge_ids) == len(set(edge_ids))
    assert integrity_problems(doc) == ()


def test_extract_repeated_calls_mint_unique_ids():
    source = SourceText("a.py", "foo()\nfoo(); foo()\n", 20, "")
    cset = extract_candidates(source)
    call_ids = [c.id.value for c in cset.items if c.kind == "call"]
    rel_ids = [c.id.value for c in cset.items if c.kind == "relation"]
    assert call_ids == [
        "call:a.py:foo:1:1",
        "call:a.py:foo:2:1",
        "call:a.py:foo:2:2",
    ]
    assert len(rel_ids) == len(set(rel_ids)) == 3


def test_extract_typescript_symbols():
    text = (
        "export function createCli(argv: string[]) {\n"
        "  parseArgs(argv);\n"
        "}\n"
    )
    cset = extract_candidates(SourceText("src/cli.ts", text, len(text), ""))
    names = {c.name for c in cset.items}
    assert "createCli" in names
    assert any(c.kind == "file" and c.loc.path == "src/cli.ts" for c in cset.items)
    assert any(c.kind == "symbol" and c.name == "createCli" for c in cset.items)
    assert any(c.kind == "call" and c.name == "parseArgs" for c in cset.items)
