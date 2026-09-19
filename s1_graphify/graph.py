from __future__ import annotations

import json
import os
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from s1_graphify.engine import CandidateJudgment, Judgments, Usage, ZERO_USAGE
from s1_graphify.extract import Candidate, CandidateId, CandidateSet, SourceLoc

NODE_KINDS = frozenset({"file", "symbol"})
EDGE_KINDS = frozenset({"import", "call", "relation"})


@dataclass(frozen=True)
class Provenance:
    extractor: Literal["stdlib_regex"]
    judged_by: str
    endpoint: str = ""


@dataclass(frozen=True)
class Node:
    id: CandidateId
    kind: Literal["file", "symbol"]
    name: str
    loc: SourceLoc
    confidence: float
    provenance: Provenance


@dataclass(frozen=True)
class Edge:
    id: CandidateId
    kind: Literal["import", "call", "relation"]
    src: CandidateId
    dst: CandidateId
    loc: SourceLoc
    confidence: float
    provenance: Provenance


@dataclass(frozen=True)
class FileFacts:
    path: str
    candidates: CandidateSet
    judgments: Judgments

    def to_dict(self) -> dict:
        return {
            "path": self.path,
            "candidates": [_candidate_dict(c) for c in self.candidates.items],
            "judgments": {
                "items": [_judgment_dict(j) for j in self.judgments.items],
                "usage": _usage_dict(self.judgments.usage),
            },
        }

    @staticmethod
    def from_dict(data: dict) -> FileFacts:
        items = tuple(_candidate_from(c) for c in data["candidates"])
        jitems = tuple(_judgment_from(j) for j in data["judgments"]["items"])
        usage = _usage_from(data["judgments"].get("usage") or {})
        return FileFacts(
            data["path"],
            CandidateSet(data["path"], items),
            Judgments(jitems, usage),
        )


@dataclass(frozen=True)
class GraphDocument:
    commit: str
    endpoint: str
    model: str
    status: Literal["complete"]
    nodes: tuple[Node, ...]
    edges: tuple[Edge, ...]

    @staticmethod
    def from_facts(
        facts: Sequence[FileFacts],
        *,
        commit: str,
        endpoint: str,
        model: str,
        rss_check: Callable[[], None] | None = None,
    ) -> GraphDocument:
        prov = Provenance("stdlib_regex", model, endpoint)
        nodes: dict[str, Node] = {}
        edges: list[Edge] = []
        seen_edge_ids: set[str] = set()

        def add_node(node: Node) -> None:
            if node.id.value not in nodes:
                nodes[node.id.value] = node

        def ensure_endpoint(cid: CandidateId, loc: SourceLoc, confidence: float) -> None:
            if cid.value in nodes:
                return
            if cid.value.startswith("file:"):
                kind: Literal["file", "symbol"] = "file"
            else:
                kind = "symbol"
            add_node(Node(cid, kind, _endpoint_name(cid), loc, confidence, prov))

        if rss_check is not None:
            rss_check()
        for fact in facts:
            if rss_check is not None:
                rss_check()
            jmap = fact.judgments.by_id()
            for c in fact.candidates.items:
                j = jmap.get(c.id)
                if j is None or not j.keep:
                    continue
                if c.kind in NODE_KINDS:
                    add_node(
                        Node(c.id, c.kind, c.name, c.loc, j.confidence, prov)  # type: ignore[arg-type]
                    )
        for fact in facts:
            jmap = fact.judgments.by_id()
            for c in fact.candidates.items:
                j = jmap.get(c.id)
                if j is None or not j.keep:
                    continue
                if c.kind not in EDGE_KINDS:
                    continue
                if c.id.value in seen_edge_ids:
                    continue
                seen_edge_ids.add(c.id.value)
                src = CandidateId(c.extra.get("src") or c.id.value)
                dst = CandidateId(c.extra.get("dst") or c.id.value)
                ensure_endpoint(src, c.loc, j.confidence)
                ensure_endpoint(dst, c.loc, j.confidence)
                edges.append(
                    Edge(c.id, c.kind, src, dst, c.loc, j.confidence, prov)  # type: ignore[arg-type]
                )
        doc = GraphDocument(
            commit=commit,
            endpoint=endpoint,
            model=model,
            status="complete",
            nodes=tuple(nodes.values()),
            edges=tuple(edges),
        )
        problems = integrity_problems(doc)
        if problems:
            raise ValueError(f"graph integrity: {problems}")
        return doc

    @staticmethod
    def load(path: Path) -> GraphDocument:
        data = json.loads(Path(path).read_text())
        nodes = tuple(_node_from(n, data.get("endpoint") or "") for n in data.get("nodes") or [])
        edges = tuple(_edge_from(e, data.get("endpoint") or "") for e in data.get("edges") or [])
        return GraphDocument(
            commit=data.get("commit") or "unknown",
            endpoint=data.get("endpoint") or "",
            model=data.get("model") or "",
            status="complete",
            nodes=nodes,
            edges=edges,
        )

    def to_dict(self) -> dict:
        return {
            "status": "complete",
            "commit": self.commit,
            "endpoint": self.endpoint,
            "model": self.model,
            "nodes": [_node_dict(n) for n in self.nodes],
            "edges": [_edge_dict(e) for e in self.edges],
        }

    def publish_atomic(self, out_dir: Path) -> None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        final = out_dir / "graph.json"
        tmp = out_dir / "graph.json.tmp"
        payload = json.dumps(self.to_dict(), indent=2)
        with open(tmp, "w", encoding="utf-8") as fh:
            fh.write(payload)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, final)


def _endpoint_name(cid: CandidateId) -> str:
    val = cid.value
    if val.startswith("name:"):
        return val[len("name:"):]
    if val.startswith("file:"):
        return val.split(":", 1)[-1].rsplit("/", 1)[-1] or val
    parts = val.split(":")
    if len(parts) >= 3:
        return parts[2]
    return parts[-1]


def integrity_problems(doc: GraphDocument) -> tuple[str, ...]:
    problems: list[str] = []
    node_ids = [n.id.value for n in doc.nodes]
    if len(node_ids) != len(set(node_ids)):
        problems.append("duplicate_node_ids")
    edge_ids = [e.id.value for e in doc.edges]
    if len(edge_ids) != len(set(edge_ids)):
        problems.append("duplicate_edge_ids")
    nodes = set(node_ids)
    if any(e.src.value not in nodes for e in doc.edges):
        problems.append("missing_src")
    if any(e.dst.value not in nodes for e in doc.edges):
        problems.append("missing_dst")
    return tuple(problems)


def _node_dict(n: Node) -> dict:
    return {
        "id": n.id.value,
        "kind": n.kind,
        "name": n.name,
        "path": n.loc.path,
        "start_line": n.loc.start_line,
        "end_line": n.loc.end_line,
        "confidence": n.confidence,
        "provenance": {
            "extractor": n.provenance.extractor,
            "judged_by": n.provenance.judged_by,
        },
    }


def _edge_dict(e: Edge) -> dict:
    return {
        "id": e.id.value,
        "kind": e.kind,
        "src": e.src.value,
        "dst": e.dst.value,
        "path": e.loc.path,
        "start_line": e.loc.start_line,
        "end_line": e.loc.end_line,
        "confidence": e.confidence,
        "provenance": {
            "extractor": e.provenance.extractor,
            "judged_by": e.provenance.judged_by,
        },
    }


def _node_from(n: dict, endpoint: str) -> Node:
    prov = n.get("provenance") or {}
    return Node(
        id=CandidateId(n["id"]),
        kind=n["kind"],
        name=n["name"],
        loc=SourceLoc(n["path"], int(n["start_line"]), int(n["end_line"])),
        confidence=float(n.get("confidence") or 0),
        provenance=Provenance(
            "stdlib_regex",
            prov.get("judged_by") or "",
            endpoint,
        ),
    )


def _edge_from(e: dict, endpoint: str) -> Edge:
    prov = e.get("provenance") or {}
    return Edge(
        id=CandidateId(e["id"]),
        kind=e["kind"],
        src=CandidateId(e["src"]),
        dst=CandidateId(e["dst"]),
        loc=SourceLoc(e["path"], int(e["start_line"]), int(e["end_line"])),
        confidence=float(e.get("confidence") or 0),
        provenance=Provenance(
            "stdlib_regex",
            prov.get("judged_by") or "",
            endpoint,
        ),
    )


def _candidate_dict(c: Candidate) -> dict:
    return {
        "id": c.id.value,
        "kind": c.kind,
        "name": c.name,
        "path": c.loc.path,
        "start_line": c.loc.start_line,
        "end_line": c.loc.end_line,
        "extra": dict(c.extra),
    }


def _candidate_from(c: dict) -> Candidate:
    return Candidate(
        CandidateId(c["id"]),
        c["kind"],
        c["name"],
        SourceLoc(c["path"], int(c["start_line"]), int(c["end_line"])),
        c.get("extra") or {},
    )


def _judgment_dict(j: CandidateJudgment) -> dict:
    return {
        "id": j.candidate_id.value,
        "keep": j.keep,
        "noul": j.noul,
        "salience": j.salience,
        "confidence": j.confidence,
        "role": j.role,
    }


def _judgment_from(j: dict) -> CandidateJudgment:
    return CandidateJudgment(
        CandidateId(j["id"]),
        bool(j["keep"]),
        float(j["noul"]),
        float(j["salience"]),
        float(j["confidence"]),
        j.get("role"),
    )


def _usage_dict(u: Usage) -> dict:
    return {
        "input_tokens": u.input_tokens,
        "output_tokens": u.output_tokens,
        "cost": u.cost,
        "requests": u.requests,
        "retries": u.retries,
        "latencies_ms": list(u.latencies_ms),
    }


def _usage_from(u: dict) -> Usage:
    if not u:
        return ZERO_USAGE
    cost = u.get("cost")
    return Usage(
        int(u.get("input_tokens") or 0),
        int(u.get("output_tokens") or 0),
        float(cost) if cost is not None else None,
        int(u.get("requests") or 0),
        int(u.get("retries") or 0),
        tuple(float(x) for x in (u.get("latencies_ms") or ())),
    )
