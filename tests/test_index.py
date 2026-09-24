from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from s1_graphify.budget import Budget, BudgetExceeded
from s1_graphify.config import EngineConfig
from s1_graphify.engine import DecisionsEngine
from s1_graphify.extract import SourceText, extract_candidates
from s1_graphify.index import CHECKPOINT_EVERY, benchmark_manifest, index_repo
from s1_graphify.report import Checkpoint
from s1_graphify.stream import stream_sources
from tests.conftest import ScriptedTransport, TOY


def _engine(transport) -> DecisionsEngine:
    return DecisionsEngine(
        config=EngineConfig(max_retries=3),
        api_key="test-key",
        transport=transport,
        sleeper=transport.sleeper,
    )


def test_extract_does_not_invent_files():
    text = (TOY / "pkg" / "user.py").read_text()
    source = SourceText(path="pkg/user.py", text=text, size_bytes=len(text.encode()), content_sha256="")
    found = extract_candidates(source)
    assert all(c.loc.path == "pkg/user.py" for c in found.items)
    file_paths = {c.loc.path for c in found.items if c.kind == "file"}
    assert file_paths == {"pkg/user.py"}
    names = {c.name for c in found.items}
    assert "load_user" in names
    assert "parse_config" in names
    assert not any(c.kind == "file" and "config.py" in c.loc.path for c in found.items)
    assert not any(c.kind == "symbol" and c.name == "parse_config" for c in found.items)


def test_stream_includes_typescript(tmp_path):
    (tmp_path / "cli.ts").write_text("export function createCli() {}\n")
    (tmp_path / "x.py").write_text("def x():\n    return 1\n")
    paths = {s.path for s in stream_sources(tmp_path, Budget.default())}
    assert "cli.ts" in paths
    assert "x.py" in paths


def test_index_mixed_languages_is_integral(tmp_path):
    from s1_graphify.graph import GraphDocument, integrity_problems

    (tmp_path / "cli.ts").write_text(
        "export function createCli() {\n  parseArgs();\n}\n"
    )
    (tmp_path / "x.py").write_text("def x():\n    return 1\n")
    out = tmp_path / "out"
    index_repo(tmp_path, out, _engine(ScriptedTransport()), Budget.default())
    doc = GraphDocument.load(out / "graph.json")
    paths = {n.loc.path for n in doc.nodes}
    assert "cli.ts" in paths
    assert "x.py" in paths
    assert any(n.name == "createCli" for n in doc.nodes)
    assert integrity_problems(doc) == ()


def test_stream_is_bounded_not_corpus(tmp_path, monkeypatch):
    for i in range(20):
        (tmp_path / f"f{i}.py").write_text(f"x{i}=1\n")
    reads: list[str] = []
    orig = Path.read_bytes

    def spy(self, *a, **k):
        reads.append(str(self))
        return orig(self, *a, **k)

    monkeypatch.setattr(Path, "read_bytes", spy)
    it = stream_sources(tmp_path, Budget.default())
    first = next(it)
    assert first.text
    assert len(reads) == 1
    rest = list(it)
    assert len(rest) == 19
    assert len(reads) == 20


def test_walk_fallback_does_not_buffer_all_paths(tmp_path, monkeypatch):
    import s1_graphify.stream as stream_mod

    d1 = tmp_path / "a"
    d2 = tmp_path / "z"
    d1.mkdir()
    d2.mkdir()
    (d1 / "a.py").write_text("a=1\n")
    (d2 / "z.py").write_text("z=1\n")
    events: list[tuple[str, str]] = []
    orig_walk = stream_mod.os.walk

    def tracking_walk(*args, **kwargs):
        for item in orig_walk(*args, **kwargs):
            events.append(("walk", item[0]))
            yield item

    orig_read = Path.read_bytes

    def spy_read(self, *a, **k):
        events.append(("read", str(self)))
        return orig_read(self, *a, **k)

    monkeypatch.setattr(stream_mod.os, "walk", tracking_walk)
    monkeypatch.setattr(Path, "read_bytes", spy_read)
    first = next(iter(stream_sources(tmp_path, Budget.default())))
    assert first.path == "a/a.py"
    walked = {Path(p).name for kind, p in events if kind == "walk"}
    assert "a" in walked
    assert "z" not in walked


def test_skip_dirs_use_relative_parts(tmp_path):
    repo = tmp_path / "build" / "proj"
    repo.mkdir(parents=True)
    (repo / "ok.py").write_text("def ok():\n    return 1\n")
    (repo / "build").mkdir()
    (repo / "build" / "gen.py").write_text("def gen():\n    return 2\n")
    out = tmp_path / "out"
    index_repo(repo, out, _engine(ScriptedTransport()), Budget.default())
    data = json.loads((out / "graph.json").read_text())
    names = {n["name"] for n in data["nodes"]}
    assert "ok" in names
    assert "gen" not in names


def test_empty_repo_does_not_publish(tmp_path):
    out = tmp_path / "out"
    index_repo(tmp_path, out, _engine(ScriptedTransport()), Budget.default())
    assert not (out / "graph.json").exists()
    report = (out / "INDEX_REPORT.md").read_text().lower()
    assert "abort" in report


def test_file_cap_aborts(tmp_path):
    (tmp_path / "big.py").write_bytes(b"x" * 500)
    budget = Budget.default()
    budget.max_file_bytes = 100
    with pytest.raises(BudgetExceeded) as ei:
        list(stream_sources(tmp_path, budget))
    assert ei.value.kind == "file_size"


def test_chunk_cap_aborts(tmp_path):
    (tmp_path / "wide.py").write_text("a" * 50)
    budget = Budget.default()
    budget.max_chunk_chars = 10
    budget.max_chunks = 2
    with pytest.raises(BudgetExceeded) as ei:
        list(stream_sources(tmp_path, budget))
    assert ei.value.kind in {"chunk", "chunks"}


def test_rss_abort(tmp_path):
    (tmp_path / "a.py").write_text("def a():\n    return 1\n")
    out = tmp_path / "out"
    out.mkdir()
    t = ScriptedTransport()
    budget = Budget.default()
    budget.max_rss_bytes = 100
    index_repo(tmp_path, out, _engine(t), budget, rss_reader=lambda: 10_000)
    assert not (out / "graph.json").exists()
    assert (out / "INDEX_REPORT.md").exists()
    assert (out / "checkpoint.json").exists()


def test_atomic_publish_success(tmp_path):
    out = tmp_path / "out"
    t = ScriptedTransport()
    index_repo(TOY, out, _engine(t), Budget.default())
    graph_path = out / "graph.json"
    assert graph_path.exists()
    data = json.loads(graph_path.read_text())
    assert data["status"] == "complete"
    assert data["nodes"]
    names = {n["name"] for n in data["nodes"]}
    assert "parse_config" in names
    assert "load_user" in names


def test_abort_writes_report_and_checkpoint_not_graph(tmp_path):
    (tmp_path / "a.py").write_text("def a():\n    return 1\n")
    out = tmp_path / "out"
    out.mkdir()
    t = ScriptedTransport()
    budget = Budget.default()
    budget.max_rss_bytes = 1
    index_repo(tmp_path, out, _engine(t), budget, rss_reader=lambda: 999)
    assert not (out / "graph.json").exists()
    assert (out / "INDEX_REPORT.md").exists()
    assert (out / "checkpoint.json").exists()
    report = (out / "INDEX_REPORT.md").read_text()
    assert "abort" in report.lower() or "rss" in report.lower()


def test_checkpoint_resume_skips_done_paths(tmp_path):
    (tmp_path / "one.py").write_text("def one():\n    return 1\n")
    (tmp_path / "two.py").write_text("def two():\n    return 2\n")
    out = tmp_path / "out"
    judged: list[str] = []

    class Counting(ScriptedTransport):
        def __call__(self, url, data, headers, timeout):
            body = json.loads(data.decode() if isinstance(data, bytes) else data)
            state = body.get("state")
            blob = json.dumps(state)
            if "def two" in blob or "two.py" in blob:
                raise TimeoutError("stop on two")
            judged.append(blob)
            return super().__call__(url, data, headers, timeout)

    t = Counting()
    budget = Budget.default()
    index_repo(tmp_path, out, _engine(t), budget)
    assert not (out / "graph.json").exists()
    first_calls = len(t.calls)

    t2 = ScriptedTransport()
    index_repo(tmp_path, out, _engine(t2), Budget.default())
    assert (out / "graph.json").exists()
    resumed = json.dumps([c["body"] for c in t2.calls])
    assert "one.py" not in resumed
    assert "two.py" in resumed
    assert first_calls >= 1


def test_checkpoint_repo_mismatch_does_not_resume_or_clobber(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "one.py").write_text("def one():\n    return 1\n")
    other = tmp_path / "other-repo"
    other.mkdir()
    out = tmp_path / "out"
    out.mkdir()
    payload = {
        "repo": str(other),
        "commit": "unknown",
        "facts": [
            {
                "path": "one.py",
                "candidates": [],
                "judgments": {"items": [], "usage": {}},
            }
        ],
        "file_hashes": {"one.py": "deadbeef"},
    }
    ckpt = out / "checkpoint.json"
    original = json.dumps(payload)
    ckpt.write_text(original)
    t = ScriptedTransport()
    ok = index_repo(repo, out, _engine(t), Budget.default())
    assert ok is False
    assert t.calls == []
    assert ckpt.read_text() == original
    report = (out / "INDEX_REPORT.md").read_text()
    assert "identity" in report
    assert "fresh output directory" in report.lower()


def test_checkpoint_resume_rejudges_changed_file(tmp_path):
    (tmp_path / "one.py").write_text("def one():\n    return 1\n")
    (tmp_path / "two.py").write_text("def two():\n    return 2\n")
    out = tmp_path / "out"

    class Counting(ScriptedTransport):
        def __call__(self, url, data, headers, timeout):
            body = json.loads(data.decode() if isinstance(data, bytes) else data)
            blob = json.dumps(body.get("state"))
            if "def two" in blob or "two.py" in blob:
                raise TimeoutError("stop on two")
            return super().__call__(url, data, headers, timeout)

    index_repo(tmp_path, out, _engine(Counting()), Budget.default())
    assert (out / "checkpoint.json").exists()
    (tmp_path / "one.py").write_text("def one():\n    return 99\n")

    t2 = ScriptedTransport()
    index_repo(tmp_path, out, _engine(t2), Budget.default())
    resumed = json.dumps([c["body"] for c in t2.calls])
    assert "one.py" in resumed
    assert "return 99" in resumed
    assert "two.py" in resumed


def test_report_write_is_atomic(tmp_path, monkeypatch):
    import os

    from s1_graphify.report import IndexReport

    replaced: list[tuple[str, str]] = []
    fsynced: list[int] = []
    real_replace = os.replace
    real_fsync = os.fsync

    def spy_replace(src, dst):
        replaced.append((Path(src).name, Path(dst).name))
        return real_replace(src, dst)

    def spy_fsync(fd):
        fsynced.append(fd)
        return real_fsync(fd)

    monkeypatch.setattr(os, "replace", spy_replace)
    monkeypatch.setattr(os, "fsync", spy_fsync)
    path = tmp_path / "INDEX_REPORT.md"
    IndexReport(
        wall_s=1.0,
        files=1,
        chunks=1,
        requests=1,
        latency_ms_p50=1.0,
        latency_ms_p95=1.0,
        input_tokens=10,
        cost=None,
        peak_rss_bytes=1,
        retries=0,
        abort=None,
    ).write_md(path)
    assert replaced == [("INDEX_REPORT.md.tmp", "INDEX_REPORT.md")]
    assert fsynced
    assert path.exists()
    assert not (tmp_path / "INDEX_REPORT.md.tmp").exists()
    assert "status: complete" in path.read_text()


def test_report_accounts_requests_retries_usage(tmp_path):
    out = tmp_path / "out"
    t = ScriptedTransport(script=[429])
    index_repo(TOY, out, _engine(t), Budget.default())
    report = (out / "INDEX_REPORT.md").read_text()
    assert "312" in report or "input" in report.lower()
    assert "retries" in report.lower()
    assert "0.000013" in report or "cost" in report.lower()
    assert "rss" in report.lower()
    assert "request" in report.lower()


def test_rss_abort_during_graph_assembly(tmp_path):
    import traceback

    (tmp_path / "a.py").write_text("def a():\n    return 1\n")
    out = tmp_path / "out"

    def reader():
        if "from_facts" in "".join(traceback.format_stack()):
            return 10_000
        return 10

    budget = Budget.default()
    budget.max_rss_bytes = 100
    index_repo(tmp_path, out, _engine(ScriptedTransport()), budget, rss_reader=reader)
    assert not (out / "graph.json").exists()
    report = (out / "INDEX_REPORT.md").read_text().lower()
    assert "abort" in report
    assert "rss" in report


def test_from_facts_rss_check_aborts():
    from s1_graphify.engine import CandidateJudgment, Judgments, ZERO_USAGE
    from s1_graphify.extract import Candidate, CandidateId, CandidateSet, SourceLoc
    from s1_graphify.graph import FileFacts, GraphDocument

    cid = CandidateId("symbol:a.py:a:1")
    cand = Candidate(cid, "symbol", "a", SourceLoc("a.py", 1, 1), {})
    judged = CandidateJudgment(cid, True, 0.9, 2.0, 0.8, "symbol")

    def boom():
        raise BudgetExceeded("rss", "assembly")

    with pytest.raises(BudgetExceeded) as ei:
        GraphDocument.from_facts(
            [FileFacts("a.py", CandidateSet("a.py", (cand,)), Judgments((judged,), ZERO_USAGE))],
            commit="abc",
            endpoint="https://example",
            model="typesafe/jev-1.13",
            rss_check=boom,
        )
    assert ei.value.kind == "rss"


def test_role_cannot_reclassify_extractor_kind():
    from s1_graphify.engine import CandidateJudgment, Judgments, ZERO_USAGE
    from s1_graphify.extract import Candidate, CandidateId, CandidateSet, SourceLoc
    from s1_graphify.graph import FileFacts, GraphDocument

    cid = CandidateId("call:pkg/user.py:parse_config:4")
    cand = Candidate(
        cid,
        "call",
        "parse_config",
        SourceLoc("pkg/user.py", 4, 4),
        {"src": "symbol:pkg/user.py:load_user:3", "dst": "name:parse_config"},
    )
    judged = CandidateJudgment(cid, True, 0.9, 2.0, 0.8, "symbol")
    doc = GraphDocument.from_facts(
        [FileFacts("pkg/user.py", CandidateSet("pkg/user.py", (cand,)), Judgments((judged,), ZERO_USAGE))],
        commit="abc",
        endpoint="https://example",
        model="typesafe/jev-1.13",
    )
    assert not any(n.id == cid for n in doc.nodes)
    assert len(doc.edges) == 1
    assert doc.edges[0].kind == "call"


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)


def test_source_symlink_outside_repo_is_not_read(tmp_path):
    marker = "OUTSIDE_MARKER_s1_issue7"
    outside = tmp_path / "outside" / "secret.txt"
    outside.parent.mkdir()
    outside.write_text(f"# {marker}\n")
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "ok.py").write_text("def ok():\n    return 1\n")
    (repo / "leak.py").symlink_to(outside)

    def yielded_text() -> str:
        sources = list(stream_sources(repo, Budget.default()))
        paths = {s.path for s in sources}
        blob = "\n".join(s.text for s in sources)
        assert marker not in blob
        assert paths == {"ok.py"}
        return blob

    yielded_text()

    def transport_body(repo_out: Path) -> bytes:
        transport = ScriptedTransport()
        index_repo(repo, repo_out, _engine(transport), Budget.default())
        chunks: list[bytes] = []
        for call in transport.calls:
            data = call["data"]
            chunks.append(data if isinstance(data, (bytes, bytearray)) else str(data).encode())
        return b"".join(chunks)

    assert marker.encode() not in transport_body(tmp_path / "out")

    _git(repo, "init")
    _git(repo, "add", "--", "leak.py", "ok.py")
    assert marker not in yielded_text()
    assert marker.encode() not in transport_body(tmp_path / "out-git")


def test_benchmark_requires_every_expected_name(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "a.py").write_text("def parse_config():\n    return 1\n\ndef load_user():\n    return 2\n")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({
        "repo": str(repo),
        "questions": [
            {
                "question": "parse_config load_user",
                "expect_names": ["parse_config", "load_user"],
            },
            {
                "question": "parse_config load_user",
                "expect_names": ["parse_config", "missing_symbol"],
            },
        ],
    }))
    benchmark_manifest(manifest, _engine(ScriptedTransport()), Budget.default())
    data = json.loads((tmp_path / "benchmark_metrics.json").read_text())
    assert data["match"] == "all"
    assert data["questions"][0]["hit"] is True
    assert data["questions"][1]["hit"] is False


def test_benchmark_abort_does_not_load_graph(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    repo = tmp_path / "empty"
    repo.mkdir()
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"repo": str(repo), "questions": [{"question": "x", "expect_names": ["y"]}]}))
    ok = benchmark_manifest(manifest, _engine(ScriptedTransport()), Budget.default())
    assert ok is False
    assert not (tmp_path / "benchmark_metrics.json").exists()


def test_checkpoint_rewrites_on_interval_not_every_file(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    count = CHECKPOINT_EVERY + 5
    for i in range(count):
        (repo / f"f{i}.py").write_text(f"def f{i}():\n    return {i}\n")
    saves = {"n": 0}
    real = Checkpoint.save_atomic

    def spy(self, path):
        saves["n"] += 1
        return real(self, path)

    monkeypatch.setattr(Checkpoint, "save_atomic", spy)
    assert index_repo(repo, tmp_path / "out", _engine(ScriptedTransport()), Budget.default())
    assert saves["n"] == 1
    assert saves["n"] < count


def test_interrupt_saves_unsaved_judgments(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    for i in range(12):
        (repo / f"f{i:02}.py").write_text(f"def f{i}():\n    return {i}\n")

    class Interrupting(ScriptedTransport):
        def __call__(self, url, data, headers, timeout):
            if b"f10.py" in data:
                raise KeyboardInterrupt
            return super().__call__(url, data, headers, timeout)

    out = tmp_path / "out"
    with pytest.raises(KeyboardInterrupt):
        index_repo(repo, out, _engine(Interrupting()), Budget.default())
    saved = json.loads((out / "checkpoint.json").read_text())
    assert [f["path"] for f in saved["facts"]] == [f"f{i:02}.py" for i in range(10)]


def test_interrupt_before_new_judgments_keeps_checkpoint(tmp_path, monkeypatch):
    import s1_graphify.index as index_mod

    (tmp_path / "one.py").write_text("def one():\n    return 1\n")
    (tmp_path / "two.py").write_text("def two():\n    return 2\n")
    out = tmp_path / "out"

    class StopOnTwo(ScriptedTransport):
        def __call__(self, url, data, headers, timeout):
            if b"two.py" in data:
                raise TimeoutError("stop on two")
            return super().__call__(url, data, headers, timeout)

    index_repo(tmp_path, out, _engine(StopOnTwo()), Budget.default())
    ckpt = out / "checkpoint.json"
    original = ckpt.read_bytes()
    assert [f["path"] for f in json.loads(original)["facts"]] == ["one.py"]

    def interrupted(repo, ckpt):
        raise KeyboardInterrupt

    monkeypatch.setattr(index_mod, "_resumable", interrupted)
    with pytest.raises(KeyboardInterrupt):
        index_repo(tmp_path, out, _engine(ScriptedTransport()), Budget.default())
    assert ckpt.read_bytes() == original
