from __future__ import annotations

import json
from unittest.mock import patch

from s1_graphify.cli import main
from tests.conftest import ScriptedTransport, TOY, answers_for


class _FakeUrlOpen:
    def __init__(self):
        self.calls = 0

    def __call__(self, req, timeout=None):
        self.calls += 1
        body = json.loads(req.data.decode() if isinstance(req.data, bytes) else req.data)
        payload = json.dumps({
            "model": "typesafe/jev-1.13",
            "answers": answers_for(body.get("questions", {})),
            "usage": {"input_tokens": 10, "output_tokens": 1, "cost": 0.000001},
        }).encode()

        class Resp:
            status = 200

            def read(self):
                return payload

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        return Resp()


def test_index_cli_mocked(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    out = tmp_path / "out"
    fake = _FakeUrlOpen()
    with patch("urllib.request.urlopen", fake):
        rc = main(["index", str(TOY), "--out", str(out)])
    assert rc == 0
    assert (out / "graph.json").exists()
    assert fake.calls >= 1


def test_index_abort_names_report_and_checkpoint(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    repo = tmp_path / "empty"
    repo.mkdir()
    out = tmp_path / "out"
    rc = main(["index", str(repo), "--out", str(out)])
    assert rc == 1
    err = capsys.readouterr().err
    assert str(out / "INDEX_REPORT.md") in err
    assert str(out / "checkpoint.json") in err
    assert "abort" in err.lower()
    assert (out / "INDEX_REPORT.md").exists()
    assert (out / "checkpoint.json").exists()


def test_query_cli_mocked(tmp_path, capsys):
    graph = {
        "status": "complete",
        "commit": "abc",
        "endpoint": "https://example",
        "model": "typesafe/jev-1.13",
        "nodes": [{
            "id": "symbol:pkg/config.py:parse_config:1",
            "kind": "symbol",
            "name": "parse_config",
            "path": "pkg/config.py",
            "start_line": 1,
            "end_line": 2,
            "confidence": 0.8,
            "provenance": {
                "extractor": "stdlib_regex",
                "judged_by": "typesafe/jev-1.13",
            },
        }],
        "edges": [],
    }
    path = tmp_path / "graph.json"
    path.write_text(json.dumps(graph))
    rc = main(["query", "where is parse_config defined?", "--graph", str(path)])
    assert rc == 0
    captured = capsys.readouterr()
    assert "parse_config" in captured.out
    assert captured.err == ""


def test_query_cli_empty_prints_no_matches(tmp_path, capsys):
    graph = {
        "status": "complete",
        "commit": "abc",
        "endpoint": "https://example",
        "model": "typesafe/jev-1.13",
        "nodes": [{
            "id": "symbol:pkg/config.py:parse_config:1",
            "kind": "symbol",
            "name": "parse_config",
            "path": "pkg/config.py",
            "start_line": 1,
            "end_line": 2,
            "confidence": 0.8,
            "provenance": {
                "extractor": "stdlib_regex",
                "judged_by": "typesafe/jev-1.13",
            },
        }],
        "edges": [],
    }
    path = tmp_path / "graph.json"
    path.write_text(json.dumps(graph))
    rc = main(["query", "where is missing_symbol defined?", "--graph", str(path)])
    assert rc == 0
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "no matches" in captured.err


def test_benchmark_cli_mocked(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.chdir(tmp_path)
    repo = tmp_path / "toy"
    pkg = repo / "pkg"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("")
    (pkg / "config.py").write_text("def parse_config(path):\n    return path\n")
    (pkg / "user.py").write_text(
        "from pkg.config import parse_config\n\ndef load_user(user_id):\n    return parse_config('u')\n"
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({
        "repo": str(repo),
        "questions": [
            {"question": "where is parse_config defined?", "expect_names": ["parse_config"]},
        ],
    }))
    fake = _FakeUrlOpen()
    with patch("urllib.request.urlopen", fake):
        rc = main(["benchmark", str(manifest)])
    assert rc == 0
    results = tmp_path / "benchmark_metrics.json"
    assert results.exists()
    data = json.loads(results.read_text())
    assert data["questions"]


def test_benchmark_cli_abort_is_nonzero(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.chdir(tmp_path)
    repo = tmp_path / "empty"
    repo.mkdir()
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"repo": str(repo), "questions": []}))
    rc = main(["benchmark", str(manifest)])
    assert rc == 1
    assert not (tmp_path / "benchmark_metrics.json").exists()
