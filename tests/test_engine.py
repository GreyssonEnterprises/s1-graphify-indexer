from __future__ import annotations

import json

import pytest

from s1_graphify.budget import Budget
from s1_graphify.config import EngineConfig
from s1_graphify.engine import (
    DecisionsEngine,
    EngineHttpError,
    EngineTimeoutError,
    MalformedResponseError,
    MissingKeyError,
)
from s1_graphify.extract import Candidate, CandidateId, CandidateSet, SourceLoc
from tests.conftest import ScriptedTransport


def _one_symbol() -> CandidateSet:
    cid = CandidateId("symbol:pkg/config.py:parse_config:1")
    cand = Candidate(
        id=cid,
        kind="symbol",
        name="parse_config",
        loc=SourceLoc("pkg/config.py", 1, 1),
        extra={},
    )
    return CandidateSet("pkg/config.py", (cand,))


def _engine(transport, **kwargs) -> DecisionsEngine:
    cfg = EngineConfig(max_retries=kwargs.pop("max_retries", 3), timeout_s=5.0)
    return DecisionsEngine(
        config=cfg,
        api_key="test-key",
        transport=transport,
        sleeper=transport.sleeper,
    )


def test_engine_request_response_mapping():
    t = ScriptedTransport()
    engine = _engine(t)
    judgments = engine.judge(_one_symbol(), budget=Budget.default())
    assert len(t.calls) == 1
    req = t.calls[0]["body"]
    assert req["model"] == "typesafe/jev-1.13"
    assert "state" in req
    assert "questions" in req
    assert t.calls[0]["headers"].get("Authorization") == "Bearer test-key"
    cid = CandidateId("symbol:pkg/config.py:parse_config:1")
    item = judgments.by_id()[cid]
    assert item.keep is True
    assert item.noul == pytest.approx(0.92)
    assert item.confidence == pytest.approx(0.8)
    assert judgments.usage.input_tokens == 312
    assert judgments.usage.cost == pytest.approx(0.000013)


def test_engine_drops_invented_ids():
    t = ScriptedTransport(extra_answers={
        "keep_invented-file": {"type": "noul", "noul": 0.99},
        "role_invented-file": {"type": "choice", "choice": "file", "confidence": 0.99},
        "strength_invented-file": {"type": "score", "score": 2.0},
    })
    engine = _engine(t)
    judgments = engine.judge(_one_symbol(), budget=Budget.default())
    ids = {j.candidate_id.value for j in judgments.items}
    assert "invented-file" not in ids
    assert "symbol:pkg/config.py:parse_config:1" in ids


def test_malformed_response():
    class Bad:
        sleeps = []

        def sleeper(self, seconds):
            self.sleeps.append(seconds)

        def __call__(self, url, data, headers, timeout):
            return 200, {"nope": True}

    engine = _engine(Bad())
    with pytest.raises(MalformedResponseError):
        engine.judge(_one_symbol(), budget=Budget.default())


def test_missing_key(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    called = []

    def transport(*a, **k):
        called.append(1)
        raise AssertionError("must not hit the network")

    with pytest.raises(MissingKeyError):
        DecisionsEngine.from_env(transport=transport)
    assert called == []


def test_retry_429_then_success():
    t = ScriptedTransport(script=[429])
    engine = _engine(t)
    judgments = engine.judge(_one_symbol(), budget=Budget.default())
    assert judgments.items
    assert len(t.calls) == 2
    assert t.sleeps
    assert judgments.usage.retries == 1
    assert judgments.usage.requests == 1


def test_retry_5xx_exhausted():
    t = ScriptedTransport(script=[503, 503, 503, 503])
    engine = _engine(t, max_retries=3)
    with pytest.raises(EngineHttpError):
        engine.judge(_one_symbol(), budget=Budget.default())
    assert len(t.calls) == 4
    assert len(t.sleeps) == 3


def test_timeout_retries_then_abort():
    t = ScriptedTransport(script=["timeout", "timeout", "timeout", "timeout"])
    engine = _engine(t, max_retries=3)
    with pytest.raises(EngineTimeoutError):
        engine.judge(_one_symbol(), budget=Budget.default())
    assert len(t.calls) == 4
    assert len(t.sleeps) == 3


def test_state_cap_aborts():
    t = ScriptedTransport()
    engine = _engine(t)
    budget = Budget.default()
    budget.max_state_chars = 1
    from s1_graphify.budget import BudgetExceeded

    with pytest.raises(BudgetExceeded) as ei:
        engine.judge(_one_symbol(), budget=budget)
    assert ei.value.kind == "state"
    assert t.calls == []


def test_state_cap_counts_serialized_payload_not_candidate_lines():
    from s1_graphify.budget import BudgetExceeded
    from s1_graphify.extract import SourceText, extract_candidates

    text = "def parse_config(path):\n    return path\n" + ("# pad\n" * 400)
    source = SourceText("pkg/config.py", text, len(text.encode()))
    cset = extract_candidates(source)
    t = ScriptedTransport()
    budget = Budget.default()
    budget.max_state_chars = 500
    budget.max_chunk_chars = 8000
    with pytest.raises(BudgetExceeded) as ei:
        _engine(t).judge(cset, budget=budget, source=source)
    assert ei.value.kind == "state"
    assert t.calls == []


def test_chunks_are_windows_not_whole_file():
    from s1_graphify.extract import SourceText, extract_candidates

    text = "def parse_config(path):\n    return path\n" + ("# tail\n" * 3000)
    source = SourceText("pkg/config.py", text, len(text.encode()))
    cset = extract_candidates(source)
    t = ScriptedTransport()
    budget = Budget.default()
    budget.max_chunk_chars = 80
    _engine(t).judge(cset, budget=budget, source=source)
    assert t.calls
    chunks = t.calls[0]["body"]["state"]["chunks"]
    blob = json.dumps(chunks)
    assert "parse_config" in blob
    assert blob.count("# tail") < 50
