from __future__ import annotations

import io
import json
import urllib.error
from unittest.mock import patch

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
    cfg = EngineConfig(
        max_retries=kwargs.pop("max_retries", 3),
        timeout_s=5.0,
        concurrency=kwargs.pop("concurrency", 1),
    )
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


def _many_symbols(n: int) -> CandidateSet:
    items = []
    for i in range(n):
        cid = CandidateId(f"symbol:pkg/f.py:fn{i}:{i}")
        items.append(
            Candidate(
                id=cid,
                kind="symbol",
                name=f"fn{i}",
                loc=SourceLoc("pkg/f.py", i + 1, i + 1),
                extra={},
            )
        )
    return CandidateSet("pkg/f.py", tuple(items))


def test_concurrency_runs_batches_in_parallel():
    import threading
    import time

    from s1_graphify.engine import _payload_chars, _questions_for, _windows_for

    in_flight = 0
    peak = {"n": 0}
    lock = threading.Lock()

    class Slow(ScriptedTransport):
        def __call__(self, url, data, headers, timeout):
            nonlocal in_flight
            with lock:
                in_flight += 1
                peak["n"] = max(peak["n"], in_flight)
            try:
                time.sleep(0.05)
                with lock:
                    return super().__call__(url, data, headers, timeout)
            finally:
                with lock:
                    in_flight -= 1

    one = _many_symbols(1).items[0]
    one_sz = _payload_chars(
        "typesafe/jev-1.13",
        [one],
        _windows_for([one], None, 8000),
        _questions_for([one]),
    )
    t = Slow()
    budget = Budget.default()
    budget.max_state_chars = one_sz + 80
    engine = _engine(t, concurrency=2)
    engine.judge(_many_symbols(8), budget=budget)
    assert len(t.calls) >= 2
    assert peak["n"] >= 2


def _json_of_size(n: int) -> bytes:
    prefix = b'{"p":"'
    suffix = b'"}'
    pad = n - len(prefix) - len(suffix)
    assert pad >= 0
    return prefix + (b"a" * pad) + suffix


class _StubUrlOpen:
    def __init__(self, payload: bytes, *, status: int = 200, http_error: int | None = None):
        self.payload = payload
        self.status = status
        self.http_error = http_error

    def __call__(self, req, timeout=None):
        body = self.payload
        status = self.status

        class Resp:
            def read(self, amt=-1):
                return body

            def __enter__(self):
                self.status = status
                return self

            def __exit__(self, *a):
                return False

        Resp.status = status

        if self.http_error is not None:
            raise urllib.error.HTTPError(
                "https://example.test/v1/decide",
                self.http_error,
                "Bad Request",
                None,
                io.BytesIO(self.payload),
            )
        return Resp()


def _max_response_bytes() -> int:
    import s1_graphify.engine as engine

    return getattr(engine, "MAX_RESPONSE_BYTES", 1_048_576)


def test_urllib_transport_rejects_oversized_success_body():
    from s1_graphify.engine import urllib_transport

    MAX_RESPONSE_BYTES = _max_response_bytes()

    payload = _json_of_size(MAX_RESPONSE_BYTES + 2)
    assert len(payload) > MAX_RESPONSE_BYTES + 1
    fake = _StubUrlOpen(payload)
    with patch("urllib.request.urlopen", fake):
        with pytest.raises(MalformedResponseError):
            urllib_transport("https://example.test/v1/decide", b"{}", {}, 1.0)


def test_urllib_transport_rejects_oversized_http_error_body():
    from s1_graphify.engine import urllib_transport

    MAX_RESPONSE_BYTES = _max_response_bytes()

    payload = _json_of_size(MAX_RESPONSE_BYTES + 2)
    fake = _StubUrlOpen(payload, http_error=400)
    with patch("urllib.request.urlopen", fake):
        with pytest.raises(MalformedResponseError):
            urllib_transport("https://example.test/v1/decide", b"{}", {}, 1.0)


def test_bounded_reader_rejects_oversized_error_body():
    from s1_graphify.engine import MAX_RESPONSE_BYTES, read_bounded_response

    payload = _json_of_size(MAX_RESPONSE_BYTES + 2)
    with pytest.raises(MalformedResponseError):
        read_bounded_response(io.BytesIO(payload))


def test_urllib_transport_parses_exact_max_json():
    from s1_graphify.engine import urllib_transport

    MAX_RESPONSE_BYTES = _max_response_bytes()

    payload = _json_of_size(MAX_RESPONSE_BYTES)
    assert len(payload) == MAX_RESPONSE_BYTES
    fake = _StubUrlOpen(payload)
    with patch("urllib.request.urlopen", fake):
        status, parsed = urllib_transport("https://example.test/v1/decide", b"{}", {}, 1.0)
    assert status == 200
    assert isinstance(parsed, dict)
    assert parsed["p"].startswith("a")


def test_pack_splits_source_lines_once(monkeypatch):
    import s1_graphify.engine as eng
    from s1_graphify.extract import SourceText, extract_candidates

    text = "\n".join(f"def sym_{i}():\n    return {i}\n" for i in range(30))
    source = SourceText("pkg/dense.py", text, len(text.encode()))
    cset = extract_candidates(source)
    calls = {"n": 0}
    real = eng._source_lines

    def spy(src):
        calls["n"] += 1
        return real(src)

    monkeypatch.setattr(eng, "_source_lines", spy)
    budget = Budget.default()
    budget.max_state_chars = 24_000
    engine = _engine(ScriptedTransport())
    first = engine._pack(cset, budget=Budget.default(), source=source)
    assert calls["n"] == 1
    calls["n"] = 0
    second = engine._pack(cset, budget=budget, source=source)
    assert calls["n"] == 1
    assert first == second


def test_non_json_success_body_is_malformed():
    from s1_graphify.engine import urllib_transport

    fake = _StubUrlOpen(b"not-json")
    with patch("urllib.request.urlopen", fake):
        with pytest.raises(MalformedResponseError, match="not JSON"):
            urllib_transport("https://example.test/v1/decide", b"{}", {}, 1.0)


def test_invalid_numeric_usage_is_malformed():
    class Bad(ScriptedTransport):
        def __call__(self, url, data, headers, timeout):
            status, payload = super().__call__(url, data, headers, timeout)
            payload["usage"]["cost"] = "nope"
            return status, payload

    with pytest.raises(MalformedResponseError):
        _engine(Bad()).judge(_one_symbol(), budget=Budget.default())


def test_invalid_noul_is_malformed():
    class Bad(ScriptedTransport):
        def __call__(self, url, data, headers, timeout):
            status, payload = super().__call__(url, data, headers, timeout)
            for key in payload["answers"]:
                if key.startswith("keep_"):
                    payload["answers"][key]["noul"] = "bad"
            return status, payload

    with pytest.raises(MalformedResponseError):
        _engine(Bad()).judge(_one_symbol(), budget=Budget.default())
