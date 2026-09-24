from __future__ import annotations

import json
import socket
import time
from concurrent.futures import ThreadPoolExecutor
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from s1_graphify.budget import Budget, BudgetExceeded
from s1_graphify.config import EngineConfig, MissingKeyError
from s1_graphify.extract import Candidate, CandidateId, CandidateSet, SourceLoc, SourceText


class MalformedResponseError(Exception):
    pass


class EngineTimeoutError(Exception):
    pass


class EngineHttpError(Exception):
    def __init__(self, status: int) -> None:
        self.status = status
        super().__init__(f"HTTP {status}")


@dataclass(frozen=True)
class Usage:
    input_tokens: int
    output_tokens: int
    cost: float | None
    requests: int
    retries: int
    latencies_ms: tuple[float, ...]

    def merged(self, other: Usage) -> Usage:
        cost: float | None
        if self.cost is None and other.cost is None:
            cost = None
        else:
            cost = (self.cost or 0.0) + (other.cost or 0.0)
        return Usage(
            self.input_tokens + other.input_tokens,
            self.output_tokens + other.output_tokens,
            cost,
            self.requests + other.requests,
            self.retries + other.retries,
            self.latencies_ms + other.latencies_ms,
        )


ZERO_USAGE = Usage(0, 0, None, 0, 0, ())


@dataclass(frozen=True)
class CandidateJudgment:
    candidate_id: CandidateId
    keep: bool
    noul: float
    salience: float
    confidence: float
    role: str | None = None


@dataclass(frozen=True)
class Judgments:
    items: tuple[CandidateJudgment, ...]
    usage: Usage

    def ids(self) -> frozenset[CandidateId]:
        return frozenset(j.candidate_id for j in self.items)

    def by_id(self) -> Mapping[CandidateId, CandidateJudgment]:
        return {j.candidate_id: j for j in self.items}


@dataclass(frozen=True)
class Hit:
    node_id: CandidateId
    name: str
    loc: SourceLoc
    score: float


@dataclass(frozen=True)
class RankedHits:
    hits: tuple[Hit, ...]
    sufficient: bool
    sufficiency_noul: float
    usage: Usage


# Success and error bodies are capped so a bad endpoint cannot allocate past the RSS budget.
# One extra byte is read only to detect overflow.
MAX_RESPONSE_BYTES = 1_048_576


def read_bounded_response(source, limit: int = MAX_RESPONSE_BYTES) -> bytes:
    read = source.read if hasattr(source, "read") else source
    try:
        raw = read(limit + 1)
    except TypeError:
        raw = read()
    if not raw:
        return b""
    if len(raw) > limit:
        raise MalformedResponseError(f"response body exceeds {limit} bytes")
    return raw


def urllib_transport(url: str, data: bytes, headers: dict[str, str], timeout: float):
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = read_bounded_response(resp)
            status = getattr(resp, "status", 200)
            return status, json.loads(raw.decode())
    except urllib.error.HTTPError as e:
        raw = read_bounded_response(e) if e.fp else b""
        try:
            parsed = json.loads(raw.decode()) if raw else {}
        except json.JSONDecodeError:
            parsed = {}
        return e.code, parsed
    except urllib.error.URLError as e:
        reason = e.reason
        if isinstance(reason, (TimeoutError, socket.timeout)):
            raise TimeoutError(str(e)) from e
        if "timed out" in str(e).lower():
            raise TimeoutError(str(e)) from e
        raise


@dataclass
class DecisionsEngine:
    config: EngineConfig
    api_key: str
    transport: Callable[..., Any] | None = None
    sleeper: Callable[[float], None] = field(default=time.sleep)

    def __post_init__(self) -> None:
        if not self.api_key:
            raise MissingKeyError(f"missing {self.config.key_env}")
        if self.transport is None:
            self.transport = urllib_transport

    @staticmethod
    def from_env(
        config: EngineConfig | None = None,
        transport: Callable[..., Any] | None = None,
        sleeper: Callable[[float], None] | None = None,
    ) -> DecisionsEngine:
        import os

        config = config or EngineConfig.from_env()
        key = os.environ.get(config.key_env, "")
        if not key:
            raise MissingKeyError(f"missing {config.key_env}")
        return DecisionsEngine(
            config=config,
            api_key=key,
            transport=transport,
            sleeper=sleeper or time.sleep,
        )

    def judge(
        self,
        candidates: CandidateSet,
        *,
        budget: Budget,
        source: SourceText | None = None,
    ) -> Judgments:
        batches = self._pack(candidates, budget=budget, source=source)
        if not batches:
            return Judgments((), ZERO_USAGE)

        def run(
            packed: tuple[tuple[Candidate, ...], list, dict],
        ) -> tuple[tuple[CandidateJudgment, ...], Usage]:
            batch, chunks, questions = packed
            state = {
                "candidates": [_candidate_dict(c) for c in batch],
                "chunks": chunks,
            }
            payload, usage = self._decide(state, questions)
            answers = payload.get("answers")
            if not isinstance(answers, dict):
                raise MalformedResponseError("missing answers")
            items = self._bind(answers, frozenset(c.id for c in batch))
            return items, usage

        workers = max(1, int(self.config.concurrency))
        if workers == 1 or len(batches) < 2:
            parts = [run(b) for b in batches]
        else:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                parts = list(pool.map(run, batches))
        items: list[CandidateJudgment] = []
        usage = ZERO_USAGE
        for part_items, part_usage in parts:
            items.extend(part_items)
            usage = usage.merged(part_usage)
        return Judgments(tuple(items), usage)

    def rerank(self, hits: Sequence[Hit], question: str, *, budget: Budget) -> RankedHits:
        state = {
            "question": question,
            "hits": [
                {"id": h.node_id.value, "name": h.name, "path": h.loc.path}
                for h in hits
            ],
        }
        questions: dict[str, object] = {
            "sufficient": {
                "type": "noul",
                "instructions": "Are these hits sufficient to answer the question?",
                "criteria": {"true": "The hits answer it.", "false": "They do not."},
            }
        }
        for h in hits:
            questions[f"rank_{h.node_id.value}"] = {
                "type": "choice",
                "instructions": "keep or drop this hit for the question",
                "criteria": {"keep": "relevant", "drop": "irrelevant"},
            }
        budget.charge_state(len(json.dumps({"state": state, "questions": questions})))
        payload, usage = self._decide(state, questions)
        answers = payload.get("answers")
        if not isinstance(answers, dict):
            raise MalformedResponseError("missing answers")
        known = {h.node_id.value for h in hits}
        kept: list[Hit] = []
        for h in hits:
            key = f"rank_{h.node_id.value}"
            raw = answers.get(key) or {}
            if not isinstance(raw, dict):
                kept.append(h)
                continue
            if raw.get("choice") == "drop":
                continue
            kept.append(h)
        extra_keys = [
            k for k in answers
            if k.startswith("rank_") and k[len("rank_"):] not in known and k != "sufficient"
        ]
        _ = extra_keys
        suff = answers.get("sufficient") or {}
        noul = float(suff.get("noul") or 0.0) if isinstance(suff, dict) else 0.0
        return RankedHits(tuple(kept), noul >= 0.5, noul, usage)

    def _decide(self, state: object, questions: Mapping[str, object]) -> tuple[dict, Usage]:
        body = json.dumps({
            "model": self.config.model,
            "state": state,
            "questions": questions,
        }).encode()
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        transport = self.transport or urllib_transport
        retries = 0
        latencies: list[float] = []
        last_timeout: TimeoutError | None = None
        last_status = 0
        for attempt in range(self.config.max_retries + 1):
            t0 = time.perf_counter()
            try:
                status, payload = transport(
                    self.config.url, body, headers, self.config.timeout_s
                )
            except TimeoutError as e:
                latencies.append((time.perf_counter() - t0) * 1000)
                last_timeout = e
                if attempt == self.config.max_retries:
                    raise EngineTimeoutError("timeout") from e
                self.sleeper(min(2 ** attempt, 8))
                retries += 1
                continue
            latencies.append((time.perf_counter() - t0) * 1000)
            if status == 429 or status >= 500:
                last_status = status
                if attempt == self.config.max_retries:
                    raise EngineHttpError(status)
                self.sleeper(min(2 ** attempt, 8))
                retries += 1
                continue
            if status != 200:
                raise EngineHttpError(status)
            if not isinstance(payload, dict) or "answers" not in payload:
                raise MalformedResponseError("missing answers")
            raw_usage = payload.get("usage") or {}
            if not isinstance(raw_usage, dict):
                raw_usage = {}
            cost = raw_usage.get("cost")
            return payload, Usage(
                input_tokens=int(raw_usage.get("input_tokens") or 0),
                output_tokens=int(raw_usage.get("output_tokens") or 0),
                cost=float(cost) if cost is not None else None,
                requests=1,
                retries=retries,
                latencies_ms=tuple(latencies),
            )
        if last_timeout is not None:
            raise EngineTimeoutError("timeout") from last_timeout
        raise EngineHttpError(last_status)

    def _pack(
        self,
        candidates: CandidateSet,
        *,
        budget: Budget,
        source: SourceText | None = None,
    ) -> list[tuple[tuple[Candidate, ...], list, dict]]:
        batches: list[tuple[tuple[Candidate, ...], list, dict]] = []
        current: list[Candidate] = []
        for c in candidates.items:
            trial = current + [c]
            chunks = _windows_for(trial, source, budget.max_chunk_chars)
            questions = _questions_for(trial)
            sz = _payload_chars(self.config.model, trial, chunks, questions)
            if sz > budget.max_state_chars:
                if not current:
                    raise BudgetExceeded("state", f"{sz} > {budget.max_state_chars}")
                packed = _windows_for(current, source, budget.max_chunk_chars)
                packed_q = _questions_for(current)
                budget.charge_state(_payload_chars(self.config.model, current, packed, packed_q))
                batches.append((tuple(current), packed, packed_q))
                current = [c]
                chunks = _windows_for(current, source, budget.max_chunk_chars)
                questions = _questions_for(current)
                sz = _payload_chars(self.config.model, current, chunks, questions)
                if sz > budget.max_state_chars:
                    raise BudgetExceeded("state", f"{sz} > {budget.max_state_chars}")
            else:
                current = trial
        if current:
            chunks = _windows_for(current, source, budget.max_chunk_chars)
            questions = _questions_for(current)
            budget.charge_state(_payload_chars(self.config.model, current, chunks, questions))
            batches.append((tuple(current), chunks, questions))
        return batches

    def _bind(
        self,
        raw_answers: Mapping[str, object],
        known: frozenset[CandidateId],
    ) -> tuple[CandidateJudgment, ...]:
        known_vals = {k.value: k for k in known}
        buckets: dict[str, dict[str, dict]] = {}
        for key, raw in raw_answers.items():
            if not isinstance(raw, dict):
                continue
            if key.startswith("keep_"):
                cid_s, slot = key[len("keep_") :], "keep"
            elif key.startswith("role_"):
                cid_s, slot = key[len("role_") :], "role"
            elif key.startswith("strength_"):
                cid_s, slot = key[len("strength_") :], "strength"
            else:
                continue
            if cid_s not in known_vals:
                continue
            buckets.setdefault(cid_s, {})[slot] = raw
        items: list[CandidateJudgment] = []
        for cid_s, slots in buckets.items():
            role = slots.get("role") or {}
            keep_raw = slots.get("keep") or {}
            strength = slots.get("strength") or {}
            choice = role.get("choice")
            noul = float(keep_raw["noul"]) if "noul" in keep_raw else 0.0
            keep = str(choice) != "drop" if choice is not None else noul >= 0.5
            if "confidence" in role:
                conf = float(role["confidence"])
            elif "noul" in keep_raw:
                conf = noul
            else:
                conf = 0.0
            salience = float(strength.get("score") or 0.0)
            role_s = str(choice) if isinstance(choice, str) else None
            items.append(
                CandidateJudgment(known_vals[cid_s], keep, noul, salience, conf, role_s)
            )
        return tuple(items)


def _candidate_dict(c: Candidate) -> dict:
    return {
        "id": c.id.value,
        "kind": c.kind,
        "name": c.name,
        "path": c.loc.path,
        "start_line": c.loc.start_line,
        "end_line": c.loc.end_line,
    }


def _payload_chars(model: str, batch: Sequence[Candidate], chunks: list, questions: dict) -> int:
    return len(json.dumps({
        "model": model,
        "state": {"candidates": [_candidate_dict(c) for c in batch], "chunks": chunks},
        "questions": questions,
    }))


def _windows_for(
    batch: Sequence[Candidate],
    source: SourceText | None,
    max_chars: int,
) -> list:
    if source is None:
        return [
            {"path": c.loc.path, "start_line": c.loc.start_line, "text": _candidate_line(c)}
            for c in batch
        ]
    lines = source.text.splitlines() or [""]
    spans: list[tuple[int, int]] = []
    for c in batch:
        if c.kind == "file":
            acc = 0
            end = 1
            for i, line in enumerate(lines, 1):
                acc += len(line) + 1
                end = i
                if acc >= max_chars:
                    break
            spans.append((1, end))
            continue
        start = max(1, c.loc.start_line - 2)
        end = min(len(lines), c.loc.end_line + 2)
        spans.append((start, end))
    spans.sort()
    merged: list[tuple[int, int]] = []
    for start, end in spans:
        if merged and start <= merged[-1][1] + 1:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    windows = []
    for start, end in merged:
        text = "\n".join(lines[start - 1 : end])
        if len(text) > max_chars:
            text = text[:max_chars]
        windows.append({"path": source.path, "start_line": start, "text": text})
    return windows


def _candidate_line(c: Candidate) -> str:
    return (
        f"{c.id.value} kind={c.kind} name={c.name} path={c.loc.path} "
        f"start={c.loc.start_line} end={c.loc.end_line}"
    )


def _questions_for(batch: Sequence[Candidate]) -> dict[str, object]:
    questions: dict[str, object] = {}
    for c in batch:
        i = c.id.value
        questions[f"keep_{i}"] = {
            "type": "noul",
            "instructions": "Is this candidate a real entity at the given source span?",
            "criteria": {
                "true": "The span names a real file, symbol, import, call, or relation.",
                "false": "The span is noise or not an entity.",
            },
        }
        questions[f"role_{i}"] = {
            "type": "choice",
            "instructions": "keep as file|symbol|import|call|relation or drop",
            "criteria": {
                "file": "a source file",
                "symbol": "a function or class",
                "import": "an import",
                "call": "a call",
                "relation": "a relation pair",
                "drop": "not a real entity",
            },
        }
        questions[f"strength_{i}"] = {
            "type": "score",
            "instructions": "weak/ok/strong evidence",
            "criteria": ["weak", "ok", "strong"],
        }
    return questions
