from __future__ import annotations

import json
import socket
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

from s1_graphify.budget import Budget, BudgetExceeded
from s1_graphify.config import EngineConfig, MissingKeyError
from s1_graphify.extract import Candidate, CandidateId, CandidateSet, SourceLoc


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


def urllib_transport(url: str, data: bytes, headers: dict[str, str], timeout: float):
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
            status = getattr(resp, "status", 200)
            return status, json.loads(raw.decode())
    except urllib.error.HTTPError as e:
        raw = e.read() if e.fp else b""
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
        chunks: list | None = None,
    ) -> Judgments:
        batches = self._pack(candidates, budget=budget)
        if not batches:
            return Judgments((), ZERO_USAGE)

        def run(batch: tuple[Candidate, ...]) -> tuple[tuple[CandidateJudgment, ...], Usage]:
            state = {
                "candidates": [
                    {
                        "id": c.id.value,
                        "kind": c.kind,
                        "name": c.name,
                        "path": c.loc.path,
                        "start_line": c.loc.start_line,
                        "end_line": c.loc.end_line,
                    }
                    for c in batch
                ],
                "chunks": chunks if chunks is not None else [_candidate_line(c) for c in batch],
            }
            questions = _questions_for(batch)
            payload, usage = self._decide(state, questions)
            answers = payload.get("answers")
            if not isinstance(answers, dict):
                raise MalformedResponseError("missing answers")
            items = self._bind(answers, frozenset(c.id for c in batch))
            return items, usage

        if len(batches) == 1 or self.config.concurrency <= 1:
            parts = [run(b) for b in batches]
        else:
            with ThreadPoolExecutor(max_workers=self.config.concurrency) as pool:
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
        budget.charge_state(len(json.dumps(state)))
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

    def _pack(self, candidates: CandidateSet, *, budget: Budget) -> list[tuple[Candidate, ...]]:
        batches: list[tuple[Candidate, ...]] = []
        current: list[Candidate] = []
        current_size = 0
        for c in candidates.items:
            line = _candidate_line(c)
            sz = len(line)
            if sz > budget.max_state_chars:
                raise BudgetExceeded("state", f"{sz} > {budget.max_state_chars}")
            if current and current_size + sz > budget.max_state_chars:
                budget.charge_state(current_size)
                batches.append(tuple(current))
                current = []
                current_size = 0
            current.append(c)
            current_size += sz
        if current:
            budget.charge_state(current_size)
            batches.append(tuple(current))
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
