from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from s1_graphify.graph import FileFacts

AbortReason = Literal["rss", "budget", "http", "malformed", "missing_key", "timeout", "identity"]


@dataclass(frozen=True)
class AbortDetail:
    reason: AbortReason
    message: str


class CheckpointIdentityError(Exception):
    pass


@dataclass
class Checkpoint:
    repo: str
    commit: str
    facts: tuple[FileFacts, ...]
    file_hashes: dict[str, str] = field(default_factory=dict)

    def paths_done(self) -> frozenset[str]:
        return frozenset(f.path for f in self.facts)

    def save_atomic(self, path: Path) -> None:
        payload = json.dumps({
            "repo": self.repo,
            "commit": self.commit,
            "facts": [f.to_dict() for f in self.facts],
            "file_hashes": dict(self.file_hashes),
        })
        _atomic_write(path, payload)

    @staticmethod
    def load(path: Path) -> Checkpoint | None:
        path = Path(path)
        if not path.exists():
            return None
        data = json.loads(path.read_text())
        facts = tuple(FileFacts.from_dict(f) for f in data.get("facts") or [])
        raw_hashes = data.get("file_hashes") or {}
        if not isinstance(raw_hashes, dict):
            raw_hashes = {}
        file_hashes = {str(k): str(v) for k, v in raw_hashes.items()}
        return Checkpoint(data.get("repo") or "", data.get("commit") or "", facts, file_hashes)


@dataclass
class IndexReport:
    wall_s: float
    files: int
    chunks: int
    requests: int
    latency_ms_p50: float
    latency_ms_p95: float
    input_tokens: int
    cost: float | None
    peak_rss_bytes: int
    retries: int
    abort: AbortDetail | None

    def write_md(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        cost_line = "not returned" if self.cost is None else str(self.cost)
        status = "aborted" if self.abort else "complete"
        lines = [
            "# Index report",
            "",
            f"status: {status}",
            f"wall_s: {self.wall_s:.4f}",
            f"files: {self.files}",
            f"chunks: {self.chunks}",
            f"requests: {self.requests}",
            f"retries: {self.retries}",
            f"latency_p50_ms: {self.latency_ms_p50:.2f}",
            f"latency_p95_ms: {self.latency_ms_p95:.2f}",
            f"input_tokens: {self.input_tokens}",
            f"cost: {cost_line}",
            f"peak_rss_bytes: {self.peak_rss_bytes}",
        ]
        if self.abort:
            lines.append(f"abort_reason: {self.abort.reason}")
            lines.append(f"abort_message: {self.abort.message}")
        _atomic_write(path, "\n".join(lines) + "\n")


def _atomic_write(path: Path, payload: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(payload)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def percentile(xs: tuple[float, ...] | list[float], p: float) -> float:
    if not xs:
        return 0.0
    ordered = sorted(xs)
    idx = min(len(ordered) - 1, max(0, round((p / 100) * (len(ordered) - 1))))
    return ordered[idx]
