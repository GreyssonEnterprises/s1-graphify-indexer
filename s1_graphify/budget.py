from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from s1_graphify.rss import rss_bytes

AbortKind = Literal["file_size", "chunk", "chunks", "state", "rss"]


class BudgetExceeded(Exception):
    def __init__(self, kind: AbortKind, detail: str) -> None:
        self.kind = kind
        super().__init__(detail)


@dataclass
class Budget:
    max_file_bytes: int = 1_000_000
    max_chunk_chars: int = 8000
    max_chunks: int = 10_000
    max_state_chars: int = 24_000
    max_rss_bytes: int = 512_000_000
    files_seen: int = 0
    chunks_seen: int = 0
    state_chars_seen: int = 0
    peak_rss_bytes: int = 0

    @staticmethod
    def default() -> Budget:
        return Budget()

    @staticmethod
    def query() -> Budget:
        return Budget()

    def charge_file(self, size: int) -> None:
        if size > self.max_file_bytes:
            raise BudgetExceeded("file_size", f"{size} > {self.max_file_bytes}")
        self.files_seen += 1

    def charge_chunks(self, n: int, chars: int) -> None:
        if chars > self.max_chunk_chars:
            raise BudgetExceeded("chunk", f"{chars} > {self.max_chunk_chars}")
        if self.chunks_seen + n > self.max_chunks:
            raise BudgetExceeded("chunks", f"{self.chunks_seen + n} > {self.max_chunks}")
        self.chunks_seen += n

    def charge_state(self, chars: int) -> None:
        if chars > self.max_state_chars:
            raise BudgetExceeded("state", f"{chars} > {self.max_state_chars}")
        self.state_chars_seen += chars

    def check_rss(self, reader: Callable[[], int] | None = None) -> None:
        current = rss_bytes(reader)
        if current > self.peak_rss_bytes:
            self.peak_rss_bytes = current
        if current > self.max_rss_bytes:
            raise BudgetExceeded("rss", f"{current} > {self.max_rss_bytes}")
