from __future__ import annotations

import os
from dataclasses import dataclass


class MissingKeyError(RuntimeError):
    pass


@dataclass(frozen=True)
class EngineConfig:
    url: str = "https://openrouter.ai/api/alpha/decisions"
    model: str = "typesafe/jev-1.13"
    key_env: str = "OPENROUTER_API_KEY"
    timeout_s: float = 30.0
    max_retries: int = 3
    max_state_chars: int = 24_000
    concurrency: int = 4

    @staticmethod
    def from_env() -> EngineConfig:
        def env(name: str, default, conv):
            raw = os.environ.get(name)
            return conv(raw) if raw is not None and raw != "" else default

        return EngineConfig(
            url=env("S1_DECISIONS_URL", "https://openrouter.ai/api/alpha/decisions", str),
            model=env("S1_DECISIONS_MODEL", "typesafe/jev-1.13", str),
            key_env=env("S1_DECISIONS_KEY_ENV", "OPENROUTER_API_KEY", str),
            timeout_s=env("S1_DECISIONS_TIMEOUT", 30.0, float),
            max_retries=env("S1_DECISIONS_RETRIES", 3, int),
            max_state_chars=env("S1_DECISIONS_BATCH_CHARS", 24_000, int),
            concurrency=env("S1_DECISIONS_CONCURRENCY", 4, int),
        )


@dataclass(frozen=True)
class Caps:
    max_file_bytes: int = 1_000_000
    max_chunk_chars: int = 8000
    max_chunks: int = 10_000
    max_state_chars: int = 24_000
    max_rss_bytes: int = 512_000_000

    @staticmethod
    def from_env() -> Caps:
        def env(name: str, default, conv):
            raw = os.environ.get(name)
            return conv(raw) if raw is not None and raw != "" else default

        return Caps(
            max_file_bytes=env("S1_MAX_FILE_BYTES", 1_000_000, int),
            max_chunk_chars=env("S1_MAX_CHUNK_CHARS", 8000, int),
            max_chunks=env("S1_MAX_CHUNKS", 10_000, int),
            max_state_chars=env("S1_MAX_STATE_CHARS", 24_000, int),
            max_rss_bytes=env("S1_MAX_RSS_BYTES", 512_000_000, int),
        )
