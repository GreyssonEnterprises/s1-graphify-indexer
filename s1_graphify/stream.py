from __future__ import annotations

import hashlib
import os
import subprocess
from collections.abc import Callable, Iterator
from pathlib import Path

from s1_graphify.budget import Budget
from s1_graphify.extract import SourceText

SKIP_DIRS = {
    "node_modules", ".git", "__pycache__", "dist", "build",
    ".venv", "target", "vendor",
}
BIN_EXT = {
    ".png", ".jpg", ".jpeg", ".gif", ".webp", ".ico", ".pdf", ".zip", ".gz",
    ".tar", ".whl", ".so", ".dylib", ".dll", ".exe", ".bin", ".pyc", ".pyo",
    ".class", ".o", ".a", ".woff", ".woff2", ".ttf", ".eot",
}
SOURCE_EXT = {
    ".py", ".pyi",
    ".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs", ".mts", ".cts",
    ".go", ".rs", ".swift", ".kt", ".kts",
    ".c", ".h", ".cc", ".cpp", ".hpp", ".cs",
    ".java", ".rb", ".php",
}


def stream_sources(
    repo: Path,
    budget: Budget,
    *,
    out_dir: Path | None = None,
    skip: frozenset[str] | None = None,
    rss_reader: Callable[[], int] | None = None,
) -> Iterator[SourceText]:
    repo = Path(repo)
    skip = skip or frozenset()
    out_res = out_dir.resolve() if out_dir else None
    for path in _iter_paths(repo):
        budget.check_rss(rss_reader)
        rel = path.relative_to(repo).as_posix()
        if rel in skip:
            continue
        if any(part in SKIP_DIRS for part in Path(rel).parts):
            continue
        if path.name in {"graph.json", "INDEX_REPORT.md", "checkpoint.json"}:
            continue
        if out_res is not None:
            resolved = path.resolve()
            if resolved == out_res or resolved.is_relative_to(out_res):
                continue
        if path.suffix.lower() in BIN_EXT:
            continue
        if path.suffix.lower() not in SOURCE_EXT:
            continue
        if path.is_symlink():
            continue
        resolved = path.resolve()
        if not resolved.is_relative_to(repo.resolve()):
            continue
        size = path.stat().st_size
        budget.charge_file(size)
        data = path.read_bytes()
        if b"\0" in data[:8192]:
            continue
        digest = hashlib.sha256(data).hexdigest()
        text = data.decode("utf-8", errors="replace")
        max_c = max(1, budget.max_chunk_chars)
        n = max(1, (len(text) + max_c - 1) // max_c)
        for i in range(n):
            piece_len = min(max_c, len(text) - i * max_c)
            budget.charge_chunks(1, piece_len)
        yield SourceText(rel, text, size, digest)


def _iter_paths(repo: Path) -> Iterator[Path]:
    listed = _git_ls(repo)
    if listed is not None:
        paths = [repo / p for p in listed]
        paths.sort()
        for path in paths:
            if path.is_file():
                yield path
        return
    for root, dirs, files in os.walk(repo):
        dirs[:] = sorted(d for d in dirs if d not in SKIP_DIRS and not d.startswith("."))
        rootp = Path(root)
        for name in sorted(files):
            yield rootp / name


def _git_ls(repo: Path) -> list[str] | None:
    if not (repo / ".git").exists():
        return None
    try:
        proc = subprocess.run(
            ["git", "ls-files", "-z"],
            cwd=repo,
            capture_output=True,
            check=False,
        )
    except FileNotFoundError:
        return None
    if proc.returncode != 0:
        return None
    return [p.decode("utf-8", "surrogateescape") for p in proc.stdout.split(b"\0") if p]
