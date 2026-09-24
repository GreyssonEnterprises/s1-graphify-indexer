from __future__ import annotations

import hashlib
import json
import subprocess
import time
from collections.abc import Callable
from pathlib import Path

from s1_graphify.budget import Budget, BudgetExceeded
from s1_graphify.config import MissingKeyError
from s1_graphify.engine import (
    DecisionsEngine,
    EngineHttpError,
    EngineTimeoutError,
    MalformedResponseError,
)
from s1_graphify.extract import extract_candidates
from s1_graphify.graph import FileFacts, GraphDocument
from s1_graphify.report import (
    AbortDetail,
    Checkpoint,
    CheckpointIdentityError,
    IndexReport,
    percentile,
)
from s1_graphify.rss import rss_bytes
from s1_graphify.stream import stream_sources


# A killed process re-judges files since the last write; any other exit saves first.
CHECKPOINT_EVERY = 25


def _commit(repo: Path) -> str:
    proc = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, check=False)
    if proc.returncode == 0:
        return proc.stdout.decode().strip()
    return "unknown"


def _file_sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _require_checkpoint_identity(repo: Path, ckpt: Checkpoint) -> None:
    repo_resolved = repo.resolve()
    if ckpt.repo and Path(ckpt.repo).resolve() != repo_resolved:
        raise CheckpointIdentityError(
            f"checkpoint was written for {ckpt.repo}, not {repo_resolved}; use a fresh output directory"
        )
    current = _commit(repo)
    if ckpt.commit and ckpt.commit != current:
        raise CheckpointIdentityError(
            f"checkpoint commit {ckpt.commit} does not match {current}; use a fresh output directory"
        )


def _resumable(repo: Path, ckpt: Checkpoint) -> tuple[list[FileFacts], dict[str, str]]:
    facts: list[FileFacts] = []
    hashes: dict[str, str] = {}
    for fact in ckpt.facts:
        stored = ckpt.file_hashes.get(fact.path)
        if not stored:
            continue
        digest = _file_sha256(repo / fact.path)
        if digest != stored:
            continue
        facts.append(fact)
        hashes[fact.path] = stored
    return facts, hashes


def _abort_reason(exc: BaseException) -> AbortDetail:
    if isinstance(exc, CheckpointIdentityError):
        return AbortDetail("identity", str(exc))
    if isinstance(exc, BudgetExceeded):
        reason = "rss" if exc.kind == "rss" else "budget"
        return AbortDetail(reason, str(exc))
    if isinstance(exc, EngineHttpError):
        return AbortDetail("http", str(exc))
    if isinstance(exc, (EngineTimeoutError, TimeoutError)):
        return AbortDetail("timeout", str(exc))
    if isinstance(exc, MalformedResponseError):
        return AbortDetail("malformed", str(exc))
    if isinstance(exc, MissingKeyError):
        return AbortDetail("missing_key", str(exc))
    return AbortDetail("budget", str(exc))


def index_repo(
    repo: Path,
    out: Path,
    engine: DecisionsEngine,
    budget: Budget,
    rss_reader: Callable[[], int] | None = None,
) -> bool:
    started = time.perf_counter()
    reader = rss_reader or rss_bytes
    out.mkdir(parents=True, exist_ok=True)
    graph_path = out / "graph.json"
    if graph_path.exists():
        graph_path.unlink()
    tmp = out / "graph.json.tmp"
    if tmp.exists():
        tmp.unlink()

    ckpt_path = out / "checkpoint.json"
    ckpt = Checkpoint.load(ckpt_path)
    facts: list[FileFacts] = []
    file_hashes: dict[str, str] = {}
    done: set[str] = set()
    abort: AbortDetail | None = None
    request_count = 0
    retries = 0
    input_tokens = 0
    cost: float | None = None
    latencies: list[float] = []
    files = 0
    unsaved = False
    commit = _commit(repo)
    try:
        if ckpt is not None:
            _require_checkpoint_identity(repo, ckpt)
            facts, file_hashes = _resumable(repo, ckpt)
            done = {f.path for f in facts}
        budget.check_rss(reader)
        for source in stream_sources(
            repo,
            budget,
            out_dir=out,
            skip=frozenset(done),
            rss_reader=rss_reader,
        ):
            budget.check_rss(reader)
            files += 1
            cset = extract_candidates(source)
            judged = engine.judge(cset, budget=budget, source=source)
            latencies.extend(judged.usage.latencies_ms)
            request_count += judged.usage.requests
            retries += judged.usage.retries
            input_tokens += judged.usage.input_tokens
            if judged.usage.cost is not None:
                cost = (cost or 0.0) + float(judged.usage.cost)
            unsaved = True
            facts.append(FileFacts(source.path, cset, judged))
            if source.content_sha256:
                file_hashes[source.path] = source.content_sha256
            if files % CHECKPOINT_EVERY == 0:
                Checkpoint(str(repo), commit, tuple(facts), dict(file_hashes)).save_atomic(ckpt_path)
                unsaved = False
        if not facts:
            abort = AbortDetail("budget", "no source files judged")
            Checkpoint(str(repo), commit, tuple(facts), dict(file_hashes)).save_atomic(ckpt_path)
        else:
            doc = GraphDocument.from_facts(
                facts,
                commit=commit,
                endpoint=engine.config.url,
                model=engine.config.model,
                rss_check=lambda: budget.check_rss(reader),
            )
            doc.publish_atomic(out)
            if ckpt_path.exists():
                ckpt_path.unlink()
    except CheckpointIdentityError as exc:
        abort = _abort_reason(exc)
    except (
        BudgetExceeded,
        EngineHttpError,
        EngineTimeoutError,
        MalformedResponseError,
        TimeoutError,
        MissingKeyError,
    ) as exc:
        abort = _abort_reason(exc)
        Checkpoint(str(repo), commit, tuple(facts), dict(file_hashes)).save_atomic(ckpt_path)
        if graph_path.exists():
            graph_path.unlink()
    except BaseException:
        if unsaved:
            Checkpoint(str(repo), commit, tuple(facts), dict(file_hashes)).save_atomic(ckpt_path)
        raise
    IndexReport(
        wall_s=time.perf_counter() - started,
        files=files,
        chunks=budget.chunks_seen,
        requests=request_count,
        latency_ms_p50=percentile(latencies, 50),
        latency_ms_p95=percentile(latencies, 95),
        input_tokens=input_tokens,
        cost=cost,
        peak_rss_bytes=budget.peak_rss_bytes or int(reader()),
        retries=retries,
        abort=abort,
    ).write_md(out / "INDEX_REPORT.md")
    return abort is None


def benchmark_manifest(manifest_path: Path, engine: DecisionsEngine, budget: Budget) -> bool:
    from s1_graphify.query import retrieve

    manifest = json.loads(Path(manifest_path).read_text())
    repo = Path(manifest["repo"])
    out = Path("benchmark_out")
    if not index_repo(repo, out, engine, budget):
        return False
    graph = GraphDocument.load(out / "graph.json")
    results = []
    for item in manifest.get("questions") or []:
        q = item.get("q") or item.get("question")
        expect = item.get("expect_names") or item.get("expect_name")
        if isinstance(expect, str):
            expect = [expect]
        hits = retrieve(graph, q)
        names = [h.name for h in hits]
        expected = list(expect or [])
        hit = bool(expected) and all(name in names for name in expected)
        results.append({"question": q, "hit": hit, "top": names[:5]})
    Path("benchmark_metrics.json").write_text(
        json.dumps({"match": "all", "questions": results}, indent=2)
    )
    return True
