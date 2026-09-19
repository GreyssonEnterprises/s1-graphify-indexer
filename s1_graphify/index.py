from __future__ import annotations

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
from s1_graphify.report import AbortDetail, Checkpoint, IndexReport, percentile
from s1_graphify.rss import rss_bytes
from s1_graphify.stream import stream_sources


def _commit(repo: Path) -> str:
    proc = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, check=False)
    if proc.returncode == 0:
        return proc.stdout.decode().strip()
    return "unknown"


def _abort_reason(exc: BaseException) -> AbortDetail:
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
    facts: list[FileFacts] = list(ckpt.facts) if ckpt else []
    done = {f.path for f in facts}
    abort: AbortDetail | None = None
    request_count = 0
    retries = 0
    input_tokens = 0
    cost: float | None = None
    latencies: list[float] = []
    files = 0
    try:
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
            facts.append(FileFacts(source.path, cset, judged))
            Checkpoint(str(repo), _commit(repo), tuple(facts)).save_atomic(ckpt_path)
        if not facts:
            abort = AbortDetail("budget", "no source files judged")
            Checkpoint(str(repo), _commit(repo), tuple(facts)).save_atomic(ckpt_path)
        else:
            doc = GraphDocument.from_facts(
                facts,
                commit=_commit(repo),
                endpoint=engine.config.url,
                model=engine.config.model,
                rss_check=lambda: budget.check_rss(reader),
            )
            doc.publish_atomic(out)
            if ckpt_path.exists():
                ckpt_path.unlink()
    except (
        BudgetExceeded,
        EngineHttpError,
        EngineTimeoutError,
        MalformedResponseError,
        TimeoutError,
        MissingKeyError,
    ) as exc:
        abort = _abort_reason(exc)
        Checkpoint(str(repo), _commit(repo), tuple(facts)).save_atomic(ckpt_path)
        if graph_path.exists():
            graph_path.unlink()
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


def benchmark_manifest(manifest_path: Path, engine: DecisionsEngine, budget: Budget) -> None:
    from s1_graphify.query import retrieve

    manifest = json.loads(Path(manifest_path).read_text())
    repo = Path(manifest["repo"])
    out = Path("benchmark_out")
    index_repo(repo, out, engine, budget)
    graph = GraphDocument.load(out / "graph.json")
    results = []
    for item in manifest.get("questions") or []:
        q = item.get("q") or item.get("question")
        expect = item.get("expect_names") or item.get("expect_name")
        if isinstance(expect, str):
            expect = [expect]
        hits = retrieve(graph, q)
        names = [h.name for h in hits]
        hit = bool(expect) and expect[0] in names
        results.append({"question": q, "hit": hit, "top": names[:5]})
    Path("benchmark_metrics.json").write_text(json.dumps({"questions": results}, indent=2))
