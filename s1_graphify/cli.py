from __future__ import annotations

import argparse
import sys
from pathlib import Path

from s1_graphify.budget import Budget
from s1_graphify.config import Caps, EngineConfig, MissingKeyError
from s1_graphify.engine import DecisionsEngine, RankedHits
from s1_graphify.index import benchmark_manifest, index_repo
from s1_graphify.query import query_graph


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    parser = argparse.ArgumentParser(prog="s1-graphify")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_index = sub.add_parser("index")
    p_index.add_argument("repo")
    p_index.add_argument("--out", required=True)
    _add_engine_flags(p_index)
    _add_cap_flags(p_index)

    p_query = sub.add_parser("query")
    p_query.add_argument("question")
    p_query.add_argument("--graph", required=True)
    p_query.add_argument("--rerank", action="store_true")
    _add_engine_flags(p_query)

    p_bench = sub.add_parser("benchmark")
    p_bench.add_argument("manifest")
    _add_engine_flags(p_bench)
    _add_cap_flags(p_bench)

    args = parser.parse_args(argv)
    try:
        return _dispatch(args)
    except MissingKeyError:
        print("missing API key", file=sys.stderr)
        return 2


def _dispatch(args: argparse.Namespace) -> int:
    if args.cmd == "index":
        engine = _engine(args)
        budget = _budget(args)
        ok = index_repo(Path(args.repo), Path(args.out), engine, budget)
        return 0 if ok else 1
    if args.cmd == "query":
        engine = _engine(args) if args.rerank else None
        result = query_graph(
            args.question, Path(args.graph), engine=engine, rerank=args.rerank
        )
        hits = result.hits if isinstance(result, RankedHits) else result
        if not hits:
            print("no matches", file=sys.stderr)
        for h in hits:
            print(f"{h.name}\t{h.loc.path}:{h.loc.start_line}\t{h.score}")
        if isinstance(result, RankedHits):
            print(f"sufficient\t{result.sufficient}\t{result.sufficiency_noul}")
        return 0
    if args.cmd == "benchmark":
        engine = _engine(args)
        budget = _budget(args)
        ok = benchmark_manifest(Path(args.manifest), engine, budget)
        return 0 if ok else 1
    return 2


def _add_engine_flags(p: argparse.ArgumentParser) -> None:
    p.add_argument("--url")
    p.add_argument("--model")
    p.add_argument("--key-env")
    p.add_argument("--timeout", type=float)
    p.add_argument("--retries", type=int)
    p.add_argument("--batch-chars", type=int)
    p.add_argument("--concurrency", type=int)


def _add_cap_flags(p: argparse.ArgumentParser) -> None:
    p.add_argument("--max-file-bytes", type=int)
    p.add_argument("--max-chunk-chars", type=int)
    p.add_argument("--max-chunks", type=int)
    p.add_argument("--max-state-chars", type=int)
    p.add_argument("--max-rss-bytes", type=int)


def _engine(args: argparse.Namespace) -> DecisionsEngine:
    base = EngineConfig.from_env()
    cfg = EngineConfig(
        url=args.url or base.url,
        model=args.model or base.model,
        key_env=args.key_env or base.key_env,
        timeout_s=base.timeout_s if args.timeout is None else args.timeout,
        max_retries=base.max_retries if args.retries is None else args.retries,
        max_state_chars=base.max_state_chars if args.batch_chars is None else args.batch_chars,
        concurrency=base.concurrency if args.concurrency is None else args.concurrency,
    )
    return DecisionsEngine.from_env(config=cfg)


def _budget(args: argparse.Namespace) -> Budget:
    caps = Caps.from_env()
    state_chars = caps.max_state_chars
    if getattr(args, "batch_chars", None) is not None:
        state_chars = args.batch_chars
    elif getattr(args, "max_state_chars", None) is not None:
        state_chars = args.max_state_chars
    return Budget(
        max_file_bytes=_or(args, "max_file_bytes", caps.max_file_bytes),
        max_chunk_chars=_or(args, "max_chunk_chars", caps.max_chunk_chars),
        max_chunks=_or(args, "max_chunks", caps.max_chunks),
        max_state_chars=state_chars,
        max_rss_bytes=_or(args, "max_rss_bytes", caps.max_rss_bytes),
    )


def _or(args: argparse.Namespace, name: str, default):
    val = getattr(args, name, None)
    return default if val is None else val


if __name__ == "__main__":
    raise SystemExit(main())
