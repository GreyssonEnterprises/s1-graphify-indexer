# How to index a repo with s1-graphify

s1-graphify-indexer streams git-tracked source, extracts candidate files, symbols, imports, calls, and relation pairs, then asks TypeSafe Jev 1.13 to judge those candidates. It does not download or run local models.

Requires Python 3.11+. Runtime stays on the standard library. pytest is the only required dev dependency.

## Install

```bash
python -m pip install -e ".[dev]"
```

Set `OPENROUTER_API_KEY` for live Decisions calls. Tests mock the network and do not need a key.

## Index

```bash
python -m s1_graphify index ./path/to/repo --out ./out
```

On success `./out/graph.json` and `./out/INDEX_REPORT.md` appear. On abort you get the report and a checkpoint. There is no complete `graph.json`.

The indexer streams one file at a time. It never materializes the whole corpus. It skips binaries, generated directories, and its own output files. File size, chunk count, request state, and RSS caps abort the run instead of producing a partial graph.

## Query

```bash
python -m s1_graphify query "where is parse_config defined?" --graph ./out/graph.json
python -m s1_graphify query "where is parse_config defined?" --graph ./out/graph.json --rerank
```

`--rerank` sends one Decisions request. Retrieval without it is deterministic and offline.

## Benchmark

```bash
python -m s1_graphify benchmark ./tests/fixtures/manifest.json
```

Writes `benchmark_metrics.json` in the current directory.

## Tests

```bash
python -m pytest -q
```

## Live canary

If `OPENROUTER_API_KEY` is set:

```bash
python scripts/canary.py
```

That script makes exactly one Decisions request against the public toy fixture. It is not a full-repo index.

## Configure the Decisions engine

Defaults match OpenRouter Jev. Point these at another engine that speaks the same JSON contract.

| Variable | Default |
|---|---|
| `S1_DECISIONS_URL` | `https://openrouter.ai/api/alpha/decisions` |
| `S1_DECISIONS_MODEL` | `typesafe/jev-1.13` |
| `S1_DECISIONS_KEY_ENV` | `OPENROUTER_API_KEY` |
| `S1_DECISIONS_TIMEOUT` | `30` |
| `S1_DECISIONS_RETRIES` | `3` |
| `S1_DECISIONS_CONCURRENCY` | `4` |

| Variable | Default | Unit | What it caps |
|---|---|---|---|
| `S1_MAX_FILE_BYTES` | `1000000` | bytes | One source file before it is read |
| `S1_MAX_CHUNK_CHARS` | `8000` | characters | One source window sent to Decisions |
| `S1_MAX_CHUNKS` | `10000` | windows | Windows charged across the run |
| `S1_MAX_STATE_CHARS` | `24000` | characters | Serialized Decisions request |
| `S1_MAX_RSS_BYTES` | `512000000` | bytes | Process resident memory |

Each cap has a matching CLI flag, such as `--max-state-chars` for `S1_MAX_STATE_CHARS`, and the flag wins over the variable. `S1_DECISIONS_TIMEOUT` is seconds. `S1_DECISIONS_RETRIES` and `S1_DECISIONS_CONCURRENCY` are counts.

The HTTP contract is in `docs/endpoint-protocol.md`.
