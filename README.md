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

The indexer streams one file at a time. It never materializes the whole corpus. It skips binaries, generated directories, symlinks, and its own output files. A path that resolves outside the repository is not read. File size, chunk count, request state, and RSS caps abort the run instead of producing a partial graph.

Extraction is line-oriented regular expressions, not a language parser. It runs on every source suffix the walker accepts, but imports are extracted only for Python and JavaScript or TypeScript. In other languages it can miss symbols or label a definition as a call. Names and spans can be wrong on syntax the patterns do not recognize.

If the selected directory has no `.git`, or `git ls-files` fails, the indexer walks the tree instead of using the git index. That walk can see untracked files.

Judge requests send source text to the configured Decisions URL. For each file that is the first `S1_MAX_CHUNK_CHARS` characters (8000 by default), which is the whole file when the file is smaller, plus a window of two lines on each side of every extracted symbol, import, and call. Nearby windows merge, and each is capped at `S1_MAX_CHUNK_CHARS`, so most of a large file can be sent. Requests also carry each candidate's name, kind, repository-relative path, and line numbers. Treat all of it as leaving the machine. Do not index a repository whose contents must not reach that endpoint.

## Resume and a fresh output directory

A later `index` into the same `--out` resumes when the checkpoint's repository path and git commit match this run. Each resumed file must also still have the same bytes. A changed file is judged again. Files after the last checkpoint are judged again too.

The checkpoint is rewritten every 25 files and whenever a run stops early, including an abort or Ctrl-C. A checkpoint identity mismatch is the exception: that abort leaves the existing checkpoint untouched. If the process is killed, the judgments since the last write are paid for again. Use a new `--out` directory when the repository path or commit does not match, when you want to discard partial judgments, or when the CLI says the checkpoint identity does not match. Do not point two repositories at one output directory.

On abort the CLI prints the paths of `INDEX_REPORT.md` and `checkpoint.json`. An offline query that finds nothing prints `no matches` to stderr and leaves stdout empty.

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

Writes `benchmark_metrics.json` in the current directory. A question hits only when every name in `expect_names` is in the retrieved list (`"match": "all"`). If indexing aborts, the command exits nonzero and does not write that file.

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
