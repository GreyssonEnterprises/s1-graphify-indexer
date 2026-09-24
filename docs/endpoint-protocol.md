# Decisions endpoint protocol

s1-graphify-indexer talks to a System One Decisions HTTP API. It does not call chat completions.

## Request

`POST` the configured URL. Default: `https://openrouter.ai/api/alpha/decisions`

Headers:

```
Authorization: Bearer <api-key>
Content-Type: application/json
```

Body:

```json
{
  "model": "typesafe/jev-1.13",
  "state": {
    "chunks": [{"path": "a.py", "start_line": 1, "text": "def parse_config():\n    pass\n"}],
    "candidates": [
      {
        "id": "symbol:a.py:parse_config:1",
        "kind": "symbol",
        "name": "parse_config",
        "path": "a.py",
        "start_line": 1,
        "end_line": 1
      }
    ]
  },
  "questions": {
    "keep_symbol:a.py:parse_config:1": {
      "type": "noul",
      "instructions": "Is this candidate a real entity at the given source span?",
      "criteria": {
        "true": "The span names a real file, symbol, import, call, or relation.",
        "false": "The span is noise or not an entity."
      }
    },
    "role_symbol:a.py:parse_config:1": {
      "type": "choice",
      "instructions": "keep as file|symbol|import|call|relation or drop",
      "criteria": {
        "file": "a source file",
        "symbol": "a function or class",
        "import": "an import",
        "call": "a call",
        "relation": "a relation pair",
        "drop": "not a real entity"
      }
    },
    "strength_symbol:a.py:parse_config:1": {
      "type": "score",
      "instructions": "weak/ok/strong evidence",
      "criteria": ["weak", "ok", "strong"]
    }
  }
}
```

`state` may be a string, object, or array of text. This indexer sends an object. Questions are independent. Keys are chosen by the client and returned unchanged.

Candidate ids are minted only by the extractor. Jev judges those ids. It does not invent files or symbols.

Multiple candidates from one or more chunks share a request as long as `state` stays under `S1_MAX_STATE_CHARS`, well inside Jev's 32K context.

## Response

```json
{
  "model": "typesafe/jev-1.13",
  "answers": {
    "keep_symbol:a.py:parse_config:1": {"type": "noul", "noul": 0.91},
    "role_symbol:a.py:parse_config:1": {
      "type": "choice",
      "choice": "symbol",
      "probabilities": {"file": 0.01, "symbol": 0.9, "import": 0.03, "call": 0.03, "relation": 0.02, "drop": 0.01},
      "confidence": 0.88
    },
    "strength_symbol:a.py:parse_config:1": {
      "type": "score",
      "score": 1.7,
      "legend": {"0": "weak", "1": "ok", "2": "strong"},
      "probabilities": {"0": 0.05, "1": 0.2, "2": 0.75},
      "confidence": 0.8
    }
  },
  "usage": {"input_tokens": 400, "output_tokens": 40, "cost": 0.000017}
}
```

`usage.cost` may be omitted. Noul answers have no separate confidence field. Choice and score answers include `confidence` in `[0, 1]`.

Answer keys that were not in the request, or that name a candidate id the extractor did not mint, are dropped. They do not become graph nodes.

## Errors and retries

| Status | Indexer behavior |
|---|---|
| Missing key | Fail before any request |
| 400, 401, 403, 404, 413, 422 | Abort. No retry |
| 429, 500, 502, 503, 524, 529 | Retry until the configured bound |
| Timeout | Counts toward the retry bound, then abort |

Retry sleep is exponential and injectable in tests.

## Caps

The indexer bounds file bytes, chunk count, request `state` size, and process RSS while streaming, judging, and assembling. It does not load the whole corpus into memory. Graph output is written to a temp file and renamed only after a complete successful run.

On abort the indexer writes `INDEX_REPORT.md` and `checkpoint.json`. It does not publish `graph.json`.
