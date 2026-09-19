from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

from s1_graphify.budget import Budget
from s1_graphify.engine import DecisionsEngine
from s1_graphify.extract import SourceText, extract_candidates

TOY = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "toy" / "pkg" / "config.py"


def main() -> int:
    if not os.environ.get("OPENROUTER_API_KEY"):
        print("SKIP: OPENROUTER_API_KEY not set", file=sys.stderr)
        return 2
    text = TOY.read_text()
    source = SourceText("pkg/config.py", text, len(text.encode()))
    cset = extract_candidates(source)
    engine = DecisionsEngine.from_env()
    started = time.perf_counter()
    judged = engine.judge(cset, budget=Budget.default())
    latency_ms = (time.perf_counter() - started) * 1000
    payload = {
        "model": engine.config.model,
        "endpoint": engine.config.url,
        "latency_ms": round(latency_ms, 2),
        "requests": judged.usage.requests,
        "retries": judged.usage.retries,
        "input_tokens": judged.usage.input_tokens,
        "cost": judged.usage.cost,
        "decisions": [
            {
                "id": item.candidate_id.value,
                "keep": item.keep,
                "noul": item.noul,
                "role": item.role,
                "confidence": item.confidence,
            }
            for item in judged.items
        ],
    }
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
