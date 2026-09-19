from __future__ import annotations

import json
from pathlib import Path

TOY = Path(__file__).resolve().parent / "fixtures" / "toy"


def answers_for(questions: dict) -> dict:
    answers: dict = {}
    for key, question in questions.items():
        kind = question.get("type")
        if kind == "noul":
            answers[key] = {"type": "noul", "noul": 0.92}
        elif kind == "choice":
            opts = list((question.get("criteria") or {}).keys()) or ["symbol"]
            cid = key.split("_", 1)[-1]
            prefix = cid.split(":", 1)[0]
            choice = prefix if prefix in opts else ("symbol" if "symbol" in opts else opts[0])
            answers[key] = {
                "type": "choice",
                "choice": choice,
                "probabilities": {choice: 1.0},
                "confidence": 0.8,
            }
        elif kind == "score":
            answers[key] = {
                "type": "score",
                "score": 2.0,
                "legend": {"0": "weak", "1": "ok", "2": "strong"},
                "probabilities": {"2": 1.0},
                "confidence": 0.8,
            }
    return answers


class ScriptedTransport:
    def __init__(self, script: list | None = None, extra_answers: dict | None = None) -> None:
        self.script = list(script or [])
        self.extra_answers = dict(extra_answers or {})
        self.calls: list[dict] = []
        self.sleeps: list[float] = []

    def sleeper(self, seconds: float) -> None:
        self.sleeps.append(seconds)

    def __call__(self, url: str, data, headers: dict, timeout: float):
        raw = data.decode() if isinstance(data, (bytes, bytearray)) else data
        body = json.loads(raw) if isinstance(raw, str) else raw
        self.calls.append({"url": url, "headers": headers, "timeout": timeout, "body": body, "data": data})
        if self.script:
            step = self.script.pop(0)
            if step == "timeout":
                raise TimeoutError("timeout")
            if isinstance(step, int) and step != 200:
                return step, {"error": "retry"}
        questions = body.get("questions") or {}
        payload = {
            "model": "typesafe/jev-1.13",
            "answers": {**answers_for(questions), **self.extra_answers},
            "usage": {"input_tokens": 312, "output_tokens": 48, "cost": 0.000013},
        }
        return 200, payload
