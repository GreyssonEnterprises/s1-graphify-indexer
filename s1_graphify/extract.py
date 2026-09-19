from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

CandidateKind = Literal["file", "symbol", "import", "call", "relation"]


@dataclass(frozen=True)
class CandidateId:
    value: str


def _mint_id(value: str) -> CandidateId:
    return CandidateId(value)


@dataclass(frozen=True)
class SourceLoc:
    path: str
    start_line: int
    end_line: int


@dataclass(frozen=True)
class SourceText:
    path: str
    text: str
    size_bytes: int


@dataclass(frozen=True)
class Candidate:
    id: CandidateId
    kind: CandidateKind
    name: str
    loc: SourceLoc
    extra: Mapping[str, str]


@dataclass(frozen=True)
class CandidateSet:
    source_path: str
    items: tuple[Candidate, ...]

    def ids(self) -> frozenset[CandidateId]:
        return frozenset(c.id for c in self.items)


_DEF = re.compile(r"^(?:async\s+)?def\s+(\w+)")
_CLASS = re.compile(r"^class\s+(\w+)")
_FROM = re.compile(r"^from\s+([\w.]+)\s+import\s+(.+)$")
_IMPORT = re.compile(r"^import\s+([\w.]+)")
_CALL = re.compile(r"\b(\w+)\s*\(")
_NOT_CALL = frozenset({
    "if", "elif", "while", "for", "def", "class", "return", "await",
    "not", "and", "or", "in", "is", "lambda", "with", "match", "case",
})


def extract_candidates(source: SourceText) -> CandidateSet:
    lines = source.text.splitlines() or [""]
    path = source.path
    end = max(1, len(lines))
    file_id = _mint_id(f"file:{path}")
    items: list[Candidate] = [
        Candidate(file_id, "file", Path(path).name, SourceLoc(path, 1, end), {}),
    ]
    current = file_id
    current_name = Path(path).name
    for i, line in enumerate(lines, 1):
        stripped = line.strip()
        loc = SourceLoc(path, i, i)
        def_m = _DEF.match(stripped)
        if def_m:
            name = def_m.group(1)
            cid = _mint_id(f"symbol:{path}:{name}:{i}")
            items.append(Candidate(cid, "symbol", name, loc, {}))
            current = cid
            current_name = name
            continue
        class_m = _CLASS.match(stripped)
        if class_m:
            name = class_m.group(1)
            cid = _mint_id(f"symbol:{path}:{name}:{i}")
            items.append(Candidate(cid, "symbol", name, loc, {}))
            current = cid
            current_name = name
            continue
        from_m = _FROM.match(stripped)
        if from_m:
            _mod, names = from_m.group(1), from_m.group(2)
            for raw in names.split(","):
                raw = raw.strip()
                if not raw or raw == "*":
                    continue
                name = raw.split()[0]
                iid = _mint_id(f"import:{path}:{name}:{i}")
                extra = {"src": file_id.value, "dst": f"name:{name}"}
                items.append(Candidate(iid, "import", name, loc, extra))
                items.append(Candidate(_mint_id(f"symbol:{path}:{name}:{i}"), "symbol", name, loc, {}))
                rid = _mint_id(f"relation:{path}:{Path(path).name}:{name}:{i}")
                items.append(Candidate(rid, "relation", f"{Path(path).name}->{name}", loc, extra))
            continue
        imp_m = _IMPORT.match(stripped)
        if imp_m:
            name = imp_m.group(1)
            iid = _mint_id(f"import:{path}:{name}:{i}")
            extra = {"src": file_id.value, "dst": f"name:{name}"}
            items.append(Candidate(iid, "import", name, loc, extra))
            continue
        for call_m in _CALL.finditer(stripped):
            name = call_m.group(1)
            if name in _NOT_CALL:
                continue
            cid = _mint_id(f"call:{path}:{name}:{i}")
            extra = {"src": current.value, "dst": f"name:{name}"}
            items.append(Candidate(cid, "call", name, loc, extra))
            rid = _mint_id(f"relation:{path}:{current_name}:{name}:{i}")
            items.append(
                Candidate(rid, "relation", f"{current_name}->{name}", loc, extra)
            )
    return CandidateSet(path, tuple(items))
