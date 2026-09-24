from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

CandidateKind = Literal["file", "symbol", "import", "call", "relation"]

PY_EXT = {".py", ".pyi"}
JS_EXT = {".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs", ".mts", ".cts"}


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
    content_sha256: str = ""


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


_SYMBOL = re.compile(
    r"^(?:export\s+(?:default\s+)?)?(?:pub(?:lic)?\s+)?(?:async\s+)?"
    r"(?:def|function|func|fun|fn)\s+(\w+)"
)
_CLASS = re.compile(
    r"^(?:export\s+(?:default\s+)?)?(?:pub(?:lic)?\s+)?class\s+(\w+)"
)
_FROM = re.compile(r"^from\s+([\w.]+)\s+import\s+(.+)$")
_IMPORT = re.compile(r"^import\s+([\w.]+)$")
_JS_FROM = re.compile(
    r"^import\s+(?:type\s+)?(.+?)\s+from\s+['\"]([^'\"]+)['\"]"
)
_JS_SIDE = re.compile(r"^import\s+['\"]([^'\"]+)['\"]")
_CALL = re.compile(r"\b(\w+)\s*\(")
_NOT_CALL = frozenset({
    "if", "elif", "else", "while", "for", "def", "class", "return", "await",
    "not", "and", "or", "in", "is", "lambda", "with", "match", "case",
    "function", "typeof", "switch", "catch", "finally", "void", "super",
    "new", "import", "export", "async", "from", "try", "throw", "yield",
    "interface", "package", "struct", "impl", "trait", "fn", "fun", "func",
    "let", "const", "var", "type", "do", "select", "range", "go", "defer",
})


def extract_candidates(source: SourceText) -> CandidateSet:
    lines = source.text.splitlines() or [""]
    path = source.path
    suffix = Path(path).suffix.lower()
    end = max(1, len(lines))
    file_id = _mint_id(f"file:{path}")
    items: list[Candidate] = [
        Candidate(file_id, "file", Path(path).name, SourceLoc(path, 1, end), {}),
    ]
    current = file_id
    current_name = Path(path).name
    for i, line in enumerate(lines, 1):
        stripped = line.strip().rstrip("{").strip()
        loc = SourceLoc(path, i, i)
        sym_m = _SYMBOL.match(stripped)
        if sym_m:
            name = sym_m.group(1)
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
        if suffix in PY_EXT:
            from_m = _FROM.match(stripped)
            if from_m:
                names = from_m.group(2)
                for k, raw in enumerate(names.split(","), 1):
                    raw = raw.strip()
                    if not raw or raw == "*":
                        continue
                    name = raw.split()[0]
                    extra = {"src": file_id.value, "dst": f"name:{name}"}
                    items.append(
                        Candidate(
                            _mint_id(f"import:{path}:{name}:{i}:{k}"),
                            "import", name, loc, extra,
                        )
                    )
                    items.append(
                        Candidate(
                            _mint_id(f"relation:{path}:{Path(path).name}:{name}:{i}:{k}"),
                            "relation", f"{Path(path).name}->{name}", loc, extra,
                        )
                    )
                continue
            imp_m = _IMPORT.match(stripped)
            if imp_m:
                name = imp_m.group(1)
                extra = {"src": file_id.value, "dst": f"name:{name}"}
                items.append(
                    Candidate(
                        _mint_id(f"import:{path}:{name}:{i}:1"),
                        "import", name, loc, extra,
                    )
                )
                continue
        if suffix in JS_EXT:
            js_from = _JS_FROM.match(stripped.rstrip(";"))
            if js_from:
                for k, name in enumerate(_js_imported_names(js_from.group(1)), 1):
                    extra = {"src": file_id.value, "dst": f"name:{name}"}
                    items.append(
                        Candidate(
                            _mint_id(f"import:{path}:{name}:{i}:{k}"),
                            "import", name, loc, extra,
                        )
                    )
                    items.append(
                        Candidate(
                            _mint_id(f"relation:{path}:{Path(path).name}:{name}:{i}:{k}"),
                            "relation", f"{Path(path).name}->{name}", loc, extra,
                        )
                    )
                continue
            js_side = _JS_SIDE.match(stripped.rstrip(";"))
            if js_side:
                name = js_side.group(1)
                extra = {"src": file_id.value, "dst": f"name:{name}"}
                items.append(
                    Candidate(
                        _mint_id(f"import:{path}:{name}:{i}:1"),
                        "import", name, loc, extra,
                    )
                )
                continue
        occ = 0
        for call_m in _CALL.finditer(stripped):
            name = call_m.group(1)
            if name in _NOT_CALL:
                continue
            occ += 1
            extra = {"src": current.value, "dst": f"name:{name}"}
            items.append(
                Candidate(
                    _mint_id(f"call:{path}:{name}:{i}:{occ}"),
                    "call", name, loc, extra,
                )
            )
            items.append(
                Candidate(
                    _mint_id(f"relation:{path}:{current_name}:{name}:{i}:{occ}"),
                    "relation", f"{current_name}->{name}", loc, extra,
                )
            )
    return CandidateSet(path, tuple(items))


def _js_imported_names(clause: str) -> list[str]:
    clause = clause.strip()
    if clause.startswith("{") and "}" in clause:
        inner = clause[1:clause.index("}")]
        names: list[str] = []
        for part in inner.split(","):
            toks = [t for t in part.replace(",", " ").split() if t not in {"type", "typeof"}]
            if not toks:
                continue
            names.append(toks[-1])
        return names
    if "*" in clause:
        toks = clause.split()
        return [toks[-1]] if toks else []
    token = clause.split()[0] if clause else ""
    return [token] if token else []
