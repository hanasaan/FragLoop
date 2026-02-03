"""Utility helpers."""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
from typing import Dict, Iterable, Iterator, List

import numpy as np


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def sha1_text(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()


_MULTI_NL_RE = re.compile(r"\n{3,}")


def normalize_text(text: str, strip_trailing: bool = True) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    if "\t" in text:
        text = text.replace("\t", " " * 4)
    if text.startswith("\n"):
        text = text.lstrip("\n")
    if strip_trailing:
        text = "\n".join(line.rstrip() for line in text.split("\n"))
    if "\n\n\n" in text:
        text = _MULTI_NL_RE.sub("\n\n", text)
    return text


def ensure_trailing_newline(text: str) -> str:
    if not text:
        return "\n"
    return text if text.endswith("\n") else text + "\n"


def is_mostly_text(text: str, max_binary_ratio: float = 0.01) -> bool:
    if not text:
        return False
    # Count characters that look like binary/control.
    bad = 0
    for ch in text:
        code = ord(ch)
        if code == 0:
            bad += 1
            continue
        if code < 9 or (13 < code < 32):
            bad += 1
    ratio = bad / max(len(text), 1)
    return ratio <= max_binary_ratio


def iter_jsonl(path: str) -> Iterator[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def read_jsonl(path: str) -> List[Dict]:
    return list(iter_jsonl(path))


def write_jsonl(path: str, items: Iterable[Dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


_WHITESPACE_RE = re.compile(r"\s+")


def normalize_for_dedup(text: str) -> str:
    text = normalize_text(text)
    text = _WHITESPACE_RE.sub(" ", text)
    return text.strip()


def strip_glsl_comments(text: str, strip_spaces: bool = False) -> str:
    # Remove // and /* */ comments, preserve strings and optional verbatim blocks (//[ ... //]).
    if not text:
        return text
    if "//" not in text and "/*" not in text:
        cleaned = text
    else:
        out: list[str] = []
        out_append = out.append
        i = 0
        n = len(text)
        in_block = False
        in_line = False
        in_string: str | None = None
        escape = False
        in_verbatim = False

        while i < n:
            ch = text[i]

            if in_verbatim:
                if ch == "/" and i + 2 < n and text[i + 1] == "/" and text[i + 2] == "]":
                    i += 3
                    in_verbatim = False
                    continue
                out_append(ch)
                i += 1
                continue

            if in_line:
                if ch == "\n":
                    in_line = False
                    out_append(ch)
                i += 1
                continue

            if in_block:
                if ch == "*" and i + 1 < n and text[i + 1] == "/":
                    in_block = False
                    i += 2
                    continue
                if ch == "\n":
                    out_append(ch)
                i += 1
                continue

            if in_string:
                out_append(ch)
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == in_string:
                    in_string = None
                i += 1
                continue

            if ch == "/":
                nxt = text[i + 1] if i + 1 < n else ""
                if nxt == "/" and i + 2 < n and text[i + 2] == "[":
                    in_verbatim = True
                    i += 3
                    while i < n and text[i] in (" ", "\t"):
                        i += 1
                    continue
                if nxt == "/" and i + 2 < n and text[i + 2] == "]":
                    out_append(ch)
                    out_append(nxt)
                    i += 2
                    continue
                if nxt == "/":
                    in_line = True
                    i += 2
                    continue
                if nxt == "*":
                    in_block = True
                    i += 2
                    continue
            if ch in ('"', "'"):
                in_string = ch
                out_append(ch)
                i += 1
                continue

            out_append(ch)
            i += 1

        cleaned = "".join(out)
    if not strip_spaces:
        return cleaned

    lines = cleaned.split("\n")
    stripped_lines: list[str] = []
    for line in lines:
        if not line:
            continue
        if line.lstrip().startswith("#"):
            stripped_lines.append(line.strip())
            continue
        out_line: list[str] = []
        in_str: str | None = None
        esc = False
        last_space = False
        for ch in line:
            if in_str:
                out_line.append(ch)
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == in_str:
                    in_str = None
                continue
            if ch in ('"', "'"):
                in_str = ch
                out_line.append(ch)
                last_space = False
                continue
            if ch in (" ", "\t"):
                if not last_space:
                    out_line.append(" ")
                    last_space = True
                continue
            last_space = False
            out_line.append(ch)
        collapsed = "".join(out_line).strip()
        if collapsed:
            stripped_lines.append(collapsed)
    return "\n".join(stripped_lines) + ("\n" if cleaned.endswith("\n") else "")
