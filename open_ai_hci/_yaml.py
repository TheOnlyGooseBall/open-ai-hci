"""Tiny YAML reader/writer for ``config.yaml`` so the GUI works without PyYAML.

PyYAML is used when it is installed. Otherwise this module handles the subset that
:meth:`HCIConfig.save` produces: nested mappings, scalars (``str``, ``int``, ``float``,
``bool``, ``null``) and flow lists/maps written as JSON (``["q"]``, ``{}``), plus ``#``
comments and blank lines. Anything fancier (anchors, block lists of mappings, multi-line
strings) raises ``ValueError`` and asks for PyYAML.
"""
from __future__ import annotations

import json
import re
from typing import Any

from ._optional import is_available

__all__ = ["dump_yaml", "load_yaml"]

_PLAIN = re.compile(r"^[A-Za-z_][A-Za-z0-9_./-]*$")
_RESERVED = {"true", "false", "null", "yes", "no", "on", "off", "~", ""}


def _scalar(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, str):
        if _PLAIN.match(value) and value.lower() not in _RESERVED:
            return value
        return json.dumps(value, ensure_ascii=False)
    return json.dumps(value, ensure_ascii=False)  # list / tuple -> flow sequence


def dump_yaml(data: dict[str, Any]) -> str:
    """Serialise a nested dict (uses PyYAML when available)."""
    if is_available("yaml"):
        import yaml

        return yaml.safe_dump(data, sort_keys=False, allow_unicode=True)
    lines: list[str] = []

    def walk(mapping: dict[str, Any], indent: int) -> None:
        pad = "  " * indent
        for raw_key, value in mapping.items():
            key = _scalar(str(raw_key))
            if isinstance(value, dict) and value:
                lines.append(f"{pad}{key}:")
                walk(value, indent + 1)
            elif isinstance(value, dict):
                lines.append(f"{pad}{key}: {{}}")
            else:
                lines.append(f"{pad}{key}: {_scalar(list(value) if isinstance(value, tuple) else value)}")

    walk(data, 0)
    return "\n".join(lines) + "\n"


def _strip_comment(line: str) -> str:
    out, quote = [], ""
    for ch in line:
        if quote:
            if ch == quote:
                quote = ""
        elif ch in "\"'":
            quote = ch
        elif ch == "#" and (not out or out[-1] in " \t"):
            break
        out.append(ch)
    return "".join(out).rstrip()


def _parse_scalar(text: str) -> Any:
    t = text.strip()
    if t in ("null", "~", "Null", "NULL"):
        return None
    if t.lower() in ("true", "yes", "on"):
        return True
    if t.lower() in ("false", "no", "off"):
        return False
    if t[:1] in "[{":
        try:
            return json.loads(t)
        except ValueError:
            try:
                return json.loads(t.replace("'", '"'))
            except ValueError as exc:
                raise ValueError(f"Unsupported YAML flow value {t!r}; install PyYAML for full YAML support") from exc
    if t[:1] == '"' and t[-1:] == '"' and len(t) >= 2:
        return json.loads(t)
    if t[:1] == "'" and t[-1:] == "'" and len(t) >= 2:
        return t[1:-1].replace("''", "'")
    try:
        return int(t)
    except ValueError:
        pass
    try:
        return float(t)
    except ValueError:
        return t


def load_yaml(text: str) -> dict[str, Any]:
    """Parse YAML text into a dict (uses PyYAML when available)."""
    if is_available("yaml"):
        import yaml

        return yaml.safe_load(text) or {}
    root: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(-1, root)]
    for number, raw in enumerate(text.splitlines(), 1):
        line = _strip_comment(raw)
        if not line.strip():
            continue
        if "\t" in line[: len(line) - len(line.lstrip())]:
            raise ValueError(f"line {number}: tabs are not allowed for indentation")
        indent = len(line) - len(line.lstrip(" "))
        body = line.strip()
        if body.startswith("- "):
            raise ValueError(f"line {number}: block lists are not supported by the built-in reader; install PyYAML")
        key, sep, rest = body.partition(":")
        if not sep:
            raise ValueError(f"line {number}: expected 'key: value', got {body!r}")
        key = key.strip().strip("\"'")
        while stack and indent <= stack[-1][0]:
            stack.pop()
        if not stack:
            raise ValueError(f"line {number}: bad indentation")
        parent = stack[-1][1]
        rest = rest.strip()
        if rest == "":
            child: dict[str, Any] = {}
            parent[key] = child
            stack.append((indent, child))
        else:
            parent[key] = _parse_scalar(rest)
    return root
