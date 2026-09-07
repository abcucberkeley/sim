#!/usr/bin/env python3
"""Assert that the project version is stated identically everywhere.

`project(SIRIUS VERSION x.y.z)` in CMakeLists.txt is canonical. The two
other places carry the same literal because pip (pyproject.toml) and the
compute worker (sirius_worker.__version__) must know their version without
CMake. The CI lint job runs this script; locally:

    python tools/check_versions.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

SOURCES = {
    "CMakeLists.txt": re.compile(r"^\s*project\s*\(\s*\w+\s+VERSION\s+([0-9][0-9.]*)", re.M),
    "pyproject.toml": re.compile(r"^version\s*=\s*\"([^\"]+)\"", re.M),
    "app/python/sirius_worker/__init__.py": re.compile(r"^__version__\s*=\s*\"([^\"]+)\"", re.M),
}


def main() -> int:
    found: dict[str, str] = {}
    for rel, pattern in SOURCES.items():
        text = (ROOT / rel).read_text(encoding="utf-8")
        m = pattern.search(text)
        if not m:
            print(f"{rel}: no version found (pattern {pattern.pattern!r})", file=sys.stderr)
            return 1
        found[rel] = m.group(1)
    canonical = found["CMakeLists.txt"]
    wrong = {rel: v for rel, v in found.items() if v != canonical}
    if wrong:
        print(f"version mismatch: CMakeLists.txt says {canonical}", file=sys.stderr)
        for rel, v in wrong.items():
            print(f"  {rel}: {v}", file=sys.stderr)
        return 1
    print(f"version {canonical} agrees in {', '.join(found)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
