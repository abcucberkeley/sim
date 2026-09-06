"""Locates the step implementations (``sirius.workbench``).

There is one copy of the step code, in ``bindings/python/sirius/workbench.py``.
The worker uses it through the installed ``sirius`` package when there is
one (``pip install -e .`` or a wheel); otherwise -- a bare interpreter with
just numpy, the build tree, a cluster node without the wheel -- it loads that
file directly, found via ``$SIRIUS_WORKBENCH_PY`` or by walking up from this
directory to a checkout / build tree containing
``bindings/python/sirius/workbench.py``. Loaded that way the module has no
``sirius`` extension, so SIM reconstruction reports itself as unavailable
while every numpy / torch step works.
"""

from __future__ import annotations

import importlib
import importlib.util
import os
import sys
import types
from typing import Optional

_REL = os.path.join("bindings", "python", "sirius", "workbench.py")


def _candidate_files() -> list:
    out = []
    env = os.environ.get("SIRIUS_WORKBENCH_PY")
    if env:
        out.append(env)
    here = os.path.dirname(os.path.abspath(__file__))
    d = here
    for _ in range(8):
        out.append(os.path.join(d, _REL))
        out.append(os.path.join(d, "workbench.py"))   # a copy placed next to the worker
        parent = os.path.dirname(d)
        if parent == d:
            break
        d = parent
    return out


def load_workbench() -> types.ModuleType:
    """The ``sirius.workbench`` module, imported or loaded from a file."""
    try:
        return importlib.import_module("sirius.workbench")
    except Exception:  # noqa: BLE001 - the package may be absent or broken; fall back to the file
        pass
    existing = sys.modules.get("sirius_workbench_file")
    if existing is not None:
        return existing
    for path in _candidate_files():
        if os.path.isfile(path):
            spec = importlib.util.spec_from_file_location("sirius_workbench_file", path)
            if spec is None or spec.loader is None:
                continue
            module = importlib.util.module_from_spec(spec)
            sys.modules["sirius_workbench_file"] = module
            spec.loader.exec_module(module)
            module.__dict__.setdefault("__source_file__", path)
            return module
    raise ImportError(
        "cannot find the SIRIUS step library: install the sirius package, set SIRIUS_WORKBENCH_PY to "
        "bindings/python/sirius/workbench.py, or run the worker from a checkout / build tree")


_workbench: Optional[types.ModuleType] = None


def workbench() -> types.ModuleType:
    global _workbench
    if _workbench is None:
        _workbench = load_workbench()
    return _workbench
