"""SIRIUS compute worker.

A small TCP service the SIRIUS desktop application talks to for work that
lives in Python -- Torch segmentation models above all -- and the same
service that runs on a cluster node for the HPC backend. Protocol: see
``protocol.py`` (mirrors ``app/core/rpc.hpp``); step implementations: see
``sirius.workbench`` (located by ``steps.py``).

    python -m sirius_worker --host 127.0.0.1 --port 0 --token X --device auto
"""

# Same literal as `project(... VERSION ...)` in the top-level CMakeLists.txt, which
# is canonical, and as pyproject.toml; tools/check_versions.py keeps them in step.
__version__ = "0.1.0"

__all__ = ["__version__"]
