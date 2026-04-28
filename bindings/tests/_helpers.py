"""Shared helpers for the sirius binding tests."""

from __future__ import annotations

import contextlib
import os


@contextlib.contextmanager
def silenced_stderr():
    """Suppress C-level stderr writes for the duration of the block.

    libtiff and FFTW print diagnostics straight to fd 2 (e.g. when we
    deliberately ask them to open a missing file). Python's
    `sys.stderr` redirection does not catch these because they bypass
    the Python layer, so we duplicate fd 2 to /dev/null instead.
    """
    saved = os.dup(2)
    devnull = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull, 2)
        yield
    finally:
        os.dup2(saved, 2)
        os.close(devnull)
        os.close(saved)
