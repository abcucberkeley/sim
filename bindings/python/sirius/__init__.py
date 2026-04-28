import os
from pathlib import Path

if os.name == "nt":
    os.add_dll_directory(str(Path(__file__).parent))

from sirius._sirius_ext import PlanRigor, FFT
from sirius._sirius_ext import TiffCompression
from sirius._sirius_ext import read_tiff, write_tiff

__all__ = [
    "PlanRigor",
    "FFT",
    "TiffCompression",
    "read_tiff",
    "write_tiff",
]