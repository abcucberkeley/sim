import os
from pathlib import Path

if os.name == "nt":
    os.add_dll_directory(str(Path(__file__).parent))

from sirius._sirius_ext import PlanRigor, FFT
