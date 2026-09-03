"""Benchmark the reusable Python SIM API without timing TIFF I/O or upload."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import sirius


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()

    params = sirius.load_legacy_parameters(str(args.data_dir / "config.txt"))
    device = sirius.Device(args.device)
    raw = sirius.read_tiff(
        str(args.data_dir / "raw.tif"), dtype=np.float64, device=device
    )
    recon = sirius.SimReconstructor(
        params, str(args.data_dir / "otf.tif"), device=device
    )

    recon.reconstruct(raw)  # plan/allocation warmup
    samples = []
    for _ in range(max(1, args.repeats)):
        start = time.perf_counter()
        output = recon.reconstruct(raw)
        samples.append(time.perf_counter() - start)
    print(
        f"sim-{args.device}: {min(samples) * 1e3:.3f} ms "
        f"(shape={output.shape}, best of {len(samples)})"
    )


if __name__ == "__main__":
    main()
