"""Benchmark sirius.FFT vs numpy.fft.

Measures per-call time for forward and inverse transforms across a grid of
shapes and ranks, for three execution models:

    numpy           np.fft.fft / fft2 / fftn  (always allocates output)
    sirius alloc    f.fft(x)                  (sirius allocates output)
    sirius inplace  f.fft(x, out=y)           (no per-call allocation)

Methodology
-----------
* Planning cost is excluded — the `sirius.FFT` instance is constructed once
  per case before timing starts.
* First call warms caches (FFTW touches its plan, numpy touches its pocketfft
  cache), then the timer auto-sizes `number` so each repeat takes ~0.1s.
* We report the **minimum** of `repeats` runs. Min filters out OS scheduling
  noise better than mean/median for a benchmark like this.
* Inputs are fixed complex128 arrays. Both libraries see the same data.

Caveats
-------
* Both backends run single-threaded (sirius's FFTW build doesn't call
  `fftw_init_threads`; numpy's pocketfft is single-threaded by default).
  If you enable FFTW threading in sirius, pin `OMP_NUM_THREADS=1` here for
  an apples-to-apples comparison or report both.
* PlanRigor is `Measure` by default (matches production). Use `--rigor Patient`
  for the best sirius numbers at the cost of longer planning.

Usage
-----
    python bindings/benchmarks/bench_fft.py
    python bindings/benchmarks/bench_fft.py --rigor Patient --repeats 7
    python bindings/benchmarks/bench_fft.py --wisdom ~/.sirius_wisdom
"""
from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass
from typing import Callable

import numpy as np

import sirius


# --------------------------------------------------------------------------- #
# Timing core                                                                 #
# --------------------------------------------------------------------------- #

def _autorange(fn: Callable[[], None], target_sec: float = 0.1) -> int:
    """Find `number` so a batch of `number` calls takes >= target_sec."""
    number = 1
    while True:
        t0 = time.perf_counter()
        for _ in range(number):
            fn()
        if time.perf_counter() - t0 >= target_sec:
            return number
        number *= 10


def bench(fn: Callable[[], None], *, repeats: int = 5) -> float:
    """Return the minimum per-call wall time in seconds."""
    fn()  # warm-up
    number = _autorange(fn)
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        for _ in range(number):
            fn()
        times.append((time.perf_counter() - t0) / number)
    return min(times)


def fmt_time(t: float) -> str:
    for scale, unit in ((1e-9, "ns"), (1e-6, "us"), (1e-3, "ms"), (1.0, "s")):
        if t < scale * 1000:
            return f"{t / scale:7.2f} {unit}"
    return f"{t:7.2f}  s"


# --------------------------------------------------------------------------- #
# Cases                                                                       #
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Case:
    rank: int
    shape: tuple[int, ...]

    @property
    def label(self) -> str:
        return f"{self.rank}D {' x '.join(str(s) for s in self.shape)}"

    @property
    def n_elements(self) -> int:
        n = 1
        for s in self.shape:
            n *= s
        return n


DEFAULT_CASES: tuple[Case, ...] = (
    Case(1, (256,)),
    Case(1, (4096,)),
    Case(1, (65_536,)),
    Case(1, (1 << 20,)),              # 1 M
    Case(2, (64, 64)),
    Case(2, (256, 256)),
    Case(2, (1024, 1024)),
    Case(3, (16, 64, 64)),
    Case(3, (64, 64, 64)),
    Case(3, (32, 128, 128)),
)


_NUMPY_FWD = {1: np.fft.fft, 2: np.fft.fft2, 3: np.fft.fftn}
_NUMPY_INV = {1: np.fft.ifft, 2: np.fft.ifft2, 3: np.fft.ifftn}


# --------------------------------------------------------------------------- #
# Runner                                                                      #
# --------------------------------------------------------------------------- #

@dataclass
class Row:
    label: str
    direction: str
    numpy_s: float
    sirius_alloc_s: float
    sirius_inplace_s: float

    @property
    def speedup_alloc(self) -> float:
        return self.numpy_s / self.sirius_alloc_s

    @property
    def speedup_inplace(self) -> float:
        return self.numpy_s / self.sirius_inplace_s


def run_case(case: Case, rigor: sirius.PlanRigor, repeats: int) -> list[Row]:
    rng = np.random.default_rng(0xC0FFEE ^ case.n_elements)
    x = (rng.standard_normal(case.shape)
         + 1j * rng.standard_normal(case.shape)).astype(np.complex128)
    y = np.empty_like(x)

    fft = sirius.FFT(list(case.shape), rigor=rigor)
    np_fwd = _NUMPY_FWD[case.rank]
    np_inv = _NUMPY_INV[case.rank]

    # Fresh buffer for inverse so forward output is stable across reps
    y_ref = fft.fft(x)
    x_back = np.empty_like(x)

    return [
        Row(
            case.label, "fft",
            numpy_s=bench(lambda: np_fwd(x), repeats=repeats),
            sirius_alloc_s=bench(lambda: fft.fft(x), repeats=repeats),
            sirius_inplace_s=bench(lambda: fft.fft(x, y), repeats=repeats),
        ),
        Row(
            case.label, "ifft",
            numpy_s=bench(lambda: np_inv(y_ref), repeats=repeats),
            sirius_alloc_s=bench(lambda: fft.ifft(y_ref, normalize=True),
                                 repeats=repeats),
            sirius_inplace_s=bench(lambda: fft.ifft(y_ref, x_back, normalize=True),
                                   repeats=repeats),
        ),
    ]


# --------------------------------------------------------------------------- #
# Presentation                                                                #
# --------------------------------------------------------------------------- #

def print_table(rows: list[Row]) -> None:
    header = (f"{'shape':<20} {'dir':<5} "
              f"{'numpy':>11} {'sirius alloc':>14} {'sirius inplace':>16} "
              f"{'alloc x':>8} {'inplace x':>10}")
    print(header)
    print("-" * len(header))
    for r in rows:
        print(f"{r.label:<20} {r.direction:<5} "
              f"{fmt_time(r.numpy_s):>11} "
              f"{fmt_time(r.sirius_alloc_s):>14} "
              f"{fmt_time(r.sirius_inplace_s):>16} "
              f"{r.speedup_alloc:>7.2f}x "
              f"{r.speedup_inplace:>9.2f}x")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--rigor", default="Measure",
                   choices=["Estimate", "Measure", "Patient", "Exhaustive"],
                   help="FFTW planning rigor (default: Measure)")
    p.add_argument("--repeats", type=int, default=5,
                   help="number of timing repeats; best is reported (default: 5)")
    p.add_argument("--wisdom", type=str, default=None,
                   help="load FFTW wisdom from this path before planning, "
                        "save it back after")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rigor = getattr(sirius.PlanRigor, args.rigor)

    if args.wisdom:
        path = os.path.expanduser(args.wisdom)
        sirius.FFT.load_wisdom(path)

    print(f"PlanRigor = {args.rigor}, repeats = {args.repeats}, "
          f"numpy = {np.__version__}")
    print()

    all_rows: list[Row] = []
    for case in DEFAULT_CASES:
        print(f"{case.label:<20} ...", end=" ", flush=True)
        t0 = time.perf_counter()
        rows = run_case(case, rigor=rigor, repeats=args.repeats)
        print(f"done in {time.perf_counter() - t0:.2f}s "
              f"(plan + warmup + timing)")
        all_rows.extend(rows)

    print()
    print_table(all_rows)

    if args.wisdom:
        sirius.FFT.save_wisdom(os.path.expanduser(args.wisdom))


if __name__ == "__main__":
    main()
