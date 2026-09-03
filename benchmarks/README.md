# SIRIUS benchmarks

Build with `SIRIUS_ENABLE_BENCHMARKS=ON`. The SIM benchmark measures one
steady-state in-memory reconstruction; file I/O, host/device upload, FFT
planning, and first-use allocation are excluded. It performs one untimed
warmup and reports the minimum of the requested repetitions to reduce scheduler
noise.

```text
bench_sim <tests/data> [repeats] [cpu|cuda|cuda:N]
```

Use a Release build, keep the machine otherwise idle, and report the CPU/GPU,
thread count, compiler, and build options with results. The bundled 64x64x9
fixture is useful for regression tracking; add representative production-sized
data when making capacity or latency decisions.

The Python entry point has matching timing boundaries:

```text
python bindings/benchmarks/bench_sim.py tests/data --device cpu --repeats 5
```

`bench_tiff_sirius` measures TIFF decoding separately so reconstruction and I/O
regressions can be attributed rather than conflated.
