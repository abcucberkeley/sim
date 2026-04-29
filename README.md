# SIRIUS — Structured Illumination Reconstruction and Image Utility Suite
Cross-platform SIM reconstruction tool that runs on the CPU, GPU and HPC.

## Development guide
TODO

## TODO
- detect/handle int overflow and use fftw_plan_guru64_dft instead of fftw_plan_many_dft
- Remove port overlay after the next nanobind release (due to missing tensor header)
- Add tensorstore 

## Python Bindings
Dev install
```
pip install -e .
```

Run unit tests
```
python -m unittest discover -s bindings/tests
```