# SIRIUS — Structured Illumination Reconstruction and Image Utility Suite
Cross-platform SIM reconstruction tool that runs on the CPU, GPU and HPC.

## Development guide
Fiona
```
module load ninja
module load cmake
module load nvhpc
module load gcc/13.2.0
module load python
```

Configure, build and test
```
# Configure
cmake --preset fiona-avx2-dev

# Build
cmake --build --preset fiona-avx2-dev

# Test
ctest --preset fiona-avx2-dev
```

Check intrinsics supported
```
lscpu | grep -i flags | tr ' ' '\n' | grep -E "sse|avx|fma" | sort -u
```

Fiona supports avx2 so make sure to pass `-DSIRIUS_ENABLE_AVX2=ON`; the `fiona-avx2-cuda-*` presets add CUDA/nvTIFF.

## TODO
- detect/handle int overflow and use fftw_plan_guru64_dft instead of fftw_plan_many_dft
- Remove port overlay after the next nanobind release (due to missing tensor header)
- Add tensorstore
- Sanitizers.cmake only enables ASan+UBSan for non-MSVC. Add tsan/msan as separate options (mutually exclusive with ASan), and on MSVC, /fsanitize=address doesn't co-exist with /RTC1, which Debug enables by default. Worth a string(REGEX REPLACE) to strip /RTC* when sanitizers are on.
- cmake install command so the downstream user can simply do
```cmake
find_package(SIRIUS CONFIG REQUIRED)
target_link_libraries(myapp PRIVATE sirius::sirius)
```
- MPI: bind each rank to `Device::cuda(localRank % cudaDeviceCount())`, distribute pages via `TiffFile::readPages`, and add a `Buffer`-aware halo exchange
- nvTIFF encoder (`nvtiffEncode`) for writing stacks straight from device memory; the writers currently stage device buffers through the host
- GPU reads of Deflate TIFFs need `libnvcomp.so.5` at run time (fetched and linked by default); wheels should bundle it via auditwheel

## CPU / GPU execution model

Every algorithm works on `sirius::BufferView<T>` (pointer + `Shape` + `Device`),
and `sirius::Buffer<T>` owns such memory on the CPU (pageable or pinned) or on
a CUDA device. The public headers contain no CUDA types and no build macros:
a CPU-only build exposes the same API and `Device::cuda()` fails at run time
with a clear error. Query `builtWithCuda()` / `cudaAvailable()` to choose.

```cpp
#include <sirius/buffer.hpp>
#include <sirius/fft.hpp>
#include <sirius/tiff_io.hpp>
using namespace sirius;

const Device dev = cudaAvailable() ? Device::cuda(0) : Device::cpu();
Stream stream(dev);                                   // no-op on the CPU

TiffFile file("raw.tif");                             // metadata: pages, pyramid levels, tiles, codec
TiffReadOptions opts;  opts.device = dev;
Buffer<float> stack = file.readStack<float>(opts, stream);   // (pages, rows, cols), decoded on the GPU by nvTIFF

FFT fft({int(stack.dim(1)), int(stack.dim(2))}, int(stack.dim(0)), PlanRigor::Measure, dev);  // FFTW or cuFFT
Buffer<std::complex<double>> spectrum(stack.shape(), dev, HostMemory::Pageable, stream);
// ... convert / fft(view, view, stream) / ifft(..., normalize=true, stream)

auto host = toEigen<3>(stack, stream);                // ImageStack<float> on the host
copy(host, stack, stream);                            // Eigen tensors are views too
```

Design rules (see `include/sirius/buffer.hpp`):
- No implicit copies. `Buffer` is move-only; `clone()`, `to(device)`, `copy()` are explicit.
- Everything taking a `Stream` is asynchronous and stream-ordered on CUDA; pinned host
  memory (`HostMemory::Pinned`) makes host<->device copies truly asynchronous.
- Device memory comes from the per-device CUDA memory pool (`cudaMallocAsync`), so
  allocating working buffers per frame is cheap. Host memory is 64-byte aligned.
- Row-major, innermost dimension contiguous: identical to the Eigen `RowMajor` tensors
  and to TIFF scanlines, so Eigen interop is a pointer reinterpretation.

### TIFF reading (`TiffFile`)

`TiffFile` opens a file once and exposes `info()` (every IFD, the full-resolution
`pages`, and pyramid `levels` discovered from SubIFDs or reduced-resolution IFDs).
`readStack`, `readPages`, `readLevel` and `readRegion` decode into a `Buffer` on any
device; `decode` fills a caller-provided view. Strips and tiles, None/LZW/Deflate
(with horizontal predictor), classic and BigTIFF are all decoded on the GPU by nvTIFF
in one batched call per 512 MiB of pages, with pixel conversion on the device. Files
nvTIFF cannot decode (e.g. the floating-point predictor sirius itself writes for
compressed float data, or Deflate without nvCOMP) fall back to libtiff plus an
upload unless `TiffReadOptions::allowCpuFallback` is off; `gpuDecodable()` tells you
in advance. Read calls return with the data ready (they synchronize the stream).

The Eigen API (`readTiff`, `readTiffStack`, `readTiffStackAny`, `writeTiff*`) is
unchanged and implemented on top of `TiffFile`; the writers also accept `BufferView`s
on either device.

### Building with CUDA

```
cmake --preset linux-cuda-dev        # native GPU arch, tests, python bindings
cmake --build --preset linux-cuda-dev
ctest --preset linux-cuda-dev        # GPU cases SKIP when no device is usable
```

`SIRIUS_ENABLE_CUDA=ON` adds cuFFT and, by default, nvTIFF (`SIRIUS_ENABLE_NVTIFF`)
plus nvCOMP (`SIRIUS_ENABLE_NVCOMP`, needed for Deflate on the GPU). Both are NVIDIA
redistributables fetched by `cmake/NvidiaRedist.cmake` from
developer.download.nvidia.com, pinned by version and SHA256 and selected for the
toolkit's CUDA major (12 or 13). On a machine that already has them (an HPC module,
an air-gapped node) point `SIRIUS_NVTIFF_ROOT` / `SIRIUS_NVCOMP_ROOT` at the extracted
archives instead. CMake prefers the nvcc under `$CUDA_HOME`, `$CUDA_PATH` or
`/usr/local/cuda` over a distro-packaged one; set `CUDACXX` to pin it. Presets:
`linux-cuda-dev`, `linux-cuda-release` (portable arch list) and the `fiona-avx2-cuda-*`
pair for the cluster.

Container images with the full toolchain (identical for Docker locally and Apptainer on
the cluster) are in [containers/](containers/README.md).

## Masked registration and stitching

`registration.hpp` implements the masked FFT translation registration of
D. Padfield, *Masked Object Registration in the Fourier Domain*, IEEE TIP 21(5),
2012, generalized from the paper's 2D derivation to 2D **and** 3D volumes. Each
image carries a mask of the voxels that may take part in the match, and the
masking is folded into the correlation rather than applied afterwards, so the
score at every candidate displacement is the true normalized cross-correlation
of exactly the overlapping unmasked voxels -- which is what makes it usable on
tiles whose overlap strip contains saturated pixels, an illumination roll-off,
or the zero fill a deskew leaves behind.

```cpp
#include <sirius/registration.hpp>
using namespace sirius;

MaskedNccOptions opts;
opts.maxShift = {2, 40, 40};              // the stage cannot be further off than this
opts.requiredOverlapFraction = 0.25;      // reject displacements that barely touch

TranslationResult t = registerTranslationMasked<float>(fixed, moving, fixedMask, movingMask, opts);
// moving[p] matches fixed[p + t.shift]; t.shift is sub-voxel, t.integerShift is not
```

`maskedNormalizedCrossCorrelation` returns the whole map (coefficients plus the
per-displacement overlap counts) when you want to inspect it, and
`MaskedCorrelator` keeps the plans and padded buffers alive across many pairs of
the same size. The cost is 6 forward and 6 inverse **real** FFTs of the volume
padded to the next 2/3/5/7-smooth size (`nextFastFFTSize`), batched into two
planned transforms, so it does not grow with the size of the search range. All
of it is double precision: the algorithm forms differences of large, nearly
equal sums and is not usable in single precision.

`stitching.hpp` builds a mosaic on top of it in three steps -- pairwise
registration of every nominally overlapping tile pair (only the overlap strip is
correlated, grown by the search radius), a global least-squares fit of the tile
origins from all the pairwise displacements at once (one sparse Cholesky
factorization shared by the three axes, anchored on one tile), and a blended
fusion pass (`Overwrite`, `Average`, `Feather`, `Maximum`).

```cpp
#include <sirius/stitching.hpp>

StitchOptions options;
options.searchRadius = {2, 64, 64};
options.maskBackground = true;            // ignore the deskew fill when correlating
options.blend = BlendMode::Feather;

StitchLayout layout;
Buffer<float> mosaic = stitchTiffTiles<float>(
    {{"tile0.tif", {0, 0, 0}}, {"tile1.tif", {0, 0, 1800}}}, options, &layout, "mosaic.tif");
for (const TileMatch& m : layout.matches)
    std::printf("%zu->%zu  r=%.3f  dx=%.2f
", m.fixed, m.moving, m.correlation, m.displacement[2]);
```

From Python:

```python
import numpy as np, sirius

r = sirius.register_translation_masked(fixed, moving, moving_mask=mask)
r.shift, r.correlation, r.valid

options = sirius.StitchOptions()
options.search_radius = [2, 64, 64]
mosaic, layout = sirius.stitch_tiff_tiles(
    ["tile0.tif", "tile1.tif"], [(0, 0, 0), (0, 0, 1800)], options, output_path="mosaic.tif")
layout.positions      # refined tile origins, (z, y, x) voxels
```

Scope, relative to [PetaKit5D](https://github.com/abcucberkeley/PetaKit5D): what is
here is the registration and the translation-only mosaic (planning, global fit,
fusion, TIFF in and out). Not here yet: deskew/rotate of the raw light-sheet
frames, flat-field correction, Zarr and the large-scale out-of-core paths,
deconvolution, cross-channel and multi-round registration, and the cluster job
distribution. `stitchTiffTiles` holds every tile and the canvas in memory;
larger mosaics need `planStitch`/`fuseTiles` driven over a tile-at-a-time
reader. Tiles are placed on the voxel grid (positions are rounded), so
sub-voxel placement would need a resampling step that does not exist yet.

## Desktop application (`app/`)

`sirius-app` is a Qt Widgets front end for the reconstruction pipeline: load a raw
stack, optionally an OTF, and a parameter file (TOML or legacy cudasirecon `key=value`),
edit the parameters, reconstruct on the CPU or any CUDA device, inspect the volumes,
read off the fitted pattern vectors and save the result as a float32 TIFF.
Reconstructions run on a worker thread; the `SimReconstructor` (FFT plans) and the
device copy of the raw stack are kept between runs and only rebuilt when the
parameters, OTF or device change.

- **OTF is optional.** Without a file the theoretical OTF of an aberration-free objective
  is computed from NA, immersion index and emission wavelength (`sirius::idealOTF`; 3D
  with the missing cone when the stack has several z planes, in-focus 2D otherwise).
  The `Ideal` button next to the OTF drops a loaded file again.
- **Viewers** (Raw, OTF, Reconstruction, crops): mouse-wheel zoom around the cursor,
  drag to pan, `Fit`/`1:1`, min/max contrast sliders with `Auto` (percentiles) and
  `Reset`, `Log` display, `Select` + `Crop` to open a rectangle (all slices) in a new
  closable tab, `Ortho` for XZ/YZ views through a click-positioned crosshair
  (`Physical z` scales them by dz/dx), and `Spectrum` to show the centered |FFT| of the
  displayed planes. In spectrum mode the raw and result views overlay the OTF support
  circle (2NA/λ), the pattern vectors predicted from the parameters (yellow crosses)
  and, after a run, the fitted ones with their modulation amplitudes (cyan circles).
- **OTF tab**: one order of the OTF (loaded or ideal) resampled onto the stack's grid,
  exactly as the reconstruction interpolates it; step through kz with the slice slider.
- **Bands tab**: with `Capture intermediate spectra` ticked, a run keeps the separated
  band spectra and their Wiener-filtered versions (`SimReconstructor::setCaptureDiagnostics`),
  browsable by direction, order (± side bands) and stage. Off by default: it holds two
  host copies of every band volume.

```
export SIRIUS_QT_DIR=/path/to/Qt/6.x/gcc_64        # any Qt 6 or Qt 5.15 prefix
cmake --preset linux-gcc-app-dev
cmake --build --preset linux-gcc-app-dev
ctest --preset linux-gcc-app-dev                    # includes the app core tests
./build/linux-gcc-app-dev/app/sirius-app --raw raw.tif [--otf otf.tif] --params config.txt [--reconstruct]
```

`SIRIUS_ENABLE_APP=ON` locates Qt with `find_package` (Qt is a system dependency
like the CUDA toolkit, not something FetchContent builds); the `*-app-*` presets
take the prefix from `$SIRIUS_QT_DIR`, or pass `-DQt6_DIR=.../lib/cmake/Qt6`
(or `Qt5_DIR`) for a Qt that lives inside a larger prefix such as a conda
environment. The layout separates a Qt-free model (`app/core`: `ReconSession`,
parameter-format detection, display mapping) that `tests/test_app_core.cpp`
covers without a display from the Widgets layer (`app/qt`). On Windows,
`windeployqt` copies the Qt runtime next to the executable after every link
(`SIRIUS_APP_DEPLOY_QT`; turn it off for Qt builds it cannot process, such as
conda-forge's renamed `Qt5*_conda.dll`, and run with the Qt `bin` directory on
`PATH` instead). Add `-DSIRIUS_ENABLE_APP=ON` to any CUDA preset to get the GPU
devices in the device list.

## Python Bindings
Dev install
```
pip install -e .
```

On fiona or any computer with avx2 support
```
pip install -e . --config-settings cmake.args="-DSIRIUS_ENABLE_AVX2=ON"
```

Run unit tests
```
python -m unittest discover -s bindings/tests
```

GPU reads return a `sirius.Buffer` that implements DLPack, so PyTorch/CuPy adopt
the device memory without a copy:
```python
import sirius, torch
f = sirius.TiffFile("raw.tif")
f.info.shape, f.info.level_count            # (pages, height, width), pyramid levels
stack = f.read_stack(device="cuda", dtype="float32")   # nvTIFF decode into GPU memory
t = torch.from_dlpack(stack)                # zero-copy
region = f.read_region(x, y, w, h, level=1) # numpy on the CPU
```
`sirius.read_tiff(path)` keeps returning numpy; pass `device="cuda"` for a Buffer.

The planned FFT runs on either device too: `sirius.FFT(dims, device="cuda")` takes
complex128 arrays that export DLPack (`sirius.Buffer`, torch, cupy) and returns a
`sirius.Buffer`; `f.fft(x, out=x)` transforms in place on both devices.

Build the extension with the GPU paths enabled (editable or wheel):
```
pip install -e . --config-settings=cmake.define.SIRIUS_ENABLE_CUDA=ON \
                 --config-settings=cmake.define.CMAKE_CUDA_ARCHITECTURES=native
```
The wheel bundles `libnvtiff.so.0` / `libnvcomp.so.5` next to the extension (found
via `$ORIGIN`); cuFFT and the CUDA runtime are resolved from the toolkit that built it.

## Benchmarks
The TIFF benchmark compares the SIRIUS parallel reader against
[cpp-tiff](https://github.com/abcucberkeley/cpp-tiff) at both the C++ and
Python-binding levels; `bench_fft.py` compares the FFT against NumPy.

Setup — extra deps plus the two C++ benchmark binaries. On CMake 4.x the
`CMAKE_POLICY_VERSION_MINIMUM=3.5` flag is required: FFTW/libtiff/zlib still
declare pre-3.5 minimums that CMake 4 rejects outright.
```
pip install numpy tifffile imagecodecs cpp-tiff "cmake>=3.25" ninja
pip install -e . --config-settings=cmake.define.CMAKE_POLICY_VERSION_MINIMUM=3.5

# SIRIUS C++ bench (built by SIRIUS's CMake)
cmake --preset linux-gcc-release -DSIRIUS_ENABLE_BENCHMARKS=ON -DCMAKE_POLICY_VERSION_MINIMUM=3.5
cmake --build --preset linux-gcc-release --target bench_tiff_sirius

# GPU variant: same binary from a CUDA preset, third argument selects the device
cmake --preset linux-cuda-release -DSIRIUS_ENABLE_BENCHMARKS=ON
cmake --build --preset linux-cuda-release --target bench_tiff_sirius
build/linux-cuda-release/benchmarks/bench_tiff_sirius stack.tif 3 cuda   # nvTIFF decode into device memory
build/linux-cuda-release/benchmarks/bench_tiff_sirius stack.tif 3 cpu

# cpp-tiff C++ bench (standalone; clones latest cpp-tiff, builds libcppTiff.so)
bash benchmarks/setup_cpptiff.sh
```

Run the full ~18 GB TIFF case (all four readers; the dataset is written to the
gitignored `./.bench_tmp/` and deleted afterwards):
```
python bindings/benchmarks/bench_tiff.py --shape 10000 1800 512 --repeats 3 \
    --cpp-sirius  build/linux-gcc-release/benchmarks/bench_tiff_sirius \
    --cpp-cpptiff .bench_tmp/cpptiff/bench_tiff_cpptiff
```

Quick correctness check on a tiny file. `--verify` reads the dataset with every
Python reader (`sirius.read_tiff`, `cpptiff.read_tiff`) and asserts the arrays
are bit-for-bit identical before the timed runs, raising on any shape/data
mismatch (lossless for every supported compression). The C++ benches only report
timing, so they are not part of this cross-check; it needs both Python readers
importable, otherwise it warns and skips.
```
python bindings/benchmarks/bench_tiff.py --shape 8 64 64 --verify
```
`--keep` retains the dataset, `--path P` uses an explicit file, and
`--dtype` / `--compression` vary the data (`imagecodecs` is required for
compressed datasets such as `--compression lzw`); see `--help` for all options.

FFT vs NumPy:
```
python bindings/benchmarks/bench_fft.py
```
## Python worker and HPC backend

`app/python/sirius_worker` is a small TCP service the application uses for
work that lives in Python -- Torch segmentation models locally -- and the
same service that serves the **HPC** backend from a cluster node. It needs
only the standard library and `numpy`; `torch` enables models, `scipy` the
label post-processing and resampling, and the `sirius` wheel SIM
reconstruction on the node. The protocol (length-prefixed JSON headers plus
raw tensors, mirrored by `app/core/rpc.hpp`), the run kinds and their
parameter keys are documented in [app/python/README.md](app/python/README.md).

```
python -m sirius_worker --host 127.0.0.1 --port 0 --token X --device auto   # prints {"port": N, ...}
python -m unittest discover -s app/python/tests -v
```

Every step the worker can run is implemented once, in
`sirius.workbench` (`bindings/python/sirius/workbench.py`); the worker
imports the installed package or loads that file from the checkout / build
tree. `sirius.workbench.run_pipeline(dataset, pipeline_json)` is what the
application's "Export pipeline as Python script" produces a call to: it
loads a TIFF / OME-TIFF / zarr dataset as `(c, t, z, y, x)` float32 and
runs the numpy, SIM (via the bindings) and Torch steps, raising
`NotImplementedError` for the steps only the C++ application implements
(deconvolution, deskew, volume rendering, stitching, registration).

On a cluster, submit [app/python/slurm/sirius_worker.sbatch](app/python/slurm/sirius_worker.sbatch)
with `SIRIUS_TOKEN` set, tunnel the port (`ssh -N -L 7645:<node>:7645 <login-node>`)
and enter host, port and token under Preferences ▸ HPC; see
[app/python/slurm/README.md](app/python/slurm/README.md).
