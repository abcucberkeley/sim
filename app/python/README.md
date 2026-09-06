# SIRIUS compute worker (`sirius_worker`)

A small TCP service that runs the parts of a pipeline that live in Python:
Torch segmentation models, and -- when it runs on a cluster node -- every
step the Python step library implements, which is how the application's
**HPC** backend works. The application launches one locally for Torch
models (`python -m sirius_worker --port 0`), reads the port it announces
and talks to it over the protocol below; on a cluster the same worker is a
Slurm job reached through an SSH tunnel (see `slurm/README.md`).

No third-party dependency is needed for the service itself: the standard
library plus `numpy`. `torch` enables `model_info` / `torch_segment` /
`seg`, `scipy` the label post-processing and resampling, the `sirius`
package SIM reconstruction.

```
python -m sirius_worker [--host 127.0.0.1] [--port 0] [--token T] [--device auto|cuda|cpu] [--log-level INFO]
```

Once listening it prints exactly one JSON line to stdout,
`{"port": 41237, "pid": 12345, "host": "127.0.0.1", "device": "cuda"}`, and
logs to stderr. `--token` (or `$SIRIUS_TOKEN`) is a shared secret the client
must present in `hello`; always set one on a shared machine.

## Where the step code lives

There is one implementation of the steps: `bindings/python/sirius/workbench.py`
(`sirius.workbench`). The worker uses the installed `sirius` package when
there is one; otherwise `sirius_worker/steps.py` loads that file directly
-- from `$SIRIUS_WORKBENCH_PY`, from a checkout or build tree found by
walking up from the worker directory (`build/<preset>/app/python` reaches
the repository root), or from a `workbench.py` copied next to the package.
Loaded that way there is no `sirius` extension, so SIM reconstruction
reports itself unavailable while the numpy and Torch steps work.

`sirius.workbench.run_pipeline(dataset_path, pipeline_json)` is also what
the application's "Export pipeline as Python script" calls.

## Protocol

Mirrors `app/core/rpc.hpp`. One frame:

```
u32 header_len (LE) | header: UTF-8 JSON | u64 payload_len (LE) | payload
```

Header fields: `id` (request id, echoed by every reply), `type`
(`request` | `progress` | `result` | `error`), `method`, `params`,
`tensors` (`[{name, dtype, shape, offset, nbytes}]` describing raw
little-endian C-order arrays concatenated in the payload), `message`,
`fraction`. dtypes: `float32 float64 uint8 int8 uint16 int16 uint32 int32
uint64 int64`.

| method | params | reply |
| --- | --- | --- |
| `hello` | `{token}` | `result`: `{version, methods, cuda, device, hostname, python, torch, sirius, workbench}` |
| `ping` | | `result`: `{time}` |
| `model_info` | `{path}` | `result`: `{format, input_shape, output_shape, dtype, size_bytes, channels_out}` |
| `run` | `{kind, params, meta?}` + tensors | `progress`\* (`{fraction, message}`) then `result` + tensors, or `error` |
| `cancel` | `{id}` | `result`: `{cancelled: id}`; the cancelled run replies `error` `"cancelled"` |
| `shutdown` | | `result` `{}` and the worker exits |

`methods` lists `run:<kind>` for every kind the worker can run, so the
application knows what to route. One run executes at a time (a second
`run` gets `error "busy"`); the reader keeps running so `cancel` is
honoured between tiles / volumes. Every `result` carries `seconds`.

### Run kinds and tensors

| kind | input tensors | output tensors | result |
| --- | --- | --- | --- |
| `torch_segment` | `input` (z, y, x) float32 | `prob` (C, z, y, x) float32 | `{channels, device}` |
| `sim` | `input` (sections, y, x) or (c, t, sections, y, x) float32 | `output` (same rank, zoomed) | `{meta, info: {fits, wiener, ...}}` |
| any other kind | `input` (c, t, z, y, x) float32, optional `labels` (t, z, y, x) uint32 | `output` (c', t', z', y', x') float32, optional `labels`, `prob` | `{meta, info}` |

`meta` in `params`/results is the dataset metadata dict of `sirius.workbench`
(`dims`, `voxel_um` [x, y, z], `channels`, `rgb`, `sim`).

### Step parameters

`params` are the step's parameters as saved by the application; the keys
below are canonical (the library accepts a few aliases, see the docstrings
in `workbench.py`).

* `torch_segment`: `model` (path on the worker's host), `tile` [z, y, x],
  `overlap` (int, or [z, y, x]), `normalize` (percentile 1..99.9 → 0..1,
  default true), `activation` (`auto` | `sigmoid` | `softmax` | `none`),
  `pad_to` (multiple the tile is padded to). TorchScript models take
  (1, 1, z, y, x) float32 and return (1, C, z, y, x); ONNX via `onnxruntime`.
* `seg` (= `torch_segment` + post-processing, labels out): the above plus
  `channel`, `threshold`, `post` (`Watershed on boundary channel` |
  `Connected components` | `None`), `fg_channel`, `boundary_channel`,
  `min_voxels`.
* `sim`: any `SIMParameters` field (`ndirs`, `nphases`, `wiener`, `na`,
  `nimm`, `wavelength_nm`, `linespacing_um`, `k0_start_angle`, `k0_angles`,
  `dx`/`dy`/`dz`, `zoomfact`, `z_zoom`, `background`, `dampen_order0`,
  `do_rescale`, ...) with the aliases `angles`, `phases`, `wavelength`,
  `linespacing`, `apodization` (`Cosine` | `Triangle` | `None` → output
  apodization), `suppress_zero`, `bleach`; `params_file` (TOML or legacy
  cudasirecon config, loaded first); `otf` (radially averaged OTF TIFF,
  required -- the theoretical OTF exists only in the application).
* `einsum`: `axes` (kept axes, e.g. `czyx`) and `reduction`
  (`sum` | `mean` | `max` | `min`); `maxproj`: `axis`; `meant`: none.
* `contrast`: `low`, `high` (percentiles), `gamma`, `per_channel`.
* `merge`: `blend` (`Additive` | `Screen` | `Max`), `colors` (`#rrggbb` per channel).
* `croppad`: `origin` [z, y, x] (negative pads), `size` [z, y, x] (0 = to the end), `fill`.
* `resample`: `voxel` [z, y, x] µm (or one isotropic value) or `factor`
  [z, y, x]; `interpolation` (`linear` | `nearest` | `cubic`).
* `bleach`: `to_mean`; `flatfield`: `flat`, `dark` (paths);
  `threshold`: `channel`, `threshold` or `percentile`, `min_voxels`.

Kinds the Python side does not implement (`decon`, `deskew`, `volrec`,
`stitch`, `register`, `label_cleanup`) are reported as unsupported; the
application runs those natively.

## Tests

```
python -m unittest discover -s app/python/tests -v      # protocol, socket, torch (skipped without torch)
python -m unittest bindings/tests/test_workbench.py -v  # run_pipeline / steps
```
