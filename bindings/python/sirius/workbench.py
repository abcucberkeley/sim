"""Pipeline steps of the SIRIUS workbench, in numpy.

This module is the single implementation behind two entry points:

* ``run_pipeline(dataset_path, pipeline)`` -- what the desktop app's
  "Export pipeline as Python script" calls: it loads the dataset and runs
  the saved pipeline (the app's ``.sirius.toml`` converted to JSON) step by
  step, returning the final ``(c, t, z, y, x)`` float32 array and its
  metadata.
* the compute worker (``app/python/sirius_worker``) that the app's HPC
  backend talks to: every ``run`` request maps onto :func:`run_step` here.
  The worker imports ``sirius.workbench`` when the ``sirius`` package is
  installed and otherwise loads this file straight from the source tree, so
  there is exactly one copy of the step code.

Arrays are always ``(c, t, z, y, x)`` float32 (x fastest), labels are
``(t, z, y, x)`` uint32 with 0 = background. Only numpy is required;
``scipy`` (resampling, connected components, distance transforms),
``torch`` (segmentation models), ``tifffile`` / ``zarr`` (loaders) and the
``sirius`` extension (SIM reconstruction, GPU TIFF decode) are used when
importable and reported as missing otherwise.

Parameter keys follow the operation definitions in ``app/core/ops``; the
``_get`` helper accepts the aliases listed with each step so pipelines saved
by older app versions keep running.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "AXES",
    "Cancelled",
    "NotAvailable",
    "StepResult",
    "load_dataset",
    "load_model",
    "model_info",
    "resolve_device",
    "run_pipeline",
    "run_step",
    "step_kinds",
    "tiled_inference",
]

AXES = "ctzyx"

ProgressFn = Optional[Callable[[float, str], None]]
CancelFn = Optional[Callable[[], bool]]


class NotAvailable(NotImplementedError):
    """A step cannot run here: a dependency is missing or the kind is not
    implemented in Python. The message names the step and what is missing."""


class Cancelled(RuntimeError):
    """Raised inside a step when the caller's cancel callback fired."""


# --------------------------------------------------------------------------
# parameter helpers
# --------------------------------------------------------------------------


def _get(params: Dict[str, Any], keys: Sequence[str], default: Any = None) -> Any:
    for k in keys:
        if k in params and params[k] is not None:
            return params[k]
    return default


def _float(params, keys, default: float) -> float:
    v = _get(params, keys, default)
    try:
        return float(v)
    except (TypeError, ValueError):
        return float(default)


def _int(params, keys, default: int) -> int:
    v = _get(params, keys, default)
    try:
        return int(round(float(v)))
    except (TypeError, ValueError):
        return int(default)


def _bool(params, keys, default: bool) -> bool:
    v = _get(params, keys, default)
    if isinstance(v, str):
        return v.strip().lower() in ("1", "true", "on", "yes")
    return bool(v)


def _str(params, keys, default: str = "") -> str:
    v = _get(params, keys, default)
    return "" if v is None else str(v)


def _list(params, keys, n: int, default: Sequence[float]) -> List[float]:
    v = _get(params, keys, None)
    if v is None:
        return [float(d) for d in default]
    if isinstance(v, str):
        parts = [p for p in v.replace("×", ",").replace("x", ",").replace(";", ",").split(",") if p.strip()]
        try:
            v = [float(p) for p in parts]
        except ValueError:
            return [float(d) for d in default]
    if isinstance(v, (int, float)):
        v = [float(v)] * n
    v = [float(x) for x in v]
    if len(v) < n:
        v = v + [float(d) for d in default[len(v):]]
    return v[:n]


def _choice(value: Any, options: Sequence[str], default: str) -> str:
    """Case-insensitive match of `value` against `options`, accepting prefixes
    ("cos" -> "Cosine")."""
    if value is None:
        return default
    s = str(value).strip().lower()
    if not s:
        return default
    for o in options:
        if o.lower() == s:
            return o
    for o in options:
        if o.lower().startswith(s) or s.startswith(o.lower()):
            return o
    return default


def _progress(progress: ProgressFn, fraction: float, message: str = "") -> None:
    if progress is not None:
        progress(max(0.0, min(1.0, float(fraction))), message)


def _check_cancel(cancelled: CancelFn) -> None:
    if cancelled is not None and cancelled():
        raise Cancelled("cancelled")


def _as5(a: np.ndarray) -> np.ndarray:
    """Any rank <= 5 array as (c, t, z, y, x) float32 (missing leading axes are 1)."""
    a = np.asarray(a)
    if a.ndim > 5:
        raise ValueError(f"expected at most 5 dimensions, got shape {a.shape}")
    while a.ndim < 5:
        a = a[np.newaxis]
    return np.ascontiguousarray(a, dtype=np.float32)


def _dims(a: np.ndarray) -> Dict[str, int]:
    return dict(zip(AXES, (int(n) for n in a.shape)))


def _channel_index(params, keys, meta: Dict[str, Any], c: int, default: int = 0) -> int:
    """A channel given as an index or as a label / wavelength string."""
    v = _get(params, keys, default)
    if isinstance(v, str):
        s = v.strip().lower()
        for i, ch in enumerate(meta.get("channels", [])[:c]):
            label = str(ch.get("label", "")).lower()
            nm = ch.get("wavelength_nm", 0)
            if s == label or (nm and s.startswith(str(int(nm)))) or (s and label.startswith(s)):
                return i
        try:
            v = int(float(s))
        except ValueError:
            v = default
    idx = int(v)
    if idx < 0 or idx >= c:
        raise ValueError(f"channel {idx} does not exist (input has {c})")
    return idx


# --------------------------------------------------------------------------
# metadata and loading
# --------------------------------------------------------------------------


def _default_meta(a: np.ndarray, source: str = "", fmt: str = "memory") -> Dict[str, Any]:
    c = int(a.shape[0])
    return {
        "name": os.path.splitext(os.path.basename(source.rstrip("/\\")))[0] if source else "array",
        "source": source,
        "format": fmt,
        "dims": _dims(a),
        "voxel_um": [0.1, 0.1, 0.2],  # x, y, z
        "frame_interval_s": 0.0,
        "channels": [{"label": f"ch {i}", "wavelength_nm": 0.0, "color": "#ffffff"} for i in range(c)],
        "rgb": False,
        "sim": {"present": False, "ndirs": 3, "nphases": 5, "fast_si": False},
    }


def _reorder_to_ctzyx(a: np.ndarray, axes: str) -> np.ndarray:
    """Permute an array whose axes are named by `axes` (letters of "ctzyx";
    other letters are folded into t) into (c, t, z, y, x)."""
    axes = axes.lower()
    a = np.asarray(a)
    if len(axes) != a.ndim:
        raise ValueError(f"axes '{axes}' do not match shape {a.shape}")
    known = [ax if ax in AXES else "t" for ax in axes]
    order = [[i for i, k in enumerate(known) if k == ax] for ax in AXES]
    perm = [i for idx in order for i in idx]
    out = np.transpose(a, perm)
    shape = []
    for idx in order:
        n = 1
        for i in idx:
            n *= a.shape[i]
        shape.append(n)
    return out.reshape(shape)


def _tiff_metadata(path: str) -> Tuple[Optional[str], Dict[str, Any]]:
    """(axes string of the stored series, metadata dict) via tifffile, or (None, {})."""
    try:
        import tifffile  # type: ignore
    except ImportError:
        return None, {}
    info: Dict[str, Any] = {}
    with tifffile.TiffFile(path) as tf:
        series = tf.series[0] if tf.series else None
        axes = series.axes if series is not None else None
        if tf.is_ome:
            info["format"] = "ome-tiff"
            try:
                info.update(_parse_ome_xml(tf.ome_metadata or ""))
            except Exception:  # noqa: BLE001 - metadata is best effort
                pass
        elif tf.is_imagej:
            info["format"] = "tiff"
            ij = tf.imagej_metadata or {}
            voxel = [0.0, 0.0, 0.0]
            if "spacing" in ij:
                voxel[2] = float(ij["spacing"])
            if "finterval" in ij:
                info["frame_interval_s"] = float(ij["finterval"])
            page = tf.pages[0]
            xres = page.tags.get("XResolution")
            yres = page.tags.get("YResolution")
            unit = str(ij.get("unit") or "").lower()
            if xres is not None and yres is not None and unit in ("micron", "um", "µm", "μm"):
                if xres.value[0] and yres.value[0]:
                    voxel[0] = xres.value[1] / xres.value[0]
                    voxel[1] = yres.value[1] / yres.value[0]
            if any(voxel):
                info["voxel_um"] = voxel
        else:
            info["format"] = "tiff"
        info["dtype"] = str(tf.pages[0].dtype)
        info["bytes_on_disk"] = os.path.getsize(path)
    return axes, info


def _parse_ome_xml(xml: str) -> Dict[str, Any]:
    import xml.etree.ElementTree as ET

    out: Dict[str, Any] = {}
    if not xml.strip():
        return out
    root = ET.fromstring(xml)
    ns = root.tag[: root.tag.index("}") + 1] if root.tag.startswith("{") else ""
    pixels = root.find(f".//{ns}Pixels")
    if pixels is None:
        return out
    g = pixels.get
    voxel = [float(g("PhysicalSizeX") or 0), float(g("PhysicalSizeY") or 0), float(g("PhysicalSizeZ") or 0)]
    if any(voxel):
        out["voxel_um"] = voxel
    if g("TimeIncrement"):
        out["frame_interval_s"] = float(g("TimeIncrement"))
    channels = []
    for ch in pixels.findall(f"{ns}Channel"):
        nm = ch.get("EmissionWavelength") or ch.get("ExcitationWavelength") or 0
        entry: Dict[str, Any] = {"label": ch.get("Name") or "", "wavelength_nm": float(nm)}
        color = ch.get("Color")
        if color:
            try:
                rgba = int(color) & 0xFFFFFFFF
                entry["color"] = "#%02x%02x%02x" % ((rgba >> 24) & 255, (rgba >> 16) & 255, (rgba >> 8) & 255)
            except ValueError:
                pass
        channels.append(entry)
    if channels:
        out["channels"] = channels
    return out


def load_dataset(path: str, page_order: str = "czt", c: Optional[int] = None, t: Optional[int] = None,
                 z: Optional[int] = None, progress: ProgressFn = None) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Load a TIFF / OME-TIFF or a zarr / N5 store as (c, t, z, y, x) float32.

    Plain multi-page TIFFs without dimension metadata are reshaped with
    `page_order` (fastest axis first, ImageJ's "czt") and the explicit
    counts; an unspecified count is derived from the page count.
    """
    path = str(path)
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    _progress(progress, 0.0, f"reading {os.path.basename(path)}")
    lower = path.lower().rstrip("/\\")
    if os.path.isdir(path) or lower.endswith((".zarr", ".n5")):
        a, meta = _load_zarr(path)
    else:
        a, meta = _load_tiff(path, page_order, c, t, z)
    _progress(progress, 1.0, "loaded")
    return a, meta


def _load_zarr(path: str) -> Tuple[np.ndarray, Dict[str, Any]]:
    try:
        import zarr  # type: ignore
    except ImportError as e:
        raise NotAvailable("loading zarr stores needs the 'zarr' package") from e
    node = zarr.open(path, mode="r")
    axes_names: Optional[List[str]] = None
    scale: Optional[List[float]] = None
    channels: List[Dict[str, Any]] = []
    arr = node
    attrs = dict(getattr(node, "attrs", {}) or {})
    if not hasattr(node, "shape"):  # a group: OME-NGFF multiscales or the first array
        ms = attrs.get("multiscales") or (attrs.get("ome") or {}).get("multiscales")
        if ms:
            m0 = ms[0]
            axes_names = [ax["name"] if isinstance(ax, dict) else str(ax) for ax in m0.get("axes", [])]
            ds0 = m0["datasets"][0]
            arr = node[ds0["path"]]
            for tr in ds0.get("coordinateTransformations", []):
                if tr.get("type") == "scale":
                    scale = [float(s) for s in tr["scale"]]
        else:
            keys = sorted(node.keys())
            if not keys:
                raise ValueError(f"{path}: empty group")
            arr = node[keys[0]]
        omero = attrs.get("omero") or (attrs.get("ome") or {}).get("omero")
        if omero:
            for ch in omero.get("channels", []):
                channels.append({"label": ch.get("label", ""),
                                 "wavelength_nm": float(ch.get("emission_wavelength") or 0),
                                 "color": "#" + ch["color"] if ch.get("color") else "#ffffff"})
    data = np.asarray(arr[...])
    if axes_names is None or len(axes_names) != data.ndim:
        if data.ndim > 5:
            raise ValueError(f"{path}: cannot map {data.ndim} axes onto (c, t, z, y, x)")
        axes_names = list(AXES[5 - data.ndim:])
    a = _as5(_reorder_to_ctzyx(data, "".join(ax[0] for ax in axes_names)))
    meta = _default_meta(a, path, "n5" if path.lower().rstrip("/\\").endswith(".n5") else "zarr")
    if scale is not None and len(scale) == len(axes_names):
        v = meta["voxel_um"]
        for ax, s in zip(axes_names, scale):
            if ax[0] == "x":
                v[0] = s
            elif ax[0] == "y":
                v[1] = s
            elif ax[0] == "z":
                v[2] = s
    if channels:
        meta["channels"] = (channels + meta["channels"])[: a.shape[0]]
    meta["dtype"] = str(data.dtype)
    meta["dims_from_metadata"] = True
    return a, meta


def _read_tiff_pages(path: str) -> Tuple[np.ndarray, Optional[str]]:
    """(array, axes) -- the sirius extension for plain stacks, tifffile otherwise."""
    axes, _ = _tiff_metadata(path)
    if axes is None or len(axes) <= 3:
        try:
            import sirius  # type: ignore

            return np.asarray(sirius.read_tiff(path, dtype=np.float32)), None
        except Exception:  # noqa: BLE001 - fall back to tifffile
            pass
    try:
        import tifffile  # type: ignore
    except ImportError as e:
        raise NotAvailable("loading TIFF needs the 'sirius' extension or 'tifffile'") from e
    return np.asarray(tifffile.imread(path)), axes


def _load_tiff(path: str, page_order: str, c: Optional[int], t: Optional[int], z: Optional[int]):
    _, info = _tiff_metadata(path)
    data, axes = _read_tiff_pages(path)
    dims_from_meta = False
    if axes is not None and len(axes) == data.ndim and data.ndim > 3:
        norm = axes.upper().replace("Q", "T").replace("S", "C").replace("I", "T")
        a = _as5(_reorder_to_ctzyx(data, norm))
        dims_from_meta = True
    else:
        pages = data.reshape((-1,) + data.shape[-2:])
        n = pages.shape[0]
        counts = {"c": int(c or 0), "t": int(t or 0), "z": int(z or 0)}
        unknown = [k for k, v in counts.items() if v <= 0]
        known = 1
        for k, v in counts.items():
            if v > 0:
                known *= v
        if len(unknown) > 1:
            for k in unknown:
                counts[k] = 1
            counts["z" if "z" in unknown else unknown[0]] = max(1, n // known)
        elif unknown:
            counts[unknown[0]] = max(1, n // known)
        if counts["c"] * counts["t"] * counts["z"] != n:
            raise ValueError(f"{n} pages do not factor into c{counts['c']} t{counts['t']} z{counts['z']}")
        order = page_order.lower()
        slowest_first = "".join(reversed(order))
        shape = tuple(counts[ax] for ax in slowest_first)
        a = _as5(_reorder_to_ctzyx(pages.reshape(shape + pages.shape[1:]), slowest_first + "yx"))
    meta = _default_meta(a, path, info.get("format", "tiff"))
    if info.get("voxel_um") and any(info["voxel_um"]):
        v = meta["voxel_um"]
        for i in range(3):
            if info["voxel_um"][i]:
                v[i] = float(info["voxel_um"][i])
    if info.get("frame_interval_s"):
        meta["frame_interval_s"] = float(info["frame_interval_s"])
    for i, ch in enumerate(info.get("channels", [])[: a.shape[0]]):
        meta["channels"][i].update({k: v for k, v in ch.items() if v not in ("", None)})
    meta["dtype"] = info.get("dtype", str(data.dtype))
    meta["bytes_on_disk"] = info.get("bytes_on_disk", os.path.getsize(path))
    meta["dims_from_metadata"] = dims_from_meta
    return a, meta


# --------------------------------------------------------------------------
# steps
# --------------------------------------------------------------------------


@dataclass
class StepResult:
    array: np.ndarray
    meta: Dict[str, Any]
    labels: Optional[np.ndarray] = None       # (t, z, y, x) uint32
    prob: Optional[np.ndarray] = None         # (C, z, y, x) float32 of the last time point
    info: Dict[str, Any] = field(default_factory=dict)


def _kept_axes(params) -> str:
    v = _get(params, ("axes", "keep", "kept"), None)
    if isinstance(v, dict):  # {"c": true, "t": false, ...}
        return "".join(ax for ax in AXES if v.get(ax, True))
    if isinstance(v, (list, tuple)):
        return "".join(str(x)[0].lower() for x in v)
    if v is None:
        reduced = _get(params, ("reduce", "reduced"), "t")
        if isinstance(reduced, (list, tuple)):
            reduced = "".join(str(x)[0].lower() for x in reduced)
        return "".join(ax for ax in AXES if ax not in str(reduced).lower())
    s = str(v).lower().replace("->", " ").split()
    s = s[-1] if s else ""
    return "".join(ax for ax in AXES if ax in s)


def _reduce(a: np.ndarray, reduce_axes: Sequence[int], op: str) -> np.ndarray:
    if not reduce_axes:
        return a
    axes = tuple(sorted(reduce_axes))
    if op == "sum":
        return a.sum(axis=axes, keepdims=True, dtype=np.float32)
    if op == "mean":
        return a.mean(axis=axes, keepdims=True, dtype=np.float32)
    if op == "max":
        return np.nanmax(a, axis=axes, keepdims=True)
    if op == "min":
        return np.nanmin(a, axis=axes, keepdims=True)
    raise ValueError(f"unknown reduction '{op}'")


def step_einsum(a: np.ndarray, params: Dict[str, Any], meta: Dict[str, Any]) -> StepResult:
    """axes: kept axes ("czyx"); reduction: sum | mean | max | min."""
    kept = _kept_axes(params)
    op = _choice(_get(params, ("reduction", "op", "red"), "mean"), ["sum", "mean", "max", "min"], "mean")
    reduce_axes = [i for i, ax in enumerate(AXES) if ax not in kept]
    out = _reduce(a, reduce_axes, op)
    meta = dict(meta, dims=_dims(out))
    if "c" not in kept:
        meta["channels"] = [{"label": f"{op} over c", "wavelength_nm": 0.0, "color": "#ffffff"}]
        meta["rgb"] = False
    return StepResult(np.ascontiguousarray(out, dtype=np.float32), meta,
                      info={"expression": f"{AXES} -> {kept or '·'}", "reduction": op})


def step_maxproj(a: np.ndarray, params: Dict[str, Any], meta: Dict[str, Any]) -> StepResult:
    """axis: z (default) | t | c."""
    axis = _choice(_get(params, ("axis",), "z"), list(AXES), "z")
    return step_einsum(a, {"axes": AXES.replace(axis, ""), "reduction": "max"}, meta)


def step_meant(a: np.ndarray, params: Dict[str, Any], meta: Dict[str, Any]) -> StepResult:
    return step_einsum(a, {"axes": "czyx", "reduction": "mean"}, meta)


def _percentile_pair(v: np.ndarray, lo: float, hi: float, max_samples: int = 1 << 22) -> Tuple[float, float]:
    flat = v.reshape(-1)
    if flat.size > max_samples:
        flat = flat[:: max(1, flat.size // max_samples)]
    flat = flat[np.isfinite(flat)]
    if flat.size == 0:
        return 0.0, 1.0
    p = np.percentile(flat, [lo, hi])
    return float(p[0]), float(p[1])


def step_contrast(a: np.ndarray, params: Dict[str, Any], meta: Dict[str, Any]) -> StepResult:
    """low / high percentiles (0.2 / 99.8), gamma (1), per_channel (true)."""
    lo = _float(params, ("low", "lo", "low_percentile", "p_low", "percentile_low"), 0.2)
    hi = _float(params, ("high", "hi", "high_percentile", "p_high", "percentile_high"), 99.8)
    gamma = max(_float(params, ("gamma",), 1.0), 1e-3)
    per_channel = _bool(params, ("per_channel", "perChannel"), True)
    out = np.empty_like(a, dtype=np.float32)
    windows = []
    groups: List[Any] = list(range(a.shape[0])) if per_channel else [slice(None)]
    for g in groups:
        vlo, vhi = _percentile_pair(a[g], lo, hi)
        if vhi <= vlo:
            vhi = vlo + 1.0
        x = (a[g] - vlo) / np.float32(vhi - vlo)
        np.clip(x, 0.0, 1.0, out=x)
        if gamma != 1.0:
            x = np.power(x, np.float32(1.0 / gamma))
        out[g] = x
        windows.append([vlo, vhi])
    return StepResult(out, dict(meta), info={"windows": windows, "gamma": gamma})


def _hex_to_rgb(s: str) -> Tuple[float, float, float]:
    s = str(s).strip().lstrip("#")
    if len(s) != 6:
        return 1.0, 1.0, 1.0
    try:
        return tuple(int(s[i:i + 2], 16) / 255.0 for i in (0, 2, 4))  # type: ignore[return-value]
    except ValueError:
        return 1.0, 1.0, 1.0


def step_merge(a: np.ndarray, params: Dict[str, Any], meta: Dict[str, Any]) -> StepResult:
    """blend: Additive | Screen | Max; colors: list of "#rrggbb" per channel
    (defaults to the channel colours of the metadata)."""
    blend = _choice(_get(params, ("blend",), "Additive"), ["Additive", "Screen", "Max"], "Additive")
    c = a.shape[0]
    colors = _get(params, ("colors", "colours"), None)
    if isinstance(colors, str):
        colors = [x.strip() for x in colors.split(",") if x.strip()]
    if not colors:
        colors = [ch.get("color", "#ffffff") for ch in meta.get("channels", [])]
    colors = [colors[i] if i < len(colors) else "#ffffff" for i in range(c)]
    rgbs = [_hex_to_rgb(col) for col in colors]
    out = np.zeros((3,) + a.shape[1:], dtype=np.float32)
    for i in range(c):
        ch = a[i]
        vmax = float(np.nanmax(ch)) if ch.size else 1.0
        norm = ch / np.float32(vmax) if vmax > 1.0 else ch
        for k in range(3):
            w = rgbs[i][k]
            if w == 0.0:
                continue
            contribution = norm * np.float32(w)
            if blend == "Additive":
                out[k] += contribution
            elif blend == "Screen":
                out[k] = 1.0 - (1.0 - out[k]) * (1.0 - np.clip(contribution, 0.0, 1.0))
            else:
                np.maximum(out[k], contribution, out=out[k])
    np.clip(out, 0.0, 1.0, out=out)
    m = dict(meta, dims=_dims(out), rgb=True,
             channels=[{"label": n, "wavelength_nm": 0.0, "color": col}
                       for n, col in (("R", "#ff0000"), ("G", "#00ff00"), ("B", "#0000ff"))])
    return StepResult(out, m, info={"blend": blend, "colors": colors})


def step_croppad(a: np.ndarray, params: Dict[str, Any], meta: Dict[str, Any]) -> StepResult:
    """origin [z, y, x] (may be negative = pad), size [z, y, x] (0 = to the end), fill."""
    z, y, x = a.shape[2:]
    origin = [int(v) for v in _list(params, ("origin", "offset", "start"), 3, [0, 0, 0])]
    size = [int(v) for v in _list(params, ("size", "extent", "shape"), 3, [0, 0, 0])]
    fill = _float(params, ("fill", "pad_value"), 0.0)
    ext = [size[i] if size[i] > 0 else (z, y, x)[i] - origin[i] for i in range(3)]
    if any(e <= 0 for e in ext):
        raise ValueError(f"crop box {origin} + {size} is empty for z{z} y{y} x{x}")
    out = np.full(a.shape[:2] + tuple(ext), np.float32(fill), dtype=np.float32)
    src = []
    dst = []
    for i, n in enumerate((z, y, x)):
        s0 = max(origin[i], 0)
        s1 = min(origin[i] + ext[i], n)
        if s1 <= s0:
            return StepResult(out, dict(meta, dims=_dims(out)), info={"origin": origin, "size": ext})
        src.append(slice(s0, s1))
        dst.append(slice(s0 - origin[i], s1 - origin[i]))
    out[(slice(None), slice(None)) + tuple(dst)] = a[(slice(None), slice(None)) + tuple(src)]
    return StepResult(out, dict(meta, dims=_dims(out)), info={"origin": origin, "size": ext})


def _zoom_volume(v: np.ndarray, factors: Sequence[float], order: int) -> np.ndarray:
    try:
        from scipy import ndimage  # type: ignore

        return ndimage.zoom(v, factors, order=order, mode="nearest", grid_mode=True).astype(np.float32, copy=False)
    except ImportError:
        idx = [np.clip(np.round(np.arange(int(round(n * f))) / max(f, 1e-9)).astype(int), 0, n - 1)
               for n, f in zip(v.shape, factors)]
        return v[np.ix_(*idx)].astype(np.float32, copy=False)


def step_resample(a: np.ndarray, params: Dict[str, Any], meta: Dict[str, Any]) -> StepResult:
    """voxel [z, y, x] target size in µm (0 keeps an axis) or a single
    isotropic size; factor [z, y, x] zoom factors; interpolation: linear |
    nearest | cubic."""
    vx, vy, vz = (list(meta.get("voxel_um", [0.1, 0.1, 0.2])) + [0.1, 0.1, 0.2])[:3]
    interp = _choice(_get(params, ("interpolation", "interp"), "linear"), ["linear", "nearest", "cubic"], "linear")
    order = {"nearest": 0, "linear": 1, "cubic": 3}[interp]
    if _get(params, ("factor", "factors", "zoom"), None) is not None:
        f = _list(params, ("factor", "factors", "zoom"), 3, [1, 1, 1])
    else:
        raw = _get(params, ("voxel", "voxel_um", "target", "isotropic"), None)
        if isinstance(raw, (int, float)):
            target = [float(raw)] * 3
        else:
            target = _list(params, ("voxel", "voxel_um", "target", "isotropic"), 3, [0, 0, 0])
        cur = [vz, vy, vx]
        f = [cur[i] / target[i] if target[i] > 0 else 1.0 for i in range(3)]
    if all(abs(fi - 1.0) < 1e-9 for fi in f):
        return StepResult(a, dict(meta), info={"factor": f})
    out = None
    for c in range(a.shape[0]):
        for t in range(a.shape[1]):
            v = _zoom_volume(a[c, t], f, order)
            if out is None:
                out = np.empty(a.shape[:2] + v.shape, dtype=np.float32)
            out[c, t] = v
    assert out is not None
    m = dict(meta, dims=_dims(out), voxel_um=[vx / f[2], vy / f[1], vz / f[0]])
    return StepResult(out, m, info={"factor": f, "interpolation": interp})


def step_bleach(a: np.ndarray, params: Dict[str, Any], meta: Dict[str, Any]) -> StepResult:
    """Scale every time point's volume so its total matches the first one
    (to_mean: the mean over time)."""
    to_mean = _bool(params, ("to_mean", "toMean", "reference_mean"), False)
    out = a.copy()
    sums = a.sum(axis=(2, 3, 4), dtype=np.float64)  # (c, t)
    ref = sums.mean(axis=1) if to_mean else sums[:, 0]
    scales = np.ones_like(sums)
    for c in range(a.shape[0]):
        for t in range(a.shape[1]):
            if sums[c, t] > 0:
                scales[c, t] = ref[c] / sums[c, t]
                out[c, t] *= np.float32(scales[c, t])
    return StepResult(out, dict(meta), info={"scales": scales.tolist()})


def step_flatfield(a: np.ndarray, params: Dict[str, Any], meta: Dict[str, Any]) -> StepResult:
    """flat: path of the flat-field image (TIFF, (y, x) or (z, y, x)); dark: optional dark frame."""
    flat_path = _str(params, ("flat", "flat_field", "flat_path"))
    if not flat_path:
        raise ValueError("flat-field: no flat image given")
    flat, _ = load_dataset(flat_path)
    flat2 = flat[0, 0].mean(axis=0) if flat.shape[2] > 1 else flat[0, 0, 0]
    dark_path = _str(params, ("dark", "dark_path"))
    if dark_path:
        dark, _ = load_dataset(dark_path)
        dark2 = dark[0, 0].mean(axis=0) if dark.shape[2] > 1 else dark[0, 0, 0]
    else:
        dark2 = np.zeros_like(flat2)
    if flat2.shape != a.shape[-2:]:
        raise ValueError(f"flat field {flat2.shape} does not match planes {a.shape[-2:]}")
    gain = flat2 - dark2
    gain = np.where(gain > 1e-6, gain, 1e-6).astype(np.float32)
    scale = np.float32(gain.mean())
    out = (a - dark2) / gain * scale
    return StepResult(out.astype(np.float32, copy=False), dict(meta))


def _label_components(mask: np.ndarray) -> np.ndarray:
    try:
        from scipy import ndimage  # type: ignore
    except ImportError as e:
        raise NotAvailable("connected components need 'scipy'") from e
    labels, _ = ndimage.label(mask)
    return labels.astype(np.uint32, copy=False)


def _remove_small(labels: np.ndarray, min_voxels: int) -> np.ndarray:
    if min_voxels <= 1:
        return labels
    counts = np.bincount(labels.reshape(-1))
    keep = counts >= min_voxels
    keep[0] = False
    remap = np.zeros_like(counts, dtype=np.uint32)
    remap[keep] = np.arange(1, int(keep.sum()) + 1, dtype=np.uint32)
    return remap[labels]


def step_threshold(a: np.ndarray, params: Dict[str, Any], meta: Dict[str, Any]) -> StepResult:
    """channel; threshold (absolute value) or percentile (0..100); min_voxels;
    labels via 3D connected components."""
    c = _channel_index(params, ("channel", "input_channel"), meta, a.shape[0])
    percentile = _get(params, ("percentile",), None)
    labels = np.zeros((a.shape[1],) + a.shape[2:], dtype=np.uint32)
    thresholds = []
    min_voxels = _int(params, ("min_voxels", "minVoxels", "min_size"), 0)
    for t in range(a.shape[1]):
        v = a[c, t]
        if percentile is not None:
            thr = float(np.percentile(v, float(percentile)))
        else:
            thr = _float(params, ("threshold", "value"), 0.5)
        labels[t] = _remove_small(_label_components(v > thr), min_voxels)
        thresholds.append(thr)
    return StepResult(a, dict(meta), labels=labels,
                      info={"thresholds": thresholds, "channel": c, "labels": int(labels.max())})


# --- SIM ----------------------------------------------------------------------

_SIM_ALIASES = {
    "angles": "ndirs", "directions": "ndirs", "phases": "nphases", "orders": "norders",
    "wavelength": "wavelength_nm", "linespacing": "linespacing_um", "line_spacing": "linespacing_um",
    "k0angle": "k0_start_angle", "k0_angle": "k0_start_angle", "k0angles": "k0_angles",
    "zoom": "zoomfact", "bleach": "do_rescale", "bleach_correction": "do_rescale",
    "suppress_zero": "dampen_order0", "suppress_zero_order": "dampen_order0",
    "otf_cutoff": "otfcutoff", "immersion": "nimm",
}
_sim_cache: Dict[str, Any] = {}


def _sim_parameters(params: Dict[str, Any], meta: Dict[str, Any]):
    try:
        import sirius  # type: ignore
    except ImportError as e:
        raise NotAvailable("SIM reconstruction needs the 'sirius' extension") from e
    p = None
    cfg = _str(params, ("params_file", "config", "parameter_file"))
    if cfg:
        try:
            p = sirius.load_parameters(cfg) if cfg.lower().endswith(".toml") else sirius.load_legacy_parameters(cfg)
        except Exception:  # noqa: BLE001 - try the other format
            p = sirius.load_legacy_parameters(cfg)
    if p is None:
        p = sirius.SIMParameters()
        vx, vy, vz = (list(meta.get("voxel_um", [0.1, 0.1, 0.2])) + [0.1, 0.1, 0.2])[:3]
        p.dx, p.dy, p.dz = float(vx), float(vy), float(vz)
        sim = meta.get("sim") or {}
        if sim.get("ndirs"):
            p.ndirs = int(sim["ndirs"])
        if sim.get("nphases"):
            p.nphases = int(sim["nphases"])
        if sim.get("fast_si"):
            p.fast_si = True
    apod = {"None": sirius.ApodizationType.None_, "Cosine": sirius.ApodizationType.Cosine,
            "Triangle": sirius.ApodizationType.Triangle}
    explicit_orders = _get(params, ("norders", "orders"), None) is not None
    for key, value in params.items():
        k = _SIM_ALIASES.get(key, key)
        if k == "apodization":
            p.apodize_output = apod[_choice(value, list(apod), "Triangle")]
            continue
        if k in ("apodize_input", "apodize_output") and isinstance(value, str):
            value = apod[_choice(value, list(apod), "Triangle")]
        if k.startswith("_") or not hasattr(p, k):
            continue
        cur = getattr(p, k)
        try:
            if isinstance(cur, bool):
                setattr(p, k, _bool({k: value}, (k,), cur))
            elif isinstance(cur, int):
                setattr(p, k, int(round(float(value))))
            elif isinstance(cur, float):
                setattr(p, k, float(value))
            elif k == "k0_angles":
                seq = value if isinstance(value, (list, tuple)) else str(value).split(",")
                setattr(p, k, [float(x) for x in seq])
            else:
                setattr(p, k, value)
        except (TypeError, ValueError):
            pass
    if not explicit_orders and not cfg:
        p.norders = p.nphases // 2 + 1
    p.validate()
    return p


def _params_key(p) -> Dict[str, str]:
    out = {}
    for k in dir(p):
        if k.startswith("_") or k == "validate":
            continue
        v = getattr(p, k)
        if not callable(v):
            out[k] = str(v)
    return out


def _to_numpy(result) -> np.ndarray:
    """numpy from what SimReconstructor.reconstruct returned (numpy on the CPU,
    a DLPack sirius.Buffer on CUDA)."""
    if isinstance(result, np.ndarray):
        return result
    if hasattr(result, "numpy"):          # sirius.Buffer copies device memory to the host
        return np.asarray(result.numpy())
    try:
        import torch  # type: ignore

        return torch.from_dlpack(result).cpu().numpy()
    except ImportError as e:
        raise NotAvailable("reading a CUDA sirius.Buffer back needs 'torch' (DLPack)") from e


def step_sim(a: np.ndarray, params: Dict[str, Any], meta: Dict[str, Any], progress: ProgressFn = None,
             cancelled: CancelFn = None, device: str = "cpu") -> StepResult:
    """Structured illumination reconstruction of a raw stack whose z axis holds
    ndirs * nphases * nz sections. Needs `otf` (radially averaged OTF TIFF) and
    the acquisition parameters (any SIMParameters field, or params_file)."""
    import sirius  # type: ignore  (its absence is reported by _sim_parameters)

    p = _sim_parameters(params, meta)
    otf = _str(params, ("otf", "otf_path", "otf_file"))
    if not otf or not os.path.exists(otf):
        raise ValueError("SIM reconstruction in Python needs a measured OTF file ('otf'); "
                         "the theoretical OTF is only available in the application")
    sections = p.ndirs * p.nphases
    if a.shape[2] % sections:
        raise ValueError(f"{a.shape[2]} sections is not a multiple of ndirs * nphases = {sections}")
    device = resolve_device(device)
    use_cuda = device.startswith("cuda") and sirius.cuda_available()
    dev = sirius.Device.cuda(int(device.split(":")[1]) if ":" in device else 0) if use_cuda else sirius.Device.cpu()
    key = json.dumps({"otf": os.path.abspath(otf), "dev": str(dev), "p": _params_key(p)}, sort_keys=True)
    recon = _sim_cache.get(key)
    if recon is None:
        _sim_cache.clear()
        recon = sirius.SimReconstructor(p, otf, dev)
        _sim_cache[key] = recon
    out = None
    fits = []
    n = a.shape[0] * a.shape[1]
    k = 0
    for c in range(a.shape[0]):
        for t in range(a.shape[1]):
            _check_cancel(cancelled)
            _progress(progress, k / max(n, 1), f"reconstructing c{c} t{t}")
            raw = np.ascontiguousarray(a[c, t], dtype=np.float64)
            src = sirius.to_device(raw, dev) if use_cuda else raw
            res = _to_numpy(recon.reconstruct(src)).astype(np.float32, copy=False)
            if out is None:
                out = np.empty(a.shape[:2] + res.shape, dtype=np.float32)
            out[c, t] = res
            fit = recon.last_fit
            fits.append({"k0": [[float(v[0]), float(v[1])] for v in fit.k0],
                         "amps": [[[float(x.real), float(x.imag)] for x in row] for row in fit.amps]})
            k += 1
    assert out is not None
    vx, vy, vz = (list(meta.get("voxel_um", [0.1, 0.1, 0.2])) + [0.1, 0.1, 0.2])[:3]
    m = dict(meta, dims=_dims(out), voxel_um=[vx / p.zoomfact, vy / p.zoomfact, vz / max(p.z_zoom, 1)],
             sim={"present": False, "ndirs": p.ndirs, "nphases": p.nphases, "fast_si": p.fast_si})
    _progress(progress, 1.0, "done")
    return StepResult(out, m, info={"fits": fits, "wiener": p.wiener, "ndirs": p.ndirs, "nphases": p.nphases,
                                    "device": str(dev)})


# --- Torch segmentation ------------------------------------------------------


def _torch():
    try:
        import torch  # type: ignore
    except ImportError as e:
        raise NotAvailable("Torch models need the 'torch' package") from e
    return torch


def resolve_device(device: str = "auto") -> str:
    """'auto' -> 'cuda' when torch sees a GPU, else 'cpu'."""
    device = (device or "auto").lower()
    if device == "auto":
        try:
            return "cuda" if _torch().cuda.is_available() else "cpu"
        except NotAvailable:
            return "cpu"
    return device


_model_cache: Dict[Tuple[str, str], Any] = {}


class _OnnxModel:
    """Callable wrapper so ONNX sessions look like a TorchScript module taking numpy."""

    def __init__(self, session):
        self.session = session
        self.input_name = session.get_inputs()[0].name
        self.input_shape = session.get_inputs()[0].shape
        self.output_shape = session.get_outputs()[0].shape

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.session.run(None, {self.input_name: np.asarray(x, dtype=np.float32)})[0]


def load_model(path: str, device: str = "auto"):
    """Load a TorchScript (.pt / .pth / .ts) or ONNX (.onnx, via onnxruntime)
    model once per (path, device)."""
    device = resolve_device(device)
    key = (os.path.abspath(path), device)
    m = _model_cache.get(key)
    if m is not None:
        return m
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    if path.lower().endswith(".onnx"):
        try:
            import onnxruntime as ort  # type: ignore
        except ImportError as e:
            raise NotAvailable("ONNX models need 'onnxruntime'") from e
        providers = (["CUDAExecutionProvider", "CPUExecutionProvider"] if device.startswith("cuda")
                     else ["CPUExecutionProvider"])
        m = _OnnxModel(ort.InferenceSession(path, providers=providers))
    else:
        torch = _torch()
        m = torch.jit.load(path, map_location=device)
        m.eval()
    _model_cache[key] = m
    return m


def model_info(path: str, device: str = "cpu") -> Dict[str, Any]:
    """format, input_shape, output_shape, dtype, size_bytes, channels_out of a
    model file. TorchScript shapes are probed with a (1, 1, 16, 32, 32) tensor."""
    m = load_model(path, device)
    info: Dict[str, Any] = {"path": path, "size_bytes": os.path.getsize(path), "dtype": "float32"}
    if isinstance(m, _OnnxModel):
        info["format"] = "ONNX"
        info["input_shape"] = [int(s) if isinstance(s, int) else -1 for s in m.input_shape]
        info["output_shape"] = [int(s) if isinstance(s, int) else -1 for s in m.output_shape]
        out = info["output_shape"]
        info["channels_out"] = out[1] if len(out) > 1 and out[1] > 0 else -1
        return info
    torch = _torch()
    info["format"] = "TorchScript"
    dev = resolve_device(device)
    probe = torch.zeros((1, 1, 16, 32, 32), dtype=torch.float32, device=dev)
    info["input_shape"] = [1, 1, -1, -1, -1]
    with torch.no_grad():
        try:
            y = m(probe)
        except Exception as e:  # noqa: BLE001 - report instead of failing model_info
            info["output_shape"] = [1, -1, -1, -1, -1]
            info["channels_out"] = -1
            info["probe_error"] = str(e).splitlines()[0][:200]
            return info
    y = y[0] if isinstance(y, (tuple, list)) else y
    info["output_shape"] = [1, int(y.shape[1]), -1, -1, -1] if y.dim() == 5 else [int(s) for s in y.shape]
    info["channels_out"] = int(y.shape[1]) if y.dim() >= 2 else 1
    return info


def _blend_window(shape: Sequence[int], overlap: Sequence[int]) -> np.ndarray:
    """Separable raised-cosine window: 1 in the tile core, tapering over each
    overlap band, so overlapping predictions cross-fade without seams."""
    w = np.ones(tuple(shape), dtype=np.float32)
    for ax, (n, o) in enumerate(zip(shape, overlap)):
        o = int(min(max(o, 0), n // 2))
        prof = np.ones(n, dtype=np.float32)
        if o > 0:
            ramp = (0.5 - 0.5 * np.cos(np.pi * (np.arange(o) + 0.5) / o)).astype(np.float32)
            prof[:o] = ramp
            prof[n - o:] = ramp[::-1]
        prof = np.maximum(prof, 1e-3)
        view = [1] * len(shape)
        view[ax] = n
        w *= prof.reshape(view)
    return w


def _activation(out: np.ndarray, activation: str) -> np.ndarray:
    activation = (activation or "auto").lower()
    if activation == "auto":
        if out.size == 0 or (out.min() >= 0.0 and out.max() <= 1.0):
            return out
        activation = "softmax" if out.shape[0] > 1 else "sigmoid"
    if activation == "sigmoid":
        return 1.0 / (1.0 + np.exp(-out))
    if activation == "softmax":
        e = np.exp(out - out.max(axis=0, keepdims=True))
        return e / e.sum(axis=0, keepdims=True)
    return out


def tiled_inference(volume: np.ndarray, model, tile: Sequence[int] = (32, 256, 256),
                    overlap: Sequence[int] = (4, 32, 32), device: str = "auto", pad_to: int = 1,
                    activation: str = "auto", normalize: bool = True, progress: ProgressFn = None,
                    cancelled: CancelFn = None) -> np.ndarray:
    """Run a (1, 1, z, y, x) -> (1, C, z, y, x) model over `volume` (z, y, x)
    tile by tile with overlap blending. Returns (C, z, y, x) float32 probabilities."""
    volume = np.ascontiguousarray(volume, dtype=np.float32)
    if volume.ndim != 3:
        raise ValueError(f"tiled_inference expects (z, y, x), got {volume.shape}")
    if normalize:
        lo, hi = _percentile_pair(volume, 1.0, 99.9)
        if hi <= lo:
            hi = lo + 1.0
        volume = np.clip((volume - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)
    shape = volume.shape
    tile = [int(min(max(int(t), 1), n)) for t, n in zip(tile, shape)]
    overlap = [int(min(max(int(o), 0), t // 2)) for o, t in zip(overlap, tile)]
    starts = []
    for n, t, o in zip(shape, tile, overlap):
        if t >= n:
            starts.append([0])
            continue
        step = max(t - o, 1)
        pos = list(range(0, n - t, step)) + [n - t]
        starts.append(sorted(set(pos)))
    tiles = [(z0, y0, x0) for z0 in starts[0] for y0 in starts[1] for x0 in starts[2]]
    window = _blend_window(tile, overlap)
    is_onnx = isinstance(model, _OnnxModel)
    torch = None if is_onnx else _torch()
    dev = resolve_device(device)
    acc: Optional[np.ndarray] = None
    weight = np.zeros(shape, dtype=np.float32)
    for i, (z0, y0, x0) in enumerate(tiles):
        _check_cancel(cancelled)
        _progress(progress, i / len(tiles), f"tile {i + 1} / {len(tiles)}")
        patch = volume[z0:z0 + tile[0], y0:y0 + tile[1], x0:x0 + tile[2]]
        patch_in = patch
        if pad_to > 1:
            pads = [(0, (-n) % pad_to) for n in patch.shape]
            if any(p[1] for p in pads):
                mode = "reflect" if all(n > 1 for n in patch.shape) else "edge"
                patch_in = np.pad(patch, pads, mode=mode)
        x = patch_in[np.newaxis, np.newaxis]
        if is_onnx:
            y = np.asarray(model(x))
        else:
            with torch.no_grad():  # type: ignore[union-attr]
                yt = model(torch.from_numpy(np.ascontiguousarray(x)).to(dev))  # type: ignore[union-attr]
                yt = yt[0] if isinstance(yt, (tuple, list)) else yt
                y = yt.float().cpu().numpy()
        if y.ndim == 4:
            y = y[np.newaxis]
        if y.ndim != 5:
            raise ValueError(f"model returned shape {y.shape}, expected (1, C, z, y, x)")
        y = y[0][:, : patch.shape[0], : patch.shape[1], : patch.shape[2]]
        if acc is None:
            acc = np.zeros((y.shape[0],) + shape, dtype=np.float32)
        w = window[: patch.shape[0], : patch.shape[1], : patch.shape[2]]
        sl = (slice(z0, z0 + patch.shape[0]), slice(y0, y0 + patch.shape[1]), slice(x0, x0 + patch.shape[2]))
        acc[(slice(None),) + sl] += y * w
        weight[sl] += w
    assert acc is not None
    acc /= np.maximum(weight, 1e-6)
    _progress(progress, 1.0, "tiles done")
    return _activation(acc, activation).astype(np.float32, copy=False)


def _watershed_labels(prob: np.ndarray, threshold: float, boundary_channel: int, fg_channel: int) -> np.ndarray:
    fg = prob[fg_channel] > threshold
    has_boundary = 0 <= boundary_channel < prob.shape[0] and boundary_channel != fg_channel
    if not has_boundary:
        return _label_components(fg)
    boundary = prob[boundary_channel]
    try:
        from scipy import ndimage  # type: ignore
    except ImportError as e:
        raise NotAvailable("watershed post-processing needs 'scipy'") from e
    seeds, _ = ndimage.label(fg & (boundary < 0.5))
    try:
        from skimage.segmentation import watershed  # type: ignore

        return watershed(boundary, markers=seeds, mask=fg).astype(np.uint32)
    except ImportError:
        # nearest-seed assignment of the remaining foreground voxels
        _, idx = ndimage.distance_transform_edt(seeds == 0, return_indices=True)
        nearest = seeds[tuple(idx)]
        return np.where(fg, nearest, 0).astype(np.uint32)


def step_seg(a: np.ndarray, params: Dict[str, Any], meta: Dict[str, Any], progress: ProgressFn = None,
             cancelled: CancelFn = None, device: str = "auto") -> StepResult:
    """Torch segmentation: model, channel, tile [z, y, x], overlap, threshold,
    post (Watershed on boundary channel | Connected components | None),
    fg_channel / boundary_channel, min_voxels, normalize, activation, pad_to."""
    path = _str(params, ("model", "model_path", "torch_model"))
    if not path:
        raise ValueError("segmentation: no model given")
    c = _channel_index(params, ("channel", "input_channel"), meta, a.shape[0])
    tile = [int(v) for v in _list(params, ("tile", "tile_size"), 3, [32, 256, 256])]
    ov = _get(params, ("overlap",), 32)
    if isinstance(ov, (list, tuple, str)):
        overlap = [int(v) for v in _list(params, ("overlap",), 3, [4, 32, 32])]
    else:
        overlap = [max(1, int(ov) // 8), int(ov), int(ov)]
    threshold = _float(params, ("threshold", "tau"), 0.5)
    post = _choice(_get(params, ("post", "post_processing", "postprocess"), "Watershed on boundary channel"),
                   ["Watershed on boundary channel", "Connected components", "None"],
                   "Watershed on boundary channel")
    model = load_model(path, device)
    labels = np.zeros((a.shape[1],) + a.shape[2:], dtype=np.uint32)
    prob_last = None
    nt = a.shape[1]
    for t in range(nt):
        prob = tiled_inference(a[c, t], model, tile, overlap, device, _int(params, ("pad_to",), 1),
                               _str(params, ("activation",), "auto"), _bool(params, ("normalize",), True),
                               progress=lambda f, m, _t=t: _progress(progress, (_t + f) / nt, m),
                               cancelled=cancelled)
        fg = _int(params, ("fg_channel", "foreground_channel"), 0 if prob.shape[0] < 3 else 1)
        fg = min(max(fg, 0), prob.shape[0] - 1)
        bd = _int(params, ("boundary_channel",), prob.shape[0] - 1 if prob.shape[0] >= 2 else -1)
        if post == "Connected components":
            lab = _label_components(prob[fg] > threshold)
        elif post == "None":
            lab = np.zeros(prob.shape[1:], dtype=np.uint32)
        else:
            lab = _watershed_labels(prob, threshold, bd, fg)
        labels[t] = _remove_small(lab, _int(params, ("min_voxels", "minVoxels"), 0))
        prob_last = prob
    return StepResult(a, dict(meta), labels=labels, prob=prob_last,
                      info={"model": path, "channels_out": int(prob_last.shape[0]) if prob_last is not None else 0,
                            "labels": int(labels.max()), "post": post})


# --------------------------------------------------------------------------
# dispatch
# --------------------------------------------------------------------------

_STEPS: Dict[str, Callable[..., StepResult]] = {
    "einsum": step_einsum,
    "maxproj": step_maxproj,
    "meant": step_meant,
    "contrast": step_contrast,
    "merge": step_merge,
    "croppad": step_croppad,
    "resample": step_resample,
    "bleach": step_bleach,
    "flatfield": step_flatfield,
    "threshold": step_threshold,
    "sim": step_sim,
    "seg": step_seg,
}
_KIND_ALIASES = {
    "max_projection": "maxproj", "max": "maxproj", "mean_over_time": "meant", "mean_t": "meant",
    "crop": "croppad", "crop_pad": "croppad", "flat_field": "flatfield", "torch": "seg",
    "torch_segment": "seg", "segmentation": "seg", "sim_reconstruction": "sim",
}
_UNSUPPORTED = {
    "decon": "Richardson-Lucy deconvolution", "deskew": "deskew + rotate", "volrec": "volume reconstruction",
    "stitch": "tile stitching", "register": "registration", "label_cleanup": "label cleanup",
}
# Steps that do not change the array (viewer-side) are skipped by run_pipeline.
_PASSTHROUGH = {"load", "label_cleanup", "volrec"}


def step_kinds() -> List[str]:
    """Kinds run_step implements here (canonical names)."""
    return sorted(_STEPS)


def run_step(kind: str, params: Dict[str, Any], array: np.ndarray, meta: Optional[Dict[str, Any]] = None,
             labels: Optional[np.ndarray] = None, progress: ProgressFn = None, cancelled: CancelFn = None,
             device: str = "auto") -> StepResult:
    """Run one step on a (c, t, z, y, x) array. Raises NotAvailable for kinds
    only the C++ application implements."""
    a = _as5(array)
    meta = dict(meta) if meta else _default_meta(a)
    meta["dims"] = _dims(a)
    k = _KIND_ALIASES.get(kind, kind)
    fn = _STEPS.get(k)
    if fn is None:
        what = _UNSUPPORTED.get(k, k)
        raise NotAvailable(f"step '{kind}' ({what}) is not implemented in Python; run it in the SIRIUS application")
    if k in ("sim", "seg"):
        res = fn(a, params or {}, meta, progress=progress, cancelled=cancelled, device=device)
    else:
        res = fn(a, params or {}, meta)
    if res.labels is None and labels is not None and res.array.shape[1:] == a.shape[1:]:
        res.labels = labels
    res.meta["dims"] = _dims(res.array)
    return res


def run_pipeline(dataset_path: str, pipeline: Any, progress: ProgressFn = None, device: str = "auto",
                 strict: bool = True) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Load `dataset_path` and run the pipeline (the app's JSON: {"steps": [...]}
    or a list of steps, each {"kind", "params", "enabled"}). Returns the final
    (c, t, z, y, x) float32 array and its metadata (with "labels" when a
    segmentation step produced them). Steps the Python side cannot run raise
    NotAvailable, or are skipped with their kinds in meta["skipped"] when
    strict is False.
    """
    if isinstance(pipeline, str):
        pipeline = json.loads(pipeline)
    steps = pipeline.get("steps", pipeline) if isinstance(pipeline, dict) else pipeline
    load_params: Dict[str, Any] = {}
    for s in steps:
        if s.get("kind") == "load":
            load_params = s.get("params", {}) or {}
            break
    order = _str(load_params, ("page_order", "pageOrder"), "czt") or "czt"
    array, meta = load_dataset(dataset_path, order, _get(load_params, ("c", "channels"), None),
                               _get(load_params, ("t", "timepoints"), None), _get(load_params, ("z", "planes"), None),
                               progress=lambda f, m: _progress(progress, 0.0, m))
    if _get(load_params, ("voxel", "voxel_um"), None) is not None:
        meta["voxel_um"] = _list(load_params, ("voxel", "voxel_um"), 3, meta["voxel_um"])
    labels: Optional[np.ndarray] = None
    skipped: List[str] = []
    n = max(len(steps), 1)
    for i, s in enumerate(steps):
        kind = s.get("kind", "")
        name = s.get("name") or kind
        if kind == "load" or not s.get("enabled", True) or kind in _PASSTHROUGH:
            continue
        _progress(progress, i / n, name)
        try:
            res = run_step(kind, s.get("params", {}) or {}, array, meta, labels,
                           progress=lambda f, m, _i=i: _progress(progress, (_i + f) / n, f"{name}: {m}"),
                           cancelled=None, device=device)
        except NotAvailable:
            if strict:
                raise
            skipped.append(kind)
            continue
        array, meta = res.array, res.meta
        if res.labels is not None:
            labels = res.labels
    if labels is not None:
        meta["labels"] = labels
    if skipped:
        meta["skipped"] = skipped
    _progress(progress, 1.0, "done")
    return array, meta
