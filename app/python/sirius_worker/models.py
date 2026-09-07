"""Segmentation models: where they come from and how they run.

A model *spec* is the string the application's Torch segmentation step holds:

    /path/to/model.pt            TorchScript (.pt / .pts / .pth) or ONNX (.onnx) file
    hf:<repo_id>[:<filename>]    a file from a Hugging Face repository, downloaded once
                                 into the cache ($SIRIUS_MODEL_CACHE or ~/.sirius/models)
    cellpose:<model>             the `cellpose` package: cyto3, nuclei, cyto2, ... or a
                                 path / hf: spec of a custom Cellpose model file
    microsam:<model_type>        the `micro_sam` package's automatic instance
                                 segmentation: vit_b_lm, vit_l_lm, vit_t_lm, vit_b_em_organelles, ...

File models return per-tile probabilities (the application turns them into
labels); the package families return instance labels directly.  The optional
packages are imported lazily so the worker starts without them and reports a
`pip install` hint instead of failing.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

ProgressFn = Optional[Callable[[float, str], None]]
CancelFn = Optional[Callable[[], bool]]

MODEL_EXTENSIONS = (".pt", ".pts", ".pth", ".onnx")
FAMILIES = ("file", "hf", "cellpose", "microsam")

CELLPOSE_MODELS = ("cyto3", "nuclei", "cyto2", "cyto", "livecell", "tissuenet", "yeast_PhC", "yeast_BF",
                   "bact_phase", "bact_fluor", "deepbacs", "cyto2_cp3")
MICROSAM_MODELS = ("vit_b_lm", "vit_l_lm", "vit_t_lm", "vit_b_em_organelles", "vit_l_em_organelles",
                   "vit_t_em_organelles", "vit_b", "vit_l", "vit_h")

INSTALL_HINTS = {
    "cellpose": "pip install cellpose",
    "microsam": "conda install -c conda-forge micro_sam   (or: pip install micro-sam)",
    "hf": "pip install huggingface_hub",
    "onnx": "pip install onnxruntime",
}


class ModelError(Exception):
    pass


class NotAvailable(ModelError):
    """An optional package (cellpose, micro_sam, huggingface_hub) is missing."""


# --- specs -----------------------------------------------------------------------------


class ModelSpec:
    """Parsed model spec."""

    def __init__(self, family: str, name: str = "", filename: str = "", raw: str = "") -> None:
        self.family = family          # "file" | "hf" | "cellpose" | "microsam"
        self.name = name              # path, repo id, or model name
        self.filename = filename      # hf: file inside the repo (may be empty)
        self.raw = raw or self.text()

    def text(self) -> str:
        if self.family == "file":
            return self.name
        if self.family == "hf":
            return f"hf:{self.name}:{self.filename}" if self.filename else f"hf:{self.name}"
        return f"{self.family}:{self.name}"

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"ModelSpec({self.text()!r})"


def parse_spec(spec: str) -> ModelSpec:
    s = (spec or "").strip()
    if not s:
        raise ModelError("no model given")
    low = s.lower()
    if low.startswith("hf:") or low.startswith("huggingface:"):
        body = s.split(":", 1)[1]
        if body.startswith("//"):
            body = body[2:]
        repo, _, filename = body.partition(":")
        repo = repo.strip().strip("/")
        if repo.count("/") != 1 or not all(repo.split("/")):
            raise ModelError(f"hf spec '{s}': expected hf:<owner>/<repo>[:<filename>]")
        return ModelSpec("hf", repo, filename.strip(), s)
    if low.startswith("cellpose:"):
        name = s.split(":", 1)[1].strip()
        if not name:
            raise ModelError("cellpose spec needs a model name, e.g. cellpose:cyto3")
        return ModelSpec("cellpose", name, "", s)
    if low.startswith("microsam:") or low.startswith("micro-sam:") or low.startswith("micro_sam:"):
        name = s.split(":", 1)[1].strip()
        if not name:
            raise ModelError("microsam spec needs a model type, e.g. microsam:vit_b_lm")
        return ModelSpec("microsam", name, "", s)
    return ModelSpec("file", s, "", s)


def is_family_spec(spec: str) -> bool:
    try:
        return parse_spec(spec).family in ("cellpose", "microsam")
    except ModelError:
        return False


def is_file_spec(spec: str) -> bool:
    try:
        return parse_spec(spec).family == "file"
    except ModelError:
        return False


# --- cache -----------------------------------------------------------------------------


def cache_dir() -> Path:
    env = os.environ.get("SIRIUS_MODEL_CACHE", "").strip()
    return Path(env).expanduser() if env else Path.home() / ".sirius" / "models"


def repo_dir(repo: str) -> Path:
    # one flat directory per repository; "/" is not a legal file-name character
    return cache_dir() / "hf" / repo.replace("/", "--")


def cached_path(repo: str, filename: str) -> Optional[str]:
    """Local path of an already-downloaded repository file, else None."""
    if not filename:
        return None
    p = repo_dir(repo) / filename
    return str(p) if p.is_file() else None


def list_cached_models() -> List[Dict[str, Any]]:
    """Model files in the cache: [{spec, path, bytes, repo, file}]."""
    out: List[Dict[str, Any]] = []
    root = cache_dir()
    hf = root / "hf"
    if hf.is_dir():
        for rdir in sorted(hf.iterdir()):
            if not rdir.is_dir():
                continue
            repo = rdir.name.replace("--", "/", 1)
            for f in sorted(rdir.rglob("*")):
                if f.is_file() and f.suffix.lower() in MODEL_EXTENSIONS and not f.name.startswith("."):
                    rel = f.relative_to(rdir).as_posix()
                    out.append({"spec": f"hf:{repo}:{rel}", "path": str(f), "bytes": f.stat().st_size,
                                "repo": repo, "file": rel})
    local = root / "local"
    if local.is_dir():
        for f in sorted(local.iterdir()):
            if f.is_file() and f.suffix.lower() in MODEL_EXTENSIONS:
                out.append({"spec": str(f), "path": str(f), "bytes": f.stat().st_size, "repo": "", "file": f.name})
    return out


# --- Hugging Face ------------------------------------------------------------------------


def _hf_api():
    try:
        from huggingface_hub import HfApi  # type: ignore
    except ImportError as e:
        raise NotAvailable(f"Hugging Face access needs the 'huggingface_hub' package ({INSTALL_HINTS['hf']})") from e
    return HfApi()


def hub_search(query: str, limit: int = 25, filter_tag: str = "") -> List[Dict[str, Any]]:
    api = _hf_api()
    kwargs: Dict[str, Any] = {"search": query or None, "limit": max(1, int(limit)), "sort": "downloads",
                              "direction": -1}
    if filter_tag:
        kwargs["filter"] = filter_tag
    out = []
    for m in api.list_models(**kwargs):
        out.append({
            "id": m.id,
            "downloads": int(getattr(m, "downloads", 0) or 0),
            "likes": int(getattr(m, "likes", 0) or 0),
            "tags": list(getattr(m, "tags", []) or [])[:8],
            "last_modified": str(getattr(m, "last_modified", "") or ""),
            "pipeline_tag": str(getattr(m, "pipeline_tag", "") or ""),
        })
    return out


def hub_files(repo: str) -> List[Dict[str, Any]]:
    api = _hf_api()
    info = api.model_info(repo, files_metadata=True)
    files = []
    for s in getattr(info, "siblings", None) or []:
        name = getattr(s, "rfilename", "")
        size = getattr(s, "size", None)
        files.append({"name": name, "size": int(size) if size is not None else -1,
                      "model": name.lower().endswith(MODEL_EXTENSIONS)})
    files.sort(key=lambda f: (not f["model"], f["name"]))
    return files


def pick_model_file(repo: str) -> str:
    """The single model file of a repository, or an error naming the candidates."""
    candidates = [f["name"] for f in hub_files(repo) if f["model"]]
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise ModelError(f"hf:{repo} holds no TorchScript / ONNX file; give hf:{repo}:<filename>")
    raise ModelError(f"hf:{repo} holds several model files: " + ", ".join(candidates[:10]) +
                     f"; choose one as hf:{repo}:<filename>")


def _progress_tqdm(progress: ProgressFn):
    """A tqdm subclass whose updates call `progress(fraction, message)`; None
    when tqdm is not importable (then only start/end are reported)."""
    if progress is None:
        return None
    try:
        from tqdm.auto import tqdm  # type: ignore
    except ImportError:
        return None

    class _Tqdm(tqdm):  # type: ignore[misc]
        def update(self, n=1):
            super().update(n)
            total = self.total or 0
            if total > 0:
                progress(min(1.0, self.n / total), f"{self.n / 2**20:.0f} / {total / 2**20:.0f} MB")

    return _Tqdm


def hub_download(repo: str, filename: str = "", progress: ProgressFn = None) -> str:
    """Download one repository file into the cache; returns the local path.
    Files already present are not fetched again."""
    try:
        from huggingface_hub import hf_hub_download  # type: ignore
    except ImportError as e:
        raise NotAvailable(f"Hugging Face downloads need 'huggingface_hub' ({INSTALL_HINTS['hf']})") from e
    if not filename:
        filename = pick_model_file(repo)
    have = cached_path(repo, filename)
    if have:
        if progress:
            progress(1.0, "cached")
        return have
    target = repo_dir(repo)
    target.mkdir(parents=True, exist_ok=True)
    if progress:
        progress(0.0, f"downloading {filename}")
    kwargs: Dict[str, Any] = {"repo_id": repo, "filename": filename, "local_dir": str(target)}
    tqdm_class = _progress_tqdm(progress)
    if tqdm_class is not None:
        kwargs["tqdm_class"] = tqdm_class
    try:
        path = hf_hub_download(**kwargs)
    except TypeError:   # older huggingface_hub without tqdm_class
        kwargs.pop("tqdm_class", None)
        path = hf_hub_download(**kwargs)
    if progress:
        progress(1.0, filename)
    return str(Path(path).resolve())


def resolve(spec: str, progress: ProgressFn = None) -> Tuple[ModelSpec, str]:
    """Spec -> (parsed spec, local file path) for file and hf models; family
    specs resolve to (spec, model name)."""
    ms = parse_spec(spec)
    if ms.family == "file":
        if not os.path.exists(ms.name):
            raise FileNotFoundError(ms.name)
        return ms, ms.name
    if ms.family == "hf":
        return ms, hub_download(ms.name, ms.filename, progress)
    return ms, ms.name


# --- families ----------------------------------------------------------------------------


def family_available(family: str) -> Tuple[bool, str]:
    """(importable, install hint)."""
    if family == "cellpose":
        try:
            import cellpose  # type: ignore  # noqa: F401
            return True, ""
        except ImportError:
            return False, INSTALL_HINTS["cellpose"]
    if family == "microsam":
        try:
            import micro_sam  # type: ignore  # noqa: F401
            return True, ""
        except ImportError:
            return False, INSTALL_HINTS["microsam"]
    if family == "hf":
        try:
            import huggingface_hub  # type: ignore  # noqa: F401
            return True, ""
        except ImportError:
            return False, INSTALL_HINTS["hf"]
    return True, ""


def family_info(spec: str) -> Dict[str, Any]:
    """What model_info reports for cellpose:/microsam: specs."""
    ms = parse_spec(spec)
    available, hint = family_available(ms.family)
    info: Dict[str, Any] = {
        "spec": ms.text(), "family": ms.family, "model": ms.name,
        "format": {"cellpose": "cellpose", "microsam": "micro-sam"}.get(ms.family, ms.family),
        "available": available, "install_hint": hint, "returns": "labels",
        "dtype": "float32", "input_shape": [1, 1, -1, -1, -1], "output_shape": ["labels", -1, -1, -1],
    }
    if ms.family == "cellpose":
        info["known_models"] = list(CELLPOSE_MODELS)
    elif ms.family == "microsam":
        info["known_models"] = list(MICROSAM_MODELS)
    return info


def _normalize(volume: np.ndarray) -> np.ndarray:
    v = np.asarray(volume, dtype=np.float32)
    lo, hi = np.percentile(v, (1.0, 99.9))
    if hi <= lo:
        hi = lo + 1.0
    return np.clip((v - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def _check(cancelled: CancelFn) -> None:
    if cancelled and cancelled():
        raise RuntimeError("cancelled")


def run_cellpose(volume: np.ndarray, model_name: str, params: Dict[str, Any], device: str = "auto",
                 progress: ProgressFn = None, cancelled: CancelFn = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Instance labels (uint32 (z, y, x)) from the Cellpose package, plus the
    cell probability as one channel when the model reports flows."""
    try:
        from cellpose import models  # type: ignore
    except ImportError as e:
        raise NotAvailable(f"cellpose:{model_name} needs the 'cellpose' package ({INSTALL_HINTS['cellpose']})") from e
    gpu = device != "cpu"
    name = model_name
    if name.lower().startswith("hf:") or os.path.exists(name):
        _, name = resolve(name, progress)
        model = models.CellposeModel(gpu=gpu, pretrained_model=name)
    else:
        try:
            model = models.CellposeModel(gpu=gpu, model_type=name)
        except TypeError:   # cellpose >= 4 dropped model_type (one built-in model)
            model = models.CellposeModel(gpu=gpu)
    z = volume.shape[0]
    diameter = params.get("diameter")
    diameter = float(diameter) if diameter not in (None, "", 0, "0") else None
    do_3d = bool(params.get("do_3d", z > 1 and str(params.get("mode", "3D")) == "3D"))
    kwargs: Dict[str, Any] = {"diameter": diameter, "channels": [0, 0], "do_3D": bool(do_3d and z > 1)}
    if params.get("anisotropy") not in (None, "", 0):
        kwargs["anisotropy"] = float(params["anisotropy"])
    if not kwargs["do_3D"] and z > 1:
        kwargs["stitch_threshold"] = float(params.get("stitch_threshold", 0.5))
    for key in ("flow_threshold", "cellprob_threshold"):
        if params.get(key) not in (None, ""):
            kwargs[key] = float(params[key])
    if progress:
        progress(0.05, f"cellpose {model_name}")
    data = _normalize(volume) if params.get("normalize", True) else np.asarray(volume, dtype=np.float32)
    result = model.eval(data if z > 1 else data[0], **kwargs)
    _check(cancelled)
    masks = result[0]
    flows = result[1] if len(result) > 1 else None
    labels = np.asarray(masks, dtype=np.uint32)
    if labels.ndim == 2:
        labels = labels[np.newaxis]
    prob = None
    try:
        cellprob = flows[2] if flows is not None else None
        if cellprob is not None:
            p = 1.0 / (1.0 + np.exp(-np.asarray(cellprob, dtype=np.float32)))
            if p.ndim == 2:
                p = p[np.newaxis]
            if p.shape == labels.shape:
                prob = np.ascontiguousarray(p[np.newaxis], dtype=np.float32)   # (1, z, y, x)
    except Exception:  # noqa: BLE001 - the probability map is optional
        prob = None
    if progress:
        progress(1.0, f"{int(labels.max()) if labels.size else 0} labels")
    return np.ascontiguousarray(labels), prob


def run_microsam(volume: np.ndarray, model_type: str, params: Dict[str, Any], device: str = "auto",
                 progress: ProgressFn = None, cancelled: CancelFn = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Instance labels from micro-SAM's automatic instance segmentation, plane
    by plane (or through its 3D z-linking for stacks in 3D mode)."""
    try:
        from micro_sam.automatic_segmentation import (  # type: ignore
            automatic_instance_segmentation, get_predictor_and_segmenter)
    except ImportError as e:
        raise NotAvailable(f"microsam:{model_type} needs the 'micro_sam' package ({INSTALL_HINTS['microsam']})") from e
    z = volume.shape[0]
    checkpoint = params.get("checkpoint") or None
    if checkpoint and str(checkpoint).lower().startswith("hf:"):
        _, checkpoint = resolve(str(checkpoint), progress)
    if progress:
        progress(0.05, f"micro-sam {model_type}")
    predictor, segmenter = get_predictor_and_segmenter(
        model_type=model_type, checkpoint=checkpoint, device=None if device == "auto" else device,
        amg=bool(params.get("amg", False)), is_tiled=False)
    data = _normalize(volume) if params.get("normalize", True) else np.asarray(volume, dtype=np.float32)
    data8 = (np.clip(data, 0.0, 1.0) * 255.0).astype(np.uint8)
    if z > 1 and str(params.get("mode", "3D")) == "3D":
        labels = np.asarray(automatic_instance_segmentation(predictor=predictor, segmenter=segmenter,
                                                            input_path=data8, ndim=3, verbose=False), dtype=np.uint32)
    else:
        planes = []
        for k in range(z):
            _check(cancelled)
            lab = automatic_instance_segmentation(predictor=predictor, segmenter=segmenter, input_path=data8[k],
                                                  ndim=2, verbose=False)
            planes.append(np.asarray(lab, dtype=np.uint32))
            if progress:
                progress(0.05 + 0.9 * (k + 1) / z, f"plane {k + 1} / {z}")
        labels = np.stack(planes)
    if progress:
        progress(1.0, f"{int(labels.max()) if labels.size else 0} labels")
    return np.ascontiguousarray(labels), None


def run_family(spec: str, volume: np.ndarray, params: Dict[str, Any], device: str = "auto",
               progress: ProgressFn = None, cancelled: CancelFn = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Dispatch cellpose:/microsam: specs; returns (labels uint32 (z, y, x), prob (1, z, y, x) or None)."""
    ms = parse_spec(spec)
    volume = np.asarray(volume, dtype=np.float32)
    while volume.ndim > 3 and volume.shape[0] == 1:
        volume = volume[0]
    if volume.ndim == 2:
        volume = volume[np.newaxis]
    if volume.ndim != 3:
        raise ModelError(f"family models take a (z, y, x) volume, got shape {volume.shape}")
    if ms.family == "cellpose":
        return run_cellpose(volume, ms.name, params, device, progress, cancelled)
    if ms.family == "microsam":
        return run_microsam(volume, ms.name, params, device, progress, cancelled)
    raise ModelError(f"'{spec}' is not a model family spec")
