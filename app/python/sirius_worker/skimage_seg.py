"""Segmentation methods that scikit-image has and the application does not.

The application's Classical segmentation step covers the filters, thresholds,
watersheds and morphology it can implement natively in C++ and mirror exactly
in :mod:`sirius.workbench`.  What is left is the handful of methods that are
worth having but are large pieces of numerical code in their own right: a
seeded random walker, an edge-driven active contour, and two superpixel
over-segmentations.  Those run here, in the worker, against scikit-image.

Every method takes a (z, y, x) float32 volume and returns (z, y, x) uint32
instance labels with 0 for background, so the application treats the result
exactly as it treats the labels a Torch model returns.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np

METHODS = ("Random walker", "Active contour (geodesic)", "Superpixels (SLIC)",
           "Superpixels (Felzenszwalb)", "Watershed (compact)")


class SkimageError(RuntimeError):
    """Raised with a message the application shows as the step's error."""


def available() -> Tuple[bool, str]:
    """(is scikit-image importable, why not)."""
    try:
        import skimage  # noqa: F401
    except Exception as e:  # noqa: BLE001 - any import failure is the same to the caller
        return False, f"scikit-image is not installed in the worker's Python: pip install scikit-image ({e})"
    return True, ""


def _require():
    ok, why = available()
    if not ok:
        raise SkimageError(why)


def _mask(volume: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
    """The rough foreground the seeded methods start from."""
    from skimage.filters import threshold_otsu

    manual = params.get("threshold")
    if manual is not None and str(manual) != "":
        cut = float(manual)
    else:
        finite = volume[np.isfinite(volume)]
        cut = float(threshold_otsu(finite)) if finite.size and finite.max() > finite.min() else 0.0
    return volume > cut


def _markers(volume: np.ndarray, mask: np.ndarray, depth: float) -> Tuple[np.ndarray, int]:
    """Background as marker 1 and one marker per object from the h-maxima of
    the distance transform, which is what the Classical step seeds with too."""
    from scipy import ndimage
    from skimage.morphology import h_maxima, label

    if not mask.any():
        return np.zeros(volume.shape, dtype=np.int32), 0
    distance = ndimage.distance_transform_edt(mask).astype(np.float32)
    peaks = h_maxima(distance, max(1e-6, float(depth)))
    seeds, count = label(peaks * mask, connectivity=3, return_num=True)
    markers = np.zeros(volume.shape, dtype=np.int32)
    markers[~mask] = 1
    markers[seeds > 0] = seeds[seeds > 0] + 1
    return markers, int(count)


def _clean(labels: np.ndarray, min_voxels: int) -> np.ndarray:
    """Drop anything too small and renumber 1..n densely."""
    out = np.asarray(labels, dtype=np.int64)
    out[out < 0] = 0
    counts = np.bincount(out.ravel())
    keep = np.zeros(counts.size, dtype=np.uint32)
    nxt = 0
    for i in range(1, counts.size):
        if counts[i] and counts[i] >= max(1, int(min_voxels)):
            nxt += 1
            keep[i] = nxt
    return keep[out].astype(np.uint32)


def _normalized(volume: np.ndarray) -> np.ndarray:
    lo, hi = float(np.nanmin(volume)), float(np.nanmax(volume))
    if not hi > lo:
        return np.zeros_like(volume, dtype=np.float32)
    return ((volume - lo) / (hi - lo)).astype(np.float32)


def run(volume: np.ndarray,
        params: Dict[str, Any],
        progress: Optional[Callable[[float, str], None]] = None,
        cancelled: Optional[Callable[[], bool]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Run one method; returns (labels, info)."""
    # what the caller got wrong is worth saying before what the machine is
    # missing: an unknown method is an unknown method whether or not
    # scikit-image is installed here
    method = str(params.get("method", METHODS[0]))
    if method not in METHODS:
        raise SkimageError(f"unknown method '{method}'; expected one of {', '.join(METHODS)}")
    volume = np.ascontiguousarray(np.asarray(volume, dtype=np.float32))
    if volume.ndim != 3:
        raise SkimageError(f"expected a (z, y, x) volume, got {volume.ndim} dimensions")
    _require()

    def report(fraction: float, message: str) -> None:
        if progress:
            progress(float(fraction), message)

    def check() -> None:
        if cancelled and cancelled():
            raise SkimageError("cancelled")

    min_voxels = int(params.get("min_voxels", 20) or 0)
    info: Dict[str, Any] = {"method": method}

    if method == "Random walker":
        from skimage.segmentation import random_walker

        report(0.1, "seeding")
        mask = _mask(volume, params)
        markers, count = _markers(volume, mask, float(params.get("seed_depth", 2.0)))
        info["seeds"] = count
        if count == 0:
            return np.zeros(volume.shape, dtype=np.uint32), {**info, "note": "no seeds: nothing above the threshold"}
        check()
        report(0.3, "solving")
        # beta: how strongly the diffusion is stopped by an intensity step
        solved = random_walker(_normalized(volume), markers, beta=float(params.get("beta", 130.0)),
                               mode="cg_j", tol=float(params.get("tolerance", 1e-3)))
        check()
        labels = np.where(solved > 1, solved - 1, 0)

    elif method == "Active contour (geodesic)":
        from skimage.morphology import label
        from skimage.segmentation import inverse_gaussian_gradient, morphological_geodesic_active_contour

        report(0.2, "edge map")
        edges = inverse_gaussian_gradient(_normalized(volume), alpha=float(params.get("alpha", 100.0)),
                                          sigma=float(params.get("edge_sigma", 2.0)))
        check()
        report(0.4, "evolving")
        contour = morphological_geodesic_active_contour(
            edges, num_iter=int(params.get("iterations", 30)), init_level_set=_mask(volume, params).astype(np.int8),
            smoothing=int(params.get("smoothing", 1)), balloon=float(params.get("balloon", 0.0)),
            threshold=float(params.get("edge_threshold", 0.69)))
        check()
        labels = label(contour > 0, connectivity=1)

    elif method == "Superpixels (SLIC)":
        from skimage.segmentation import slic

        report(0.3, "clustering")
        labels = slic(_normalized(volume), n_segments=max(2, int(params.get("n_segments", 200))),
                      compactness=float(params.get("compactness", 0.1)), channel_axis=None,
                      start_label=1, enforce_connectivity=True)
        check()

    elif method == "Superpixels (Felzenszwalb)":
        from skimage.segmentation import felzenszwalb

        # felzenszwalb is a 2D method: run it plane by plane and keep the
        # planes apart, so each is its own over-segmentation
        report(0.2, "graph merging")
        planes = []
        offset = 0
        for z in range(volume.shape[0]):
            check()
            seg = felzenszwalb(_normalized(volume[z]), scale=float(params.get("scale", 100.0)),
                               sigma=float(params.get("edge_sigma", 0.8)),
                               min_size=max(1, int(params.get("min_size", 20))))
            seg = seg.astype(np.int64) + 1 + offset
            offset = int(seg.max())
            planes.append(seg)
            report(0.2 + 0.7 * (z + 1) / volume.shape[0], "graph merging")
        labels = np.stack(planes)
        info["note"] = "Felzenszwalb is a 2D method: each plane is segmented on its own"

    else:   # Watershed (compact)
        from skimage.segmentation import watershed

        report(0.1, "seeding")
        mask = _mask(volume, params)
        markers, count = _markers(volume, mask, float(params.get("seed_depth", 2.0)))
        info["seeds"] = count
        if count == 0:
            return np.zeros(volume.shape, dtype=np.uint32), {**info, "note": "no seeds: nothing above the threshold"}
        markers[markers == 1] = 0   # the background marker is not an object here
        check()
        report(0.4, "flooding")
        # compactness pulls the regions towards compact shapes, which is what
        # the plain watershed cannot be asked for
        labels = watershed(-_normalized(volume), markers=markers, mask=mask,
                           compactness=float(params.get("compactness", 0.01)))

    check()
    report(0.9, "labelling")
    out = _clean(labels, min_voxels)
    info["labels"] = int(out.max())
    return out, info
