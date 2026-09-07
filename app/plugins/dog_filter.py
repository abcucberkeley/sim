"""Example SIRIUS plugin: a difference-of-Gaussians band-pass filter.

Copy this file to ~/.sirius/plugins (or set SIRIUS_PLUGIN_DIRS) and edit it;
Process ▸ Reload plugins picks the change up.
"""

import numpy as np

STEP = {
    "kind": "dog_filter",
    "name": "Difference of Gaussians",
    "group": "Intensity",
    "params": [
        {"key": "sigma_lo", "label": "σ low", "type": "double", "default": 1.0, "min": 0.1, "max": 50.0,
         "unit": "px", "help": "Width of the narrow Gaussian; structures smaller than this are smoothed away"},
        {"key": "sigma_hi", "label": "σ high", "type": "double", "default": 4.0, "min": 0.1, "max": 100.0,
         "unit": "px", "help": "Width of the wide Gaussian; the local background it estimates is subtracted"},
        {"key": "in_3d", "label": "Filter in 3D", "type": "bool", "default": False,
         "help": "Also smooth along z (with σ scaled by the voxel aspect)"},
        {"key": "clip", "label": "Clip negatives", "type": "bool", "default": True},
    ],
    "separable_over_t": True,
}


def _gaussian(volume, sigma, in_3d, meta):
    """scipy when available, a separable numpy kernel otherwise."""
    try:
        from scipy.ndimage import gaussian_filter
        if in_3d:
            voxel = meta.get("voxel_um") or [1.0, 1.0, 1.0]
            sz = sigma * voxel[0] / max(voxel[2], 1e-9)
            return gaussian_filter(volume, (sz, sigma, sigma))
        return np.stack([gaussian_filter(p, sigma) for p in volume])
    except ImportError:
        radius = max(1, int(3 * sigma))
        x = np.arange(-radius, radius + 1, dtype=np.float32)
        k = np.exp(-0.5 * (x / sigma) ** 2)
        k /= k.sum()
        out = volume.astype(np.float32, copy=True)
        for axis in (1, 2) + ((0,) if in_3d else ()):
            out = np.apply_along_axis(lambda v: np.convolve(v, k, mode="same"), axis, out)
        return out


def run(data, params, meta, ctx):
    """# Difference of Gaussians

    Band-pass filter: the image smoothed with a narrow Gaussian minus the
    image smoothed with a wide one,

    $$ I' = G_{\\sigma_{lo}} * I - G_{\\sigma_{hi}} * I $$

    which removes noise below σ low and background above σ high. Structures
    between the two scales (roughly 2 σ low to 2 σ high in diameter) are kept.

    ## Parameters

    | Parameter | Explanation |
    |---|---|
    | **σ low** <br> 0.1 – 50 px | Noise scale; 1 px keeps everything but shot noise. |
    | **σ high** <br> 0.1 – 100 px | Background scale; larger keeps bigger structures. |
    | **Filter in 3D** <br> on · off | Smooth along z too, with σ scaled by the voxel aspect ratio. |
    | **Clip negatives** <br> on · off | Set values below 0 (darker than the background) to 0. |
    """
    lo, hi = float(params["sigma_lo"]), float(params["sigma_hi"])
    if hi <= lo:
        raise ValueError("σ high must be larger than σ low")
    c, t = data.shape[:2]
    out = np.empty_like(data, dtype=np.float32)
    n = c * t
    k = 0
    for ci in range(c):
        for ti in range(t):
            if ctx.cancelled():
                raise RuntimeError("cancelled")
            vol = data[ci, ti]
            out[ci, ti] = _gaussian(vol, lo, params["in_3d"], meta) - _gaussian(vol, hi, params["in_3d"], meta)
            k += 1
            ctx.progress(k / n, f"channel {ci} t {ti}")
    if params["clip"]:
        np.maximum(out, 0.0, out=out)
    facts = {"σ low": f"{lo:g} px", "σ high": f"{hi:g} px", "Band": f"≈ {2 * lo:g} – {2 * hi:g} px"}
    return out, {"summary": f"DoG σ {lo:g} – {hi:g} px", "facts": facts}
