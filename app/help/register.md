---
title: Register
figure: Masked normalised cross-correlation map
---

Aligns a moving channel or time point to a fixed one by a translation, found as the peak of the masked normalised cross-correlation of Padfield (2012). Each image carries a mask of the voxels that may take part in the match, and the masking is folded into the correlation, so the score at every candidate displacement is the true normalised correlation of exactly the overlapping unmasked voxels.

$$
\text{NCC}(\mathbf{d}) = \frac{\sum_{\mathbf{x}\in\Omega_{\mathbf{d}}} \big(F(\mathbf{x}) - \bar F_{\mathbf{d}}\big)\big(M(\mathbf{x} - \mathbf{d}) - \bar M_{\mathbf{d}}\big)}{\sqrt{\sum_{\Omega_{\mathbf{d}}} (F - \bar F_{\mathbf{d}})^2 \;\sum_{\Omega_{\mathbf{d}}} (M - \bar M_{\mathbf{d}})^2}}
$$

## Parameters

| Parameter | Explanation |
|---|---|
| **Fixed** <br> channel · time point | The reference; everything else is moved onto it. |
| **Moving** <br> channel · all | Which channel (or every channel against the first time point) to register. |
| **Max shift** <br> z, y, x voxels | Search range; the correlation is computed with 12 real FFTs of the padded volume, so the cost does not grow with the range. |
| **Required overlap** <br> 0 – 1 | Displacements whose masks overlap less than this fraction of the largest overlap are rejected: tiny overlaps correlate perfectly by accident. |
| **Sub-voxel** <br> on · off | Refine the integer peak with a parabolic fit and resample; off keeps whole-voxel shifts (no interpolation). |

## Note

Everything is computed in double precision — the algorithm forms differences of large, nearly equal sums and is not usable in single precision.
