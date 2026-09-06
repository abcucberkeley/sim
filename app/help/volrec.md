---
title: Volume reconstruction
figure: Transfer function editor
---

Resamples the stack to isotropic voxels and renders it with ray casting through a transfer function. Each ray accumulates colour and opacity front to back; the transfer function maps intensity to opacity so dim background stays transparent and bright structures become solid.

$$
C = \sum_{i} T_i\, c_i \alpha_i,\qquad T_i = \prod_{j<i}(1-\alpha_j)
$$

## Parameters

| Parameter | Explanation |
|---|---|
| **Method** <br> ray casting · maximum intensity | Ray casting composites opacity along each ray; maximum intensity projection shows the brightest voxel and needs no transfer function. |
| **Resample to** <br> isotropic µm | Target voxel size of the isotropic grid; by default the lateral pixel size, so z is interpolated up to it. |
| **Step size** <br> 0.25 – 1 voxel | Ray sampling distance; smaller is smoother and slower. |
| **Opacity ramp** <br> lo – hi | Intensities below *lo* are transparent, above *hi* opaque, linear in between. $\alpha(I) = \text{clip}\left(\frac{I - I_{lo}}{I_{hi} - I_{lo}}, 0, 1\right)$ |

## Note

Selecting a Volume reconstruction step switches the viewer to 3D; yaw and pitch, the presets and the Z clip range live in the viewer overlays.
