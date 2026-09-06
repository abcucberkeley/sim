---
title: Resample
figure: Anisotropic grid resampled to isotropic voxels
---

Interpolates the volume onto a new voxel grid — isotropic voxels for rendering and segmentation, coarser voxels to shrink a dataset, or an exact factor to match another channel.

$$
O(\mathbf{p}) = I\!\left(\mathbf{A}\,\mathbf{p} + \mathbf{b}\right),\qquad \mathbf{A} = \text{diag}\!\left(\frac{dz'}{dz}, \frac{dy'}{dy}, \frac{dx'}{dx}\right)
$$

## Parameters

| Parameter | Explanation |
|---|---|
| **Target voxel** <br> z, y, x µm | Voxel size of the output grid; 0 keeps that axis unchanged. *Isotropic* sets all three to the smallest input voxel. |
| **Interpolation** <br> nearest · linear · cubic | Nearest keeps labels intact; linear is the fast default; cubic preserves fine detail when up-sampling. |
| **Anti-alias** <br> on · off | Box-filter before down-sampling by more than 2× so that fine structures do not alias. |

## Note

The output voxel size is written to the metadata, so scale bars and later steps stay correct.
