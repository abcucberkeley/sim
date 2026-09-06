---
title: Maximum projection
figure: Stack collapsed along z
---

Collapses the z axis to its brightest voxel per (y, x): the classic maximum-intensity projection. It is the einsum reduction `ctzyx -> ctyx` with *max*, kept as its own step because it is the most common reduction.

$$
P_{ctyx} = \max_{z}\; I_{ctzyx}
$$

## Parameters

| Parameter | Explanation |
|---|---|
| **Axis** <br> z · t · c | Axis to project; z by default. |
| **Range** <br> first – last | Optional sub-range of the axis to project (for example a slab of z planes). |

## Note

A projection discards depth: use it for overviews and figures, not as input for deconvolution or segmentation.
