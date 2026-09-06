---
title: Bleach correction
figure: Total intensity per frame before and after
---

Fluorophores fade under illumination, so later frames of a time series (or later phases of a SIM acquisition) are dimmer than earlier ones. Bleach correction rescales every frame so its total intensity matches a reference, removing the trend without changing spatial structure.

$$
I'_t = I_t \cdot \frac{\sum I_{\text{ref}}}{\sum I_t}
$$

## Parameters

| Parameter | Explanation |
|---|---|
| **Reference** <br> first · mean | Match every frame to the first frame (keeps absolute counts of the start) or to the mean over the series. |
| **Along** <br> t · z | The axis along which fading happens: time for live imaging, z for stacks that bleach plane by plane. |
| **Exclude background** <br> on · off | Estimate the sums from foreground voxels only so that a growing background does not mask the decay. |

## Note

Ratiometric or quantitative intensity analyses should keep the raw counts; correct a copy for display and segmentation instead.
