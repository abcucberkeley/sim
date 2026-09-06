---
title: Contrast adjustment
figure: Histogram with percentile handles
---

Linear rescale between two percentiles of the intensity histogram, then an optional gamma. Display-only unless baked into the export.

$$
I' = \left(\frac{\text{clip}(I, p_{lo}, p_{hi}) - p_{lo}}{p_{hi} - p_{lo}}\right)^{1/\gamma}
$$

## Parameters

| Parameter | Explanation |
|---|---|
| **Percentiles** <br> 0.1 – 99.9 | Robust min/max that ignore hot pixels and background. The histogram panel greys out the tails that are clipped. |
| **Gamma** <br> 0.5 – 2 | Below 1 lifts dim structures; above 1 emphasises bright ones. |
| **Per channel** <br> on · off | Compute the percentiles for each channel separately (the default) or once over all channels, which preserves relative brightness between them. |

## Note

The output is in the range 0 – 1. Export with *uint16 (rescale)* to keep 16-bit precision, or *float32* to keep the exact values.
