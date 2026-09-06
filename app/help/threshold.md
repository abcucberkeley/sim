---
title: Threshold segmentation
figure: Histogram with the threshold and the two classes
---

Separates foreground from background by an intensity threshold and labels the connected foreground regions. Otsu's method picks the threshold that minimises the within-class variance of the histogram; a manual value or a percentile can replace it.

$$
\tau^\ast = \arg\max_\tau\; \omega_0(\tau)\,\omega_1(\tau)\,\big[\mu_0(\tau) - \mu_1(\tau)\big]^2
$$

## Parameters

| Parameter | Explanation |
|---|---|
| **Method** <br> Otsu · manual · percentile | How the threshold is chosen. Otsu assumes a bimodal histogram; use a percentile for sparse signal on a dark background. |
| **Threshold** <br> value | Manual cut (in the input's units) or the percentile of voxels below the cut. |
| **Channel** <br> channel | The channel to threshold. |
| **Minimum size** <br> voxels | Components smaller than this are discarded as noise. |
| **Split touching** <br> on · off | Run a distance-transform watershed on the mask so touching objects become separate labels. $L = \text{watershed}(-\text{EDT}(M),\; \text{seeds})$ |

## Note

Labels from this step carry no confidence; the cleanup panel flags only size and border contact for them.
