---
title: Channel merge
figure: Channel → colour mapping
---

Maps each channel to a display colour and blends them into one RGB image.

$$
\text{RGB} = \sum_c \text{LUT}_c\big(I_c\big)
$$

## Parameters

| Parameter | Explanation |
|---|---|
| **Colours** <br> per channel | Display colour of each channel; defaults follow the emission wavelength (405 blue, 488 green, 561 magenta, 640 orange). |
| **Blend** <br> additive · screen · max | Additive is physically faithful; screen avoids clipping; max keeps the brightest channel only. $\text{screen}(a, b) = 1 - (1 - a)(1 - b)$ |
| **Normalise** <br> on · off | Rescale each channel to 0 – 1 before blending (use Contrast first for control over the window). |

## Note

The output has three channels flagged as RGB; steps that need intensities (segmentation, deconvolution) should run before the merge.
