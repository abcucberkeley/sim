---
title: Flat-field correction
figure: Illumination profile and its correction
---

Divides every plane by the illumination profile so that a uniform sample looks uniform across the field. The profile comes from a flat-field image (a dye slide or the average of many fields); a dark frame removes the camera offset first.

$$
I' = \frac{I - D}{F - D}\cdot\overline{(F - D)}
$$

## Parameters

| Parameter | Explanation |
|---|---|
| **Flat image** <br> file | Image of a uniform sample with the same objective, channel and camera settings; blurred flat fields are fine, noisy ones add noise everywhere. |
| **Dark image** <br> file · constant | Camera offset frame taken with the shutter closed, or a constant offset when no frame exists. |
| **Normalise** <br> mean · max | Rescale the corrected image by the mean (keeps average brightness) or the maximum (keeps the brightest region) of the flat field. |

## Note

Apply flat-field correction before stitching so tile edges match; the feather blend cannot hide a vignetted field.
