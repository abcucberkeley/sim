---
title: Classical segmentation
figure: Top-hat → smooth → threshold → clean → instances
---

The conventional recipe for nuclei and blobs, no model needed: the background is flattened with a white top-hat, the image is smoothed, a threshold turns it into a mask, the mask is cleaned, and touching objects are split by a distance watershed. Every stage is a parameter, so the step can be tuned in seconds and it runs on any machine.

$$
M = \left[\, G_\sigma * \big(I - (I \circ B_r)\big) > \tau \,\right],\qquad L = \text{watershed}\big(-d(M)\big)
$$

## Parameters

| Parameter | Explanation |
|---|---|
| **Background radius** <br> px | White top-hat with a box of this radius: structures larger than the box are treated as background and removed. Use about the object size; 0 leaves the background alone. |
| **Smoothing σ** <br> px | Gaussian blur before the threshold; suppresses noise that would otherwise fragment the mask. |
| **Threshold** <br> Otsu · Manual · Percentile · Local mean | Otsu picks the cut that best separates two intensity classes; Percentile keeps the brightest fraction; Manual is a fixed value in the (smoothed) intensity units; Local mean compares each pixel with the mean of its window, which follows an uneven background. |
| **Local window · ratio · offset** <br> local mean | Window side in pixels; foreground where value > ratio × local mean + offset. |
| **Opening radius** <br> px | Binary opening (erode then dilate) drops specks and thin bridges narrower than the radius. |
| **Fill holes** <br> on · off | Background enclosed by an object, per plane, becomes object: hollow nuclei stay one object. |
| **Instances** <br> watershed · components | The distance watershed seeds on the maxima of the distance transform (at least *Seed distance* apart) and splits touching objects; connected components keeps every blob as one object. |
| **Min. voxels** | Objects smaller than this are dropped. |

## Note

Start with Otsu, σ 1, opening 1 and the watershed; switch to Local mean when the illumination is uneven, or add a top-hat of about the object radius. For crowded or textured data a learned model (the Segmentation step with Cellpose) separates objects the classical route cannot.
