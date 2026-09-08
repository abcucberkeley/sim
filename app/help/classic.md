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
| **Enhance** <br> none · blobs · tubes | What to bring out before the threshold. *Blobs* is a difference of Gaussians, a band-pass that answers to round objects about *Feature σ* across and flattens anything broader, so nuclei survive a textured or uneven background. *Tubes* is Frangi vesselness in 3D: the Hessian's eigenvalues at several widths score how tube-like each voxel is, so a filament is found whatever direction it runs in, including through z. It keeps filaments continuous where an intensity cut breaks them into dashes. |
| **Feature σ · σ max · Scales** <br> px | Blobs: the radius to respond to. Tubes: the smallest and largest width, and how many widths in between; the score is the best over the range. |
| **Background radius** <br> px | White top-hat with a box of this radius: structures larger than the box are treated as background and removed. Use about the object size; 0 leaves the background alone. |
| **Smoothing σ** <br> px | Gaussian blur before the threshold; suppresses noise that would otherwise fragment the mask. |
| **Threshold** <br> Otsu · Multi-Otsu · Manual · Percentile · Local mean · Local contrast | Otsu picks the cut that best separates two intensity classes. *Multi-Otsu* fits three classes and keeps only the brightest, which drops a mid-grey halo the single cut swallows. Percentile keeps the brightest fraction; Manual is a fixed value in the (smoothed) intensity units. *Local mean* compares each pixel with the mean of its window. *Local contrast* puts the cut *k* local standard deviations above that mean, so it follows the background level and the local noise together and does not turn a flat region into speckle. |
| **Local window · ratio · offset · contrast k** <br> local methods | Window side in pixels. Local mean: foreground where value > ratio × mean + offset. Local contrast: where value > mean + k × SD + offset. |
| **Opening radius** <br> px | Binary opening (erode then dilate) drops specks and thin bridges narrower than the radius. |
| **Fill holes** <br> on · off | Background enclosed by an object, per plane, becomes object: hollow nuclei stay one object. |
| **Instances** <br> watershed · components | The distance watershed splits touching objects; connected components keeps every connected blob as one object. |
| **Seeds** <br> distance maxima · h-maxima · blob centres | What the watershed starts from. *Distance maxima* takes every peak of the distance map at least *Seed distance* apart, so a lumpy or elongated object is often cut into pieces. *H-maxima* keeps only the peaks that stand *Seed depth* above their surroundings, so one object stays one object. Raise the depth when an object is split, lower it when two are merged. *Blob centres* instead finds the objects in the image itself: the scale-normalised Laplacian of Gaussian is evaluated over a range of sizes and peaks once at the centre of each round object, whatever its size. That is the one to use when the objects are not all the same size, which is where a distance map has to be retuned for each. |
| **Object radius · max · scales** <br> px, blob centres | The smallest and largest object radius to look for, and how many sizes in between. The detector answers to every size in the range at once. |
| **Min. voxels** | Objects smaller than this are dropped. |

## Note

Start with Otsu, σ 1, opening 1 and the watershed on h-maxima seeds. When the objects vary in size, switch the seeds to blob centres and give their radius range.

For filaments rather than compact objects the recipe is different, and these defaults are wrong for it: turn on *Tubes*, set the widths to the filament thickness (σ 0.8 – 2.0 over three scales suits the bundled SIM reconstruction), leave the smoothing and the opening off, and take *Connected components* — a distance watershed has nothing to split there. A plain intensity threshold on that data returns one object: the whole cell. When the illumination is uneven, switch the threshold to Local contrast or add a top-hat of about the object radius. When objects sit in texture, turn on the blob enhancement at their radius. When one object keeps being split, raise the seed depth. For crowded or low-contrast data a learned model (the Segmentation step with Cellpose) still separates objects the classical route cannot.
