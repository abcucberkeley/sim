---
title: scikit-image segmentation
figure: Worker → method → instance labels
---

The methods scikit-image has that this application does not implement itself. *Classical segmentation* covers everything that can be written in C++ and checked against a Python mirror voxel for voxel — the filters, the thresholds, the watersheds, the morphology — and runs on any machine with nothing installed. What is left over are a few large numerical methods that are not worth reimplementing, and they run in the Python worker instead.

The step therefore **needs the worker** (Preferences ▸ Python) with `scikit-image` installed in its interpreter (`pip install scikit-image`). It hands a volume over and gets instance labels back, exactly as the Torch *Segmentation* step does, so painting, the review table, tracking and the training-data export all treat the result the same way.

$$
\nabla \cdot \big(w \nabla p_s\big) = 0, \qquad w_{ij} = e^{-\beta (I_i - I_j)^2}, \qquad L(x) = \arg\max_s p_s(x)
$$

## Parameters

| Parameter | Explanation |
|---|---|
| **Method** | Which of the five. |
| **Random walker** | Every voxel is asked which seed a random walk from it would most likely reach first, where a step across an intensity change is unlikely. That fills outwards from the seeds and stops where the image changes, which is what makes it hold a boundary too faint or too broken for a threshold to follow — the case a watershed floods straight through. Seeds are the h-maxima of the distance transform, as in the Classical step. |
| **Diffusion stiffness · Solver tolerance** | How strongly an intensity step stops the walk: higher follows fainter boundaries, lower lets the regions spread past them. The tolerance is the linear solver's. |
| **Active contour (geodesic)** | A morphological geodesic active contour: the mask's boundary is driven towards the image's *edges*, where the Classical step's Chan–Vese is driven by the two regions' means. Use it when the object has a clear edge but an interior too uneven for a region fit. |
| **Contour steps · smoothing · Balloon · Edge sharpness · Edge σ · Edge threshold** | How many times the contour moves and how much curvature is applied each time; the balloon inflates (positive) or deflates (negative) it where the edges say nothing; the rest shape the edge map the contour follows. |
| **Superpixels (SLIC)** | Cuts the volume into roughly **Superpixels** pieces that follow the intensity, each compact. Not a segmentation on its own: an over-segmentation whose pieces are merged or measured afterwards, and a good starting point when the objects have no single threshold. |
| **Superpixels (Felzenszwalb)** | The graph-merging over-segmentation. It is a 2D method, so each plane is segmented on its own and the planes are kept apart; **Merge scale** decides how big the pieces come out. |
| **Watershed (compact)** | The seeded watershed with a compactness term, which pulls the regions towards round shapes instead of letting them follow every intensity ridge. What to reach for when the Classical step's watershed leaks along a faint ridge out of an object. |
| **Threshold** | The rough foreground the seeded methods start from; 0 asks the worker for an Otsu cut. |
| **Seed depth** | Random walker and compact watershed: how far a peak of the distance map must stand above its surroundings to seed its own object. Raise it when one object is split, lower it when two are merged. |
| **Min. voxels** | Objects smaller than this are dropped. |

## Note

Everything here is slower than *Classical segmentation*, sends the volume over the wire and needs a package installed, so reach for it when the classical recipe has actually failed rather than first. The random walker is the one that most often succeeds where a threshold cannot: weak, broken or low-contrast boundaries.

Unlike every other step, this one has no Python mirror in `sirius.workbench` and no parity fixture — the method *is* the Python, so there is nothing to compare it against.
