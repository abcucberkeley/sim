---
title: Loading data
figure: File layout: chunk grid over Z–Y–X
---

Opens the dataset lazily: only the planes needed for the current view or the running step are read. Metadata (voxel size, channels, acquisition mode) is parsed from the file header — OME-XML or ImageJ tags in a TIFF, the OME-NGFF attributes of a zarr store — and drives downstream defaults.

$$
I(c,t,z,y,x) \in \mathbb{N}^{C\times T\times Z\times Y\times X},\quad \text{uint16}
$$

## Parameters

| Parameter | Explanation |
|---|---|
| **Source** <br> file or directory | A multi-page TIFF / OME-TIFF (decoded on the GPU by nvTIFF when possible) or a zarr / N5 store. Plain TIFFs without dimension metadata ask how the pages map onto channels, time points and z planes. |
| **Read as** <br> lazy · full | Lazy reads planes on demand and keeps a bounded RAM cache; full load reads everything once — faster scrubbing, but needs the whole dataset in memory. |
| **SIM layout** <br> directions × phases | For raw structured-illumination stacks: how many pattern directions and phase steps the z axis interleaves, so the SIM step can unmix them. $Z_{\text{file}} = N_{\text{dir}} \cdot N_{\text{phase}} \cdot Z$ |

## Note

Voxel sizes and channel names can be overridden here when the file's metadata is wrong; every step downstream reads the corrected values.
