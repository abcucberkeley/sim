---
title: Stitch tiles
figure: Tile grid, pairwise shifts and the fused mosaic
---

A specimen that does not fit in one field is imaged as overlapping tiles whose nominal origins come from the stage — good to within a few percent of the field, not to a voxel. Stitching registers every overlapping pair with masked cross-correlation, solves for all tile origins at once and fuses the tiles onto one canvas.

$$
\min_{\mathbf{p}} \sum_{(i,j)} w_{ij}\,\big\|(\mathbf{p}_j - \mathbf{p}_i) - \mathbf{d}_{ij}\big\|^2 + \epsilon \sum_i \|\mathbf{p}_i - \mathbf{p}^{\,0}_i\|^2
$$

## Parameters

| Parameter | Explanation |
|---|---|
| **Tiles** <br> files + positions | One file per tile with its nominal (z, y, x) origin in voxels, or a layout file. |
| **Search radius** <br> z, y, x voxels | How far a tile may move from its nominal position; bounds the correlation search, so keep it near the stage's real repeatability. |
| **Minimum correlation** <br> 0 – 1 | Pair matches scoring below this are dropped from the global fit instead of dragging the mosaic. |
| **Mask background** <br> on · off | Ignore voxels at or below the background level when correlating: the zero fill of a deskew, unilluminated borders. |
| **Blend** <br> overwrite · average · feather · maximum | How overlapping voxels are combined on the canvas. Feather (distance-to-border weighted) hides seams. |

## Note

The alignment panel shows a checkerboard of fixed and moving tile in their overlap, the tile map with the selected pair and the pairwise shift statistics (mean and maximum displacement, normalised cross-correlation).
