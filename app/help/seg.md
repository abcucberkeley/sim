---
title: Segmentation
figure: Tile grid with overlap halo
---

Runs any TorchScript model tile-wise over the volume and stitches predictions back with overlap blending. Probabilities are then turned into instance labels by the post-processing step; a model family (Cellpose, micro-SAM) returns labels directly and skips it.

$$
p = \sigma\left(f_\theta(x)\right),\qquad L = \text{watershed}(p_{\text{fg}} > \tau,\; p_{\text{boundary}})
$$

## Parameters

| Parameter | Explanation |
|---|---|
| **Model** <br> file · hub · family | A TorchScript / ONNX file taking (1, 1, Z, Y, X) float32 and returning (1, K, Z, Y, X) logits or probabilities; `hf:<repo>[:<file>]`, a file fetched from Hugging Face on first run and cached in `~/.sirius/models`; or a model family that returns labels itself: `cellpose:<model>` (`default` for the installed Cellpose's built-in model, one of its model names, or a custom file) and `microsam:<model_type>` (vit_b_lm, vit_l_lm, vit_b_em_organelles, …). *Hub…* offers the families, installs a missing package on request, fetches weights, and searches Hugging Face. The model runs in the Python worker (locally or on the HPC backend), never inside the app process. |
| **Input channel** <br> channel | The channel fed to the model, normalised to 0 – 1 by percentiles. |
| **Tile · Overlap** <br> GPU bound | Tile must fit GPU memory; overlap should exceed the receptive-field radius so edges are not visible in the stitched prediction. |
| **Threshold τ** <br> 0 – 1 | Foreground probability cut. Lower recovers dim objects, higher separates touching ones. |
| **Post-processing** <br> watershed · components · none | Watershed on the boundary channel splits touching nuclei; connected components is faster but merges neighbours; none keeps the raw probabilities. |
| **Label opacity** <br> 10 – 90 % | Opacity of the label overlay in the viewer. |

## Choosing a model

For nuclei and cells in fluorescence, Cellpose (`cellpose:default`) is the safest first choice: trained on microscopy, 2D and 3D, no prompts; Cellpose 4 ships one built-in model (about 1.2 GB, fetched from the authors on first use, no account needed), Cellpose 3 offers `cyto3`, `nuclei` and more. micro-SAM adapts the Segment Anything models to light and electron microscopy (`vit_b_lm`, `vit_b_em_organelles`) and is the better generalist when the specimen is unusual. A family whose package is missing on the worker's host is installed from the hub dialog after a confirmation (`pip install cellpose`; `conda install -c conda-forge micro_sam` in a conda environment). Plain SAM checkpoints on Hugging Face are in the transformers format, expect prompts and are not run directly; SAM 3 there is also gated, so it needs an accepted licence and an access token (Hub… ▸ Token…, or Preferences).

## Note

Every label gets a mean foreground probability as its confidence; labels below 0.6, touching the border or far from the median size are flagged for review in the cleanup panel.
