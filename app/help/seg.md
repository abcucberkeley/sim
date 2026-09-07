---
title: Segmentation with a Torch model
figure: Tile grid with overlap halo
---

Runs any TorchScript model tile-wise over the volume and stitches predictions back with overlap blending. Probabilities are then turned into instance labels by the post-processing step; a model family (Cellpose, micro-SAM) returns labels directly and skips it.

$$
p = \sigma\left(f_\theta(x)\right),\qquad L = \text{watershed}(p_{\text{fg}} > \tau,\; p_{\text{boundary}})
$$

## Parameters

| Parameter | Explanation |
|---|---|
| **Model** <br> file · hub · family | A TorchScript / ONNX file taking (1, 1, Z, Y, X) float32 and returning (1, K, Z, Y, X) logits or probabilities; `hf:<repo>[:<file>]`, a file fetched from Hugging Face on first run and cached in `~/.sirius/models`; or a model family that returns labels itself: `cellpose:<model>` (cyto3, nuclei, cyto2, or a custom file) and `microsam:<model_type>` (vit_b_lm, vit_l_lm, vit_b_em_organelles, …). *Hub…* searches Hugging Face, downloads with progress and fills this in. The model runs in the Python worker (locally or on the HPC backend), never inside the app process. |
| **Input channel** <br> channel | The channel fed to the model, normalised to 0 – 1 by percentiles. |
| **Tile · Overlap** <br> GPU bound | Tile must fit GPU memory; overlap should exceed the receptive-field radius so edges are not visible in the stitched prediction. |
| **Threshold τ** <br> 0 – 1 | Foreground probability cut. Lower recovers dim objects, higher separates touching ones. |
| **Post-processing** <br> watershed · components · none | Watershed on the boundary channel splits touching nuclei; connected components is faster but merges neighbours; none keeps the raw probabilities. |
| **Label opacity** <br> 10 – 90 % | Opacity of the label overlay in the viewer. |

## Choosing a model

For nuclei and cells in fluorescence, Cellpose (`cellpose:cyto3`, `cellpose:nuclei`) is the safest default: it is trained on microscopy, handles 2D and 3D and needs no prompt. micro-SAM adapts the Segment Anything models to light and electron microscopy (`vit_b_lm`, `vit_b_em_organelles`) and is the better generalist when the specimen is unusual. Both families need their package on the worker's host (`pip install cellpose`, `pip install micro-sam`); the parameters panel says which are installed. Plain SAM checkpoints from Hugging Face expect prompts and are not run directly.

## Note

Every label gets a mean foreground probability as its confidence; labels below 0.6, touching the border or far from the median size are flagged for review in the cleanup panel.
