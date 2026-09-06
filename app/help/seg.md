---
title: Segmentation with a Torch model
figure: Tile grid with overlap halo
---

Runs any TorchScript model tile-wise over the volume and stitches predictions back with overlap blending. Probabilities are then turned into instance labels by the post-processing step.

$$
p = \sigma\left(f_\theta(x)\right),\qquad L = \text{watershed}(p_{\text{fg}} > \tau,\; p_{\text{boundary}})
$$

## Parameters

| Parameter | Explanation |
|---|---|
| **Torch model** <br> .pt file | A TorchScript module taking (1, 1, Z, Y, X) float32 and returning (1, K, Z, Y, X) logits or probabilities. The model runs in the Python worker (locally or on the HPC backend), never inside the app process. |
| **Input channel** <br> channel | The channel fed to the model, normalised to 0 – 1 by percentiles. |
| **Tile · Overlap** <br> GPU bound | Tile must fit GPU memory; overlap should exceed the receptive-field radius so edges are not visible in the stitched prediction. |
| **Threshold τ** <br> 0 – 1 | Foreground probability cut. Lower recovers dim objects, higher separates touching ones. |
| **Post-processing** <br> watershed · components · none | Watershed on the boundary channel splits touching nuclei; connected components is faster but merges neighbours; none keeps the raw probabilities. |
| **Label opacity** <br> 10 – 90 % | Opacity of the label overlay in the viewer. |

## Note

Every label gets a mean foreground probability as its confidence; labels below 0.6, touching the border or far from the median size are flagged for review in the cleanup panel.
