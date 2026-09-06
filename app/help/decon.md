---
title: Richardson–Lucy deconvolution
figure: Convergence curve and stopping criterion
---

Iteratively estimates the object that, blurred by the PSF, best explains the image under Poisson noise. Each iteration multiplies the estimate by a correction ratio; more iterations sharpen and eventually amplify noise.

$$
\hat{o}^{(k+1)} = \hat{o}^{(k)} \cdot \left[ \frac{i}{\hat{o}^{(k)} \ast h} \ast h^{\star} \right]
$$

## Parameters

| Parameter | Explanation |
|---|---|
| **Iterations** <br> 10 – 50 | Stop when the relative change drops below ~10⁻⁴ or ringing appears around bright structures. The convergence panel plots $\|\hat{o}^{(k+1)} - \hat{o}^{(k)}\| / \|\hat{o}^{(k)}\|$ per iteration. |
| **TV regularisation** <br> 0 – 0.01 | Total-variation penalty $\lambda$ that suppresses noise amplification at the cost of fine texture. $\min_o\; -\log p(i\mid o) + \lambda \,\|\nabla o\|_1$ |
| **PSF** <br> file · theoretical | Measured bead PSF or a Gaussian model from NA, wavelength and immersion index; must match the channel wavelength and objective. The PSF is centred, normalised to unit sum and embedded in a grid of the image size. |
| **Stop at relative change** <br> 0 – 10⁻³ | Early-stopping threshold on the relative change; 0 always runs every iteration. |

## Note

Every iteration costs two FFT-based convolutions of the volume padded to the next fast size. The residual panel shows $i - \hat{o}^{(k)} \ast h$, which should look like noise when the model fits.
