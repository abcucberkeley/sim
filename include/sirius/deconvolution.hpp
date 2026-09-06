#ifndef SIRIUS_DECONVOLUTION_HPP
#define SIRIUS_DECONVOLUTION_HPP

#include <functional>
#include <vector>

#include "sirius/buffer.hpp"
#include "sirius/device.hpp"

// Richardson-Lucy deconvolution with an optional total-variation prior
// (Dey et al. 2006), FFT based. The PSF is centred, normalized to unit sum
// and embedded in a grid of the image size (wrapped so the centre sits at
// the origin), so the blur is a circular convolution of the volume padded
// to the next fast FFT size.

namespace sirius {

    struct DeconvolutionOptions {
        int iterations = 20;
        double tvLambda = 0.0;                 // 0 = plain Richardson-Lucy
        // Stop early when the relative change ||o_k+1 - o_k|| / ||o_k|| drops
        // below this (0 = never).
        double stopRelativeChange = 0.0;
        Device device = Device::cpu();         // CUDA runs the FFTs on the device (falls back to CPU when unavailable)
        // Called after every iteration; return false to stop.
        std::function<bool(int iteration, double relativeChange)> onIteration;
    };

    struct DeconvolutionResult {
        std::vector<double> relativeChange;    // one per iteration run
        int iterations = 0;
        bool stoppedEarly = false;
        bool ranOnGpu = false;
    };

    // In place on a host (z, y, x) float volume (rank 2 = one plane) with a
    // (pz, py, px) PSF of the same voxel size; the PSF may be smaller or
    // larger than the volume.
    DeconvolutionResult richardsonLucy(BufferView<float> image, BufferView<const float> psf,
                                       const DeconvolutionOptions& options = {});

    // Gaussian approximation of a widefield PSF for a (pz, py, px) grid with
    // voxel (dz, dxy) um: sigma_xy = 0.21 lambda / NA, sigma_z = 0.66 lambda
    // nimm / NA^2 (Zhang et al. 2007). Odd extents keep the peak centred.
    Buffer<float> gaussianPsf(Index pz, Index py, Index px, double dzUm, double dxyUm, double na,
                              double wavelengthNm, double nimm);

} // namespace sirius

#endif // SIRIUS_DECONVOLUTION_HPP
