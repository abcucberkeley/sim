#ifndef SIRIUS_SIM_RECONSTRUCTION_HPP
#define SIRIUS_SIM_RECONSTRUCTION_HPP

#include <array>
#include <complex>
#include <memory>
#include <string>
#include <vector>

#include "sirius/buffer.hpp"
#include "sirius/device.hpp"
#include "sirius/fft_common.hpp"
#include "sirius/otf.hpp"
#include "sirius/sim_parameters.hpp"

namespace sirius {

    // Per-direction results of the pattern-vector search and modulation
    // amplitude fit of the last reconstruct() call.
    struct SimFit {
        std::vector<std::array<double, 2>> k0;                // (ndirs) fitted {kx, ky}, 1/um
        std::vector<std::vector<std::complex<double>>> amps;  // (ndirs, norders); amps[d][0] == 1
    };

    // 3-beam structured illumination reconstruction (Gustafsson et al. 2008),
    // matching the cudasirecon algorithm: preprocessing, band separation,
    // pattern-vector / modulation-amplitude fitting, generalized Wiener
    // filtering and real-space assembly.
    //
    // The same object API runs on the CPU (FFTW + OpenMP) or on a CUDA device
    // (cuFFT + kernels); pass Device::cuda(n) and device-resident input to
    // run on the GPU -- everything else is identical. FFT plans and work
    // buffers are created lazily for the input shape and reused across calls
    // (e.g. over a time series), so construct once and reconstruct many.
    class SimReconstructor {
    public:
        // The OTF must be radially averaged with at least norders orders
        // (see loadOTF below). PlanRigor affects only the FFTW backend.
        SimReconstructor(SIMParameters params, OTFRadiallyAveraged otf,
                         Device device = Device::cpu(), PlanRigor rigor = PlanRigor::Measure);
        ~SimReconstructor();

        SimReconstructor(const SimReconstructor&) = delete;
        SimReconstructor& operator=(const SimReconstructor&) = delete;
        SimReconstructor(SimReconstructor&&) noexcept;
        SimReconstructor& operator=(SimReconstructor&&) noexcept;

        Device device() const noexcept;

        // raw: (ndirs*nphases*nz, ny, nx) camera frames on device(), in the
        // standard direction->z->phase section order (fast_si selects the
        // z->direction->phase order instead). nx and ny must be even.
        // Returns the (z_zoom*nz, zoomfact*ny, zoomfact*nx) super-resolution
        // volume on device(); all enqueued work has completed on return.
        Buffer<double> reconstruct(BufferView<const double> raw);

        // Convenience for Buffers and host Eigen tensors.
        template <typename Src>
        Buffer<double> reconstruct(const Src& src) {
            return reconstruct(BufferView<const double>(toConstView(src)));
        }

        // Fit diagnostics of the last reconstruct() call.
        const SimFit& lastFit() const noexcept;

    private:
        struct Impl;
        std::unique_ptr<Impl> impl_;
    };

    // Load a radially averaged OTF TIFF, deriving its reciprocal-space
    // sampling from the file dimensions and the acquisition parameters
    // (dkr = 1/(dx*(nkr-1)*2), dkz = 1/(dz_psf*nzotf)), as cudasirecon's
    // determine_otf_dimensions does for otfRA files.
    OTFRadiallyAveraged loadOTF(const std::string& filename, const SIMParameters& p);

} // namespace sirius

#endif // SIRIUS_SIM_RECONSTRUCTION_HPP
