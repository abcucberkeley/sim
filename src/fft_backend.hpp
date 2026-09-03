#ifndef SIRIUS_FFT_BACKEND_HPP
#define SIRIUS_FFT_BACKEND_HPP

// Internal: one backend per device type behind sirius::FFT.

#include <complex>
#include <cstddef>
#include <memory>
#include <vector>

#include "sirius/device.hpp"
#include "sirius/fft.hpp"

namespace sirius::detail {

    class FftBackend {
    public:
        virtual ~FftBackend() = default;
        virtual void execute(const std::complex<double>* in, std::complex<double>* out, bool forward,
                             const Stream& stream) const = 0;
        // out[i] *= scale for the full batch (used by ifft normalization).
        virtual void scale(std::complex<double>* out, std::size_t n, double scale,
                           const Stream& stream) const = 0;
    };

    std::unique_ptr<FftBackend> makeFftwBackend(const std::vector<int>& dims, int howmany, PlanRigor rigor);
    // Defined in fft_cufft.cpp (only compiled with SIRIUS_HAS_CUDA).
    std::unique_ptr<FftBackend> makeCufftBackend(const std::vector<int>& dims, int howmany, Device device);

} // namespace sirius::detail

#endif // SIRIUS_FFT_BACKEND_HPP
