#ifndef SIRIUS_CUDA_KERNELS_HPP
#define SIRIUS_CUDA_KERNELS_HPP

// Device kernels used by the buffer layer and the readers. Declared here with
// plain C++ types so the callers can be compiled by the host compiler; only
// kernels.cu needs nvcc. Every function enqueues on `stream` and returns.

#include <cuda_runtime.h>

#include <complex>
#include <cstddef>
#include <cstdint>

namespace sirius::cuda {

    template <typename T>
    void fillDevice(T* dst, std::size_t n, T value, cudaStream_t stream);

    template <typename From, typename To>
    void convertDevice(const From* src, To* dst, std::size_t n, cudaStream_t stream);

    // Multiply n complex<double> by a real scalar (used by ifft normalization).
    void scaleComplexDouble(std::complex<double>* p, std::size_t n, double scale, cudaStream_t stream);

    // Multiply n doubles by a scalar (used by irfft normalization).
    void scaleDouble(double* p, std::size_t n, double scale, cudaStream_t stream);

} // namespace sirius::cuda

#endif // SIRIUS_CUDA_KERNELS_HPP
