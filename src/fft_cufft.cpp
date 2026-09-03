// cuFFT backend for sirius::FFT (compiled only with SIRIUS_ENABLE_CUDA).

#include "fft_backend.hpp"
#include "cuda_check.hpp"
#include "cuda/kernels.hpp"

#include <cufft.h>

#include <numeric>
#include <stdexcept>
#include <string>

namespace sirius::detail {

    namespace {
        const char* cufftErrorName(cufftResult r) {
            switch (r) {
                case CUFFT_SUCCESS:        return "CUFFT_SUCCESS";
                case CUFFT_INVALID_PLAN:   return "CUFFT_INVALID_PLAN";
                case CUFFT_ALLOC_FAILED:   return "CUFFT_ALLOC_FAILED";
                case CUFFT_INVALID_TYPE:   return "CUFFT_INVALID_TYPE";
                case CUFFT_INVALID_VALUE:  return "CUFFT_INVALID_VALUE";
                case CUFFT_INTERNAL_ERROR: return "CUFFT_INTERNAL_ERROR";
                case CUFFT_EXEC_FAILED:    return "CUFFT_EXEC_FAILED";
                case CUFFT_SETUP_FAILED:   return "CUFFT_SETUP_FAILED";
                case CUFFT_INVALID_SIZE:   return "CUFFT_INVALID_SIZE";
                case CUFFT_INVALID_DEVICE: return "CUFFT_INVALID_DEVICE";
                case CUFFT_NO_WORKSPACE:   return "CUFFT_NO_WORKSPACE";
                case CUFFT_NOT_IMPLEMENTED:return "CUFFT_NOT_IMPLEMENTED";
                case CUFFT_NOT_SUPPORTED:  return "CUFFT_NOT_SUPPORTED";
                default:                   return "CUFFT_UNKNOWN";
            }
        }

        void check(cufftResult r, const char* what) {
            if (r != CUFFT_SUCCESS)
                throw CudaError(std::string(what) + ": " + cufftErrorName(r));
        }

        class CufftBackend final : public FftBackend {
        public:
            CufftBackend(const std::vector<int>& dims, int howmany, Device device) : device_(device) {
                cuda::DeviceGuard g(device.index);
                const int total = std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<int>{});
                n_ = static_cast<std::size_t>(total) * static_cast<std::size_t>(howmany);
                // inembed/onembed = nullptr: contiguous, tightly packed batches
                // laid out exactly like the FFTW plan (row-major dims, batch stride = total).
                check(cufftPlanMany(&plan_, static_cast<int>(dims.size()), const_cast<int*>(dims.data()),
                                    nullptr, 1, total, nullptr, 1, total, CUFFT_Z2Z, howmany),
                      "cufftPlanMany");
            }

            ~CufftBackend() override {
                cuda::DeviceGuard g(device_.index);
                (void)cufftDestroy(plan_);
            }

            void execute(const std::complex<double>* in, std::complex<double>* out, bool forward,
                         const Stream& stream) const override {
                if (stream.device().isCuda() && stream.device() != device_)
                    throw std::invalid_argument("FFT: stream on " + toString(stream.device()) +
                                                " used with a plan on " + toString(device_));
                cuda::DeviceGuard g(device_.index);
                // cufftSetStream mutates the plan: a plan must not be executed
                // concurrently from several threads (same restriction as cuFFT itself).
                check(cufftSetStream(plan_, cuda::handle(stream)), "cufftSetStream");
                auto* i = reinterpret_cast<cufftDoubleComplex*>(const_cast<std::complex<double>*>(in));
                auto* o = reinterpret_cast<cufftDoubleComplex*>(out);
                check(cufftExecZ2Z(plan_, i, o, forward ? CUFFT_FORWARD : CUFFT_INVERSE), "cufftExecZ2Z");
            }

            void scale(std::complex<double>* out, std::size_t n, double s, const Stream& stream) const override {
                cuda::DeviceGuard g(device_.index);
                cuda::scaleComplexDouble(out, n, s, cuda::handle(stream));
                cuda::check(cudaGetLastError(), "scale kernel launch");
            }

        private:
            cufftHandle plan_{};
            Device device_;
            std::size_t n_ = 0;
        };
    } // namespace

    std::unique_ptr<FftBackend> makeCufftBackend(const std::vector<int>& dims, int howmany, Device device) {
        return std::make_unique<CufftBackend>(dims, howmany, device);
    }

} // namespace sirius::detail
