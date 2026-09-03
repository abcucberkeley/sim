#ifndef SIRIUS_FFT_HPP
#define SIRIUS_FFT_HPP

#include <complex>
#include <memory>
#include <string>
#include <vector>

#include "sirius/buffer.hpp"
#include "sirius/device.hpp"
#include "sirius/fft_common.hpp"
#include "sirius/tensor_util.hpp"

namespace sirius {

    // Planned complex-to-complex double precision FFT, batched over `howmany`
    // contiguous transforms. Backed by FFTW on Device::cpu() and by cuFFT on
    // Device::cuda(n); the same object API is used for both, so algorithms
    // written against BufferView run unchanged on either device. PlanRigor
    // (see fft_common.hpp) only affects the FFTW backend.
    class FFT {
    public:
        // dims: {n}                 for 1D
        //       {rows, cols}        for 2D
        //       {depth, rows, cols} for 3D
        explicit FFT(std::vector<int> dims, int howmany = 1, PlanRigor rigor = PlanRigor::Measure,
                     Device device = Device::cpu());
        ~FFT();

        // delete copy constructors
        FFT(const FFT&) = delete;
        FFT& operator=(const FFT&) = delete;

        // move constructors
        FFT(FFT&&) noexcept;
        FFT& operator=(FFT&&) noexcept;

        Device device() const noexcept;
        const std::vector<int>& dims() const noexcept;
        int howmany() const noexcept;
        Index size() const noexcept;        // product(dims) * howmany: elements per call
        Shape shape() const;                // {howmany, dims...} (howmany omitted when 1)

        // Raw interface. Pointers must reference memory on device(): host
        // memory for the CPU plan, device memory for a CUDA plan. On CUDA the
        // call is asynchronous on `stream`.
        void fft(const std::complex<double>* in, std::complex<double>* out,
                 const Stream& stream = Stream::null()) const;
        void ifft(const std::complex<double>* in, std::complex<double>* out,
                  const Stream& stream = Stream::null()) const;

        // Buffer interface: element counts must equal size() and both views
        // must live on device(). `normalize` divides by product(dims).
        void fft(BufferView<const std::complex<double>> in, BufferView<std::complex<double>> out,
                 const Stream& stream = Stream::null()) const;
        void ifft(BufferView<const std::complex<double>> in, BufferView<std::complex<double>> out,
                  bool normalize = false, const Stream& stream = Stream::null()) const;

        // Convenience functions for eigen (host memory; CPU plans only)
        template<int Rank>
        void fft(const TensorXcd<Rank>& in, TensorXcd<Rank>& out) const;

        template<int Rank>
        void ifft(const TensorXcd<Rank>& in, TensorXcd<Rank>& out, bool normalize = false) const;

        // Load/Save FFTW wisdom from file (no effect on cuFFT plans)
        static void loadWisdom(const std::string& path);
        static void saveWisdom(const std::string& path);

    private:
        // Use Pimpl (pointer to implementation) pattern for fftw plan vars
        // otherwise fftw details would have to be exposed to the consumer of the header
        struct Impl;
        std::unique_ptr<Impl> impl_;
    };
} // namespace sirius

#endif // SIRIUS_FFT_HPP
