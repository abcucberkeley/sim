#ifndef SIRIUS_REAL_FFT_HPP
#define SIRIUS_REAL_FFT_HPP

#include "sirius/device.hpp"
#include "sirius/fft_common.hpp"
#include "sirius/tensor_util.hpp"

#include <complex>
#include <memory>
#include <stdexcept>
#include <vector>

namespace sirius {

    // Half-complex band storage used by real-to-complex 3D FFTs.
    // Shape is explicitly (nz, ny, nx/2 + 1), where nx is the original real
    // spatial width. This matches FFTW/NumPy rfftn and cudasirecon band layout.
    // NOTE: standalone storage helper — RealFFT operates on raw pointers / Eigen
    // tensors and does not consume HalfComplexBand directly yet.
    template <typename Real>
    class HalfComplexBand {
    public:
        using Complex = std::complex<Real>;
        using Tensor = TensorXc<Real, 3>;

        HalfComplexBand(Eigen::Index nz, Eigen::Index ny, Eigen::Index nx)
            : nz_(requirePositive(nz)), ny_(requirePositive(ny)), nx_(requirePositive(nx)),
              half_nx_(nx_ / 2 + 1),
              values_(nz_, ny_, half_nx_) {}

        Eigen::Index nz() const { return nz_; }
        Eigen::Index ny() const { return ny_; }
        Eigen::Index nx() const { return nx_; }
        Eigen::Index halfNx() const { return half_nx_; }
        Eigen::Index size() const { return values_.size(); }

        Tensor& tensor() { return values_; }
        const Tensor& tensor() const { return values_; }
        Complex* data() { return values_.data(); }
        const Complex* data() const { return values_.data(); }

        Complex& operator()(Eigen::Index z, Eigen::Index y, Eigen::Index x) {
            return values_(z, y, x);
        }
        const Complex& operator()(Eigen::Index z, Eigen::Index y, Eigen::Index x) const {
            return values_(z, y, x);
        }

    private:
        static Eigen::Index requirePositive(Eigen::Index n) {
            if (n <= 0)
                throw std::invalid_argument("HalfComplexBand dimensions must be positive");
            return n;
        }

        Eigen::Index nz_;
        Eigen::Index ny_;
        Eigen::Index nx_;
        Eigen::Index half_nx_;
        Tensor values_;
    };

    // Planned real-to-complex / complex-to-real double precision FFT, batched
    // over `howmany` contiguous transforms. Backed by FFTW on Device::cpu()
    // and by cuFFT (D2Z/Z2D) on Device::cuda(n) -- the same object API for
    // both, mirroring sirius::FFT. PlanRigor only affects the FFTW backend.
    class RealFFT {
    public:
        using Real = double;
        using Complex = std::complex<double>;

        // dims: {n}, {rows, cols}, or {depth, rows, cols}; the last dimension
        // is the real axis compressed to n/2 + 1 complex samples on output.
        explicit RealFFT(std::vector<int> dims, int howmany = 1, PlanRigor rigor = PlanRigor::Measure,
                         Device device = Device::cpu());
        ~RealFFT();

        RealFFT(const RealFFT&) = delete;
        RealFFT& operator=(const RealFFT&) = delete;
        RealFFT(RealFFT&&) noexcept;
        RealFFT& operator=(RealFFT&&) noexcept;

        Device device() const noexcept;
        int rank() const;            // dims.size()
        int howmany() const;         // batch count
        int realSize() const;        // product(dims), e.g. depth*rows*cols
        int complexSize() const;     // product(dims[:-1]) * (dims.back()/2 + 1), e.g. depth*rows*(cols/2+1)
        int fullRealSize() const;    // realSize() * howmany()
        int fullComplexSize() const; // complexSize() * howmany()
        const std::vector<int>& dims() const;  // original real dims, e.g. {depth, rows, cols}

        // Pointers must reference memory on device(): host memory for a CPU
        // plan, device memory for a CUDA plan (where calls are asynchronous
        // on `stream`). The complex input of irfft is preserved.
        void rfft(const Real* in, Complex* out, const Stream& stream = Stream::null()) const;
        void irfft(const Complex* in, Real* out, bool normalize = false,
                   const Stream& stream = Stream::null()) const;

        template<int Rank>
        void rfft(const TensorXr<double, Rank>& in, TensorXc<double, Rank>& out) const;

        template<int Rank>
        void irfft(const TensorXc<double, Rank>& in, TensorXr<double, Rank>& out,
                   bool normalize = false) const;

    private:
        struct Impl;
        std::unique_ptr<Impl> impl_;
    };

} // namespace sirius

#endif // SIRIUS_REAL_FFT_HPP
