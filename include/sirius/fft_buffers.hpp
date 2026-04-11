#ifndef FFT_BUFFERS_HPP
#define FFT_BUFFERS_HPP

#include <Eigen/Core>
#include <stdexcept>
#include <fftw3.h>

namespace sirius {

    // 1D Complex buffer
    class FFTWBuffer1D {
    public:
        FFTWBuffer1D(Eigen::Index n): n_(n) {
            if (n <= 0) throw std::invalid_argument("Size must be positive");
            if (n > static_cast<Eigen::Index>(std::numeric_limits<int>::max())) {
                throw std::invalid_argument("FFT dimension exceeds FFTW's int limit.");
            }
            data_ = fftw_alloc_complex(static_cast<size_t>(n));
            if (!data_) throw std::bad_alloc();
        }

        ~FFTWBuffer1D() {fftw_free(data_);}

        // delete copy constructor
        FFTWBuffer1D(const FFTWBuffer1D&) = delete;
        FFTWBuffer1D& operator=(const FFTWBuffer1D&) = delete;

        // move constructor
        FFTWBuffer1D(FFTWBuffer1D&& other) noexcept: data_(other.data_), n_(other.n_) {
            other.data_ = nullptr;
            other.n_ = 0;
        }
        FFTWBuffer1D& operator=(FFTWBuffer1D&& other) noexcept {
            if (this != &other) {
                fftw_free(data_);
                data_ = other.data_;
                n_ = other.n_;
                other.data_ = nullptr;
                other.n_ = 0;
            }
            return *this;
        }

        // access raw data pointers
        fftw_complex* data() {return data_;}
        const fftw_complex* data() const {return data_;}

        // lightweight eigen conversion
        Eigen::Map<Eigen::VectorXcd> as_eigen() {
            return {reinterpret_cast<std::complex<double>*>(data_), n_};
        }

        Eigen::Index size() const {return n_;}

    private:
        fftw_complex* data_ = nullptr;
        Eigen::Index n_ = 0;
    };


}


#endif // FFT_BUFFERS_HPP